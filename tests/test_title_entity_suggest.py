"""title_entity_suggest candidate source tests.

Covers the observable contract: entity spans are extracted from the **raw**
question string where casing survives (the T1/T2 distinction),
``suggest`` calls are capped by ``max_spans`` per archive, distinctive terms
are gated behind their param, and candidates are emitted span-major with
per-archive path dedupe — degrading to ``[]`` (never raising) on empty scope,
missing archives, or a failing suggest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from vesta.retrieval.contracts import PreparedQuery, Scope
from vesta.retrieval.impls.title_entity_suggest import TitleEntitySuggest, extract_spans
from vesta.retrieval.trace import Trace

# ── Fakes (follow tests/test_alias_title_resolve.py's FakeArchive style) ─────


@dataclass
class FakeArchive:
    """Minimal Archive fake — records every suggest prefix it is asked."""

    id: int = 1
    uuid: str = "fake-uuid"
    title: str = "Fake Archive"
    language: str = "en"
    has_fulltext_index: bool = True
    article_count: int = 10
    suggest_paths: list[str] = field(default_factory=list)
    raise_on_suggest: bool = False
    calls: list[str] = field(default_factory=list)

    async def search(self, terms: Any, limit: int) -> list[str]:
        return []

    async def suggest(self, prefix: str, limit: int) -> list[str]:
        self.calls.append(prefix)
        if self.raise_on_suggest:
            raise RuntimeError("boom")
        return list(self.suggest_paths)[:limit]

    async def read(self, path: str) -> None:
        raise NotImplementedError

    async def extract(self, path: str) -> Any:
        raise NotImplementedError

    async def main_path(self) -> str:
        return "Fake/Article_1"


class FakeRegistry:
    def __init__(self, archives: list[FakeArchive] | None = None) -> None:
        self._archives = archives if archives is not None else [FakeArchive()]

    def enabled(self, scope: Any = None) -> list[FakeArchive]:
        if scope is not None and getattr(scope, "zim_ids", None) is not None:
            return [a for a in self._archives if a.id in scope.zim_ids]
        return list(self._archives)

    async def ids_for_labels(self, labels: Any) -> frozenset[int]:
        return frozenset()


def _pq(raw: str, terms: tuple[str, ...] = ()) -> PreparedQuery:
    """A query whose ``terms`` are already normalize-destroyed (lowercased)."""
    return PreparedQuery(
        raw=raw,
        terms=terms or tuple(raw.lower().split()),
        text=raw,
        aliases=(),
        is_keyword_query=False,
        rung="initial",
        history=(),
    )


def _tr() -> Trace:
    return Trace()


# ── Span extraction: casing is the signal ───────────────────────────────────


class TestExtractSpans:
    @pytest.mark.parametrize(
        ("query", "expected_spans"),
        [
            ("How old was Napoleon when he became emperor?", ["Napoleon"]),
            ("How did the Marconi wireless telegraph work?", ["Marconi"]),
            (
                "Elizabeth II was queen regnant of how many sovereign states during her lifetime?",
                ["Elizabeth II"],
            ),
            ("The Great Depression lasted from 1929 to 1939.", ["Great Depression"]),
            ("Where is the University of Toronto located?", ["University of Toronto"]),
            ("What did Vincent van Gogh paint?", ["Vincent van Gogh"]),
            (
                "I've read that the German philosopher Max Scheler's heavy smoking killed him.",
                ["Max Scheler"],
            ),
            (
                "According to the article, on what date was Abraham Lincoln admitted to the Illinois bar?",
                ["Abraham Lincoln"],
            ),
            ("How many moons orbit Jupiter, and how many orbit Saturn?", ["Jupiter", "Saturn"]),
            ("How was the brand name 'Nembutal' derived?", ["Nembutal"]),
            (
                "How was the brand name \u2018Nembutal\u2019 derived, and who coined it?",
                ["Nembutal"],
            ),
            ("", []),
            ("   ", []),
            ("napoleon emperor", []),
            (
                "Besides raloxifene, what is the name of another benzothiophene modulator?",
                [],
            ),
        ],
    )
    def test_extract_spans_casing_and_punctuation(
        self, query: str, expected_spans: list[str]
    ) -> None:
        assert extract_spans(query) == expected_spans

    @pytest.mark.parametrize(
        ("query", "kwargs", "expected"),
        [
            # double quotes
            (
                'For the Nazi "Greater Aryan certificate", how far back?',
                {},
                {"in": ["Greater Aryan certificate"]},
            ),
            # possessive tail is stripped
            (
                "In Norbert Wiener's 1959 fiction The Tempter, who is Oliver?",
                {},
                {"in": ["Norbert Wiener"]},
            ),
            # demonyms do not become spans
            (
                "I've read that the German philosopher Max Scheler's heavy smoking killed him.",
                {},
                {"in": ["Max Scheler"], "not_in": ["German"]},
            ),
            # distinctive terms enabled
            (
                "Besides raloxifene, what is the name of another benzothiophene modulator?",
                {"include_distinctive_terms": True},
                {"in": ["raloxifene"]},
            ),
            # distinctive terms skip stopwords/openers
            (
                "During the early lockdown in Italy, what was the name of the radio station?",
                {"include_distinctive_terms": True, "max_spans": 10},
                {"in": ["Italy"], "not_in": ["During"]},
            ),
            # strength ordering multi-token before single
            (
                "What role did John Densmore portray in his guest appearance on Beverly Hills?",
                {},
                {"prefix": ["John Densmore", "Beverly Hills"]},
            ),
            # max spans cap
            (
                "How many moons orbit Jupiter, and how many orbit Saturn, Mars, Venus and Mercury?",
                {"max_spans": 3},
                {"len": 3},
            ),
            # subtoken span not repeated alone
            (
                "Elizabeth II was queen regnant of how many sovereign states?",
                {"max_spans": 8},
                {"in": ["Elizabeth II"], "not_in": ["Elizabeth"]},
            ),
        ],
    )
    def test_extract_spans_options_and_ordering(
        self,
        query: str,
        kwargs: dict[str, Any],
        expected: dict[str, Any],
    ) -> None:
        spans = extract_spans(query, **kwargs)
        for item in expected.get("in", []):
            assert item in spans
        for item in expected.get("not_in", []):
            assert item not in spans
        if "len" in expected:
            assert len(spans) == expected["len"]
        if "prefix" in expected:
            assert spans[: len(expected["prefix"])] == expected["prefix"]


# ── Source behaviour: capping, gating, emission ──────────────────────────────


class TestFind:
    async def test_suggest_calls_capped_at_max_spans_per_archive(self):
        archive = FakeArchive(suggest_paths=["A/Napoleon"])
        source = TitleEntitySuggest(
            params=TitleEntitySuggest.Params(max_spans=2), archives=FakeRegistry([archive])
        )
        q = _pq("How many moons orbit Jupiter, Saturn, Mars and Venus, really?")
        out = await source.find(q, Scope(), _tr())
        # 4+ spans available, but only max_spans suggest calls happen.
        assert len(archive.calls) == 2
        assert out

    async def test_no_suggest_calls_when_no_spans(self):
        archive = FakeArchive(suggest_paths=["A/Raloxifene"])
        source = TitleEntitySuggest(archives=FakeRegistry([archive]))
        # Lowercase-only question, distinctive terms off by default.
        out = await source.find(
            _pq("besides raloxifene what is the name of another modulator"), Scope(), _tr()
        )
        assert out == []
        assert archive.calls == []

    async def test_distinctive_terms_param_gates_the_extra_suggests(self):
        archive = FakeArchive(suggest_paths=["A/Raloxifene"])
        source = TitleEntitySuggest(
            params=TitleEntitySuggest.Params(include_distinctive_terms=True),
            archives=FakeRegistry([archive]),
        )
        q = _pq("Besides raloxifene, what is the name of another modulator?")
        out = await source.find(q, Scope(), _tr())
        # The gold-naming lowercase term is looked up (it is first — earliest
        # distinctive term), and the total is still capped by max_spans.
        assert archive.calls[0] == "raloxifene"
        assert len(archive.calls) <= 4
        assert [c.path for c in out] == ["A/Raloxifene"]

    async def test_candidates_are_emitted_span_major_and_deduped(self):
        archive = FakeArchive(suggest_paths=["A/Shared", "A/Only"])
        archive2 = FakeArchive(id=2, suggest_paths=["A/Shared"])
        source = TitleEntitySuggest(archives=FakeRegistry([archive, archive2]))
        q = _pq("What did Vincent van Gogh and Napoleon Bonaparte both do?")
        out = await source.find(q, Scope(), _tr())
        by_zim: dict[int, list[str]] = {}
        for c in out:
            assert c.source == "title_entity_suggest"
            assert c.score is None
            by_zim.setdefault(c.zim_id, []).append(c.path)
        # Both spans hit the same two paths in archive 1: deduped, the
        # stronger span's order kept, ranks dense from rank_offset.
        assert by_zim[1] == ["A/Shared", "A/Only"]
        ranks1 = [c.rank for c in out if c.zim_id == 1]
        assert ranks1 == [4, 5]
        assert by_zim[2] == ["A/Shared"]

    async def test_exact_title_match_collapses_to_the_exact_article(self):
        """When the span's own article is among the suggest results, it is
        the intent and the non-exact tail is dilution — year variants
        ("Academy Awards" -> Academy_Awards_2017), sibling articles
        ("Abraham Lincoln" -> Abraham_Lincoln_Assassination), case-variant
        crowds — each nomination costs funnel width a dense-found gold is
        competing for (the three measured hybrid full losses)."""
        archive = FakeArchive(
            suggest_paths=[
                "A/ABRAHAM_LINCOLN",
                "A/Abraham_Lincoln_Assassination",
                "A/Abraham_Lincoln_II",
            ]
        )
        source = TitleEntitySuggest(archives=FakeRegistry([archive]))
        out = await source.find(
            _pq("On what date was Abraham Lincoln admitted to the Illinois bar?"),
            Scope(),
            _tr(),
        )
        assert [c.path for c in out] == ["A/ABRAHAM_LINCOLN"]
        assert [c.rank for c in out] == [4]

    async def test_exact_match_folds_case_underscores_and_namespace(self):
        """The comparison mirrors ``alias_title_resolve._exact_matches``:
        basename (namespace stripped), underscores as spaces, lowercase."""
        archive = FakeArchive(suggest_paths=["A/Napoleon_Bonaparte", "A/Napoleon_II"])
        source = TitleEntitySuggest(archives=FakeRegistry([archive]))
        out = await source.find(_pq("What did Napoleon Bonaparte do?"), Scope(), _tr())
        assert [c.path for c in out] == ["A/Napoleon_Bonaparte"]

    async def test_no_exact_match_keeps_the_top_limit_fallback(self):
        """A span that names a person but not their article's title keeps the
        fallback: top-``limit`` results in suggest order."""
        archive = FakeArchive(
            suggest_paths=["A/Derek_McLane_Set_Designer", "A/McLane", "A/McLane_(surname)"]
        )
        source = TitleEntitySuggest(
            params=TitleEntitySuggest.Params(limit=2), archives=FakeRegistry([archive])
        )
        out = await source.find(_pq("What did Derek McLane design?"), Scope(), _tr())
        assert [c.path for c in out] == ["A/Derek_McLane_Set_Designer", "A/McLane"]

    async def test_rank_offset_offsets_emission_without_reordering(self):
        """RRF tempering: ranks shift up (weight 1/(k+rank) shrinks) but the
        span-major ordering — the observable candidate order — is unchanged."""
        paths = [f"A/P{i}" for i in range(20)]
        for offset in (0, 4, 12):
            archive = FakeArchive(suggest_paths=paths)
            source = TitleEntitySuggest(
                params=TitleEntitySuggest.Params(max_spans=1, limit=3, rank_offset=offset),
                archives=FakeRegistry([archive]),
            )
            out = await source.find(_pq("What did Napoleon Bonaparte do?"), Scope(), _tr())
            assert [c.path for c in out] == ["A/P0", "A/P1", "A/P2"]
            assert [c.rank for c in out] == [offset, offset + 1, offset + 2]

    async def test_suggest_failure_degrades_to_remaining_spans(self):
        archive = FakeArchive(raise_on_suggest=True)
        archive.calls.clear()
        source = TitleEntitySuggest(archives=FakeRegistry([archive]))
        out = await source.find(_pq("What did Napoleon Bonaparte do?"), Scope(), _tr())
        assert out == []
        assert len(archive.calls) >= 1  # it tried, and swallowed the error

    async def test_empty_scope_returns_empty(self):
        archive = FakeArchive(suggest_paths=["A/Napoleon"])
        source = TitleEntitySuggest(archives=FakeRegistry([archive]))
        out = await source.find(_pq("What did Napoleon do?"), Scope(zim_ids=frozenset({9})), _tr())
        assert out == []
        assert archive.calls == []


# ── Registration + profile opt-in ────────────────────────────────────────────


class TestRegistration:
    def test_registered_as_candidate_source(self):
        from vesta.retrieval.registry import resolve

        assert resolve("candidate_source", "title_entity_suggest") is TitleEntitySuggest

    def test_requires_no_capabilities(self):
        """The title index is universal — depth-0 boxes keep it (S5)."""
        assert TitleEntitySuggest.requires == frozenset()

    def test_defaults_match_the_measured_ship_config(self):
        """Defaults = the measured ship config (see the impl's Params docstring)."""
        params = TitleEntitySuggest.Params()
        assert params.include_distinctive_terms is False
        assert params.max_spans == 4
        assert params.limit == 3
        assert params.rank_offset == 4
