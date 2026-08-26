"""Walkthrough + contract tests for the retrieval framework.

Two acceptance tests live here:

* **The walkthrough.** Proves adding a new ``CandidateSource``
  is one file + one import line. The throwaway ``random_articles`` source is
  defined *in this test module* (it is the "one new file"), registers itself on
  import (the "one import line"), and appears in a real pipeline trace. It is
  deliberately NOT shipped under ``src/`` — delete it after the walk.

* **The contract suite.** Every registered impl satisfies its
  Protocol, handles empty input, and (when run through the pipeline) its stage
  appears in the trace.

Plus degradation, profile-hash-in-trace, and confidence-signal coverage.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from vesta.retrieval.contracts import (
    Candidate,
    PreparedQuery,
    Scope,
)
from vesta.retrieval.pipeline import Deps, run_pipeline
from vesta.retrieval.profiles import ProfileComponent, RetrievalProfile
from vesta.retrieval.registry import register, registered, resolve

# ── Throwaway walkthrough source (the "one new file") ───────────────────────


@register("candidate_source", "random_articles")
class RandomArticles:
    """Returns a fixed set of Candidates for the walkthrough demonstration.

    Exists only inside this test module to prove the invariant: a new
    CandidateSource is one file + one import line, with zero other file changes.
    """

    requires = frozenset()

    class Params:  # minimal stand-in; the real registry introspection uses Pydantic
        pass

    def __init__(self, params: Any = None, archives: Any = None) -> None:
        self._params = params
        self._archives = archives

    async def find(self, q: PreparedQuery, scope: Scope, tr: Any) -> list[Candidate]:
        out: list[Candidate] = []
        if self._archives is not None:
            archives = self._archives.enabled()
            if archives:
                archive = archives[0]
                try:
                    main = await archive.main_path()
                    out.append(
                        Candidate(
                            zim_id=archive.id,
                            path=main,
                            source="random_articles",
                            rank=0,
                            score=0.99,
                        )
                    )
                except Exception:
                    pass
        return out


# ── Fakes ───────────────────────────────────────────────────────────────────


@dataclass
class FakeArchive:
    """Minimal Archive fake for pipeline tests."""

    id: int = 1
    uuid: str = "fake-uuid"
    title: str = "Fake Archive"
    language: str = "en"
    has_fulltext_index: bool = True
    article_count: int = 10

    async def search(self, terms: Any, limit: int) -> list[str]:
        return ["Fake/Article_1", "Fake/Article_2"]

    async def suggest(self, prefix: str, limit: int) -> list[str]:
        return ["Fake/Article_1"]

    async def read(self, path: str) -> None:
        raise NotImplementedError

    async def extract(self, path: str) -> Any:
        from vesta.zim.types import EntryFlags, ExtractedArticle

        return ExtractedArticle(
            path=path,
            title="Fake Article",
            text=(
                "This is a test article. It has multiple sentences. "
                "Each sentence is meant for testing the pipeline."
            ),
            sections=(),
            flags=EntryFlags.NONE,
        )

    async def main_path(self) -> str:
        return "Fake/Article_1"


class FakeRegistry:
    """Minimal ArchiveRegistry fake, including the alias/label hooks."""

    def __init__(self, archive: FakeArchive | None = None):
        self._archive = archive or FakeArchive()

    def get(self, zim_id: int) -> FakeArchive:
        return self._archive

    def enabled(self, scope: Any = None) -> list[FakeArchive]:
        return [self._archive]

    def has_any_fulltext(self) -> bool:
        return self._archive.has_fulltext_index

    async def lookup_aliases(self, terms: Any, *, max_aliases: int) -> list[str]:
        return []

    async def ids_for_labels(self, labels: Any) -> frozenset[int]:
        return frozenset()


def make_profile(**overrides: Any) -> RetrievalProfile:
    """Build a minimal profile for testing."""
    return RetrievalProfile(
        name="test",
        description="Test profile",
        hash="test-hash",
        preparers=overrides.get("preparers", (ProfileComponent(impl="normalize"),)),
        sources=overrides.get(
            "sources",
            (
                ProfileComponent(impl="random_articles"),
                ProfileComponent(impl="title_suggest", params={"limit": 5}),
            ),
        ),
        fusion=overrides.get(
            "fusion", ProfileComponent(impl="rrf", params={"k": 20, "across_archives": "union"})
        ),
        passages=overrides.get(
            "passages",
            ProfileComponent(
                impl="candidate_articles", params={"max_articles": 5, "max_passages": 20}
            ),
        ),
        scorers=overrides.get("scorers", (ProfileComponent(impl="lexical_overlap"),)),
        assembler=overrides.get(
            "assembler",
            ProfileComponent(
                impl="topk_budget",
                params={"budget_tokens": 1000, "max_per_article": 2, "dedup": "none"},
            ),
        ),
    )


def _deps(**overrides: Any) -> Deps:
    return Deps(
        archives=overrides.get("archives", FakeRegistry()),
        capabilities=overrides.get("capabilities", frozenset()),
        semaphore=overrides.get("semaphore", asyncio.Semaphore(4)),
    )


# ── Walkthrough test ────────────────────────────────────────────────────────


class TestWalkthrough:
    """Prove that adding a new CandidateSource is one file + one import line."""

    @pytest.mark.asyncio
    async def test_pipeline_with_random_articles_produces_trace(self) -> None:
        """A pipeline run with random_articles must show it in the trace."""
        profile = make_profile(sources=(ProfileComponent(impl="random_articles"),))
        result = await run_pipeline(
            profile=profile, query="test query", scope=Scope(), deps=_deps()
        )
        trace_dict = result.trace.to_dict()
        source_stages = [s for s in trace_dict["stages"] if s["name"] == "candidate_source"]
        assert len(source_stages) > 0, "No candidate_source stages in trace"
        components = {s["component"] for s in source_stages}
        assert "random_articles" in components, (
            f"random_articles not found in trace components: {components}"
        )
        # Profile content hash must appear in every trace (03 spec).
        assert trace_dict["profile"] == profile.name
        assert trace_dict["profile_hash"] == profile.hash


# ── Degradation tests ───────────────────────────────────────────────────────


class TestDegradation:
    """Capability checks: drop unmet components, record in trace, loud failure."""

    @pytest.mark.asyncio
    async def test_component_with_unmet_requires_is_dropped_and_recorded(self) -> None:
        """A source requiring ZIM_FULLTEXT is dropped when capability is absent."""
        profile = make_profile(
            sources=(
                ProfileComponent(impl="xapian_fts", params={"limit": 5}),
                ProfileComponent(impl="title_suggest", params={"limit": 5}),
            ),
            preparers=(),
            scorers=(),
        )
        # No ZIM_FULLTEXT, but title_suggest requires nothing → it survives.
        result = await run_pipeline(
            profile=profile, query="test", scope=Scope(), deps=_deps(capabilities=frozenset())
        )
        trace_dict = result.trace.to_dict()
        assert trace_dict["degradations"], "Expected degradation entries"
        dropped = {d["component"] for d in trace_dict["degradations"]}
        assert any("xapian_fts" in c for c in dropped), f"xapian_fts should be dropped: {dropped}"

    @pytest.mark.asyncio
    async def test_all_sources_unmet_capabilities_raises_error(self) -> None:
        """When ALL sources require unmet capabilities, pipeline must raise."""
        fts_only_profile = make_profile(
            sources=(ProfileComponent(impl="xapian_fts", params={"limit": 5}),),
            preparers=(),
            scorers=(),
        )
        with pytest.raises(RuntimeError, match="All candidate sources were dropped"):
            await run_pipeline(
                profile=fts_only_profile,
                query="test",
                scope=Scope(),
                deps=_deps(capabilities=frozenset()),
            )


# ── Hybrid depth-0 equivalence ──────────────────────────────────────────────


class TestHybridDepth0Equivalence:
    """``hybrid`` with ``VECTORS`` unmet must be byte-equivalent to ``standard``.

    The ``retrieval.active_profile`` default is ``hybrid``;
    a depth-0 box (no semantic index) must therefore behave *exactly* as it
    did under ``standard`` — not merely "still run". Two invariants:

    1. structurally, ``hybrid`` is ``standard`` plus exactly one source
       (``vector_knn``) and nothing else;
    2. at runtime with ``VECTORS`` (and the encoders) unmet, that one source
       is capability-dropped with a recorded degradation and the resulting
       passages / cards / confidence are equal field-for-field.
    """

    def test_hybrid_is_standard_plus_exactly_vector_knn(self) -> None:
        """Guard the structural premise: no other component may drift."""
        from vesta.retrieval.profiles import load_profile

        standard = load_profile("standard")
        hybrid = load_profile("hybrid")
        assert standard is not None and hybrid is not None
        dense_only = [c for c in hybrid.sources if c not in standard.sources] + [
            c for c in standard.sources if c not in hybrid.sources
        ]
        assert [(c.impl, c.params) for c in dense_only] == [
            ("vector_knn", {"k": 40, "enabled": True})
        ]
        assert hybrid.preparers == standard.preparers
        assert hybrid.fusion == standard.fusion
        assert hybrid.passages == standard.passages
        assert hybrid.scorers == standard.scorers
        assert hybrid.assembler == standard.assembler

    @pytest.mark.asyncio
    async def test_hybrid_without_vectors_is_byte_equivalent_to_standard(self) -> None:
        from vesta.config.capabilities import Capability
        from vesta.retrieval.profiles import load_profile

        standard = load_profile("standard")
        hybrid = load_profile("hybrid")
        # A depth-0 box: Xapian full text yes; no semantic index, no ONNX
        # encoders, no LLM. This is the capability reality of a fresh install
        # that has registered a ZIM but never indexed it.
        deps = _deps(capabilities=frozenset({Capability.ZIM_FULLTEXT}))
        res_std = await run_pipeline(profile=standard, query="test query", scope=Scope(), deps=deps)
        res_hyb = await run_pipeline(profile=hybrid, query="test query", scope=Scope(), deps=deps)

        # Not a vacuous pass: both runs retrieved real passages through the
        # surviving lexical sources.
        assert res_std.passages, "standard produced no passages — fixture broke"

        # The one and only difference: hybrid dropped vector_knn for missing
        # `vectors`. Every other degradation (encoders, llm) is shared.
        std_degs = {d["component"] for d in res_std.trace.to_dict()["degradations"]}
        hyb_degs = {d["component"] for d in res_hyb.trace.to_dict()["degradations"]}
        assert hyb_degs - std_degs == {"candidate_source/vector_knn"}

        # Byte-equivalence of everything downstream of the sources.
        assert res_hyb.passages == res_std.passages
        assert res_hyb.cards == res_std.cards
        assert res_hyb.confidence == res_std.confidence


# ── Contract tests ──────────────────────────────────────────────────────────


class TestContracts:
    """Every registered implementation satisfies its Protocol, handles empty
    input, and writes to the trace (03 spec DoD item 3)."""

    def _make_trace(self) -> Any:
        from vesta.retrieval.trace import Trace

        return Trace()

    def _empty_pq(self) -> PreparedQuery:
        return PreparedQuery(raw="", terms=(), text="", aliases=(), is_keyword_query=False, rung="")

    # ── Structural checks: every impl has the contract method + Params ─────

    @pytest.mark.parametrize(
        "kind,method",
        [
            ("query_preparer", "prepare"),
            ("candidate_source", "find"),
            ("fuser", "fuse"),
            ("passage_builder", "build"),
            ("passage_scorer", "score"),
            ("context_assembler", "assemble"),
        ],
    )
    def test_every_impl_has_contract_method(self, kind: str, method: str) -> None:
        for name in registered(kind):
            cls = resolve(kind, name)
            assert cls is not None
            assert hasattr(cls, method), f"{kind}/{name}.{method} missing"

    # ── Behavioural checks: empty input must not raise ────────────────────

    @pytest.mark.asyncio
    async def test_all_preparers_handle_empty_query(self) -> None:
        pq = self._empty_pq()
        for name in registered("query_preparer"):
            cls = resolve("query_preparer", name)
            instance = cls(params=cls.Params())  # type: ignore[attr-defined]
            result = await instance.prepare(pq, self._make_trace())
            assert isinstance(result, PreparedQuery), f"{name} did not return PreparedQuery"

    def test_all_fusers_handle_empty_input(self) -> None:
        tr = self._make_trace()
        for name in registered("fuser"):
            cls = resolve("fuser", name)
            instance = cls(params=cls.Params())  # type: ignore[attr-defined]
            result = instance.fuse({}, tr)
            assert isinstance(result, list)
            assert len(result) == 0, f"{name} returned results for empty input"

    @pytest.mark.asyncio
    async def test_all_passage_builders_handle_empty_input(self) -> None:
        for name in registered("passage_builder"):
            cls = resolve("passage_builder", name)
            instance = cls(params=cls.Params(), archives=FakeRegistry())  # type: ignore[attr-defined]
            result = await instance.build([], self._empty_pq(), self._make_trace())
            assert isinstance(result, list)
            assert len(result) == 0, f"{name} returned passages for empty input"

    @pytest.mark.asyncio
    async def test_all_passage_scorers_handle_empty_input(self) -> None:
        for name in registered("passage_scorer"):
            cls = resolve("passage_scorer", name)
            instance = cls(params=cls.Params())  # type: ignore[attr-defined]
            result = await instance.score([], self._empty_pq(), self._make_trace())
            assert isinstance(result, list)

    def test_all_context_assemblers_handle_empty_input(self) -> None:
        from vesta.retrieval.contracts import Budget

        tr = self._make_trace()
        pq = self._empty_pq()
        for name in registered("context_assembler"):
            cls = resolve("context_assembler", name)
            instance = cls(params=cls.Params(), archives=FakeRegistry())  # type: ignore[attr-defined]
            result = instance.assemble([], Budget(token_total=1000, max_per_article=2), pq, tr)
            from vesta.retrieval.contracts import RetrievalResult

            assert isinstance(result, RetrievalResult), f"{name} did not return RetrievalResult"

    # ── Trace coverage: every kind's stage appears in a real pipeline run ─

    @pytest.mark.asyncio
    async def test_pipeline_trace_records_every_stage_kind(self) -> None:
        result = await run_pipeline(
            profile=make_profile(), query="test query", scope=Scope(), deps=_deps()
        )
        stage_names = {s["name"] for s in result.trace.to_dict()["stages"]}
        # Every pipeline stage that has a registered, capability-met component
        # must appear in the trace.
        assert "preparer" in stage_names
        assert "candidate_source" in stage_names
        assert "fuser" in stage_names
        assert "passage_builder" in stage_names
        assert "passage_scorer" in stage_names
        assert "context_assembler" in stage_names


# ── Registry tests ──────────────────────────────────────────────────────────


class TestRegistry:
    """The registry introspection API works correctly."""

    def test_component_schemas_returns_all_kinds(self) -> None:
        from vesta.retrieval.registry import component_schemas

        schemas = component_schemas()
        for kind in ("preparers", "sources", "fusion", "passages", "scorers", "assemblers"):
            assert kind in schemas, f"{kind} missing from component_schemas"
            assert isinstance(schemas[kind], list)

    @pytest.mark.parametrize(
        "kind,expected",
        [
            ("candidate_source", {"xapian_fts", "title_suggest"}),
            ("query_preparer", {"normalize", "alias_expand", "conversational_rewrite"}),
            ("fuser", {"rrf"}),
            ("passage_builder", {"candidate_articles"}),
            ("passage_scorer", {"lexical_overlap"}),
            ("context_assembler", {"topk_budget"}),
        ],
    )
    def test_baseline_components_registered(self, kind: str, expected: set[str]) -> None:
        actual = registered(kind)
        assert expected <= set(actual), f"{expected} not subset of {kind} registered: {actual}"


# ── Confidence signals tests ────────────────────────────────────────────────


class TestConfidenceSignals:
    """Confidence signals are computed and recorded, including agreement."""

    @pytest.mark.asyncio
    async def test_confidence_signals_appear_in_result(self) -> None:
        result = await run_pipeline(
            profile=make_profile(), query="test query", scope=Scope(), deps=_deps()
        )
        assert result.confidence is not None
        for attr in ("top_score", "score_dropoff", "density", "agreement"):
            assert hasattr(result.confidence, attr)

    @pytest.mark.parametrize(
        "scores,expected_check",
        [
            # top (0.9) stands well above tail (0.1) -> low ratio (< 0.2)
            ([0.9, 0.5, 0.1], lambda d: 0.0 <= d < 0.2),
            # flat scores -> ratio is ~1.0
            ([0.5, 0.5, 0.5], lambda d: abs(d - 1.0) < 1e-9),
        ],
    )
    def test_score_dropoff(self, scores: list[float], expected_check: Any) -> None:
        from vesta.retrieval.assemblers._shared import compute_confidence as _compute_confidence
        from vesta.retrieval.contracts import ScoredPassage
        from vesta.zim.types import Passage

        def sp(score: float, path: str) -> ScoredPassage:
            p = Passage(
                zim_id=1,
                path=path,
                ordinal=0,
                char_start=0,
                char_end=10,
                breadcrumb="",
                text="a b c d e f",
                is_lead=False,
            )
            return ScoredPassage(passage=p, score=score, source_info="x")

        c = _compute_confidence([sp(s, chr(65 + i)) for i, s in enumerate(scores)])
        assert c.score_dropoff is not None
        assert expected_check(c.score_dropoff)


# ── Budget-ceiling test (non-binding, profile-authoritative) ────────────────

_captured_budgets: list[Any] = []


@register("context_assembler", "capture_budget")
class _CaptureBudgetAssembler:
    """Throwaway assembler recording the ``Budget`` the pipeline hands it.

    Proves the pipeline's ``token_total`` is a non-binding ceiling — matching
    ``max_per_article``'s established pattern — so a profile's own
    ``Params.budget_tokens`` is always authoritative. A prior bug bound the
    pipeline's ceiling to a global default (2400), silently clamping every
    profile's larger budget (e.g. ``standard``'s 12 000 tokens) back down.
    """

    requires = frozenset()

    class Params:
        pass

    def __init__(self, params: Any = None, archives: Any = None) -> None:
        self._params = params

    def assemble(self, scored: Any, budget: Any, q: Any, tr: Any) -> Any:
        from vesta.retrieval.contracts import ConfidenceSignals, RetrievalResult

        _captured_budgets.append(budget)
        return RetrievalResult(
            passages=(),
            cards=(),
            trace=tr,
            confidence=ConfidenceSignals(
                top_score=None, score_dropoff=None, density=0.0, agreement=0.0
            ),
        )


class TestBudgetCeiling:
    @pytest.mark.asyncio
    async def test_token_total_is_not_clamped_to_the_global_default(self) -> None:
        _captured_budgets.clear()
        profile = make_profile(assembler=ProfileComponent(impl="capture_budget", params={}))
        await run_pipeline(profile=profile, query="test query", scope=Scope(), deps=_deps())
        assert len(_captured_budgets) == 1
        # far above any plausible pipeline-side ceiling — the pipeline must
        # not bind the assembler to it.
        assert _captured_budgets[0].token_total >= 100_000


# ── Agreement signal tests ─────────────────────────────────────────────────


class TestAgreement:
    """The agreement signal is computed from per-source candidate sets."""

    @pytest.mark.parametrize(
        "groups_spec,expected",
        [
            # fully overlap
            ([("a", ["A/1", "A/2"]), ("b", ["A/1", "A/2"])], 1.0),
            # disjoint
            ([("a", ["A/1"]), ("b", ["B/1"])], 0.0),
            # one source
            ([("a", ["A/1"])], 0.0),
        ],
    )
    def test_source_agreement(
        self, groups_spec: list[tuple[str, list[str]]], expected: float
    ) -> None:
        from vesta.retrieval.contracts import FusionKey
        from vesta.retrieval.pipeline import _source_agreement

        groups = {
            FusionKey(zim_id=1, source=src): [
                Candidate(zim_id=1, path=p, source=src, rank=i, score=None)
                for i, p in enumerate(paths)
            ]
            for src, paths in groups_spec
        }
        assert _source_agreement(groups) == expected


# ── Profile tests ───────────────────────────────────────────────────────────


class TestProfiles:
    """Built-in profiles load and validate correctly."""

    def test_builtin_profiles_integrity(self) -> None:
        """Built-in profiles exist and contain expected component pipelines."""
        from vesta.retrieval.profiles import BUILTIN_PROFILES

        for name in ("lexical", "standard", "hybrid"):
            assert name in BUILTIN_PROFILES, f"{name} profile not loaded"

        lexical = BUILTIN_PROFILES["lexical"]
        prep_impls = {c.impl for c in lexical.preparers}
        assert {"normalize", "alias_expand"} <= prep_impls
        source_impls = {c.impl for c in lexical.sources}
        assert {"xapian_fts", "title_suggest"} <= source_impls
        assert lexical.fusion.impl == "rrf"
        assert lexical.passages.impl == "candidate_articles"
        assert "lexical_overlap" in {c.impl for c in lexical.scorers}
        assert lexical.assembler.impl == "topk_budget"

        hybrid = BUILTIN_PROFILES["hybrid"]
        hybrid_sources = {c.impl for c in hybrid.sources}
        assert "vector_knn" in hybrid_sources
        assert {"xapian_fts", "title_suggest"} <= hybrid_sources

    def test_profiles_have_hash(self) -> None:
        """Every built-in profile has a valid 64-character SHA-256 hash."""
        from vesta.retrieval.profiles import BUILTIN_PROFILES

        for name, profile in BUILTIN_PROFILES.items():
            assert len(profile.hash) == 64, f"{name} hash is not SHA-256"

    def test_profile_loading_and_validation(self) -> None:
        """Profile resolution and invalid profile rejection."""
        from vesta.retrieval.profiles import _validate_components, load_profile

        assert load_profile("nonexistent") is None

        with pytest.raises(ValueError, match="at least one candidate source"):
            _validate_components(
                "empty",
                {
                    "fusion": {"impl": "rrf"},
                    "passages": {"impl": "candidate_articles"},
                    "assembler": {"impl": "topk_budget"},
                    "sources": [],
                },
            )
