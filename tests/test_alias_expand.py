"""Alias expand preparer mode tests.

Covers the three ``mode`` values: ``fts_terms`` (today's unchanged default —
aliases feed both ``q.terms`` and ``q.aliases``), ``title_hints`` (aliases
populate ``q.aliases`` only, never poisoning the AND-mandatory FTS term
list), and ``both`` (the union). The default-mode test is the load-bearing one:
no existing profile sets ``mode`` explicitly, so ``fts_terms`` must reproduce
today's behaviour byte-for-byte or every profile's measured eval numbers move
by stealth.
"""

from __future__ import annotations

import pytest

from vesta.retrieval.contracts import PreparedQuery
from vesta.retrieval.impls.alias_expand import AliasExpand
from vesta.retrieval.trace import Trace


class _FakeArchives:
    """Minimal ArchiveRegistry fake exposing only lookup_aliases."""

    def __init__(self, expansions: list[str] | None = None) -> None:
        self._expansions = expansions if expansions is not None else ["Atmosphere of Earth"]
        self.calls: list[tuple[list[str], int]] = []

    async def lookup_aliases(self, terms: list[str], *, max_aliases: int) -> list[str]:
        self.calls.append((list(terms), max_aliases))
        return self._expansions[:max_aliases]


def _pq(terms: tuple[str, ...] = ("how", "much", "nitrogen", "in", "the", "air")) -> PreparedQuery:
    return PreparedQuery(
        raw="how much nitrogen in the air",
        terms=terms,
        text="how much nitrogen in the air",
        aliases=(),
        is_keyword_query=False,
        rung="initial",
        history=(),
    )


@pytest.mark.asyncio
async def test_default_mode_is_fts_terms() -> None:
    """No params at all ⇒ mode defaults to fts_terms."""
    assert AliasExpand.Params().mode == "fts_terms"


@pytest.mark.asyncio
async def test_fts_terms_mode_matches_todays_behaviour() -> None:
    """fts_terms (default): aliases appended to BOTH q.terms and q.aliases —
    exactly the pre-mode behaviour, so no existing profile's numbers move."""
    archives = _FakeArchives(["Atmosphere of Earth"])
    prep = AliasExpand(archives=archives)  # type: ignore[arg-type]
    q = _pq()
    out = await prep.prepare(q, Trace())
    assert out.terms == (*q.terms, "Atmosphere of Earth")
    assert out.aliases == ("Atmosphere of Earth",)


@pytest.mark.asyncio
async def test_title_hints_mode_does_not_poison_terms() -> None:
    """title_hints: aliases populate ONLY q.aliases — q.terms is untouched, so
    the alias never reaches the all-terms-mandatory FTS ladder."""
    archives = _FakeArchives(["Atmosphere of Earth"])
    params = AliasExpand.Params(mode="title_hints")
    prep = AliasExpand(params=params, archives=archives)  # type: ignore[arg-type]
    q = _pq()
    out = await prep.prepare(q, Trace())
    assert out.terms == q.terms  # unchanged
    assert out.aliases == ("Atmosphere of Earth",)


@pytest.mark.asyncio
async def test_both_mode_is_the_union() -> None:
    archives = _FakeArchives(["Atmosphere of Earth"])
    params = AliasExpand.Params(mode="both")
    prep = AliasExpand(params=params, archives=archives)  # type: ignore[arg-type]
    q = _pq()
    out = await prep.prepare(q, Trace())
    assert out.terms == (*q.terms, "Atmosphere of Earth")
    assert out.aliases == ("Atmosphere of Earth",)


@pytest.mark.asyncio
async def test_no_expansions_is_noop_regardless_of_mode() -> None:
    archives = _FakeArchives([])
    for mode in ("fts_terms", "title_hints", "both"):
        params = AliasExpand.Params(mode=mode)  # type: ignore[arg-type]
        prep = AliasExpand(params=params, archives=archives)  # type: ignore[arg-type]
        q = _pq()
        out = await prep.prepare(q, Trace())
        assert out is q


@pytest.mark.asyncio
async def test_no_archives_is_noop() -> None:
    prep = AliasExpand(archives=None)
    q = _pq()
    out = await prep.prepare(q, Trace())
    assert out is q


@pytest.mark.asyncio
async def test_history_preserved_across_all_modes() -> None:
    archives = _FakeArchives(["Atmosphere of Earth"])
    history = (("user", "how much nitrogen in the air"),)
    for mode in ("fts_terms", "title_hints", "both"):
        params = AliasExpand.Params(mode=mode)  # type: ignore[arg-type]
        prep = AliasExpand(params=params, archives=archives)  # type: ignore[arg-type]
        q = PreparedQuery(
            raw="how much nitrogen in the air",
            terms=("how", "much", "nitrogen", "in", "the", "air"),
            text="how much nitrogen in the air",
            aliases=(),
            is_keyword_query=False,
            rung="initial",
            history=history,
        )
        out = await prep.prepare(q, Trace())
        assert out.history == history
