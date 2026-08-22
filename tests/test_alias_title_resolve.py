"""alias_title_resolve candidate source tests.

Covers: alias hits resolve directly (no capability required), exact title
matches, near-but-not-exact title matches are correctly rejected, empty
scope/no archives degrades to ``[]`` (never raises), and dedup between the two
resolution mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from vesta.retrieval.contracts import PreparedQuery, Scope
from vesta.retrieval.impls.alias_title_resolve import AliasTitleResolve
from vesta.retrieval.trace import Trace

# ── Fakes (follow tests/test_walkthrough.py's FakeArchive/FakeRegistry style) ─


@dataclass
class FakeArchive:
    """Minimal Archive fake — only ``suggest`` matters for this source."""

    id: int = 1
    uuid: str = "fake-uuid"
    title: str = "Fake Archive"
    language: str = "en"
    has_fulltext_index: bool = True
    article_count: int = 10
    suggest_paths: list[str] = field(default_factory=list)
    raise_on_suggest: bool = False

    async def search(self, terms: Any, limit: int) -> list[str]:
        return []

    async def suggest(self, prefix: str, limit: int) -> list[str]:
        if self.raise_on_suggest:
            raise RuntimeError("boom")
        return list(self.suggest_paths)

    async def read(self, path: str) -> None:
        raise NotImplementedError

    async def extract(self, path: str) -> Any:
        raise NotImplementedError

    async def main_path(self) -> str:
        return "Fake/Article_1"


class FakeRegistry:
    """Minimal ArchiveRegistry fake including ``resolve_alias_targets``."""

    def __init__(
        self,
        archives: list[FakeArchive] | None = None,
        alias_targets: list[tuple[int, str]] | None = None,
    ) -> None:
        self._archives = archives if archives is not None else [FakeArchive()]
        self._alias_targets = alias_targets or []
        self.resolve_calls: list[tuple[list[str], Any, int]] = []

    def get(self, zim_id: int) -> FakeArchive:
        return next(a for a in self._archives if a.id == zim_id)

    def enabled(self, scope: Any = None) -> list[FakeArchive]:
        if scope is not None and getattr(scope, "zim_ids", None) is not None:
            return [a for a in self._archives if a.id in scope.zim_ids]
        return list(self._archives)

    def has_any_fulltext(self) -> bool:
        return any(a.has_fulltext_index for a in self._archives)

    async def lookup_aliases(self, terms: Any, *, max_aliases: int) -> list[str]:
        return []

    async def ids_for_labels(self, labels: Any) -> frozenset[int]:
        return frozenset()

    async def resolve_alias_targets(
        self, terms: list[str], *, zim_ids: Any = None, max_aliases: int
    ) -> list[tuple[int, str]]:
        self.resolve_calls.append((list(terms), zim_ids, max_aliases))
        return self._alias_targets[:max_aliases]


def _pq(
    terms: tuple[str, ...] = (), aliases: tuple[str, ...] = (), raw: str = "query"
) -> PreparedQuery:
    return PreparedQuery(
        raw=raw,
        terms=terms,
        text=raw,
        aliases=aliases,
        is_keyword_query=False,
        rung="initial",
        history=(),
    )


# ── Alias hit resolves directly ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_alias_hit_resolves_directly() -> None:
    """q.aliases non-empty ⇒ resolve_alias_targets is called and its exact
    (zim_id, path) pairs become candidates — no capability required."""
    archive = FakeArchive(id=7, suggest_paths=[])
    registry = FakeRegistry(archives=[archive], alias_targets=[(7, "A/Atmosphere_of_Earth")])
    source = AliasTitleResolve(archives=registry)  # type: ignore[arg-type]
    q = _pq(terms=("how", "much", "nitrogen", "in", "the", "air"), aliases=("Atmosphere of Earth",))
    out = await source.find(q, Scope(), Trace())
    assert len(out) == 1
    assert out[0].zim_id == 7
    assert out[0].path == "A/Atmosphere_of_Earth"
    assert out[0].source == "alias_title_resolve"
    assert out[0].rank == 0
    assert out[0].score is None
    # resolve_alias_targets got the aliases and the scoped zim_ids.
    assert registry.resolve_calls[0][0] == ["Atmosphere of Earth"]
    assert registry.resolve_calls[0][1] == frozenset({7})


@pytest.mark.asyncio
async def test_no_capability_gate() -> None:
    assert AliasTitleResolve.requires == frozenset()


# ── Exact title match ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_exact_title_match_found() -> None:
    """suggest() returns a path whose basename (underscore→space, lowercased)
    exactly equals the joined query terms."""
    archive = FakeArchive(id=1, suggest_paths=["A/Atmosphere_of_Earth"])
    registry = FakeRegistry(archives=[archive])
    source = AliasTitleResolve(archives=registry)  # type: ignore[arg-type]
    q = _pq(terms=("atmosphere", "of", "earth"))
    out = await source.find(q, Scope(), Trace())
    assert len(out) == 1
    assert out[0].path == "A/Atmosphere_of_Earth"
    assert out[0].source == "alias_title_resolve"


@pytest.mark.asyncio
async def test_near_but_not_exact_title_match_rejected() -> None:
    """A suggest() hit that is a superset/prefix but not an exact match must
    NOT become a candidate — this source only trusts exact matches."""
    archive = FakeArchive(id=1, suggest_paths=["A/Atmosphere_of_Earth_and_Venus", "A/Nitrogen"])
    registry = FakeRegistry(archives=[archive])
    source = AliasTitleResolve(archives=registry)  # type: ignore[arg-type]
    q = _pq(terms=("atmosphere", "of", "earth"))
    out = await source.find(q, Scope(), Trace())
    assert out == []


@pytest.mark.asyncio
async def test_exact_title_match_disabled_by_param() -> None:
    archive = FakeArchive(id=1, suggest_paths=["A/Atmosphere_of_Earth"])
    registry = FakeRegistry(archives=[archive])
    params = AliasTitleResolve.Params(exact_title_match=False)
    source = AliasTitleResolve(params=params, archives=registry)  # type: ignore[arg-type]
    q = _pq(terms=("atmosphere", "of", "earth"))
    out = await source.find(q, Scope(), Trace())
    assert out == []


@pytest.mark.asyncio
async def test_suggest_exception_degrades_to_empty() -> None:
    archive = FakeArchive(id=1, raise_on_suggest=True)
    registry = FakeRegistry(archives=[archive])
    source = AliasTitleResolve(archives=registry)  # type: ignore[arg-type]
    q = _pq(terms=("atmosphere", "of", "earth"))
    out = await source.find(q, Scope(), Trace())
    assert out == []


# ── Degradation: empty scope / no archives ───────────────────────────────────


@pytest.mark.asyncio
async def test_empty_scope_returns_empty() -> None:
    """A scope whose zim_ids resolve to no enabled archives degrades to []."""
    registry = FakeRegistry(archives=[])
    source = AliasTitleResolve(archives=registry)  # type: ignore[arg-type]
    q = _pq(terms=("a",), aliases=("b",))
    out = await source.find(q, Scope(), Trace())
    assert out == []


# ── Dedup between the two mechanisms ──────────────────────────────────────


@pytest.mark.asyncio
async def test_dedup_between_alias_and_title_match() -> None:
    """The same article nominated by both the alias resolver and the exact
    title matcher appears once, keeping the first (alias) occurrence, and the
    combined list is re-ranked 0..n-1."""
    archive = FakeArchive(id=3, suggest_paths=["A/Atmosphere_of_Earth"])
    registry = FakeRegistry(archives=[archive], alias_targets=[(3, "A/Atmosphere_of_Earth")])
    source = AliasTitleResolve(archives=registry)  # type: ignore[arg-type]
    q = _pq(terms=("atmosphere", "of", "earth"), aliases=("Atmosphere of Earth",))
    out = await source.find(q, Scope(), Trace())
    assert len(out) == 1
    assert out[0].path == "A/Atmosphere_of_Earth"
    assert out[0].rank == 0


@pytest.mark.asyncio
async def test_alias_and_distinct_title_match_both_kept_and_reranked() -> None:
    archive = FakeArchive(id=3, suggest_paths=["A/Nitrogen"])
    registry = FakeRegistry(archives=[archive], alias_targets=[(3, "A/Atmosphere_of_Earth")])
    source = AliasTitleResolve(archives=registry)  # type: ignore[arg-type]
    q = _pq(terms=("nitrogen",), aliases=("Atmosphere of Earth",))
    out = await source.find(q, Scope(), Trace())
    paths = {c.path for c in out}
    assert paths == {"A/Atmosphere_of_Earth", "A/Nitrogen"}
    ranks = sorted(c.rank for c in out)
    assert ranks == [0, 1]
