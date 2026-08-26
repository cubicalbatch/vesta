"""Multi-archive rank-contract tests (AUDIT_0822 M4).

The ``Candidate`` contract scopes ``rank`` to its ``(zim_id, source)`` group
only — RRF consumes ``1/(k + rank)`` per group, so a global counter over the
concatenated per-archive results offsets every archive after the first,
shrinking its RRF mass and flipping within-archive source order. Invisible to
single-archive benchmarks.

The load-bearing properties covered here:

* every candidate source emits ranks that restart at 0 within each
  ``(zim_id, source)`` group on a multi-archive scope;
* **single-archive invariance**: an archive's emitted candidates are identical
  whether it is alone in the scope or accompanied (offset 0 ⇒ byte-equal);
* RRF-level consequence: with group-local ranks, global RRF interleaves two
  archives' hit lists instead of letting archive #1 dominate, and
  within-archive fused order no longer depends on other archives' volume;
* end-to-end: a two-fake-archive pipeline produces passages from both.

Fakes stand in for the registry/archives/store/encoders per the existing
candidate-source test patterns.
"""

from __future__ import annotations

import asyncio
import itertools
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from vesta.config.capabilities import Capability
from vesta.retrieval.contracts import (
    Candidate,
    FusionKey,
    PreparedQuery,
    Scope,
)
from vesta.retrieval.impls.alias_title_resolve import AliasTitleResolve
from vesta.retrieval.impls.rrf import RRF, _rrf_fuse
from vesta.retrieval.impls.title_suggest import TitleSuggest
from vesta.retrieval.impls.vector_knn import VectorKnn
from vesta.retrieval.impls.xapian_fts import XapianFTS
from vesta.retrieval.pipeline import Deps, run_pipeline
from vesta.retrieval.profiles import ProfileComponent, RetrievalProfile
from vesta.retrieval.trace import Trace
from vesta.vectors.contracts import IndexMeta, VectorHit
from vesta.zim.types import EntryFlags, ExtractedArticle

# ── Fakes ────────────────────────────────────────────────────────────────────


@dataclass
class FakeLexArchive:
    """Minimal Archive fake — configurable search/suggest results per archive."""

    id: int = 1
    uuid: str = "fake-uuid"
    title: str = "Fake Archive"
    language: str = "en"
    has_fulltext_index: bool = True
    article_count: int = 10
    search_paths: list[str] = field(default_factory=list)
    suggest_paths: list[str] = field(default_factory=list)
    raise_on_search: bool = False

    async def search(self, terms: Any, limit: int) -> list[str]:
        if self.raise_on_search:
            raise RuntimeError("boom")
        return list(self.search_paths)[:limit]

    async def suggest(self, prefix: str, limit: int) -> list[str]:
        return list(self.suggest_paths)[:limit]

    async def read(self, path: str) -> None:
        raise NotImplementedError

    async def extract(self, path: str) -> ExtractedArticle:
        name = path.rsplit("/", 1)[-1].replace("_", " ")
        return ExtractedArticle(
            path=path,
            title=name,
            text=(f"{name} article. " * 20),
            sections=(),
            flags=EntryFlags.NONE,
        )

    async def main_path(self) -> str:
        return f"A{self.id}/Article_1"


class FakeLexRegistry:
    """Minimal ArchiveRegistry fake for the lexical sources + pipeline."""

    def __init__(
        self,
        archives: list[FakeLexArchive] | None = None,
        alias_targets: list[tuple[int, str]] | None = None,
    ) -> None:
        self._archives = archives if archives is not None else [FakeLexArchive()]
        self._alias_targets = alias_targets or []

    def get(self, zim_id: int) -> FakeLexArchive:
        return next(a for a in self._archives if a.id == zim_id)

    def enabled(self, scope: Any = None) -> list[FakeLexArchive]:
        if scope is not None and getattr(scope, "zim_ids", None) is not None:
            return [a for a in self._archives if a.id in scope.zim_ids]
        return list(self._archives)

    def has_any_fulltext(self) -> bool:
        return any(a.has_fulltext_index for a in self._archives)

    async def ids_for_labels(self, labels: Any) -> frozenset[int]:
        return frozenset()

    async def lookup_aliases(self, terms: Any, *, max_aliases: int) -> list[str]:
        return []

    async def resolve_alias_targets(
        self, terms: list[str], *, zim_ids: Any = None, max_aliases: int
    ) -> list[tuple[int, str]]:
        pairs = [
            (zid, path) for zid, path in self._alias_targets if zim_ids is None or zid in zim_ids
        ]
        return pairs[:max_aliases]


def _pq(raw: str = "test query", aliases: tuple[str, ...] = ()) -> PreparedQuery:
    return PreparedQuery(
        raw=raw,
        terms=tuple(raw.lower().split()),
        text=raw,
        aliases=aliases,
        is_keyword_query=False,
        rung="initial",
        history=(),
    )


def _cand(zim: int, path: str, rank: int, source: str = "xapian_fts") -> Candidate:
    return Candidate(zim_id=zim, path=path, source=source, rank=rank, score=None)


# ── xapian_fts ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_xapian_fts_ranks_restart_per_archive() -> None:
    registry = FakeLexRegistry(
        archives=[
            FakeLexArchive(id=1, search_paths=["A/P1", "A/P2", "A/P3"]),
            FakeLexArchive(id=2, search_paths=["B/Q1", "B/Q2"]),
        ]
    )
    out = await XapianFTS(archives=registry).find(_pq(), Scope(), Trace())  # type: ignore[arg-type]
    assert [(c.zim_id, c.path, c.rank) for c in out] == [
        (1, "A/P1", 0),
        (1, "A/P2", 1),
        (1, "A/P3", 2),
        (2, "B/Q1", 0),
        (2, "B/Q2", 1),
    ]


@pytest.mark.asyncio
async def test_xapian_fts_single_archive_output_invariant_to_extra_archives() -> None:
    """Offset-0 invariant: archive 1's candidates are identical alone or when
    another archive joins the scope."""
    one = await XapianFTS(
        archives=FakeLexRegistry([FakeLexArchive(id=1, search_paths=["A/P1", "A/P2"])])
    ).find(_pq(), Scope(), Trace())
    both = await XapianFTS(
        archives=FakeLexRegistry(
            [
                FakeLexArchive(id=1, search_paths=["A/P1", "A/P2"]),
                FakeLexArchive(id=2, search_paths=["B/Q1", "B/Q2", "B/Q3"]),
            ]
        )
    ).find(_pq(), Scope(), Trace())
    assert [c for c in one if c.zim_id == 1] == [c for c in both if c.zim_id == 1]
    assert one[0].rank == 0


# ── title_suggest ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_title_suggest_ranks_restart_per_archive() -> None:
    registry = FakeLexRegistry(
        archives=[
            FakeLexArchive(id=1, suggest_paths=["A/S1", "A/S2"]),
            FakeLexArchive(id=2, suggest_paths=["B/T1"]),
            FakeLexArchive(id=3, suggest_paths=["C/U1", "C/U2", "C/U3"]),
        ]
    )
    out = await TitleSuggest(archives=registry).find(_pq(), Scope(), Trace())  # type: ignore[arg-type]
    assert [(c.zim_id, c.rank) for c in out] == [
        (1, 0),
        (1, 1),
        (2, 0),
        (3, 0),
        (3, 1),
        (3, 2),
    ]


@pytest.mark.asyncio
async def test_title_suggest_single_archive_output_invariant_to_extra_archives() -> None:
    one = await TitleSuggest(
        archives=FakeLexRegistry([FakeLexArchive(id=1, suggest_paths=["A/S1"])])
    ).find(_pq(), Scope(), Trace())
    both = await TitleSuggest(
        archives=FakeLexRegistry(
            [
                FakeLexArchive(id=1, suggest_paths=["A/S1"]),
                FakeLexArchive(id=2, suggest_paths=["B/T1", "B/T2"]),
            ]
        )
    ).find(_pq(), Scope(), Trace())
    assert one == [c for c in both if c.zim_id == 1]


# ── vector_knn ───────────────────────────────────────────────────────────────

_META = IndexMeta(
    embedder_id="embed/model",
    dim=4,
    query_prefix="",
    passage_prefix="",
    pooling="mean",
    normalize=True,
)


class _FakeVectorArchive:
    def __init__(self, zim_id: int) -> None:
        self.id = zim_id


class _FakeVectorRegistry:
    def __init__(self, ids: list[int]) -> None:
        self._archives = [_FakeVectorArchive(i) for i in ids]

    def enabled(self, scope: Any = None) -> list[_FakeVectorArchive]:
        if scope is not None and scope.zim_ids is not None:
            return [a for a in self._archives if a.id in scope.zim_ids]
        return list(self._archives)


class _FakeEncoder:
    id = "embed/model"
    dim = 4
    query_prefix = ""
    passage_prefix = ""
    pooling = "mean"
    normalize = True

    async def embed(self, texts: list[str], *, kind: str) -> list[np.ndarray]:
        return [np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) for _ in texts]


class _FakeEncoders:
    def __init__(self, encoder: Any) -> None:
        self._encoder = encoder

    async def get_embed(self) -> Any:
        return self._encoder


class _FakeStore:
    def __init__(self, metas: dict[int, IndexMeta], hits: list[VectorHit]) -> None:
        self._metas = metas
        self._hits = hits

    async def describe(self, zim_id: int) -> IndexMeta | None:
        return self._metas.get(zim_id)

    async def search(self, q: Any, *, zim_ids: list[int], k: int) -> list[VectorHit]:
        return [h for h in self._hits if h.zim_id in zim_ids][:k]


def _hit(zim: int, art: int, chunk: int, path: str, score: float) -> VectorHit:
    return VectorHit(
        zim_id=zim,
        article_id=art,
        chunk_id=chunk,
        path=path,
        score=score,
        char_start=0,
        char_end=100,
    )


@pytest.mark.asyncio
async def test_vector_knn_ranks_restart_per_archive_on_interleaved_hits() -> None:
    """Hits arrive score-interleaved across archives; ranks must count per zim."""
    hits = [
        _hit(1, 10, 1, "A/One", -0.05),  # z1 best
        _hit(2, 20, 1, "B/Four", -0.10),  # z2 best — would be rank 1 under the old global counter
        _hit(1, 11, 1, "A/Two", -0.15),  # z1 second — was rank 2 globally
        _hit(2, 21, 1, "B/Five", -0.20),
        _hit(1, 12, 1, "A/Three", -0.25),
    ]
    src = VectorKnn(
        archives=_FakeVectorRegistry([1, 2]),  # type: ignore[arg-type]
        vectors=_FakeStore({1: _META, 2: _META}, hits),  # type: ignore[arg-type]
        encoders=_FakeEncoders(_FakeEncoder()),  # type: ignore[arg-type]
    )
    out = await src.find(_pq("photosynthesis"), Scope(), Trace())
    assert [(c.zim_id, c.path, c.rank) for c in out] == [
        (1, "A/One", 0),
        (2, "B/Four", 0),
        (1, "A/Two", 1),
        (2, "B/Five", 1),
        (1, "A/Three", 2),
    ]


@pytest.mark.asyncio
async def test_vector_knn_single_archive_output_invariant_to_extra_archives() -> None:
    hits_one = [_hit(1, 10, 1, "A/One", -0.05), _hit(1, 11, 1, "A/Two", -0.15)]
    hits_both = [*hits_one, _hit(2, 20, 1, "B/Four", -0.10)]
    src_one = VectorKnn(
        archives=_FakeVectorRegistry([1]),  # type: ignore[arg-type]
        vectors=_FakeStore({1: _META}, hits_one),  # type: ignore[arg-type]
        encoders=_FakeEncoders(_FakeEncoder()),  # type: ignore[arg-type]
    )
    src_both = VectorKnn(
        archives=_FakeVectorRegistry([1, 2]),  # type: ignore[arg-type]
        vectors=_FakeStore({1: _META, 2: _META}, hits_both),  # type: ignore[arg-type]
        encoders=_FakeEncoders(_FakeEncoder()),  # type: ignore[arg-type]
    )
    q = _pq("photosynthesis")
    one = await src_one.find(q, Scope(zim_ids=frozenset({1})), Trace())
    both = await src_both.find(q, Scope(zim_ids=frozenset({1})), Trace())
    assert one == both


# ── alias_title_resolve ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_alias_title_resolve_ranks_restart_per_archive() -> None:
    """Alias targets land in two archives; the old global renumber gave the
    second archive's matches ranks 2..n — they must restart at 0."""
    registry = FakeLexRegistry(
        archives=[
            FakeLexArchive(id=7),
            FakeLexArchive(id=9, suggest_paths=["Nitrogen"]),
        ],
        alias_targets=[(7, "A/X"), (9, "B/Y"), (7, "A/Z")],
    )
    src = AliasTitleResolve(archives=registry)  # type: ignore[arg-type]
    q = _pq(raw="nitrogen", aliases=("Atmosphere",))
    out = await src.find(q, Scope(), Trace())
    assert [(c.zim_id, c.path, c.rank) for c in out] == [
        (7, "A/X", 0),
        (9, "B/Y", 0),
        (7, "A/Z", 1),
        (9, "Nitrogen", 1),
    ]


# ── RRF consequences ─────────────────────────────────────────────────────────


def _fts_groups(buggy_second_block: bool) -> dict[FusionKey, list[Candidate]]:
    """Two equal-strength archives' FTS lists. With ``buggy_second_block``, the
    second archive's ranks are renumbered 5..9 as a global counter would emit."""
    second = [(5 + i) if buggy_second_block else i for i in range(5)]
    return {
        FusionKey(1, "xapian_fts"): [_cand(1, f"A/{i}", i) for i in range(5)],
        FusionKey(2, "xapian_fts"): [_cand(2, f"B/{i}", r) for i, r in enumerate(second)],
    }


def test_rrf_global_mode_interleaves_with_group_local_ranks() -> None:
    """Global RRF over two equal archives: contract-correct ranks interleave
    (both rank-0 candidates top), while the old offset input lets archive #1
    sweep the top five."""
    fuser = RRF(RRF.Params(k=20, across_archives="rrf"))
    tr = Trace()

    fused = fuser.fuse(_fts_groups(buggy_second_block=False), tr)
    assert [c.path for c in fused[:2]] == ["A/0", "B/0"]
    assert [c.zim_id for c in fused[:6]] == [1, 2, 1, 2, 1, 2]

    dominated = fuser.fuse(_fts_groups(buggy_second_block=True), tr)
    assert {c.zim_id for c in dominated[:5]} == {1}, (
        "offset ranks let archive #1 dominate — exactly the M4 bias"
    )


def test_rrf_union_mode_within_archive_order_ignores_other_archives_volume() -> None:
    """Within one archive, fused order must depend only on group-local ranks.
    An fts hit at correct rank 0 outranks a deep suggest-only hit; the same hit
    carrying a global-counter offset (60 earlier archives' hits) would lose to it."""
    tr = Trace()

    correct = {
        FusionKey(2, "xapian_fts"): [_cand(2, "B/P", 0)],
        FusionKey(2, "title_suggest"): [_cand(2, "B/Q", 10)],
    }
    fused = RRF(RRF.Params(k=20, across_archives="union")).fuse(correct, tr)
    assert [c.path for c in fused] == ["B/P", "B/Q"]

    offset = {
        FusionKey(2, "xapian_fts"): [_cand(2, "B/P", 60)],
        FusionKey(2, "title_suggest"): [_cand(2, "B/Q", 10)],
    }
    flipped = RRF(RRF.Params(k=20, across_archives="union")).fuse(offset, tr)
    assert [c.path for c in flipped] == ["B/Q", "B/P"], (
        "an offset fts rank flips the within-archive order the contract forbids"
    )


# ── RRF determinism (AUDIT_0824 N41) ─────────────────────────────────────────


def _tied_groups() -> dict[FusionKey, list[Candidate]]:
    """One archive, three sources of 100 disjoint candidates each.

    Every candidate appears in exactly one group, so all candidates sharing a
    rank are exact RRF-score ties (1/(k + r) each) — the deliberate
    ``rank_offset`` collisions the built-in profiles construct, at a scale
    where the downstream ``max_articles=40`` cut lands inside the tie field.
    """
    sources = ("xapian_fts", "title_suggest", "title_entity_suggest")
    return {
        FusionKey(1, src): [_cand(1, f"A/{src[0]}{i}", i, source=src) for i in range(100)]
        for src in sources
    }


def test_rrf_tie_order_is_independent_of_group_delivery_order() -> None:
    """The same candidate groups delivered in any dict insertion order fuse to
    a byte-identical ranking — including who survives the top-40 truncation."""
    fuser = RRF(RRF.Params(k=20, across_archives="union"))
    tr = Trace()
    groups = _tied_groups()

    reference: list[Candidate] | None = None
    for perm in itertools.permutations(groups):
        fused = fuser.fuse({k: groups[k] for k in perm}, tr)
        if reference is None:
            reference = fused
            assert len(fused) == len({c.path for c in fused})
        else:
            assert fused == reference, "group delivery order changed the fused ranking"

    # The candidate_articles cap (max_articles=40) cuts inside the tie field:
    # the survivor set must be stable too.
    assert reference is not None
    survivors = [c.path for c in reference[:40]]
    assert len(survivors) == 40
    for perm in itertools.permutations(groups):
        permuted = fuser.fuse({k: groups[k] for k in perm}, tr)
        assert [c.path for c in permuted[:40]] == survivors


def test_rrf_flat_input_is_permutation_invariant() -> None:
    """_rrf_fuse over one flattened list is invariant under input shuffles —
    float summation order must not leak into scores or ordering."""
    flat = [c for cands in _tied_groups().values() for c in cands]
    base = _rrf_fuse(flat)
    rng = random.Random(824)
    for _ in range(25):
        shuffled = flat[:]
        rng.shuffle(shuffled)
        assert _rrf_fuse(shuffled) == base


# ── End-to-end: two fake archives through the pipeline ───────────────────────


def _two_archive_profile() -> RetrievalProfile:
    return RetrievalProfile(
        name="multi",
        description="Multi-archive lexical profile",
        hash="multi-hash",
        preparers=(),
        sources=(
            ProfileComponent(impl="xapian_fts", params={"limit": 5}),
            ProfileComponent(impl="title_suggest", params={"limit": 5}),
        ),
        fusion=ProfileComponent(impl="rrf", params={"k": 20, "across_archives": "union"}),
        passages=ProfileComponent(
            impl="candidate_articles", params={"max_articles": 8, "max_passages": 12}
        ),
        scorers=(),
        assembler=ProfileComponent(
            impl="topk_budget",
            params={"budget_tokens": 4000, "max_per_article": 2, "dedup": "none"},
        ),
    )


@pytest.mark.asyncio
async def test_pipeline_two_archives_produces_passages_from_both() -> None:
    archives = [
        FakeLexArchive(
            id=1,
            search_paths=["A/Photosynthesis", "A/Chloroplast"],
            suggest_paths=["A/Photosynthesis"],
        ),
        FakeLexArchive(id=2, search_paths=["B/Keratin", "B/Collagen"], suggest_paths=["B/Keratin"]),
    ]
    deps = Deps(
        archives=FakeLexRegistry(archives),  # type: ignore[arg-type]
        capabilities=frozenset({Capability.ZIM_FULLTEXT}),
        semaphore=asyncio.Semaphore(4),
    )
    result = await run_pipeline(
        profile=_two_archive_profile(),
        query="photosynthesis keratin",
        scope=Scope(),
        deps=deps,
    )
    assert result.passages, "expected passages from both archives"
    assert {sp.passage.zim_id for sp in result.passages} == {1, 2}
