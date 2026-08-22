"""Stage A3 dense candidate source tests.

The load-bearing properties:

* ``requires = {Capability.VECTORS}`` — a depth-0 box drops the source by
  capability, no branching in the pipeline;
* it is the first source returning ``Candidate`` **with** a score;
* per-article dedup: depths 2/3 can hit several chunks of one article, Stage A
  nominates articles, so the strongest hit per article wins;
* index-settings enforcement: an archive whose recorded embedder differs from
  the live query embedder is REFUSED with a trace entry, never searched — a
  mismatched embedder returns plausible-looking garbage.

Fakes stand in for the store/manager/registry; the scope path exercises the
real ``archives_for_scope`` helper.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vesta.config.capabilities import Capability
from vesta.retrieval.contracts import PreparedQuery, Scope
from vesta.retrieval.impls.vector_knn import VectorKnn
from vesta.retrieval.trace import Trace
from vesta.vectors.contracts import IndexMeta, VectorHit

_META = IndexMeta(
    embedder_id="embed/model",
    dim=4,
    query_prefix="",
    passage_prefix="",
    pooling="mean",
    normalize=True,
)


class _FakeArchive:
    def __init__(self, zim_id: int) -> None:
        self.id = zim_id


class _FakeRegistry:
    """The slice of ``ArchiveRegistry`` ``archives_for_scope`` touches."""

    def __init__(self, ids: list[int]) -> None:
        self._archives = [_FakeArchive(i) for i in ids]

    def enabled(self, scope: Any = None) -> list[_FakeArchive]:
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
        self.searched_zim_ids: list[int] | None = None

    async def describe(self, zim_id: int) -> IndexMeta | None:
        return self._metas.get(zim_id)

    async def search(self, q: Any, *, zim_ids: list[int], k: int) -> list[VectorHit]:
        self.searched_zim_ids = list(zim_ids)
        return [h for h in self._hits if h.zim_id in zim_ids][:k]


def _query() -> PreparedQuery:
    return PreparedQuery(
        raw="what is photosynthesis",
        terms=("what", "is", "photosynthesis"),
        text="what is photosynthesis",
        aliases=(),
        is_keyword_query=False,
        rung="test",
    )


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


# ── capability + construction ────────────────────────────────────────────────


def test_requires_vectors_capability() -> None:
    assert VectorKnn.requires == frozenset({Capability.VECTORS})


def test_constructs_through_the_di_ladder() -> None:
    # The pipeline's first ladder rung passes params+archives+vectors+encoders.
    src = VectorKnn(
        params=VectorKnn.Params(k=5),
        archives=_FakeRegistry([1]),  # type: ignore[arg-type]
        vectors=_FakeStore({}, []),  # type: ignore[arg-type]
        encoders=_FakeEncoders(_FakeEncoder()),  # type: ignore[arg-type]
    )
    assert src._params.k == 5


# ── empty-path guards ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("params", "vectors", "encoders"),
    [
        (VectorKnn.Params(), None, _FakeEncoders(_FakeEncoder())),
        (
            VectorKnn.Params(enabled=False),
            _FakeStore({1: _META}, [_hit(1, 1, 1, "A/X", -0.1)]),
            _FakeEncoders(_FakeEncoder()),
        ),
        (VectorKnn.Params(), _FakeStore({1: _META}, []), _FakeEncoders(None)),
    ],
)
async def test_missing_dependency_or_disabled_yields_no_candidates(
    params: VectorKnn.Params, vectors: Any, encoders: Any
) -> None:
    src = VectorKnn(
        params=params,
        archives=_FakeRegistry([1]),  # type: ignore[arg-type]
        vectors=vectors,  # type: ignore[arg-type]
        encoders=encoders,  # type: ignore[arg-type]
    )
    assert await src.find(_query(), Scope(), Trace()) == []
    if isinstance(vectors, _FakeStore):
        assert vectors.searched_zim_ids is None


# ── the happy path: scored candidates, deduped per article ───────────────────


@pytest.mark.asyncio
async def test_scored_candidates_deduped_per_article() -> None:
    hits = [
        _hit(1, 10, 1, "A/Photosynthesis", -0.05),  # strongest chunk of the article
        _hit(1, 10, 2, "A/Photosynthesis", -0.20),  # same article, weaker chunk
        _hit(1, 11, 3, "A/Chloroplast", -0.30),
    ]
    store = _FakeStore({1: _META}, hits)
    src = VectorKnn(
        params=VectorKnn.Params(k=10),
        archives=_FakeRegistry([1]),  # type: ignore[arg-type]
        vectors=store,  # type: ignore[arg-type]
        encoders=_FakeEncoders(_FakeEncoder()),  # type: ignore[arg-type]
    )
    out = await src.find(_query(), Scope(), Trace())
    assert store.searched_zim_ids == [1]
    assert [c.path for c in out] == ["A/Photosynthesis", "A/Chloroplast"]
    # Dense is the first source that can return a real score.
    assert all(c.score is not None for c in out)
    assert out[0].score == pytest.approx(-0.05)
    assert [c.rank for c in out] == [0, 1]
    assert all(c.source == "vector_knn" for c in out)


@pytest.mark.asyncio
async def test_scope_restricts_searched_archives() -> None:
    store = _FakeStore(
        {1: _META, 2: _META}, [_hit(1, 1, 1, "A/X", -0.1), _hit(2, 2, 2, "A/Y", -0.1)]
    )
    src = VectorKnn(
        archives=_FakeRegistry([1, 2]),  # type: ignore[arg-type]
        vectors=store,  # type: ignore[arg-type]
        encoders=_FakeEncoders(_FakeEncoder()),  # type: ignore[arg-type]
    )
    out = await src.find(_query(), Scope(zim_ids=frozenset({2})), Trace())
    assert store.searched_zim_ids == [2]
    assert [c.zim_id for c in out] == [2]


# ── index-settings enforcement ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mismatched_embedder_refused_with_trace_never_searched() -> None:
    mismatched = IndexMeta(
        embedder_id="other/model",
        dim=768,
        query_prefix="",
        passage_prefix="",
        pooling="mean",
        normalize=True,
    )
    store = _FakeStore({1: mismatched}, [_hit(1, 1, 1, "A/X", -0.1)])
    src = VectorKnn(
        archives=_FakeRegistry([1]),  # type: ignore[arg-type]
        vectors=store,  # type: ignore[arg-type]
        encoders=_FakeEncoders(_FakeEncoder()),  # type: ignore[arg-type]
    )
    tr = Trace()
    out = await src.find(_query(), Scope(), tr)
    assert out == []
    assert store.searched_zim_ids is None, "a mismatched index is never searched"
    degs = tr.to_dict()["degradations"]
    assert len(degs) == 1 and "refused" in degs[0]["reason"]
    assert "other/model" in degs[0]["reason"]


@pytest.mark.asyncio
async def test_unindexed_archive_skipped_silently() -> None:
    # describe() → None ("no index") is a skip, not a refusal: archives at
    # depth 0 are the norm, not a degradation worth tracing per query.
    store = _FakeStore({}, [_hit(1, 1, 1, "A/X", -0.1)])
    src = VectorKnn(
        archives=_FakeRegistry([1]),  # type: ignore[arg-type]
        vectors=store,  # type: ignore[arg-type]
        encoders=_FakeEncoders(_FakeEncoder()),  # type: ignore[arg-type]
    )
    tr = Trace()
    assert await src.find(_query(), Scope(), tr) == []
    assert store.searched_zim_ids is None
    assert tr.to_dict()["degradations"] == []


@pytest.mark.asyncio
async def test_mixed_archives_search_only_compatible() -> None:
    mismatched = IndexMeta(
        embedder_id="other/model",
        dim=768,
        query_prefix="",
        passage_prefix="",
        pooling="mean",
        normalize=True,
    )
    store = _FakeStore(
        {1: _META, 2: mismatched},
        [_hit(1, 1, 1, "A/X", -0.1), _hit(2, 2, 2, "A/Y", -0.1)],
    )
    tr = Trace()
    src = VectorKnn(
        archives=_FakeRegistry([1, 2]),  # type: ignore[arg-type]
        vectors=store,  # type: ignore[arg-type]
        encoders=_FakeEncoders(_FakeEncoder()),  # type: ignore[arg-type]
    )
    out = await src.find(_query(), Scope(), tr)
    assert store.searched_zim_ids == [1], "only the compatible archive is searched"
    assert [c.zim_id for c in out] == [1]
    assert len(tr.to_dict()["degradations"]) == 1
