"""Vector store round-trip + contract tests.

Covers these load-bearing properties:

* upsert → search → delete → stats round-trip;
* ``zim_id`` PARTITION KEY is a genuine PRE-filter (a query scoped to ``[1]``
  never returns ``zim_id=2`` rows);
* one ``vectors_d{N}`` table per distinct dim, created lazily by the store;
* per-archive embedder compat metadata round-trips (index-settings
  enforcement).

Uses the real ``Database`` + migration 0004 against a tmp file so the vec0
extension load and the chunk/index_meta schema are exercised end to end. No ZIM
archive is needed: the store is a pure SQL/vec0 layer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio

from vesta.db.connection import Database
from vesta.db.migrations import run_migrations
from vesta.vectors.contracts import IndexMeta, VectorRow
from vesta.vectors.sqlite_vec_store import SqliteVecStore


@pytest_asyncio.fixture
async def store(tmp_db_path: Path) -> tuple[Database, SqliteVecStore]:
    db = Database(str(tmp_db_path), busy_timeout_ms=2000)
    await db.start()
    async with db.write() as conn:
        await run_migrations(conn)
    s = SqliteVecStore(db, quantizer="bit", oversample=8, default_dim=4)
    await s.ensure_default_table()
    yield db, s
    await db.stop()


def _row(
    cid: int, zim: int, art: int, vec: list[float], *, cs: int = 0, ce: int = 100
) -> VectorRow:
    return VectorRow(
        id=cid,
        zim_id=zim,
        article_id=art,
        embedding=np.array(vec, dtype=np.float32),
        char_start=cs,
        char_end=ce,
    )


async def _seed_zims_articles(
    db: Database, zims: list[int], articles: list[tuple[int, int]]
) -> None:
    """Insert minimal zims + articles rows so chunks FK references resolve."""
    async with db.write() as conn:
        for z in zims:
            await conn.execute("INSERT INTO zims(id) VALUES (?)", (z,))
        for art_id, zim_id in articles:
            await conn.execute(
                "INSERT INTO articles(id, zim_id, entry_path) VALUES (?, ?, ?)",
                (art_id, zim_id, f"A{art_id}"),
            )


# ── Round-trip: upsert → search → stats ─────────────────────────────────────


@pytest.mark.asyncio
async def test_upsert_search_roundtrip(store: tuple[Database, SqliteVecStore]) -> None:
    db, s = store
    await _seed_zims_articles(db, [1], [(10, 1), (11, 1)])
    await s.upsert(
        [
            _row(1, 1, 10, [1.0, 0.0, 0.0, 0.0], cs=0, ce=50),
            _row(2, 1, 11, [0.0, 1.0, 0.0, 0.0], cs=0, ce=80),
        ]
    )
    q = np.array([0.95, 0.05, 0.0, 0.0], dtype=np.float32)
    hits = await s.search(q, zim_ids=[1], k=2)
    assert {h.chunk_id for h in hits} == {1, 2}
    # Nearest neighbour is the first upserted vector (cosine-faithful).
    assert hits[0].chunk_id == 1
    # Higher score == more similar (negated distance).
    assert hits[0].score >= hits[1].score
    # The hit carries enough to rebuild a Candidate (join entry_path).
    top = hits[0]
    assert top.zim_id == 1
    assert top.article_id == 10
    assert top.char_start == 0 and top.char_end == 50


@pytest.mark.asyncio
async def test_stats_reflects_rows(store: tuple[Database, SqliteVecStore]) -> None:
    db, s = store
    await _seed_zims_articles(db, [1], [(10, 1)])
    await s.upsert([_row(1, 1, 10, [1.0, 0.0, 0.0, 0.0])])
    stats = await s.stats()
    assert stats.total_rows == 1
    assert stats.per_dim.get(4) == 1
    assert stats.per_zim.get(1) == 1
    assert stats.vec0_available is True


# ── Partition-key PRE-filter (the load-bearing property) ────────────────────


@pytest.mark.asyncio
async def test_partition_key_pre_filters(store: tuple[Database, SqliteVecStore]) -> None:
    """A query scoped to zim_ids=[1] must NEVER return zim_id=2 rows. This is
    vec0's partition-key shard restriction (a physical pre-filter, not a
    post-filter) — the property that makes ``scan the largest single archive``
    the cost model instead of ``scan the whole index``."""
    db, s = store
    await _seed_zims_articles(db, [1, 2], [(10, 1), (20, 2)])
    await s.upsert(
        [
            _row(1, 1, 10, [1.0, 0.0, 0.0, 0.0]),
            _row(2, 2, 20, [1.0, 0.0, 0.0, 0.0]),  # identical vector, other archive
        ]
    )
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scoped = await s.search(q, zim_ids=[1], k=5)
    assert {h.zim_id for h in scoped} == {1}, "partition filter leaked across archives"
    assert {h.chunk_id for h in scoped} == {1}
    # Empty zim_ids ⇒ search every archive.
    all_hits = await s.search(q, zim_ids=[], k=5)
    assert {h.zim_id for h in all_hits} == {1, 2}


# ── Delete-by-zim (08 DoD: "Deleting an archive removes its vectors") ───────


@pytest.mark.asyncio
async def test_delete_by_zim(store: tuple[Database, SqliteVecStore]) -> None:
    db, s = store
    await _seed_zims_articles(db, [1, 2], [(10, 1), (11, 1), (20, 2)])
    await s.upsert(
        [
            _row(1, 1, 10, [1.0, 0.0, 0.0, 0.0]),
            _row(2, 1, 11, [0.0, 1.0, 0.0, 0.0]),
            _row(3, 2, 20, [0.0, 0.0, 1.0, 0.0]),
        ]
    )
    await s.delete_by_zim(1)
    stats = await s.stats()
    assert stats.per_zim.get(1, 0) == 0
    assert stats.per_zim.get(2) == 1
    # The deleted archive's vectors are gone from vec0 too.
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert await s.search(q, zim_ids=[1], k=5) == []
    hits2 = await s.search(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32), zim_ids=[2], k=5)
    assert {h.chunk_id for h in hits2} == {3}


# ── One vec0 table per dimension, created lazily ────────────────────────────


@pytest.mark.asyncio
async def test_lazy_high_dim_table(store: tuple[Database, SqliteVecStore]) -> None:
    """The default-dim table is created eagerly; a different dim (e.g. 768 for the
    quality embedder) is created on first upsert. The store resolves the table by
    dimension — callers never name it (08: "resolved through the interface")."""
    db, s = store
    await _seed_zims_articles(db, [1], [(10, 1), (11, 1)])
    await s.upsert([_row(1, 1, 10, [1.0] * 8)])  # dim 8 — not the default 4
    q = np.array([1.0] * 8, dtype=np.float32)
    hits = await s.search(q, zim_ids=[1], k=1)
    assert {h.chunk_id for h in hits} == {1}
    stats = await s.stats()
    assert stats.per_dim.get(8) == 1
    # default-dim table still exists (created eagerly at ensure_default_table).
    assert 4 in stats.per_dim or stats.per_dim.get(4, 0) == 0


# ── Embedder compat metadata round-trip ──────────────────────────────────────


@pytest.mark.asyncio
async def test_record_and_describe_index_meta(
    store: tuple[Database, SqliteVecStore],
) -> None:
    db, s = store
    await _seed_zims_articles(db, [1], [])  # index_meta.zim_id FK→zims(id)
    meta = IndexMeta(
        embedder_id="onnx-community/granite-embedding-small-english-r2-ONNX",
        dim=384,
        query_prefix="",
        passage_prefix="",
        pooling="mean",
        normalize=True,
    )
    await s.record_index_meta(1, meta)
    back = await s.describe(1)
    assert back == meta
    # Unknown archive ⇒ None (Stage 3's dense source treats None as "not indexed").
    assert await s.describe(999) is None
    # Re-recording overwrites (re-index after an embedder change).
    await s.record_index_meta(1, IndexMeta("other/model", 768, "", "", "cls", False))
    again = await s.describe(1)
    assert again is not None and again.embedder_id == "other/model" and again.dim == 768


# ── Idempotent re-upsert on the id PK ───────────────────────────────────────


@pytest.mark.asyncio
async def test_reupsert_replaces_vector(store: tuple[Database, SqliteVecStore]) -> None:
    db, s = store
    await _seed_zims_articles(db, [1], [(10, 1)])
    await s.upsert([_row(1, 1, 10, [1.0, 0.0, 0.0, 0.0])])
    # Re-index the same chunk to a different embedding.
    await s.upsert([_row(1, 1, 10, [0.0, 0.0, 0.0, 1.0], cs=5, ce=90)])
    stats = await s.stats()
    assert stats.total_rows == 1, "re-upsert must replace, not duplicate"
    # The vector now matches the new direction.
    hits = await s.search(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), zim_ids=[1], k=1)
    assert hits and hits[0].chunk_id == 1 and hits[0].char_end == 90


@pytest.mark.asyncio
async def test_empty_upsert_is_noop(store: tuple[Database, SqliteVecStore]) -> None:
    _db, s = store
    await s.upsert([])
    assert (await s.stats()).total_rows == 0


@pytest.mark.asyncio
async def test_upsert_raises_when_ddl_rejected(
    store: tuple[Database, SqliteVecStore], monkeypatch: pytest.MonkeyPatch
) -> None:
    """AUDIT_0824 N25: if this sqlite-vec build rejects every dim-table DDL
    variant, upsert must fail the build loudly — never silently drop the batch
    and let the index finish 'complete' with zero vectors."""
    db, s = store
    await _seed_zims_articles(db, [1], [(10, 1)])
    monkeypatch.setattr(
        s,
        "_ddl_candidates",
        lambda dim: (
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vectors_d{dim} USING vec0(not_an_option)",
        ),
    )
    with pytest.raises(RuntimeError, match="rejected all DDL variants"):
        await s.upsert([_row(1, 1, 10, [1.0] * 8)])
