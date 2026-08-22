"""``sqlite-vec`` ``vec0``-backed :class:`~vesta.vectors.contracts.VectorStore`.

One file, one transaction, one ``cp`` for backup: the vec0 virtual tables live in
the same ``vesta.db`` as the metadata. ``zim_id`` is a vec0
PARTITION KEY — a genuine physical pre-filter,
so a scoped query scans only the named archive(s), not the whole index. That is
what makes "can it scan 25 M vectors" become "can it scan the largest single
archive".

``rescore(quantizer=bit, oversample=8)`` is the intended default.
``_ensure_dim_table`` tries the rescore DDL first and falls back to a plain
flat index if the rescore table option is unsupported by the loaded build.

This module imports only ``db`` (+ stdlib/numpy/sqlite_vec); the dense source
receives the store via DI, never by importing it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from vesta.vectors.contracts import IndexMeta, StoreStats, VectorHit, VectorRow

if TYPE_CHECKING:
    from collections.abc import Sequence
    from sqlite3 import Row

    import numpy as np
    from numpy.typing import NDArray

    from vesta.db.connection import Database

log = structlog.get_logger("vesta.vectors")


def _table_for(dim: int) -> str:
    """The vec0 table name for a given embedding dimensionality (one table per
    distinct dim, resolved through the interface)."""
    return f"vectors_d{dim}"


class SqliteVecStore:
    """The one :class:`VectorStore` implementation.

    Constructed by the composition root (``main`` lifespan) after the DB is open
    and migrated. If vec0 did not load (``db.vec0_available()`` is False), every
    method is a no-op/empty — ``Capability.VECTORS`` stays off and the dense source
    is capability-dropped, so the app keeps working without a semantic index.
    """

    def __init__(
        self,
        db: Database,
        *,
        quantizer: str = "bit",
        oversample: int = 8,
        default_dim: int = 384,
    ) -> None:
        self._db = db
        self._quantizer = quantizer
        self._oversample = oversample
        #: The default embedder dim (e.g. Granite embedding small = 384).
        #: The lazy ``vectors_d{default_dim}`` is created eagerly at
        #: ``ensure_default_table()`` so a freshly-migrated DB has the table ready.
        self._default_dim = default_dim

    @property
    def vec0_available(self) -> bool:
        return self._db.vec0_available()

    async def ensure_default_table(self) -> None:
        """Create the default-dim vec0 table up front (called by ``main`` at bind
        time). Other dims are created lazily on first upsert."""
        if self.vec0_available:
            await self._ensure_dim_table(self._default_dim)

    def _ddl_candidates(self, dim: int) -> tuple[str, ...]:
        """vec0 DDL variants, most-preferred first.

        The rescore variant is tried first; the plain flat index is the fallback
        for stock wheels without the ``rescore`` table option. ``partition_key``
        on ``zim_id`` is the load-bearing pre-filter and is present in every variant.
        """
        base = (
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {_table_for(dim)} USING vec0("
            f"id INTEGER PRIMARY KEY, embedding FLOAT[{dim}], "
            f"zim_id INTEGER PARTITION KEY"
        )
        rescore = (
            f", rescore quantizer={self._quantizer} oversample={self._oversample}"
            if self._quantizer and self._oversample > 1
            else ""
        )
        return (base + rescore + ")", base + ")")

    async def _ensure_dim_table(self, dim: int) -> bool:
        """Create ``vectors_d{dim}`` if absent. Returns True on success, False if
        vec0 is unavailable or every DDL variant is rejected by this build."""
        if not self.vec0_available:
            return False
        async with self._db.write() as conn:
            for ddl in self._ddl_candidates(dim):
                try:
                    await conn.execute(ddl)
                    return True
                except Exception as exc:
                    log.debug("vectors.ddl_variant_rejected", dim=dim, error=repr(exc))
        log.warning("vectors.dim_table_unavailable", dim=dim)
        return False

    # ── VectorStore ──────────────────────────────────────────────────────────

    async def upsert(self, rows: Sequence[VectorRow]) -> None:
        if not rows:
            return
        if not self.vec0_available:
            log.warning("vectors.upsert_skipped_no_vec0", count=len(rows))
            return
        # Group by dimension so each dim's table is created once and written in a
        # single transaction. All rows from one embedder share a dim; grouping
        # keeps the common case (one dim per upsert call) a single tx.
        by_dim: dict[int, list[VectorRow]] = {}
        for r in rows:
            dim = _dim_of(r.embedding)
            by_dim.setdefault(dim, []).append(r)
        for dim, group in by_dim.items():
            if not await self._ensure_dim_table(dim):
                continue
            table = _table_for(dim)
            async with self._db.write() as conn:
                for r in group:
                    blob = _to_blob(r.embedding)
                    # Chunk row: reconcile on the id PK. The store owns
                    # (zim_id, article_id, char_start, char_end); the indexer
                    # (Stage 2) owns (ordinal, depth) and pre-inserts/sets them,
                    # so ON CONFLICT(id) leaves those columns untouched.
                    await conn.execute(
                        "INSERT INTO chunks(id, zim_id, article_id, char_start, char_end) "
                        "VALUES(?, ?, ?, ?, ?) "
                        "ON CONFLICT(id) DO UPDATE SET "
                        "zim_id=excluded.zim_id, article_id=excluded.article_id, "
                        "char_start=excluded.char_start, char_end=excluded.char_end",
                        (r.id, r.zim_id, r.article_id, r.char_start, r.char_end),
                    )
                    # vec0 has no upsert/REPLACE (its xUpdate rejects a rowid
                    # that already exists with "UNIQUE constraint failed"), so a
                    # re-index of an existing id is DELETE-then-INSERT. Cheap: the
                    # common case (new id) deletes zero rows.
                    await conn.execute(f"DELETE FROM {table} WHERE id = ?", (r.id,))
                    await conn.execute(
                        f"INSERT INTO {table}(id, embedding, zim_id) VALUES(?, ?, ?)",
                        (r.id, blob, r.zim_id),
                    )

    async def search(
        self, q: NDArray[np.float32], *, zim_ids: Sequence[int], k: int
    ) -> list[VectorHit]:
        if not self.vec0_available or k <= 0:
            return []
        dim = _dim_of(q)
        table = _table_for(dim)
        if not await _table_exists(self._db, table):
            return []
        blob = _to_blob(q)
        # Partition-key pre-filter: vec0 applies zim_id IN (...) as a physical
        # shard restriction BEFORE the KNN scan. Empty zim_ids ⇒ scope-everything
        # (search all archives); the caller (dense source) normally passes the
        # scoped set.
        in_clause, params = _in_clause(zim_ids, blob, k)
        sql = (
            f"SELECT v.id, v.distance, v.zim_id, "
            f"c.article_id, c.char_start, c.char_end, a.entry_path "
            f"FROM {table} v JOIN chunks c ON c.id = v.id "
            f"JOIN articles a ON a.id = c.article_id "
            f"WHERE v.embedding MATCH ? AND v.k = ?{in_clause} "
            f"ORDER BY v.distance"
        )
        hits: list[VectorHit] = []
        async with self._db.read() as conn, conn.execute(sql, params) as cur:
            async for row in cur:
                hits.append(
                    VectorHit(
                        zim_id=row["zim_id"],
                        article_id=row["article_id"],
                        chunk_id=row["id"],
                        path=row["entry_path"],
                        # Negate distance so higher == more similar
                        # (Candidate.score convention).
                        score=-float(row["distance"]),
                        char_start=row["char_start"],
                        char_end=row["char_end"],
                    )
                )
        return hits

    async def delete_by_zim(self, zim_id: int) -> None:
        """Remove every vector + chunk row for an archive. vec0 tables are NOT
        FK-cascaded, so each dim table gets an explicit partition-key delete
        (DELETE reclaims space per ~1024-vector chunk)."""
        if not self.vec0_available:
            # Still clear chunk rows (plain table, no extension needed) so an
            # archive removal on a vec0-less build doesn't orphan chunk metadata.
            async with self._db.write() as conn:
                await conn.execute("DELETE FROM chunks WHERE zim_id = ?", (zim_id,))
            return
        for dim in await _existing_dims(self._db):
            async with self._db.write() as conn:
                await conn.execute(f"DELETE FROM {_table_for(dim)} WHERE zim_id = ?", (zim_id,))
        async with self._db.write() as conn:
            await conn.execute("DELETE FROM chunks WHERE zim_id = ?", (zim_id,))
        log.info("vectors.deleted_by_zim", zim_id=zim_id)

    async def delete_by_article(self, zim_id: int, article_id: int) -> None:
        """Remove one article's vectors + chunks.

        A worker killed mid-article leaves partial chunks; before re-inserting we
        delete that article's chunk ids from every ``vectors_d{N}`` (vec0 is not
        FK-cascaded) so no orphaned vectors survive the restart."""
        ids = await _all(
            self._db,
            "SELECT id FROM chunks WHERE zim_id = ? AND article_id = ?",
            (zim_id, article_id),
        )
        if not ids:
            return
        id_vals = [int(r["id"]) for r in ids]
        if self.vec0_available:
            placeholders = ",".join("?" for _ in id_vals)
            for dim in await _existing_dims(self._db):
                async with self._db.write() as conn:
                    await conn.execute(
                        f"DELETE FROM {_table_for(dim)} WHERE id IN ({placeholders})",
                        tuple(id_vals),
                    )
        async with self._db.write() as conn:
            await conn.execute(
                "DELETE FROM chunks WHERE zim_id = ? AND article_id = ?",
                (zim_id, article_id),
            )

    async def stats(self) -> StoreStats:
        per_dim: dict[int, int] = {}
        total = 0
        if self.vec0_available:
            for dim in await _existing_dims(self._db):
                count = await _scalar(self._db, f"SELECT count(*) FROM {_table_for(dim)}")
                per_dim[dim] = count
                total += count
        per_zim_rows = await _all(
            self._db, "SELECT zim_id, count(*) AS n FROM chunks GROUP BY zim_id"
        )
        per_zim = {int(row["zim_id"]): int(row["n"]) for row in per_zim_rows}
        disk_bytes = await _disk_bytes(self._db)
        return StoreStats(
            total_rows=total,
            per_dim=per_dim,
            per_zim=per_zim,
            disk_bytes=disk_bytes,
            vec0_available=self.vec0_available,
        )

    async def record_index_meta(self, zim_id: int, meta: IndexMeta) -> None:
        async with self._db.write() as conn:
            await conn.execute(
                "INSERT INTO index_meta(zim_id, embedder_id, dim, query_prefix, "
                "passage_prefix, pooling, normalize) VALUES(?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(zim_id) DO UPDATE SET "
                "embedder_id=excluded.embedder_id, dim=excluded.dim, "
                "query_prefix=excluded.query_prefix, "
                "passage_prefix=excluded.passage_prefix, pooling=excluded.pooling, "
                "normalize=excluded.normalize",
                (
                    zim_id,
                    meta.embedder_id,
                    meta.dim,
                    meta.query_prefix,
                    meta.passage_prefix,
                    meta.pooling,
                    1 if meta.normalize else 0,
                ),
            )

    async def describe(self, zim_id: int) -> IndexMeta | None:
        async with (
            self._db.read() as conn,
            conn.execute("SELECT * FROM index_meta WHERE zim_id = ?", (zim_id,)) as cur,
        ):
            row = await cur.fetchone()
        if row is None:
            return None
        return IndexMeta(
            embedder_id=row["embedder_id"],
            dim=int(row["dim"]),
            query_prefix=row["query_prefix"],
            passage_prefix=row["passage_prefix"],
            pooling=row["pooling"],
            normalize=bool(row["normalize"]),
        )


# ── Helpers (module-private; no vector state) ────────────────────────────────


def _dim_of(vec: NDArray[np.float32]) -> int:
    """Embedding dimensionality from a 1-D vector or a (1, dim) row matrix."""
    shape = vec.shape
    if len(shape) == 1:
        return int(shape[0])
    # (1, dim) query-vector shape; last axis is the dimensionality.
    return int(shape[-1])


def _to_blob(vec: NDArray[np.float32]) -> bytes:
    """vec0 wants float32 vectors as packed bytes."""
    import numpy as _np

    return _np.ascontiguousarray(vec, dtype=_np.float32).tobytes()


def _in_clause(zim_ids: Sequence[int], blob: bytes, k: int) -> tuple[str, tuple[object, ...]]:
    """Build the optional ``AND zim_id IN (...)`` clause + bound params.

    Match vector and k always come first (``embedding MATCH ? AND k = ?``); the
    partition-key filter is appended when a scope is given. An empty ``zim_ids``
    means "search every archive" — vec0 then scans the whole table, which is the
    caller's explicit choice (the dense source scopes per request by default)."""
    if not zim_ids:
        return "", (blob, k)
    placeholders = ",".join("?" for _ in zim_ids)
    return f" AND v.zim_id IN ({placeholders})", (blob, k, *zim_ids)


async def _table_exists(db: Database, name: str) -> bool:
    async with (
        db.read() as conn,
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ) as cur,
    ):
        return await cur.fetchone() is not None


async def _existing_dims(db: Database) -> list[int]:
    """Dimensions currently materialised as ``vectors_d{N}`` tables."""
    async with (
        db.read() as conn,
        conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vectors_d%'"
        ) as cur,
    ):
        rows = await cur.fetchall()
    dims: list[int] = []
    for row in rows:
        try:
            dims.append(int(row["name"].removeprefix("vectors_d")))
        except ValueError:
            continue
    return sorted(dims)


async def _scalar(db: Database, sql: str) -> int:
    async with db.read() as conn, conn.execute(sql) as cur:
        row = await cur.fetchone()
    return int(row[0]) if row else 0


async def _all(db: Database, sql: str, params: tuple[object, ...] = ()) -> list[Row]:
    async with db.read() as conn, conn.execute(sql, params) as cur:
        return list(await cur.fetchall())


async def _disk_bytes(db: Database) -> int:
    """On-disk footprint estimate: SQLite page count x page size (covers the main
    db; WAL/shm are transient and excluded)."""
    async with db.read() as conn:
        async with conn.execute("PRAGMA page_count") as cur:
            pages_row = await cur.fetchone()
        async with conn.execute("PRAGMA page_size") as cur:
            size_row = await cur.fetchone()
    if pages_row is None or size_row is None:
        return 0
    return int(pages_row[0]) * int(size_row[0])
