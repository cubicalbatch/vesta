"""Vector-store scale harness: measures sqlite-vec and LanceDB backends on an
apples-to-apples synthetic corpus at configurable scale points.

Measures ``SqliteVecStore`` (src/vesta/vectors/sqlite_vec_store.py) and, when
``--backend`` includes it, a minimal in-script LanceDB harness (there is no
``LanceDbStore`` product implementation yet; this script's
Lance wrapper exists ONLY to produce comparable numbers and is not reused by
product code) against **synthetic** vectors at configurable scale points. Pure
store-scaling measurement (p50/p95 search latency, cold-cache latency,
recall@10 vs. brute-force exact, disk, RAM, build time), not a content or
retrieval-quality measurement — synthetic data is sufficient because only
store throughput/latency/recall is under test, not content.

**sqlite-vec DDL**: the live production index is the FLAT f32 variant, not the
bit-rescored one (the rescore DDL is rejected by the installed wheel and
`_ddl_candidates` silently falls back).
This harness forces the same flat DDL explicitly (``quantizer=""``,
``oversample=0``), rather than relying on that fallback, so the comparison is
against the variant this deployment actually runs, not a hypothetical faster
one.

Standalone script, not wired into ``vesta bench``, by deliberate choice: the
harness needs a scratch ``Database`` + migrations + seeded ``zims``/``articles``
FK rows to drive the real sqlite-vec store, which pulls in ``vesta.db`` and
``vesta.vectors`` directly. ``eval/`` may import only ``retrieval`` + ``config``,
so a real module living there would need the same
callable-injection shape ``eval/bench/extraction.py`` uses — considerably more
machinery than this one-off, DoD-checkbox measurement justifies. This script is
the composition root instead (same role ``cli.py`` plays for the real bench
harness).

Design of the synthetic corpus at each scale point N (unchanged from the
original sqlite-vec-only harness, so both backends see the identical corpus):
    - vectors are random unit-normalized float32, dim = the default ``embed``
      encoder's dim (Granite embedding small = 384, encoders/registry.py).
    - spread across 25 fake zim_ids so a scoped/partitioned query is genuinely
      exercised.
    - one "big" zim_id holds 90% of N and the rest is split round-robin across
      the other 24 — this directly targets the switch trigger's literal
      wording ("a single archive exceeding ~5M vectors"), rather than diluting
      the worst case by spreading evenly. Searches are scoped to that single big
      zim_id (k=40, matching ``retrieval.dense.k``'s default), which is exactly
      "can it scan the largest single archive". Because
      every search in this harness is scoped to exactly one archive, the vec0
      "k is applied per PARTITION KEY" semantics
      do not come into play here — there is only ever one partition in scope.
      That per-archive-k question is a *retrieval* semantics question a
      product store implementation must answer; this harness measures raw
      store throughput/latency/recall, not multi-archive fan-out.
    - corpus generation is a pure function of ``n`` (``_gen_corpus_batches``),
      replayed independently by each arm's upsert phase AND by the exact-recall
      reference builder, so every arm at a given ``n`` sees byte-identical
      vectors without persisting them between arms.

Cold-cache measurement: this box grants passwordless ``sudo`` for
``/proc/sys/vm/drop_caches`` (verified interactively before this script was
written); ``_drop_page_cache()`` uses it. If that is unavailable in some other
environment, the cold arm is skipped with a warning rather than silently
reporting warm numbers as cold.

Recall@10 vs. exact: for each scale point, before any store is built, a
brute-force reference is computed once (shared across arms) over the
``BIG_ZIM_ID`` vectors for a small sample of the query set, using plain numpy
matmul (vectors are unit-normalized, so max dot product == min L2 distance ==
exact top-k). Each arm then reports the mean fraction of the exact top-10 that
appears in its own top-10 for that same sample.

Usage:
    .venv/bin/python scripts/bench_vector_scale.py \\
        --scales 1000000 5000000 \\
        --backend both \\
        --lance-index-types flat ivf_pq ivf_hnsw_sq \\
        --out bench_results/vector_scale.md

Writes progress to stdout and a result table to stdout **and** to
``--out`` (markdown) + a sibling ``.json`` (machine-readable) at the end.
Does NOT touch ``data/vesta.db`` — every (scale, arm) gets its own scratch dir
under a temp directory, deleted as soon as that arm's measurement is done.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vesta.db.connection import Database
from vesta.db.migrations import run_migrations
from vesta.vectors.contracts import VectorRow
from vesta.vectors.sqlite_vec_store import SqliteVecStore

if TYPE_CHECKING:
    from collections.abc import Iterator

DIM = 384  # onnx-community/granite-embedding-small-english-r2-ONNX (default `embed`)
NUM_ZIMS = 25
BIG_ZIM_ID = 1  # holds ~90% of the corpus — the "single large archive" case
UPSERT_BATCH = 10_000
SEARCH_K = 40  # retrieval.dense.k default
NUM_SEARCH_QUERIES = 100
NUM_COLD_QUERIES = 5  # each preceded by an explicit page-cache drop — expensive, keep small
RECALL_SAMPLE_QUERIES = 20  # first N of the query sequence, checked against brute-force exact
RECALL_AT = 10
# Lance defaults under test.
LANCE_NPROBES = 20
LANCE_REFINE_FACTOR = 10
# Switch trigger: a single archive > ~5M vectors with p95 > 400ms on an
# unfiltered search (sqlite-vec only — LanceDB is expected to beat this by
# construction; the column is kept for continuity with the original harness).
SWITCH_TRIGGER_P95_MS = 400.0


def _rss_kb() -> int:
    """Peak RSS so far in this process, KB (Linux ``ru_maxrss`` is already KB;
    stdlib-only, matching the house style of ``eval/bench/hardware.py`` — no
    ``psutil`` dependency exists elsewhere in this codebase)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _zim_for_row(i: int) -> int:
    """90% of rows land on ``BIG_ZIM_ID``; the rest round-robin across the other
    24 zim_ids (2..NUM_ZIMS) — see module docstring."""
    if i % 10 != 0:
        return BIG_ZIM_ID
    other = 2 + (i // 10) % (NUM_ZIMS - 1)
    return other


def _big_zim_count(n: int) -> int:
    """Exact count of rows landing on BIG_ZIM_ID for a corpus of size n — a
    closed form of ``_zim_for_row``, used to preallocate the exact-recall
    reference arrays without a scan."""
    return n - n // 10


def _make_batch(start_id: int, count: int, rng: np.random.Generator) -> list[VectorRow]:
    vecs = rng.standard_normal((count, DIM)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    rows = []
    for j in range(count):
        cid = start_id + j
        zim_id = _zim_for_row(cid)
        rows.append(
            VectorRow(
                id=cid,
                zim_id=zim_id,
                article_id=cid,  # 1:1 chunk<->article — FK satisfied, ratio doesn't matter here
                embedding=vecs[j],
                char_start=0,
                char_end=400,
            )
        )
    return rows


def _gen_corpus_batches(n: int) -> Iterator[list[VectorRow]]:
    """Pure, deterministic (seed 42) replay of the synthetic corpus in
    ``UPSERT_BATCH``-sized chunks. Every arm's upsert phase AND the exact-recall
    reference builder call this independently for the same ``n`` and get
    byte-identical vectors — this is what makes cross-arm/cross-backend
    comparison at a given scale point valid without persisting the corpus."""
    rng = np.random.default_rng(42)
    inserted = 0
    next_id = 1
    while inserted < n:
        batch_n = min(UPSERT_BATCH, n - inserted)
        yield _make_batch(next_id, batch_n, rng)
        inserted += batch_n
        next_id += batch_n


def _query_vec(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(DIM).astype(np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _gen_query_sequence(count: int) -> list[np.ndarray]:
    """Deterministic (seed 7) replay of the query sequence: index 0 is the
    warmup query (discarded by the search phase but consumed here too, to stay
    lockstep), 1..count are the measured queries. Every arm's search phase and
    the exact-recall reference builder call this independently and get the
    same vectors in the same order."""
    rng = np.random.default_rng(7)
    return [_query_vec(rng) for _ in range(count + 1)]


# ── Exact (brute-force) recall reference ─────────────────────────────────────


@dataclass(frozen=True)
class ExactReference:
    n: int
    query_vecs: list[np.ndarray]  # length RECALL_SAMPLE_QUERIES, measured-queries order
    exact_top_ids: list[list[int]]  # per query, chunk ids of the exact top-RECALL_AT


def _build_exact_reference(n: int) -> ExactReference:
    """Brute-force top-``RECALL_AT`` over BIG_ZIM_ID's vectors for the first
    ``RECALL_SAMPLE_QUERIES`` measured queries, via a single numpy matmul.
    Vectors are unit-normalized, so argmax(dot) == exact nearest-neighbour by
    L2 — no need to materialize distances.

    Freed immediately after use (the big_n x DIM float32 matrix is
    ~1.5 GiB/million rows) so it never coexists with a store's own memory."""
    big_n = _big_zim_count(n)
    print(f"  building exact-recall reference over {big_n:,} BIG_ZIM_ID vectors...", flush=True)
    t0 = time.perf_counter()
    vecs = np.empty((big_n, DIM), dtype=np.float32)
    ids = np.empty(big_n, dtype=np.int64)
    pos = 0
    for batch in _gen_corpus_batches(n):
        for row in batch:
            if row.zim_id != BIG_ZIM_ID:
                continue
            vecs[pos] = row.embedding
            ids[pos] = row.id
            pos += 1
    assert pos == big_n, f"expected {big_n} BIG_ZIM_ID rows, collected {pos}"

    queries = _gen_query_sequence(RECALL_SAMPLE_QUERIES)[1:]  # drop the warmup query
    q_mat = np.stack(queries)  # (RECALL_SAMPLE_QUERIES, DIM)
    dots = q_mat @ vecs.T  # (RECALL_SAMPLE_QUERIES, big_n) — higher = more similar
    exact_top_ids: list[list[int]] = []
    for row in dots:
        # argpartition for the top-RECALL_AT candidates, then sort just those.
        top_idx = np.argpartition(-row, RECALL_AT - 1)[:RECALL_AT]
        top_idx = top_idx[np.argsort(-row[top_idx])]
        exact_top_ids.append([int(ids[i]) for i in top_idx])
    del vecs, ids, dots
    gc.collect()
    print(f"  exact reference built in {time.perf_counter() - t0:.1f}s", flush=True)
    return ExactReference(n=n, query_vecs=queries, exact_top_ids=exact_top_ids)


def _recall_at_k(returned_ids: list[int], exact_ids: list[int], k: int) -> float:
    exact_k = exact_ids[:k]
    if not exact_k:
        return 1.0
    returned_k = set(returned_ids[:k])
    return len(returned_k & set(exact_k)) / len(exact_k)


# ── Cold-cache control ────────────────────────────────────────────────────────

_COLD_CACHE_AVAILABLE: bool | None = None


def _drop_page_cache() -> bool:
    """Best-effort OS page-cache drop via passwordless sudo. Retries once (a
    transient sudo/process hiccup should not sacrifice an entire arm's cold
    measurement). Returns False (and only warns once) if truly unavailable, so
    callers can skip the cold arm instead of silently mislabeling warm numbers
    as cold."""
    global _COLD_CACHE_AVAILABLE
    last_exc: Exception | None = None
    for _attempt in range(2):
        try:
            subprocess.run(
                ["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
                check=True,
                capture_output=True,
                timeout=10,
            )
            _COLD_CACHE_AVAILABLE = True
            return True
        except Exception as exc:
            last_exc = exc
            time.sleep(0.5)
    if _COLD_CACHE_AVAILABLE is None:
        print(f"  [cold] page-cache drop unavailable ({last_exc!r}); cold arm will be skipped")
    _COLD_CACHE_AVAILABLE = False
    return False


# ── sqlite-vec arm ────────────────────────────────────────────────────────────


async def _seed_zims(db: Database) -> None:
    async with db.write() as conn:
        await conn.executemany(
            "INSERT INTO zims(id) VALUES (?)", [(z,) for z in range(1, NUM_ZIMS + 1)]
        )


async def _seed_articles_batch(db: Database, rows: list[VectorRow]) -> None:
    """Bulk-insert the article FK rows a batch of chunks needs, via
    ``executemany`` (bypassing the store — this is scaffolding, not the
    measured operation)."""
    async with db.write() as conn:
        await conn.executemany(
            "INSERT INTO articles(id, zim_id, entry_path) VALUES (?, ?, ?)",
            [(r.id, r.zim_id, f"A/{r.id}") for r in rows],
        )


def _disk_usage_sqlite(db_path: Path) -> int:
    total = 0
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(db_path) + suffix)
        if p.exists():
            total += p.stat().st_size
    return total


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


@dataclass(frozen=True)
class ScaleResult:
    arm: str  # "sqlite_vec" | "lancedb:flat" | "lancedb:ivf_pq" | "lancedb:ivf_hnsw_sq"
    n: int
    disk_bytes: int
    upsert_seconds: float
    index_build_seconds: float  # 0.0 for arms with no separate ANN build step (sqlite-vec, flat)
    upsert_rss_kb: int
    search_p50_ms: float
    search_p95_ms: float
    search_rss_kb: int
    cold_p50_ms: float | None  # None if cold measurement was skipped (no drop_caches access)
    cold_p95_ms: float | None
    recall_at_10: float | None  # None if no exact reference was available
    big_zim_count: int


async def _run_sqlite_vec_arm(n: int, scratch_dir: Path, exact_ref: ExactReference) -> ScaleResult:
    db_path = scratch_dir / f"scale_{n}.db"
    print(f"\n--- arm=sqlite_vec n={n:,} -> {db_path} ---", flush=True)
    db = Database(str(db_path), busy_timeout_ms=30_000, read_pool_size=4)
    await db.start()
    if not db.vec0_available():
        raise RuntimeError("vec0 extension did not load — cannot run the scale harness")
    async with db.write() as conn:
        await run_migrations(conn)
    # Force the FLAT DDL (no rescore clause) — matches the live production
    # index, not the bit-rescored variant this
    # deployment has never actually run. See module docstring.
    store = SqliteVecStore(db, quantizer="", oversample=0, default_dim=DIM)
    await store.ensure_default_table()
    await _seed_zims(db)

    t0 = time.perf_counter()
    inserted = 0
    big_zim_count = 0
    for batch in _gen_corpus_batches(n):
        await _seed_articles_batch(db, batch)
        await store.upsert(batch)
        big_zim_count += sum(1 for r in batch if r.zim_id == BIG_ZIM_ID)
        inserted += len(batch)
        if inserted % (UPSERT_BATCH * 5) == 0 or inserted == n:
            elapsed = time.perf_counter() - t0
            rate = inserted / max(elapsed, 1e-9)
            print(
                f"  upserted {inserted:,}/{n:,} ({100 * inserted / n:.1f}%) "
                f"in {elapsed:.1f}s -- {rate:.0f} rows/s",
                flush=True,
            )
    upsert_elapsed = time.perf_counter() - t0
    upsert_rss = _rss_kb()

    async with db.write() as conn:
        await conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    stats = await store.stats()
    assert stats.total_rows == n, f"expected {n} rows, store reports {stats.total_rows}"
    disk_bytes = _disk_usage_sqlite(db_path)

    async def search_fn(q: np.ndarray) -> list[int]:
        hits = await store.search(q, zim_ids=[BIG_ZIM_ID], k=SEARCH_K)
        return [h.chunk_id for h in hits]

    p50, p95, search_rss, recall = await _search_and_recall_phase(search_fn, exact_ref)
    cold_p50, cold_p95 = await _cold_phase(search_fn)

    await db.stop()
    print(
        f"  done: disk={disk_bytes / (1024 * 1024):.1f} MiB upsert={upsert_elapsed:.1f}s "
        f"p50={p50:.2f}ms p95={p95:.2f}ms recall@10={recall}",
        flush=True,
    )
    return ScaleResult(
        arm="sqlite_vec",
        n=n,
        disk_bytes=disk_bytes,
        upsert_seconds=upsert_elapsed,
        index_build_seconds=0.0,
        upsert_rss_kb=upsert_rss,
        search_p50_ms=p50,
        search_p95_ms=p95,
        search_rss_kb=search_rss,
        cold_p50_ms=cold_p50,
        cold_p95_ms=cold_p95,
        recall_at_10=recall,
        big_zim_count=big_zim_count,
    )


# ── LanceDB arm ───────────────────────────────────────────────────────────────


async def _lance_upsert_corpus(table: Any, n: int) -> tuple[float, int, int]:
    """Replay the deterministic corpus into the Lance table (UPSERT_BATCH-sized
    adds); returns (upsert_seconds, upsert_rss_kb, big_zim_count)."""
    t0 = time.perf_counter()
    inserted = 0
    big_zim_count = 0
    buf: list[dict[str, object]] = []
    for batch in _gen_corpus_batches(n):
        for r in batch:
            buf.append(
                {
                    "chunk_id": r.id,
                    "zim_id": r.zim_id,
                    "vector": r.embedding.tolist(),
                }
            )
            if r.zim_id == BIG_ZIM_ID:
                big_zim_count += 1
        if len(buf) >= UPSERT_BATCH:
            await table.add(buf)
            buf = []
        inserted += len(batch)
        if inserted % (UPSERT_BATCH * 5) == 0 or inserted == n:
            elapsed = time.perf_counter() - t0
            rate = inserted / max(elapsed, 1e-9)
            print(
                f"  upserted {inserted:,}/{n:,} ({100 * inserted / n:.1f}%) "
                f"in {elapsed:.1f}s -- {rate:.0f} rows/s",
                flush=True,
            )
    if buf:
        await table.add(buf)
    return time.perf_counter() - t0, _rss_kb(), big_zim_count


async def _lance_build_vector_index(table: Any, lindex: Any, index_type: str) -> float:
    """Build the configured vector index; returns build seconds. "flat" builds
    none (0.0) — Lance's exact KNN scan applies when no vector index exists."""
    if index_type == "ivf_pq":
        t0 = time.perf_counter()
        await table.create_index("vector", config=lindex.IvfPq(distance_type="l2"))
        return time.perf_counter() - t0
    if index_type == "ivf_hnsw_sq":
        t0 = time.perf_counter()
        await table.create_index("vector", config=lindex.IvfHnswSq(distance_type="l2"))
        return time.perf_counter() - t0
    if index_type != "flat":
        raise ValueError(f"unknown lance index_type: {index_type!r}")
    return 0.0


async def _run_lancedb_arm(
    n: int, index_type: str, scratch_dir: Path, exact_ref: ExactReference
) -> ScaleResult:
    """``index_type`` in {"flat", "ivf_pq", "ivf_hnsw_sq"}. "flat" builds no ANN
    index at all — Lance's documented behaviour is an exact KNN scan when no
    vector index exists (the "flat fallback" property that makes a
    searchable-during-build index possible), so it is the
    correct backend-neutral comparison point against sqlite-vec's flat scan,
    not a synonym for `IvfFlat`."""
    import lancedb
    from lancedb import index as lindex

    lance_dir = scratch_dir / f"lance_{n}_{index_type}"
    print(f"\n--- arm=lancedb:{index_type} n={n:,} -> {lance_dir} ---", flush=True)
    import pyarrow as pa

    db = await lancedb.connect_async(str(lance_dir))
    schema = pa.schema(
        [
            pa.field("chunk_id", pa.int64()),
            pa.field("zim_id", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), DIM)),
        ]
    )
    table = await db.create_table(f"vectors_d{DIM}", schema=schema, mode="overwrite")

    upsert_elapsed, upsert_rss, big_zim_count = await _lance_upsert_corpus(table, n)

    # Scalar index on zim_id — the pre-filter column — mirrors the vec0
    # PARTITION KEY's role: makes the `where("zim_id = ...")` prefilter cheap.
    await table.create_index("zim_id", config=lindex.Bitmap())

    index_build_elapsed = await _lance_build_vector_index(table, lindex, index_type)

    row_count = await table.count_rows()
    assert row_count == n, f"expected {n} rows, table reports {row_count}"
    disk_bytes = _dir_size_bytes(lance_dir)

    async def search_fn(q: np.ndarray) -> list[int]:
        # AsyncTable.search() itself is a coroutine — await it to get the
        # AsyncVectorQuery builder, THEN chain .where()/.nprobes()/etc.
        query = await table.search(q)
        query = query.where(f"zim_id = {BIG_ZIM_ID}").limit(SEARCH_K)
        if index_type != "flat":
            query = query.nprobes(LANCE_NPROBES).refine_factor(LANCE_REFINE_FACTOR)
        rows = await query.to_list()
        return [int(r["chunk_id"]) for r in rows]

    p50, p95, search_rss, recall = await _search_and_recall_phase(search_fn, exact_ref)
    cold_p50, cold_p95 = await _cold_phase(search_fn)

    db.close()
    print(
        f"  done: disk={disk_bytes / (1024 * 1024):.1f} MiB upsert={upsert_elapsed:.1f}s "
        f"index_build={index_build_elapsed:.1f}s p50={p50:.2f}ms p95={p95:.2f}ms recall@10={recall}",
        flush=True,
    )
    return ScaleResult(
        arm=f"lancedb:{index_type}",
        n=n,
        disk_bytes=disk_bytes,
        upsert_seconds=upsert_elapsed,
        index_build_seconds=index_build_elapsed,
        upsert_rss_kb=upsert_rss,
        search_p50_ms=p50,
        search_p95_ms=p95,
        search_rss_kb=search_rss,
        cold_p50_ms=cold_p50,
        cold_p95_ms=cold_p95,
        recall_at_10=recall,
        big_zim_count=big_zim_count,
    )


# ── Shared search/recall/cold phases ─────────────────────────────────────────


async def _search_and_recall_phase(
    search_fn, exact_ref: ExactReference
) -> tuple[float, float, int, float | None]:
    """Runs the warm-cache measured-query loop (NUM_SEARCH_QUERIES), and — for
    the first RECALL_SAMPLE_QUERIES of them — checks the returned ids against
    the brute-force exact reference. Returns (p50_ms, p95_ms, peak_rss_kb,
    mean_recall_at_10 | None)."""
    queries = _gen_query_sequence(NUM_SEARCH_QUERIES)
    await search_fn(queries[0])  # warmup

    latencies_ms: list[float] = []
    recalls: list[float] = []
    for i, q in enumerate(queries[1:], start=0):
        t0 = time.perf_counter()
        ids = await search_fn(q)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        assert len(ids) <= SEARCH_K
        if i < RECALL_SAMPLE_QUERIES and exact_ref.n:
            recalls.append(_recall_at_k(ids, exact_ref.exact_top_ids[i], RECALL_AT))
    latencies_ms.sort()
    p50 = statistics.median(latencies_ms)
    p95 = latencies_ms[min(int(len(latencies_ms) * 0.95), len(latencies_ms) - 1)]
    mean_recall = statistics.mean(recalls) if recalls else None
    return p50, p95, _rss_kb(), mean_recall


async def _cold_phase(search_fn) -> tuple[float | None, float | None]:
    """NUM_COLD_QUERIES queries, each preceded by an explicit OS page-cache
    drop. Returns (None, None) if the cache drop is unavailable — never
    silently reports warm numbers as cold."""
    queries = _gen_query_sequence(NUM_COLD_QUERIES)[1:]
    latencies_ms: list[float] = []
    for q in queries:
        if not _drop_page_cache():
            return None, None
        t0 = time.perf_counter()
        await search_fn(q)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    latencies_ms.sort()
    p50 = statistics.median(latencies_ms)
    p95 = latencies_ms[min(int(len(latencies_ms) * 0.95), len(latencies_ms) - 1)]
    return p50, p95


# ── Orchestration ─────────────────────────────────────────────────────────────


def _resolve_arms(backend: str, lance_index_types: list[str]) -> list[tuple[str, str | None]]:
    """Returns [(kind, lance_index_type)] pairs; kind in {"sqlite_vec", "lancedb"}."""
    arms: list[tuple[str, str | None]] = []
    if backend in ("sqlite_vec", "both"):
        arms.append(("sqlite_vec", None))
    if backend in ("lancedb", "both"):
        arms.extend(("lancedb", t) for t in lance_index_types)
    return arms


async def main(scales: list[int], backend: str, lance_index_types: list[str]) -> list[ScaleResult]:
    results: list[ScaleResult] = []
    arms = _resolve_arms(backend, lance_index_types)
    with tempfile.TemporaryDirectory(prefix="vesta-vecscale-") as tmpdir:
        tmp_root = Path(tmpdir)
        for n in scales:
            print(f"\n\n=== scale point n={n:,} ===", flush=True)
            exact_ref = _build_exact_reference(n)
            for kind, lance_index_type in arms:
                scratch = tmp_root / f"n{n}"
                scratch.mkdir(parents=True, exist_ok=True)
                if kind == "sqlite_vec":
                    result = await _run_sqlite_vec_arm(n, scratch, exact_ref)
                else:
                    assert lance_index_type is not None
                    result = await _run_lancedb_arm(n, lance_index_type, scratch, exact_ref)
                results.append(result)
                gc.collect()
                # Delete this arm's scratch data immediately — 5M/7M-point
                # data can be multiple GB; no need to keep more than one
                # arm's data on disk at once.
                shutil.rmtree(scratch, ignore_errors=True)
    return results


def _fmt(v: float | None, spec: str) -> str:
    return format(v, spec) if v is not None else "n/a"


def _print_report(results: list[ScaleResult]) -> str:
    lines: list[str] = []
    lines.append("\n\n=== Scale harness results ===\n")
    header = (
        f"| {'arm':<18} | {'n':>10} | {'disk (MiB)':>10} | {'upsert (s)':>10} | "
        f"{'idx build (s)':>13} | {'rows/s':>8} | {'RSS (MB)':>9} | "
        f"{'warm p50':>9} | {'warm p95':>9} | {'cold p50':>9} | {'cold p95':>9} | "
        f"{'recall@10':>9} |"
    )
    sep = "|" + "|".join("-" * (len(c) + 2) for c in header.strip("|").split("|")) + "|"
    lines.append(header)
    lines.append(sep)
    for r in results:
        lines.append(
            f"| {r.arm:<18} | {r.n:>10,} | {r.disk_bytes / (1024 * 1024):>10.1f} | "
            f"{r.upsert_seconds:>10.1f} | {r.index_build_seconds:>13.1f} | "
            f"{r.n / max(r.upsert_seconds, 1e-9):>8.0f} | {r.upsert_rss_kb / 1024:>9.1f} | "
            f"{r.search_p50_ms:>9.2f} | {r.search_p95_ms:>9.2f} | "
            f"{_fmt(r.cold_p50_ms, '9.2f')} | {_fmt(r.cold_p95_ms, '9.2f')} | "
            f"{_fmt(r.recall_at_10, '9.3f') if r.recall_at_10 is not None else 'n/a':>9} |"
        )
    text = "\n".join(lines)
    print(text)
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[1_000_000, 5_000_000],
        help="Scale points to measure (row counts).",
    )
    parser.add_argument(
        "--backend",
        choices=["sqlite_vec", "lancedb", "both"],
        default="both",
    )
    parser.add_argument(
        "--lance-index-types",
        nargs="+",
        default=["flat", "ivf_pq", "ivf_hnsw_sq"],
        choices=["flat", "ivf_pq", "ivf_hnsw_sq"],
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write the markdown table + a sibling .json to this path.",
    )
    ns = parser.parse_args()
    out = asyncio.run(main(ns.scales, ns.backend, ns.lance_index_types))
    table_text = _print_report(out)
    if ns.out:
        out_path = Path(ns.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table_text + "\n", encoding="utf-8")
        json_path = out_path.with_suffix(".json")
        json_path.write_text(json.dumps([asdict(r) for r in out], indent=2), encoding="utf-8")
        print(f"\nwrote {out_path} and {json_path}")
