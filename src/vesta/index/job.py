"""The ``index_zim`` job — resumable, multi-process, preemptible semantic indexing.

One job indexes one archive at one depth. The shape follows a few hard
requirements:

* **Multi-process extraction** (threads scale *negatively* for extraction).
  The job builds its own spawn pool (``index.worker_processes``) so it controls
  the worker PIDs — those are the SIGSTOP targets for CPU preemption, and
  each worker ``nice``s/``ionice``s itself down at startup.
* **Resumable**. Articles are processed in the archive's stable entry-id order
  (``text_entry_paths``); the checkpoint is a high-water count of fully-
  committed articles. On resume the job skips the committed prefix and, for the
  first article it touches, deletes any partial chunks+vectors first (a crashed
  worker must not silently skip a range — and must not *leak* one).
* **Skips redirects, soft redirects, disambiguation and list pages** — but they
  remain readable (the ZIM is never touched).
* **Embedding is injectable** so the job is unit-testable without ONNX: the
  composition root binds an embedder provider returning the live ``Encoder``;
  tests bind a deterministic fake.
* **Throughput measured continuously** and fed to the calibrated estimator; the
  job's progress/ETA reflects preemption suspension honestly.

``index/`` depends on zim/encoders/vectors/config/db and must not import
api/answer/retrieval (enforced by tests/test_boundaries.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from vesta import config
from vesta.index import (
    INDEX_BATCH_SIZE,
    INDEX_DEFAULT_DEPTH,
    INDEX_NICE,
    INDEX_WORKER_PROCESSES,
    INDEX_WORKER_RECYCLE_EVERY,
    get_coordinator,
    get_db,
    get_embedder_provider,
    get_registry,
    set_indexed_state,
)
from vesta.index.depth import chunks_for_article
from vesta.index.estimate import ThroughputTracker
from vesta.index.preempt import apply_ionice_idle, apply_nice
from vesta.jobs.types import RESUME_CHECKPOINT_KEY, JobHandle, register_job_type
from vesta.vectors import get_store
from vesta.vectors.contracts import IndexMeta, VectorRow
from vesta.zim.entries import classify_entry
from vesta.zim.extract import extract_entry
from vesta.zim.types import EntryFlags, EntryPath, ExtractedArticle

if TYPE_CHECKING:
    from libzim.reader import Archive as LibzimArchive

    from vesta.db.connection import Database
    from vesta.vectors.contracts import VectorStore
    from vesta.zim.registry import ArchiveRegistry

_log = logging.getLogger(__name__)

#: Flags that mean "do not index this article". They remain
#: readable — skipping is an indexing decision, never a content decision.
_SKIP_FLAGS = (
    EntryFlags.REDIRECT | EntryFlags.SOFT_REDIRECT | EntryFlags.DISAMBIGUATION | EntryFlags.LIST
)

#: Index format version. Bumped when the set/order of indexed entries changes
#: (e.g. harvesting vtt/plain/markdown sidecars alongside html) so a stale
#: positional checkpoint from an older build is ignored and the index rebuilds
#: fresh rather than resuming against a mismatched path list.
_INDEX_FMT_VERSION = 3

#: Progress-report cadence (articles). The runner throttles writes to ~1/s
#: anyway, but we avoid building a huge SSE event per article.
_PROGRESS_EVERY = 25


class EmbedderProvider(Protocol):
    """``() -> awaitable Encoder | None``. The composition root binds one that
    returns the live embed encoder (``EncoderManager.get_embed``); tests bind a
    fake. ``None`` ⇒ no embedder available ⇒ the job cannot run."""

    def __call__(self) -> Awaitable[Any]: ...


# ── spawn-pool worker (module-level: must be picklable) ─────────────────────


_POOL_ARCHIVE: LibzimArchive | None = None


def _worker_init(archive_path: str, nice: int) -> None:
    """Open the archive once per worker, then demote scheduling/I/O priority
    (``nice 15`` / ``ionice -c3``) so interactive work wins the CPU."""
    from libzim.reader import Archive as LibzimArchive

    global _POOL_ARCHIVE
    _POOL_ARCHIVE = LibzimArchive(archive_path)
    apply_nice(nice)
    apply_ionice_idle()


def _worker_extract_classify(path: EntryPath) -> ExtractedArticle | None:
    """Read, classify and extract one path. Returns ``None`` for skip-flagged
    entries (redirect/soft-redirect/disambig/list) so the indexer never spends an
    embed call on them."""
    assert _POOL_ARCHIVE is not None
    from vesta.zim.reader import read_entry_sync

    raw = read_entry_sync(_POOL_ARCHIVE, path)
    flags = classify_entry(path, raw.title, raw.content, is_redirect=raw.is_redirect)
    if flags & _SKIP_FLAGS:
        return None
    # Mimetype-aware: HTML bodies via resiliparse; vtt/plain/markdown sidecars
    # via plain-text extraction (media/SPA ZIMs whose html entries are stubs).
    article = extract_entry(raw)
    return ExtractedArticle(
        path=article.path,
        title=article.title,
        text=article.text,
        sections=article.sections,
        flags=flags | article.flags,
    )


def _worker_extract_batch(paths: list[EntryPath]) -> list[ExtractedArticle | None]:
    """Extract + classify a batch in one worker round-trip (fewer pickles)."""
    return [_worker_extract_classify(p) for p in paths]


# ── the job ─────────────────────────────────────────────────────────────────


class IndexZimJob:
    """Registered as job type ``index_zim``. Params: ``zim_id``, ``depth``."""

    name = "index_zim"

    async def run(self, job: JobHandle, params: Mapping[str, Any]) -> None:
        zim_id = int(params["zim_id"])
        depth = int(params.get("depth", int(config.get(INDEX_DEFAULT_DEPTH))))
        if depth < 1 or depth > 3:
            raise ValueError(f"index_zim: depth must be 1..3, got {depth}")

        db = get_db()
        registry = get_registry()
        store = get_store()
        if db is None or registry is None or store is None:
            raise RuntimeError(
                "index_zim: db/registry/store not bound (run inside the app lifespan)"
            )
        try:
            await _run_build(
                job, params, zim_id=zim_id, depth=depth, db=db, registry=registry, store=store
            )
        except Exception:
            # A failed build is surfaced on the archive row, not just the job
            # row (08: index_status includes 'error'; the checkpoint keeps the
            # resume position, so retrying the job resumes where it stopped).
            await _set_index_status(db, zim_id, "error", depth=depth)
            raise


async def _run_build(  # noqa: PLR0912, PLR0915 — the resume/fresh/cancel/preempt ladder
    job: JobHandle,
    params: Mapping[str, Any],
    *,
    zim_id: int,
    depth: int,
    db: Database,
    registry: ArchiveRegistry,
    store: VectorStore,
) -> None:
    archive = await _get_archive(registry, zim_id)
    archive_path = await _archive_path(db, zim_id)
    paths = await archive.text_entry_paths()
    total = len(paths)
    if total == 0:
        await _set_index_status(db, zim_id, "complete", depth=depth)
        set_indexed_state(True)
        await job.progress(0, 0, "no text entries")
        return

    # Resume: high-water count of fully-committed entries (strict order).
    resume = params.get(RESUME_CHECKPOINT_KEY)
    start_at = 0
    checkpoint_fmt: object = None
    if isinstance(resume, Mapping):
        start_at = max(0, int(resume.get("done_count", 0)))
        checkpoint_fmt = resume.get("fmt")
    recorded_depth = await _recorded_depth(db, zim_id)
    # A format bump (the set/order of indexed mimetypes changed) invalidates any
    # stale positional checkpoint — otherwise resuming against a differently-
    # ordered path list would corrupt the index. Mirrors the depth-change rule.
    fresh = (start_at == 0) or (recorded_depth != depth) or (checkpoint_fmt != _INDEX_FMT_VERSION)
    if fresh:
        # Clean slate: a new build or a depth change. Lowering/raising depth
        # deletes the old vectors and never touches the ZIM.
        await store.delete_by_zim(zim_id)
        await _reset_articles(db, zim_id)
        start_at = 0

    await _set_index_status(db, zim_id, "running", depth=depth, total=total)
    set_indexed_state(True)
    await job.progress(start_at, total, f"indexing depth {depth}")

    embedder_provider = get_embedder_provider()
    if embedder_provider is None:
        raise RuntimeError("index_zim: no embedder provider bound")
    encoder = await embedder_provider()
    if encoder is None:
        raise RuntimeError("index_zim: no embed model available — run `vesta models` then retry")

    batch_size = max(1, int(config.get(INDEX_BATCH_SIZE)))
    workers = max(1, int(config.get(INDEX_WORKER_PROCESSES)))
    nice = int(config.get(INDEX_NICE))
    recycle_every = int(config.get(INDEX_WORKER_RECYCLE_EVERY))
    coord = get_coordinator()
    tracker = ThroughputTracker(total_articles=total, depth=depth)
    t0 = time.monotonic()
    done = start_at

    pool = _ExtractionPool(archive_path, workers, nice, recycle_every=recycle_every)
    await pool.start()
    if coord is not None:
        coord.set_worker_pids(pool.pids())

    def _start_extract(at: int) -> tuple[int, asyncio.Task[list[ExtractedArticle | None]]] | None:
        """Kick off extraction for the batch at ``at``; ``None`` past the end.

        Returns the batch's END cursor alongside the task. Nothing about `done`
        moves here — the checkpoint only ever advances past articles that have
        been fully committed by ``_materialize_batch`` below.
        """
        if at >= total:
            return None
        batch = paths[at : at + batch_size]
        return at + len(batch), asyncio.create_task(pool.extract(batch))

    cursor = start_at
    # One batch of extraction runs (in the pool's worker processes) while the
    # previous batch embeds (in an ONNX thread), so the two halves of the
    # pipeline stop taking turns. Only ever ONE extraction is in flight — the
    # next is started after the previous is awaited — so the pool is never
    # entered concurrently and its recycle logic stays single-threaded.
    pending = _start_extract(cursor)
    try:
        while pending is not None:
            batch_end, task = pending
            if job.cancelled():
                # Resumable pause, not an error: the checkpoint holds the
                # high-water mark and a later resume continues from it.
                # Drain the in-flight extraction first — `pool.stop()` in the
                # finally block would otherwise close the pool mid-`map`.
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
                pending = None
                await _set_index_status(db, zim_id, "paused", depth=depth, total=total)
                await _update_index_progress(db, zim_id, done)
                return
            await _yield(coord)
            extracted = await task
            # A recycle (if configured) may have respawned the workers with new
            # PIDs; re-register them every batch so the coordinator's SIGSTOP
            # targets stay current. Idempotent and cheap (a set assignment).
            if coord is not None:
                coord.set_worker_pids(pool.pids())
            cursor = batch_end
            # Start the NEXT extraction before embedding this batch — this is
            # the overlap. Both `pool.extract` and `encoder.embed` hand off to
            # threads, so they genuinely run at the same time.
            pending = _start_extract(cursor)
            await _materialize_batch(
                db=db,
                store=store,
                encoder=encoder,
                extracted=extracted,
                zim_id=zim_id,
                depth=depth,
                fresh_build=fresh,
            )
            done = batch_end
            tracker.record(done, time.monotonic() - t0)
            if done % _PROGRESS_EVERY < batch_size or done >= total:
                est = tracker.estimate()
                rate = (
                    est.rate_articles_per_s / max(coord.effective_rate_fraction(), 0.05)
                    if coord
                    else est.rate_articles_per_s
                )
                msg = f"depth {depth}: {done}/{total} articles ({rate:.1f}/s effective)"
                await job.progress(done, total, msg)
                _log_rss(pool, done, total)
                await job.checkpoint(
                    {"done_count": done, "depth": depth, "fmt": _INDEX_FMT_VERSION}
                )
                await _update_index_progress(db, zim_id, done)
    finally:
        if pending is not None:  # an exception escaped mid-pipeline
            pending[1].cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pending[1]
        if coord is not None:
            coord.set_worker_pids(set())
        await pool.stop()

    # Stamp the embedder fingerprint so a mismatched future query is refused,
    # and record it on the zims row for the API/UI surface.
    meta = _meta_from_encoder(encoder)
    await store.record_index_meta(zim_id, meta)
    await _record_embedder(db, zim_id, meta)
    await _set_index_status(db, zim_id, "complete", depth=depth, total=total)
    set_indexed_state(True)
    await job.progress(total, total, "done")


# ── helpers ─────────────────────────────────────────────────────────────────


async def _yield(coord: Any) -> None:
    if coord is not None:
        await coord.yield_if_busy()
    else:
        await asyncio.sleep(0)


def _read_rss_kib(status_path: str) -> int | None:
    """Read ``VmRSS:`` (KiB) from a Linux ``/proc/<pid>/status`` file.

    Returns ``None`` on any error or when the ``VmRSS:`` line is absent (non-
    Linux, or the process exited between listing its PID and reading it). Best-
    effort: never raises — RSS logging must not break indexing.
    """
    try:
        with open(status_path, encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
            return None
    except (OSError, ValueError, IndexError):
        return None


def _log_rss(pool: Any, done: int, total: int) -> None:
    """Log main-process + worker RSS (MB) on the progress tick so the memory
    curve is visible during a long index run.

    Reads ``VmRSS`` from ``/proc/self/status`` (main) and each ``pool.pids()``
    worker's ``/proc/<pid>/status``. Best-effort: any failure (non-Linux, a
    worker that exited) is swallowed — this never raises into the indexer.
    """
    main_kib = _read_rss_kib("/proc/self/status")
    if main_kib is None:
        return  # non-Linux or unreadable; nothing useful to report
    worker_kibs: list[int] = []
    try:
        pids = pool.pids()
    except Exception:
        pids = set()
    for pid in pids:
        kib = _read_rss_kib(f"/proc/{pid}/status")
        if kib is not None:
            worker_kibs.append(kib)
    worker_mbs = [k // 1024 for k in worker_kibs]
    _log.info(
        "index.rss done=%d/%d main=%dMB workers=%s",
        done,
        total,
        main_kib // 1024,
        worker_mbs,
    )


async def _get_archive(registry: ArchiveRegistry, zim_id: int) -> Any:
    return registry.get(zim_id)


async def _archive_path(db: Database, zim_id: int) -> str:
    async with (
        db.read() as conn,
        conn.execute("SELECT path FROM zims WHERE id=?", (zim_id,)) as cur,
    ):
        row = await cur.fetchone()
    if row is None:
        raise RuntimeError(f"index_zim: zims row {zim_id} not found")
    return str(row["path"])


async def _recorded_depth(db: Database, zim_id: int) -> int:
    async with (
        db.read() as conn,
        conn.execute("SELECT index_depth FROM zims WHERE id=?", (zim_id,)) as cur,
    ):
        row = await cur.fetchone()
    return int(row["index_depth"] or 0) if row is not None else 0


async def _set_index_status(
    db: Database, zim_id: int, status: str, *, depth: int | None = None, total: int | None = None
) -> None:
    import datetime as _dt

    now = _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()
    # zims has event timestamps (indexed_at), no generic updated_at (0001_init).
    sets = ["index_status=?", "index_progress=?"]
    vals: list[Any] = [status, 0]
    if depth is not None:
        sets.append("index_depth=?")
        vals.append(depth)
    if total is not None:
        sets.append("index_total=?")
        vals.append(total)
    if status == "complete":
        sets.append("indexed_at=?")
        vals.append(now)
    vals.append(zim_id)
    async with db.write() as conn:
        await conn.execute(f"UPDATE zims SET {', '.join(sets)} WHERE id=?", vals)


async def _reset_articles(db: Database, zim_id: int) -> None:
    """Drop articles rows for a fresh build (chunks+vectors cleared by the store)."""
    async with db.write() as conn:
        await conn.execute("DELETE FROM articles WHERE zim_id=?", (zim_id,))


async def _update_index_progress(db: Database, zim_id: int, done: int) -> None:
    """Mirror live progress onto the zims row so ``GET /api/zims/{id}/index``
    shows real movement between status transitions."""
    async with db.write() as conn:
        await conn.execute("UPDATE zims SET index_progress=? WHERE id=?", (done, zim_id))


async def _record_embedder(db: Database, zim_id: int, meta: IndexMeta) -> None:
    """Record the embedder an archive was indexed with on its zims row."""
    async with db.write() as conn:
        await conn.execute(
            "UPDATE zims SET embedding_model=?, embedding_dim=? WHERE id=?",
            (meta.embedder_id, meta.dim, zim_id),
        )


async def _materialize_batch(
    *,
    db: Database,
    store: VectorStore,
    encoder: Any,
    extracted: Sequence[ExtractedArticle | None],
    zim_id: int,
    depth: int,
    fresh_build: bool,
) -> int:
    """Insert article + chunk rows, embed, and upsert vectors for one batch.

    Returns the count of articles actually indexed (skip-flagged entries are
    dropped). Unless ``fresh_build``, every article touched is
    ``delete_by_article``'d first so a partial write from a crashed worker is
    cleaned before re-insertion (no orphan vectors). Idempotent on re-index.

    ``fresh_build`` is the caller's build-level "we already cleared this
    archive" flag: ``_run_build`` calls ``store.delete_by_zim`` before the first
    batch of a fresh build, so no chunk row for this ``zim_id`` can exist and the
    per-article cleanup is a guaranteed no-op — one indexed SELECT per article
    per batch, each a round-trip through aiosqlite's worker thread, for nothing.
    A RESUMED build skips that clearing step and does need the cleanup: its first
    touched batch may hold a partially-written article from the crashed run.

    Transaction shape (load-bearing): the store opens its OWN ``db.write()``
    transactions, and the Database's write lock is not reentrant — so the
    phases below never nest a store call inside this function's ``db.write()``
    block: (1) articles upsert, (2) per-article cleanup, (3) chunk rows,
    (4) embed + vector upsert.
    """
    kept = [a for a in extracted if a is not None and a.text.strip()]
    if not kept:
        return 0

    # Documents manifest (0013): for documents-kind archives (nautiluszim), a
    # PDF entry's libzim title is the stub-shell/filename ("Water (1).pdf"), not
    # the human title. Override it from the manifest (article_documents.title)
    # so a chunk/card reads "Distillation For Home Water Treatment". One batched
    # round-trip; the SELECT returns no rows for non-documents archives, so this
    # is a no-op for article/media/SPA ZIMs.
    titles = await _document_titles(db, zim_id, [a.path for a in kept])

    # (1) Upsert the article rows and collect their ids.
    article_ids: list[int] = []
    async with db.write() as conn:
        for article in kept:
            assert article is not None
            article_ids.append(
                await _upsert_article(
                    conn=conn,
                    zim_id=zim_id,
                    path=article.path,
                    title=titles.get(article.path, article.title),
                    text=article.text,
                    flags=article.flags,
                    sections=article.sections,
                )
            )

    # (2) Clean any partial prior write for these articles before re-inserting
    # (outside any write tx of ours — the store manages its own). Skipped on a
    # fresh build, where delete_by_zim already guaranteed there is nothing here.
    if not fresh_build:
        for article_id in article_ids:
            await store.delete_by_article(zim_id, article_id)

    # (3) Insert the chunk rows (indexer-owned ordinal/depth) and stage the
    # texts + rows for the batch embed.
    pending_rows: list[VectorRow] = []
    pending_texts: list[str] = []
    async with db.write() as conn:
        for article, article_id in zip(kept, article_ids, strict=True):
            assert article is not None
            for ordinal, spec in enumerate(chunks_for_article(article, depth)):
                cur = await conn.execute(
                    "INSERT INTO chunks(zim_id, article_id, ordinal, char_start, char_end, depth) "
                    "VALUES(?,?,?,?,?,?)",
                    (zim_id, article_id, ordinal, spec.char_start, spec.char_end, depth),
                )
                chunk_id = int(cur.lastrowid) if cur.lastrowid is not None else 0
                pending_texts.append(spec.text)
                pending_rows.append(
                    VectorRow(
                        id=chunk_id,
                        zim_id=zim_id,
                        article_id=article_id,
                        # placeholder embedding; filled after the batch embed
                        embedding=np.zeros(1, dtype=np.float32),
                        char_start=spec.char_start,
                        char_end=spec.char_end,
                    )
                )

    # (4) Embed the batch and upsert the vectors.
    if pending_rows:
        vecs = await encoder.embed(pending_texts, kind="passage")
        rows = [
            VectorRow(
                id=r.id,
                zim_id=r.zim_id,
                article_id=r.article_id,
                embedding=vecs[i],
                char_start=r.char_start,
                char_end=r.char_end,
            )
            for i, r in enumerate(pending_rows)
        ]
        await store.upsert(rows)
    return len(kept)


async def _upsert_article(
    *,
    conn: Any,
    zim_id: int,
    path: EntryPath,
    title: str,
    text: str,
    flags: EntryFlags,
    sections: Sequence[Any],
) -> int:
    """Insert/refresh an articles row, returning its id (08 is the first populator)."""
    await conn.execute(
        "INSERT INTO articles(zim_id, entry_path, title, char_len, n_sections, flags) "
        "VALUES(?,?,?,?,?,?) "
        "ON CONFLICT(zim_id, entry_path) DO UPDATE SET "
        "title=excluded.title, char_len=excluded.char_len, n_sections=excluded.n_sections, "
        "flags=excluded.flags",
        (zim_id, path, title, len(text), len(sections), int(flags)),
    )
    cur = await conn.execute(
        "SELECT id FROM articles WHERE zim_id=? AND entry_path=?", (zim_id, path)
    )
    row = await cur.fetchone()
    return int(row["id"])


async def _document_titles(db: Database, zim_id: int, paths: list[EntryPath]) -> dict[str, str]:
    """Batched manifest-title lookup for documents-kind archives.

    Returns ``{doc_path: title}`` only for paths that have an ``article_documents``
    row with a non-null title. Empty for non-documents archives (no rows), so the
    override is a no-op everywhere except nautiluszim ZIMs. One round-trip,
    de-duped; mirrors ``fetch_media_for_paths``'s IN-list shape.
    """
    if not paths:
        return {}
    unique = list(dict.fromkeys(paths))
    placeholders = ",".join("?" for _ in unique)
    query = (
        "SELECT doc_path, title FROM article_documents "
        f"WHERE zim_id=? AND doc_path IN ({placeholders})"
    )
    async with db.read() as conn:
        cur = await conn.execute(query, (zim_id, *unique))
        rows = await cur.fetchall()
    return {str(r["doc_path"]): str(r["title"]) for r in rows if r["title"] is not None}


def _meta_from_encoder(encoder: Any) -> IndexMeta:
    """Stamp the embedder fingerprint an index was built with. Read from
    the live encoder's spec-derived attributes."""
    return IndexMeta(
        embedder_id=str(encoder.id),
        dim=int(encoder.dim),
        query_prefix=str(encoder.query_prefix),
        passage_prefix=str(encoder.passage_prefix),
        pooling=str(encoder.pooling),
        normalize=bool(encoder.normalize),
    )


# ── extraction pool (PID-controlled spawn pool) ────────────────────────────


class _ExtractionPool:
    """A spawn pool whose worker PIDs the coordinator SIGSTOPs.

    Thin wrapper over ``multiprocessing`` so the indexer controls the process
    group (extraction's libzim reads are the CPU-bound, SIGSTOP-able work). The
    initializer opens the archive + demotes priority once per worker.

    ``recycle_every`` bounds worker-process RSS on very long runs (full
    Wikipedia): libzim's per-read allocations accumulate in each worker, so
    every N extracted batches the pool is closed and respawned with fresh
    workers. The job re-registers the (possibly changed) PIDs with the
    coordinator after each ``extract`` — the pool itself never touches the
    coordinator (clean separation).
    """

    def __init__(
        self, archive_path: str, processes: int, nice: int, *, recycle_every: int = 0
    ) -> None:
        self._archive_path = archive_path
        self._processes = max(1, processes)
        self._nice = nice
        self._recycle_every = max(0, int(recycle_every))
        self._extract_count = 0
        self._pool: Any = None

    async def start(self) -> None:
        # Build synchronously off the event loop; spawn is slow but one-time.
        await asyncio.to_thread(self._start_sync)

    def _start_sync(self) -> None:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self._pool = ctx.Pool(
            processes=self._processes,
            initializer=_worker_init,
            initargs=(self._archive_path, self._nice),
        )

    def pids(self) -> set[int]:
        if self._pool is None:
            return set()
        # ``Pool._pool`` is the list of worker Process objects (stable CPython API).
        return {int(p.pid) for p in getattr(self._pool, "_pool", []) if p.pid is not None}

    async def _recycle(self) -> None:
        """Close and respawn the worker pool to reset per-worker RSS.

        Runs the close + respawn off the event loop (both block on process
        join/spawn). Does NOT touch the coordinator — the caller re-registers
        PIDs after this returns (the pool has no coordinator dependency).
        """
        old = self._pool
        self._pool = None
        await asyncio.to_thread(_close_pool, old)
        await asyncio.to_thread(self._start_sync)
        self._extract_count = 0

    async def extract(self, paths: Sequence[EntryPath]) -> list[ExtractedArticle | None]:
        if not paths or self._pool is None:
            return []
        # Recycle BEFORE doing work once the batch threshold is reached, so the
        # accumulated worker RSS is released before this batch's allocations.
        if (
            self._recycle_every > 0
            and self._extract_count >= self._recycle_every
            and self._pool is not None
        ):
            await self._recycle()
        # ``Pool.map`` distributes its ITERABLE across workers and returns one
        # result per item, so the batch is split into contiguous per-worker
        # sub-batches (one worker would otherwise serialize the whole batch —
        # and contiguous chunks keep the flattened result in work order, which
        # the batch-committed checkpoint relies on) and flattened one level.
        items = list(paths)
        n = min(self._processes, len(items))
        size = -(-len(items) // n)  # ceil
        sub_batches = [items[i * size : (i + 1) * size] for i in range(n)]
        results = await asyncio.to_thread(self._pool.map, _worker_extract_batch, sub_batches)
        self._extract_count += 1
        return [article for sub in results for article in sub]

    async def stop(self) -> None:
        pool = self._pool
        self._pool = None
        if pool is not None:
            await asyncio.to_thread(_close_pool, pool)


def _close_pool(pool: Any) -> None:
    try:
        pool.close()
        pool.join()
    except Exception:  # pragma: no cover - best-effort shutdown
        pool.terminate()


# Register the built-in index job type at import.
register_job_type(IndexZimJob())


__all__ = ["IndexZimJob"]
