"""``index_zim`` job tests (resume + failure semantics).

The job runs against a REAL ``Database`` + migration 0004 + REAL
``SqliteVecStore`` (vec0), with fakes only at the true external edges: the
archive (``text_entry_paths``), the spawn pool (``_ExtractionPool`` —
monkeypatched so no processes spawn in tests), the encoder, and the job handle.
That exercises these load-bearing properties:

* resume = high-water count in the archive's stable entry order; a depth
  change is a fresh build (old vectors deleted, ZIM never touched);
* skip-flagged entries (redirect/soft-redirect/disambig/list) never reach the
  encoder but remain readable;
* a crashed worker must not silently skip a range OR leak one — every touched
  article is ``delete_by_article``-cleaned before re-insertion;
* a failed build surfaces ``index_status='error'`` on the archive row, and a
  cancel surfaces ``paused``.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
import pytest_asyncio

import vesta.index
from vesta import config
from vesta.db.connection import Database
from vesta.db.migrations import run_migrations
from vesta.index import bind_coordinator, bind_runtime, set_indexed_state
from vesta.index.job import _INDEX_FMT_VERSION, IndexZimJob
from vesta.jobs.runner import JobRunner
from vesta.jobs.types import RESUME_CHECKPOINT_KEY
from vesta.vectors import bind_store
from vesta.vectors.contracts import IndexMeta
from vesta.vectors.sqlite_vec_store import SqliteVecStore
from vesta.zim.types import EntryFlags, EntryPath, ExtractedArticle

# ── fakes at the external edges ───────────────────────────────────────────────


class _FakeJobHandle:
    def __init__(self, *, cancelled: bool = False) -> None:
        self.progress_calls: list[tuple[int, int, str]] = []
        self.checkpoints: list[dict[str, Any]] = []
        self._cancelled = cancelled

    async def progress(self, done: int, total: int, message: str) -> None:
        self.progress_calls.append((done, total, message))

    async def checkpoint(self, blob: Mapping[str, Any]) -> None:
        self.checkpoints.append(dict(blob))

    def cancelled(self) -> bool:
        return self._cancelled


class _FakeArchive:
    def __init__(self, paths: list[EntryPath]) -> None:
        self._paths = paths

    async def text_entry_paths(self) -> list[EntryPath]:
        return list(self._paths)


class _FakeRegistry:
    def __init__(self, archive: _FakeArchive) -> None:
        self._archive = archive

    def get(self, zim_id: int) -> _FakeArchive:
        return self._archive


class _FakeEncoder:
    """Deterministic dim-4 "embedder" — a text hash, no ONNX."""

    id = "test/embedder"
    dim = 4
    query_prefix = ""
    passage_prefix = ""
    pooling = "mean"
    normalize = True

    def __init__(self) -> None:
        self.embed_calls: list[list[str]] = []

    async def embed(self, texts: list[str], *, kind: str) -> list[np.ndarray]:
        self.embed_calls.append(list(texts))
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            out.append(np.array([h[0] / 255, h[1] / 255, h[2] / 255, h[3] / 255], dtype=np.float32))
        return out


class _FakePool:
    """Stands in for ``_ExtractionPool`` — no spawn, records requests."""

    articles: ClassVar[dict[EntryPath, ExtractedArticle | None]] = {}
    instances: ClassVar[list[_FakePool]] = []

    def __init__(
        self, archive_path: str, processes: int, nice: int, recycle_every: int = 0
    ) -> None:
        self.requested: list[list[EntryPath]] = []
        self.stopped = False
        _FakePool.instances.append(self)

    async def start(self) -> None:
        pass

    def pids(self) -> set[int]:
        return set()

    async def extract(self, paths: Any) -> list[ExtractedArticle | None]:
        self.requested.append(list(paths))
        return [_FakePool.articles.get(p) for p in paths]

    async def stop(self) -> None:
        self.stopped = True


class _SpyStore:
    """Delegates to the real store, counting the crash-safety/fresh-build calls."""

    def __init__(self, inner: SqliteVecStore) -> None:
        self._inner = inner
        self.deleted_zims: list[int] = []
        self.deleted_articles: list[tuple[int, int]] = []

    async def delete_by_zim(self, zim_id: int) -> None:
        self.deleted_zims.append(zim_id)
        await self._inner.delete_by_zim(zim_id)

    async def delete_by_article(self, zim_id: int, article_id: int) -> None:
        self.deleted_articles.append((zim_id, article_id))
        await self._inner.delete_by_article(zim_id, article_id)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# ── fixture: real db + store, fake archive/pool/encoder, bound singletons ─────


def _article(path: EntryPath, i: int) -> ExtractedArticle:
    return ExtractedArticle(
        path=path,
        title=f"Article {i}",
        text=f"Lead text of article {i}. " * 8,
        sections=(),
        flags=EntryFlags.NONE,
    )


class _Rig:
    def __init__(
        self,
        db: Database,
        store: _SpyStore,
        encoder: _FakeEncoder,
        handle: _FakeJobHandle,
        paths: list[EntryPath],
    ) -> None:
        self.db = db
        self.store = store
        self.encoder = encoder
        self.handle = handle
        self.paths = paths

    async def zims_row(self) -> dict[str, Any]:
        async with self.db.read() as conn, conn.execute("SELECT * FROM zims WHERE id=1") as cur:
            row = await cur.fetchone()
        assert row is not None
        return dict(row)

    async def chunk_rows(self) -> list[dict[str, Any]]:
        async with (
            self.db.read() as conn,
            conn.execute("SELECT * FROM chunks WHERE zim_id=1 ORDER BY id") as cur,
        ):
            return [dict(r) for r in await cur.fetchall()]


@pytest_asyncio.fixture
async def rig(tmp_db_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    db = Database(str(tmp_db_path), busy_timeout_ms=2000)
    await db.start()
    async with db.write() as conn:
        await run_migrations(conn)
        await conn.execute("INSERT INTO zims(id, path) VALUES (1, '/fake.zim')")
    inner = SqliteVecStore(db, default_dim=4)
    await inner.ensure_default_table()
    store = _SpyStore(inner)
    encoder = _FakeEncoder()

    paths = [EntryPath(f"A/Article_{i}") for i in range(5)]
    _FakePool.articles = {p: _article(p, i) for i, p in enumerate(paths)}
    _FakePool.instances = []
    monkeypatch.setattr("vesta.index.job._ExtractionPool", _FakePool)

    config.configure(
        env={"index.batch_size": "2", "index.worker_processes": "1", "index.nice": "0"}
    )
    archive = _FakeArchive(paths)

    async def _provider() -> Any:
        return encoder

    bind_runtime(db, _FakeRegistry(archive), _provider)
    bind_store(store)  # type: ignore[arg-type]
    bind_coordinator(None)
    set_indexed_state(False)
    handle = _FakeJobHandle()
    yield _Rig(db, store, encoder, handle, paths)
    bind_runtime(None, None, None)
    bind_store(None)
    set_indexed_state(False)
    config.configure(env={})
    await db.stop()


async def _zims_status(rig: _Rig) -> str:
    return str((await rig.zims_row())["index_status"] or "none")


# ── the tests ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fresh_depth1_build_indexes_everything(rig: _Rig) -> None:
    await IndexZimJob().run(rig.handle, {"zim_id": 1, "depth": 1})

    # Every article got one depth-1 chunk + one vector, searchable afterwards.
    chunks = await rig.chunk_rows()
    assert len(chunks) == 5
    assert all(c["depth"] == 1 for c in chunks)
    stats = await rig.store.stats()
    assert stats.per_zim.get(1) == 5
    q = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    hits = await rig.store.search(q, zim_ids=[1], k=3)
    assert len(hits) == 3
    assert all(h.path.startswith("A/Article_") for h in hits)

    # The archive row carries the full honest state.
    row = await rig.zims_row()
    assert row["index_status"] == "complete"
    assert row["index_depth"] == 1
    assert row["index_total"] == 5
    assert row["embedding_model"] == "test/embedder"
    assert row["embedding_dim"] == 4
    assert row["indexed_at"] is not None

    # The compat record matches the encoder (index-settings enforcement).
    meta = await rig.store.describe(1)
    assert meta == IndexMeta("test/embedder", 4, "", "", "mean", True)

    # Checkpoints carry the high-water mark; the last one is the full count.
    assert rig.handle.checkpoints[-1] == {"done_count": 5, "depth": 1, "fmt": _INDEX_FMT_VERSION}
    assert rig.handle.progress_calls[-1][0:2] == (5, 5)
    # Every article was embedded exactly once.
    assert sum(len(c) for c in rig.encoder.embed_calls) == 5


@pytest.mark.asyncio
async def test_skip_flagged_entries_never_reach_the_encoder(rig: _Rig) -> None:
    # Redirects/soft-redirects/disambiguation/list pages are skipped by the
    # worker (returned as None) — but remain readable; skipping is an indexing
    # decision, never a content decision.
    _FakePool.articles[rig.paths[1]] = None
    _FakePool.articles[rig.paths[3]] = None
    await IndexZimJob().run(rig.handle, {"zim_id": 1, "depth": 1})
    assert (await rig.store.stats()).per_zim.get(1) == 3
    embedded = [t for call in rig.encoder.embed_calls for t in call]
    assert not any("article 1" in t for t in embedded)
    assert not any("article 3" in t for t in embedded)


@pytest.mark.asyncio
async def test_resumed_build_cleans_every_touched_article_before_reinsert(rig: _Rig) -> None:
    # The crash trap: a worker killed mid-article leaves partial chunks, so a
    # RESUMED run deletes each article's prior chunks+vectors BEFORE
    # re-inserting and no orphaned vectors survive (08 Traps). The resume path
    # is where this can actually happen — a resume deliberately does NOT wipe
    # the archive first (test_resume_skips_committed_prefix), so partial rows
    # from the dead run are still on disk.
    await IndexZimJob().run(rig.handle, {"zim_id": 1, "depth": 1})
    rig.store.deleted_articles.clear()

    handle2 = _FakeJobHandle()
    await IndexZimJob().run(
        handle2,
        {
            "zim_id": 1,
            "depth": 1,
            RESUME_CHECKPOINT_KEY: {"done_count": 2, "depth": 1, "fmt": _INDEX_FMT_VERSION},
        },
    )
    async with rig.db.read() as conn, conn.execute("SELECT id FROM articles WHERE zim_id=1") as cur:
        article_ids = {int(r["id"]) for r in await cur.fetchall()}
    cleaned = {a for (_z, a) in rig.store.deleted_articles}
    # Every article the resumed run re-processed was cleaned first. The skipped
    # committed prefix is never touched, so it is never cleaned.
    assert cleaned, "a resumed build must clean before re-inserting"
    assert cleaned <= article_ids


@pytest.mark.asyncio
async def test_fresh_build_skips_the_per_article_cleanup(rig: _Rig) -> None:
    # A fresh build calls store.delete_by_zim up front, so no chunk row for the
    # archive can exist and the per-article cleanup is a guaranteed no-op — one
    # indexed SELECT per article per batch, each a round-trip through
    # aiosqlite's worker thread, for nothing. Skipping it must not change what
    # ends up in the index.
    await IndexZimJob().run(rig.handle, {"zim_id": 1, "depth": 1})

    assert rig.store.deleted_zims == [1], "a fresh build wipes the archive up front"
    assert rig.store.deleted_articles == [], "…so per-article cleanup is redundant"
    assert (await rig.store.stats()).per_zim.get(1) == 5
    assert await _zims_status(rig) == "complete"


@pytest.mark.asyncio
async def test_resume_skips_committed_prefix(rig: _Rig) -> None:
    # First full run at depth 1.
    await IndexZimJob().run(rig.handle, {"zim_id": 1, "depth": 1})
    pool1 = _FakePool.instances[-1]
    assert sum(len(b) for b in pool1.requested) == 5

    # Resume from the high-water mark (as if the process died after article 2):
    # the committed prefix is skipped in whole-batch steps, no wipe happens.
    rig.store.deleted_zims.clear()
    handle2 = _FakeJobHandle()
    await IndexZimJob().run(
        handle2,
        {
            "zim_id": 1,
            "depth": 1,
            RESUME_CHECKPOINT_KEY: {"done_count": 2, "depth": 1, "fmt": _INDEX_FMT_VERSION},
        },
    )
    assert rig.store.deleted_zims == [], "a same-depth resume never wipes the index"
    pool2 = _FakePool.instances[-1]
    requested = [p for batch in pool2.requested for p in batch]
    assert rig.paths[0] not in requested and rig.paths[1] not in requested
    assert set(requested) == set(rig.paths[2:])
    assert await _zims_status(rig) == "complete"


@pytest.mark.asyncio
async def test_depth_change_is_a_fresh_build(rig: _Rig) -> None:
    await IndexZimJob().run(rig.handle, {"zim_id": 1, "depth": 1})
    assert (await rig.store.stats()).per_zim.get(1) == 5

    # Re-index at depth 2 with a stale checkpoint: the old vectors are deleted
    # (the ZIM is never touched) and the build restarts from article 0.
    rig.store.deleted_zims.clear()
    handle2 = _FakeJobHandle()
    await IndexZimJob().run(
        handle2,
        {"zim_id": 1, "depth": 2, RESUME_CHECKPOINT_KEY: {"done_count": 5, "depth": 1}},
    )
    assert rig.store.deleted_zims == [1], "lowering/raising depth deletes the old vectors"
    chunks = await rig.chunk_rows()
    assert chunks and all(c["depth"] == 2 for c in chunks)
    assert (await rig.store.stats()).per_zim.get(1) == 5
    pool2 = _FakePool.instances[-1]
    assert sum(len(b) for b in pool2.requested) == 5, "a fresh build reprocesses everything"
    assert (await rig.zims_row())["index_depth"] == 2


@pytest.mark.asyncio
async def test_rebuild_zeroes_the_checkpoint_before_any_batch_work(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    # AUDIT_0822 M8: the rebuild branch wipes vectors/articles, but its next
    # checkpoint only lands after the first batch materializes. Dying in that
    # window used to leave a stale done_count=N behind, and a later plain
    # resume would replay at N into the wiped store — articles 0..N silently
    # missing from a "complete" index. So the wipe point immediately writes a
    # {done_count: 0} checkpoint.
    async with rig.db.write() as conn:
        await conn.execute("UPDATE zims SET index_depth=2 WHERE id=1")

    import vesta.index.job as _job_mod

    real_materialize = _job_mod._materialize_batch
    die_calls: list[int] = []

    async def _die_before_first_batch(**kwargs: Any) -> None:
        # Die exactly once — between the wipe and the first committed batch —
        # then let the real materializer through so the follow-up resume runs.
        if not die_calls:
            die_calls.append(1)
            raise RuntimeError("died between the wipe and the first batch")
        await real_materialize(**kwargs)

    monkeypatch.setattr(_job_mod, "_materialize_batch", _die_before_first_batch)
    with pytest.raises(RuntimeError, match="died between"):
        await IndexZimJob().run(
            rig.handle,
            {
                "zim_id": 1,
                "depth": 1,
                RESUME_CHECKPOINT_KEY: {"done_count": 5, "depth": 2, "fmt": _INDEX_FMT_VERSION},
            },
        )

    # The wipe happened, and the ONLY checkpoint this crashed run wrote says 0
    # — nothing can mistake it for resumable progress at N.
    assert rig.store.deleted_zims == [1]
    assert rig.handle.checkpoints == [{"done_count": 0, "depth": 1, "fmt": _INDEX_FMT_VERSION}]

    # A later plain resume picks up that zeroed cursor and starts from 0, not N.
    handle2 = _FakeJobHandle()
    await IndexZimJob().run(
        handle2,
        {"zim_id": 1, "depth": 1, RESUME_CHECKPOINT_KEY: rig.handle.checkpoints[-1]},
    )
    pool2 = _FakePool.instances[-1]
    requested = {p for batch in pool2.requested for p in batch}
    assert requested == set(rig.paths), "the zeroed cursor resumes from scratch"
    assert handle2.checkpoints[-1] == {"done_count": 5, "depth": 1, "fmt": _INDEX_FMT_VERSION}


@pytest.mark.asyncio
async def test_cancel_marks_paused_and_keeps_checkpoint(rig: _Rig) -> None:
    rig.handle._cancelled = True
    await IndexZimJob().run(rig.handle, {"zim_id": 1, "depth": 1})
    assert await _zims_status(rig) == "paused"
    assert (await rig.store.stats()).total_rows == 0
    # The pool was torn down, not left running.
    assert _FakePool.instances[-1].stopped


async def _submit_parked_build(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> tuple[Database, JobRunner, int]:
    """Start a real index_zim build through the JobRunner and return once it is
    parked mid-batch with the archive row stamped 'running'."""
    gate = asyncio.Event()
    real_embed = rig.encoder.embed

    async def gated_embed(texts: list[str], *, kind: str) -> list[np.ndarray]:
        out = await real_embed(texts, kind=kind)
        await gate.wait()  # park mid-build like a real batch in flight
        return out

    monkeypatch.setattr(rig.encoder, "embed", gated_embed)
    r = JobRunner(rig.db)
    await r.start()
    jid = await r.submit("index_zim", None, {"zim_id": 1, "depth": 1})
    while not rig.encoder.embed_calls:
        await asyncio.sleep(0)
    assert await _zims_status(rig) == "running"
    return rig.db, r, jid


async def _poll_status(r: JobRunner, jid: int, status: str) -> bool:
    rec = None
    for _ in range(100):
        await asyncio.sleep(0.005)
        rec = await r.get(jid)
        if rec is not None and rec.status == status:
            return True
    return False


@pytest.mark.asyncio
async def test_server_pause_mid_build_stamps_zims_paused_not_running(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AUDIT_0824 M13: the runner's pause button sets the flag and then cancels
    the task outright, so CancelledError lands at whatever await the build is
    parked on — the loop-top ``job.cancelled()`` check never runs. The archive
    row must still leave 'running': reseed counts 'running' as indexed and
    would silently serve a partial build's vectors."""
    _, r, jid = await _submit_parked_build(rig, monkeypatch)
    try:
        assert await r.pause(jid)
        assert await _poll_status(r, jid, "paused")
        assert await _zims_status(rig) == "paused"
        # And the capability flag dropped in the same unwind (S3): no restart
        # needed before dense profiles stop serving the partial build.
        assert vesta.index._ANY_INDEXED is False
    finally:
        await r.stop()


@pytest.mark.asyncio
async def test_server_cancel_mid_build_stamps_zims_paused_not_running(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same unwind path as a server pause (both converge on task cancellation);
    the job row goes terminal 'cancelled', the archive row still must not stay
    'running'."""
    _, r, jid = await _submit_parked_build(rig, monkeypatch)
    try:
        assert await r.cancel(jid)
        assert await _poll_status(r, jid, "cancelled")
        assert await _zims_status(rig) == "paused"
        assert vesta.index._ANY_INDEXED is False
    finally:
        await r.stop()


@pytest.mark.asyncio
async def test_missing_embedder_fails_with_error_status(rig: _Rig) -> None:
    async def _none_provider() -> Any:
        return None

    bind_runtime(rig.db, _FakeRegistry(_FakeArchive(rig.paths)), _none_provider)
    with pytest.raises(RuntimeError, match="no embed model"):
        await IndexZimJob().run(rig.handle, {"zim_id": 1, "depth": 1})
    assert await _zims_status(rig) == "error"


@pytest.mark.asyncio
async def test_failed_build_reseeds_vectors_flag(rig: _Rig) -> None:
    """AUDIT_0824 S3: the 'running' stamp raises the capability flag mid-build;
    when the build then fails, the error arm stamps 'error' — which the seed
    query does not count as indexed — and must lower the flag in the same
    unwind instead of claiming VECTORS for a partial store until restart."""

    async def _boom_provider() -> Any:
        raise RuntimeError("embedder exploded")

    bind_runtime(rig.db, _FakeRegistry(_FakeArchive(rig.paths)), _boom_provider)
    with pytest.raises(RuntimeError, match="embedder exploded"):
        await IndexZimJob().run(rig.handle, {"zim_id": 1, "depth": 1})
    assert await _zims_status(rig) == "error"
    assert vesta.index._ANY_INDEXED is False


@pytest.mark.asyncio
async def test_empty_archive_completes_immediately(rig: _Rig) -> None:
    bind_runtime(
        rig.db,
        _FakeRegistry(_FakeArchive([])),
        (lambda: None),  # type: ignore[arg-type]
    )
    handle = _FakeJobHandle()
    await IndexZimJob().run(handle, {"zim_id": 1, "depth": 1})
    assert await _zims_status(rig) == "complete"
    assert handle.progress_calls[-1][2] == "no text entries"


@pytest.mark.asyncio
async def test_depth_zero_or_four_rejected(rig: _Rig) -> None:
    with pytest.raises(ValueError, match=r"depth must be 1\.\.3"):
        await IndexZimJob().run(rig.handle, {"zim_id": 1, "depth": 0})
    with pytest.raises(ValueError, match=r"depth must be 1\.\.3"):
        await IndexZimJob().run(rig.handle, {"zim_id": 1, "depth": 4})


# ── _ExtractionPool.extract: Pool.map semantics (live-run regression) ────────


@pytest.mark.asyncio
async def test_extraction_pool_flattens_map_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """``Pool.map(fn, [sub1, sub2])`` returns ``[fn(sub1), fn(sub2)]`` — a list
    of per-sub-batch LISTS. The pool must flatten one level and distribute the
    batch across workers; the first live run returned the nested list and the
    job died on ``'list' object has no attribute 'text'``."""
    from vesta.index.job import _ExtractionPool

    calls: list[list[EntryPath]] = []

    class _MapSemantics:
        """Mimics multiprocessing.Pool.map: fn applied per item, list returned."""

        def map(self, fn: Any, items: list[list[EntryPath]]) -> list[Any]:
            calls.extend(items)
            return [fn(batch) for batch in items]

    pool = _ExtractionPool("/fake.zim", processes=3, nice=0)
    pool._pool = _MapSemantics()
    paths = [EntryPath(f"A/{i}") for i in range(7)]

    # Worker fn needs the module-global archive; stub it to echo paths as articles.
    monkeypatch.setattr(
        "vesta.index.job._worker_extract_batch",
        lambda batch: [_article(p, 0) for p in batch],
    )
    out = await pool.extract(paths)
    assert len(out) == 7
    assert all(isinstance(a, ExtractedArticle) for a in out)
    # Distributed across workers: 3 sub-batches, not one serialized batch.
    assert len(calls) == 3
    assert [p for batch in calls for p in batch] == paths
    # Order is preserved by map (results align with sub-batch order).
    assert [a.path for a in out if a is not None] == paths


@pytest.mark.asyncio
async def test_extraction_pool_recycles_after_n_extracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``recycle_every`` bounds worker-process RSS by closing + respawning the
    spawn pool every N extracts. The recycle must run BEFORE the
    threshold-crossing batch's work and reset the counter, so
    ``recycle_every=2`` recycles exactly once across three extracts."""
    from vesta.index.job import _ExtractionPool

    closes: list[int] = []
    starts: list[int] = []

    class _MapSemantics:
        """Mimics multiprocessing.Pool.map: fn applied per item, list returned."""

        def map(self, fn: Any, items: list[list[EntryPath]]) -> list[Any]:
            return [fn(batch) for batch in items]

    pool = _ExtractionPool("/fake.zim", processes=2, nice=0, recycle_every=2)
    stub = _MapSemantics()
    pool._pool = stub

    def fake_close(p: Any) -> None:
        closes.append(1)

    def fake_start() -> None:
        starts.append(1)
        pool._pool = stub  # restore a usable pool so extract can proceed

    monkeypatch.setattr("vesta.index.job._close_pool", fake_close)
    pool._start_sync = fake_start  # type: ignore[method-assign,assignment]
    monkeypatch.setattr(
        "vesta.index.job._worker_extract_batch",
        lambda batch: [_article(p, 0) for p in batch],
    )
    paths = [EntryPath("A/0"), EntryPath("A/1")]

    await pool.extract(paths)  # count 0 -> no recycle, then count=1
    assert starts == []
    assert closes == []
    await pool.extract(paths)  # count 1 -> no recycle, then count=2
    assert starts == []
    assert closes == []
    await pool.extract(paths)  # count 2 >= 2 -> recycle once, then count=1
    assert len(starts) == 1
    assert len(closes) == 1

    assert isinstance(pool.pids(), set)


@pytest.mark.asyncio
async def test_document_entry_paths_reads_manifest_doc_paths(rig: _Rig) -> None:
    # The kind-aware path source for documents-kind archives: returns ONLY the
    # manifest doc_paths (the 7 PDFs), excluding the viewer junk. This is what
    # makes the 19 sourcemap/template/stub rows stop being indexed.
    from vesta.zim.registry import _document_entry_paths

    doc_paths = ["files/Water (1).pdf", "files/Water (3).pdf", "files/Water (2).pdf"]
    async with rig.db.write() as conn:
        await conn.executemany(
            "INSERT INTO article_documents(zim_id, doc_path, title, doc_mime) VALUES(?,?,?,?)",
            [(1, p, f"title {p}", "application/pdf") for p in doc_paths],
        )
    paths = await _document_entry_paths(rig.db, 1)
    # Exactly the manifest doc_paths, in stable (sorted) order, nothing else.
    assert paths == ["files/Water (1).pdf", "files/Water (2).pdf", "files/Water (3).pdf"]


@pytest.mark.asyncio
async def test_manifest_title_overrides_extracted_title(rig: _Rig) -> None:
    # documents-kind (nautiluszim) archives: a PDF entry's libzim/filename
    # title is the stub ("Water (1).pdf"); ``article_documents.title`` holds the
    # human title. The indexer overrides the title from the manifest in one
    # batched lookup; paths without a manifest row keep their extracted title.
    manifest = {
        rig.paths[0]: "Distillation For Home Water Treatment",
        rig.paths[2]: "Giardia: Drinking Water Factsheet",
    }
    async with rig.db.write() as conn:
        await conn.executemany(
            "INSERT INTO article_documents(zim_id, doc_path, title, description, author, doc_mime) "
            "VALUES(?,?,?,?,?,?)",
            [(1, p, t, None, None, "application/pdf") for p, t in manifest.items()],
        )
    await IndexZimJob().run(rig.handle, {"zim_id": 1, "depth": 1})

    async with rig.db.read() as conn:
        cur = await conn.execute("SELECT entry_path, title FROM articles WHERE zim_id=1")
        rows = {str(r["entry_path"]): str(r["title"]) for r in await cur.fetchall()}
    # Manifest titles win for the two indexed rows that have a catalog entry.
    assert rows[rig.paths[0]] == "Distillation For Home Water Treatment"
    assert rows[rig.paths[2]] == "Giardia: Drinking Water Factsheet"
    # Paths without a manifest row keep their extracted title ("Article N").
    assert rows[rig.paths[1]] == "Article 1"
    assert rows[rig.paths[3]] == "Article 3"
    assert rows[rig.paths[4]] == "Article 4"


@pytest.mark.asyncio
async def test_documents_kind_indexes_only_manifest_paths(rig: _Rig) -> None:
    # A documents-kind archive indexes EXACTLY the manifest doc_paths (the kind-
    # aware path source in registry.py returns those only). Here the rig's fake
    # archive already plays the role of that filtered path set, so we prove the
    # combination: every indexed article is a manifest doc_path, every manifest
    # doc_path is indexed, and all carry the manifest title. No junk survives.
    manifest = {p: f"Document {i}" for i, p in enumerate(rig.paths)}
    async with rig.db.write() as conn:
        await conn.executemany(
            "INSERT INTO article_documents(zim_id, doc_path, title, description, author, doc_mime) "
            "VALUES(?,?,?,?,?,?)",
            [(1, p, t, None, None, "application/pdf") for p, t in manifest.items()],
        )
    await IndexZimJob().run(rig.handle, {"zim_id": 1, "depth": 1})

    async with rig.db.read() as conn:
        cur = await conn.execute(
            "SELECT entry_path, title, char_len FROM articles WHERE zim_id=1 ORDER BY entry_path"
        )
        rows = [dict(r) for r in await cur.fetchall()]
    indexed_paths = {r["entry_path"] for r in rows}
    assert indexed_paths == set(rig.paths)  # exactly the manifest set, zero extras
    assert all(r["char_len"] > 0 for r in rows)
    assert all(r["title"] == f"Document {i}" for i, r in enumerate(rows))
