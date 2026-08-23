"""Cross-process index-build leases (AUDIT_0822 M7).

The detached ``vesta index`` CLI and the server-side ``index_zim`` job are two
OS processes writing the same tables for one ``zim_id``. Migration 0014's
``index_leases`` row is the gate both claim through ``vesta.index.leases``;
liveness is an ``os.kill(pid, 0)`` probe against the recorded holder pid, so
tests simulate the OTHER indexer with a real short-lived subprocess (a live
foreign pid) or its reaped corpse (a dead pid) rather than spawning full index
builds. Covered here:

* claim protocol: free ⇒ claim; live foreign ⇒ :class:`IndexLeaseHeld`;
  dead/stale/own-process holder ⇒ takeover; guarded release;
* the job refuses to build under a foreign lease and leaves the archive row
  and store untouched, and releases on every exit path;
* the API trigger answers 409 naming the holder instead of enqueueing a
  doomed build (and enqueues normally once the lease is free/dead);
* the CLI refuses cleanly (nonzero, nothing cancelled, nothing written).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
import pytest_asyncio

from vesta.cli import _run_index
from vesta.db.connection import Database
from vesta.db.migrations import run_migrations
from vesta.index import bind_runtime
from vesta.index.job import IndexZimJob
from vesta.index.leases import (
    IndexLeaseHeld,
    acquire_index_lease,
    active_holder,
    describe,
    release_index_lease,
)

pytestmark = pytest.mark.asyncio


# ── simulating the other process ──────────────────────────────────────────────


@contextmanager
def _live_foreign_process() -> Iterator[int]:
    """Yield the pid of a REAL other process (the 'other indexer') for the
    duration of the block, then kill it."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        yield proc.pid
    finally:
        proc.kill()
        proc.wait()


def _dead_pid() -> int:
    """A pid whose process is already reaped — the crashed-indexer case."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.kill()
    proc.wait()
    return proc.pid


def _iso(*, ago_s: int = 0) -> str:
    when = _dt.datetime.now(_dt.UTC) - _dt.timedelta(seconds=ago_s)
    return when.replace(microsecond=0).isoformat()


async def _hold_lease(
    db: Database, zim_id: int, owner_id: str, pid: int, *, acquired_at: str | None = None
) -> None:
    """Seed a lease row directly, as the foreign holder would have written it."""
    async with db.write() as conn:
        await conn.execute(
            "INSERT OR REPLACE INTO index_leases(zim_id, owner_id, pid, acquired_at) "
            "VALUES(?,?,?,?)",
            (zim_id, owner_id, pid, acquired_at or _iso()),
        )


async def _lease_row(db: Database, zim_id: int) -> dict[str, Any]:
    async with (
        db.read() as conn,
        conn.execute("SELECT * FROM index_leases WHERE zim_id=?", (zim_id,)) as cur,
    ):
        row = await cur.fetchone()
    assert row is not None, "expected a lease row"
    return dict(row)


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def lease_db(tmp_db_path: Path) -> AsyncIterator[Database]:
    db = Database(str(tmp_db_path), busy_timeout_ms=2000)
    await db.start()
    try:
        async with db.write() as conn:
            await run_migrations(conn)
            await conn.execute("INSERT INTO zims(id, path) VALUES (1, '/fake.zim')")
        yield db
    finally:
        await db.stop()


class _EmptyArchive:
    async def text_entry_paths(self) -> list[str]:
        return []


class _EmptyArchiveRegistry:
    def get(self, zim_id: int) -> _EmptyArchive:
        return _EmptyArchive()


class _DummyStore:  # never touched: the empty-archive build returns before store use
    pass


class _StubHandle:
    def __init__(self) -> None:
        self.checkpoints: list[dict[str, Any]] = []

    async def progress(self, done: int, total: int, message: str) -> None:
        pass

    async def checkpoint(self, blob: Mapping[str, Any]) -> None:
        self.checkpoints.append(dict(blob))

    def cancelled(self) -> bool:
        return False


@pytest_asyncio.fixture
async def job_rig(lease_db: Database) -> AsyncIterator[Database]:
    """Bind just enough runtime for ``IndexZimJob.run`` to reach the lease
    claim and complete an empty-archive build without encoders/pools."""
    from vesta import config

    config.configure(env={})
    bind_runtime(lease_db, _EmptyArchiveRegistry(), (lambda: None))
    from vesta.vectors import bind_store

    bind_store(_DummyStore())  # type: ignore[arg-type]
    try:
        yield lease_db
    finally:
        bind_runtime(None, None, None)
        bind_store(None)
        config.configure(env={})


# ── the claim protocol ────────────────────────────────────────────────────────


async def test_free_archive_claims_then_releases(lease_db: Database) -> None:
    await acquire_index_lease(lease_db, 1, owner_id="cli")
    row = await _lease_row(lease_db, 1)
    assert row["owner_id"] == "cli" and row["pid"] == os.getpid()

    # Our own process never reads as contention...
    assert await active_holder(lease_db, 1) is None
    # ...and release drops the row so the next acquirer proceeds.
    await release_index_lease(lease_db, 1, owner_id="cli")
    async with (
        lease_db.read() as conn,
        conn.execute("SELECT COUNT(*) FROM index_leases WHERE zim_id=1") as cur,
    ):
        assert (await cur.fetchone())[0] == 0


async def test_live_foreign_holder_refuses_the_claim(lease_db: Database) -> None:
    with _live_foreign_process() as foreign_pid:
        await _hold_lease(lease_db, 1, "server", foreign_pid)
        with pytest.raises(IndexLeaseHeld) as excinfo:
            await acquire_index_lease(lease_db, 1, owner_id="cli")
        assert "server" in excinfo.value.holder and str(foreign_pid) in excinfo.value.holder

        # The read-only view agrees (this is what the API 409 names).
        holder = await active_holder(lease_db, 1)
        assert holder == describe("server", foreign_pid)

    # Refusal left the holder's lease exactly as it was.
    row = await _lease_row(lease_db, 1)
    assert row["owner_id"] == "server" and row["pid"] == foreign_pid


async def test_crashed_owner_is_taken_over(lease_db: Database) -> None:
    await _hold_lease(lease_db, 1, "server", _dead_pid())
    await acquire_index_lease(lease_db, 1, owner_id="cli")
    row = await _lease_row(lease_db, 1)
    assert row["owner_id"] == "cli"


async def test_stale_holder_is_taken_over_even_while_alive(lease_db: Database) -> None:
    # A wedged-but-alive holder past the staleness ceiling must not wedge the
    # archive forever (reboot / pid-reuse escape hatch).
    with _live_foreign_process() as foreign_pid:
        await _hold_lease(lease_db, 1, "server", foreign_pid, acquired_at=_iso(ago_s=13 * 3600))
        await acquire_index_lease(lease_db, 1, owner_id="cli")
    assert (await _lease_row(lease_db, 1))["owner_id"] == "cli"


async def test_release_never_deletes_a_successors_lease(lease_db: Database) -> None:
    # A superseded holder's late release (guarded by owner+pid) cannot drop a
    # newer holder's lease.
    with _live_foreign_process() as foreign_pid:
        await _hold_lease(lease_db, 1, "server", foreign_pid)
        await release_index_lease(lease_db, 1, owner_id="cli")
        row = await _lease_row(lease_db, 1)
        assert row["owner_id"] == "server" and row["pid"] == foreign_pid


# ── the job is the hard gate ──────────────────────────────────────────────────


async def test_job_refuses_under_a_foreign_lease_and_touches_nothing(
    job_rig: Database,
) -> None:
    with _live_foreign_process() as foreign_pid:
        await _hold_lease(job_rig, 1, "server", foreign_pid)
        with pytest.raises(IndexLeaseHeld):
            await IndexZimJob().run(_StubHandle(), {"zim_id": 1, "depth": 1})

        # The refused build left no trace: status untouched, no chunks, no wipe…
        async with (
            job_rig.read() as conn,
            conn.execute("SELECT index_status FROM zims WHERE id=1") as cur,
        ):
            assert (await cur.fetchone())["index_status"] in (None, "none")
    async with (
        job_rig.read() as conn,
        conn.execute("SELECT COUNT(*) FROM chunks WHERE zim_id=1") as cur,
    ):
        assert (await cur.fetchone())[0] == 0
    # …and the holder keeps its lease.
    assert (await _lease_row(job_rig, 1))["pid"] == foreign_pid


async def test_job_releases_after_completion_and_can_rerun(job_rig: Database) -> None:
    handle = _StubHandle()
    await IndexZimJob().run(handle, {"zim_id": 1, "depth": 1})
    assert await active_holder(job_rig, 1) is None

    # The normal sequential flow is unchanged: build → release → build.
    await IndexZimJob().run(handle, {"zim_id": 1, "depth": 1})


# ── the API trigger fails fast with a 409 ─────────────────────────────────────


async def test_trigger_conflicts_with_a_live_cli_build(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    client, zim_id = app_client_with_zim
    app = client._transport.app  # type: ignore[attr-defined]
    db = app.state.vesta.db  # type: ignore[attr-defined]
    with _live_foreign_process() as foreign_pid:
        await _hold_lease(db, zim_id, "cli", foreign_pid)

        resp = await client.post(f"/api/zims/{zim_id}/index", json={"depth": 1})
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert "cli" in detail and str(foreign_pid) in detail

        # No doomed job was enqueued behind the 409.
        jobs = (await client.get("/api/jobs")).json()["jobs"]
        assert not [j for j in jobs if j["type"] == "index_zim"]


async def test_trigger_enqueues_once_the_blocking_lease_is_dead(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    client, zim_id = app_client_with_zim
    app = client._transport.app  # type: ignore[attr-defined]
    db = app.state.vesta.db  # type: ignore[attr-defined]
    await _hold_lease(db, zim_id, "cli", _dead_pid())

    resp = await client.post(f"/api/zims/{zim_id}/index", json={"depth": 1})
    assert resp.status_code == 200  # a dead holder never blocks; the job takes over
    jobs = (await client.get("/api/jobs")).json()["jobs"]
    assert [j for j in jobs if j["type"] == "index_zim" and j["id"] == resp.json()["job_id"]]


# ── the CLI refuses cleanly, writing nothing ──────────────────────────────────


class _MustNotRunJob:
    """Stands in for ``IndexZimJob``: construction alone means the CLI got far
    enough to launch a second writer, which must not happen."""

    instances: ClassVar[list[_MustNotRunJob]] = []

    def __init__(self) -> None:
        type(self).instances.append(self)

    async def run(self, handle: Any, params: Mapping[str, Any]) -> None:
        raise AssertionError("a refused CLI build must not enter the index job")


async def test_cli_index_refuses_cleanly_while_server_build_is_live(
    lease_db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vesta import config

    config.configure(env={})
    _MustNotRunJob.instances.clear()
    monkeypatch.setattr("vesta.index.job.IndexZimJob", _MustNotRunJob)

    # A live server-side build holds the archive, and its (still non-terminal)
    # job row is present — the refusal must happen BEFORE the CLI cancels it.
    with _live_foreign_process() as foreign_pid:
        await _hold_lease(lease_db, 1, "server", foreign_pid)
        async with lease_db.write() as conn:
            await conn.execute(
                "INSERT INTO jobs(type, target, params, status, progress, total, "
                "created_at, updated_at) VALUES('index_zim', '1', ?, 'running', 0, 0, "
                "'2026-01-01T00:00:00', '2026-01-01T00:00:00')",
                ('{"zim_id": 1, "depth": 1}',),
            )

        code = await _run_index(
            type("State", (), {"db": lease_db})(),  # type: ignore[arg-type]
            argparse.Namespace(depth=1, zim="1", fresh=False, data_dir=str(tmp_path)),
        )

    assert code == 1
    assert _MustNotRunJob.instances == [], "no second writer may start"

    # Nothing was written: the stranded job row survives untouched, the holder
    # keeps its lease, and no chunks appeared.
    async with (
        lease_db.read() as conn,
        conn.execute("SELECT status FROM jobs WHERE type='index_zim'") as cur,
    ):
        assert (await cur.fetchone())["status"] == "running"
    async with (
        lease_db.read() as conn,
        conn.execute("SELECT COUNT(*) FROM chunks WHERE zim_id=1") as cur,
    ):
        assert (await cur.fetchone())[0] == 0
