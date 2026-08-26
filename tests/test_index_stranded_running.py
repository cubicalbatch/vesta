"""AUDIT_0824 M11 — a force-quit must not strand ``index_status='running'``.

The second Ctrl+C raises :class:`KeyboardInterrupt` — a ``BaseException`` —
which used to bypass every error-stamping arm of :meth:`IndexZimJob.run`, and
SIGKILL / power loss runs no arm at all. The row stayed 'running' forever
while the VECTORS seed query counts 'running' as indexed: the next boot
served a partial build's vectors as if complete.

Two mechanisms are covered here:

* the job's BaseException arm stamps resumable-paused before re-raising;
* the startup sweep (``main._reconcile_stranded_index_rows``) flips orphaned
  'running' rows to 'paused' unless a LIVE foreign lease holder owns them.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
import pytest_asyncio

from vesta.cli import _run_index
from vesta.db.connection import Database
from vesta.db.migrations import run_migrations

pytestmark = pytest.mark.asyncio


# ── shared rig ────────────────────────────────────────────────────────────────


class _EmptyArchive:
    async def text_entry_paths(self) -> list[str]:
        return []


class _EmptyArchiveRegistry:
    def get(self, zim_id: int) -> _EmptyArchive:
        return _EmptyArchive()


class _DummyStore:  # never touched: _run_build is stubbed out below
    pass


async def _seed_zim(db: Database, *, status: str = "running") -> None:
    await db.start()
    async with db.write() as conn:
        await run_migrations(conn)
        await conn.execute(
            "INSERT INTO zims(id, path, status, enabled) VALUES (1, '/fake.zim', 'ready', 1)"
        )
        await conn.execute(
            "UPDATE zims SET index_status=?, index_depth=1, index_progress=40 WHERE id=1",
            (status,),
        )


@pytest_asyncio.fixture
async def db(tmp_path: Any) -> AsyncIterator[Database]:
    database = Database(str(tmp_path / "vesta.db"), busy_timeout_ms=2000)
    await _seed_zim(database)
    try:
        yield database
    finally:
        await database.stop()


async def _status(db: Database, zim_id: int = 1) -> str:
    async with (
        db.read() as conn,
        conn.execute("SELECT index_status FROM zims WHERE id=?", (zim_id,)) as cur,
    ):
        row = await cur.fetchone()
    assert row is not None
    return str(row["index_status"])


# ── mechanism 1: the job's BaseException arm ─────────────────────────────────


@contextlib.contextmanager
def _run_build_raises(exc: BaseException) -> Iterator[None]:
    import vesta.index.job as job_mod

    async def _boom(*args: Any, **kwargs: Any) -> None:
        raise exc

    saved = job_mod._run_build
    job_mod._run_build = _boom  # type: ignore[assignment]
    try:
        yield
    finally:
        job_mod._run_build = saved  # type: ignore[assignment]


@pytest.fixture
def _job_runtime(db: Database) -> Iterator[None]:
    from vesta import config
    from vesta.index import bind_runtime
    from vesta.vectors import bind_store

    config.configure(env={})
    bind_runtime(db, _EmptyArchiveRegistry(), (lambda: None))
    bind_store(_DummyStore())  # type: ignore[arg-type]
    try:
        yield
    finally:
        bind_runtime(None, None, None)
        bind_store(None)
        config.configure(env={})


async def test_keyboardinterrupt_through_the_job_stamps_paused(
    db: Database, _job_runtime: None
) -> None:
    """A force-quit (KeyboardInterrupt is a BaseException) leaves the archive
    row resumable-paused — never stranded on 'running' — and drops the lease."""
    from vesta.index import set_indexed_state
    from vesta.index.job import IndexZimJob

    class _StubHandle:
        async def progress(self, done: int, total: int, message: str) -> None: ...
        async def checkpoint(self, blob: dict[str, Any]) -> None: ...
        def cancelled(self) -> bool:
            return False

    # Prove the arm reseeds: pretend the running build had seeded VECTORS.
    set_indexed_state(True)

    with (
        _run_build_raises(KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        await IndexZimJob().run(_StubHandle(), {"zim_id": 1, "depth": 1, "owner": "cli"})

    assert await _status(db) == "paused"
    async with db.read() as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM index_leases")
        assert (await cur.fetchone())[0] == 0
    from vesta.index import _ANY_INDEXED

    assert _ANY_INDEXED is False, "reseed must drop the VECTORS claim of a partial build"


# ── mechanism 1 through the CLI entry point ──────────────────────────────────


async def test_cli_force_quit_exits_130_and_leaves_row_paused(
    db: Database, tmp_path: Any, _job_runtime: None
) -> None:
    """The CLI path end to end: KeyboardInterrupt out of the real job → exit
    code 130, and the zims row is not left 'running'."""
    with _run_build_raises(KeyboardInterrupt):
        code = await _run_index(
            type("State", (), {"db": db})(),  # type: ignore[arg-type]
            argparse.Namespace(depth=1, zim="1", fresh=False, data_dir=str(tmp_path)),
        )

    assert code == 130
    assert await _status(db) == "paused"


# ── mechanism 2: the startup sweep ───────────────────────────────────────────


class _Log:
    def __init__(self) -> None:
        self.infos: list[tuple[str, dict[str, Any]]] = []
        self.warnings: list[tuple[str, dict[str, Any]]] = []

    def info(self, event: str, **kw: Any) -> None:
        self.infos.append((event, kw))

    def warning(self, event: str, **kw: Any) -> None:
        self.warnings.append((event, kw))


async def test_startup_sweep_flips_an_orphaned_running_row(db: Database) -> None:
    """SIGKILL / power loss ran no exit arm; the sweep pauses the orphan."""
    from vesta.main import _reconcile_stranded_index_rows

    log = _Log()
    await _reconcile_stranded_index_rows(db, log)  # type: ignore[arg-type]

    assert await _status(db) == "paused"
    assert any(event == "index.stranded_rows_paused" for event, _ in log.infos)


async def test_startup_sweep_spares_a_live_foreign_holder(db: Database) -> None:
    """A detached indexer actually running right now keeps its 'running' row."""
    from vesta.main import _reconcile_stranded_index_rows

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        now = _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()
        async with db.write() as conn:
            await conn.execute(
                "INSERT INTO index_leases(zim_id, owner_id, pid, acquired_at) VALUES(?,?,?,?)",
                (1, "cli", proc.pid, now),
            )
        log = _Log()
        await _reconcile_stranded_index_rows(db, log)  # type: ignore[arg-type]

        assert await _status(db) == "running"
        assert log.infos == []
    finally:
        proc.kill()
        proc.wait()


async def test_startup_sweep_takes_a_dead_lease_as_orphaned(db: Database) -> None:
    """A leftover lease row whose pid is dead is exactly the SIGKILL case."""
    from vesta.main import _reconcile_stranded_index_rows

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.kill()
    proc.wait()
    now = _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()
    async with db.write() as conn:
        await conn.execute(
            "INSERT INTO index_leases(zim_id, owner_id, pid, acquired_at) VALUES(?,?,?,?)",
            (1, "cli", proc.pid, now),
        )

    await _reconcile_stranded_index_rows(db, _Log())  # type: ignore[arg-type]

    assert await _status(db) == "paused"
