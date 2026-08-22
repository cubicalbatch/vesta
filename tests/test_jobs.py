"""Job runner: submit/run/progress, checkpoint, pause/resume, cancel, resume
across a fresh runner instance (the restart-survival DoD)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from vesta import config
from vesta.db.connection import Database
from vesta.db.migrations import run_migrations
from vesta.jobs.runner import JobRunner
from vesta.jobs.types import RESUME_CHECKPOINT_KEY, JobHandle, register_job_type


@pytest.fixture
async def runner(tmp_db_path: Path) -> tuple[Database, JobRunner]:
    config.configure(env={})
    db = Database(str(tmp_db_path), busy_timeout_ms=2000)
    await db.start()
    async with db.write() as conn:
        await run_migrations(conn)
    r = JobRunner(db)
    await r.start()
    return db, r


async def _stop(db: Database, r: JobRunner) -> None:
    await r.stop()
    await db.stop()


@pytest.mark.asyncio
async def test_submit_runs_to_completion(runner: tuple[Database, JobRunner]) -> None:
    db, r = runner
    jid = await r.submit("noop", None, {"total": 5, "delay": 0.0})
    for _ in range(50):
        await asyncio.sleep(0.005)
        rec = await r.get(jid)
        if rec and rec.status == "done":
            break
    rec = await r.get(jid)
    assert rec is not None
    assert rec.status == "done"
    assert rec.progress == rec.total == 5
    await _stop(db, r)


@pytest.mark.asyncio
async def test_unknown_job_type_rejected(runner: tuple[Database, JobRunner]) -> None:
    db, r = runner
    with pytest.raises(ValueError):
        await r.submit("does_not_exist", None)
    await _stop(db, r)


@pytest.mark.asyncio
async def test_checkpoint_is_persisted(runner: tuple[Database, JobRunner]) -> None:
    db, r = runner
    jid = await r.submit("noop", None, {"total": 10, "delay": 0.0})
    rec = None
    for _ in range(50):
        await asyncio.sleep(0.005)
        rec = await r.get(jid)
        if rec and rec.checkpoint is not None:
            break
    assert rec is not None
    # The noop job writes a checkpoint each step; one must have landed.
    assert rec.checkpoint is not None
    assert '"i"' in rec.checkpoint
    await r.cancel(jid)
    await _stop(db, r)


@pytest.mark.asyncio
async def test_pause_resume_continues_from_checkpoint(runner: tuple[Database, JobRunner]) -> None:
    db, r = runner
    jid = await r.submit("noop", None, {"total": 20, "delay": 0.001})
    mid = None
    for _ in range(50):
        await asyncio.sleep(0.005)
        mid = await r.get(jid)
        if mid and mid.status == "running" and mid.progress >= 1:
            break
    assert mid is not None and mid.status == "running"
    paused_ok = await r.pause(jid)
    assert paused_ok
    paused = None
    for _ in range(50):
        await asyncio.sleep(0.005)
        paused = await r.get(jid)
        if paused and paused.status == "paused":
            break
    assert paused is not None and paused.status == "paused"
    paused_at = paused.progress

    await r.resume(jid)
    after = None
    for _ in range(50):
        await asyncio.sleep(0.005)
        after = await r.get(jid)
        if after and (after.progress > paused_at or after.status == "done"):
            break
    assert after is not None
    # Resumed past the pause point (or finished).
    assert after.progress > paused_at or after.status == "done"
    await _stop(db, r)


@pytest.mark.asyncio
async def test_cancel_marks_cancelled(runner: tuple[Database, JobRunner]) -> None:
    db, r = runner
    jid = await r.submit("noop", None, {"total": 1000, "delay": 0.001})
    for _ in range(50):
        await asyncio.sleep(0.005)
        rec = await r.get(jid)
        if rec and rec.status == "running":
            break
    assert await r.cancel(jid) is True
    for _ in range(50):
        await asyncio.sleep(0.005)
        rec = await r.get(jid)
        if rec and rec.status == "cancelled":
            break
    rec = await r.get(jid)
    assert rec is not None and rec.status == "cancelled"
    await _stop(db, r)


@pytest.mark.asyncio
async def test_cancel_while_queued_behind_semaphore(runner: tuple[Database, JobRunner]) -> None:
    """Regression: cancelling a job that's genuinely queued — blocked on the
    per-type concurrency semaphore, never reached status='running' — used to
    die silently. `cancel()` calls `state.task.cancel()`, and a task still
    suspended at `await sem.acquire()` (inside `_acquire_and_run`) delivered
    CancelledError *before* the `cancel_requested` check ever ran, with no
    handler to catch it: `_finish` was never called, so the DB row stayed
    'queued' forever and no SSE status event fired. `test_cancel_marks_cancelled`
    doesn't catch this because its single job has no concurrency contention and
    is already 'running' by the time it's cancelled."""
    db, r = runner
    # noop's default concurrency limit is 1 (jobs.max_concurrent.noop), so the
    # second job is forced to genuinely wait on the semaphore.
    first = await r.submit("noop", None, {"total": 1000, "delay": 0.001})
    second = await r.submit("noop", None, {"total": 5, "delay": 0.001})

    for _ in range(50):
        await asyncio.sleep(0.005)
        first_rec = await r.get(first)
        second_rec = await r.get(second)
        if (
            first_rec
            and first_rec.status == "running"
            and second_rec
            and second_rec.status == "queued"
        ):
            break

    first_rec = await r.get(first)
    second_rec = await r.get(second)
    assert first_rec is not None and first_rec.status == "running"
    assert second_rec is not None and second_rec.status == "queued"

    # Subscribe before cancelling so we can observe the terminal SSE event.
    queue = r._subscribe(second)

    assert await r.cancel(second) is True

    for _ in range(50):
        await asyncio.sleep(0.005)
        rec = await r.get(second)
        if rec is not None and rec.status == "cancelled":
            break
    rec = await r.get(second)
    assert rec is not None
    assert rec.status == "cancelled"
    assert rec.finished_at is not None

    event = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert event["event"] == "status"
    assert event["data"]["status"] == "cancelled"
    assert event["data"]["id"] == second

    await _stop(db, r)


@pytest.mark.asyncio
async def test_list_jobs(runner: tuple[Database, JobRunner]) -> None:
    db, r = runner
    await r.submit("noop", None, {"total": 1, "delay": 0.0})
    await r.submit("noop", None, {"total": 1, "delay": 0.0})
    for _ in range(50):
        await asyncio.sleep(0.005)
        jobs = await r.list_jobs()
        if len(jobs) >= 2:
            break
    jobs = await r.list_jobs()
    assert len(jobs) >= 2
    await _stop(db, r)


@dataclass
class _ResumeProbe:
    """A job that records the index it *resumed from*, so a test can prove the
    second run continued past the persisted checkpoint rather than restarting
    from zero (the "from checkpoint" half of the restart-survival DoD)."""

    name: str = "resume_probe"
    resumed_from: list[int] = field(default_factory=list)

    async def run(self, job: JobHandle, params: Mapping[str, Any]) -> None:
        total = int(params.get("total", 10))
        delay = float(params.get("delay", 0.0))
        resume = params.get(RESUME_CHECKPOINT_KEY)
        start = 0
        if isinstance(resume, Mapping):
            start = int(resume.get("i", 0))
        self.resumed_from.append(start)  # 0 ⇒ fresh run; >0 ⇒ resumed run
        for i in range(start, total):
            if job.cancelled():
                return
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0.001)
            await job.progress(i + 1, total, "x")
            await job.checkpoint({"i": i + 1})
        await job.progress(total, total, "done")


@pytest.mark.asyncio
async def test_resume_actually_continues_from_persisted_checkpoint(
    tmp_db_path: Path,
) -> None:
    """The hard DoD: killed mid-run → resumes *from the persisted checkpoint*
    (not from zero) on restart and completes. Uses a recording job so the resume
    index is observable, and asserts the DB checkpoint cursor survived the crash."""
    config.configure(env={})
    probe = _ResumeProbe()
    register_job_type(probe)
    try:
        # First "process": run, persist a checkpoint, then hard-crash.
        db1 = Database(str(tmp_db_path), busy_timeout_ms=2000)
        await db1.start()
        async with db1.write() as conn:
            await run_migrations(conn)
        r1 = JobRunner(db1)
        await r1.start()
        jid = await r1.submit("resume_probe", None, {"total": 10, "delay": 0.0})
        before = None
        for _ in range(50):
            await asyncio.sleep(0.005)
            before = await r1.get(jid)
            if before is not None and before.progress >= 2 and before.checkpoint is not None:
                break
        before = await r1.get(jid)
        assert before is not None and before.progress >= 1
        persisted_cp = before.checkpoint
        assert persisted_cp is not None
        persisted_i = json.loads(persisted_cp)["i"]
        assert 1 <= persisted_i <= 10
        await r1.stop()
        # A real SIGKILL leaves the row at 'running' (no graceful cancel handler).
        async with db1.write() as conn:
            await conn.execute("UPDATE jobs SET status='running' WHERE id=?", (jid,))
        await db1.stop()

        # Second "process": fresh runner against the same DB.
        db2 = Database(str(tmp_db_path), busy_timeout_ms=2000)
        await db2.start()
        r2 = JobRunner(db2)
        await r2.start()
        for _ in range(50):
            await asyncio.sleep(0.005)
            rec = await r2.get(jid)
            if rec and rec.status in {"done", "error", "cancelled"}:
                break
        rec = await r2.get(jid)
        await r2.stop()
        await db2.stop()

        assert rec is not None
        assert rec.status == "done"
        assert rec.total == 10
        # Two runs: the first fresh (from 0), the second resumed (from persisted_i).
        assert probe.resumed_from[0] == 0
        assert probe.resumed_from[1] == persisted_i
        assert persisted_i > 0  # the second run did NOT restart from zero
    finally:
        from vesta.jobs.types import JOB_TYPES

        JOB_TYPES.pop("resume_probe", None)
