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
from vesta.jobs.handle import JobHandleImpl
from vesta.jobs.runner import JobRunner, _now_iso, _RunState
from vesta.jobs.types import (
    JOB_TYPES,
    RESUME_CHECKPOINT_KEY,
    JobHandle,
    JobRecord,
    register_job_type,
)


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


@pytest.mark.asyncio
async def test_graceful_stop_preserves_resumable_row(tmp_db_path: Path) -> None:
    """AUDIT_0824 M14: ``runner.stop()`` must not terminal-cancel live jobs.

    A graceful shutdown stamps a running job ``paused`` with its checkpoint
    intact (never terminally ``cancelled``), so a fresh runner can resume it
    from the checkpoint instead of restarting from zero."""
    config.configure(env={})
    probe = _ResumeProbe()
    register_job_type(probe)
    try:
        # First "process": run, then shut down gracefully mid-flight.
        db1 = Database(str(tmp_db_path), busy_timeout_ms=2000)
        await db1.start()
        async with db1.write() as conn:
            await run_migrations(conn)
        r1 = JobRunner(db1)
        await r1.start()
        jid = await r1.submit("resume_probe", None, {"total": 500, "delay": 0.001})
        await _wait_running_with_checkpoint(r1, jid)

        await r1.stop()  # graceful shutdown while the job is mid-run
        stopped = await r1.get(jid)
        await db1.stop()

        assert stopped is not None
        # Resumable-paused, NOT terminal-cancelled; finished_at stays open.
        assert stopped.status == "paused"
        assert stopped.finished_at is None
        assert stopped.checkpoint is not None
        stopped_i = json.loads(stopped.checkpoint)["i"]
        assert stopped_i >= 2

        # Second "process": fresh runner against the same DB.
        db2 = Database(str(tmp_db_path), busy_timeout_ms=2000)
        await db2.start()
        r2 = JobRunner(db2)
        await r2.start()  # paused rows stay parked until an explicit resume
        parked = await r2.get(jid)
        assert parked is not None and parked.status == "paused"
        assert await r2.resume(jid) is True
        rec = None
        for _ in range(400):
            await asyncio.sleep(0.005)
            rec = await r2.get(jid)
            if rec and rec.status in {"done", "error", "cancelled"}:
                break
        await r2.stop()
        await db2.stop()

        assert rec is not None
        assert rec.status == "done"
        # Two runs: first fresh (from 0), second resumed from the exact
        # checkpoint the shutdown flushed — not restarted from zero.
        assert probe.resumed_from[0] == 0
        assert probe.resumed_from[1] == stopped_i
    finally:
        from vesta.jobs.types import JOB_TYPES

        JOB_TYPES.pop("resume_probe", None)


async def _wait_running_with_checkpoint(r: JobRunner, jid: int) -> JobRecord:
    """Wait until ``jid`` is running with at least one flushed checkpoint."""
    rec = None
    for _ in range(100):
        await asyncio.sleep(0.005)
        rec = await r.get(jid)
        if rec is not None and rec.status == "running" and rec.checkpoint is not None:
            return rec
    raise AssertionError(f"job {jid} never reached running-with-checkpoint: {rec!r}")


# ── checkpoint throttling + progress-publish status cache (AUDIT_0822 P3) ─────


async def _running_job(db: Database, r: JobRunner) -> tuple[int, Any]:
    """Insert a 'running' job row and attach a live handle to its run state,
    so tests can drive ``checkpoint``/``progress`` without a real job task."""
    now = "2026-01-01T00:00:00+00:00"
    async with db.write() as conn:
        cur = await conn.execute(
            "INSERT INTO jobs(type, target, params, status, progress, total, "
            "created_at, updated_at) VALUES('noop', NULL, '{}', 'running', 0, 0, ?, ?)",
            (now, now),
        )
        jid = int(cur.lastrowid) if cur.lastrowid is not None else 0
    state = r._runs.setdefault(jid, _RunState())
    handle = JobHandleImpl(r, jid)
    state.handle = handle
    state.status = "running"
    return jid, handle


@pytest.mark.asyncio
async def test_rapid_checkpoints_write_once_but_final_call_flushes(
    runner: tuple[Database, JobRunner],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N checkpoint calls inside the throttle window cost one DB write; the
    newest cursor is kept pending and an explicit final flush lands it."""
    db, r = runner
    jid, handle = await _running_job(db, r)
    writes: list[dict[str, Any]] = []
    original = r._write_checkpoint

    async def spy(job_id: int, blob: Mapping[str, Any]) -> None:
        writes.append(dict(blob))
        await original(job_id, blob)

    monkeypatch.setattr(r, "_write_checkpoint", spy)

    for i in range(1, 6):
        await handle.checkpoint({"i": i})
    assert len(writes) == 1  # the whole burst cost exactly one write...
    assert writes[0] == {"i": 1}  # ...the first call (window had elapsed)

    # The FINAL call's cursor was never lost: an explicit final flush writes it.
    await handle.flush_checkpoint()
    assert len(writes) == 2
    assert writes[-1] == {"i": 5}
    rec = await r.get(jid)
    assert rec is not None
    assert json.loads(rec.checkpoint or "null") == {"i": 5}
    await _stop(db, r)


@pytest.mark.asyncio
async def test_terminal_transition_flushes_latest_pending_checkpoint(
    runner: tuple[Database, JobRunner],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A terminal transition must land the newest pending checkpoint — resume
    correctness depends on never losing the last offset. ``_finish`` is the
    funnel every cancelled/error/done path goes through."""
    db, r = runner
    jid, handle = await _running_job(db, r)
    writes: list[dict[str, Any]] = []
    original = r._write_checkpoint

    async def spy(job_id: int, blob: Mapping[str, Any]) -> None:
        writes.append(dict(blob))
        await original(job_id, blob)

    monkeypatch.setattr(r, "_write_checkpoint", spy)

    for i in range(1, 4):
        await handle.checkpoint({"i": i})
    assert len(writes) == 1

    await r._finish(jid, "cancelled")
    rec = await r.get(jid)
    assert rec is not None and rec.status == "cancelled"
    assert json.loads(rec.checkpoint or "null") == {"i": 3}
    await _stop(db, r)


@dataclass
class _CheckpointProbe:
    """Writes a rapid burst of checkpoints (well inside the throttle window),
    then blocks until released — so a test can pause it with the newest
    cursor still pending."""

    name: str = "cp_probe"
    burst_done: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)

    async def run(self, job: JobHandle, params: Mapping[str, Any]) -> None:
        for i in range(1, 6):
            await job.checkpoint({"i": i})
        self.burst_done.set()
        await self.release.wait()


@pytest.mark.asyncio
async def test_pause_flushes_latest_pending_checkpoint_and_resumes(
    runner: tuple[Database, JobRunner],
) -> None:
    """Pausing a running job flushes the newest pending checkpoint to the DB,
    and a later resume completes from that offset."""
    db, r = runner
    probe = _CheckpointProbe()
    register_job_type(probe)
    try:
        jid = await r.submit("cp_probe", None)
        for _ in range(100):
            await asyncio.sleep(0.005)
            if probe.burst_done.is_set():
                break
        assert probe.burst_done.is_set()

        assert await r.pause(jid) is True
        rec = None
        for _ in range(50):
            await asyncio.sleep(0.005)
            rec = await r.get(jid)
            if rec and rec.status == "paused":
                break
        assert rec is not None and rec.status == "paused"
        # The burst's LAST cursor ({"i": 5}) landed even though only its first
        # write fell inside the throttle window.
        assert json.loads(rec.checkpoint or "null") == {"i": 5}

        probe.release.set()
        assert await r.resume(jid) is True
        for _ in range(50):
            await asyncio.sleep(0.005)
            rec = await r.get(jid)
            if rec and rec.status == "done":
                break
        assert rec is not None and rec.status == "done"
    finally:
        from vesta.jobs.types import JOB_TYPES

        JOB_TYPES.pop("cp_probe", None)
    await _stop(db, r)


@pytest.mark.asyncio
async def test_progress_publish_does_not_select_per_event(
    runner: tuple[Database, JobRunner],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SSE progress events read the cached status instead of re-SELECTing the
    jobs row on every publish (the old per-event SELECT ran even when the DB
    write itself was throttled away)."""
    db, r = runner
    jid, handle = await _running_job(db, r)
    queue = r._subscribe(jid)
    selects = 0
    original_get = r.get

    async def get_spy(job_id: int) -> object:
        nonlocal selects
        selects += 1
        return await original_get(job_id)

    monkeypatch.setattr(r, "get", get_spy)

    for i in range(1, 8):
        await handle.progress(i, 100, f"step {i}")
    assert selects == 0  # zero row reads for seven publishes
    events: list[dict[str, Any]] = []
    while not queue.empty():
        events.append(queue.get_nowait())
    # SSE still sees every update (only DB writes throttle); the persisted
    # first tick re-publishes, hence 8 events for 7 calls.
    assert [e["data"]["progress"] for e in events] == [1, 1, 2, 3, 4, 5, 6, 7]
    assert events[-1]["event"] == "progress"
    assert events[-1]["data"] == {
        "id": jid,
        "progress": 7,
        "total": 100,
        "message": "step 7",
        "status": "running",
    }
    await _stop(db, r)


@dataclass
class _GateProbe:
    """A job that records every execution (by tag) and parks on a release
    event, so a test can hold the per-type semaphore slot and prove exactly
    which jobs ran — and how many times."""

    name: str
    runs: list[str] = field(default_factory=list)
    release: asyncio.Event = field(default_factory=asyncio.Event)

    async def run(self, job: JobHandle, params: Mapping[str, Any]) -> None:
        self.runs.append(str(params.get("tag", "")))
        await self.release.wait()
        await job.progress(1, 1, "done")


async def _wait_status(r: JobRunner, jid: int, status: str) -> JobRecord:
    rec: JobRecord | None = None
    for _ in range(400):
        await asyncio.sleep(0.005)
        rec = await r.get(jid)
        if rec is not None and rec.status == status:
            return rec
    raise AssertionError(f"job {jid} never reached {status!r}: {rec!r}")


@pytest.mark.asyncio
async def test_resume_on_queued_job_with_live_task_does_not_double_run(
    runner: tuple[Database, JobRunner],
) -> None:
    """AUDIT_0824 M15 interleaving A: resume() accepted 'queued' rows and
    _enqueue spawned unconditionally, so a second resume on an already-queued
    job spawned a twin task; both eventually crossed the per-type semaphore
    and the job body ran twice (for download_zim, re-downloading the file)."""
    db, r = runner
    probe = _GateProbe(name="gate_probe")
    register_job_type(probe)
    try:
        first = await r.submit("gate_probe", None, {"tag": "first"})
        await _wait_status(r, first, "running")
        second = await r.submit("gate_probe", None, {"tag": "second"})
        await _wait_status(r, second, "queued")

        # The queued job already has a live task parked on the semaphore;
        # resuming it must refuse instead of spawning a duplicate.
        assert await r.resume(second) is False

        probe.release.set()
        await _wait_status(r, first, "done")
        await _wait_status(r, second, "done")
        assert probe.runs == ["first", "second"]
    finally:
        JOB_TYPES.pop("gate_probe", None)
    await _stop(db, r)


@pytest.mark.asyncio
async def test_cancel_then_resume_does_not_resurrect_cancelled_job(
    runner: tuple[Database, JobRunner],
) -> None:
    """AUDIT_0824 M15 interleaving B: cancel() flags + cancels the parked
    task, but until that task delivers its CancelledError the row still reads
    'queued' — a resume() squeezed into that window cleared cancel_requested
    and spawned a fresh task, so the job the user cancelled ran anyway."""
    db, r = runner
    probe = _GateProbe(name="resurrect_probe")
    register_job_type(probe)
    try:
        first = await r.submit("resurrect_probe", None, {"tag": "first"})
        await _wait_status(r, first, "running")
        second = await r.submit("resurrect_probe", None, {"tag": "second"})
        await _wait_status(r, second, "queued")

        # Hold the terminal transition so the race window stays open: the
        # cancel has been requested and the parked task cancelled, but the
        # row still reads 'queued' because its CancelledError is undelivered.
        orig_finish = r._finish
        gate = asyncio.Event()

        async def gated_finish(job_id: int, status: str, **kw: Any) -> None:
            if job_id == second:
                await gate.wait()
            await orig_finish(job_id, status, **kw)

        r._finish = gated_finish  # type: ignore[method-assign]
        try:
            assert await r.cancel(second) is True
            # The live mid-cancel task owns the job: resume must refuse.
            assert await r.resume(second) is False
        finally:
            gate.set()
            r._finish = orig_finish  # type: ignore[method-assign]

        await _wait_status(r, second, "cancelled")
        probe.release.set()
        await _wait_status(r, first, "done")
        # The cancelled job never ran again.
        assert probe.runs == ["first"]
    finally:
        JOB_TYPES.pop("resurrect_probe", None)
    await _stop(db, r)


@pytest.mark.asyncio
async def test_cancel_paused_job_finishes_terminal(
    runner: tuple[Database, JobRunner],
) -> None:
    """AUDIT_0824 N22: cancelling a paused job took the live-task branch, but
    pausing already finished that task — Task.cancel() on a done task is a
    no-op, so the row stayed 'paused' while the API reported 'cancelled', the
    per-job SSE stream never closed, and a later resume() cleared the cancel
    intent and resurrected the job."""
    db, r = runner
    probe = _GateProbe(name="paused_cancel_probe")
    register_job_type(probe)
    try:
        jid = await r.submit("paused_cancel_probe", None, {"tag": "only"})
        await _wait_status(r, jid, "running")
        assert await r.pause(jid) is True
        await _wait_status(r, jid, "paused")

        queue = r._subscribe(jid)
        assert await r.cancel(jid) is True

        # The row lands terminally 'cancelled', matching the API's reply.
        rec = await r.get(jid)
        assert rec is not None and rec.status == "cancelled"
        # The terminal status event is published, so SSE subscribers see
        # closure instead of waiting forever.
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event["event"] == "status"
        assert event["data"]["status"] == "cancelled"

        # Resume refuses a cancelled row; the job never runs again.
        assert await r.resume(jid) is False
        probe.release.set()
        await asyncio.sleep(0.05)
        assert probe.runs == ["only"]
        still = await r.get(jid)
        assert still is not None and still.status == "cancelled"
    finally:
        JOB_TYPES.pop("paused_cancel_probe", None)
    await _stop(db, r)


@pytest.mark.asyncio
async def test_row_superseded_while_queued_is_never_run(
    runner: tuple[Database, JobRunner],
) -> None:
    """AUDIT_0824 M15 interleaving C: `vesta index` supersedes stranded
    server-side index jobs by writing status='cancelled' straight to SQLite.
    A task parked on the per-type semaphore holds no lease and no runner flag,
    so it used to run anyway once the slot freed — surfacing a spurious
    lease error. The runner must re-read the row at acquire time."""
    db, r = runner
    probe = _GateProbe(name="supersede_probe")
    register_job_type(probe)
    try:
        first = await r.submit("supersede_probe", None, {"tag": "first"})
        await _wait_status(r, first, "running")
        second = await r.submit("supersede_probe", None, {"tag": "second"})
        await _wait_status(r, second, "queued")

        queue = r._subscribe(second)
        stamp = _now_iso()
        # Mirror cli._cancel_pending_index_jobs: an out-of-band terminal write
        # the runner's in-memory flags cannot see.
        async with db.write() as conn:
            await conn.execute(
                "UPDATE jobs SET status='cancelled', "
                "error='superseded by `vesta index`', updated_at=?, finished_at=? "
                "WHERE id=?",
                (stamp, stamp, second),
            )

        probe.release.set()
        await _wait_status(r, first, "done")
        await asyncio.sleep(0.05)  # let the parked task cross the semaphore
        rec = await r.get(second)
        assert rec is not None
        assert rec.status == "cancelled"
        assert rec.error == "superseded by `vesta index`"
        assert probe.runs == ["first"]
        # SSE subscribers see closure for the already-terminal row.
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event["event"] == "status"
        assert event["data"]["status"] == "cancelled"
    finally:
        JOB_TYPES.pop("supersede_probe", None)
    await _stop(db, r)


@pytest.mark.asyncio
async def test_crash_stranded_queued_job_is_requeued_on_start(
    tmp_db_path: Path,
) -> None:
    """AUDIT_0824 N23: a job left ``queued`` by a hard crash (SIGKILL/power loss
    while its task was parked on the type semaphore) must be re-enqueued by a
    fresh runner's ``start()`` — not strand forever with no task in the new
    process. (A clean stop never strands this way: M14 checkpoints parked tasks
    to ``paused``.)"""
    config.configure(env={})
    probe = _ResumeProbe()
    register_job_type(probe)
    try:
        # Crash debris: a row sitting at 'queued' whose owning process died.
        db1 = Database(str(tmp_db_path), busy_timeout_ms=2000)
        await db1.start()
        async with db1.write() as conn:
            await run_migrations(conn)
        now = _now_iso()
        async with db1.write() as conn:
            cur = await conn.execute(
                "INSERT INTO jobs(type, target, params, status, progress, total, "
                "created_at, updated_at) VALUES('resume_probe', NULL, ?, 'queued', 0, 0, ?, ?)",
                (json.dumps({"total": 3}), now, now),
            )
            jid = int(cur.lastrowid) if cur.lastrowid is not None else 0
        await db1.stop()

        # Fresh "process": start() must pick the queued row up and run it.
        db2 = Database(str(tmp_db_path), busy_timeout_ms=2000)
        await db2.start()
        r2 = JobRunner(db2)
        await r2.start()
        rec = None
        for _ in range(200):
            await asyncio.sleep(0.005)
            rec = await r2.get(jid)
            if rec and rec.status in {"done", "error", "cancelled"}:
                break
        await r2.stop()
        await db2.stop()

        assert rec is not None
        assert rec.status == "done"
        assert rec.progress == rec.total == 3
        # It ran exactly once, fresh from zero (no checkpoint existed).
        assert probe.resumed_from == [0]
    finally:
        JOB_TYPES.pop("resume_probe", None)
