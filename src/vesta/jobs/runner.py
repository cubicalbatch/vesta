"""The async job runner.

One mechanism for all long work: jobs persist to the ``jobs`` table, survive a
restart by resuming from their last checkpoint, throttle progress writes, expose
SSE streams (per-job + global), and obey per-type concurrency limits.

Deliberately *not* a task framework: no retries, no
priorities, no DAGs. The primary job kinds are download and index; the
shape here is the minimum that makes them resumable.

Cancellation is cooperative + hard-kill: a job polls
``JobHandle.cancelled()``; the runner also cancels the asyncio task. Pause stops
a running task and keeps its checkpoint so resume picks up where it left off.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from vesta import config
from vesta.db.connection import Database
from vesta.jobs.handle import JobHandleImpl
from vesta.jobs.types import (
    JOB_TYPES,
    RESUME_CHECKPOINT_KEY,
    JobRecord,
    _maybe_json,
)

_log = logging.getLogger(__name__)

_TERMINAL = frozenset({"done", "error", "cancelled"})
_ACTIVE = frozenset({"running", "queued", "paused"})
_QUEUE_MAX = 64  # bound SSE backlog; a slow consumer drops rather than blocks a job


@dataclass
class _RunState:
    """In-memory state for a job that has been (or is being) scheduled."""

    cancel_requested: bool = False
    pause_requested: bool = False
    task: asyncio.Task[None] | None = None
    started_monotonic: float | None = None
    # The live handle, so pause/cancel/finish can flush its newest pending
    # checkpoint (the handle owns the ~1/s write throttle).
    handle: JobHandleImpl | None = None
    # Last status this runner wrote for the job. The runner is the single
    # in-process writer of ``jobs.status``, so SSE progress events read it from
    # here instead of re-SELECTing the row per event.
    status: str | None = None


class JobRunner:
    """Owns the lifetime of every running job task."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._runs: dict[int, _RunState] = {}
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._job_queues: dict[int, list[asyncio.Queue[dict[str, Any]]]] = {}
        self._global_queues: list[asyncio.Queue[dict[str, Any]]] = []
        self._stopping = False

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Resume jobs interrupted by a crash/restart.

        Any job left ``running`` was killed mid-flight: re-queue it so it
        resumes from its checkpoint. A job left ``queued`` was parked on its
        type semaphore when the process died (SIGKILL/power loss — a clean
        stop checkpoints parked tasks to ``paused`` instead): re-enqueue it,
        or it would strand forever with no task in the new process.
        ``paused`` jobs stay paused until the user resumes.
        """
        async with (
            self._db.read() as conn,
            conn.execute(
                "SELECT id, status FROM jobs WHERE status IN ('running', 'queued')"
            ) as cur,
        ):
            rows = list(await cur.fetchall())
        for job_id, _status in rows:
            await self._set_status(job_id, "queued")
            await self._enqueue(job_id, resume=True)
        if rows:
            _log.info("jobs.resumed", extra={"count": len(rows)})

    async def stop(self) -> None:
        """Cancel every running task so the process can exit cleanly.

        Shutdown cancels stamp live jobs ``paused`` (checkpoint intact) rather
        than terminally cancelled, so the next ``start()``/``resume()`` picks
        them up instead of every clean restart forfeiting the checkpoint.
        """
        self._stopping = True
        tasks = [s.task for s in self._runs.values() if s.task is not None]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
        # Each task's CancelledError handler flushes its own pending checkpoint;
        # sweep any leftover (a handler interrupted by a second cancellation)
        # so shutdown never loses the last resume offset.
        for state in self._runs.values():
            await self._flush_checkpoint(state)

    # ── public actions ─────────────────────────────────────────────────────

    async def submit(
        self,
        type_: str,
        target: str | None,
        params: Mapping[str, Any] | None = None,
    ) -> int:
        """Create a job row and schedule it. Returns the new job id."""
        if type_ not in JOB_TYPES:
            raise ValueError(f"unknown job type {type_!r}")
        params_json = json.dumps(dict(params or {}))
        now = _now_iso()
        async with self._db.write() as conn:
            cur = await conn.execute(
                "INSERT INTO jobs(type, target, params, status, progress, total, "
                "created_at, updated_at) VALUES(?, ?, ?, 'queued', 0, 0, ?, ?)",
                (type_, target, params_json, now, now),
            )
            job_id = int(cur.lastrowid) if cur.lastrowid is not None else 0
        self._runs.setdefault(job_id, _RunState()).status = "queued"
        await self._publish_status(job_id, "queued")
        await self._enqueue(job_id, resume=False)
        return job_id

    async def pause(self, job_id: int) -> bool:
        """Stop a running job, keeping its checkpoint for later resume."""
        state = self._runs.get(job_id)
        record = await self.get(job_id)
        if record is None or record.status != "running":
            return False
        if state is not None:
            state.pause_requested = True
            if state.task is not None:
                state.task.cancel()
        return True

    async def resume(self, job_id: int) -> bool:
        """Resume a paused (or queued) job from its checkpoint."""
        record = await self.get(job_id)
        if record is None or record.status not in {"paused", "queued"}:
            return False
        state = self._runs.get(job_id)
        if state is not None and state.task is not None and not state.task.done():
            # A live task already owns this job: parked on the semaphore,
            # running, or mid-cancel. Spawning a second task would re-run the
            # job body when both tasks eventually reach the semaphore, and
            # clearing the cancel flag under a task that is about to deliver
            # its own CancelledError resurrects a job the user just cancelled.
            return False
        if state is not None:
            state.pause_requested = False
            state.cancel_requested = False
        await self._set_status(job_id, "queued")
        await self._enqueue(job_id, resume=True)
        return True

    async def cancel(self, job_id: int) -> bool:
        """Cooperatively + forcibly cancel a running or queued job."""
        state = self._runs.get(job_id)
        record = await self.get(job_id)
        if record is None or record.status in _TERMINAL:
            return False
        if record.status == "paused" or state is None or state.task is None or state.task.done():
            # Nothing live to cooperate with: a paused job's task already ran
            # to its CancelledError, so cancelling it again is a no-op and the
            # row would stay 'paused' while callers report 'cancelled'. Land
            # the terminal transition here so resume() cannot resurrect it.
            await self._finish(job_id, "cancelled")
            return True
        state.cancel_requested = True
        state.task.cancel()
        return True

    # ── reads ──────────────────────────────────────────────────────────────

    async def get(self, job_id: int) -> JobRecord | None:
        async with (
            self._db.read() as conn,
            conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)) as cur,
        ):
            row = await cur.fetchone()
        return _row_to_record(row) if row is not None else None

    async def list_jobs(self, limit: int = 100) -> list[JobRecord]:
        async with (
            self._db.read() as conn,
            conn.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)) as cur,
        ):
            rows = await cur.fetchall()
        return [_row_to_record(r) for r in rows]

    # ── SSE ────────────────────────────────────────────────────────────────

    async def stream(self, job_id: int) -> AsyncIterator[dict[str, Any]]:
        """Yield events for one job: current snapshot first, then live updates."""
        record = await self.get(job_id)
        if record is None:
            return
        queue = self._subscribe(job_id)
        try:
            yield {"event": "snapshot", "data": record.to_dict()}
            while True:
                event = await queue.get()
                yield event
                data = event.get("data", {})
                if isinstance(data, dict) and data.get("status") in _TERMINAL:
                    return
        finally:
            self._unsubscribe(job_id, queue)

    async def stream_all(self) -> AsyncIterator[dict[str, Any]]:
        """Global stream: a snapshot of all jobs, then every status change."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._global_queues.append(queue)
        try:
            for record in await self.list_jobs():
                yield {"event": "snapshot", "data": record.to_dict()}
            while True:
                event = await queue.get()
                yield event
        finally:
            if queue in self._global_queues:
                self._global_queues.remove(queue)

    def _subscribe(self, job_id: int) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._job_queues.setdefault(job_id, []).append(queue)
        return queue

    def _unsubscribe(self, job_id: int, queue: asyncio.Queue[dict[str, Any]]) -> None:
        queues = self._job_queues.get(job_id)
        if queues and queue in queues:
            queues.remove(queue)

    # ── scheduling / execution ─────────────────────────────────────────────

    async def _enqueue(self, job_id: int, *, resume: bool) -> None:
        record = await self.get(job_id)
        if record is None:
            return
        state = self._runs.setdefault(job_id, _RunState())
        sem = self._semaphore_for(record.type)
        state.task = asyncio.create_task(self._acquire_and_run(job_id, sem), name=f"job-{job_id}")

    async def _acquire_and_run(self, job_id: int, sem: asyncio.Semaphore) -> None:
        # A job cancelled while still blocked on the semaphore (waiting for a
        # concurrency slot, not yet running) never reaches _run_job's own
        # CancelledError handler — cancellation lands at `await sem.acquire()`
        # instead. Catch it here so the job still ends up `cancelled` with an
        # SSE status event, exactly like a job cancelled mid-run.
        try:
            await sem.acquire()
        except asyncio.CancelledError:
            if self._stopping:
                # Shutdown cancel, not a user cancel: keep the job resumable
                # so a later start()/resume() picks it up instead of the row
                # dying terminally on every clean restart.
                await self._set_status(job_id, "paused")
                await self._publish_status(job_id, "paused")
            else:
                await self._finish(job_id, "cancelled")
            return
        try:
            if self._stopping:
                return
            state = self._runs.get(job_id)
            if state is not None and state.cancel_requested:
                await self._finish(job_id, "cancelled")
                return
            # Re-read the row now that we hold the slot: an out-of-band writer
            # (e.g. `vesta index` superseding stranded server-side index jobs)
            # can mark a queued row terminal while this task is parked on the
            # semaphore — no lease is held and no runner flag is set while
            # queued, so the in-memory checks above cannot see it. Never run a
            # job whose row went terminal; republish so SSE subscribers see
            # closure.
            record = await self.get(job_id)
            if record is not None and record.status in _TERMINAL:
                await self._publish_status(job_id, record.status)
                return
            await self._run_job(job_id)
        finally:
            sem.release()

    async def _run_job(self, job_id: int) -> None:
        state = self._runs.setdefault(job_id, _RunState())
        record = await self.get(job_id)
        if record is None:
            return
        jt = JOB_TYPES.get(record.type)
        if jt is None:
            await self._finish(job_id, "error", error=f"unknown job type {record.type}")
            return
        await self._set_status(job_id, "running")
        state.started_monotonic = time.monotonic()
        handle = JobHandleImpl(self, job_id)
        state.handle = handle
        params: dict[str, Any] = dict(record.params)
        cp = _maybe_json(record.checkpoint)
        if cp is not None:
            params[RESUME_CHECKPOINT_KEY] = cp
        try:
            await jt.run(handle, params)
        except asyncio.CancelledError:
            # Land the newest pending checkpoint before the status flip so a
            # paused/cancelled job resumes from the true last offset.
            await self._flush_checkpoint(state)
            if state.pause_requested or self._stopping:
                # A user pause or a graceful-shutdown cancel both keep the job
                # resumable; only an explicit user cancel is terminal.
                await self._set_status(job_id, "paused")
                await self._publish_status(job_id, "paused")
            else:
                await self._finish(job_id, "cancelled")
            raise
        except Exception as exc:
            await self._finish(job_id, "error", error=repr(exc))
            _log.exception("jobs.run_failed", extra={"job_id": job_id, "type": record.type})
            return
        else:
            pending = handle.pending_final()
            if pending is not None:
                done, total, message = pending
                await self._write_progress(job_id, done, total, message, final=True)
            await self._finish(job_id, "done")

    # ── cancellation flag ──────────────────────────────────────────────────

    def _is_cancelling(self, job_id: int) -> bool:
        state = self._runs.get(job_id)
        return state is not None and (state.cancel_requested or state.pause_requested)

    # ── persistence ────────────────────────────────────────────────────────

    async def _write_progress(
        self, job_id: int, done: int, total: int, message: str, *, final: bool
    ) -> None:
        rate, eta = _rate_eta(self._runs.get(job_id), done, total)
        now = _now_iso()
        async with self._db.write() as conn:
            await conn.execute(
                "UPDATE jobs SET progress = ?, total = ?, message = ?, rate = ?, "
                "eta_seconds = ?, updated_at = ? WHERE id = ?",
                (done, total, message, rate, eta, now, job_id),
            )
        await self._publish_progress(job_id, done, total, message)

    async def _write_checkpoint(self, job_id: int, blob: Mapping[str, Any]) -> None:
        now = _now_iso()
        async with self._db.write() as conn:
            await conn.execute(
                "UPDATE jobs SET checkpoint = ?, updated_at = ? WHERE id = ?",
                (json.dumps(dict(blob)), now, job_id),
            )

    async def _flush_checkpoint(self, state: _RunState | None) -> None:
        """Force the job's newest pending checkpoint to disk (no-op if none)."""
        if state is not None and state.handle is not None:
            await state.handle.flush_checkpoint()

    async def _set_status(self, job_id: int, status: str) -> None:
        self._runs.setdefault(job_id, _RunState()).status = status
        now = _now_iso()
        finished = now if status in _TERMINAL else None
        async with self._db.write() as conn:
            await conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ?, finished_at = COALESCE(?, "
                "finished_at) WHERE id = ?",
                (status, now, finished, job_id),
            )

    async def _finish(self, job_id: int, status: str, *, error: str | None = None) -> None:
        # Terminal transition: never lose the last offset (resume correctness).
        await self._flush_checkpoint(self._runs.get(job_id))
        self._runs.setdefault(job_id, _RunState()).status = status
        now = _now_iso()
        async with self._db.write() as conn:
            await conn.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = ?, finished_at = ? "
                "WHERE id = ?",
                (status, error, now, now, job_id),
            )
        await self._publish_status(job_id, status)

    # ── SSE fan-out ────────────────────────────────────────────────────────

    async def _publish_progress(self, job_id: int, done: int, total: int, message: str) -> None:
        # The runner is the single writer of ``jobs.status``: read the cached
        # value instead of re-SELECTing the row on every (throttled) progress
        # event. Fall back to the row only when no run-state exists yet.
        state = self._runs.get(job_id)
        status = state.status if state is not None else None
        if status is None:
            record = await self.get(job_id)
            status = record.status if record is not None else "running"
        await self._publish(
            job_id,
            {
                "event": "progress",
                "data": {
                    "id": job_id,
                    "progress": done,
                    "total": total,
                    "message": message,
                    "status": status,
                },
            },
        )

    async def _publish_status(self, job_id: int, status: str) -> None:
        record = await self.get(job_id)
        await self._publish(
            job_id,
            {
                "event": "status",
                "data": record.to_dict()
                if record is not None
                else {"id": job_id, "status": status},
            },
        )

    async def _publish(self, job_id: int, event: dict[str, Any]) -> None:
        for queue in self._job_queues.get(job_id, []):
            _put_nowait(queue, event)
        for queue in self._global_queues:
            _put_nowait(queue, event)

    # ── helpers ────────────────────────────────────────────────────────────

    def _semaphore_for(self, type_name: str) -> asyncio.Semaphore:
        sem = self._sems.get(type_name)
        if sem is None:
            sem = asyncio.Semaphore(self._concurrency_limit(type_name))
            self._sems[type_name] = sem
        return sem

    @staticmethod
    def _concurrency_limit(type_name: str) -> int:
        """Read ``jobs.max_concurrent.<type>`` if declared, else default to 1.

        New job types declare their own setting; until they do,
        a limit of 1 keeps the indexer CPU-bound one-at-a-time.
        """
        key = f"jobs.max_concurrent.{type_name}"
        registry = config.all_settings()
        descriptor = registry.get(key)
        if descriptor is not None:
            value = config.get(descriptor)
            return int(value) if isinstance(value, int) else 1
        return 1


# ── module helpers ──────────────────────────────────────────────────────────


def _put_nowait(queue: asyncio.Queue[dict[str, Any]], event: dict[str, Any]) -> None:
    # A slow SSE consumer misses an event; the snapshot endpoint still gives the
    # authoritative current state on reconnect. Never block the job.
    with suppress(asyncio.QueueFull):
        queue.put_nowait(event)


def _rate_eta(state: _RunState | None, done: int, total: int) -> tuple[float | None, int | None]:
    if state is None or state.started_monotonic is None or done <= 0:
        return None, None
    elapsed = max(time.monotonic() - state.started_monotonic, 1e-6)
    rate = done / elapsed
    remaining = max(total - done, 0)
    eta = int(remaining / rate) if rate > 0 else None
    return rate, eta


def _now_iso() -> str:
    # UTC, second precision, no telemetry-baiting microseconds. Good enough for a
    # single-user appliance and sorts lexicographically in SQLite TEXT columns.
    import datetime as _dt

    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()


def _row_to_record(row: Any) -> JobRecord:
    raw_params = row["params"]
    params: dict[str, Any] = {}
    if raw_params:
        with suppress(json.JSONDecodeError):
            params = dict(json.loads(raw_params))
    return JobRecord(
        id=int(row["id"]),
        type=row["type"],
        target=row["target"],
        status=row["status"],
        progress=int(row["progress"] or 0),
        total=int(row["total"] or 0),
        checkpoint=row["checkpoint"],
        message=row["message"],
        error=row["error"],
        rate=row["rate"],
        eta_seconds=row["eta_seconds"],
        params=params,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        finished_at=row["finished_at"],
    )
