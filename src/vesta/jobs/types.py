"""Job-type protocols, registry, and the row record.

A job is an ``async def run(job, params)`` registered under a name. The runner
owns persistence, concurrency limits, cancellation and SSE; the type does only
its work, reporting through the :class:`JobHandle` it's handed.

Cooperative cancellation: ``JobHandle.cancelled()`` is
polled by the job; the runner *also* cancels the asyncio task (the hard-kill
path that extends to child processes). A resumed job receives its last
checkpoint in ``params["__checkpoint__"]`` — kept inside ``params`` so the
:class:`JobType` Protocol signature stays exactly as specified.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

#: Reserved params key carrying the last checkpoint blob on a resumed run.
RESUME_CHECKPOINT_KEY = "__checkpoint__"


@runtime_checkable
class JobHandle(Protocol):
    """What a job sees of the runner."""

    async def progress(self, done: int, total: int, message: str) -> None:
        """Report progress. Throttled by the runner (~1/s); final write always lands."""
        ...

    async def checkpoint(self, blob: Mapping[str, Any]) -> None:
        """Persist a resume cursor. Throttled by the runner (~1/s); the newest
        cursor is always flushed on pause/cancel/completion and shutdown."""
        ...

    def cancelled(self) -> bool:
        """Cooperative cancellation flag. Poll between units of work."""
        ...


@runtime_checkable
class JobType(Protocol):
    """A registered job kind. ``name`` is the ``jobs.max_concurrent.<name>`` key."""

    name: str

    async def run(self, job: JobHandle, params: Mapping[str, Any]) -> None:
        """Do the work. Must poll ``job.cancelled()`` and write checkpoints."""
        ...


JOB_TYPES: dict[str, JobType] = {}


def register_job_type(jt: JobType) -> JobType:
    """Register a job-type instance by its ``name``."""
    if not jt.name:
        raise ValueError("JobType.name must be non-empty")
    JOB_TYPES[jt.name] = jt
    return jt


def job_types() -> Mapping[str, JobType]:
    """Read-only view of the registry (for the jobs API)."""
    return dict(JOB_TYPES)


@dataclass(frozen=True)
class JobRecord:
    """The ``jobs`` row, as a frozen domain object (not a Pydantic model)."""

    id: int
    type: str
    target: str | None
    status: str
    progress: int
    total: int
    checkpoint: str | None
    message: str | None
    error: str | None
    rate: float | None
    eta_seconds: int | None
    params: Mapping[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "target": self.target,
            "status": self.status,
            "progress": self.progress,
            "total": self.total,
            "checkpoint": _maybe_json(self.checkpoint),
            "params": dict(self.params),
            "message": self.message,
            "error": self.error,
            "rate": self.rate,
            "eta_seconds": self.eta_seconds,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
        }


def _maybe_json(raw: str | None) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


class NoopJob:
    """A job that counts to ``total`` sleeping ``delay`` seconds per step.

    Exists purely to exercise the runner: progress, checkpoint, cooperative
    cancellation, and resume-from-checkpoint across a restart. It demonstrates
    the resumable contract end-to-end without any product behaviour.
    """

    name = "noop"

    async def run(self, job: JobHandle, params: Mapping[str, Any]) -> None:
        total = int(params.get("total", 10))
        delay = float(params.get("delay", 0.05))
        resume = params.get(RESUME_CHECKPOINT_KEY)
        start = 0
        if isinstance(resume, Mapping):
            start = int(resume.get("i", 0))
        for i in range(start, total):
            if job.cancelled():
                return
            # sleep in small slices so cancellation lands promptly.
            await _sleep_interruptible(job, delay)
            done = i + 1
            await job.progress(done, total, f"step {done}/{total}")
            await job.checkpoint({"i": done})
        # final progress is always emitted (always write final state)
        await job.progress(total, total, "done")


async def _sleep_interruptible(job: JobHandle, seconds: float) -> None:
    """Sleep while staying responsive to ``job.cancelled()``."""
    end = asyncio.get_event_loop().time() + seconds
    while True:
        if job.cancelled():
            return
        remaining = end - asyncio.get_event_loop().time()
        if remaining <= 0:
            return
        await asyncio.sleep(min(remaining, 0.02))


async def maybe_throttle(written: int, start: int, limit_kbps: int, t0: float) -> None:
    """Best-effort per-download bandwidth throttle, shared by both download job
    types (ZIM + model): compares bytes served this run against elapsed time
    and sleeps to keep under ``limit_kbps`` KiB/s (0 = unlimited). The sleep is
    capped at 0.5 s so cancellation stays responsive.
    """
    if limit_kbps <= 0:
        return
    served = written - start
    if served <= 0:
        return
    elapsed = asyncio.get_event_loop().time() - t0
    target_seconds = served / (limit_kbps * 1024)
    ahead = target_seconds - elapsed
    if ahead > 0:
        await asyncio.sleep(min(ahead, 0.5))


# Register the built-in noop job type at import.
register_job_type(NoopJob())
