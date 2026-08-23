"""The concrete :class:`JobHandle` the runner hands to a job.

Owns the two design considerations for progress reporting:

* **Write amplification.** Progress on a multi-hour index job would hammer
  SQLite per item. We throttle DB writes to ~1/s and *always* write the final
  state. SSE events are emitted for every update (cheap) so the UI stays live.
  Checkpoint writes share the same ~1/s cadence: the newest cursor is kept
  pending and :meth:`JobHandleImpl.flush_checkpoint` lands it on every
  pause/cancel/completion (and runner shutdown), so resume never loses the
  last offset — only the redundant intermediate writes disappear.
* **Cooperative cancellation.** ``cancelled()`` reflects the runner's flag, so a
  well-behaved job polls it between units of work. The runner also cancels the
  asyncio task as the hard-kill path.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vesta.jobs.runner import JobRunner

#: Minimum spacing between throttled progress DB writes (seconds).
_PROGRESS_WRITE_INTERVAL_S = 1.0


class JobHandleImpl:
    """Runner-side implementation of the :class:`JobHandle` protocol."""

    def __init__(self, runner: JobRunner, job_id: int) -> None:
        self._runner = runner
        self._job_id = job_id
        self._last_write_s: float = 0.0
        self._last_progress: tuple[int, int, str] | None = None
        self._last_checkpoint_s: float = 0.0
        self._pending_checkpoint: dict[str, Any] | None = None

    async def progress(self, done: int, total: int, message: str) -> None:
        self._last_progress = (done, total, message)
        is_final = total > 0 and done >= total
        now = time.monotonic()
        should_persist = is_final or (now - self._last_write_s) >= _PROGRESS_WRITE_INTERVAL_S
        # Always publish to SSE (cheap); throttle only the SQLite write.
        await self._runner._publish_progress(self._job_id, done, total, message)
        if should_persist:
            await self._runner._write_progress(self._job_id, done, total, message, final=is_final)
            self._last_write_s = now

    async def checkpoint(self, blob: Mapping[str, Any]) -> None:
        # Stash first so a pause/cancel/completion flush can never lose this
        # cursor even when the throttled write itself is skipped.
        self._pending_checkpoint = dict(blob)
        now = time.monotonic()
        if (now - self._last_checkpoint_s) >= _PROGRESS_WRITE_INTERVAL_S:
            await self.flush_checkpoint()

    async def flush_checkpoint(self) -> None:
        """Write the newest pending checkpoint now, bypassing the throttle.

        The runner calls this on every terminal/paused/cancelled transition and
        on shutdown of the job; resume correctness depends on the last offset
        always landing."""
        blob = self._pending_checkpoint
        if blob is None:
            return
        self._pending_checkpoint = None
        await self._runner._write_checkpoint(self._job_id, blob)
        self._last_checkpoint_s = time.monotonic()

    def cancelled(self) -> bool:
        return self._runner._is_cancelling(self._job_id)

    def pending_final(self) -> tuple[int, int, str] | None:
        """Last reported progress, for the runner to persist once on completion."""
        return self._last_progress
