"""CPU preemption for the indexer.

The app **SIGSTOPs the indexer while an interactive request is in flight and
SIGCONTs when idle** — preemption by suspension, not by polite thread-count
negotiation. The gate is hard: search latency must stay within **2x** while
indexing runs, and without preemption the app feels broken during its
longest-running operation.

Design (two layers, both required):

* **Cooperative yield at batch granularity.** Between extraction/embed batches
  the indexer calls :meth:`PreemptionCoordinator.yield_if_busy`; while any
  interactive request (``/api/search`` / ``/api/answer``) is in flight the
  indexer simply does not schedule the next batch. This is what keeps the
  in-process embedding thread off the CPU during a search.
* **Hard SIGSTOP of worker processes.** Extraction runs in a spawn pool;
  the indexer registers those worker PIDs with the coordinator, which SIGSTOPs
  them the instant a search starts and SIGCONTs them when the box goes idle.
  This is "preemption by suspension" made literal and is demonstrably active.

A **duty-cycle floor** guarantees a minimum progress slice per minute: even under
continuous search load the indexer is allowed to run often enough to make
visible progress, and the effective (degraded) throughput is reported in the
job's ETA.

The in-flight counter is fed by a FastAPI **middleware** (added in ``main``),
NOT by an edit to ``api/search.py`` — a
middleware that matches the search/answer paths is the clean way to signal
"interactive work" without touching the handler.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from dataclasses import dataclass

_log = logging.getLogger(__name__)

#: Paths whose requests count as "interactive" and trip preemption. Health /
#: static / eval polling do NOT, so the dev console's polling can't starve the
#: indexer (the middleware in main.py mirrors this set).
INTERACTIVE_PATH_PREFIXES: tuple[str, ...] = ("/api/search", "/api/answer")

#: Poll interval while suspended (seconds). Short enough that SIGCONT resumes
#: the indexer promptly when a search finishes.
_SUSPEND_POLL_S = 0.02

#: Minimum fraction of wall-clock the indexer is allowed to actually run, even
#: under continuous search load (a guaranteed minimum progress slice per
#: minute). 0.25 ⇒ at worst the indexer runs 25% of the time.
MIN_DUTY_CYCLE = 0.25

#: Window over which the duty cycle is measured (seconds). One minute.
_DUTY_WINDOW_S = 60.0


@dataclass
class _DutyTracker:
    """Approximately-sliding accounting of running vs suspended time.

    Both accumulators decay exponentially (half-life ``_DUTY_WINDOW_S / 2``) on
    every record, so the fraction reflects the *recent* minute, not the
    lifetime average — without keeping an event log. ``record_*`` takes the
    caller's ``loop.time()`` so decay is computed on one clock.
    """

    running_s: float = 0.0
    suspended_s: float = 0.0
    _t: float | None = None

    def _decay(self, now: float) -> None:
        if self._t is None:
            self._t = now
            return
        dt = now - self._t
        self._t = now
        if dt <= 0:
            return
        factor = 0.5 ** (dt / (_DUTY_WINDOW_S / 2))
        self.running_s *= factor
        self.suspended_s *= factor

    def record_running(self, dt: float, now: float) -> None:
        self._decay(now)
        self.running_s += dt

    def record_suspended(self, dt: float, now: float) -> None:
        self._decay(now)
        self.suspended_s += dt

    def duty_fraction(self) -> float:
        total = self.running_s + self.suspended_s
        if total <= 0:
            return 1.0
        return self.running_s / total

    def starved(self) -> bool:
        """True when allowing another suspension would breach the duty floor."""
        return self.duty_fraction() <= MIN_DUTY_CYCLE


class PreemptionCoordinator:
    """Tracks in-flight interactive requests and suspends registered workers.

    The indexer registers its extraction-pool worker PIDs (:meth:`set_worker_pids`);
    the request-path middleware calls :meth:`acquire`/:meth:`release`; the indexer
    calls :meth:`yield_if_busy` between batches. All three are safe to call from
    the event-loop thread.
    """

    def __init__(self, *, enabled: bool = True, idle_ms: int = 50) -> None:
        self._enabled = enabled
        self._idle_ms = max(0, idle_ms)
        self._in_flight = 0
        self._lock = asyncio.Lock()
        self._worker_pids: set[int] = set()
        self._suspended = False
        self._duty = _DutyTracker()
        #: ``loop.time()`` when the indexer last returned from a yield — the
        #: wall-clock between yields is the indexer's *running* time, which is
        #: what feeds the duty floor. ``None`` until the first yield.
        self._last_yield_end: float | None = None

    # ── request-path side (called by the middleware) ───────────────────────

    async def acquire(self) -> None:
        async with self._lock:
            self._in_flight += 1
            if self._enabled and self._worker_pids and not self._suspended:
                self._suspend_workers()

    async def release(self) -> None:
        async with self._lock:
            if self._in_flight > 0:
                self._in_flight -= 1
            if self._enabled and self._suspended and self._in_flight <= 0:
                self._resume_workers()

    # ── indexer side ───────────────────────────────────────────────────────

    def set_worker_pids(self, pids: set[int]) -> None:
        """Register the extraction-pool worker PIDs to SIGSTOP/SIGCONT.

        Called by the indexer when it (re)builds its pool. Pass an empty set when
        the pool is torn down so no stray SIGSTOP targets a recycled PID."""
        # If workers are currently suspended, resume them before swapping the set
        # so a stale PID never holds a SIGSTOP.
        if self._suspended:
            self._resume_workers()
        self._worker_pids = {p for p in pids if p > 0}

    async def yield_if_busy(self) -> None:
        """Between batches: if interactive work is in flight, suspend and wait.

        The wall-clock since the previous yield is counted as the indexer's
        *running* time — that is what the duty-cycle floor measures against, so
        an indexer that just ran a full batch can be suspended, but one that
        has been starved below ``MIN_DUTY_CYCLE`` is let through despite the
        load (08 Risks: "guarantee a minimum progress slice per minute").
        No-op when preemption is disabled.
        """
        if not self._enabled:
            return
        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._last_yield_end is not None:
            self._duty.record_running(min(now - self._last_yield_end, _DUTY_WINDOW_S), now)
        # Always yield a slice so the indexer cooperates with the event loop.
        await asyncio.sleep(self._idle_ms / 1000.0)
        if self._in_flight > 0 and not self._duty.starved():
            # Suspend workers and wait for the box to go idle (or for the duty
            # floor to force a progress slice).
            async with self._lock:
                if self._worker_pids and not self._suspended:
                    self._suspend_workers()
            while self._in_flight > 0 and not self._duty.starved():
                self._duty.record_suspended(_SUSPEND_POLL_S, loop.time())
                await asyncio.sleep(_SUSPEND_POLL_S)
            async with self._lock:
                if self._suspended:
                    # Either the box went idle or the floor forced a resume —
                    # the indexer is about to run, so its workers run too.
                    self._resume_workers()
        self._last_yield_end = loop.time()

    def effective_rate_fraction(self) -> float:
        """The indexer's current effective throughput as a fraction of full-speed
        (1.0 = uninterrupted). The indexer divides its measured rate by this so
        the ETA reflects suspension time honestly."""
        if not self._enabled:
            return 1.0
        f = self._duty.duty_fraction()
        return max(f, 0.05)

    # ── worker signal handling ─────────────────────────────────────────────

    def _suspend_workers(self) -> None:
        if not self._worker_pids:
            return
        for pid in self._worker_pids:
            try:
                _send_signal(pid, signal.SIGSTOP)
            except (ProcessLookupError, PermissionError):
                continue
        self._suspended = True

    def _resume_workers(self) -> None:
        if not self._worker_pids:
            self._suspended = False
            return
        for pid in self._worker_pids:
            try:
                _send_signal(pid, signal.SIGCONT)
            except (ProcessLookupError, PermissionError):
                continue
        self._suspended = False


def _send_signal(pid: int, sig: int) -> None:
    """os.kill wrapper, isolated so tests can monkeypatch the signal path."""
    import os

    os.kill(pid, sig)


def apply_nice(nice_delta: int) -> None:
    """Lower this process's scheduling priority (best-effort; ``nice 15``).

    Called from a spawn-pool worker initializer so extraction/embed workers run
    at reduced priority from the start. Tolerated if the platform refuses
    (already at the floor, container without CAP_SYS_NICE) — degrade-graceful."""
    try:
        import os

        os.nice(nice_delta)
    except (OSError, PermissionError):
        pass


def apply_ionice_idle() -> None:
    """Best-effort ``ionice -c3`` (idle I/O class) for this process.

    The kernel's ioprio interface isn't in the stdlib; shell out to ``ionice``
    when present and tolerated when absent (non-Linux, no binary)."""
    try:
        import shutil
        import subprocess

        ionice = shutil.which("ionice")
        if ionice:
            subprocess.run(  # trusted binary, fixed arg list
                [ionice, "-c", "3", "-p", str(_own_pid())],
                check=False,
                capture_output=True,
            )
    except Exception:
        pass


def _own_pid() -> int:
    import os

    return os.getpid()


__all__ = [
    "INTERACTIVE_PATH_PREFIXES",
    "MIN_DUTY_CYCLE",
    "PreemptionCoordinator",
    "apply_ionice_idle",
    "apply_nice",
]
