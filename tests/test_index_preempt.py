"""CPU-preemption tests.

The load-bearing properties:

* an interactive request SIGSTOPs the registered indexer workers; the last
  release SIGCONTs them (preemption by suspension, demonstrably active);
* nested requests suspend once and resume once;
* a disabled coordinator never signals;
* swapping the PID set while suspended resumes the old PIDs first (no stray
  SIGSTOP on a recycled PID);
* ``yield_if_busy`` suspends the indexer while work is in flight, and the
  duty-cycle floor guarantees a minimum progress slice under continuous load
  (08 Risks: "Preemption starves the indexer to a standstill on a busy box").

Signals are intercepted by monkeypatching ``_send_signal`` — no real processes
are stopped in tests.
"""

from __future__ import annotations

import asyncio
import signal

import pytest

from vesta.index import preempt
from vesta.index.preempt import MIN_DUTY_CYCLE, PreemptionCoordinator


@pytest.fixture
def signals(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(preempt, "_send_signal", lambda pid, sig: sent.append((pid, sig)))
    return sent


def _stops(sent: list[tuple[int, int]]) -> list[int]:
    return [pid for pid, sig in sent if sig == signal.SIGSTOP]


def _conts(sent: list[tuple[int, int]]) -> list[int]:
    return [pid for pid, sig in sent if sig == signal.SIGCONT]


# ── acquire / release ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_first_acquire_stops_last_release_continues(signals: list) -> None:
    c = PreemptionCoordinator(enabled=True, idle_ms=0)
    c.set_worker_pids({111, 222})
    await c.acquire()
    assert sorted(_stops(signals)) == [111, 222]
    assert _conts(signals) == []
    await c.release()
    assert sorted(_conts(signals)) == [111, 222]


@pytest.mark.asyncio
async def test_nested_acquires_suspend_once_resume_once(signals: list) -> None:
    c = PreemptionCoordinator(enabled=True, idle_ms=0)
    c.set_worker_pids({42})
    await c.acquire()
    await c.acquire()
    assert _stops(signals) == [42]  # no double SIGSTOP
    await c.release()
    assert _conts(signals) == []  # still one request in flight
    await c.release()
    assert _conts(signals) == [42]


@pytest.mark.asyncio
async def test_disabled_coordinator_never_signals(signals: list) -> None:
    c = PreemptionCoordinator(enabled=False, idle_ms=0)
    c.set_worker_pids({7})
    await c.acquire()
    await c.release()
    assert signals == []
    assert c.effective_rate_fraction() == 1.0


@pytest.mark.asyncio
async def test_swapping_pid_set_resumes_old_pids_first(signals: list) -> None:
    c = PreemptionCoordinator(enabled=True, idle_ms=0)
    c.set_worker_pids({100})
    await c.acquire()  # suspends 100
    # Pool rebuilt mid-request (e.g. a worker died): the old PID must not keep
    # a stray SIGSTOP, and the coordinator is no longer suspended on the old set.
    c.set_worker_pids({200})
    assert 100 in _conts(signals)
    assert 200 not in _stops(signals)
    await c.release()


# ── yield_if_busy / duty cycle ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_yield_returns_promptly_when_idle(signals: list) -> None:
    c = PreemptionCoordinator(enabled=True, idle_ms=1)
    c.set_worker_pids({5})
    await asyncio.wait_for(c.yield_if_busy(), timeout=1.0)


@pytest.mark.asyncio
async def test_yield_waits_out_an_in_flight_request(signals: list) -> None:
    c = PreemptionCoordinator(enabled=True, idle_ms=0)
    loop = asyncio.get_running_loop()
    # The indexer just ran a batch (running history) — so a suspension does NOT
    # trip the duty floor, and the yield itself drives the SIGSTOP (the request
    # started before the PIDs were registered, so acquire signalled nothing).
    c._duty.record_running(1.0, loop.time())
    await c.acquire()
    assert signals == []
    c.set_worker_pids({9})
    yielded = asyncio.create_task(c.yield_if_busy())
    await asyncio.sleep(0.1)
    assert not yielded.done(), "indexer must stay suspended while a request is in flight"
    assert 9 in _stops(signals), "the yield SIGSTOPs the workers mid-request"
    await c.release()
    await asyncio.wait_for(yielded, timeout=2.0)
    assert 9 in _conts(signals)


@pytest.mark.asyncio
async def test_duty_floor_guarantees_progress_under_continuous_load(signals: list) -> None:
    c = PreemptionCoordinator(enabled=True, idle_ms=0)
    loop = asyncio.get_running_loop()
    c.set_worker_pids({9})
    # Simulate a long suspension history: duty fraction at/below the floor.
    now = loop.time()
    c._duty.record_running(60.0 * MIN_DUTY_CYCLE, now)
    c._duty.record_suspended(60.0 * (1 - MIN_DUTY_CYCLE), loop.time())
    await c.acquire()
    # Starved: the indexer must be allowed to run despite the in-flight request.
    await asyncio.wait_for(c.yield_if_busy(), timeout=1.0)
    await c.release()


@pytest.mark.asyncio
async def test_effective_rate_fraction_reflects_suspension(signals: list) -> None:
    c = PreemptionCoordinator(enabled=True, idle_ms=0)
    loop = asyncio.get_running_loop()
    assert c.effective_rate_fraction() == 1.0  # no history → full speed
    c._duty.record_running(1.0, loop.time())
    c._duty.record_suspended(3.0, loop.time())
    frac = c.effective_rate_fraction()
    assert 0.05 <= frac < 1.0


# ── priority demotion helpers (best-effort, never raise) ─────────────────────


def test_apply_nice_and_ionice_tolerate_refusal() -> None:
    preempt.apply_nice(15)  # a positive delta always succeeds for own process
    preempt.apply_nice(-20)  # raising priority requires CAP_SYS_NICE → tolerated
    preempt.apply_ionice_idle()  # ionice binary may be absent → tolerated
