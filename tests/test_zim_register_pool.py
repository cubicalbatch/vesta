"""Registration-time alias mining must not occupy the interactive pool.

AUDIT_0822 P5: ``mine_aliases`` walks every entry (~29 s for Simple Wikipedia's
400 k) but used to be dispatched onto the same bounded executor that serves
reader/media reads, search/suggest, random draws and query-time extraction —
so one large registration pinned an interactive worker and two measurably
stalled the appliance. These tests pin the split: mining runs on its own
one-shot registration executor and never lands on the shared pool, its output
is identical to calling ``mine_aliases`` directly, and interactive reads stay
prompt while a slow mining pass is still running.
"""

from __future__ import annotations

import asyncio
import functools
import threading
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from fixtures.tiny_zim import LONG_ARTICLE_PATH, REDIRECT_TARGET, build_tiny_zim
from vesta.db.connection import Database
from vesta.db.migrations import run_migrations
from vesta.zim import aliases as aliases_mod
from vesta.zim import bind_registry
from vesta.zim import registry as registry_mod
from vesta.zim.registry import ArchiveRegistry


class _RecordingPool:
    """Executor stand-in that records every ``submit`` and delegates to a real
    thread pool, so instrumented submissions still actually run."""

    def __init__(self, inner: ThreadPoolExecutor) -> None:
        self.submissions: list[object] = []
        self._inner = inner

    def submit(self, fn: object, /, *args: object, **kwargs: object) -> object:
        # Dispatches arrive as ``functools.partial(target, ...)`` (run_in_executor
        # wraps them); record the unwrapped target so identity checks work.
        target = fn.func if isinstance(fn, functools.partial) else fn
        self.submissions.append(target)
        return self._inner.submit(fn, *args, **kwargs)  # type: ignore[arg-type]

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self._inner.shutdown(wait=wait, cancel_futures=cancel_futures)


@pytest.fixture
async def registry(tmp_path: Path) -> AsyncIterator[ArchiveRegistry]:
    """A registry holding one registered fixture archive."""
    zims = tmp_path / "zims"
    zims.mkdir(parents=True, exist_ok=True)
    build_tiny_zim(zims / "one.zim")
    db = Database(str(tmp_path / "vesta.db"), busy_timeout_ms=1000)
    await db.start()
    async with db.write() as conn:
        await run_migrations(conn)
    reg = ArchiveRegistry(db=db, zims_dir=zims, read_pool_size=4, cluster_cache_mb=32)
    bind_registry(reg)
    try:
        await reg.start()
        yield reg
    finally:
        bind_registry(None)
        await reg.stop()
        await db.stop()


async def test_mining_never_submits_to_interactive_pool(
    registry: ArchiveRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Instrument both executors: mine_aliases must be submitted to the
    registration executor created per call, never to the read/search pool."""
    # Registration pools are created inside _dispatch_registration; record each.
    reg_pools: list[_RecordingPool] = []
    real_tpe = registry_mod.ThreadPoolExecutor

    def recording_factory(**kwargs: Any) -> _RecordingPool:
        pool = _RecordingPool(real_tpe(**kwargs))
        reg_pools.append(pool)
        return pool

    monkeypatch.setattr(registry_mod, "ThreadPoolExecutor", recording_factory)

    # Wrap the interactive pool in the same recording shape (still delegating).
    interactive_spy = _RecordingPool(registry._pool)
    registry._pool = interactive_spy  # type: ignore[assignment]

    # Spy the mining function itself (it still calls the real one).
    seen_archives: list[object] = []
    real_mine = aliases_mod.mine_aliases

    def spy_mine(archive: object, **kwargs: object) -> list[tuple[str, str]]:
        seen_archives.append(archive)
        return real_mine(archive, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(aliases_mod, "mine_aliases", spy_mine)

    build_tiny_zim(registry._zims_dir / "two.zim")
    result = await registry.rescan()

    assert len(result.added) == 1
    assert len(seen_archives) == 1, "mining must run exactly once for the new archive"
    assert spy_mine not in interactive_spy.submissions, (
        "mine_aliases was submitted to the INTERACTIVE read/search pool"
    )
    # Exactly one registration pool is created (per mining call), carrying
    # exactly the mining submission.
    assert len(reg_pools) == 1, f"expected 1 registration executor, saw {len(reg_pools)}"
    assert reg_pools[0].submissions == [spy_mine]


async def test_mined_output_identical_to_direct_mine(
    registry: ArchiveRegistry, tmp_path: Path
) -> None:
    """Only WHERE mining runs changes: the stored pairs equal a fresh direct
    ``mine_aliases`` call over the same archive."""
    from libzim.reader import Archive as LibzimArchive

    build_tiny_zim(tmp_path / "zims" / "two.zim")
    result = await registry.rescan()
    assert len(result.added) == 1
    zim_id = result.added[0]

    arc = LibzimArchive(str(tmp_path / "zims" / "two.zim"))
    expected = set(aliases_mod.mine_aliases(arc))
    assert ("USA", REDIRECT_TARGET) in expected  # sanity: fixture redirect present

    fetched: list[tuple[str, str]] = []
    async with registry._db.read() as conn:
        cur = await conn.execute("SELECT source, target FROM aliases WHERE zim_id=?", (zim_id,))
        for r in await cur.fetchall():
            fetched.append((str(r["source"]), str(r["target"])))

    assert set(fetched) == expected


async def test_interactive_reads_prompt_while_mining_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saturation regression: while a slow fake mine occupies the registration
    executor, an interactive read on an already-open archive completes promptly.

    Deterministic with a 1-worker interactive pool: pre-fix code queued the read
    BEHIND the slow mine on that single worker; post-fix the read rides the free
    interactive worker immediately."""
    zims = tmp_path / "zims"
    zims.mkdir(parents=True, exist_ok=True)
    build_tiny_zim(zims / "one.zim")
    db = Database(str(tmp_path / "vesta.db"), busy_timeout_ms=1000)
    await db.start()
    async with db.write() as conn:
        await run_migrations(conn)
    reg = ArchiveRegistry(db=db, zims_dir=zims, read_pool_size=1, cluster_cache_mb=32)
    bind_registry(reg)
    try:
        await reg.start()

        mine_started = threading.Event()
        mine_finished = threading.Event()

        def slow_mine(archive: object, **kwargs: object) -> list[tuple[str, str]]:
            mine_started.set()
            time.sleep(0.5)
            mine_finished.set()
            return []

        monkeypatch.setattr(aliases_mod, "mine_aliases", slow_mine)

        build_tiny_zim(zims / "two.zim")
        scan_task = asyncio.create_task(reg.rescan())

        for _ in range(500):
            if mine_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert mine_started.is_set(), "slow mine never started"

        t0 = time.perf_counter()
        raw = await reg.get(1).read(LONG_ARTICLE_PATH)
        read_seconds = time.perf_counter() - t0
        assert raw.title == "Albert Einstein"
        assert read_seconds < 0.25, f"interactive read stalled {read_seconds:.3f}s behind mining"
        assert not mine_finished.is_set(), "mine finished before the read — test race"

        result = await scan_task
        assert len(result.added) == 1
    finally:
        bind_registry(None)
        await reg.stop()
        await db.stop()
