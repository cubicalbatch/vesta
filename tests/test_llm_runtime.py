"""Tests for the LLM runtime against a fake router.

The dev host has no ``llama-server`` binary, so the router's HTTP surface is
stubbed with a tiny Starlette app on a random port, speaking the *measured*
wire shapes from the real binary:

- ``GET /models`` ids are filename STEMS (no ``.gguf``); status is nested at
  ``data[].status.value`` ∈ unloaded | loading | loaded | sleeping.
- ``POST /models/load`` returns ``{"success": true}`` immediately — loading is
  async, so the runtime must poll ``GET /models`` until ``loaded``.
- ``POST /models/unload`` is instant.

The supervisor is stubbed out (the runtime only needs ``router_url``,
``ensure_running``, ``is_running``, ``restart``, ``stop``).
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from vesta import config
from vesta.config.settings import SettingsSnapshot
from vesta.inference.models import (
    estimate_ram_bytes,
    preset_by_filename,
    preset_by_id,
    thinking_for_filename,
)
from vesta.inference.runtime import LlmRuntime, LlmRuntimeError

QWEN_ID = "Qwen3.5-4B-Q4_K_S"
LFM_ID = "LFM2.5-1.2B-Instruct-Q4_K_M"


# ── The fake llama-server router ─────────────────────────────────────────────


class FakeRouter:
    """The router control plane, per the measured wire shapes."""

    def __init__(self, model_ids: list[str]) -> None:
        self._statuses: dict[str, str] = dict.fromkeys(model_ids, "unloaded")
        #: models mid-load and how many polls remain "loading" before "loaded"
        self._loading: dict[str, int] = {}
        self.load_calls: list[str] = []
        self.unload_calls: list[str] = []
        self.fail_unload: bool = False

    async def health(self, request: Any) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def models(self, request: Any) -> JSONResponse:
        for mid in list(self._loading):
            if self._loading[mid] > 0:
                self._loading[mid] -= 1
            else:
                del self._loading[mid]
                self._statuses[mid] = "loaded"
        data = [
            {
                "id": mid,
                "aliases": [],
                "tags": [],
                "object": "model",
                "owned_by": "llamacpp",
                "created": 0,
                "status": {"value": state, "args": [], "preset": ""},
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                },
                "source": "models_dir",
                "can_remove": False,
            }
            for mid, state in self._statuses.items()
        ]
        return JSONResponse({"data": data, "object": "list"})

    async def load(self, request: Any) -> JSONResponse:
        mid = str((await request.json())["model"])
        self.load_calls.append(mid)
        self._statuses[mid] = "loading"
        self._loading[mid] = 1  # one poll reports "loading", the next "loaded"
        return JSONResponse({"success": True})

    async def unload(self, request: Any) -> JSONResponse:
        if self.fail_unload:
            return JSONResponse({"error": "unload failed"}, status_code=500)
        mid = str((await request.json())["model"])
        self.unload_calls.append(mid)
        self._statuses[mid] = "unloaded"
        return JSONResponse({"success": True})


@pytest.fixture()
async def router_server() -> AsyncIterator[tuple[FakeRouter, str]]:
    """A fake router with both preset models, served on a random port."""
    router = FakeRouter([QWEN_ID, LFM_ID])
    app = Starlette(
        routes=[
            Route("/health", router.health),
            Route("/models", router.models),
            Route("/models/load", router.load, methods=["POST"]),
            Route("/models/unload", router.unload, methods=["POST"]),
        ]
    )
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", access_log=False)
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if server.started and server.servers:
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail("fake router did not start")
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield router, f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


# ── Helpers ──────────────────────────────────────────────────────────────────


class StubSupervisor:
    """The supervisor surface LlmRuntime consumes — no real process."""

    def __init__(self, router_url: str) -> None:
        self.router_url = router_url
        self.base_url = f"{router_url}/v1"
        self.ensure_running_calls = 0
        self.restart_calls = 0
        self.stop_calls = 0
        self._running = False
        self.hardware: str | None = None

    async def ensure_running(self) -> None:
        self.ensure_running_calls += 1
        self._running = True

    def is_running(self) -> bool:
        return self._running

    async def restart(self) -> None:
        self.restart_calls += 1

    async def stop(self) -> None:
        self.stop_calls += 1
        self._running = False

    async def hw_class(self) -> str:
        return self.hardware or "cpu"


#: Runtimes created by make_runtime — closed after each test so the httpx
#: client never leaks an "unclosed transport" ResourceWarning (warnings are
#: errors in this suite).
_LIVE_RUNTIMES: list[LlmRuntime] = []


@pytest.fixture(autouse=True)
async def _close_runtimes() -> AsyncIterator[None]:
    yield
    for runtime in _LIVE_RUNTIMES:
        with contextlib.suppress(Exception):
            await runtime.stop()
    _LIVE_RUNTIMES.clear()


def make_snapshot(**overrides: Any) -> SettingsSnapshot:
    """A resolved snapshot with inference defaults plus overrides."""
    config.configure()
    values = dict(config.snapshot().values)
    values.update(overrides)
    return SettingsSnapshot(values=values)


def make_runtime(
    router_url: str,
    snapshot: SettingsSnapshot,
    *,
    models_dir: Path | None = None,
    watchdog_interval_s: float = 0.05,
) -> tuple[LlmRuntime, StubSupervisor]:
    supervisor = StubSupervisor(router_url)
    runtime = LlmRuntime(
        supervisor=supervisor,  # type: ignore[arg-type]
        snapshot=snapshot,
        models_dir=models_dir,
        watchdog_interval_s=watchdog_interval_s,
    )
    _LIVE_RUNTIMES.append(runtime)
    return runtime, supervisor


def local_snapshot(model: str = f"{QWEN_ID}.gguf", **overrides: Any) -> SettingsSnapshot:
    return make_snapshot(
        **{"inference.llm.source": "local", "inference.llm.model": model, **overrides}
    )


async def test_single_entry_fallback() -> None:
    """One model in the registry ⇒ use it even if the configured name differs."""
    from vesta.inference.runtime import _match_model_id

    assert _match_model_id("whatever.gguf", ["Only-Model"]) == "Only-Model"
    # And the other rules, matcher-level:
    assert _match_model_id("Only-Model", ["Only-Model", "Other"]) == "Only-Model"
    assert _match_model_id("dir/Only-Model.gguf", ["Only-Model", "Other"]) == "Only-Model"
    assert _match_model_id("Only-Model.gguf", ["Only-Model", "Other"]) == "Only-Model"
    assert _match_model_id("Neither.gguf", ["A-Model", "B-Model"]) is None
    assert _match_model_id("Any.gguf", []) is None


async def test_resolves_id_by_exact_match(router_server: tuple[FakeRouter, str]) -> None:
    _router, url = router_server
    runtime, _sup = make_runtime(url, local_snapshot(model=QWEN_ID))
    await runtime.ensure_ready()
    assert runtime._resolved_id == QWEN_ID


async def test_resolves_id_by_basename(router_server: tuple[FakeRouter, str]) -> None:
    _router, url = router_server
    runtime, _sup = make_runtime(url, local_snapshot(model=f"some/dir/{QWEN_ID}.gguf"))
    await runtime.ensure_ready()
    assert runtime._resolved_id == QWEN_ID


async def test_no_match_is_an_error(router_server: tuple[FakeRouter, str]) -> None:
    """No match ⇒ state=error naming both the configured filename and the ids
    the router reported (D2: diagnosable from the UI, not container logs)."""
    router, url = router_server
    runtime, _sup = make_runtime(url, local_snapshot(model="Absent-Model.gguf"))
    with pytest.raises(LlmRuntimeError) as excinfo:
        await runtime.ensure_ready()
    assert "Absent-Model.gguf" in str(excinfo.value)
    assert QWEN_ID in str(excinfo.value)
    assert router.load_calls == []
    status = await runtime.status()
    assert status.state == "error"
    assert status.error is not None


# ── ensure_ready ─────────────────────────────────────────────────────────────


async def test_ensure_ready_emits_status_strings_and_ends_loaded(
    router_server: tuple[FakeRouter, str],
) -> None:
    router, url = router_server
    runtime, _sup = make_runtime(url, local_snapshot())
    messages: list[str] = []
    await runtime.ensure_ready(on_status=messages.append)
    assert any("Starting the local model runtime" in m for m in messages)
    assert any("Loading Qwen3.5 4B" in m for m in messages)
    assert router._statuses[QWEN_ID] == "loaded"
    assert runtime._state == "loaded"
    assert runtime._last_used is not None


async def test_ensure_ready_idempotent_and_concurrent_one_load(
    router_server: tuple[FakeRouter, str],
) -> None:
    router, url = router_server
    runtime, _sup = make_runtime(url, local_snapshot())
    await asyncio.gather(runtime.ensure_ready(), runtime.ensure_ready())
    assert router.load_calls == [QWEN_ID]  # one load, two callers
    await runtime.ensure_ready()
    assert router.load_calls == [QWEN_ID]  # already loaded → no re-load


async def test_ensure_ready_warm_emits_no_start_status(
    router_server: tuple[FakeRouter, str],
) -> None:
    """Warm path is silent about "Starting the runtime" — the message claims a
    cold start, so it must only appear when the supervisor actually spawns."""
    _router, url = router_server
    runtime, _sup = make_runtime(url, local_snapshot())
    await runtime.ensure_ready()
    messages: list[str] = []
    await runtime.ensure_ready(on_status=messages.append)
    assert messages == []


async def test_ensure_ready_recovery_clears_previous_error(
    router_server: tuple[FakeRouter, str],
    tmp_path: Path,
) -> None:
    """A previous error (e.g. transient failure) is cleared when ensure_ready succeeds."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / f"{QWEN_ID}.gguf").touch()
    _router, url = router_server
    runtime, _sup = make_runtime(
        url, local_snapshot(model="Absent-Model.gguf"), models_dir=models_dir
    )
    with pytest.raises(LlmRuntimeError):
        await runtime.ensure_ready()
    status = await runtime.status()
    assert status.state == "error"
    assert status.error is not None
    assert runtime._error is not None

    # Rebuild with a valid model and re-run ensure_ready:
    await runtime.rebuild(local_snapshot(model=f"{QWEN_ID}.gguf"))
    # Manually simulate a stale _error if ensure_ready is called without rebuild:
    runtime._error = "stale error message"
    await runtime.ensure_ready()
    assert runtime._error is None
    status = await runtime.status()
    assert status.state == "loaded"
    assert status.error is None


# ── D4: idle watchdog ────────────────────────────────────────────────────────


async def test_watchdog_unloads_after_threshold(router_server: tuple[FakeRouter, str]) -> None:
    router, url = router_server
    runtime, _sup = make_runtime(
        url,
        local_snapshot(
            **{
                "inference.local.idle_unload_seconds": 1,
                "inference.local.stop_server_after_idle_seconds": 3600,
            }
        ),
    )
    await runtime.ensure_ready()
    runtime._last_used = time.monotonic() - 100.0
    await runtime._tick()
    assert router.unload_calls == [QWEN_ID]
    assert runtime._state == "unloaded"


async def test_watchdog_never_unloads_when_disabled(
    router_server: tuple[FakeRouter, str],
) -> None:
    router, url = router_server
    runtime, _sup = make_runtime(
        url,
        local_snapshot(
            **{
                "inference.local.idle_unload_seconds": 0,
                "inference.local.stop_server_after_idle_seconds": 0,
            }
        ),
    )
    await runtime.ensure_ready()
    runtime._last_used = time.monotonic() - 1000.0
    await runtime._tick()
    assert router.unload_calls == []
    assert runtime._state == "loaded"


async def test_watchdog_stops_server_after_longer_threshold(
    router_server: tuple[FakeRouter, str],
) -> None:
    """The second, longer threshold SIGTERMs the child (RSS → ~0)."""
    _router, url = router_server
    runtime, sup = make_runtime(
        url,
        local_snapshot(
            **{
                "inference.local.idle_unload_seconds": 3600,
                "inference.local.stop_server_after_idle_seconds": 1,
            }
        ),
    )
    await runtime.ensure_ready()
    assert sup.is_running()
    runtime._last_used = time.monotonic() - 100.0  # simulate long idle
    await runtime._tick()
    assert sup.stop_calls == 1
    assert runtime._state == "stopped"


async def test_watchdog_never_kills_a_load_in_flight(
    router_server: tuple[FakeRouter, str],
) -> None:
    """D4's "the next question transparently respawns it": after the
    stop-server threshold has already been crossed (stale ``last_used``), a
    respawn must not be SIGTERMed by the next tick. Found against the real
    b10373 binary: the watchdog killed each freshly spawned child mid-startup
    because idle kept its pre-respawn age until ``mark_used`` ran."""
    router, url = router_server
    runtime, sup = make_runtime(
        url,
        local_snapshot(
            **{
                "inference.local.idle_unload_seconds": 1,
                "inference.local.stop_server_after_idle_seconds": 2,
            }
        ),
    )
    await runtime.ensure_ready()
    assert sup.stop_calls == 0
    # Idle far past BOTH thresholds, load in flight: the tick must hold fire.
    runtime._last_used = time.monotonic() - 1000.0
    await runtime.unload()  # so ensure_ready must POST /models/load and poll
    loads_before = len(router.load_calls)
    load_task = asyncio.create_task(runtime.ensure_ready())
    await asyncio.sleep(0)  # let the task start and take _load_lock up to its first await
    await runtime._tick()
    assert sup.stop_calls == 0, "watchdog stopped the server during a load"
    assert router.unload_calls == [QWEN_ID], "watchdog unloaded a model mid-load"

    await asyncio.wait_for(load_task, timeout=5.0)
    assert len(router.load_calls) == loads_before + 1
    assert runtime._state == "loaded"

    # Fresh idle: the next tick is a no-op, not a kill.
    await runtime._tick()
    assert sup.stop_calls == 0
    assert runtime._state == "loaded"


async def test_watchdog_tick_unloads_deterministically(
    router_server: tuple[FakeRouter, str],
) -> None:
    router, url = router_server
    runtime, _sup = make_runtime(
        url,
        local_snapshot(
            **{
                "inference.local.idle_unload_seconds": 1,
            }
        ),
    )
    await runtime.ensure_ready()
    runtime._last_used = time.monotonic() - 100.0
    await runtime._tick()
    assert router.unload_calls == [QWEN_ID]
    assert runtime._state == "unloaded"


async def test_idle_unload_failure_does_not_latch_error_state(
    router_server: tuple[FakeRouter, str],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """I2: router failure on idle-unload must not permanently latch error state in status()."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / f"{QWEN_ID}.gguf").touch()
    router, url = router_server
    runtime, _sup = make_runtime(
        url,
        local_snapshot(
            **{
                "inference.local.idle_unload_seconds": 1,
                "inference.local.stop_server_after_idle_seconds": 3600,
            }
        ),
        models_dir=models_dir,
    )
    await runtime.ensure_ready()
    status = await runtime.status()
    assert status.state == "loaded"
    assert status.error is None
    assert runtime._error is None

    # Router unload fails (e.g. 500 / timeout):
    router.fail_unload = True
    runtime._last_used = time.monotonic() - 100.0

    with caplog.at_level("WARNING", logger="vesta.inference.runtime"):
        await runtime._tick()

    # Warning logged, error state NOT permanently latched:
    assert any("llm.idle_unload_failed" in r.getMessage() for r in caplog.records)
    assert runtime._error is None
    assert runtime._state == "loaded"  # remained loaded since unload failed
    status = await runtime.status()
    assert status.state == "loaded"
    assert status.error is None

    # Next tick after router recovers successfully unloads:
    router.fail_unload = False
    await runtime._tick()
    assert router.unload_calls == [QWEN_ID]
    assert runtime._state == "unloaded"
    status = await runtime.status()
    assert status.state == "unloaded"
    assert status.error is None


# ── status ───────────────────────────────────────────────────────────────────


async def test_status_never_stamps_last_used(router_server: tuple[FakeRouter, str]) -> None:
    _router, url = router_server
    runtime, _sup = make_runtime(url, local_snapshot())
    await runtime.ensure_ready()
    runtime._last_used = time.monotonic() - 10.0
    before = runtime._last_used
    status = await runtime.status()
    assert runtime._last_used == before  # polling must not keep the model alive
    assert status.seconds_since_last_use is not None
    assert status.seconds_since_last_use >= 10.0


async def test_status_local_fields(router_server: tuple[FakeRouter, str], tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / f"{QWEN_ID}.gguf").write_bytes(b"x" * 1000)
    _router, url = router_server
    runtime, _sup = make_runtime(url, local_snapshot(), models_dir=models_dir)
    await runtime.ensure_ready()
    status = await runtime.status()
    assert status.source == "local"
    assert status.configured and status.installed
    assert status.state == "loaded"
    assert status.model_file == f"{QWEN_ID}.gguf"
    assert status.display_name == "Qwen3.5 4B (Q4_K_S)"
    assert status.model_id == QWEN_ID
    assert status.size_bytes == 1000
    assert status.context_size == 8192
    assert status.thinking_supported is True  # Qwen = toggle
    assert status.estimated_ram_bytes == estimate_ram_bytes(1000, 8192, 17408)


async def test_status_local_never_thinking_model(
    router_server: tuple[FakeRouter, str], tmp_path: Path
) -> None:
    """LFM2.5-1.2B-Instruct: thinking=never ⇒ switch inert, thinking False."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / f"{LFM_ID}.gguf").write_bytes(b"x" * 500)
    _router, url = router_server
    runtime, _sup = make_runtime(url, local_snapshot(model=f"{LFM_ID}.gguf"), models_dir=models_dir)
    await runtime.ensure_ready()
    status = await runtime.status()
    assert status.thinking is False
    assert status.thinking_supported is False
    target = runtime.target()
    assert target.enable_thinking is None  # send no kwargs at all


# ── rebuild (D7) ─────────────────────────────────────────────────────────────


async def test_rebuild_on_model_change_drops_cached_id(
    router_server: tuple[FakeRouter, str],
) -> None:
    router, url = router_server
    runtime, sup = make_runtime(url, local_snapshot())
    await runtime.ensure_ready()
    assert runtime._resolved_id == QWEN_ID
    await runtime.rebuild(local_snapshot(model=f"{LFM_ID}.gguf"))
    assert runtime._resolved_id is None
    # The old model was unloaded and the child restarted (a running router does
    # not discover new files — measured gotcha).
    assert router.unload_calls == [QWEN_ID]
    assert sup.restart_calls == 1
    # The new model resolves fresh.
    await runtime.ensure_ready()
    assert runtime._resolved_id == LFM_ID


async def test_rebuild_leaving_local_stops_supervisor(
    router_server: tuple[FakeRouter, str],
) -> None:
    _router, url = router_server
    runtime, sup = make_runtime(url, local_snapshot())
    await runtime.ensure_ready()
    assert sup.is_running()
    assert runtime._resolved_id == QWEN_ID
    await runtime.rebuild(
        make_snapshot(
            **{
                "inference.llm.source": "remote",
                "inference.llm.endpoint_url": "http://elsewhere:1/v1",
                "inference.llm.model": "m",
            }
        )
    )
    assert sup.stop_calls == 1
    assert runtime._state == "absent"
    assert runtime._resolved_id is None


async def test_rebuild_entering_local_resets_state_and_id(
    router_server: tuple[FakeRouter, str],
) -> None:
    _router, url = router_server
    runtime, _sup = make_runtime(
        url,
        make_snapshot(
            **{
                "inference.llm.source": "remote",
                "inference.llm.endpoint_url": "http://remote:1234/v1",
                "inference.llm.model": "m",
            }
        ),
    )
    assert runtime._state == "absent"
    await runtime.rebuild(local_snapshot())
    assert runtime._state == "unloaded"
    assert runtime._resolved_id is None


async def test_rebuild_force_restart_restarts_supervisor_when_running(
    router_server: tuple[FakeRouter, str],
) -> None:
    _router, url = router_server
    runtime, sup = make_runtime(url, local_snapshot())
    await runtime.ensure_ready()
    assert runtime._resolved_id == QWEN_ID
    assert sup.is_running()
    await runtime.rebuild(local_snapshot(), force_restart=True)
    assert runtime._resolved_id is None
    assert sup.restart_calls == 1


async def test_rescan_when_model_file_present_on_disk_but_not_in_router(
    router_server: tuple[FakeRouter, str], tmp_path: Path
) -> None:
    router, url = router_server
    router._statuses = {QWEN_ID: "unloaded"}
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / f"{LFM_ID}.gguf").touch()

    runtime, sup = make_runtime(url, local_snapshot(model=f"{LFM_ID}.gguf"), models_dir=models_dir)

    async def _on_restart() -> None:
        router._statuses[LFM_ID] = "unloaded"

    orig_restart = sup.restart

    async def restart_and_update() -> None:
        await orig_restart()
        await _on_restart()

    sup.restart = restart_and_update  # type: ignore[assignment]

    await runtime.ensure_ready()
    assert sup.restart_calls == 1
    assert runtime._resolved_id == LFM_ID
    assert router.load_calls == [LFM_ID]


# ── remote source ────────────────────────────────────────────────────────────


async def test_remote_ensure_ready_is_noop_and_target_has_endpoint(
    router_server: tuple[FakeRouter, str],
) -> None:
    router, url = router_server
    runtime, sup = make_runtime(
        url,
        make_snapshot(
            **{
                "inference.llm.source": "remote",
                "inference.llm.endpoint_url": "http://remote.example:1234/v1",
                "inference.llm.api_key": "sk-x",
                "inference.llm.model": "remote-model",
            }
        ),
    )
    await runtime.ensure_ready()  # no-op: no supervisor start, no HTTP
    assert sup.ensure_running_calls == 0
    assert router.load_calls == []
    target = runtime.target()
    assert target.source == "remote"
    assert target.base_url == "http://remote.example:1234/v1"
    assert target.api_key == "sk-x"
    assert target.model_id == "remote-model"
    status = await runtime.status()
    assert status.source == "remote"
    assert status.configured
    assert status.state == "loaded"


async def test_local_target_prefers_resolved_id(router_server: tuple[FakeRouter, str]) -> None:
    _router, url = router_server
    runtime, _sup = make_runtime(url, local_snapshot())
    assert runtime.target().model_id == f"{QWEN_ID}.gguf"  # unresolved → configured
    await runtime.ensure_ready()
    assert runtime.target().model_id == QWEN_ID  # resolved → router id


# ── explicit unload ──────────────────────────────────────────────────────────


async def test_explicit_unload(router_server: tuple[FakeRouter, str]) -> None:
    router, url = router_server
    runtime, _sup = make_runtime(url, local_snapshot())
    await runtime.ensure_ready()
    await runtime.unload()
    assert router.unload_calls == [QWEN_ID]
    assert runtime._state == "unloaded"


async def test_explicit_unload_failure_does_not_latch_error_state(
    router_server: tuple[FakeRouter, str],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Explicit unload router failure logs a warning and marks unloaded without latching error."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / f"{QWEN_ID}.gguf").touch()
    router, url = router_server
    runtime, _sup = make_runtime(url, local_snapshot(), models_dir=models_dir)
    await runtime.ensure_ready()
    status = await runtime.status()
    assert status.state == "loaded"
    assert status.error is None
    assert runtime._error is None

    router.fail_unload = True
    with caplog.at_level("WARNING", logger="vesta.inference.runtime"):
        await runtime.unload()

    assert any("llm.unload_failed" in r.getMessage() for r in caplog.records)
    assert runtime._error is None
    assert runtime._state == "unloaded"
    status = await runtime.status()
    assert status.state == "unloaded"
    assert status.error is None


# ── models.py helpers (D11 + RAM estimate) ───────────────────────────────────


class TestModelHelpers:
    def test_presets(self) -> None:
        qwen = preset_by_id("qwen3.5-4b-q4_k_s")
        assert qwen is not None
        assert qwen.filename == f"{QWEN_ID}.gguf"
        assert qwen.size_bytes == 2_590_430_368
        assert qwen.thinking == "toggle"
        assert qwen.url.startswith("https://huggingface.co/unsloth/")
        assert preset_by_id("lfm2.5-1.2b-instruct-q4_k_m") is None
        assert preset_by_id("lfm2.5-2.6b-q4_k_m") is None

    def test_thinking_heuristic(self) -> None:
        assert thinking_for_filename(f"{QWEN_ID}.gguf") == "toggle"
        assert thinking_for_filename(f"{LFM_ID}.gguf") == "never"
        assert thinking_for_filename("LFM2.5-2.6B-Q4_K_M.gguf") == "always"
        assert thinking_for_filename("Some-7B-Thinking-Q4.gguf") == "always"
        assert thinking_for_filename("Mistral-7B-Q4_K_M.gguf") == "toggle"

    def test_estimate_ram_bytes(self) -> None:
        # Calibrated against real-binary measurements (see models.py docstring).
        assert estimate_ram_bytes(730_895_168, 32768, 7168) == 965_776_192
        assert estimate_ram_bytes(0, 32768, 32768) == 1024 * 1024 * 1024

    def test_preset_by_filename(self) -> None:
        preset = preset_by_filename(f"{QWEN_ID}.gguf")
        assert preset is not None
        assert preset.id == "qwen3.5-4b-q4_k_s"
        assert preset_by_filename("Unknown-Model.gguf") is None
