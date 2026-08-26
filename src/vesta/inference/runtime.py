"""The LLM runtime — the single owner of "how do I talk to the LLM right now".

``LlmRuntime`` resolves the model id the router actually answers to (never
assume; the router exposes bare ``.gguf`` files under their filename *stem*),
loads and unloads on demand, watches idleness with an app-side watchdog (belt
and braces: the CLI flag plus this task, because only the explicit
``/models/unload`` frees memory all the way), and exposes ``LlmStatus`` for the
UI.

Wire shapes are measured ones:
``POST /models/load`` returns ``{"success": true}`` *immediately* (loading is
async — you must poll ``GET /models`` until the nested ``status.value`` is
``"loaded"``); ``GET /models`` ids are filename stems without ``.gguf``.

``inference/`` depends ONLY on ``config`` plus package siblings (runtime.py
adds ``httpx`` for the router control plane).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import httpx

from vesta.config.settings import SettingsSnapshot
from vesta.inference import (
    INFERENCE_LLM_API_KEY,
    INFERENCE_LLM_ENABLE_THINKING,
    INFERENCE_LLM_ENDPOINT_URL,
    INFERENCE_LLM_MODEL,
    INFERENCE_LLM_SOURCE,
    INFERENCE_LOCAL_CONTEXT_SIZE,
    INFERENCE_LOCAL_IDLE_UNLOAD_SECONDS,
    INFERENCE_LOCAL_STOP_SERVER_AFTER_IDLE_SECONDS,
)
from vesta.inference.models import (
    display_name_for,
    estimate_ram_bytes,
    kv_bytes_per_token_for,
    thinking_for_filename,
)

if TYPE_CHECKING:
    from vesta.inference.local import LlamaServerSupervisor

_log = logging.getLogger(__name__)

#: Timeout for cheap router probes (``GET /models``, ``/models/unload``).
_MODELS_TIMEOUT_S = 5.0
#: Budget for the whole load flow — the POST (which returns immediately) plus
#: the polling loop. A cold ~2.5 GB load off a slow disk is minutes.
_LOAD_TIMEOUT_S = 300.0
#: Poll interval while waiting for ``status.value == "loaded"``.
_LOAD_POLL_S = 0.25
#: Idle-watchdog tick interval (D4). Overridable in the constructor so tests
#: can run with second-scale thresholds.
_WATCHDOG_INTERVAL_S = 30.0

#: Status values llama-server reports: ``sleeping`` is the state
#: after the router's own ``--sleep-idle-seconds`` fires — distinct from
#: ``unloaded`` and still holding ~200 MB resident.
_ROUTER_STATES = frozenset({"unloaded", "loading", "loaded", "sleeping"})


class LlmRuntimeError(RuntimeError):
    """The runtime cannot reach a ready local model (D2 no-match, load failure)."""


@dataclass(frozen=True)
class LlmTarget:
    """Everything a chat call needs, resolved for *right now* (D1)."""

    source: str  # "local" | "remote"
    base_url: str  # supervisor's OpenAI base URL (local) or the endpoint setting
    api_key: str
    model_id: str  # the id llama-server actually answers to (D2)
    enable_thinking: bool | None  # None ⇒ send no chat_template_kwargs
    #: Accelerator class of the serving host ("cpu"|"gpu"); ``None`` for the
    #: remote source or before the local probe resolved (hardware contract).
    hardware: str | None = None


@dataclass(frozen=True)
class LlmStatus:
    """A pollable snapshot of the LLM lifecycle for the UI (D9)."""

    source: str
    configured: bool  # a model is selected
    installed: bool  # local: the GGUF exists on disk
    state: str  # absent|stopped|unloaded|loading|loaded|sleeping|error
    model_file: str | None
    display_name: str | None
    model_id: str | None  # the resolved router id, once known (D2)
    size_bytes: int
    context_size: int
    thinking: bool
    thinking_supported: bool  # False ⇒ the switch is inert for this model (D11)
    idle_unload_seconds: int  # 0 = never
    seconds_since_last_use: float | None
    estimated_ram_bytes: int
    error: str | None
    hardware: str | None = None  # "cpu"|"gpu" local; None remote/unknown


def _match_model_id(configured: str, router_ids: list[str]) -> str | None:
    """Resolve the configured value against the ids ``GET /models`` reported (D2).

    In priority order: exact equality, basename equality, stem equality (the one
    that matches bare ``.gguf`` files — the router exposes filename *stems*),
    then a single-entry fallback. No match ⇒ ``None`` (caller sets an error
    naming both sides, so a mismatch is diagnosable from the UI).
    """
    if not router_ids:
        return None
    if configured in router_ids:
        return configured
    name = Path(configured).name
    if name in router_ids:
        return name
    stem = Path(name).stem
    for rid in router_ids:
        if rid.removesuffix(".gguf") == stem:
            return rid
    if len(router_ids) == 1:
        return router_ids[0]
    return None


class LlmRuntime:
    """One resolver, one lifecycle owner for the chat LLM (D1).

    All router HTTP goes through one lazily-created ``httpx.AsyncClient`` owned
    by the runtime (short timeout for ``/models``, long budget for load).
    ``ensure_ready`` is idempotent and concurrency-safe (one lock, one load).
    """

    def __init__(
        self,
        *,
        supervisor: LlamaServerSupervisor | None,
        snapshot: SettingsSnapshot,
        models_dir: Path | None = None,
        watchdog_interval_s: float = _WATCHDOG_INTERVAL_S,
    ) -> None:
        self._supervisor = supervisor
        self._snapshot = snapshot
        self._models_dir = models_dir
        self._watchdog_interval_s = watchdog_interval_s
        self._router_url = supervisor.router_url if supervisor is not None else ""
        self._http: httpx.AsyncClient | None = None
        self._load_lock = asyncio.Lock()
        self._watchdog_task: asyncio.Task[None] | None = None
        #: Cached router id (dropped by :meth:`rebuild` and on supervisor stop).
        self._resolved_id: str | None = None
        #: Last-known lifecycle state — a cache; ``status()`` prefers the live
        #: router value when reachable.
        self._state: str = "absent"
        self._error: str | None = None
        #: Stamped by ``mark_used`` (ensure_ready + every completed turn) —
        #: NEVER by status polling, or the UI's poll loop would keep the model
        #: alive forever (D4, upstream issue #23096's failure mode).
        self._last_used: float | None = None

    # ── Resolution ───────────────────────────────────────────────────────────

    def _source(self) -> str:
        return str(self._snapshot.get(INFERENCE_LLM_SOURCE))

    def _model_file(self) -> str:
        return str(self._snapshot.get(INFERENCE_LLM_MODEL))

    def target(self) -> LlmTarget:
        """The resolved connection info for a chat call, right now (D1).

        For local, ``model_id`` is the cached router id once resolved; before
        that it is the configured value as-is (call :meth:`ensure_ready` first
        on any path that chats). ``enable_thinking`` is ``None`` — send nothing
        — when the model's thinking mode makes the switch inert (``always`` /
        ``never``, D11).
        """
        source = self._source()
        api_key = str(self._snapshot.get(INFERENCE_LLM_API_KEY))
        model = self._model_file()
        if source == "remote":
            return LlmTarget(
                source="remote",
                base_url=str(self._snapshot.get(INFERENCE_LLM_ENDPOINT_URL)),
                api_key=api_key,
                model_id=model,
                enable_thinking=bool(self._snapshot.get(INFERENCE_LLM_ENABLE_THINKING)),
            )
        enable = (
            None
            if thinking_for_filename(model) != "toggle"
            else bool(self._snapshot.get(INFERENCE_LLM_ENABLE_THINKING))
        )
        return LlmTarget(
            source="local",
            base_url=self._supervisor.base_url if self._supervisor is not None else "",
            api_key=api_key or "local",
            model_id=self._resolved_id or model,
            enable_thinking=enable,
            hardware=self._supervisor.hardware if self._supervisor is not None else None,
        )

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def ensure_ready(self, *, on_status: Callable[[str], None] | None = None) -> None:
        """Bring the LLM to "ready to chat". No-op for remote (D1).

        Local: start the supervisor if needed → resolve the router id (D2) →
        ``POST /models/load`` (async!) → poll until ``loaded`` → ``mark_used``.
        ``on_status`` receives short human strings the caller turns into SSE
        ``status`` events. Raises :class:`LlmRuntimeError` (with the state set
        to ``error``) when no model is configured or the router registry has no
        match.
        """
        if self._source() == "remote":
            return
        async with self._load_lock:
            await self._ensure_ready_local(on_status)

    async def _ensure_ready_local(self, on_status: Callable[[str], None] | None) -> None:
        model = self._model_file()
        if not model:
            self._fail("no model configured (inference.llm.model is empty)")
        if self._supervisor is None:
            self._fail("local source but no llama-server supervisor bound")
        if not self._supervisor.is_running() and on_status is not None:
            on_status("Starting the local model runtime…")
        await self._supervisor.ensure_running()

        states = await self._fetch_router_states()
        if self._resolved_id is not None and self._resolved_id not in states:
            self._resolved_id = None

        if self._resolved_id is None:
            if self._file_path(model) is not None and Path(model).stem not in states:
                _log.info("llm.rescan_models_dir", extra={"model": model})
                await self._supervisor.restart()
                states = await self._fetch_router_states()
            rid = _match_model_id(model, list(states))
            if rid is None:
                self._fail(
                    f"model {model!r} not found in the llama-server registry; "
                    f"router reports: {sorted(states)}"
                )
            self._resolved_id = rid

        rid = self._resolved_id
        if states.get(rid) == "loaded":
            self._state = "loaded"
            self._error = None
            self.mark_used()
            return

        if on_status is not None:
            on_status(f"Loading {display_name_for(model)} into memory…")
        self._state = "loading"
        # The POST returns {"success": true} immediately — the
        # long timeout only guards a hung call, not the load itself.
        try:
            await self._post_router("/models/load", {"model": rid}, timeout=_LOAD_TIMEOUT_S)
        except Exception as exc:
            self._fail(f"llama-server /models/load failed: {exc!r}")
        await self._poll_until_loaded(rid)
        self._state = "loaded"
        self._error = None
        self.mark_used()

    async def _poll_until_loaded(self, rid: str) -> None:
        deadline = time.monotonic() + _LOAD_TIMEOUT_S
        while True:
            state = await self._router_state_of(rid)
            if state == "loaded":
                return
            if time.monotonic() >= deadline:
                self._fail(f"model {rid!r} did not finish loading within {_LOAD_TIMEOUT_S}s")
            await asyncio.sleep(_LOAD_POLL_S)

    async def load(self) -> None:
        """Explicit load — the models API's ``POST /api/models/load``."""
        await self.ensure_ready()

    async def unload(self) -> None:
        """Explicit unload: frees the weights all the way (~17 MB residual)."""
        if self._source() == "remote":
            return
        if self._supervisor is not None and self._resolved_id is not None:
            try:
                await self._post_router("/models/unload", {"model": self._resolved_id})
            except Exception as exc:
                _log.warning("llm.unload_failed", extra={"error": str(exc)})
        self._state = "unloaded"

    @property
    def supervisor(self) -> LlamaServerSupervisor | None:
        """The supervised child's owner (``None`` for the remote source)."""
        return self._supervisor

    @property
    def snapshot(self) -> SettingsSnapshot:
        """The settings the runtime currently resolves against."""
        return self._snapshot

    @property
    def watchdog_running(self) -> bool:
        """Whether the idle-watchdog task is alive (D4)."""
        return self._watchdog_task is not None and not self._watchdog_task.done()

    async def retire(self) -> None:
        """Full teardown for a settings rebind (D7 row 3): stop the watchdog,
        close the router client, and stop the supervised child so a fresh
        runtime (with a freshly-baked command line) can own the port."""
        await self.stop()
        if self._supervisor is not None:
            with contextlib.suppress(Exception):
                await self._supervisor.stop()

    def mark_used(self) -> None:
        """Stamp ``last_used`` — called from ``ensure_ready`` and after every
        completed turn, never from status polling (D4)."""
        self._last_used = time.monotonic()

    async def rebuild(self, snapshot: SettingsSnapshot, *, force_restart: bool = False) -> None:
        """Settings changed — take the cheapest correct action (D7).

        A model change drops the cached router id and best-effort unloads the
        old model; if the child is running it is restarted so the router
        discovers the new file (a running router does NOT see
        files dropped into ``--models-dir``). Leaving ``local`` stops the
        supervised child. Threshold changes need nothing — the watchdog reads
        the snapshot fresh on every tick.
        """
        old_model = self._model_file()
        old_source = self._source()
        self._snapshot = snapshot
        self._error = None
        new_model = self._model_file()
        new_source = self._source()

        if force_restart or old_model != new_model:
            rid = self._resolved_id
            self._resolved_id = None
            self._state = "unloaded"
            if self._supervisor is not None and self._supervisor.is_running():
                if rid is not None and old_model:
                    try:
                        await self._post_router("/models/unload", {"model": rid})
                    except Exception as exc:
                        _log.warning("llm.unload_failed", extra={"error": str(exc)})
                await self._supervisor.restart()

        if old_source != new_source and new_source == "remote":
            if self._supervisor is not None:
                with contextlib.suppress(Exception):
                    await self._supervisor.stop()
            self._state = "absent"

    # ── Status ───────────────────────────────────────────────────────────────

    async def status(self) -> LlmStatus:
        """A pollable snapshot (D9). Never stamps ``last_used`` (D4)."""
        now = time.monotonic()
        seconds = None if self._last_used is None else now - self._last_used
        source = self._source()
        model = self._model_file()
        hardware = await self._hardware()
        if source == "remote":
            configured = bool(model)
            return LlmStatus(
                source="remote",
                configured=configured,
                installed=configured,
                state="loaded" if configured else "absent",
                model_file=model or None,
                display_name=model or None,
                model_id=model or None,
                size_bytes=0,
                context_size=0,
                thinking=bool(self._snapshot.get(INFERENCE_LLM_ENABLE_THINKING)),
                thinking_supported=True,
                idle_unload_seconds=0,
                seconds_since_last_use=seconds,
                estimated_ram_bytes=0,
                error=self._error,
                hardware=None,
            )

        file_path = self._file_path(model)
        installed = file_path is not None
        size = file_path.stat().st_size if file_path is not None else 0
        mode = thinking_for_filename(model) if model else "toggle"
        thinking = {"never": False, "always": True}.get(
            mode, bool(self._snapshot.get(INFERENCE_LLM_ENABLE_THINKING))
        )
        context_size = int(self._snapshot.get(INFERENCE_LOCAL_CONTEXT_SIZE))
        return LlmStatus(
            source="local",
            configured=bool(model),
            installed=installed,
            state=self._live_state(installed),
            model_file=model or None,
            display_name=display_name_for(model) if model else None,
            model_id=self._resolved_id,
            size_bytes=size,
            context_size=context_size,
            thinking=thinking,
            thinking_supported=mode == "toggle",
            idle_unload_seconds=int(self._snapshot.get(INFERENCE_LOCAL_IDLE_UNLOAD_SECONDS)),
            seconds_since_last_use=seconds,
            estimated_ram_bytes=(
                estimate_ram_bytes(size, context_size, kv_bytes_per_token_for(model))
                if installed
                else 0
            ),
            error=self._error,
            hardware=hardware,
        )

    async def _hardware(self) -> str | None:
        """The serving host's accelerator class (``None`` remote/unknown).

        ``hw_class`` probes once and caches, so repeated status polls never
        respawn ``--list-devices``; any failure degrades to ``None`` — hardware
        detection must never take the status surface down."""
        if self._supervisor is None:
            return None
        with contextlib.suppress(Exception):
            return await self._supervisor.hw_class()
        return None

    def _live_state(self, installed: bool) -> str:
        """Best-known local state: error/absent short-circuit, then cached router
        state. An observed error outranks a missing file — the mismatch is what
        the UI needs to explain (D2)."""
        if self._error is not None:
            return "error"
        if not installed:
            return "absent"
        if self._supervisor is None or not self._supervisor.is_running():
            return "stopped"
        return self._state if self._state in _ROUTER_STATES else "unloaded"

    # ── Idle watchdog (D4) ───────────────────────────────────────────────────

    def start(self) -> None:
        """Start the idle-watchdog task (idempotent)."""
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(self._watchdog(), name="llm-idle-watchdog")

    async def stop(self) -> None:
        """Stop the watchdog and close the router client. Safe when never started."""
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watchdog_task
            self._watchdog_task = None
        if self._http is not None:
            with contextlib.suppress(Exception):
                await self._http.aclose()
            self._http = None

    async def _watchdog(self) -> None:
        while True:
            await asyncio.sleep(self._watchdog_interval_s)
            with contextlib.suppress(Exception):
                await self._tick()

    async def _tick(self) -> None:
        if self._last_used is None or self._source() == "remote":
            return
        # Never idle-collect while a load is in flight (D4's "next question
        # transparently respawns it"): ``ensure_ready`` holds ``_load_lock``
        # from spawn to ``mark_used``, and ``last_used`` is stale until then —
        # a tick in that window would SIGTERM the freshly spawned child with
        # the *pre-respawn* idle age.
        if self._load_lock.locked() or self._state == "loading":
            return
        idle = time.monotonic() - self._last_used
        unload_after = int(self._snapshot.get(INFERENCE_LOCAL_IDLE_UNLOAD_SECONDS))
        stop_after = int(self._snapshot.get(INFERENCE_LOCAL_STOP_SERVER_AFTER_IDLE_SECONDS))
        if unload_after > 0 and idle > unload_after and self._state == "loaded":
            if self._resolved_id is not None and self._supervisor is not None:
                try:
                    await self._post_router("/models/unload", {"model": self._resolved_id})
                    self._state = "unloaded"
                    _log.info("llm.idle_unloaded", extra={"idle_s": round(idle, 1)})
                except Exception as exc:
                    _log.warning(
                        "llm.idle_unload_failed",
                        extra={"error": str(exc), "idle_s": round(idle, 1)},
                    )
            else:
                self._state = "unloaded"
        if (
            stop_after > 0
            and idle > stop_after
            and self._supervisor is not None
            and self._supervisor.is_running()
        ):
            await self._supervisor.stop()
            self._state = "stopped"
            _log.info("llm.idle_stopped_server", extra={"idle_s": round(idle, 1)})

    # ── Router HTTP ──────────────────────────────────────────────────────────

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(base_url=self._router_url, timeout=_MODELS_TIMEOUT_S)
        return self._http

    def _file_path(self, model: str) -> Path | None:
        if not model or self._models_dir is None:
            return None
        path = self._models_dir / Path(model).name
        return path if path.is_file() else None

    def _fail(self, message: str) -> NoReturn:
        self._state = "error"
        self._error = message
        raise LlmRuntimeError(message)

    async def _fetch_router_states(self) -> dict[str, str]:
        """``GET /models`` → ``{id: status.value}``. Raises on any HTTP failure."""
        try:
            resp = await self._client().get("/models")
            resp.raise_for_status()
            data = resp.json()["data"]
            return {str(entry["id"]): str(entry["status"]["value"]) for entry in data}
        except Exception as exc:
            self._fail(f"cannot reach llama-server at {self._router_url!r}: {exc!r}")

    async def _router_state_of(self, rid: str) -> str:
        states = await self._fetch_router_states()
        if rid not in states:
            self._fail(f"model {rid!r} disappeared from the llama-server registry")
        return states[rid]

    async def _post_router(
        self, path: str, payload: dict[str, str], *, timeout: float = _MODELS_TIMEOUT_S
    ) -> httpx.Response:
        resp = await self._client().post(path, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp


__all__ = [
    "LlmRuntime",
    "LlmRuntimeError",
    "LlmStatus",
    "LlmTarget",
]
