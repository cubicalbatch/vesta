"""Inference gateway + local LLM runtime lifecycle.

One ``AsyncOpenAI`` client for local **and** remote.
Nothing upstream branches on backend; callers use ``gateway.chat_stream(...)`` and
that is the entire surface. Three independent roles (``llm``, ``embed``,
``rerank``), but only ``llm`` is exercised through this gateway —
embed/rerank stay in-process via ``encoders/``.

The :class:`~vesta.inference.runtime.LlmRuntime` is the single
owner of "how do I talk to the LLM right now": it resolves the router id,
loads/unloads on demand, and runs the idle watchdog. The composition root
binds it via :func:`bind_runtime` and the gateway via :func:`bind_gateway`.

Capabilities: on import this package registers a probe that
turns on ``Capability.LLM`` when a chat model is *usable* (remote:
endpoint + model set; local: the ``llama-server`` binary exists AND the
configured GGUF is on disk). The probe re-evaluates on every capability
computation (a cheap config/filesystem check, never a network round-trip —
mirrors ``encoders/__init__``'s pattern).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vesta.config.capabilities import Capability, CapabilitySet, register_probe
from vesta.config.settings import setting

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Collection

    from vesta.config.settings import SettingsSnapshot
    from vesta.inference.gateway import Gateway
    from vesta.inference.local import LlamaServerSupervisor
    from vesta.inference.runtime import LlmRuntime

# ── Settings ─────────────────────────────────────────────────────────────

INFERENCE_LLM_SOURCE = setting(
    "inference.llm.source",
    str,
    "local",
    group="Inference / LLM",
    help="Where the chat model runs: 'local' (supervised llama-server) or "
    "'remote' (an OpenAI-compatible endpoint). Nothing upstream branches on "
    "this — the gateway is identical either way.",
    choices=("local", "remote"),
    hot=False,
)
INFERENCE_LLM_ENDPOINT_URL = setting(
    "inference.llm.endpoint_url",
    str,
    "",
    group="Inference / LLM",
    help="OpenAI-compatible base URL for the remote source (e.g. "
    "'http://host:1234/v1'). Ignored when source is 'local'.",
    hot=True,
)
INFERENCE_LLM_API_KEY = setting(
    "inference.llm.api_key",
    str,
    "",
    group="Inference / LLM",
    help="API key for the remote endpoint. Empty for local or unauthenticated "
    "endpoints. Leave blank or unchanged when saving to keep the stored key.",
    hot=True,
    secret=True,
)
INFERENCE_LLM_MODEL = setting(
    "inference.llm.model",
    str,
    "",
    group="Inference / LLM",
    help="The chat model id. For local: a GGUF filename or registry key from "
    "data/models/. For remote: the endpoint's model id (e.g. "
    "'unsloth/qwen3.5-4b').",
    hot=True,
)
INFERENCE_LLM_ENABLE_THINKING = setting(
    "inference.llm.enable_thinking",
    bool,
    False,
    group="Inference / LLM",
    help="Send chat_template_kwargs.enable_thinking to the chat endpoint. "
    "Qwen3/Qwen3.5 reasoning models think by default and burn the whole "
    "max_tokens budget on hidden reasoning_content; False "
    "gives direct, fast answers (the intended instruct-class behaviour). True "
    "re-enables deep reasoning — pair it with a larger answer.max_output_tokens. "
    "Instruct-only models and endpoints that ignore the parameter are unaffected.",
    hot=True,
)

INFERENCE_LOCAL_BINARY_PATH = setting(
    "inference.local.binary_path",
    str,
    "llama-server",
    group="Inference / Local runtime",
    help="Path to the bundled llama-server binary. Default 'llama-server' "
    "resolves via PATH; the container symlinks /opt/llama.cpp/llama-server onto "
    "PATH so no configuration is needed.",
    hot=False,
)
INFERENCE_LOCAL_THREADS_GEN = setting(
    "inference.local.threads_gen",
    int,
    6,
    group="Inference / Local runtime",
    help="llama-server generation threads (-t). Never exceed 8 physical cores; "
    "SMT hurts. 6 saturates DDR5 bandwidth.",
    min=1,
    max=8,
    hot=False,
)
INFERENCE_LOCAL_THREADS_PREFILL = setting(
    "inference.local.threads_prefill",
    int,
    8,
    group="Inference / Local runtime",
    help="llama-server prompt-processing threads (-tb). Compute-bound, scales with physical cores.",
    min=1,
    max=8,
    hot=False,
)
INFERENCE_LOCAL_IDLE_UNLOAD_SECONDS = setting(
    "inference.local.idle_unload_seconds",
    int,
    900,
    group="Inference / Local runtime",
    help="Free the loaded model's weights after this idle period "
    "(--sleep-idle-seconds on the child, plus an app-side watchdog that unloads "
    "via /models/unload — only the explicit unload frees memory all the way). "
    "0 = never.",
    min=0,
    max=86400,
    hot=False,
)
INFERENCE_LOCAL_STOP_SERVER_AFTER_IDLE_SECONDS = setting(
    "inference.local.stop_server_after_idle_seconds",
    int,
    3600,
    group="Inference / Local runtime",
    help="SIGTERM the llama-server child after this longer idle period so RSS "
    "returns to ~0 rather than 'router process, no weights'. The "
    "next question transparently respawns it. 0 = never.",
    min=0,
    max=86400,
    hot=False,
)
INFERENCE_LOCAL_CONTEXT_SIZE = setting(
    "inference.local.context_size",
    int,
    8192,
    group="Inference / Local runtime",
    help="Context window (llama-server -c, written to models.ini). The input+"
    "output window; answer.max_output_tokens stays the separate output cap.",
    min=2048,
    max=131072,
    hot=False,
)
INFERENCE_LOCAL_PRELOAD_ON_READY = setting(
    "inference.local.preload_on_ready",
    bool,
    True,
    group="Inference / Local runtime",
    help="After a GGUF download finishes, load it immediately so the user's "
    "first question is fast.",
    hot=True,
)
INFERENCE_LOCAL_MODELS_MAX = setting(
    "inference.local.models_max",
    int,
    1,
    group="Inference / Local runtime",
    help="llama-server --models-max: max resident models with LRU eviction. Single user → 1.",
    min=1,
    max=8,
    hot=False,
)

#: Concurrency slot for the ``download_model`` job (mirrors the ZIM path's
#: ``jobs.max_concurrent.download_zim``). Single download at a time is plenty
#: for a one-off wizard download.
JOBS_MAX_CONCURRENT_DOWNLOAD_MODEL = setting(
    "jobs.max_concurrent.download_model",
    int,
    1,
    group="Jobs",
    help="Max concurrent GGUF model downloads.",
    min=1,
    max=4,
    hot=False,
)

#: The models directory, bound by ``main``'s lifespan. Same value as
#: ``encoders.model_dir`` (``data/models/``) — GGUFs and ONNX models share the
#: directory. Bound separately so ``inference/download.py`` doesn't need to
#: import ``encoders`` or read ``data.dir`` directly.
_MODELS_DIR: Path | None = None


def bind_models_dir(models_dir: Path | None) -> None:
    """Bind (or detach, with ``None``) the models directory."""
    global _MODELS_DIR
    _MODELS_DIR = models_dir


def get_models_dir() -> Path | None:
    """The models directory, or ``None`` if not bound (outside the lifespan)."""
    if _MODELS_DIR is None:
        return None
    return Path(_MODELS_DIR)


#: The live gateway, bound by the composition root (``main`` lifespan). A
#: module-level singleton the capability probe reads — matching ``encoders``'s
#: ``bind_manager`` precedent (a configured singleton, not a per-call import).
_GATEWAY: Gateway | None = None
_SUPERVISOR: LlamaServerSupervisor | None = None


def _config_value(descriptor: Any, default: object) -> Any:
    """A config read that never raises (the probe must stay cheap and total)."""
    from vesta import config

    try:
        return config.get(descriptor)
    except Exception:
        return default


def _probe_remote() -> bool:
    """Remote is usable iff the endpoint URL and model are both non-empty."""
    return bool(
        str(_config_value(INFERENCE_LLM_ENDPOINT_URL, ""))
        and str(_config_value(INFERENCE_LLM_MODEL, ""))
    )


def _probe_local() -> bool:
    """Local is usable iff the binary exists AND the GGUF is on disk.

    Degrade-don't-fail: a missing binary or GGUF is no-LLM;
    ``sources_only`` stays usable.
    """
    model = str(_config_value(INFERENCE_LLM_MODEL, ""))
    if not model:
        return False
    if _SUPERVISOR is not None:
        binary_present = _SUPERVISOR.binary_available()
    else:
        binary_path = str(_config_value(INFERENCE_LOCAL_BINARY_PATH, ""))
        binary_present = Path(binary_path).is_file() or shutil.which(binary_path) is not None
    if not binary_present:
        return False
    models_dir = get_models_dir()
    return models_dir is not None and (models_dir / Path(model).name).is_file()


def _capability_probe() -> CapabilitySet:
    """``Capability.LLM`` is on iff a chat model is *usable*.

    For **remote**: the endpoint URL and model are set. For **local**: the
    ``llama-server`` binary exists **and** the configured GGUF exists under the
    models dir. Both are cheap config/filesystem checks — never a network
    round-trip, never a process spawn (the probe runs on every ``/health`` and
    every chat request). Reachability is tested by the actual chat call, which
    degrades gracefully on failure.
    """
    if _GATEWAY is None:
        return frozenset()
    source = str(_config_value(INFERENCE_LLM_SOURCE, INFERENCE_LLM_SOURCE.default))
    usable = _probe_remote() if source == "remote" else _probe_local()
    return frozenset({Capability.LLM}) if usable else frozenset()


register_probe(_capability_probe)


def bind_gateway(gateway: Gateway | None, supervisor: LlamaServerSupervisor | None = None) -> None:
    """Attach (or detach, with ``None``) the live gateway + supervisor."""
    global _GATEWAY, _SUPERVISOR
    _GATEWAY = gateway
    _SUPERVISOR = supervisor


def get_gateway() -> Gateway | None:
    """The live gateway, or ``None`` if the composition root hasn't bound one."""
    return _GATEWAY


def get_supervisor() -> LlamaServerSupervisor | None:
    """The live supervisor (local source only), or ``None``."""
    return _SUPERVISOR


def build_gateway_from_settings(
    snapshot: SettingsSnapshot,
    *,
    data_dir: Path,
) -> tuple[Gateway, LlamaServerSupervisor | None]:
    """Construct the live :class:`~vesta.inference.gateway.Gateway` from settings.

    Returns ``(gateway, supervisor)``. For the **local** source the supervisor is
    constructed (lazy-started on first chat call); for **remote** it is ``None``
    and the gateway points straight at the user's endpoint.

    Kept in ``inference/`` (not ``main.py``) so a new ``inference.*`` setting
    can't be added without updating the one place that constructs the gateway —
    the same pattern ``encoders/__init__`` uses for ``build_manager_from_settings``.
    """
    from vesta.inference.gateway import OpenAIGateway
    from vesta.inference.local import build_supervisor_from_settings

    source = str(snapshot.get(INFERENCE_LLM_SOURCE))
    api_key = str(snapshot.get(INFERENCE_LLM_API_KEY))

    if source == "remote":
        endpoint = str(snapshot.get(INFERENCE_LLM_ENDPOINT_URL))
        gateway: OpenAIGateway = OpenAIGateway(
            base_url=endpoint,
            api_key=api_key,
            supervisor=None,
        )
        return gateway, None

    # local
    supervisor = build_supervisor_from_settings(snapshot, data_dir=data_dir)
    gateway = OpenAIGateway(
        base_url=supervisor.base_url,
        api_key=api_key or "local",
        supervisor=supervisor,
    )
    return gateway, supervisor


#: The live LLM runtime, bound by the composition root. Same
#: configured-singleton pattern as ``_GATEWAY`` above.
_RUNTIME: LlmRuntime | None = None


def bind_runtime(runtime: LlmRuntime | None) -> None:
    """Attach (or detach, with ``None``) the live :class:`LlmRuntime`."""
    global _RUNTIME
    _RUNTIME = runtime


def get_runtime() -> LlmRuntime | None:
    """The live LLM runtime, or ``None`` if the composition root hasn't bound one."""
    return _RUNTIME


#: Settings baked into the supervisor's command line / ``models.ini`` at
#: construction. A change leaves any running child stale, so the runtime is
#: rebound fresh (a new supervisor with the new command; the old child is
#: stopped) instead of merely ``rebuild()``-ing — ``LlmRuntime.rebuild``
#: deliberately does not restart the child for these.
_CHILD_RESTART_KEYS = frozenset(
    {
        INFERENCE_LOCAL_BINARY_PATH.key,
        INFERENCE_LOCAL_CONTEXT_SIZE.key,
        INFERENCE_LOCAL_MODELS_MAX.key,
        INFERENCE_LOCAL_THREADS_GEN.key,
        INFERENCE_LOCAL_THREADS_PREFILL.key,
    }
)


def _needs_child_rebind(
    old_runtime: LlmRuntime, snapshot: SettingsSnapshot, changed: Collection[str] | None
) -> bool:
    """Whether the change set needs a fresh runtime (supervisor restart).

    Beyond the always-restart keys: ``idle_unload_seconds`` only needs the
    child restarted when the ``--sleep-idle-seconds`` *flag's presence*
    flipped across zero — a pure threshold change is picked up by
    the watchdog, which reads the snapshot fresh on every tick.
    """
    if changed is not None and _CHILD_RESTART_KEYS.intersection(changed):
        return True
    old_snapshot = getattr(old_runtime, "snapshot", None)
    if old_snapshot is None:  # a stub runtime has no snapshot — no rebind
        return False
    old_idle = int(old_snapshot.get(INFERENCE_LOCAL_IDLE_UNLOAD_SECONDS))
    new_idle = int(snapshot.get(INFERENCE_LOCAL_IDLE_UNLOAD_SECONDS))
    return (old_idle > 0) != (new_idle > 0)


async def _rebind_runtime(old_runtime: LlmRuntime, snapshot: SettingsSnapshot) -> None:
    """Build a fresh runtime + supervisor and swap every reference to it.

    The fresh runtime is built *before* the old one is retired, so a build
    failure leaves the old runtime serving. The gateway keeps its identity
    (fixed local port ⇒ same base URL) and only swaps its supervisor, so
    ``AppState.gateway`` stays valid.
    """
    models_dir = get_models_dir()
    if models_dir is None:
        # Outside the composition root — nothing to rebind onto.
        await old_runtime.rebuild(snapshot)
        return
    fresh = build_runtime_from_settings(snapshot, data_dir=models_dir.parent)
    was_running = old_runtime.watchdog_running
    await old_runtime.retire()
    gateway = get_gateway()
    if gateway is not None and fresh.supervisor is not None:
        from vesta.inference.gateway import OpenAIGateway

        if isinstance(gateway, OpenAIGateway):
            gateway.attach_supervisor(fresh.supervisor)
        bind_gateway(gateway, fresh.supervisor)
    bind_runtime(fresh)
    if was_running:
        fresh.start()
    logging.getLogger(__name__).info("inference.runtime_rebound")


async def rebuild_runtime(
    snapshot: SettingsSnapshot | None = None,
    changed: Collection[str] | None = None,
    *,
    force_restart: bool = False,
) -> bool:
    """Rebuild the bound LLM runtime after an ``inference.*`` change.

    Defaults to the live config snapshot. ``changed`` is the set of keys the
    caller wrote; when any of them is baked into the llama-server command
    line, the runtime is rebound fresh (child restart — see
    :data:`_CHILD_RESTART_KEYS`), otherwise the cheaper in-place
    ``LlmRuntime.rebuild`` runs. Returns whether a runtime was rebuilt.
    Failures are logged, never raised — a bad endpoint must always stay
    correctable from the UI, so the settings write is not rolled back.
    """
    runtime = get_runtime()
    if runtime is None:
        return False
    if snapshot is None:
        from vesta import config

        snapshot = config.snapshot()
    try:
        if _needs_child_rebind(runtime, snapshot, changed):
            await _rebind_runtime(runtime, snapshot)
        else:
            await runtime.rebuild(snapshot, force_restart=force_restart)
    except Exception as exc:
        logging.getLogger(__name__).warning("inference.rebuild_failed", extra={"error": repr(exc)})
    return True


def build_runtime_from_settings(
    snapshot: SettingsSnapshot,
    *,
    data_dir: Path,
    supervisor: LlamaServerSupervisor | None = None,
) -> LlmRuntime:
    """Construct the live :class:`~vesta.inference.runtime.LlmRuntime` from settings.

    For the **local** source the runtime owns a supervisor (lazy — nothing is
    spawned until the first ``ensure_ready``); for **remote** there is no
    supervisor and the runtime is a thin settings resolver. Kept in
    ``inference/`` for the same one-construction-site reason as
    :func:`build_gateway_from_settings`.

    ``supervisor`` lets the composition root share the gateway's supervisor —
    one child, one port (:data:`~vesta.inference.local.DEFAULT_PORT`); two
    supervisors would race for it. ``None`` (the default) builds a fresh one.
    """
    from vesta.inference.local import build_supervisor_from_settings
    from vesta.inference.runtime import LlmRuntime

    source = str(snapshot.get(INFERENCE_LLM_SOURCE))
    if supervisor is None and source != "remote":
        supervisor = build_supervisor_from_settings(snapshot, data_dir=data_dir)
    return LlmRuntime(
        supervisor=supervisor,
        snapshot=snapshot,
        models_dir=data_dir / "models",
    )


#: Post-download callback — the same injection pattern ``catalog/``
#: uses for its post-download registration, so ``inference/`` never imports
#: ``jobs/``: the download job calls :func:`notify_model_ready`, and ``main``
#: binds the actual behaviour (refresh + optional preload).
_ON_MODEL_READY: Callable[[Path], Awaitable[None]] | None = None


def bind_on_model_ready(cb: Callable[[Path], Awaitable[None]] | None) -> None:
    """Bind (or detach, with ``None``) the post-GGUF-download callback."""
    global _ON_MODEL_READY
    _ON_MODEL_READY = cb


async def notify_model_ready(path: Path) -> None:
    """Called by the download job after the atomic rename lands the GGUF."""
    if _ON_MODEL_READY is not None:
        await _ON_MODEL_READY(path)


__all__ = [
    "INFERENCE_LLM_API_KEY",
    "INFERENCE_LLM_ENABLE_THINKING",
    "INFERENCE_LLM_ENDPOINT_URL",
    "INFERENCE_LLM_MODEL",
    "INFERENCE_LLM_SOURCE",
    "INFERENCE_LOCAL_BINARY_PATH",
    "INFERENCE_LOCAL_CONTEXT_SIZE",
    "INFERENCE_LOCAL_IDLE_UNLOAD_SECONDS",
    "INFERENCE_LOCAL_MODELS_MAX",
    "INFERENCE_LOCAL_PRELOAD_ON_READY",
    "INFERENCE_LOCAL_STOP_SERVER_AFTER_IDLE_SECONDS",
    "INFERENCE_LOCAL_THREADS_GEN",
    "INFERENCE_LOCAL_THREADS_PREFILL",
    "JOBS_MAX_CONCURRENT_DOWNLOAD_MODEL",
    "bind_gateway",
    "bind_models_dir",
    "bind_on_model_ready",
    "bind_runtime",
    "build_gateway_from_settings",
    "build_runtime_from_settings",
    "get_gateway",
    "get_models_dir",
    "get_runtime",
    "get_supervisor",
    "notify_model_ready",
    "rebuild_runtime",
]
