"""Local ``llama-server`` lifecycle — supervised child process in router mode.

Router mode (start ``llama-server`` with no ``--model``) is what makes the local
path a clean win: ``--models-dir``, ``--models-preset``, ``/models/load``,
``/models/unload``, ``--models-max`` LRU, ``--sleep-idle-seconds`` — all built-in.
We **configure** that lifecycle rather than implement it.

This supervisor does ~150 lines of glue: spawn, health-wait,
restart on crash with backoff, clean SIGTERM shutdown. It is **lazy** — started
only when an answer is actually requested, not at app startup.

Two traps drive the design:
* The healthcheck must return 200 with **no model loaded** — otherwise the
  container looks unhealthy on every idle box.
* The binary may be absent on this machine (dev box without ``llama-server``).
  The supervisor degrades: no ``Capability.LLM``, ``sources_only`` stays usable.

``inference/`` depends ONLY on ``config``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import shutil
import signal
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vesta.config.settings import SettingsSnapshot

_log = logging.getLogger(__name__)

#: How long to wait for ``/health`` to respond 200 after spawning (seconds).
_HEALTH_TIMEOUT_S = 30.0
#: Poll interval for the health check.
_HEALTH_POLL_S = 0.5
#: Initial backoff for restart-on-crash (doubles up to ``_MAX_BACKOFF_S``).
_INITIAL_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 30.0
#: Max restart attempts before giving up (degrades to no-LLM).
_MAX_RESTARTS = 5
#: Timeout for the ``--list-devices`` hardware probe (a hung binary must not
#: wedge a chat turn — kill and assume CPU).
_LIST_DEVICES_TIMEOUT_S = 10.0
#: Max logged chars per drained child-output line (banners can be enormous).
_DRAIN_LINE_MAX = 500
#: Startup banner cross-check: "offloaded 33/35 layers to GPU" with n>0 means
#: the running child actually offloaded to a GPU — outranks --list-devices.
_GPU_OFFLOAD_RE = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")

#: Router-mode default port. Not exposed as a setting — single-user, single
#: worker; a fixed localhost port is simpler than coordination.
DEFAULT_PORT = 8081


class LlamaServerSupervisor:
    """Supervised ``llama-server`` child process in router mode.

    The supervisor owns the process lifecycle. ``ensure_running`` is idempotent
    and lazy: the first caller spawns the child, subsequent callers find it
    already up. A crash triggers an async restart with exponential backoff; the
    next ``ensure_running`` waits for the restart to succeed.
    """

    def __init__(
        self,
        *,
        binary_path: str,
        models_dir: Path,
        config_dir: Path,
        port: int = DEFAULT_PORT,
        threads_gen: int = 6,
        threads_prefill: int = 8,
        idle_unload_seconds: int = 900,
        models_max: int = 1,
        cache_reuse: int = 256,
        context_size: int = 8192,
    ) -> None:
        self._binary_path = binary_path
        self._models_dir = models_dir
        self._config_dir = config_dir
        self._port = port
        self._threads_gen = threads_gen
        self._threads_prefill = threads_prefill
        self._idle_unload_seconds = idle_unload_seconds
        self._models_max = models_max
        self._cache_reuse = cache_reuse
        self._context_size = context_size
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._restart_task: asyncio.Task[None] | None = None
        self._watcher_task: asyncio.Task[None] | None = None
        self._drain_task: asyncio.Task[None] | None = None
        #: Hardware class from ``--list-devices`` ("cpu"|"gpu"), probed once.
        self._hw: str | None = None
        #: Hardware class from the startup banner — a *cross-check* that
        #: outranks ``--list-devices`` when the child actually offloaded.
        self._hw_banner: str | None = None
        self._crashed = False
        self._base_url = f"http://127.0.0.1:{port}/v1"

    @property
    def base_url(self) -> str:
        """The OpenAI-compatible base URL (``http://127.0.0.1:<port>/v1``)."""
        return self._base_url

    @property
    def router_url(self) -> str:
        """The router control-plane root (``http://127.0.0.1:<port>``).

        The OpenAI-compatible surface lives under ``/v1`` (:attr:`base_url`);
        the model-management endpoints the runtime drives (``/models``,
        ``/models/load``, ``/models/unload``) live at the root.
        """
        return f"http://127.0.0.1:{self._port}"

    def binary_available(self) -> bool:
        """True if the ``llama-server`` binary exists on this machine.

        The capability probe reads this (cheap ``shutil.which`` / ``Path.exists``
        stat, never a process spawn — mirrors the encoders filesystem-stat probe).
        """
        if Path(self._binary_path).is_file():
            return True
        return shutil.which(self._binary_path) is not None

    async def hw_class(self) -> str:
        """The accelerator class of this machine: ``"cpu"`` or ``"gpu"``.

        Computed once and cached. The ``--list-devices`` probe is the source of
        truth, but a startup banner showing real GPU offload (see
        :meth:`_drain_output`) outranks it — the running child knows better
        than the device listing. Any spawn error, timeout, or empty device
        list degrades to ``"cpu"`` (the safe assumption for sizing).
        """
        if self._hw_banner is not None:
            self._hw = self._hw_banner
            return self._hw_banner
        if self._hw is None:
            self._hw = await self._probe_devices()
        return self._hw

    @property
    def hardware(self) -> str | None:
        """The cached hardware class, synchronously (``None`` before the first
        probe/banner — :meth:`hw_class` is the async path that fills it)."""
        return self._hw_banner or self._hw

    async def _probe_devices(self) -> str:
        """Spawn ``<binary> --list-devices`` and parse the listing.

        The binary prints ``Available devices:`` followed by either
        ``  (none)`` (verified live against the bundled b10373 binary on a
        CPU-only host) or one line per device. Anything but a non-empty device
        list ⇒ ``"cpu"``.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                self._binary_path,
                "--list-devices",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except Exception:
            return "cpu"
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=_LIST_DEVICES_TIMEOUT_S)
        except TimeoutError:
            with contextlib.suppress(Exception):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
            return "cpu"
        except Exception:
            return "cpu"
        return _parse_list_devices(out.decode("utf-8", errors="replace"))

    async def ensure_running(self) -> None:
        """Ensure the supervised child is up and healthy. Lazy on first call.

        Idempotent: if the process is alive, returns immediately. If a previous
        crash triggered an async restart, waits for it. If the binary is absent,
        raises :class:`BinaryMissing` so the caller can degrade gracefully.
        """
        if not self.binary_available():
            raise BinaryMissing(f"llama-server binary not found at {self._binary_path!r}")

        # One-shot hardware probe (cached): the first bring-up is the only
        # place with a natural pause to pay the subprocess cost.
        await self.hw_class()

        # If a restart is in progress, wait for it OUTSIDE the lock — the
        # restart task itself needs ``self._lock`` to call ``_start_and_wait``,
        # so awaiting it while holding the lock would deadlock (asyncio.Lock is
        # not reentrant).
        while self._restart_task is not None and not self._restart_task.done():
            with contextlib.suppress(Exception):
                await self._restart_task

        async with self._lock:
            if self._proc is not None and self._proc.returncode is None:
                return
            await self._start_and_wait()

    def is_running(self) -> bool:
        """True if the supervised child process is currently alive.

        A cheap liveness check (no health round-trip) for the runtime's status
        resolution and idle watchdog.
        """
        return self._proc is not None and self._proc.returncode is None

    async def restart(self) -> None:
        """Stop the child, then (lazily) bring it back.

        ``_start_and_wait`` rewrites ``models.ini`` on every start, so a
        context-size or thread change takes effect here — and a restart is also
        the only way a running router discovers GGUFs added to ``--models-dir``.
        ``ensure_running`` is lazy, so when the child was
        already down this merely cleans up.
        """
        await self.stop()
        await self.ensure_running()

    async def _start_and_wait(self) -> None:
        """Spawn the child, write ``models.ini``, and poll ``/health``."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._write_models_ini()
        cmd = self._build_command()
        _log.info("llama_server.starting", extra={"cmd": " ".join(cmd), "port": self._port})
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._crashed = False
        proc = self._proc
        assert proc is not None
        self._drain_task = asyncio.create_task(self._drain_output(proc), name="llama-server-drain")
        self._watcher_task = asyncio.create_task(self._watch(), name="llama-server-watcher")
        await self._wait_for_health(proc)

    def _write_models_ini(self) -> None:
        """Write the router-mode ``models.ini`` into the config dir.

        The ``[*]`` section applies to every model; per-model sections would be
        added as the user picks models from the registry. Threads: ``-t 6``
        generation / ``-tb 8`` prefill, **never 16** — Zen 5 data shows SMT
        reduces throughput here.

        ``c`` / ``jinja`` / ``reasoning-format`` were isolation-tested against
        the real b10373 binary (unknown INI keys abort startup). ``jinja = true`` is
        required for ``chat_template_kwargs.enable_thinking`` to work at all,
        and ``reasoning-format = auto`` is **mandatory**: ``none`` leaks
        ``<think>`` inline into ``content``.
        """
        self._config_dir.mkdir(parents=True, exist_ok=True)
        ini_path = self._config_dir / "models.ini"
        ini_content = (
            f"[*]\n"
            f"threads = {self._threads_gen}\n"
            f"threads-batch = {self._threads_prefill}\n"
            f"parallel = 1\n"
            f"cache-reuse = {self._cache_reuse}\n"
            f"c = {self._context_size}\n"
            f"jinja = true\n"
            f"reasoning-format = auto\n"
        )
        ini_path.write_text(ini_content, encoding="utf-8")

    def _build_command(self) -> list[str]:
        """Build the ``llama-server`` command line (router mode).

        Router mode is activated by **not specifying a model** — upstream: "Start
        the server in router mode by not specifying a model". There is no
        ``--router`` flag, and an unknown flag aborts startup. The health
        endpoint returns 200 with no model loaded — critical for a box that
        idles 99% of the time.
        """
        cmd = [
            self._binary_path,
            "--models-dir",
            str(self._models_dir),
            "--models-preset",
            str(self._config_dir / "models.ini"),
            "--models-max",
            str(self._models_max),
        ]
        if self._idle_unload_seconds > 0:
            # 0 means "never" — the flag is OMITTED, not passed as 0 (don't
            # pass 0 and hope upstream treats it as never).
            cmd += ["--sleep-idle-seconds", str(self._idle_unload_seconds)]
        cmd += [
            "--port",
            str(self._port),
            "--host",
            "0.0.0.0",
        ]
        return cmd

    async def _wait_for_health(self, proc: asyncio.subprocess.Process) -> None:
        """Poll ``/health`` until 200 or timeout (must work with no model).

        Takes the *specific* process it is waiting for: a concurrent
        :meth:`stop` (or a new ``_start_and_wait``) nulls/replaces
        ``self._proc`` — without the identity check this loop would poll a dead
        port for the full timeout instead of failing fast.
        """
        deadline = asyncio.get_event_loop().time() + _HEALTH_TIMEOUT_S
        url = f"http://127.0.0.1:{self._port}/health"
        while asyncio.get_event_loop().time() < deadline:
            if self._proc is not proc:
                raise LlamaServerError("llama-server startup aborted (stopped concurrently)")
            if proc.returncode is not None:
                raise LlamaServerError(
                    f"llama-server exited with code {proc.returncode} during startup"
                )
            try:
                code = await asyncio.to_thread(_check_health, url)
                if code == 200:
                    _log.info("llama_server.healthy", extra={"port": self._port})
                    return
            except Exception:
                pass
            await asyncio.sleep(_HEALTH_POLL_S)
        raise LlamaServerError(f"llama-server did not become healthy within {_HEALTH_TIMEOUT_S}s")

    async def _watch(self) -> None:
        """Watch for unexpected exit; trigger restart with backoff."""
        assert self._proc is not None
        await self._proc.wait()
        if self._crashed:
            return
        rc = self._proc.returncode
        _log.warning("llama_server.crashed", extra={"returncode": rc})
        self._crashed = True
        self._proc = None
        self._restart_task = asyncio.create_task(
            self._restart_with_backoff(), name="llama-server-restart"
        )

    async def _drain_output(self, proc: asyncio.subprocess.Process) -> None:
        """Continuously drain the child's stdout/stderr.

        Regression fix: the pipes were never drained, so crash output was
        swallowed and a chatty child could block on a full pipe buffer. Every
        line is logged at debug (truncated to ``_DRAIN_LINE_MAX``); a banner
        matching ``offloaded N/M layers to GPU`` with N>0 records the hardware
        class as a cross-check that outranks ``--list-devices``.
        """

        async def pump(stream: asyncio.StreamReader | None) -> None:
            if stream is None:
                return
            while True:
                raw = await stream.readline()
                if not raw:
                    return
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                match = _GPU_OFFLOAD_RE.search(line)
                if match is not None and int(match.group(1)) > 0:
                    self._hw_banner = "gpu"
                _log.debug("llama_server.output", extra={"line": line[:_DRAIN_LINE_MAX]})

        await asyncio.gather(pump(proc.stdout), pump(proc.stderr))

    async def _restart_with_backoff(self) -> None:
        """Restart the child with exponential backoff.

        Gives up after ``_MAX_RESTARTS`` attempts and leaves the supervisor in a
        crashed state; the next ``ensure_running`` raises, the caller degrades.
        """
        backoff = _INITIAL_BACKOFF_S
        for attempt in range(1, _MAX_RESTARTS + 1):
            await asyncio.sleep(backoff)
            _log.info("llama_server.restarting", extra={"attempt": attempt})
            try:
                async with self._lock:
                    await self._start_and_wait()
                return
            except Exception as exc:
                _log.warning(
                    "llama_server.restart_failed", extra={"attempt": attempt, "error": repr(exc)}
                )
                backoff = min(backoff * 2, _MAX_BACKOFF_S)
        _log.error("llama_server.restart_gave_up", extra={"max_attempts": _MAX_RESTARTS})

    async def stop(self) -> None:
        """Clean SIGTERM shutdown (kill is the reliable ``free()``)."""
        self._crashed = True
        if self._restart_task is not None and not self._restart_task.done():
            self._restart_task.cancel()
        if self._watcher_task is not None and not self._watcher_task.done():
            self._watcher_task.cancel()
        if self._drain_task is not None and not self._drain_task.done():
            self._drain_task.cancel()
        if self._proc is not None and self._proc.returncode is None:
            try:
                self._proc.send_signal(signal.SIGTERM)
                await asyncio.wait_for(self._proc.wait(), timeout=10.0)
            except Exception:
                with contextlib.suppress(Exception):
                    self._proc.kill()
        self._proc = None
        _log.info("llama_server.stopped")


def _check_health(url: str) -> int:
    """Return the HTTP status code from ``/health`` (blocking, run in a thread)."""
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except Exception:
        return 0


def _parse_list_devices(text: str) -> str:
    """Parse ``--list-devices`` output: ``"gpu"`` iff at least one device line
    follows ``Available devices:``.

    Measured shape on a CPU-only host (bundled b10373 binary)::

        Available devices:
          (none)

    A GPU host instead lists one indented line per device. Missing header,
    ``(none)``, or an empty list all mean ``"cpu"``.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "Available devices:":
            devices = [sub.strip() for sub in lines[i + 1 :] if sub.strip()]
            return "gpu" if devices and devices != ["(none)"] else "cpu"
    return "cpu"


class BinaryMissing(RuntimeError):
    """The ``llama-server`` binary is not present on this machine."""


class LlamaServerError(RuntimeError):
    """The ``llama-server`` process failed to start or became unhealthy."""


def build_supervisor_from_settings(
    snapshot: SettingsSnapshot, *, data_dir: Path
) -> LlamaServerSupervisor:
    """Construct a supervisor from resolved settings.

    Kept in ``inference/`` (not ``main.py``) so a new ``inference.local.*`` setting
    can't be added without updating the one place that constructs the supervisor
    — the same pattern ``encoders/__init__`` uses for ``build_manager_from_settings``.
    """
    from vesta.inference import (
        INFERENCE_LOCAL_BINARY_PATH,
        INFERENCE_LOCAL_CONTEXT_SIZE,
        INFERENCE_LOCAL_IDLE_UNLOAD_SECONDS,
        INFERENCE_LOCAL_MODELS_MAX,
        INFERENCE_LOCAL_THREADS_GEN,
        INFERENCE_LOCAL_THREADS_PREFILL,
    )

    return LlamaServerSupervisor(
        binary_path=str(snapshot.get(INFERENCE_LOCAL_BINARY_PATH)),
        models_dir=data_dir / "models",
        config_dir=data_dir / "config",
        threads_gen=int(snapshot.get(INFERENCE_LOCAL_THREADS_GEN)),
        threads_prefill=int(snapshot.get(INFERENCE_LOCAL_THREADS_PREFILL)),
        idle_unload_seconds=int(snapshot.get(INFERENCE_LOCAL_IDLE_UNLOAD_SECONDS)),
        models_max=int(snapshot.get(INFERENCE_LOCAL_MODELS_MAX)),
        context_size=int(snapshot.get(INFERENCE_LOCAL_CONTEXT_SIZE)),
    )


__all__ = [
    "DEFAULT_PORT",
    "BinaryMissing",
    "LlamaServerError",
    "LlamaServerSupervisor",
    "build_supervisor_from_settings",
]
