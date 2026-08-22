"""A stub ``LlmRuntime`` for tests.

Implements the surface ``api/agent_chat`` and ``api/models`` consume:
``target()``, ``ensure_ready(on_status=)``, ``mark_used()``, ``rebuild(snapshot)``,
plus the lifecycle surface ``load()``/``unload()``/``status()``. Bind it
by monkeypatching ``vesta.inference.get_runtime`` (the seam ``agent_chat``,
the models API, and ``rebuild_runtime`` resolve at call time).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vesta.inference.runtime import LlmStatus, LlmTarget


def _default_status(state: str = "unloaded", error: str | None = None) -> LlmStatus:
    return LlmStatus(
        source="local",
        configured=True,
        installed=True,
        state=state,
        model_file="stub.gguf",
        display_name="Stub",
        model_id="stub",
        size_bytes=1,
        context_size=8192,
        thinking=False,
        thinking_supported=True,
        idle_unload_seconds=900,
        seconds_since_last_use=None,
        estimated_ram_bytes=0,
        error=error,
    )


@dataclass
class FakeLlmRuntime:
    """Configurable stub: fixed target, optional warm-up statuses / error."""

    base_url: str = "http://127.0.0.1:9999/v1"
    model_id: str = "stub-model"
    api_key: str = "local"
    enable_thinking: bool | None = False
    #: Messages ``ensure_ready`` reports through ``on_status``, in order.
    status_messages: list[str] = field(default_factory=list)
    #: Raised by ``ensure_ready``/``load`` after the statuses were reported.
    error: Exception | None = None
    #: ``mark_used`` call count.
    used: int = 0
    #: Snapshots received by ``rebuild``, in order.
    rebuild_snapshots: list[Any] = field(default_factory=list)
    #: ``load``/``unload``/``ensure_ready`` call counts.
    load_calls: int = 0
    unload_calls: int = 0
    ready_calls: int = 0
    #: Returned by ``status()`` (a default local status when ``None``).
    status_value: LlmStatus | None = None

    def target(self) -> LlmTarget:
        return LlmTarget(
            source="local",
            base_url=self.base_url,
            api_key=self.api_key,
            model_id=self.model_id,
            enable_thinking=self.enable_thinking,
        )

    async def ensure_ready(self, *, on_status: Any = None) -> None:
        self.ready_calls += 1
        for msg in self.status_messages:
            if on_status is not None:
                on_status(msg)
        if self.error is not None:
            raise self.error

    async def load(self) -> None:
        self.load_calls += 1
        if self.error is not None:
            raise self.error

    async def unload(self) -> None:
        self.unload_calls += 1

    async def status(self) -> LlmStatus:
        if self.status_value is not None:
            return self.status_value
        return _default_status()

    def mark_used(self) -> None:
        self.used += 1

    async def rebuild(self, snapshot: Any, *, force_restart: bool = False) -> None:
        self.rebuild_snapshots.append(snapshot)
