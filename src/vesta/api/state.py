"""Composition-root state shared with the API routers.

``api/`` is the only package that may import from more than two internal
packages; it holds the wires, not the logic. This dataclass is the wire bundle:
the DB, the job runner and the ZIM archive registry, each constructed once in
``main`` and read by routers via a FastAPI dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import Request

from vesta.db.connection import Database
from vesta.jobs.runner import JobRunner

if TYPE_CHECKING:
    from vesta.encoders.manager import EncoderManager
    from vesta.inference.gateway import Gateway
    from vesta.inference.local import LlamaServerSupervisor
    from vesta.zim.registry import ArchiveRegistry


@dataclass
class AppState:
    db: Database
    runner: JobRunner
    registry: ArchiveRegistry | None = None
    encoders: EncoderManager | None = None
    gateway: Gateway | None = None
    supervisor: LlamaServerSupervisor | None = None


def app_state(request: Request) -> AppState:
    """FastAPI dependency: the live :class:`AppState` attached to the app."""
    state = request.app.state.vesta
    if not isinstance(state, AppState):  # pragma: no cover - wiring bug
        raise RuntimeError("app.state.vesta is not an AppState")
    return state
