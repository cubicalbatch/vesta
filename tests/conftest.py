"""Shared pytest fixtures.

Tests run the real lifespan against a throwaway SQLite file in a tmp data dir,
so the DB/migrations/runner/settings all get exercised end to end. The tiny
env-var pin (``data.dir``) steers the app at a temp path without touching the
working tree.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from fixtures.tiny_zim import build_tiny_zim
from vesta.main import create_app


@pytest_asyncio.fixture
async def app_client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    """A real app instance with its lifespan driven, pointed at a tmp data dir.

    The ``inference.llm.*`` defaults are inert on a fresh install (source
    ``local``, empty model), so nothing dials out and
    ``Capability.LLM`` is honestly off. The env pin keeps it that way even if a
    default (or a DB stored under the tmp data dir) ever drifts: tests must
    NEVER make real LLM calls (slow + costly), and with an empty model the app
    degrades to ``sources_only`` (no network) unless a test explicitly opts
    into the LLM path (force the capability + stub the model). Set alongside
    ``data.dir``; cleaned up in ``finally``.
    """
    os.environ["data.dir"] = str(tmp_path)
    os.environ["inference.llm.model"] = ""
    try:
        app = create_app()
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                yield client
    finally:
        os.environ.pop("data.dir", None)
        os.environ.pop("inference.llm.model", None)


@pytest_asyncio.fixture
async def app_client_with_zim(tmp_path: Path) -> AsyncIterator[tuple[httpx.AsyncClient, int]]:
    """An app whose ``./data/zims`` holds the tiny fixture ZIM at startup.

    The lifespan scan registers it (probes the index, mines aliases) so the
    archive/reader/extract paths are exercisable end-to-end. Yields
    ``(client, zim_id)`` where ``zim_id`` is the registered archive's row id.

    Like :func:`app_client`, the LLM model is neutralized so no test makes a real
    LLM call (see that fixture's note).
    """
    zims_dir = tmp_path / "zims"
    zims_dir.mkdir(parents=True, exist_ok=True)
    build_tiny_zim(zims_dir / "tiny.zim")
    os.environ["data.dir"] = str(tmp_path)
    os.environ["inference.llm.model"] = ""
    try:
        app = create_app()
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/zims")
                archives = resp.json()["archives"]
                zim_id = int(archives[0]["id"])
                yield client, zim_id
    finally:
        os.environ.pop("data.dir", None)
        os.environ.pop("inference.llm.model", None)


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "vesta.db"
