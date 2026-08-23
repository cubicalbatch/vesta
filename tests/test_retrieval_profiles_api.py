"""Retrieval profiles listing API.

The read-only surface that survived the AUDIT_0822 dead-code sweep: the SPA
consumes only ``GET /api/retrieval/profiles``. User profiles still round-trip
through the ``retrieval.profiles`` settings blob (loaded for the listing);
built-ins stay read-only.
"""

from __future__ import annotations

import httpx
import pytest


@pytest.mark.asyncio
async def test_list_profiles_includes_builtins(app_client: httpx.AsyncClient) -> None:
    resp = await app_client.get("/api/retrieval/profiles")
    assert resp.status_code == 200
    items = resp.json()["profiles"]
    names = {p["name"]: p["builtin"] for p in items}
    assert names.get("lexical") is True


@pytest.mark.asyncio
async def test_removed_editor_endpoints_are_gone(app_client: httpx.AsyncClient) -> None:
    """The orphaned profile-editor surface (detail/save/delete/components) was
    removed — no client ever called it. Probed via the OpenAPI table because the
    SPA catch-all answers stray GETs with the app shell; the spec only ever
    lists real API routes. The listing must be the sole surviving path and it
    must be read-only."""
    spec = (await app_client.get("/openapi.json")).json()
    retrieval_paths = {
        path: set(methods)
        for path, methods in spec["paths"].items()
        if path.startswith("/api/retrieval")
    }
    assert retrieval_paths == {"/api/retrieval/profiles": {"get"}}

    # Behavioral backstop: the surviving listing rejects mutations.
    save = await app_client.post("/api/retrieval/profiles", json={"yaml": "name: x\n"})
    assert save.status_code == 405
