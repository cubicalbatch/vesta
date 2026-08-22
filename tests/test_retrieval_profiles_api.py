"""Retrieval profile CRUD API.

User profiles round-trip through the ``retrieval.profiles`` settings blob;
built-ins stay read-only. A saved profile must be immediately usable by
``GET /api/search?profile=`` — no restart.
"""

from __future__ import annotations

import httpx
import pytest

_VALID_YAML = """
name: console_test_profile
description: a saved test profile
sources:
  - impl: xapian_fts
    params: {limit: 10, fallback_ladder: true}
  - impl: title_suggest
    params: {limit: 10, keyword_boost: 1.5}
fusion:
  impl: rrf
  params: {k: 20, group_by: archive, across_archives: union}
passages:
  impl: candidate_articles
  params: {max_articles: 5, max_passages: 50}
scorers:
  - impl: lexical_overlap
assembler:
  impl: topk_budget
  params: {budget_tokens: 1200, max_per_article: 2, dedup: near_exact}
"""


@pytest.mark.asyncio
async def test_list_profiles_includes_builtins(app_client: httpx.AsyncClient) -> None:
    resp = await app_client.get("/api/retrieval/profiles")
    assert resp.status_code == 200
    items = resp.json()["profiles"]
    names = {p["name"]: p["builtin"] for p in items}
    assert names.get("lexical") is True


@pytest.mark.asyncio
async def test_get_builtin_profile_detail(app_client: httpx.AsyncClient) -> None:
    resp = await app_client.get("/api/retrieval/profiles/lexical")
    assert resp.status_code == 200
    body = resp.json()
    assert body["builtin"] is True
    assert body["name"] == "lexical"
    assert "impl" in body["profile"]["fusion"]
    assert "name: lexical" in body["yaml"]


@pytest.mark.asyncio
async def test_save_list_get_delete_user_profile(app_client: httpx.AsyncClient) -> None:
    save = await app_client.post("/api/retrieval/profiles", json={"yaml": _VALID_YAML})
    assert save.status_code == 200
    saved = save.json()
    assert saved["builtin"] is False
    assert saved["name"] == "console_test_profile"
    assert saved["hash"]

    listed = await app_client.get("/api/retrieval/profiles")
    names = {p["name"]: p["builtin"] for p in listed.json()["profiles"]}
    assert names.get("console_test_profile") is False

    got = await app_client.get("/api/retrieval/profiles/console_test_profile")
    assert got.status_code == 200
    assert got.json()["hash"] == saved["hash"]

    deleted = await app_client.delete("/api/retrieval/profiles/console_test_profile")
    assert deleted.status_code == 200
    missing = await app_client.get("/api/retrieval/profiles/console_test_profile")
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_save_profile_via_structured_form(app_client: httpx.AsyncClient) -> None:
    """The generated-form fast path: send the ``profile`` dict, not YAML text."""
    base = await app_client.get("/api/retrieval/profiles/lexical")
    profile = base.json()["profile"]
    profile["name"] = "console_form_clone"
    profile["description"] = "cloned via form"
    save = await app_client.post("/api/retrieval/profiles", json={"profile": profile})
    assert save.status_code == 200
    body = save.json()
    assert body["name"] == "console_form_clone"
    assert body["profile"]["fusion"]["impl"] == profile["fusion"]["impl"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"yaml": _VALID_YAML.replace("console_test_profile", "lexical")},
        {"yaml": "name: bad\nsources: []\n"},
    ],
)
async def test_save_rejects_invalid_payload(
    app_client: httpx.AsyncClient, payload: dict[str, str]
) -> None:
    resp = await app_client.post("/api/retrieval/profiles", json=payload)
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_delete_builtin_is_rejected(app_client: httpx.AsyncClient) -> None:
    resp = await app_client.delete("/api/retrieval/profiles/lexical")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_delete_unknown_user_profile_404(app_client: httpx.AsyncClient) -> None:
    resp = await app_client.delete("/api/retrieval/profiles/does_not_exist")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_components_endpoint_lists_registered_impls(app_client: httpx.AsyncClient) -> None:
    resp = await app_client.get("/api/retrieval/components")
    assert resp.status_code == 200
    components = resp.json()["components"]
    source_names = {c["name"] for c in components["sources"]}
    assert "xapian_fts" in source_names
    fusion = components["fusion"][0]
    assert fusion["params_schema"] is not None
