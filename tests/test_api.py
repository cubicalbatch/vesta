"""API: health, settings schema/get/put, jobs create/stream/control."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest


@pytest.mark.asyncio
async def test_health_returns_ok_with_capability_breakdown(app_client: httpx.AsyncClient) -> None:
    resp = await app_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["components"]["database"] == "ok"
    assert "capabilities" in body
    # The test fixtures neutralize ``inference.llm.model`` (conftest) so no test
    # makes a real LLM call — so ``llm`` is correctly ABSENT from the test app's
    # reported capabilities. The cheap config probe (no network) is what's
    # exercised here; the production app reports ``llm`` when a model is set.
    assert "llm" not in body["capabilities_available"]


@pytest.mark.asyncio
async def test_settings_schema_lists_every_setting(app_client: httpx.AsyncClient) -> None:
    resp = await app_client.get("/api/settings/schema")
    assert resp.status_code == 200
    items = resp.json()["settings"]
    keys = {i["key"] for i in items}
    assert {
        "server.host",
        "server.port",
        "data.dir",
        "jobs.max_concurrent.noop",
        "logging.level",
        "db.busy_timeout_ms",
    } <= keys
    # Each carries type/bounds/group/help (the settings UI is generated from this).
    for item in items:
        assert item["type"] in {"string", "integer", "float", "boolean"}
        assert item["group"]
        assert item["help"]
        assert "hot" in item


@pytest.mark.asyncio
async def test_get_settings_returns_resolved_values(app_client: httpx.AsyncClient) -> None:
    resp = await app_client.get("/api/settings")
    assert resp.status_code == 200
    values = resp.json()["values"]
    assert values["server.host"] == "127.0.0.1"
    assert values["server.port"] == 8080


@pytest.mark.asyncio
async def test_put_hot_setting_observed_without_restart(
    app_client: httpx.AsyncClient,
) -> None:
    """The table is authoritative; a hot change shows on the next request."""
    resp = await app_client.put("/api/settings", json={"values": {"logging.level": "DEBUG"}})
    assert resp.status_code == 200
    assert "logging.level" in resp.json()["applied"]
    got = await app_client.get("/api/settings")
    assert got.json()["values"]["logging.level"] == "DEBUG"


@pytest.mark.asyncio
async def test_put_rejects_unknown_key(app_client: httpx.AsyncClient) -> None:
    resp = await app_client.put("/api/settings", json={"values": {"no.such.key": "x"}})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_put_rejects_out_of_bounds(app_client: httpx.AsyncClient) -> None:
    resp = await app_client.put("/api/settings", json={"values": {"server.port": "99999"}})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_create_list_get_job(app_client: httpx.AsyncClient) -> None:
    create = await app_client.post(
        "/api/jobs", json={"type": "noop", "params": {"total": 3, "delay": 0.01}}
    )
    assert create.status_code == 200
    jid = create.json()["id"]
    # let it finish
    for _ in range(100):
        await asyncio.sleep(0.01)
        got = await app_client.get(f"/api/jobs/{jid}")
        if got.json().get("status") in {"done", "error", "cancelled"}:
            break
    got = await app_client.get(f"/api/jobs/{jid}")
    assert got.status_code == 200
    assert got.json()["status"] == "done"

    listed = await app_client.get("/api/jobs")
    assert listed.status_code == 200
    assert any(j["id"] == jid for j in listed.json()["jobs"])


@pytest.mark.asyncio
async def test_create_job_rejects_unknown_type(app_client: httpx.AsyncClient) -> None:
    resp = await app_client.post("/api/jobs", json={"type": "bogus"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_create_download_model_job_validates_params(
    app_client: httpx.AsyncClient,
) -> None:
    """``POST /api/jobs`` must not bypass the download endpoint's filename
    hygiene: hostile filenames, missing keys, and unknown params are rejected
    before any job row exists (audit M1)."""
    bad_submissions = [
        {"url": "https://x/a.gguf", "filename": "../evil.gguf"},
        {"url": "https://x/a.gguf", "filename": "/abs/a.gguf"},
        {"url": "https://x/a.gguf", "filename": "sub/dir/a.gguf"},
        {"url": "https://x/a.gguf"},  # missing filename
        {"filename": "ok.gguf"},  # missing url
        {"url": "https://x/a.gguf", "filename": "a.gguf", "resume": {}},  # unknown param
    ]
    for params in bad_submissions:
        resp = await app_client.post("/api/jobs", json={"type": "download_model", "params": params})
        assert resp.status_code == 400, params
    assert (await app_client.get("/api/jobs")).json()["jobs"] == []


@pytest.mark.asyncio
async def test_create_job_rejects_bad_params_for_every_type(
    app_client: httpx.AsyncClient,
) -> None:
    """Every registered type gates its params at ``POST /api/jobs`` (audit
    AUDIT_0824 N2): unknown keys, wrong-typed values, out-of-range values, and
    the reserved checkpoint key are refused before any job row exists — params
    are never passed through verbatim to the handler."""
    bad_submissions = [
        ("refresh_catalog", {"url": 123}),  # wrong type
        ("refresh_catalog", {"feed": "https://lan/opds"}),  # unknown key
        ("refresh_catalog", {"__checkpoint__": {}}),  # reserved key
        ("index_zim", {"depth": 2}),  # missing zim_id
        ("index_zim", {"zim_id": "one"}),  # wrong type
        ("index_zim", {"zim_id": True}),  # bool is not an int here
        ("index_zim", {"zim_id": 1, "depth": 0}),  # out of range
        ("index_zim", {"zim_id": 1, "depth": 4}),  # out of range
        ("download_zim", {}),  # missing url
        ("download_zim", {"url": "https://x/a.zim.meta4", "size": -1}),
        ("download_zim", {"url": "https://x/a.zim.meta4", "title": 5}),
        ("download_zim", {"url": "https://x/a.zim.meta4", "name": "../evil"}),
        ("noop", {"total": 0}),
        ("noop", {"delay": "fast"}),
    ]
    for jtype, params in bad_submissions:
        resp = await app_client.post("/api/jobs", json={"type": jtype, "params": params})
        assert resp.status_code == 400, (jtype, params)
    assert (await app_client.get("/api/jobs")).json()["jobs"] == []


def test_job_param_validators_accept_handler_shaped_params() -> None:
    """The per-type gates accept exactly what each handler consumes, including
    boundary values — so a legitimate client submission is never refused."""
    from vesta.api.jobs import _validate_job_params

    valid: dict[str, list[dict[str, object]]] = {
        "refresh_catalog": [{}, {"url": "https://lan/portal/opds"}],
        "index_zim": [{"zim_id": 1}, {"zim_id": 7, "depth": 1}, {"zim_id": 7, "depth": 3}],
        "download_zim": [
            {"url": "https://x/a.zim.meta4"},
            {
                "url": "https://x/a.zim.meta4",
                "name": "wikipedia",
                "title": "Wikipedia",
                "sha256": "ab" * 32,
                "size": 0,
            },
        ],
        "noop": [{}, {"total": 3, "delay": 0}, {"total": 1, "delay": 0.01}],
    }
    for jtype, cases in valid.items():
        for params in cases:
            _validate_job_params(jtype, dict(params))  # must not raise


@pytest.mark.asyncio
async def test_job_sse_stream_emits_snapshot_then_progress(
    app_client: httpx.AsyncClient,
) -> None:
    """The noop job reports progress over SSE (01-foundations DoD)."""
    create = await app_client.post(
        "/api/jobs", json={"type": "noop", "params": {"total": 6, "delay": 0.02}}
    )
    jid = create.json()["id"]
    # Drain a few SSE events from the per-job stream.
    async with app_client.stream("GET", f"/api/jobs/{jid}/stream") as response:
        assert response.status_code == 200
        events: list[tuple[str, dict]] = []
        name = ""
        async for line in response.aiter_lines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
                events.append((name, data))
                if name in {"status"} and data.get("status") in {"done", "error", "cancelled"}:
                    break
            if len(events) > 50:
                break
    names = [n for n, _ in events]
    assert "snapshot" in names
    # At least one progress or status update landed.
    assert any(n in {"progress", "status"} for n in names)


@pytest.mark.asyncio
async def test_pause_resume_cancel_endpoints(app_client: httpx.AsyncClient) -> None:
    create = await app_client.post(
        "/api/jobs", json={"type": "noop", "params": {"total": 500, "delay": 0.01}}
    )
    jid = create.json()["id"]
    await asyncio.sleep(0.1)
    assert (await app_client.post(f"/api/jobs/{jid}/pause")).status_code in (200, 409)
    assert (await app_client.post(f"/api/jobs/{jid}/resume")).status_code in (200, 409)
    cancel = await app_client.post(f"/api/jobs/{jid}/cancel")
    assert cancel.status_code == 200


@pytest.mark.asyncio
async def test_health_reports_setup_not_completed_by_default(
    app_client: httpx.AsyncClient,
) -> None:
    """A fresh install reports setup_completed=False — the SPA uses this to route
    `/` → `/welcome` on a zero-archive state."""
    resp = await app_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["setup_completed"] is False


@pytest.mark.asyncio
async def test_setup_complete_endpoint_flips_health_flag(
    app_client: httpx.AsyncClient,
) -> None:
    """POST /api/setup/complete persists the flag; /health then reports True so
    the SPA never bounces a finished user back to /welcome."""
    post = await app_client.post("/api/setup/complete")
    assert post.status_code == 200
    assert post.json()["ok"] is True
    # Idempotent: a second call is a no-op that keeps the flag set.
    assert (await app_client.post("/api/setup/complete")).status_code == 200
    assert (await app_client.get("/health")).json()["setup_completed"] is True


@pytest.mark.asyncio
async def test_model_presets_report_downloaded_flag(app_client: httpx.AsyncClient) -> None:
    """Each preset carries a `downloaded` bool (file-on-disk check). The test
    app's models dir is empty, so every preset reports False — the wizard uses
    this to show "Downloaded" instead of a Download button when True."""
    resp = await app_client.get("/api/models/presets")
    assert resp.status_code == 200
    presets = resp.json()["presets"]
    assert len(presets) >= 1
    for p in presets:
        assert "downloaded" in p
        assert p["downloaded"] is False
        assert p["url"].startswith("https://")
