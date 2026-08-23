"""Library & catalog API.

Exercises the new surface against a live app: ``GET /api/catalog`` server-side
FTS filtering (never the full list), ``GET /api/catalog/curated`` (offline-safe),
``POST /api/catalog/refresh`` (job), ``POST /api/zims/download`` (job enqueue),
and the offline-first invariant — an empty cache yields an empty listing, never a
500. The catalog cache is seeded directly into SQLite (no network) so these tests
run hermetically.
"""

from __future__ import annotations

import httpx
import pytest

from vesta.catalog.opds import refresh_catalog_cache
from vesta.db.connection import Database

FIXTURE_BYTES = (
    '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">'
    "<entry><id>urn:uuid:eee</id><name>wikipedia_en_top</name>"
    "<title>Wikipedia Top Nopic</title><summary>Top articles, no images.</summary>"
    "<language>eng</language><flavour>nopic</flavour>"
    "<tags>wikipedia;_ftindex:yes</tags><articleCount>875265</articleCount>"
    '<link rel="http://opds-spec.org/acquisition/open-access" type="application/x-zim" '
    'href="https://lb.download.kiwix.org/zim/wikipedia/wikipedia_en_top_nopic_2026-06.zim.meta4" '
    'length="2403523584"/></entry>'
    "<entry><id>urn:uuid:fff</id><name>devdocs_en_lit</name>"
    "<title>Lit Docs</title><summary>Lit documentation.</summary>"
    "<language>eng</language><tags>devdocs;_ftindex:no</tags><articleCount>57</articleCount>"
    '<link rel="http://opds-spec.org/acquisition/open-access" type="application/x-zim" '
    'href="https://lb.download.kiwix.org/zim/devdocs/devdocs_en_lit_2026-07.zim.meta4" '
    'length="739328"/></entry>'
    "<entry><id>urn:uuid:ggg</id><name>wiktionary_fr_all</name>"
    "<title>Wiktionary FR</title><summary>Dictionnaire.</summary>"
    "<language>fra</language><tags>wiktionary</tags><articleCount>2100000</articleCount>"
    '<link rel="http://opds-spec.org/acquisition/open-access" type="application/x-zim" '
    'href="https://lb.download.kiwix.org/zim/wiktionary/wiktionary_fr.zim.meta4" '
    'length="4500000000"/></entry>'
    "</feed>"
)


async def _seed_catalog(client: httpx.AsyncClient) -> None:
    """Seed the catalog cache by running the refresh backing function through a
    mock transport against the app's bound DB — no real network."""
    transport = httpx.MockTransport(lambda req: httpx.Response(200, text=FIXTURE_BYTES))
    mock = httpx.AsyncClient(transport=transport)
    try:
        # Reach the bound DB via the ASGI transport's app reference.
        app = client._transport.app  # type: ignore[attr-defined]
        db: Database = app.state.vesta.db
        await refresh_catalog_cache(db, client=mock)
    finally:
        await mock.aclose()


@pytest.mark.asyncio
async def test_catalog_empty_cache_returns_empty_not_error(app_client: httpx.AsyncClient) -> None:
    """First boot / offline: no catalog cached → empty listing, never a 500."""
    resp = await app_client.get("/api/catalog")
    assert resp.status_code == 200
    body = resp.json()
    assert body["entries"] == []
    assert body["total"] == 0
    assert body["available"] is False


@pytest.mark.asyncio
async def test_catalog_state_reports_empty(app_client: httpx.AsyncClient) -> None:
    resp = await app_client.get("/api/catalog/state")
    assert resp.status_code == 200
    assert resp.json() == {"count": 0, "fetched_at": None, "available": False}


@pytest.mark.asyncio
async def test_curated_list_always_available_offline(app_client: httpx.AsyncClient) -> None:
    """The curated starter list is shipped data — available with no network."""
    resp = await app_client.get("/api/catalog/curated")
    assert resp.status_code == 200
    entries = resp.json()["entries"]
    names = [e["name"] for e in entries]
    assert names[0] == "wikipedia_en_100"  # rank 1 (Fastest start)
    assert any(e["warning"] and "221" in e["warning"] for e in entries)  # gutenberg


@pytest.mark.asyncio
async def test_catalog_fts_filters_server_side(app_client: httpx.AsyncClient) -> None:
    await _seed_catalog(app_client)
    # The client receives only the matching rows, never the full list.
    resp = await app_client.get("/api/catalog", params={"q": "lit documentation"})
    assert resp.status_code == 200
    body = resp.json()
    names = {e["name"] for e in body["entries"]}
    assert names == {"devdocs_en_lit"}
    assert body["available"] is True


@pytest.mark.asyncio
async def test_catalog_language_and_recommended_filters(app_client: httpx.AsyncClient) -> None:
    await _seed_catalog(app_client)
    # Language filter.
    resp = await app_client.get("/api/catalog", params={"language": "fra"})
    assert {e["name"] for e in resp.json()["entries"]} == {"wiktionary_fr_all"}
    # Recommended-only filter (curated_rank IS NOT NULL).
    resp = await app_client.get("/api/catalog", params={"recommended": "true"})
    names = {e["name"] for e in resp.json()["entries"]}
    assert "wikipedia_en_top" in names
    assert "devdocs_en_lit" not in names  # not curated


@pytest.mark.asyncio
async def test_catalog_sort_by_size(app_client: httpx.AsyncClient) -> None:
    await _seed_catalog(app_client)
    resp = await app_client.get("/api/catalog", params={"sort": "size_desc", "limit": 10})
    names = [e["name"] for e in resp.json()["entries"]]
    # 4.5 GB wiktionary > 2.4 GB wikipedia > 739 KB devdocs.
    assert names == ["wiktionary_fr_all", "wikipedia_en_top", "devdocs_en_lit"]


@pytest.mark.asyncio
async def test_catalog_max_size_filter(app_client: httpx.AsyncClient) -> None:
    await _seed_catalog(app_client)
    resp = await app_client.get("/api/catalog", params={"max_size": 1_000_000_000})
    names = {e["name"] for e in resp.json()["entries"]}
    assert names == {"devdocs_en_lit"}  # the only sub-1 GB archive


@pytest.mark.asyncio
async def test_catalog_languages_endpoint(app_client: httpx.AsyncClient) -> None:
    await _seed_catalog(app_client)
    resp = await app_client.get("/api/catalog/languages")
    assert resp.status_code == 200
    langs = resp.json()["languages"]
    assert {"code": "eng", "count": 2} in langs
    assert {"code": "fra", "count": 1} in langs


@pytest.mark.asyncio
async def test_catalog_curated_rank_and_warning_on_entries(app_client: httpx.AsyncClient) -> None:
    await _seed_catalog(app_client)
    resp = await app_client.get("/api/catalog", params={"q": "wikipedia"})
    entries = resp.json()["entries"]
    wiki = next(e for e in entries if e["name"] == "wikipedia_en_top")
    assert wiki["curated_rank"] == 2


@pytest.mark.asyncio
async def test_catalog_pagination(app_client: httpx.AsyncClient) -> None:
    await _seed_catalog(app_client)
    resp = await app_client.get("/api/catalog", params={"limit": 1, "offset": 0})
    body = resp.json()
    assert len(body["entries"]) == 1
    assert body["total"] >= 3


@pytest.mark.asyncio
async def test_get_catalog_entry_by_id(app_client: httpx.AsyncClient) -> None:
    await _seed_catalog(app_client)
    resp = await app_client.get("/api/catalog/urn:uuid:fff")
    assert resp.status_code == 200
    assert resp.json()["name"] == "devdocs_en_lit"
    # Unknown id → 404.
    resp = await app_client.get("/api/catalog/urn:uuid:nope")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_refresh_catalog_enqueues_job(app_client: httpx.AsyncClient) -> None:
    resp = await app_client.post("/api/catalog/refresh")
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_type"] == "refresh_catalog"
    assert body["job_id"] > 0


@pytest.mark.asyncio
async def test_download_enqueues_job_by_entry_id(app_client: httpx.AsyncClient) -> None:
    await _seed_catalog(app_client)
    resp = await app_client.post("/api/zims/download", json={"entry_id": "urn:uuid:fff"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_type"] == "download_zim"
    assert body["target"] == "devdocs_en_lit"


@pytest.mark.asyncio
async def test_download_requires_url_or_entry_id(app_client: httpx.AsyncClient) -> None:
    resp = await app_client.post("/api/zims/download", json={})
    assert resp.status_code == 400
    resp = await app_client.post("/api/zims/download", json={"url": "https://x/x.zim"})
    assert resp.status_code == 400  # name required


@pytest.mark.asyncio
async def test_download_accepts_manual_url_when_catalog_empty(
    app_client: httpx.AsyncClient,
) -> None:
    """The catalog is never the only path: a raw URL works too."""
    resp = await app_client.post(
        "/api/zims/download",
        json={"url": "https://example.com/x.zim.meta4", "name": "manual"},
    )
    assert resp.status_code == 200
    assert resp.json()["job_type"] == "download_zim"


@pytest.mark.asyncio
async def test_download_unknown_entry_id_404(app_client: httpx.AsyncClient) -> None:
    resp = await app_client.post("/api/zims/download", json={"entry_id": "urn:uuid:nope"})
    assert resp.status_code == 404


# ── POST /api/zims/download — filename hygiene (audit M2) ───────────────────


@pytest.mark.asyncio
async def test_download_rejects_unsafe_names_before_submitting(
    app_client: httpx.AsyncClient,
) -> None:
    """Absolute paths, traversal, and separators are rejected with a 400 before
    any job row exists — ``zims_dir / name`` would otherwise escape the zims
    dir (pathlib lets absolute paths win and ``..`` climb out)."""
    bad_names = (
        "/etc/evil.zim",  # absolute wins outright in pathlib
        "../evil",
        "sub/dir/evil",  # nested names are rejected, not rewritten
        "a\\b",  # backslash separator
        "..evil.zim",  # contains ..
        ".zim",  # empty stem
    )
    for bad in bad_names:
        resp = await app_client.post(
            "/api/zims/download",
            json={"url": "https://example.com/x.zim.meta4", "name": bad},
        )
        assert resp.status_code == 400, bad
    jobs = (await app_client.get("/api/jobs")).json()["jobs"]
    assert jobs == []


@pytest.mark.asyncio
async def test_download_validates_name_even_with_entry_id(
    app_client: httpx.AsyncClient,
) -> None:
    """An explicit name overrides the catalog entry's — the override must pass
    the same guard."""
    await _seed_catalog(app_client)
    resp = await app_client.post(
        "/api/zims/download",
        json={"entry_id": "urn:uuid:fff", "name": "../../evil"},
    )
    assert resp.status_code == 400
    jobs = (await app_client.get("/api/jobs")).json()["jobs"]
    assert jobs == []
