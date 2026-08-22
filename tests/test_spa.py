"""SPA static file server (11-production-frontend.md "How it is served"):
the router-order guarantee and the path-traversal guard.

The phase doc's hard requirement: ``spa.router``'s catch-all
``GET /{path:path}`` must never shadow a real ``/api/*`` or ``/health``
route. ``api/__init__.py`` registers ``spa.router`` last so every
other router's routes are tried first; this is asserted here behaviourally
(hit the real endpoints and check for their real payload/headers), which is
robust to whatever routing-priority implementation FastAPI uses internally,
rather than introspecting the route table's private structure.

``src/vesta/static/app/`` is build output (``npm run build``), gitignored,
and may not exist in every environment this test runs in — tests that need
the built ``index.html`` skip cleanly when it's absent, same pattern as
``settings-basic.test.ts``'s no-backend skip on the frontend side.
"""

from __future__ import annotations

import httpx
import pytest

from vesta.api import spa

_BUILD_PRESENT = (spa._STATIC_DIR / spa._INDEX).is_file()
_skip_no_build = pytest.mark.skipif(
    not _BUILD_PRESENT,
    reason="frontend not built (src/vesta/static/app/index.html absent — run `npm run build`)",
)


@pytest.mark.asyncio
async def test_health_not_shadowed_by_spa_catch_all(app_client: httpx.AsyncClient) -> None:
    resp = await app_client.get("/health")
    assert resp.status_code == 200
    assert "capabilities" in resp.json()  # health's real shape, not index.html


@pytest.mark.asyncio
async def test_api_routes_not_shadowed_by_spa_catch_all(app_client: httpx.AsyncClient) -> None:
    resp = await app_client.get("/api/zims")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    assert "archives" in resp.json()

    resp = await app_client.get("/api/settings/schema")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")


@_skip_no_build
@pytest.mark.asyncio
async def test_spa_index_served_for_root(app_client: httpx.AsyncClient) -> None:
    resp = await app_client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert resp.headers.get("cache-control") == "no-store"
    assert "<html" in resp.text.lower()


@_skip_no_build
@pytest.mark.asyncio
async def test_spa_index_served_for_unknown_client_route(app_client: httpx.AsyncClient) -> None:
    """A hard refresh on a client-side route (`/catalog`, `/ask/abc123`, ...)
    has no matching backend route and must fall back to `index.html`, not a
    404 — this is what keeps client-side routing alive across a reload."""
    for path in ("/catalog", "/ask/abc123", "/settings", "/some/deeply/nested/route"):
        resp = await app_client.get(path)
        assert resp.status_code == 200, path
        assert resp.headers.get("cache-control") == "no-store", path
        assert "<html" in resp.text.lower(), path


@_skip_no_build
@pytest.mark.asyncio
async def test_spa_build_asset_served_immutable(app_client: httpx.AsyncClient) -> None:
    assets_dir = spa._STATIC_DIR / "_app"
    hashed_js = next(assets_dir.rglob("*.js"))
    rel = hashed_js.relative_to(spa._STATIC_DIR).as_posix()
    resp = await app_client.get(f"/{rel}")
    assert resp.status_code == 200
    assert resp.headers.get("cache-control") == "public, max-age=31536000, immutable"


def test_spa_asset_path_traversal_is_blocked() -> None:
    """Same reasoning as the retired dev console's traversal test: called
    directly (bypassing an HTTP client) because a client normalizes literal
    dot-segments in a URL before the request is even sent, which would test
    the client, not the guard."""
    assert spa._resolve_static_file("../../main.py") is None
    assert spa._resolve_static_file("_app/../../../../etc/passwd") is None


@_skip_no_build
@pytest.mark.asyncio
async def test_spa_asset_traversal_via_http_is_rejected(app_client: httpx.AsyncClient) -> None:
    """Belt-and-braces HTTP-level check for the `/_app/{path:path}` route
    specifically, using a URL-encoded traversal so the client doesn't
    normalize it away before the request is sent."""
    resp = await app_client.get("/_app/%2e%2e/%2e%2e/%2e%2e/%2e%2e/etc/passwd")
    assert resp.status_code == 404
