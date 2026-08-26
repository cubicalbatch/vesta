"""Archive & article HTTP API over the fixture, via the real lifespan.

Covers the reader passthrough route (correct MIME for the WebP trap, 302s for
redirects, nested paths), the extracted-article JSON route, scan, and
enable/disable/delete — all through the composition root the way a client does.
"""

from __future__ import annotations

import urllib.parse

import httpx

from fixtures.tiny_zim import (
    DISAMBIGUATION_PATH,
    LONG_ARTICLE_PATH,
    REDIRECT_PATH,
    REDIRECT_TARGET,
    SOFT_REDIRECT_PATH,
    WEBP_ASSET_PATH,
)


def _url(zim_id: int, path: str) -> str:
    return f"/api/zim/{zim_id}/{urllib.parse.quote(path, safe='/')}"


async def test_list_zims_and_scan(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    client, zim_id = app_client_with_zim
    resp = await client.get("/api/zims")
    archives = resp.json()["archives"]
    assert len(archives) == 1
    arc = archives[0]
    assert arc["id"] == zim_id
    # has_fulltext_index is a PROBED bool; article_count is an int from
    # Counter['text/html'], not the over-counted archive.article_count.
    assert isinstance(arc["has_fulltext_index"], bool)
    assert arc["article_count"] > 0
    assert arc["enabled"] is True
    # POST /api/zims/scan re-scans and stays consistent.
    scan = (await client.post("/api/zims/scan")).json()
    assert scan["total"] == 1


async def test_zim_passthrough_routing_and_mime_traps(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    """MIME trap, redirects, and fragment resolution over passthrough & article routes."""
    client, zim_id = app_client_with_zim

    # MIME trap: .jpg-named asset served as image/webp.
    webp = await client.get(_url(zim_id, WEBP_ASSET_PATH), follow_redirects=False)
    assert webp.status_code == 200
    assert webp.headers["content-type"].startswith("image/webp")
    assert webp.content[:4] == b"RIFF" and webp.content[8:12] == b"WEBP"
    assert "immutable" in webp.headers.get("cache-control", "")

    # Hard redirect is 302.
    hard = await client.get(_url(zim_id, REDIRECT_PATH), follow_redirects=False)
    assert hard.status_code == 302
    assert REDIRECT_TARGET in hard.headers["location"]

    # Soft redirect (<meta refresh>) becomes server-side 302 with #Section fragment.
    soft = await client.get(_url(zim_id, SOFT_REDIRECT_PATH), follow_redirects=False)
    assert soft.status_code == 302
    assert soft.headers["location"] == f"/api/zim/{zim_id}/{REDIRECT_TARGET}#Government"

    # Client-quoted fragment (%23) resolves on both passthrough and article routes.
    fragment_path = f"{REDIRECT_TARGET}#Government"
    passthrough = await client.get(_url(zim_id, fragment_path), follow_redirects=False)
    assert passthrough.status_code == 200
    assert passthrough.headers["content-type"].startswith("text/html")
    article = await client.get(
        f"/api/article/{zim_id}/{urllib.parse.quote(fragment_path, safe='/')}"
    )
    assert article.status_code == 200
    assert article.json()["path"] == REDIRECT_TARGET


async def test_article_route_returns_text_and_sections(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    client, zim_id = app_client_with_zim
    resp = await client.get(f"/api/article/{zim_id}/{LONG_ARTICLE_PATH}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["path"] == LONG_ARTICLE_PATH
    assert body["text"].strip()
    assert len(body["sections"]) >= 2
    # Section offsets index into the returned text.
    for s in body["sections"]:
        assert 0 <= s["char_start"] <= s["char_end"] <= len(body["text"])


async def test_article_route_disambiguation_classified(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    client, zim_id = app_client_with_zim
    resp = await client.get(f"/api/article/{zim_id}/{DISAMBIGUATION_PATH}")
    assert resp.status_code == 200
    # Disambiguation flag bit is set (entries.py classification).
    from vesta.zim.types import EntryFlags

    assert int(resp.json()["flags"]) & EntryFlags.DISAMBIGUATION


async def test_patch_enable_disable_and_delete(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    client, zim_id = app_client_with_zim
    # Disable.
    resp = await client.patch(f"/api/zims/{zim_id}", json={"enabled": False})
    assert resp.json()["ok"] is True
    arc = (await client.get("/api/zims")).json()["archives"][0]
    assert arc["enabled"] is False
    # Corpus label.
    await client.patch(f"/api/zims/{zim_id}", json={"corpus_label": "wiki"})
    arc = (await client.get("/api/zims")).json()["archives"][0]
    assert arc["corpus_label"] == "wiki"
    # Delete.
    resp = await client.delete(f"/api/zims/{zim_id}")
    assert resp.json()["ok"] is True
    assert (await client.get("/api/zims")).json()["archives"] == []


async def test_disable_and_delete_recompute_vectors_capability(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    """AUDIT_0824 S3: disabling or deleting the last indexed archive must
    recompute the cached VECTORS capability flag in the same request, not
    leave it claiming vectors until the next restart."""
    import vesta.index

    client, zim_id = app_client_with_zim
    try:
        vesta.index.set_indexed_state(True)
        # Disabling the archive drops its seed qualification.
        resp = await client.patch(f"/api/zims/{zim_id}", json={"enabled": False})
        assert resp.json()["ok"] is True
        assert vesta.index._ANY_INDEXED is False

        # Deletion likewise: row + vectors gone, the claim must follow.
        vesta.index.set_indexed_state(True)
        resp = await client.delete(f"/api/zims/{zim_id}")
        assert resp.json()["ok"] is True
        assert (await client.get("/api/zims")).json()["archives"] == []
        assert vesta.index._ANY_INDEXED is False
    finally:
        vesta.index.set_indexed_state(False)


async def test_missing_archive_404(app_client_with_zim: tuple[httpx.AsyncClient, int]) -> None:
    client, _ = app_client_with_zim
    resp = await client.get("/api/zim/99999/Foo", follow_redirects=False)
    assert resp.status_code == 404


async def test_random_article_returns_article_shape(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    """GET /api/zims/{id}/random — same ArticleOut shape as /api/article, and
    repeatable without ever erroring (the "Random article" action)."""
    client, zim_id = app_client_with_zim
    for _ in range(5):
        resp = await client.get(f"/api/zims/{zim_id}/random")
        assert resp.status_code == 200
        body = resp.json()
        assert body["zim_id"] == zim_id
        assert body["path"]
        assert "text" in body and "sections" in body and "flags" in body


async def test_samples_endpoint(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    """GET /api/zims/{id}/samples — deduplicated cards, default count, bounds, and termination."""
    client, zim_id = app_client_with_zim

    # Default count is six.
    default_resp = await client.get(f"/api/zims/{zim_id}/samples")
    assert default_resp.status_code == 200
    assert len(default_resp.json()) <= 6

    # Explicit count returns deduplicated ArticleOut-shaped items.
    resp = await client.get(f"/api/zims/{zim_id}/samples?count=6")
    assert resp.status_code == 200
    items = resp.json()
    assert isinstance(items, list)
    assert len(items) <= 6
    paths: set[str] = set()
    for item in items:
        assert item["zim_id"] == zim_id
        assert item["path"]
        assert item["text"].strip()  # empty-text entries are skipped
        assert "sections" in item and "flags" in item
        paths.add(item["path"])
    assert len(paths) == len(items), "samples must be deduped by path"

    # Count bounds [1, 24] rejected with 422.
    assert (await client.get(f"/api/zims/{zim_id}/samples?count=0")).status_code == 422
    assert (await client.get(f"/api/zims/{zim_id}/samples?count=25")).status_code == 422

    # Small archive terminates promptly across repeated calls without hanging.
    for _ in range(10):
        body = await client.get(f"/api/zims/{zim_id}/samples?count=24")
        assert body.status_code == 200
        assert 0 < len(body.json()) <= 24
