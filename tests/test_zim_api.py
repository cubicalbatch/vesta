"""Archive & article HTTP API over the fixture, via the real lifespan.

Covers the reader passthrough route (correct MIME for the WebP trap, 302s for
redirects, nested paths), the extracted-article JSON route, scan, and
enable/disable/delete — all through the composition root the way a client does.
"""

from __future__ import annotations

import urllib.parse
from pathlib import Path

import httpx
import pytest
from starlette.requests import Request

from fixtures.tiny_zim import (
    DISAMBIGUATION_PATH,
    LONG_ARTICLE_PATH,
    REDIRECT_PATH,
    REDIRECT_TARGET,
    SOFT_REDIRECT_PATH,
    WEBP_ASSET_PATH,
)
from vesta.api.zim import _parse_range, _RangeUnsatisfiable, _serve


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


def test_parse_range_valid_and_edge_cases() -> None:
    """_parse_range unit tests covering valid, suffix, open-ended, unsatisfiable, and zero-size cases."""
    # Valid range (bytes=0-49 on 100 bytes)
    assert _parse_range("bytes=0-49", 100) == (0, 49)

    # Suffix range (bytes=-50 on 100 bytes)
    assert _parse_range("bytes=-50", 100) == (50, 99)
    # Suffix range larger than representation clamps to start at 0
    assert _parse_range("bytes=-150", 100) == (0, 99)

    # Open-ended range (bytes=50- on 100 bytes)
    assert _parse_range("bytes=50-", 100) == (50, 99)
    assert _parse_range("bytes=0-", 100) == (0, 99)

    # Range with end beyond size clamps to size - 1
    assert _parse_range("bytes=50-200", 100) == (50, 99)

    # Inverted range (bytes=500-400 -> unsatisfiable)
    with pytest.raises(_RangeUnsatisfiable):
        _parse_range("bytes=500-400", 100)

    with pytest.raises(_RangeUnsatisfiable):
        _parse_range("bytes=10-5", 20)

    # Past-EOF start range (bytes=500-600 on 100-byte file -> unsatisfiable)
    with pytest.raises(_RangeUnsatisfiable):
        _parse_range("bytes=500-600", 100)

    with pytest.raises(_RangeUnsatisfiable):
        _parse_range("bytes=100-", 100)

    # Suffix range with n <= 0
    with pytest.raises(_RangeUnsatisfiable):
        _parse_range("bytes=-0", 100)

    # Zero-size resource range handling (all ranges unsatisfiable)
    with pytest.raises(_RangeUnsatisfiable):
        _parse_range("bytes=0-49", 0)

    with pytest.raises(_RangeUnsatisfiable):
        _parse_range("bytes=0-", 0)

    with pytest.raises(_RangeUnsatisfiable):
        _parse_range("bytes=-50", 0)

    # Malformed / unrecognized syntax (ignored per RFC 9110 §14.2 -> None)
    assert _parse_range("not-a-range", 100) is None
    assert _parse_range("bytes=", 100) is None
    assert _parse_range("bytes=-", 100) is None
    assert _parse_range("bytes=abc-def", 100) is None
    assert _parse_range("items=0-10", 100) is None
    assert _parse_range("bytes=-10-20", 100) is None


def test_serve_range_headers_and_responses() -> None:
    """_serve unit tests covering 206, 416, 200 responses on media and non-media."""
    dummy_100_bytes = b"x" * 100

    def make_req(range_val: str | None) -> Request:
        headers = [(b"range", range_val.encode())] if range_val else []
        return Request({"type": "http", "method": "GET", "headers": headers})

    # 1. Valid range on media -> 206 Partial Content
    resp = _serve(dummy_100_bytes, "video/webm", make_req("bytes=0-49"), zim_id=1)
    assert resp.status_code == 206
    assert resp.headers["Content-Range"] == "bytes 0-49/100"
    assert resp.headers["Content-Length"] == "50"
    assert resp.body == b"x" * 50

    # 2. Suffix range on media -> 206 Partial Content
    resp = _serve(dummy_100_bytes, "video/webm", make_req("bytes=-50"), zim_id=1)
    assert resp.status_code == 206
    assert resp.headers["Content-Range"] == "bytes 50-99/100"
    assert resp.headers["Content-Length"] == "50"
    assert resp.body == b"x" * 50

    # 3. Open-ended range on media -> 206 Partial Content
    resp = _serve(dummy_100_bytes, "video/webm", make_req("bytes=50-"), zim_id=1)
    assert resp.status_code == 206
    assert resp.headers["Content-Range"] == "bytes 50-99/100"
    assert resp.headers["Content-Length"] == "50"
    assert resp.body == b"x" * 50

    # 4. Inverted range -> 416 Range Not Satisfiable
    resp = _serve(dummy_100_bytes, "video/webm", make_req("bytes=500-400"), zim_id=1)
    assert resp.status_code == 416
    assert resp.headers["Content-Range"] == "bytes */100"
    assert resp.body == b""

    # 5. Past-EOF range -> 416 Range Not Satisfiable
    resp = _serve(dummy_100_bytes, "video/webm", make_req("bytes=500-600"), zim_id=1)
    assert resp.status_code == 416
    assert resp.headers["Content-Range"] == "bytes */100"
    assert resp.body == b""

    # 6. Zero-size resource range -> 416 Range Not Satisfiable
    resp = _serve(b"", "video/webm", make_req("bytes=0-49"), zim_id=1)
    assert resp.status_code == 416
    assert resp.headers["Content-Range"] == "bytes */0"
    assert resp.body == b""

    # 7. Malformed range on media -> ignored -> 200 OK
    resp = _serve(dummy_100_bytes, "video/webm", make_req("bytes=malformed"), zim_id=1)
    assert resp.status_code == 200
    assert "Content-Range" not in resp.headers
    assert resp.body == dummy_100_bytes

    # 8. Range request on non-media (e.g. text/html) -> ignored -> 200 OK
    resp = _serve(b"<html></html>", "text/html", make_req("bytes=0-4"), zim_id=1)
    assert resp.status_code == 200
    assert "Content-Range" not in resp.headers
    assert resp.body == b"<html></html>"


async def test_passthrough_media_range_endpoint(
    tmp_path: Path,
) -> None:
    """HTTP endpoint test for Range handling over a real media ZIM fixture."""
    import os

    from fixtures.tiny_media_zim import VIDEO_PATH, build_tiny_media_zim
    from vesta.main import create_app

    zims_dir = tmp_path / "zims"
    zims_dir.mkdir(parents=True, exist_ok=True)
    build_tiny_media_zim(zims_dir / "tiny_media.zim")
    os.environ["data.dir"] = str(tmp_path)
    os.environ["inference.llm.model"] = ""
    try:
        app = create_app()
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/zims")
                zim_id = int(resp.json()["archives"][0]["id"])
                video_url = f"/api/zim/{zim_id}/{VIDEO_PATH}"

                # 1. Valid range: bytes=0-3
                r206 = await client.get(video_url, headers={"Range": "bytes=0-3"})
                assert r206.status_code == 206
                assert r206.headers["content-range"] == "bytes 0-3/15"
                assert r206.headers["content-length"] == "4"
                assert r206.content == b"\x1a\x45\xdf\xa3"

                # 2. Suffix range: bytes=-4
                r_suffix = await client.get(video_url, headers={"Range": "bytes=-4"})
                assert r_suffix.status_code == 206
                assert r_suffix.headers["content-range"] == "bytes 11-14/15"
                assert r_suffix.headers["content-length"] == "4"
                assert r_suffix.content == b"tial"

                # 3. Open-ended range: bytes=10-
                r_open = await client.get(video_url, headers={"Range": "bytes=10-"})
                assert r_open.status_code == 206
                assert r_open.headers["content-range"] == "bytes 10-14/15"
                assert r_open.headers["content-length"] == "5"
                assert r_open.content == b"rtial"

                # 4. Inverted range: bytes=500-400 -> 416
                r416_inv = await client.get(video_url, headers={"Range": "bytes=500-400"})
                assert r416_inv.status_code == 416
                assert r416_inv.headers["content-range"] == "bytes */15"

                # 5. Past-EOF range: bytes=500-600 -> 416
                r416_eof = await client.get(video_url, headers={"Range": "bytes=500-600"})
                assert r416_eof.status_code == 416
                assert r416_eof.headers["content-range"] == "bytes */15"
    finally:
        os.environ.pop("data.dir", None)
        os.environ.pop("inference.llm.model", None)
