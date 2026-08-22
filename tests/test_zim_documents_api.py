"""Documents API surface (nautiluszim document-library support).

``GET /api/zims/{zim_id}/documents`` returns the archive's manifest-backed
document catalog as ``{documents: [DocumentOut…]}`` where each ``url`` is the
path-preserving reader URL; article/random/samples/search enrich a documents-kind
hit with a ``document`` dict so it reads by its manifest title/author/description
instead of a bare reader URL.

Two fixtures:
* ``app_client_with_documents`` — the tiny ZIM registered with one
  ``article_documents`` row for a real article path (deterministic, no real
  nautilus archive needed). Exercises the endpoint DTO + ArticleOut/search-card
  enrichment + empty-list for a registered non-documents archive.
* a gated real-archive test — the water archive (kind='documents', 7 PDFs) via a
  full app whose data dir symlinks the real ``.zim`` files, proving the endpoint
  returns 7 docs whose reader ``url`` serves ``application/pdf``, and that a
  normal article ZIM returns an empty list.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from fixtures.tiny_zim import LONG_ARTICLE_PATH, build_tiny_zim
from vesta.db.connection import Database
from vesta.db.migrations import run_migrations
from vesta.main import create_app

DATA = Path("data/zims")
WATER = DATA / "zimgit-water_en_2024-08.zim"
ARTICLE = DATA / "devdocs_en_liquid_2026-07.zim"  # a small normal article ZIM

water = pytest.mark.skipif(not WATER.exists(), reason=f"{WATER} not present")

# The seeded document row: matches a real tiny-zim article path.
_DOC_PATH = LONG_ARTICLE_PATH  # "A/Albert_Einstein"
_DOC_TITLE = "Seeded Document Title"
_DOC_DESC = "A seeded document description"
_DOC_AUTHOR = "A Seed Author"


@pytest_asyncio.fixture
async def app_client_with_documents(
    tmp_path: Path,
) -> AsyncIterator[tuple[httpx.AsyncClient, int]]:
    """The tiny ZIM registered with one ``article_documents`` row for a real
    article path — end-to-end documents surface without the real nautilus archive.

    The zims row is pre-seeded keyed by the tiny ZIM's UUID so the lifespan scan
    refreshes it (keeping id=1 and the seeded article_documents row) rather than
    inserting a fresh row. The tiny ZIM is ``kind='articles'`` (no database.js),
    so only the seeded document row populates ``/documents``.
    """
    zims_dir = tmp_path / "zims"
    zims_dir.mkdir(parents=True, exist_ok=True)
    zim_path = build_tiny_zim(zims_dir / "tiny.zim")

    from libzim.reader import Archive as LibzimArchive

    uuid = str(LibzimArchive(str(zim_path)).uuid)
    db = Database(str(tmp_path / "vesta.db"), busy_timeout_ms=1000)
    await db.start()
    async with db.write() as conn:
        await run_migrations(conn)
        await conn.execute(
            "INSERT INTO zims(id, uuid, filename, path, name, title, kind, enabled, status) "
            "VALUES(1, ?, 'tiny.zim', ?, 'tiny', 'Tiny', 'articles', 1, 'known')",
            (uuid, str(zim_path)),
        )
        await conn.execute(
            "INSERT INTO article_documents(zim_id, doc_path, title, description, author, doc_mime) "
            "VALUES(1, ?, ?, ?, ?, ?)",
            (_DOC_PATH, _DOC_TITLE, _DOC_DESC, _DOC_AUTHOR, "application/pdf"),
        )
    await db.stop()

    os.environ["data.dir"] = str(tmp_path)
    os.environ["inference.llm.model"] = ""
    try:
        app = create_app()
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                yield client, 1
    finally:
        os.environ.pop("data.dir", None)
        os.environ.pop("inference.llm.model", None)


# --- GET /api/zims/{zim_id}/documents ----------------------------------------


async def test_documents_endpoint_returns_seeded_catalog(
    app_client_with_documents: tuple[httpx.AsyncClient, int],
) -> None:
    client, zim_id = app_client_with_documents
    resp = await client.get(f"/api/zims/{zim_id}/documents")
    assert resp.status_code == 200
    body = resp.json()
    assert body["documents"] == [
        {
            "doc_path": _DOC_PATH,
            "title": _DOC_TITLE,
            "description": _DOC_DESC,
            "author": _DOC_AUTHOR,
            "doc_mime": "application/pdf",
            "url": f"/api/zim/{zim_id}/{_DOC_PATH}",
        }
    ]


# --- document enrichment on ArticleOut / search cards -------------------------


async def test_article_and_search_routes_carry_document_enrichment(
    app_client_with_documents: tuple[httpx.AsyncClient, int],
) -> None:
    """An ArticleOut and search hit whose path matches a document carries the
    manifest title/author/description + a resolvable reader url."""
    client, zim_id = app_client_with_documents

    # Article route enrichment.
    resp = await client.get(f"/api/article/{zim_id}/{_DOC_PATH}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["document"] is not None
    assert body["document"]["doc_path"] == _DOC_PATH
    assert body["document"]["title"] == _DOC_TITLE
    assert body["document"]["description"] == _DOC_DESC
    assert body["document"]["author"] == _DOC_AUTHOR
    assert body["document"]["doc_mime"] == "application/pdf"
    assert body["document"]["url"] == f"/api/zim/{zim_id}/{_DOC_PATH}"

    # Search card enrichment.
    s_resp = await client.get("/api/search", params={"q": "Einstein", "profile": "lexical"})
    assert s_resp.status_code == 200
    cards = s_resp.json()["cards"]
    doc_cards = [c for c in cards if c.get("document") is not None]
    assert doc_cards, f"expected a document-enriched card, got {len(cards)} cards"
    for card in doc_cards:
        assert card["path"] == _DOC_PATH
        assert card["title"] == _DOC_TITLE  # display title = manifest title, not the path
        assert card["document"]["title"] == _DOC_TITLE
        assert card["document"]["author"] == _DOC_AUTHOR
        assert card["document"]["description"] == _DOC_DESC
        assert card["document"]["url"] == f"/api/zim/{zim_id}/{_DOC_PATH}"


# --- gated real-archive integration ------------------------------------------


@water
@pytest.mark.asyncio
async def test_water_documents_endpoint_serves_seven_pdfs(tmp_path: Path) -> None:
    """The real water archive (kind='documents', 7 PDFs) returns all 7 documents
    via ``/documents`` with manifest titles and a resolvable reader ``url`` that
    serves ``application/pdf``; a normal article ZIM returns an empty list."""
    zims_dir = tmp_path / "zims"
    zims_dir.mkdir(parents=True, exist_ok=True)
    # Symlink (not copy) the real archives — libzim opens by path, and the
    # 20 MB water ZIM needn't be duplicated per run.
    os.symlink(WATER.resolve(), zims_dir / WATER.name)
    if ARTICLE.exists():
        os.symlink(ARTICLE.resolve(), zims_dir / ARTICLE.name)

    os.environ["data.dir"] = str(tmp_path)
    os.environ["inference.llm.model"] = ""
    try:
        app = create_app()
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                archives = (await client.get("/api/zims")).json()["archives"]
                water_id = next(a["id"] for a in archives if "zimgit-water" in (a["name"] or ""))

                resp = await client.get(f"/api/zims/{water_id}/documents")
                assert resp.status_code == 200
                docs = resp.json()["documents"]
                assert len(docs) == 7
                titles = {d["title"] for d in docs}
                assert "Distillation For Home Water Treatment" in titles
                assert "Water Treatment" in titles
                for d in docs:
                    assert d["doc_mime"] == "application/pdf"
                    assert d["url"] == f"/api/zim/{water_id}/{d['doc_path']}"

                # A real document url serves application/pdf through the reader.
                pdf = next(d for d in docs if "Water (1).pdf" in d["doc_path"])
                served = await client.get(pdf["url"], follow_redirects=False)
                assert served.status_code == 200
                assert served.headers["content-type"].startswith("application/pdf")

                if ARTICLE.exists():
                    devdocs_id = next(a["id"] for a in archives if "devdocs" in (a["name"] or ""))
                    non_docs = await client.get(f"/api/zims/{devdocs_id}/documents")
                    assert non_docs.status_code == 200
                    assert non_docs.json() == {"documents": []}
    finally:
        os.environ.pop("data.dir", None)
        os.environ.pop("inference.llm.model", None)
