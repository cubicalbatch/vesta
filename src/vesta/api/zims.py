"""Archive & article API.

* ``GET /api/zims`` — installed archives (from the ``zims`` table).
* ``PATCH /api/zims/{id}`` — enable/disable, corpus label.
* ``DELETE /api/zims/{id}`` — remove (cascades articles + aliases).
* ``POST /api/zims/scan`` — pick up files dropped in ``./data/zims``.
* ``GET /api/article/{zim}/{path:path}`` — extracted text + sections (JSON).
* ``GET /api/zims/{id}/random`` — a random article, same shape as the above.
* ``GET /api/zims/{id}/samples`` — a deduplicated set of random articles.

DTOs (Pydantic) live here; the mapping from the frozen dataclasses in ``zim/`` is
explicit and stays in this package — DTOs are not domain
objects. This router never decides ranking; it exposes raw material only.
"""

from __future__ import annotations

import asyncio
import urllib.parse
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from vesta.api.state import AppState, app_state
from vesta.jobs.types import JobRecord
from vesta.zim.reader import EntryNotFound
from vesta.zim.types import EntryPath, ExtractedArticle

router = APIRouter(tags=["archives"])

_NON_TERMINAL_JOB_STATUSES = frozenset({"queued", "running", "paused"})

# Per-zim_id locks guarding trigger_index's check-then-act (read pending jobs,
# then submit). Without this, two requests that both arrive before either has
# committed its submit() both observe "no pending job" and both enqueue —
# a real, reproducible race (two concurrent curl POSTs against the same
# zim_id reliably produced two jobs), not just a theoretical one. A
# process-local dict is enough: this is a single-process uvicorn app, so a
# module-level asyncio.Lock per zim_id serializes the check+submit within
# this process without needing a DB-level lock.
_index_trigger_locks: dict[int, asyncio.Lock] = {}


def _index_trigger_lock(zim_id: int) -> asyncio.Lock:
    lock = _index_trigger_locks.get(zim_id)
    if lock is None:
        lock = asyncio.Lock()
        _index_trigger_locks[zim_id] = lock
    return lock


def _archive_or_404(state: AppState, zim_id: int) -> Any:
    """Resolve one registered archive by id — the single copy of the guard
    every per-archive endpoint shares. A missing registry is the standard
    not-ready 503; an unknown id is the standard 404."""
    if state.registry is None:
        raise HTTPException(status_code=503, detail="archive registry not ready")
    try:
        return state.registry.get(zim_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="archive not found") from exc


class ArchiveOut(BaseModel):
    id: int
    uuid: str
    name: str | None = None
    title: str | None = None
    language: str | None = None
    flavour: str | None = None
    file_size: int | None = None
    article_count: int  # from Counter['text/html'] only — never all entries
    has_fulltext_index: bool  # PROBED at runtime, not trusted from metadata
    corpus_label: str | None = None
    kind: str = "articles"  # "articles" | "media" | "spa" (0007_zim_kind.sql)
    scraper: str | None = None  # raw Scraper metadata (transparency only)
    tags: str | None = None  # raw Tags metadata (transparency only)
    enabled: bool
    status: str
    index_depth: int = 0  # 08: 0..3 semantic-index depth
    index_status: str = "none"  # 08: none|running|paused|complete|stale|error
    embedding_model: str | None = None  # 08: the embedder an index was built with


class ArchivePatch(BaseModel):
    enabled: bool | None = None
    corpus_label: str | None = None


class ScanResultOut(BaseModel):
    added: list[int]
    updated: list[int]
    missing: list[int]
    total: int


class IndexTriggerRequest(BaseModel):
    depth: int = 1


class SectionOut(BaseModel):
    heading_path: list[str]
    level: int
    char_start: int
    char_end: int


class MediaOut(BaseModel):
    """Playable-media assets for one entry (media/SPA ZIMs, 0008).

    The frontend renders a native ``<video poster=… src=…>`` from this; ``None``
    fields mean the asset wasn't in the manifest. Paths are ZIM-relative (served
    via the path-preserving ``/api/zim/{id}/...`` reader route)."""

    video_path: str | None = None
    poster_path: str | None = None
    duration: int | None = None


class DocumentOut(BaseModel):
    """One browsable document for a documents-kind ZIM (nautiluszim, 0013).

    ``url`` is the path-preserving reader URL (``/api/zim/{zim_id}/{doc_path}``)
    so the frontend opens the PDF natively (Chromium renders ``application/pdf``
    without scripts). ``None`` fields mean the manifest omitted the field."""

    doc_path: str
    title: str | None = None
    description: str | None = None
    author: str | None = None
    doc_mime: str
    url: str


class ArticleOut(BaseModel):
    zim_id: int
    path: str
    title: str
    text: str
    sections: list[SectionOut]
    flags: int
    media: MediaOut | None = None  # 0008: present for media-ZIM entries with assets
    document: DocumentOut | None = None  # 0013: present for documents-kind entry hits


def _document_out(document: object | None) -> DocumentOut | None:
    """Map a domain :class:`DocumentRef` (or None) to the wire DTO."""
    if document is None:
        return None
    return DocumentOut(
        doc_path=getattr(document, "doc_path", ""),
        title=getattr(document, "title", None),
        description=getattr(document, "description", None),
        author=getattr(document, "author", None),
        doc_mime=getattr(document, "doc_mime", ""),
        url=getattr(document, "url", ""),
    )


def _media_out(media: object | None) -> MediaOut | None:
    """Map a domain :class:`MediaRef` (or None) to the wire DTO."""
    if media is None:
        return None
    return MediaOut(
        video_path=getattr(media, "video_path", None),
        poster_path=getattr(media, "poster_path", None),
        duration=getattr(media, "duration", None),
    )


def _article_out(
    zim_id: int,
    article: ExtractedArticle,
    *,
    media: object | None = None,
    document: object | None = None,
) -> dict[str, object]:
    """Serialize an extracted article to the ``ArticleOut`` wire shape.

    Shared by the article, random, and samples routes so the DTO mapping lives
    in one place — DTOs are not domain objects."""
    return ArticleOut(
        zim_id=zim_id,
        path=article.path,
        title=article.title,
        text=article.text,
        sections=[
            SectionOut(
                heading_path=list(s.heading_path),
                level=s.level,
                char_start=s.char_start,
                char_end=s.char_end,
            )
            for s in article.sections
        ],
        flags=int(article.flags),
        media=_media_out(media),
        document=_document_out(document),
    ).model_dump()


def _archive_out(row: dict[str, Any]) -> ArchiveOut:
    return ArchiveOut(
        id=int(row["id"]),
        uuid=row["uuid"],
        name=row["name"],
        title=row["title"],
        language=row["language"],
        flavour=row["flavour"],
        file_size=row["file_size"],
        article_count=int(row["article_count"]),
        has_fulltext_index=bool(row["has_fulltext_index"]),
        corpus_label=row["corpus_label"],
        kind=str(row.get("kind") or "articles"),
        scraper=row.get("scraper"),
        tags=row.get("tags"),
        enabled=bool(row["enabled"]),
        status=row["status"],
        index_depth=int(row.get("index_depth") or 0),
        index_status=str(row.get("index_status") or "none"),
        embedding_model=row.get("embedding_model"),
    )


@router.get("/api/zims")
async def list_zims(state: AppState = Depends(app_state)) -> dict[str, object]:
    async with (
        state.db.read() as conn,
        conn.execute("SELECT * FROM zims ORDER BY id") as cur,
    ):
        rows = [dict(r) for r in await cur.fetchall()]
    return {"archives": [_archive_out(r).model_dump() for r in rows]}


@router.post("/api/zims/scan")
async def scan_zims(state: AppState = Depends(app_state)) -> dict[str, object]:
    """Pick up ``*.zim`` files dropped in the ZIM directory."""
    if state.registry is None:
        raise HTTPException(status_code=503, detail="archive registry not ready")
    result = await state.registry.rescan()
    out = ScanResultOut(
        added=list(result.added),
        updated=list(result.updated),
        missing=list(result.missing),
        total=result.total,
    )
    return out.model_dump()


@router.patch("/api/zims/{zim_id}")
async def patch_zim(
    zim_id: int, patch: ArchivePatch, state: AppState = Depends(app_state)
) -> dict[str, object]:
    if state.registry is None:
        raise HTTPException(status_code=503, detail="archive registry not ready")
    if patch.enabled is not None:
        ok = await state.registry.set_enabled(zim_id, patch.enabled)
        if not ok:
            raise HTTPException(status_code=404, detail="archive not found")
        # Recompute the capability flag: a disabled archive no longer counts
        # toward VECTORS, so dense profiles must degrade until it returns.
        from vesta.index import reseed_indexed_state

        await reseed_indexed_state(state.db)
    if patch.corpus_label is not None:
        ok = await state.registry.set_corpus_label(zim_id, patch.corpus_label)
        if not ok:
            raise HTTPException(status_code=404, detail="archive not found")
    return {"id": zim_id, "ok": True}


@router.delete("/api/zims/{zim_id}")
async def delete_zim(
    zim_id: int,
    state: AppState = Depends(app_state),
    keep_file: bool = Query(
        False,
        description="Keep the .zim file on disk. Default (false) removes the file "
        "AND every DB reference — articles, aliases, chunks, index vectors — so a "
        "deleted archive leaves no trace.",
    ),
) -> dict[str, object]:
    """Remove an archive: every DB reference is cascade-deleted (the
    user's "delete the file AND all references" requirement), with the file
    removal optional via ``keep_file``.

    The DB cascade is total regardless: ``zims`` row → articles/aliases/chunks/
    index_meta (FK CASCADE) + vectors (registry on_remove callback, since vec0
    isn't FK-cascaded). ``keep_file=true`` only spares the bytes on disk.
    """
    if state.registry is None:
        raise HTTPException(status_code=503, detail="archive registry not ready")
    ok = await state.registry.remove(zim_id, delete_file=not keep_file)
    if not ok:
        raise HTTPException(status_code=404, detail="archive not found")
    # The cascade removed this archive's vectors; recompute the capability
    # flag so VECTORS turns off when this was the last indexed archive.
    from vesta.index import reseed_indexed_state

    await reseed_indexed_state(state.db)
    return {"id": zim_id, "ok": True, "file_removed": not keep_file}


@router.get("/api/article/{zim_id}/{path:path}")
async def get_article(
    zim_id: int, path: str, state: AppState = Depends(app_state)
) -> dict[str, object]:
    """Extracted text + section structure for one article."""
    archive = _archive_or_404(state, zim_id)
    decoded = urllib.parse.unquote(path)
    try:
        article = await archive.extract(decoded)
    except EntryNotFound as exc:
        raise HTTPException(status_code=404, detail=f"not found: {decoded}") from exc
    media = await _media_for(state, zim_id, decoded)
    document = await _document_for(state, zim_id, decoded)
    return _article_out(zim_id, article, media=media, document=document)


@router.get("/api/zims/{zim_id}/random")
async def get_random_article(
    zim_id: int, state: AppState = Depends(app_state)
) -> dict[str, object]:
    """A random article's extracted text + sections — same ``ArticleOut`` shape
    as ``GET /api/article/{zim}/{path}``, for the archive-browse page's
    "Random article" action. A direct libzim read via ``Archive.random()``, so
    it works at any ``index_depth`` (including 0) — independent of the
    depth-based vector index.
    """
    archive = _archive_or_404(state, zim_id)
    path = await archive.random()
    try:
        article = await archive.extract(path)
    except EntryNotFound as exc:
        raise HTTPException(status_code=404, detail=f"not found: {path}") from exc
    media = await _media_for(state, zim_id, path)
    document = await _document_for(state, zim_id, path)
    return _article_out(zim_id, article, media=media, document=document)


async def _media_for(state: AppState, zim_id: int, path: EntryPath) -> object | None:
    """Fetch the media manifest row for one entry, or ``None``.

    ``zim/media.py`` owns ``article_media``; this is the api-layer's thin lookup
    so a card/Reader payload carries playable assets inline. Tolerates a missing
    table (pre-0008 DB) by returning ``None`` — media enrichment degrades, never
    breaks the article route.
    """
    from vesta.zim.media import fetch_media_for_paths

    try:
        media_map = await fetch_media_for_paths(state.db, zim_id, [path])
    except Exception:
        return None
    return media_map.get(path)


async def _document_for(state: AppState, zim_id: int, path: EntryPath) -> object | None:
    """Fetch the document manifest ref for one entry, or ``None``.

    ``zim/documents.py`` owns ``article_documents``; this is the api-layer's
    thin lookup so a documents-kind entry card carries the manifest
    title/author/description + a resolvable reader ``url``. Tolerates a missing
    table (pre-0013 DB) by returning ``None`` — enrichment degrades, never
    breaks the article route.
    """
    from vesta.zim.documents import fetch_document_refs_for_paths

    try:
        docs = await fetch_document_refs_for_paths(state.db, zim_id, [path])
    except Exception:
        return None
    return docs.get(path)


@router.get("/api/zims/{zim_id}/samples")
async def get_random_samples(
    zim_id: int,
    count: int = Query(6, ge=1, le=24),
    state: AppState = Depends(app_state),
) -> list[dict[str, object]]:
    """A deduplicated set of ``count`` random articles (``ArticleOut`` shape),
    for the archive-browse page's "discover" card grid. Like ``/random`` this is
    a direct libzim read, independent of ``index_depth``.

    Entries are deduped by path and the loop is attempt-capped, so a small
    archive simply returns fewer than ``count`` items rather than spinning
    forever; empty-text entries (redirects that slipped through, soft
    redirects) are skipped so every card has a snippet to show.
    """
    archive = _archive_or_404(state, zim_id)
    seen: set[EntryPath] = set()
    cards: list[tuple[EntryPath, ExtractedArticle, object | None]] = []
    attempts = 0
    cap = max(count * 8, 16)
    while len(cards) < count and attempts < cap:
        attempts += 1
        path = await archive.random()
        if path in seen:
            continue
        seen.add(path)
        try:
            article = await archive.extract(path)
        except EntryNotFound:
            continue
        if article.text.strip():
            # Real article body — show it as a text card.
            cards.append((path, article, None))
            continue
        # Empty body (a redirect stub). For media ZIMs the stub still has a
        # poster + title via the manifest — keep it so the browse grid shows
        # videos. For article ZIMs (no manifest) it's skipped as before.
        media = await _media_for(state, zim_id, path)
        if media is None:
            continue
        cards.append((path, article, media))
    # Batch document enrichment (0013): one path-keyed lookup across all card
    # paths so documents-kind ZIMs carry the manifest title/author/description,
    # not a bare reader URL + per-card DB round-trip.
    from vesta.zim.documents import fetch_document_refs_for_paths
    from vesta.zim.types import DocumentRef

    try:
        doc_map: dict[EntryPath, DocumentRef] = await fetch_document_refs_for_paths(
            state.db, zim_id, [p for p, _a, _m in cards]
        )
    except Exception:
        doc_map = {}
    out: list[dict[str, object]] = [
        _article_out(zim_id, article, media=media, document=doc_map.get(path))
        for path, article, media in cards
    ]
    return out


@router.get("/api/zims/{zim_id}/documents")
async def list_documents(zim_id: int, state: AppState = Depends(app_state)) -> dict[str, object]:
    """The archive's browsable document catalog (nautiluszim, 0013).

    Returns ``{"documents": [DocumentOut…]}`` where each ``url`` is the
    path-preserving reader URL (``/api/zim/{zim_id}/{doc_path}``) — the browser
    renders ``application/pdf`` natively. Non-documents archives have no
    manifest rows, so they return an empty list; an unknown archive is a 404.
    """
    _archive_or_404(state, zim_id)
    from vesta.zim.documents import fetch_document_refs

    try:
        docs = await fetch_document_refs(state.db, zim_id)
    except Exception:
        docs = []
    return {
        "documents": [
            DocumentOut(
                doc_path=d.doc_path,
                title=d.title,
                description=d.description,
                author=d.author,
                doc_mime=d.doc_mime,
                url=d.url,
            ).model_dump()
            for d in docs
        ]
    }


async def _find_pending_index_job(state: AppState, zim_id: int) -> JobRecord | None:
    """The most recent non-terminal ``index_zim`` job targeting ``zim_id``, if any.

    ``list_jobs`` is the runner's public read primitive (id DESC, capped at its
    default limit) — good enough for a single-user appliance where at most one
    build per archive is ever queued at a time.
    """
    for record in await state.runner.list_jobs():
        if (
            record.type == "index_zim"
            and record.target == str(zim_id)
            and record.status in _NON_TERMINAL_JOB_STATUSES
        ):
            return record
    return None


@router.post("/api/zims/{zim_id}/index")
async def trigger_index(
    zim_id: int, body: IndexTriggerRequest, state: AppState = Depends(app_state)
) -> dict[str, object]:
    """Enqueue a resumable index build for one archive at one depth (08).

    Idempotent by ``zim_id``: a double-click, a repeated click during the
    queued-but-not-yet-running window, or two browser tabs must not queue two
    builds for the same archive (Workstream A, Issue 1 — the frontend already
    hides the trigger button once it sees a pending job, but this is the actual
    safety net since it can't see the *other* tab's click). If a non-terminal
    ``index_zim`` job already targets this zim, its id (and the depth it's
    actually running at) is returned instead of enqueuing a duplicate.
    """
    if state.registry is None or state.runner is None:
        raise HTTPException(status_code=503, detail="registry/runner not ready")
    _archive_or_404(state, zim_id)
    depth = body.depth
    if depth < 1 or depth > 3:
        raise HTTPException(status_code=400, detail="depth must be 1..3")

    # Hold the per-zim_id lock across the whole check-then-act: without it,
    # two requests can both await _find_pending_index_job (a DB read) before
    # either reaches runner.submit (a DB write), and both see "no pending
    # job" — a real race, not a hypothetical one.
    async with _index_trigger_lock(zim_id):
        existing = await _find_pending_index_job(state, zim_id)
        if existing is not None:
            existing_depth = existing.params.get("depth")
            return {
                "zim_id": zim_id,
                "depth": existing_depth if isinstance(existing_depth, int) else depth,
                "job_id": existing.id,
            }

        # Cross-process guard (AUDIT_0822 M7): a detached `vesta index` build
        # creates no job row at all, so the check above cannot see it. Refuse
        # with a 409 naming the holder instead of enqueueing a build destined
        # to fail its own lease claim. Dead/stale leases don't block — the job
        # takes them over when it runs.
        from vesta.index.leases import active_holder

        holder = await active_holder(state.db, zim_id)
        if holder is not None:
            raise HTTPException(
                status_code=409,
                detail=f"an index build is already running for this archive ({holder})",
            )

        job_id = await state.runner.submit(
            "index_zim",
            target=str(zim_id),
            params={"zim_id": zim_id, "depth": depth, "owner": "server"},
        )
    return {"zim_id": zim_id, "depth": depth, "job_id": job_id}


__all__ = ["router"]
