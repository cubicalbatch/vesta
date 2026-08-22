"""Library & catalog API.

The surface for getting archives onto the box without touching the
filesystem:

* ``GET /api/catalog`` — server-side FTS5-filtered catalog listing (q, language,
  recommended filters). Never ships the full ~3 000-row list to the client.
* ``POST /api/catalog/refresh`` — re-cache the OPDS feed; a job so its progress
  and any outage lands on the jobs panel.
* ``GET /api/catalog/{id}`` — one catalog entry.
* ``POST /api/zims/download`` — enqueue a resumable checksummed download.
* ``GET /api/catalog/state`` — cheap cache summary for the library page header.

Catalog outage is graceful throughout: a failed refresh surfaces on
the job row but the cached list keeps serving; an empty cache yields an empty
listing, never a 500. The catalog is never a hard dependency for anything already
installed.

DTOs (Pydantic) live here; mapping from ``catalog/`` frozen dataclasses is
explicit — DTOs are not domain objects.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from vesta.api.state import AppState, app_state
from vesta.catalog import CATALOG_ENABLED
from vesta.catalog.curated import curated_entries, curated_warning_for
from vesta.catalog.opds import (
    CatalogFilter,
    catalog_languages,
    catalog_state,
    get_entry,
    search_catalog,
)

router = APIRouter(tags=["catalog"])


class CatalogEntryOut(BaseModel):
    id: str
    name: str
    title: str
    description: str
    language: str
    flavour: str
    tags: str
    size_bytes: int
    article_count: int  # catalog figure — redirect-inflated, displayed as approx
    url: str
    illustration_url: str | None = None
    zim_date: str | None = None
    curated_rank: int | None = None
    curated_warning: str | None = None
    fetched_at: str | None = None


class CatalogListOut(BaseModel):
    entries: list[CatalogEntryOut]
    total: int
    available: bool
    fetched_at: str | None = None


class CuratedEntryOut(BaseModel):
    name: str
    rank: int
    size_note: str
    description: str = ""
    article_count: int = 0
    warning: str | None = None


class CatalogStateOut(BaseModel):
    count: int
    fetched_at: str | None = None
    available: bool


class CatalogLanguageOut(BaseModel):
    code: str
    count: int


class DownloadRequest(BaseModel):
    # The catalog entry id (preferred) OR a raw acquisition URL (manual entry —
    # the catalog is never the only path: "degrade to manual URL").
    entry_id: str | None = None
    url: str | None = None
    name: str | None = None


class DownloadResponse(BaseModel):
    job_id: int
    job_type: str
    target: str | None = None


class RefreshResponse(BaseModel):
    job_id: int
    job_type: str


def _entry_out(row: dict[str, Any]) -> CatalogEntryOut:
    name = str(row.get("name") or "")
    flavour = str(row.get("flavour") or "")
    return CatalogEntryOut(
        id=str(row["id"]),
        name=name,
        title=str(row.get("title") or name),
        description=str(row.get("description") or ""),
        language=str(row.get("language") or ""),
        flavour=flavour,
        tags=str(row.get("tags") or ""),
        size_bytes=int(row.get("size_bytes") or 0),
        article_count=int(row.get("article_count") or 0),
        url=str(row.get("url") or ""),
        illustration_url=row.get("illustration_url"),
        zim_date=row.get("zim_date"),
        curated_rank=row.get("curated_rank"),
        curated_warning=curated_warning_for(name, flavour),
        fetched_at=row.get("fetched_at"),
    )


def _require_enabled(state: AppState) -> None:
    from vesta import config

    if not bool(config.get(CATALOG_ENABLED)):
        raise HTTPException(status_code=503, detail="catalog disabled (catalog.enabled=false)")


@router.get("/api/catalog")
async def list_catalog(
    *,
    state: AppState = Depends(app_state),
    q: str = Query("", description="FTS5 search over name/title/description/tags"),
    language: str = Query("", description="ISO 639-3 code, e.g. 'eng'"),
    recommended: bool = Query(False, description="curated entries only"),
    max_size: int | None = Query(
        None, ge=0, description="exclude archives larger than this many bytes"
    ),
    sort: str = Query(
        "default",
        description="default|relevance|size_asc|size_desc|date_asc|date_desc",
    ),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict[str, object]:
    """Server-side filtered catalog listing — never ship the full list."""
    _require_enabled(state)
    flt = CatalogFilter(
        query=q.strip(),
        language=language.strip(),
        recommended_only=bool(recommended),
        max_size_bytes=max_size,
        sort=sort.strip(),
        limit=limit,
        offset=offset,
    )
    result = await search_catalog(state.db, flt)
    info = await catalog_state(state.db)
    out = CatalogListOut(
        entries=[_entry_out(r) for r in result.entries],
        total=result.total,
        available=bool(info["available"]),
        fetched_at=info["fetched_at"],
    )
    return out.model_dump()


@router.get("/api/catalog/curated")
async def list_curated(state: AppState = Depends(app_state)) -> dict[str, object]:
    """The shipped starter list (data in the repo — works with no network).

    Always available even when the live catalog is not (a catalog
    outage must not degrade first-run). Returned ordered by ``rank``.
    """
    _require_enabled(state)
    entries = [
        CuratedEntryOut(
            name=e.key,
            rank=e.rank,
            size_note=e.size_note,
            description=e.description,
            article_count=e.article_count,
            warning=e.warning,
        )
        for e in curated_entries()
    ]
    return {"entries": [e.model_dump() for e in entries]}


@router.get("/api/catalog/state", response_model=CatalogStateOut)
async def get_catalog_state(state: AppState = Depends(app_state)) -> dict[str, object]:
    _require_enabled(state)
    info = await catalog_state(state.db)
    return CatalogStateOut(
        count=int(info["count"]),
        fetched_at=info["fetched_at"],
        available=bool(info["available"]),
    ).model_dump()


@router.get("/api/catalog/languages")
async def list_catalog_languages(
    state: AppState = Depends(app_state),
) -> dict[str, object]:
    """Distinct languages with counts, for the browse picker facets.

    Aggregated by comma-member so a multilingual archive counts once per
    language it ships. Empty on an empty cache (offline first boot) — never a 500.
    """
    _require_enabled(state)
    langs = await catalog_languages(state.db)
    return {"languages": [CatalogLanguageOut(**lang).model_dump() for lang in langs]}


@router.get("/api/catalog/{entry_id}")
async def get_catalog_entry(
    entry_id: str, state: AppState = Depends(app_state)
) -> dict[str, object]:
    _require_enabled(state)
    row = await get_entry(state.db, entry_id)
    if row is None:
        raise HTTPException(status_code=404, detail="catalog entry not found")
    return _entry_out(row).model_dump()


@router.post("/api/catalog/refresh", response_model=RefreshResponse)
async def refresh_catalog(state: AppState = Depends(app_state)) -> dict[str, object]:
    """Re-cache the OPDS feed as a job (progress + outage land on the jobs panel).

    The refresh only replaces the cache after a clean parse, so a
    failure leaves the existing list serving. Returns the job id immediately.
    """
    _require_enabled(state)
    if state.runner is None:
        raise HTTPException(status_code=503, detail="job runner not ready")
    job_id = await state.runner.submit("refresh_catalog", target="catalog", params={})
    return RefreshResponse(job_id=job_id, job_type="refresh_catalog").model_dump()


@router.post("/api/zims/download", response_model=DownloadResponse)
async def download_zim(
    body: DownloadRequest, state: AppState = Depends(app_state)
) -> dict[str, object]:
    """Enqueue a resumable checksummed download for one catalog entry or raw URL.

    The catalog is never the only path: ``url`` + ``name`` support
    manual entry when the catalog is unavailable. ``entry_id`` resolves the
    cached entry's acquisition URL + name automatically.
    """
    _require_enabled(state)
    if state.runner is None:
        raise HTTPException(status_code=503, detail="job runner not ready")

    url = body.url
    name = body.name
    size = 0
    if body.entry_id:
        row = await get_entry(state.db, body.entry_id)
        if row is None:
            raise HTTPException(status_code=404, detail="catalog entry not found")
        url = url or str(row.get("url") or "")
        name = name or str(row.get("name") or "")
        size = int(row.get("size_bytes") or 0)
    if not url:
        raise HTTPException(status_code=400, detail="url or entry_id required")
    if not name:
        raise HTTPException(status_code=400, detail="name required (or supply entry_id)")

    params: dict[str, Any] = {"url": url, "name": name}
    if size:
        params["size"] = size

    job_id = await state.runner.submit("download_zim", target=name, params=params)
    return DownloadResponse(job_id=job_id, job_type="download_zim", target=name).model_dump()


__all__ = ["router"]
