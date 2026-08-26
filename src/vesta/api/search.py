"""Search API — ``GET /api/search`` — the sources-only fast path.

Runs the full retrieval pipeline with the active profile (or ``?profile=``
override) and returns source cards, a complete trace, and confidence signals.
No LLM — this is the fast path that works without any model configured.

SSE progress is optional; the pipeline is fast enough that JSON
response is fine.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from vesta import config as app_config
from vesta.api.answer import _parse_scope
from vesta.api.state import AppState, app_state
from vesta.api.zims import DocumentOut, MediaOut
from vesta.config.capabilities import compute_capabilities
from vesta.retrieval import (
    API_ALLOW_PROFILE_OVERRIDE,
    RETRIEVAL_ACTIVE_PROFILE,
    RETRIEVAL_MAX_ARCHIVES_CONCURRENT,
)
from vesta.retrieval.pipeline import Deps, NoCandidatesError, run_pipeline
from vesta.retrieval.profiles import (
    RetrievalProfile,
    resolve_profile_from_settings,
)
from vesta.vectors import get_store as get_vector_store
from vesta.zim import bind_registry  # noqa: F401 — ensures capability probe is registered

router = APIRouter(prefix="/api", tags=["search"])


class SourceCardResponse(BaseModel):
    """DTO for a source card in the API response."""

    zim_id: int
    path: str
    title: str
    snippet: str
    breadcrumb: str
    score: float | None
    source: str
    media: dict[str, Any] | None = (
        None  # 0008: {video_path, poster_path, duration} for media-ZIM cards
    )
    document: dict[str, Any] | None = (
        None  # 0013: {doc_path, title, description, author, doc_mime, url} for documents-kind cards
    )

    model_config = {"extra": "allow"}


class ConfidenceResponse(BaseModel):
    """DTO for confidence signals."""

    top_score: float | None = None
    score_dropoff: float | None = None
    density: float = 0.0
    agreement: float = 0.0


class SearchResponse(BaseModel):
    """DTO for ``GET /api/search``."""

    cards: list[SourceCardResponse]
    trace: dict[str, Any]
    confidence: ConfidenceResponse
    profile: str | None = None
    profile_hash: str | None = None


@router.get("/search", response_model=SearchResponse)
async def search(
    request: Request,
    q: str = Query(..., description="Query string (question or bare search term)"),
    scope: str | None = Query(
        None,
        description='Comma-separated zim_ids or archive names/filenames, e.g. "1,3" '
        'or "wikipedia_en_top"',
    ),
    profile: str | None = Query(None, description="Profile name override"),
) -> SearchResponse:
    """Run the retrieval pipeline and return source cards + trace."""
    state: AppState = app_state(request)

    # Resolve profile: ?profile= override (gated) else the active setting.
    # User-saved profiles (the ``retrieval.profiles`` blob) take priority over
    # built-ins of the same name so a saved profile is usable by name without a
    # restart (04-dev-console.md). Falls back to ``lexical`` if the named
    # profile is neither a user save nor a built-in.
    profile_name = _resolve_profile_name(profile)
    retrieval_profile = _resolve_retrieval_profile(profile_name)
    if retrieval_profile is None:
        raise RuntimeError(f"profile {profile_name!r} not found; no profiles loaded")

    # Parse scope via the shared parser (same contract as /api/answer): bare
    # integer zim_ids or archive name/filename tokens resolved against the
    # registry. An unresolvable token degrades to a matches-nothing scope —
    # never silently to the whole corpus.
    ret_scope = _parse_scope(scope, state.registry)
    # Build deps. The fan-out semaphore is sized from retrieval.max_archives_concurrent
    # ("async fan-out needs a bound") — NOT the libzim read pool, which is
    # a separate concern (blocking-read threads vs concurrent archive searches).
    capabilities = compute_capabilities()
    try:
        sn = app_config.snapshot()
    except RuntimeError:
        sn = None
    sem_bound = _concurrency_bound(sn)

    deps = Deps(
        archives=state.registry,
        settings=sn,
        capabilities=capabilities,
        semaphore=asyncio.Semaphore(sem_bound),
        encoders=state.encoders,
        # The dense ``vector_knn`` source receives the store via DI.
        # ``None`` on a vec0-less build → the source is capability-dropped.
        vectors=get_vector_store(),
    )

    try:
        result = await run_pipeline(
            profile=retrieval_profile,
            query=q,
            scope=ret_scope,
            deps=deps,
        )
    except NoCandidatesError as exc:
        # A hard requirement surfaced as a typed error. Return an empty result
        # set with the trace preserved (the exception carries it) rather than a
        # 500 — "nothing matched" is a valid, explainable search outcome.
        return SearchResponse(
            cards=[],
            trace=exc.trace.to_dict(),
            confidence=ConfidenceResponse(),
            profile=retrieval_profile.name,
            profile_hash=retrieval_profile.hash,
        )

    # Map to DTOs.
    cards = [
        SourceCardResponse(
            zim_id=c.zim_id,
            path=c.path,
            title=c.title,
            snippet=c.snippet,
            breadcrumb=c.breadcrumb,
            score=c.score,
            source=c.source,
        )
        for c in result.cards
    ]

    # Enrich with media assets (0008): batch-fetch per zim_id so media-ZIM cards
    # carry their poster/video path inline. Retrieval stays DB-free; this is an
    # api-layer concern. Tolerates a missing table by returning empty.
    await _enrich_cards_with_media(state.db, cards)
    # Enrich with document assets (0013): batch-fetch per zim_id so documents-kind
    # hits carry the manifest title/author/description + a resolvable reader url.
    await _enrich_cards_with_documents(state.db, cards)

    return SearchResponse(
        cards=cards,
        trace=result.trace.to_dict(),
        confidence=ConfidenceResponse(
            top_score=result.confidence.top_score,
            score_dropoff=result.confidence.score_dropoff,
            density=result.confidence.density,
            agreement=result.confidence.agreement,
        ),
        profile=retrieval_profile.name,
        profile_hash=retrieval_profile.hash,
    )


def _concurrency_bound(sn: Any) -> int:
    """The retrieval fan-out semaphore size from settings, defaulting sanely."""
    if sn is not None:
        try:
            return max(1, int(sn.get(RETRIEVAL_MAX_ARCHIVES_CONCURRENT)))
        except Exception:
            pass
    return int(RETRIEVAL_MAX_ARCHIVES_CONCURRENT.default)


def _resolve_retrieval_profile(name: str) -> RetrievalProfile | None:
    """Resolve a profile name to a validated profile, user-saved first.

    User profiles (the ``retrieval.profiles`` blob) shadow built-ins of the same
    name (04-dev-console.md). Falls back to ``lexical`` only if the name matches
    nothing — preserving the fallback behaviour for an unknown name.
    """
    return resolve_profile_from_settings(name, fallback_to_default=True)


def _resolve_profile_name(profile: str | None) -> str:
    """The profile name to use for this request.

    ``?profile=`` override (gated by ``api.allow_profile_override``), else the
    active profile from settings, else the ``lexical`` default.
    """
    if profile and _allow_override():
        return profile
    try:
        val = app_config.get(RETRIEVAL_ACTIVE_PROFILE)
        if isinstance(val, str) and val:
            return val
    except Exception:
        pass
    return "lexical"


def _allow_override() -> bool:
    """True if profile override is allowed. Fails *closed* on settings error.

    A gate flag should not silently open when the resolver is misconfigured;
    defaulting to "no override" is the safe choice for a dev-only feature.
    """
    try:
        return bool(app_config.get(API_ALLOW_PROFILE_OVERRIDE))
    except Exception:
        return False


async def _enrich_cards_with_media(db: Any, cards: list[SourceCardResponse]) -> None:
    """Attach media assets (0008) to cards in place, batched per archive.

    Media-ZIM cards (whose paths are meta-refresh stubs) get a ``media`` dict so
    the frontend can show a poster / open a native ``<video>``. Article-ZIM cards
    have no manifest rows and stay ``media=None``. One indexed ``IN(…)`` query
    per archive in the result set; tolerates a missing table by doing nothing."""
    if not cards:
        return
    from vesta.zim.media import fetch_media_for_paths

    # Group paths by zim_id so each archive is one batched lookup.
    by_zim: dict[int, list[str]] = {}
    for c in cards:
        by_zim.setdefault(c.zim_id, []).append(c.path)
    media_by_zim_path: dict[int, dict[str, Any]] = {}
    for zim_id, paths in by_zim.items():
        try:
            media_by_zim_path[zim_id] = await fetch_media_for_paths(db, zim_id, paths)
        except Exception:
            media_by_zim_path[zim_id] = {}
    for c in cards:
        ref = media_by_zim_path.get(c.zim_id, {}).get(c.path)
        if ref is not None:
            c.media = MediaOut(
                video_path=ref.video_path,
                poster_path=ref.poster_path,
                duration=ref.duration,
            ).model_dump()


async def _enrich_cards_with_documents(db: Any, cards: list[SourceCardResponse]) -> None:
    """Attach document assets (0013) to cards in place, batched per archive.

    Documents-kind hits (whose path is a document entry) get a ``document`` dict
    so the frontend names a result by its manifest title/author/description and
    can open the PDF. The card's display ``title`` is likewise overridden with
    the manifest title so a result reads "Distillation For Home Water
    Treatment", not a bare ``files/Water (1).pdf`` path (the libzim title of a
    PDF entry is the filename/stub). Article/media-ZIM cards have no manifest
    rows and stay ``document=None``, title untouched. One indexed ``IN(…)``
    query per archive in the result set; tolerates a missing table by doing
    nothing."""
    if not cards:
        return
    from vesta.zim.documents import fetch_document_refs_for_paths

    # Group paths by zim_id so each archive is one batched lookup.
    by_zim: dict[int, list[str]] = {}
    for c in cards:
        by_zim.setdefault(c.zim_id, []).append(c.path)
    doc_by_zim_path: dict[int, dict[str, Any]] = {}
    for zim_id, paths in by_zim.items():
        try:
            doc_by_zim_path[zim_id] = await fetch_document_refs_for_paths(db, zim_id, paths)
        except Exception:
            doc_by_zim_path[zim_id] = {}
    for c in cards:
        ref = doc_by_zim_path.get(c.zim_id, {}).get(c.path)
        if ref is not None:
            c.document = DocumentOut(
                doc_path=ref.doc_path,
                title=ref.title,
                description=ref.description,
                author=ref.author,
                doc_mime=ref.doc_mime,
                url=ref.url,
            ).model_dump()
            if ref.title:
                c.title = ref.title
