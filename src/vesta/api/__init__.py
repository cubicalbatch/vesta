"""HTTP API routers — ``api/`` is the composition root.

DTOs (Pydantic models) live here and in each router; internal packages use
frozen dataclasses. The mapping between them is explicit and stays in this
package — this is what stops a UI change from rippling into the retrieval core.
"""

from __future__ import annotations

from fastapi import APIRouter

from vesta.api import (
    answer,
    bench,
    chat,
    eval,
    health,
    jobs,
    library,
    models,
    retrieval,
    search,
    settings,
    setup,
    spa,
    system,
    zim,
    zims,
)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(settings.router)
api_router.include_router(jobs.router)
api_router.include_router(zims.router)
api_router.include_router(zim.router)
api_router.include_router(library.router)
api_router.include_router(search.router)
api_router.include_router(retrieval.router)
api_router.include_router(answer.router)
api_router.include_router(chat.router)
api_router.include_router(eval.router)
api_router.include_router(bench.router)
api_router.include_router(system.router)
api_router.include_router(models.router)
api_router.include_router(setup.router)
# The production SPA is registered LAST: it owns a catch-all `GET /{path:path}`
# for client-side routing, which must never get a chance to shadow a real
# route above it (11-production-frontend.md "How it is served"; asserted in
# tests/test_spa.py).
api_router.include_router(spa.router)

__all__ = [
    "answer",
    "api_router",
    "bench",
    "chat",
    "eval",
    "health",
    "jobs",
    "library",
    "models",
    "retrieval",
    "search",
    "settings",
    "setup",
    "spa",
    "system",
    "zim",
    "zims",
]
