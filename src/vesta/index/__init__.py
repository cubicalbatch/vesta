"""Semantic-index package — depth management, estimator, indexer job, preemption.

Owns the cost of dense retrieval so it is opt-in, honest, and resumable. Three
things live here:

* **Settings** (``index.*``): declared at import so ``GET /api/settings/schema``
  surfaces them with no frontend change.
* **The ``Capability.VECTORS`` probe**: "at least one archive at index depth ≥ 1
  with a usable index". It is a cheap in-memory flag (the indexer updates it;
  ``main`` seeds it at startup from the ``zims`` table) because
  ``compute_capabilities()`` runs per search request and must stay synchronous
  and fast — never a vector-table scan.
* **The preemption coordinator** singleton: bound by ``main``, read by the
  indexer job and driven by a request-path middleware (also wired in ``main``).

``index/`` depends on ``zim``, ``encoders``, ``vectors``, ``config``, ``db``
(module map: "index/ ... depends on: zim, encoders, vectors").
It must not import ``api``/``answer``/``retrieval``; the dense source receives
the store through DI, never by importing this package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vesta.config.capabilities import Capability, CapabilitySet, register_probe
from vesta.config.settings import setting

if TYPE_CHECKING:
    from starlette.middleware.base import BaseHTTPMiddleware

    from vesta.db.connection import Database
    from vesta.index.preempt import PreemptionCoordinator

# ── Settings ─────────────────────────────────────────────────────────────
# Depth/embedder/batch shape are ``hot=False``: they select what an index build
# commits to, and a build is a multi-hour job, so a runtime change cannot retrofit
# an existing index — restart to apply, then re-index (same shape as the encoders
# model-choice settings). Preemption knobs are ``hot=True`` (per-request effect).

INDEX_DEFAULT_DEPTH = setting(
    "index.default_depth",
    int,
    0,
    group="Index",
    help="Default index depth for a newly registered archive. 0 = "
    "lexical only (the default; nothing indexes unless asked). 1 = article "
    "(title+lead). 2 = + one vector per H2 section. 3 = every ~400-token passage.",
    min=0,
    max=3,
    hot=False,
)
INDEX_EMBEDDER = setting(
    "index.embedder",
    str,
    "onnx-community/granite-embedding-small-english-r2-ONNX",
    group="Index",
    help="The INDEX embedder (repo id) the indexer builds with. This is a "
    "SEPARATE role from encoders.static.model (the static embedder "
    "is not the index embedder). It must match the query-side "
    "encoders.embed.model for the index to be served — a mismatch marks the "
    "archive stale at startup and the dense source refuses its vectors "
    "(index-settings enforcement). Change both settings together, then "
    "re-index.",
    hot=False,
)
INDEX_BATCH_SIZE = setting(
    "index.batch_size",
    int,
    64,
    group="Index",
    help="Articles per extraction/embed batch during indexing. Bounded so an "
    "in-flight batch finishes promptly when an interactive request arrives and "
    "the preemption coordinator yields.",
    min=1,
    max=1024,
    hot=False,
)
INDEX_WORKER_PROCESSES = setting(
    "index.worker_processes",
    int,
    2,
    group="Index",
    help="Extraction/embedding worker processes (threads scale "
    "negatively for extraction; processes scale properly). Also the SIGSTOP "
    "targets for CPU preemption.",
    min=1,
    max=32,
    hot=False,
)
INDEX_WORKER_RECYCLE_EVERY = setting(
    "index.worker_recycle_every",
    int,
    500,
    group="Index",
    help="Recycle the extraction spawn pool every N batches to bound "
    "worker-process RSS on very long runs (e.g. full Wikipedia), where libzim "
    "per-read allocations would otherwise grow each worker monotonically. 0 = "
    "never recycle. Each recycle closes and respawns the worker processes; the "
    "job re-registers their PIDs with the preemption coordinator.",
    min=0,
    max=100000,
    hot=False,
)
INDEX_PREEMPT_ENABLED = setting(
    "index.preempt.enabled",
    bool,
    True,
    group="Index",
    help="SIGSTOP the indexer while an interactive request (search/answer) is "
    "in flight. Disable only for diagnosis.",
    hot=True,
)
INDEX_PREEMPT_IDLE_MS = setting(
    "index.preempt.idle_ms",
    int,
    50,
    group="Index",
    help="Cooperative yield between indexer batches (ms). The indexer sleeps "
    "this long between batches so the event loop and interactive requests get a "
    "turn even under sustained indexing load.",
    min=0,
    max=5000,
    hot=True,
)
INDEX_NICE = setting(
    "index.nice",
    int,
    15,
    group="Index",
    help="nice(2) delta applied to indexer worker processes ('nice 15'). "
    "Lowers scheduling priority so interactive work wins the CPU.",
    min=0,
    max=19,
    hot=True,
)

# One index job at a time (CPU-bound).
# Mirrors the jobs.max_concurrent.noop precedent.
JOBS_MAX_CONCURRENT_INDEX_ZIM = setting(
    "jobs.max_concurrent.index_zim",
    int,
    1,
    group="Jobs",
    help="Concurrent index_zim jobs. Fixed at 1: indexing is CPU-bound and a "
    "second concurrent build would only contend with the first.",
    min=1,
    max=1,
    hot=False,
)

# ── Capability probe ──────────────────────────────────────────────────────
# In-memory flag (cheap, sync — compute_capabilities runs per request). Seeded
# at startup from the zims table by main; updated by the indexer when an index
# completes, is lowered to 0, or its archive is removed. The per-archive
# mismatch refusal happens in the dense source via store.describe(), not here.
_ANY_INDEXED: bool = False


def set_indexed_state(any_indexed: bool) -> None:
    """Update the cached 'is there a usable semantic index' flag.

    Called by ``main`` at startup (seeded from ``zims``) and by the indexer job
    when an index build completes or is torn down."""
    global _ANY_INDEXED
    _ANY_INDEXED = any_indexed


async def reseed_indexed_state(db: Database) -> None:
    """Recompute :func:`set_indexed_state` from the ``zims`` table: on iff some
    enabled archive has depth ≥ 1 and a complete/running build.

    The single copy of the seed query — run at startup (main), when the CLI
    composition root opens its runtime, and whenever an archive's index rows
    change outside a build. Error policy stays with the caller: startup and CLI
    degrade to flag-off, mutation endpoints let a failure surface.
    """
    async with (
        db.read() as conn,
        conn.execute(
            "SELECT 1 FROM zims WHERE enabled=1 AND index_depth>=1 "
            "AND index_status IN ('complete','running') LIMIT 1"
        ) as cur,
    ):
        row = await cur.fetchone()
    set_indexed_state(row is not None)


def _capability_probe() -> CapabilitySet:
    if _ANY_INDEXED:
        return frozenset({Capability.VECTORS})
    return frozenset()


register_probe(_capability_probe)


# ── Preemption coordinator singleton ──────────────────────────────────────
_COORDINATOR: PreemptionCoordinator | None = None


def bind_coordinator(coordinator: PreemptionCoordinator | None) -> None:
    """Attach (or detach, with ``None``) the live preemption coordinator."""
    global _COORDINATOR
    _COORDINATOR = coordinator


def get_coordinator() -> PreemptionCoordinator | None:
    """The live coordinator, or ``None`` outside the FastAPI lifespan."""
    return _COORDINATOR


# ── Runtime bindings for the index job (composition root wires these) ──────
# The JobType protocol gives a job only ``(JobHandle, params)``; the db, the
# archive registry, and the embedder are reached through these singletons, bound
# by ``main``'s lifespan (mirroring encoders/vectors/inference bind_*). Each is
# ``None`` outside the lifespan so a unit test can run the job with fakes.
_DB: Any = None
_REGISTRY: Any = None
_EMBEDDER_PROVIDER: Any = None


def bind_runtime(
    db: Any,
    registry: Any,
    embedder_provider: Any,
) -> None:
    """Bind the db, archive registry, and embedder provider for the index job."""
    global _DB, _REGISTRY, _EMBEDDER_PROVIDER
    _DB = db
    _REGISTRY = registry
    _EMBEDDER_PROVIDER = embedder_provider


def get_db() -> Any:
    return _DB


def get_registry() -> Any:
    return _REGISTRY


def get_embedder_provider() -> Any:
    return _EMBEDDER_PROVIDER


def make_middleware() -> type[BaseHTTPMiddleware]:
    """Build the FastAPI/Starlette middleware that feeds the coordinator.

    Constructed in ``main`` (NOT in ``api/search.py``). Returns a Starlette
    ``BaseHTTPMiddleware`` subclass instance that
    increments the in-flight counter for interactive paths and decrements on
    completion. The coordinator is read at *dispatch* time (via
    :func:`get_coordinator`) because the app is built before the lifespan binds
    the coordinator."""
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request as StarletteRequest

    from vesta.index.preempt import INTERACTIVE_PATH_PREFIXES

    class IndexPreemptionMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: StarletteRequest, call_next):  # type: ignore[no-untyped-def]
            coord = get_coordinator()
            path = request.url.path
            interactive = any(path.startswith(p) for p in INTERACTIVE_PATH_PREFIXES)
            if coord is not None and interactive:
                await coord.acquire()
                try:
                    return await call_next(request)
                finally:
                    await coord.release()
            return await call_next(request)

    return IndexPreemptionMiddleware


__all__ = [
    "INDEX_BATCH_SIZE",
    "INDEX_DEFAULT_DEPTH",
    "INDEX_EMBEDDER",
    "INDEX_NICE",
    "INDEX_PREEMPT_ENABLED",
    "INDEX_PREEMPT_IDLE_MS",
    "INDEX_WORKER_PROCESSES",
    "INDEX_WORKER_RECYCLE_EVERY",
    "JOBS_MAX_CONCURRENT_INDEX_ZIM",
    "bind_coordinator",
    "get_coordinator",
    "make_middleware",
    "reseed_indexed_state",
    "set_indexed_state",
]
