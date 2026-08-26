"""Application factory + lifespan.

The lifespan is the only place the wiring order matters, and it matters a lot:

1. Configure settings with default+env only (the DB isn't open yet, so the
   settings-table layer can't participate — and the DB pragmas *are* settings).
2. Open the DB, apply migrations (before anything reads).
3. Load the ``settings`` table and add it as the authoritative layer.
4. Start the job runner, which resumes anything killed mid-run.

Shutdown drains in reverse. ``docker compose up`` on an empty ``./data`` must
reach step 4 without error — nothing is configured and that is a healthy state.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import FastAPI

if TYPE_CHECKING:
    from vesta.inference.gateway import Gateway
    from vesta.inference.local import LlamaServerSupervisor
    from vesta.inference.runtime import LlmRuntime

from vesta import config
from vesta import index as index_pkg  # noqa: F401 — registers index.* settings + VECTORS probe
from vesta.api import api_router
from vesta.api.state import AppState
from vesta.catalog import (
    bind_runtime as bind_catalog_runtime,
)
from vesta.catalog import (
    download as _download_job,  # noqa: F401 — registers the download_zim job type
)
from vesta.catalog import (
    refresh as _refresh_job,  # noqa: F401 — registers the refresh_catalog job type
)
from vesta.db.connection import Database
from vesta.db.migrations import run_migrations
from vesta.db.settings_store import load_settings
from vesta.encoders import ENCODERS_EMBED_MODEL, bind_manager, build_manager_from_settings
from vesta.encoders.registry import MODEL_SPECS
from vesta.index import (
    INDEX_EMBEDDER,
    INDEX_PREEMPT_ENABLED,
    INDEX_PREEMPT_IDLE_MS,
    bind_coordinator,
    bind_runtime,
    make_middleware,
    reseed_indexed_state,
    set_indexed_state,
)
from vesta.index import job as _index_job  # noqa: F401 — registers the index_zim job type
from vesta.index.compat import fingerprint_from_spec, reconcile_stale
from vesta.index.preempt import PreemptionCoordinator
from vesta.inference import (
    INFERENCE_LOCAL_PRELOAD_ON_READY,
    bind_gateway,
    bind_models_dir,
    bind_on_model_ready,
    build_gateway_from_settings,
    build_runtime_from_settings,
    get_gateway,
    get_runtime,
    get_supervisor,
)
from vesta.inference import (
    bind_runtime as bind_llm_runtime,
)
from vesta.inference import (
    download as _model_download_job,  # noqa: F401 — registers the download_model job type
)
from vesta.jobs.runner import JobRunner
from vesta.vectors import (
    VECTORS_BACKEND,
    VECTORS_OVERSAMPLE,
    VECTORS_QUANTIZER,
    bind_store,
    get_store,
)
from vesta.vectors.sqlite_vec_store import SqliteVecStore
from vesta.zim import bind_registry
from vesta.zim.registry import ArchiveRegistry


def _configure_logging(level: str) -> None:
    """Structured JSON to stdout, no telemetry. One place to read logs.

    Our own logs go out as JSON via structlog; stdlib logging (uvicorn, etc.) is
    bridged through the same processors so every line on stdout is JSON.
    """
    level_num = getattr(logging, level.upper(), logging.INFO)
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    structlog.configure(
        processors=[*shared_processors, structlog.processors.JSONRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(level_num),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    # Bridge stdlib logging through the same JSON renderer. ExtraAdder is
    # required: without it, every stdlib ``log.info(..., extra={...})`` field is
    # silently dropped from the JSON output (verified 2026-07-31 — this had been
    # swallowing structured fields codebase-wide: jobs, zim, encoders, answer,
    # inference). ``answer.log_prompts`` depends on those fields surviving.
    handler = logging.StreamHandler()
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=[*shared_processors, structlog.stdlib.ExtraAdder()],
        )
    )
    root = logging.getLogger()
    if not any(h is handler for h in root.handlers):
        root.addHandler(handler)
    root.setLevel(level_num)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: open DB → migrate → load settings → start jobs. Shutdown: drain."""
    # 1. default+env only — DB pragmas come from settings, so the resolver must
    #    be usable before the DB exists.
    config.configure()
    _configure_logging(str(config.get(config.LOGGING_LEVEL)))
    log = structlog.get_logger("vesta.startup")

    data_dir = Path(str(config.get(config.DATA_DIR)))
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "vesta.db"

    db = Database(
        str(db_path),
        busy_timeout_ms=int(config.get(config.DB_BUSY_TIMEOUT_MS)),
    )
    await db.start()

    # 2. Migrations run before anything reads (empty DB and a DB from a
    #    previous version both reach the same state).
    async with db.write() as conn:
        applied = await run_migrations(conn)
    if applied:
        log.info("db.migrated", migrations=applied)

    # 3. The settings table is authoritative over env.
    async with db.read() as conn:
        stored = await load_settings(conn)
    config.set_db_values(stored)

    # 4. Construct the job runner, but do NOT resume interrupted jobs yet: a
    #    resumed index_zim job needs bind_store/bind_registry/bind_runtime
    #    (below), and JobRunner.start() re-enqueues orphaned "running" jobs as
    #    asyncio tasks that can start executing as soon as this coroutine next
    #    awaits — before those bindings exist. Calling start() here reproduced
    #    exactly that race live: the resumed job
    #    crashed instantly with "db/registry/store not bound". start() is
    #    deferred to just before `yield`, after every bind_* call.
    runner = JobRunner(db)

    # 4b. Construct the encoder manager (cheap — no ONNX session loaded until a
    # scorer's first real call, encoders/manager.py) and bind it so the
    # STATIC_ENCODER/CROSS_ENCODER capability probes reflect real model presence.
    encoders = build_manager_from_settings(config.snapshot(), model_dir=data_dir / "models")
    bind_manager(encoders)

    # 4b2. Bind the models directory for the download_model job (same dir as
    # encoders — GGUFs and ONNX models share data/models/).
    bind_models_dir(data_dir / "models")

    # 4c. Construct the vector store and bind it. Cheap:
    # only creates the default-dim vec0 table if vec0 loaded on every connection
    # (db.vec0_available()). On a vec0-less build the store is bound but every
    # method is a no-op — Capability.VECTORS stays off, the dense source
    # is capability-dropped, and the app keeps working with lexical search only.
    # quantizer/oversample are read from the vectors.* settings this package
    # registered at import.
    vector_store = SqliteVecStore(
        db,
        quantizer=str(config.get(VECTORS_QUANTIZER)),
        oversample=int(config.get(VECTORS_OVERSAMPLE)),
    )
    await vector_store.ensure_default_table()
    bind_store(vector_store)
    if not db.vec0_available():
        log.warning("vectors.vec0_unavailable", msg="semantic index disabled (no vec0 extension)")
    else:
        log.info("vectors.ready", backend=str(config.get(VECTORS_BACKEND)))

    # 5. Open the ZIM archive registry: scan ./data/zims, probe indexes, mine
    #    aliases, raise the libzim cluster cache. bind_registry wires
    #    the ZIM_FULLTEXT capability probe so health reflects real state.
    zims_dir = data_dir / "zims"
    registry = ArchiveRegistry(
        db=db,
        zims_dir=zims_dir,
        read_pool_size=int(config.get(config.ZIM_READ_POOL_SIZE)),
        cluster_cache_mb=int(config.get(config.ZIM_CLUSTER_CACHE_MB)),
    )
    try:
        scan = await registry.start()
        if scan.added or scan.missing:
            log.info(
                "zim.scanned",
                added=list(scan.added),
                updated=list(scan.updated),
                missing=list(scan.missing),
                total=scan.total,
            )
    except Exception as exc:  # a startup scan failure must not kill the app
        log.warning("zim.scan_failed", error=repr(exc))
    bind_registry(registry)

    # 5c. CPU preemption for the indexer. The coordinator SIGSTOPs the
    #    indexer's worker processes while an interactive request (search/answer)
    #    is in flight; the request-path middleware (added in create_app) feeds it.
    coordinator = PreemptionCoordinator(
        enabled=bool(config.get(INDEX_PREEMPT_ENABLED)),
        idle_ms=int(config.get(INDEX_PREEMPT_IDLE_MS)),
    )
    bind_coordinator(coordinator)

    # 5d. Wire the index job's runtime singletons (db, registry, embedder) and
    #    reconcile stale indexes BEFORE seeding the VECTORS capability flag.
    #    The embedder provider honors ``index.embedder`` (a separate role from
    #    the query-side ``encoders.embed.model``), deferring to the
    #    manager's lazy per-repo loader so a model downloaded after startup is
    #    picked up on the next index run.
    index_embedder_repo = str(config.get(INDEX_EMBEDDER))

    async def _index_embedder_provider() -> Any:
        return await encoders.get_embed_for(index_embedder_repo)

    bind_runtime(db, registry, _index_embedder_provider)
    # Changing either embedder setting marks mismatched archives stale;
    # both settings are hot=False, so the change always arrives via a restart.
    configured_fps = [
        fingerprint_from_spec(MODEL_SPECS[repo])
        for repo in {index_embedder_repo, str(config.get(ENCODERS_EMBED_MODEL))}
        if repo in MODEL_SPECS
    ]
    stale = await reconcile_stale(db, vector_store, configured_fps)
    if stale:
        log.warning("index.stale_after_embedder_change", zim_ids=stale)
    await _seed_indexed_state(db)

    # 5e. Wire the vector-store cascade into archive removal (vec0 is not
    #    FK-cascaded; ``zim/`` can't import ``vectors/``, so the composition root
    #    registers the callback). ``get_store()`` is read at call time so the
    #    binding survives the lifespan teardown ordering.
    async def _cascade_vectors(zim_id: int) -> None:
        store = get_store()
        if store is not None:
            await store.delete_by_zim(zim_id)

    registry.add_on_remove(_cascade_vectors)

    # 5e2. Wire the catalog/download jobs' runtime singletons. The
    #    download job needs the zims dir + a post-download "register the archive"
    #    callback; catalog/ cannot import zim/, so the
    #    composition root injects ``registry.rescan`` here. ``rescan`` runs the
    #    full probe (fulltext index, Counter['text/html'] article count,
    #    alias mining) on the freshly-renamed file. The db backs the catalog cache
    #    + FTS used by the refresh_catalog job.
    async def _register_downloaded(path: Any) -> None:
        await registry.rescan()

    bind_catalog_runtime(db, zims_dir, _register_downloaded)

    # 5b. Wire inference: the gateway (lazy for local — the supervisor only
    #    starts when an answer is requested) plus the LLM runtime
    #    and its post-download callback.
    gateway, supervisor, llm_runtime = _wire_inference(data_dir, log)

    # 6. Now that every bind_* call above has completed, it's safe to resume
    #    jobs left "running" by a prior crash — their tasks can rely on
    #    db/registry/store/encoders/gateway all being bound already.
    await runner.start()

    # 6b. Prune stale conversation traces once at startup:
    #     multi-turn conversations with full traces
    #     grow the DB; the retention setting bounds it. A single startup pass is
    #     sufficient for a single-user appliance.
    await _prune_conversation_traces(db, log)

    # 6c. Reconcile unified benchmark runs a dead process left "running":
    #     the background task lives only in this
    #     process's memory, so any such row is orphaned and gets marked
    #     aborted at startup.
    await _reconcile_bench_runs(db, log)

    # 6c'. Reconcile golden-set eval runs a dead process left "running":
    #      same reasoning as 6c — the background task lives only in this
    #      process's memory, so any such row is orphaned and gets marked
    #      error at startup.
    await _reconcile_eval_runs(db, log)

    # 6d. Prune per-question benchmark traces past retention, once at startup
    #     (mirrors 6b): verdicts + retrieval + answer text stay forever; only
    #     the bulky trace_json is bounded by bench.trace_retention_days.
    await _prune_bench_traces(db, log)

    app.state.vesta = AppState(
        db=db,
        runner=runner,
        registry=registry,
        encoders=encoders,
        gateway=gateway,
        supervisor=supervisor,
    )
    log.info("vesta.ready", db=str(db_path))

    # 7. Start the LLM idle watchdog — unloads weights / stops the
    #    child when the model goes idle. Stopped in the finally block below,
    #    BEFORE the supervisor is stopped (it drives the supervisor).
    if llm_runtime is not None:
        llm_runtime.start()

    try:
        yield
    finally:
        # Stop the LIVE inference singletons, not the startup locals — a
        # settings rebind (inference.rebuild_runtime) may have swapped the
        # runtime + supervisor for fresh ones after startup.
        live_gateway = get_gateway()
        live_supervisor = get_supervisor()
        live_runtime = get_runtime()
        bind_registry(None)
        bind_manager(None)
        bind_models_dir(None)
        bind_store(None)
        bind_gateway(None, None)
        bind_llm_runtime(None)
        bind_on_model_ready(None)
        if live_runtime is not None:
            await live_runtime.stop()
        if live_gateway is not None:
            with suppress(Exception):
                await live_gateway.aclose()
        if live_supervisor is not None:
            await live_supervisor.stop()
        await registry.stop()
        await runner.stop()
        await db.stop()
        log.info("vesta.stopped")


def _wire_inference(
    data_dir: Path, log: Any
) -> tuple[Gateway | None, LlamaServerSupervisor | None, LlmRuntime | None]:
    """Construct + bind the inference gateway, LLM runtime, and callback.

    All three are lazy — nothing spawns until the first chat/``ensure_ready``.
    A failure degrades gracefully (no ``Capability.LLM``, ``sources_only``
    stays usable) and is logged, never raised into the lifespan.
    """
    gateway: Gateway | None = None
    supervisor: LlamaServerSupervisor | None = None
    try:
        gateway, supervisor = build_gateway_from_settings(config.snapshot(), data_dir=data_dir)
        bind_gateway(gateway, supervisor)
    except Exception as exc:
        log.warning("inference.gateway_failed", error=repr(exc))
        gateway = None
        supervisor = None

    llm_runtime: LlmRuntime | None = None
    try:
        # Share the gateway's supervisor — one child on one fixed port; a
        # second supervisor would race it (and a rebind swaps this same
        # supervisor out from under both, see inference._rebind_runtime).
        llm_runtime = build_runtime_from_settings(
            config.snapshot(), data_dir=data_dir, supervisor=supervisor
        )
        bind_llm_runtime(llm_runtime)
    except Exception as exc:
        log.warning("inference.runtime_failed", error=repr(exc))
        llm_runtime = None

    bind_on_model_ready(_model_ready)
    return gateway, supervisor, llm_runtime


async def _reconcile_bench_runs(db: Database, log: Any) -> None:
    """Mark orphaned "running" bench rows aborted (their task died with the
    process). Failures are logged, never startup-blocking."""
    try:
        from vesta.api.bench import reconcile_stale_bench_runs

        bench_aborted = await reconcile_stale_bench_runs(db)
        if bench_aborted:
            log.info("bench.runs_reconciled", aborted=bench_aborted)
    except Exception as exc:
        log.warning("bench.reconcile_failed", error=repr(exc))


async def _reconcile_eval_runs(db: Database, log: Any) -> None:
    """Mark orphaned "running" eval rows errored (their task died with the
    process). Failures are logged, never startup-blocking."""
    try:
        from vesta.api.eval import reconcile_stale_eval_runs

        errored = await reconcile_stale_eval_runs(db)
        if errored:
            log.info("eval.runs_reconciled", errored=errored)
    except Exception as exc:
        log.warning("eval.reconcile_failed", error=repr(exc))


async def _prune_bench_traces(db: Database, log: Any) -> None:
    """Null ``trace_json`` on benchmark questions past retention.

    A single startup pass bounds DB growth from accumulated per-question traces
    (mirrors :func:`_prune_conversation_traces`). Failures are logged and
    swallowed — pruning is housekeeping, never startup-blocking.
    """
    try:
        from vesta.api.bench import prune_stale_bench_traces
        from vesta.eval.bench_runner import BENCH_TRACE_RETENTION_DAYS

        retention = int(config.get(BENCH_TRACE_RETENTION_DAYS))
        pruned = await prune_stale_bench_traces(db, retention)
        if pruned:
            log.info("bench.traces_pruned", count=pruned, retention_days=retention)
    except Exception as exc:
        log.warning("bench.trace_prune_failed", error=repr(exc))


async def _model_ready(path: Path) -> None:
    """Post-GGUF-download callback, bound by ``_wire_inference``.

    The runtime refreshes (restarting the child so the router discovers the new
    GGUF — a running router does not rescan its models dir) and,
    when ``inference.local.preload_on_ready`` is set, loads the model so the
    user's first question is fast. Bound in ``main`` so ``inference/`` never
    imports ``jobs/``.
    """
    # Resolved lazily (not via the module-level import) so the seam the tests
    # patch — ``vesta.inference.get_runtime`` — is honoured here too.
    from vesta.inference import get_runtime

    runtime = get_runtime()
    if runtime is None:
        return
    await runtime.rebuild(config.snapshot(), force_restart=True)
    if bool(config.get(INFERENCE_LOCAL_PRELOAD_ON_READY)):
        try:
            await runtime.ensure_ready()
        except Exception as exc:
            logging.getLogger("vesta.startup").warning(
                "inference.preload_failed", extra={"error": repr(exc), "path": str(path)}
            )


def create_app() -> FastAPI:
    """Build the FastAPI app. Importing ``vesta.main`` gives you ``app``."""
    app = FastAPI(title="Vesta", version="0.1.0", lifespan=lifespan)
    # The preemption middleware feeds the coordinator from the request
    # path (search/answer) WITHOUT editing api/search.py. Added here
    # because middleware is wired at app setup, before the lifespan binds the
    # coordinator; the middleware reads it lazily at dispatch time.
    app.add_middleware(make_middleware())
    app.include_router(api_router)
    return app


async def _prune_conversation_traces(db: Database, log: Any) -> None:
    """Null ``trace_json`` on conversation messages past retention.

    A single startup pass bounds DB growth from accumulated multi-turn traces.
    Failures are logged and swallowed — pruning is housekeeping, never startup-
    blocking.
    """
    try:
        from vesta.answer import CHAT_TRACE_RETENTION_DAYS
        from vesta.api.conversation_store import prune_old_traces

        retention = int(config.get(CHAT_TRACE_RETENTION_DAYS))
        pruned = await prune_old_traces(db, retention)
        if pruned:
            log.info("chat.traces_pruned", count=pruned, retention_days=retention)
    except Exception as exc:
        log.warning("chat.trace_prune_failed", error=repr(exc))


async def _seed_indexed_state(db: Database) -> None:
    """Turn VECTORS on iff some enabled archive already has a usable index."""
    try:
        await reseed_indexed_state(db)
    except Exception:  # a schema/migration mismatch never blocks startup
        set_indexed_state(False)


# Importing this module yields a ready ``app`` for ``uvicorn vesta.main:app``.
app = create_app()


def main() -> None:
    """``python -m vesta.main`` — read host/port from settings, serve."""
    config.configure()  # env-only resolution for bind address
    host = str(config.get(config.SERVER_HOST))
    port = int(config.get(config.SERVER_PORT))
    log_level = str(config.get(config.LOGGING_LEVEL)).lower()

    import uvicorn

    config.reset_for_test()
    uvicorn.run(
        "vesta.main:app",
        host=host,
        port=port,
        log_level=log_level,
        workers=1,  # one uvicorn worker, ever.
    )


if __name__ == "__main__":
    main()
