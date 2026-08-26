"""Eval API — ``POST /api/eval/run``, list/get runs.

The committed, tested eval surface alongside the CLI. Runs the golden set over a
profile and persists the result to ``eval_runs`` with every comparison pin.
The API path runs the work as a job-style async task: the
POST returns immediately with a run id; the result is fetched once done. This
mirrors how every long operation in Vesta behaves.

``api/`` is the composition root: it wires the real DB-
backed :class:`SqliteEvalStore` and the live archive registry into the eval
runner, which itself imports only ``retrieval`` and ``config``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any

import aiosqlite
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, field_validator

from vesta import config as app_config
from vesta.api.state import AppState, app_state
from vesta.config.capabilities import compute_capabilities
from vesta.eval.golden import (
    EVAL_ARCHIVE_CHECKSUM,
    EVAL_ARCHIVE_PATH,
    GOLDEN_SET_NAMES,
    load_set,
)
from vesta.eval.runner import (
    EvalStore,
    PipelineRunner,
    RunRecord,
    evaluate_profile,
    now_iso,
    record_from_row,
)
from vesta.eval.runner import (
    git_sha as _runner_git_sha,
)
from vesta.eval.runner import (
    machine_id as _runner_machine_id,
)
from vesta.retrieval import RETRIEVAL_PROFILES
from vesta.retrieval.contracts import Scope as RetScope
from vesta.retrieval.pipeline import Deps, NoCandidatesError, run_pipeline
from vesta.retrieval.profiles import RetrievalProfile, resolve_profile
from vesta.vectors import get_store as get_vector_store

router = APIRouter(prefix="/api/eval", tags=["eval"])

# Background tasks for in-flight runs (the API path is job-shaped).
# Single-user, one uvicorn worker: an in-process dict is sufficient and
# survives navigation; the row's status reflects progress.
_tasks: dict[int, asyncio.Task[None]] = {}


# ── DB-backed EvalStore (composition root wires DB ↔ eval) ──────────────────


class SqliteEvalStore(EvalStore):
    """Persist eval runs to the ``eval_runs`` table (0001 + migration 0003).

    Implements the :class:`EvalStore` Protocol from ``eval.runner``. Lives in
    ``api/`` (not ``eval/``) because it imports aiosqlite — keeping the DB dep
    out of the eval package preserves the ≤2 dependency cap.
    """

    def __init__(self, db: Any) -> None:
        self._db = db

    async def insert_run(self, record: RunRecord) -> int:
        now = record.started_at
        config_json = json.dumps(record.to_config_json())
        metrics_json = json.dumps(record.to_metrics_json())
        async with self._db.write() as conn:
            cur = await conn.execute(
                "INSERT INTO eval_runs(started_at, config_json, metrics_json, "
                "profile_name, profile_hash, golden_hash, archive_checksum, "
                "git_sha, machine_id, finished_at, status) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    now,
                    config_json,
                    metrics_json,
                    record.profile_name,
                    record.profile_hash,
                    record.golden_hash,
                    record.archive_checksum,
                    record.git_sha,
                    record.machine_id,
                    now,
                    record.status,
                ),
            )
            return int(cur.lastrowid) if cur.lastrowid is not None else 0

    async def update_run(self, run_id: int, record: RunRecord) -> bool:
        """Overwrite a placeholder row with a completed run's full record.

        The API path is job-shaped: POST inserts a placeholder to return an
        id immediately, then the background task fills it in via this update.
        """
        config_json = json.dumps(record.to_config_json())
        metrics_json = json.dumps(record.to_metrics_json())
        async with self._db.write() as conn:
            cur = await conn.execute(
                "UPDATE eval_runs SET config_json=?, metrics_json=?, "
                "profile_name=?, profile_hash=?, golden_hash=?, archive_checksum=?, "
                "git_sha=?, machine_id=?, started_at=?, finished_at=?, status=? "
                "WHERE id=?",
                (
                    config_json,
                    metrics_json,
                    record.profile_name,
                    record.profile_hash,
                    record.golden_hash,
                    record.archive_checksum,
                    record.git_sha,
                    record.machine_id,
                    record.started_at,
                    now_iso(),  # finished_at: stamped when the run lands
                    record.status,
                    run_id,
                ),
            )
            return bool(cur.rowcount > 0)

    async def get_run(self, run_id: int) -> RunRecord | None:
        async with self._db.read() as conn:
            cur = await conn.execute("SELECT * FROM eval_runs WHERE id=?", (run_id,))
            row = await cur.fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    async def list_runs(self, limit: int = 50) -> list[RunRecord]:
        async with self._db.read() as conn:
            cur = await conn.execute("SELECT * FROM eval_runs ORDER BY id DESC LIMIT ?", (limit,))
            rows = await cur.fetchall()
        return [_row_to_record(r) for r in rows]


def _row_to_record(row: aiosqlite.Row) -> RunRecord:
    config = json.loads(row["config_json"]) if row["config_json"] else {}
    metrics_blob = json.loads(row["metrics_json"]) if row["metrics_json"] else {}
    return record_from_row(
        row_id=int(row["id"]),
        config=config,
        metrics_blob=metrics_blob,
        started_at=row["started_at"] or "",
        status=str(row["status"] or "done"),
        profile_name=row["profile_name"],
        profile_hash=row["profile_hash"],
        golden_hash=row["golden_hash"],
        archive_checksum=row["archive_checksum"],
        git_sha=row["git_sha"],
        machine_id=row["machine_id"],
    )


# ── Pipeline runner wired to the live archive registry ───────────────────────


class LivePipelineRunner(PipelineRunner):
    """Run one retrieval query through the real pipeline + open archives.

    The single PipelineRunner implementation for both API-driven eval runs and
    the CLI (``vesta bench``/``vesta eval`` alias it as ``CLIPipelineRunner``)
    — the two former copies could silently drift and break the
    measure-before-ship comparability guarantee.
    """

    def __init__(self, state: AppState, profile: RetrievalProfile | None = None) -> None:
        self._state = state

    async def run(
        self, profile: RetrievalProfile, query: str
    ) -> tuple[tuple[str, ...], dict[str, object]]:
        deps = Deps(
            archives=self._state.registry,
            settings=app_config.snapshot(),
            capabilities=compute_capabilities(),
            semaphore=asyncio.Semaphore(8),
            encoders=self._state.encoders,
            # 08: the dense ``vector_knn`` source receives the store via DI, the
            # same one-line amendment already applied to api/search.py and
            # api/answer.py (03-retrieval-framework.md). Without this, eval
            # runs triggered through the API always capability-drop
            # vector_knn, silently making ``hybrid`` runs
            # indistinguishable from ``standard`` (found live this phase).
            vectors=get_vector_store(),
        )
        try:
            result = await run_pipeline(profile=profile, query=query, scope=RetScope(), deps=deps)
        except NoCandidatesError as exc:
            return (), exc.trace.to_dict()
        paths = tuple(c.path for c in result.cards)
        return paths, result.trace.to_dict()


# ── DTOs ────────────────────────────────────────────────────────────────────


class EvalRunRequest(BaseModel):
    """``POST /api/eval/run`` body. ``profile`` defaults to the active profile."""

    profile: str | None = None
    golden_set: str = "full"
    notes: str = ""

    @field_validator("golden_set")
    @classmethod
    def _known_golden_set(cls, v: str) -> str:
        """Reject typos (``fixture_subsets``) instead of silently launching the
        full pinned-archive run under the misspelled name (AUDIT_0824 N4)."""
        if v not in GOLDEN_SET_NAMES:
            known = ", ".join(GOLDEN_SET_NAMES)
            raise ValueError(f"unknown golden_set {v!r}; valid sets: {known}")
        return v


class EvalRunResponse(BaseModel):
    """The created run's identity + a snapshot of its headline metrics."""

    id: int
    profile: str
    profile_hash: str
    status: str  # running | done | error

    model_config = {"extra": "allow"}


class EvalRunDetail(BaseModel):
    """Full detail for ``GET /api/eval/runs/{id}``."""

    id: int
    started_at: str
    profile: str
    profile_hash: str
    golden_hash: str
    archive_checksum: str
    git_sha: str
    machine_id: str
    status: str = "done"
    metrics: dict[str, Any] = {}
    config: dict[str, Any] = {}

    model_config = {"extra": "allow"}


# ── Routes ──────────────────────────────────────────────────────────────────


@router.post("/run", response_model=EvalRunResponse)
async def run_eval(
    request: Request,
    body: EvalRunRequest,
) -> EvalRunResponse:
    """Start a golden-set run; returns immediately with the run id (job-shape)."""
    state: AppState = app_state(request)
    profile_name = body.profile or "lexical"
    profile = _resolve_profile(state, profile_name)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"profile {profile_name!r} not found")

    store = SqliteEvalStore(state.db)
    # Insert a placeholder row synchronously so the id exists before we return.
    # The row is stamped 'running' with the real start time; the background
    # task overwrites it (status included) when the run lands or fails.
    started_at = now_iso()
    placeholder = _placeholder_record(profile, body, started_at=started_at)
    run_id = await store.insert_run(placeholder)

    task = asyncio.create_task(
        _run_to_completion(state, store, profile, body, run_id, started_at), name=f"eval-{run_id}"
    )
    _tasks[run_id] = task
    return EvalRunResponse(
        id=run_id, profile=profile.name, profile_hash=profile.hash, status="running"
    )


@router.get("/runs", response_model=list[EvalRunDetail])
async def list_runs(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
) -> list[EvalRunDetail]:
    """List recent eval runs, newest first."""
    state: AppState = app_state(request)
    store = SqliteEvalStore(state.db)
    return [_to_detail(r) for r in await store.list_runs(limit)]


@router.get("/runs/{run_id}", response_model=EvalRunDetail)
async def get_run(request: Request, run_id: int) -> EvalRunDetail:
    """Fetch one run's full detail (metrics + config + per-query)."""
    state: AppState = app_state(request)
    store = SqliteEvalStore(state.db)
    record = await store.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    detail = _to_detail(record)
    # Surface in-flight status so a client polling a just-started run sees it.
    task = _tasks.get(run_id)
    if task is not None and not task.done():
        detail.status = "running"
    return detail


# ── Helpers ─────────────────────────────────────────────────────────────────


def _resolve_profile(state: AppState, name: str) -> RetrievalProfile | None:
    """Resolve a profile name against user-saved + built-in profiles.

    Returns ``None`` for an unknown explicit name — the caller turns that
    into a 404; there is no silent fallback (an unset name is defaulted by
    the caller before this is reached).
    """
    try:
        blob = str(app_config.get(RETRIEVAL_PROFILES))
        from vesta.retrieval.profiles import load_user_profiles

        users = load_user_profiles(blob)
    except Exception:
        users = {}
    return resolve_profile(name, users)


def _placeholder_record(
    profile: RetrievalProfile,
    body: EvalRunRequest,
    *,
    started_at: str = "",
) -> RunRecord:
    from vesta.eval.metrics import LatencyPercentiles, RunMetrics

    golden = load_set(body.golden_set)
    return RunRecord(
        id=0,
        started_at=started_at,
        profile_name=profile.name,
        profile_hash=profile.hash,
        profile_yaml="",
        golden_hash=golden.hash,
        archive_path=str(EVAL_ARCHIVE_PATH.default),
        archive_checksum=str(EVAL_ARCHIVE_CHECKSUM.default),
        settings_snapshot={},
        git_sha="",
        machine_id="",
        metrics=RunMetrics({}, LatencyPercentiles(), False, (), 0),
        per_query=(),
        notes=body.notes or "running",
        status="running",
    )


async def _run_to_completion(
    state: AppState,
    store: SqliteEvalStore,
    profile: RetrievalProfile,
    body: EvalRunRequest,
    run_id: int,
    started_at: str,
) -> None:
    """Run the golden set, then overwrite the placeholder row with the result."""
    try:
        golden = load_set(body.golden_set)
        runner = LivePipelineRunner(state, profile)
        metrics, results = await evaluate_profile(profile, runner, golden)
        snapshot = app_config.strip_secret_values(app_config.snapshot().values)
        record = _build_record(
            profile,
            golden,
            metrics,
            results,
            snapshot=dict(snapshot),
            notes=body.notes,
            started_at=started_at,
        )
        await store.update_run(run_id, record)
    except Exception as exc:  # a failed run records its error, never crashes the API
        placeholder = _placeholder_record(profile, body, started_at=started_at)
        await store.update_run(
            run_id,
            replace(placeholder, id=run_id, notes=f"error: {exc!r}", status="error"),
        )
    finally:
        _tasks.pop(run_id, None)


def _build_record(
    profile: RetrievalProfile,
    golden: Any,
    metrics: Any,
    results: Any,
    *,
    snapshot: dict[str, object],
    notes: str,
    started_at: str,
) -> RunRecord:
    """Assemble a fully-populated RunRecord for an update_run write."""
    from vesta.eval.runner import RunRecord as _RR
    from vesta.retrieval.profiles import profile_to_yaml

    return _RR(
        id=0,
        started_at=started_at,
        profile_name=profile.name,
        profile_hash=profile.hash,
        profile_yaml=profile_to_yaml(profile),
        golden_hash=golden.hash,
        archive_path=str(EVAL_ARCHIVE_PATH.default),
        archive_checksum=str(EVAL_ARCHIVE_CHECKSUM.default),
        settings_snapshot=snapshot,
        git_sha=_runner_git_sha(),
        machine_id=_runner_machine_id(),
        metrics=metrics,
        per_query=tuple(r.to_dict() for r in results),
        notes=notes or "api",
    )


def _to_detail(record: RunRecord) -> EvalRunDetail:
    detail = EvalRunDetail(
        id=record.id,
        started_at=record.started_at,
        profile=record.profile_name,
        profile_hash=record.profile_hash,
        golden_hash=record.golden_hash,
        archive_checksum=record.archive_checksum,
        git_sha=record.git_sha,
        machine_id=record.machine_id,
        status=record.status,
        metrics=record.to_metrics_json(),
        config=record.to_config_json(),
    )
    return detail


async def reconcile_stale_eval_runs(db: Any) -> int:
    """Mark eval runs orphaned by a process death as ``error`` (startup).

    A ``POST /api/eval/run`` row is job-shaped: inserted with
    ``status == 'running'`` and owned by an asyncio task in this process's
    memory. After a restart every still-``running`` row is by definition
    orphaned — without the status column it would be indistinguishable from a
    legitimate all-zero retrieval run. Called once at startup alongside
    :func:`vesta.api.bench.reconcile_stale_bench_runs`.

    Returns how many runs were marked.
    """
    async with db.write() as conn:
        cur = await conn.execute("UPDATE eval_runs SET status='error' WHERE status='running'")
        return int(cur.rowcount or 0)


__all__ = ["LivePipelineRunner", "SqliteEvalStore", "reconcile_stale_eval_runs", "router"]
