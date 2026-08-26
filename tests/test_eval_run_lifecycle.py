"""Regression tests for the eval-run row lifecycle (AUDIT_0824 M5).

Three behaviors, each previously broken:

1. ``SqliteEvalStore.update_run`` never touched ``started_at``, so every
   API-created run kept the placeholder's empty string forever and rendered
   '—' in the UI. The UPDATE now persists it.
2. ``EvalRunDetail.status`` defaulted to ``"done"`` and nothing ever derived
   ``"error"`` — a failed run was recorded only inside ``config.notes``. A
   failed run now persists ``status='error'``.
3. No startup reconciliation existed, so a crashed run's stranded
   ``'running'`` row was indistinguishable from a legitimate all-zero
   retrieval run. The sweep flips orphaned rows to ``error`` at startup,
   mirroring ``reconcile_stale_bench_runs``.
"""

from __future__ import annotations

from typing import Any, cast

import httpx
import pytest

from vesta import config as app_config
from vesta.api import eval as eval_api
from vesta.db.connection import Database
from vesta.db.migrations import run_migrations
from vesta.eval.golden import load_set
from vesta.eval.metrics import LatencyPercentiles, RunMetrics
from vesta.eval.runner import PipelineRunner
from vesta.retrieval.profiles import load_profile

_STARTED = "2026-08-25T00:00:00+00:00"


@pytest.fixture
def _settings() -> None:
    """The settings registry must be configured before snapshot() works."""
    app_config.configure(env={})


@pytest.fixture
async def db(tmp_path: Any) -> Any:
    database = Database(str(tmp_path / "eval_lifecycle.db"), busy_timeout_ms=1000)
    await database.start()
    async with database.write() as conn:
        await run_migrations(conn)
    yield database
    await database.stop()


class _OkRunner(PipelineRunner):
    """Never called — evaluate_profile is patched over it."""

    async def run(self, profile: Any, query: str) -> tuple[tuple[str, ...], dict[str, object]]:
        return (), {"stages": [], "degradations": []}


def _zero_metrics() -> RunMetrics:
    return RunMetrics({}, LatencyPercentiles(), False, (), 0)


def _body() -> eval_api.EvalRunRequest:
    return eval_api.EvalRunRequest(golden_set="fixture_subset", notes="")


# ── 1. started_at survives completion ───────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.usefixtures("_settings")
async def test_completed_run_keeps_started_at(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = eval_api.SqliteEvalStore(db)
    profile = load_profile("lexical")
    assert profile is not None
    placeholder = eval_api._placeholder_record(profile, _body(), started_at=_STARTED)
    run_id = await store.insert_run(placeholder)

    # The POST path stamps the real start time + running status immediately.
    inserted = await store.get_run(run_id)
    assert inserted is not None
    assert inserted.started_at == _STARTED
    assert inserted.status == "running"

    async def fake_evaluate(*args: Any, **kwargs: Any) -> tuple[RunMetrics, tuple[(), ...]]:
        return _zero_metrics(), ()

    monkeypatch.setattr(eval_api, "LivePipelineRunner", lambda state, profile: _OkRunner())
    monkeypatch.setattr(eval_api, "evaluate_profile", fake_evaluate)
    await eval_api._run_to_completion(
        object(),
        store,
        profile,
        _body(),
        run_id,
        _STARTED,  # type: ignore[arg-type]
    )

    done = await store.get_run(run_id)
    assert done is not None
    assert done.started_at == _STARTED
    assert done.status == "done"
    detail = eval_api._to_detail(done)
    assert detail.started_at == _STARTED
    assert detail.status == "done"


# ── 2. A failed run reports error, not done ─────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.usefixtures("_settings")
async def test_failed_run_reports_error(db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    store = eval_api.SqliteEvalStore(db)
    profile = load_profile("lexical")
    assert profile is not None
    placeholder = eval_api._placeholder_record(profile, _body(), started_at=_STARTED)
    run_id = await store.insert_run(placeholder)

    async def boom(*args: Any, **kwargs: Any) -> tuple[RunMetrics, tuple[(), ...]]:
        raise RuntimeError("pipeline exploded")

    monkeypatch.setattr(eval_api, "LivePipelineRunner", lambda state, profile: _OkRunner())
    monkeypatch.setattr(eval_api, "evaluate_profile", boom)
    await eval_api._run_to_completion(
        object(),
        store,
        profile,
        _body(),
        run_id,
        _STARTED,  # type: ignore[arg-type]
    )

    failed = await store.get_run(run_id)
    assert failed is not None
    assert failed.status == "error"
    assert failed.started_at == _STARTED
    assert "RuntimeError" in failed.notes
    # The DTO surfaces it — no silent 'done' default on a failure.
    assert eval_api._to_detail(failed).status == "error"


# ── 3. Startup sweep reconciles stranded placeholders ───────────────────────


@pytest.mark.asyncio
async def test_startup_reconcile_flips_orphaned_running_rows(db: Database) -> None:
    store = eval_api.SqliteEvalStore(db)
    profile = load_profile("lexical")
    assert profile is not None

    stranded_a = await store.insert_run(
        eval_api._placeholder_record(profile, _body(), started_at=_STARTED)
    )
    stranded_b = await store.insert_run(
        eval_api._placeholder_record(profile, _body(), started_at=_STARTED)
    )
    finished = await store.insert_run(
        eval_api._placeholder_record(profile, _body(), started_at=_STARTED)
    )
    done_record = eval_api._build_record(
        profile,
        load_set("fixture_subset"),
        _zero_metrics(),
        (),
        snapshot={},
        notes="",
        started_at=_STARTED,
    )
    await store.update_run(finished, done_record)

    errored = await eval_api.reconcile_stale_eval_runs(db)
    assert errored == 2

    for run_id, expected in (
        (stranded_a, "error"),
        (stranded_b, "error"),
        (finished, "done"),  # terminal rows are never touched
    ):
        rec = await store.get_run(run_id)
        assert rec is not None
        assert rec.status == expected


# ── Legacy rows predate the status column ───────────────────────────────────


@pytest.mark.asyncio
async def test_legacy_row_without_status_reads_done(db: Database) -> None:
    async with db.write() as conn:
        await conn.execute("INSERT INTO eval_runs(started_at) VALUES('2025-01-01T00:00:00+00:00')")
    store = eval_api.SqliteEvalStore(db)
    runs = await store.list_runs()
    assert len(runs) == 1
    assert runs[0].status == "done"


# ── AUDIT_0824 B5: unknown profile errors instead of silent lexical ────────


@pytest.mark.usefixtures("_settings")
def test_resolve_profile_returns_none_for_unknown() -> None:
    """The resolver returns ``None`` for an unknown explicit name (the route
    turns that into a 404) and still resolves built-ins."""
    known = eval_api._resolve_profile(cast(Any, None), "lexical")
    assert known is not None
    assert known.name == "lexical"
    assert eval_api._resolve_profile(cast(Any, None), "no_such_profile") is None


@pytest.mark.asyncio
async def test_run_unknown_profile_is_404(app_client: httpx.AsyncClient) -> None:
    """POST /api/eval/run with a bogus profile must 404 — pre-fix it silently
    ran the lexical profile and reported it as the requested one."""
    resp = await app_client.post(
        "/api/eval/run",
        json={"profile": "no_such_profile", "golden_set": "fixture_subset"},
    )
    assert resp.status_code == 404
    assert "no_such_profile" in resp.json()["detail"]
