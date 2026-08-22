"""Integration tests for the unified benchmark API (``/api/bench/*``).

Exercises the real ``SqliteBenchStore`` against an in-memory/tmp SQLite DB, the
REST routes against a live app, and the ``_drive_answer_events`` event-reduction
logic — all offline. The pipeline-heavy ``POST /run`` is exercised with a fake
``run_benchmark`` seam (monkeypatched) so no live LLM or ZIM archive is needed.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio

from vesta.answer.contracts import (
    CitationsEvent,
    DoneEvent,
    ErrorEvent,
    SourcesEvent,
    StatusEvent,
    TokenEvent,
    TraceEvent,
)
from vesta.api import bench
from vesta.eval.bench_dataset import BenchQuestion, BenchSource
from vesta.eval.bench_runner import BenchQuestionResult, BenchRunRecord

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def app_with_db(tmp_path: Path) -> AsyncIterator[tuple[httpx.AsyncClient, Any]]:
    """A live app + its ``app.state.vesta.db`` (migrations already applied)."""
    os.environ["data.dir"] = str(tmp_path)
    try:
        from vesta.main import create_app

        app = create_app()
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                yield client, app
    finally:
        os.environ.pop("data.dir", None)


def _q(qid: str, *, capability: str = "lookup", answer: str = "42") -> BenchQuestion:
    src = BenchSource(
        zim="wikipedia_en_top_nopic_2026-06.zim",
        article_title="A",
        article_path="A",
    )
    return BenchQuestion(
        id=qid,
        question=f"Question {qid}?",
        capability=capability,
        difficulty="easy",
        slice="core",
        expected_behavior="answer",
        answer=answer,
        sources=(src,),
        oracle={"model": "model-a", "verdict": "correct", "checked_at": "now"},
        closed_book={"model": "model-a", "verdict": "incorrect", "checked_at": "now"},
    )


def _run_record(
    *, system: str, run_group: str = "group-1", status: str = "complete"
) -> BenchRunRecord:
    return BenchRunRecord(
        run_group=run_group,
        label="test",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:01:00+00:00",
        status=status,
        dataset_name="vesta_test",
        dataset_hash="abc123",
        subset_hash="",
        system=system,
        profile_name="profile-x",
        profile_hash="phash",
        answer_model="model-a",
        judge_model="judge-b",
        trusted=True,
        calibration=0.9,
        metrics_json={
            "answer": {
                "strict_accuracy": 1.0,
                "weighted_accuracy": 1.0,
                "unjudged": 0,
                "complete": True,
            },
            "source": {"recall_at_10": 1.0, "source_coverage": 1.0},
            "reference": {"headroom_realised": 1.0, "ceiling": 1, "system": 1, "floor": 0},
            "attribution": {
                "correct_source_found": 1,
                "correct_source_missed": 0,
                "failed_source_found": 0,
                "failed_source_missed": 0,
            },
            "by_capability": {"lookup": {"strict_accuracy": 1.0, "n": 1}},
        },
    )


def _result_row(run_id: int, qid: str, verdict: str, shr: int | None = 1) -> BenchQuestionResult:
    return BenchQuestionResult(
        run_id=run_id,
        question_id=qid,
        capability="lookup",
        difficulty="easy",
        question_text=f"Question {qid}?",
        expected_answer="42",
        answer_text="42",
        abstained=False,
        verdict=verdict,
        source_hit_rank=shr,
        source_coverage=1.0,
        retrieved_paths=("A", "B"),
        trace={"stages": [{"name": "answer"}]},
    )


async def _seed_run(app: Any, *, system: str, verdicts: dict[str, str]) -> int:
    """Insert a run + its per-question rows directly into the app's DB."""
    from vesta.api.bench import SqliteBenchStore

    store = SqliteBenchStore(app.state.vesta.db)
    run_id = await store.insert_run(_run_record(system=system))
    for qid, v in verdicts.items():
        await store.insert_question_result(run_id, _result_row(run_id, qid, v))
    return run_id


# ── POST /api/bench/run (offline-safe: fake run_benchmark seam) ─────────────


@pytest.mark.asyncio
async def test_run_and_list(
    app_with_db: tuple[httpx.AsyncClient, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _app = app_with_db
    data_dir = Path(os.environ["data.dir"])
    dataset_path = data_dir / "tiny_bench.json"
    dataset_path.write_text(
        json.dumps(
            {
                "name": "tiny",
                "version": 1,
                "questions": [
                    {
                        "id": "q1",
                        "question": "What is 42?",
                        "capability": "lookup",
                        "difficulty": "easy",
                        "slice": "core",
                        "expected_behavior": "answer",
                        "answer": "42",
                        "sources": [
                            {
                                "zim": "wikipedia_en_top_nopic_2026-06.zim",
                                "article_title": "A",
                                "article_path": "A",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    # Fake the judge (none) and the orchestration seam so no live LLM/archive
    # runs. The fake grades every question correct and completes the pre-seeded
    # rows through the real store (via the pre-seeded wrapper).
    monkeypatch.setattr("vesta.api.bench.make_judge_llm", lambda _state, _m: (None, None))

    completed: list[int] = []

    async def _fake_run_benchmark(
        *,
        dataset,
        questions,
        systems,
        store,
        judge,
        judge_model,
        run_group,
        label,
        scope,
        config_snapshot,
        judge_concurrency,
        judge_shares_endpoint,
        repeats,
        max_concurrent,
        progress,
        level=None,
    ):
        from vesta.eval.bench_dataset import subset_hash
        from vesta.eval.bench_scoring import Verdict as V

        sub = subset_hash(list(questions))
        records = []
        for _ in range(repeats):
            for system in systems:
                run_id = await store.insert_run(
                    BenchRunRecord(
                        run_group=run_group,
                        label=label,
                        started_at="2026-01-01T00:00:00+00:00",
                        status="running",
                        dataset_name=dataset.name,
                        dataset_hash=dataset.hash,
                        subset_hash=sub,
                        system=system.name,
                        profile_name=getattr(system, "profile_name", ""),
                        profile_hash="",
                        answer_model=getattr(system, "answer_model", ""),
                        judge_model=judge_model,
                        scope=scope,
                    )
                )
                for q in questions:
                    await store.insert_question_result(
                        run_id,
                        BenchQuestionResult(
                            run_id=run_id,
                            question_id=q.id,
                            capability=q.capability,
                            difficulty=q.difficulty,
                            question_text=q.question,
                            expected_answer=q.answer,
                            answer_text="yes",
                            abstained=False,
                            verdict=V.CORRECT.value,
                            source_hit_rank=1,
                            source_coverage=1.0,
                        ),
                    )
                    if progress is not None:
                        progress(
                            type(
                                "PU",
                                (),
                                {
                                    "system": system.name,
                                    "run_id": run_id,
                                    "stage": "judging",
                                    "done": 1,
                                    "total": len(questions),
                                },
                            )()
                        )
                final = BenchRunRecord(
                    run_group=run_group,
                    label=label,
                    started_at="2026-01-01T00:00:00+00:00",
                    finished_at="2026-01-01T00:01:00+00:00",
                    status="complete",
                    dataset_name=dataset.name,
                    dataset_hash=dataset.hash,
                    subset_hash=sub,
                    system=system.name,
                    profile_name=getattr(system, "profile_name", ""),
                    profile_hash="",
                    answer_model=getattr(system, "answer_model", ""),
                    judge_model=judge_model,
                    scope=scope,
                    id=run_id,
                    config_json={"git_sha": "abc", "machine_id": "test"},
                    metrics_json={
                        "answer": {
                            "strict_accuracy": 1.0,
                            "weighted_accuracy": 1.0,
                            "unjudged": 0,
                            "complete": True,
                        },
                        "source": {"recall_at_10": 1.0, "source_coverage": 1.0},
                        "reference": {
                            "headroom_realised": 1.0,
                            "ceiling": 1,
                            "system": 1,
                            "floor": 0,
                        },
                        "attribution": {
                            "correct_source_found": len(questions),
                            "correct_source_missed": 0,
                            "failed_source_found": 0,
                            "failed_source_missed": 0,
                        },
                        "by_capability": {"lookup": {"strict_accuracy": 1.0, "n": len(questions)}},
                    },
                )
                await store.update_run(run_id, final)
                completed.append(run_id)
                records.append(final)
        return records

    monkeypatch.setattr("vesta.api.bench.run_benchmark", _fake_run_benchmark)

    resp = await client.post(
        "/api/bench/run",
        json={
            "systems": ["sources_only"],
            "models": ["model-a"],
            "dataset": str(dataset_path),
            "judge_model": "judge-b",
            "label": "smoke",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "running"
    assert body["run_group"]
    assert len(body["run_ids"]) == 1
    run_id = int(body["run_ids"][0])
    assert body["matrix_size"] == 1
    assert body["dataset_name"] == "tiny"

    # Background task completes the pre-seeded run.
    deadline = asyncio.get_event_loop().time() + 15.0
    while asyncio.get_event_loop().time() < deadline:
        detail = (await client.get(f"/api/bench/runs/{run_id}")).json()
        if detail["status"] != "running":
            break
        await asyncio.sleep(0.1)
    assert detail["status"] == "complete"
    assert detail["metrics"]["answer"]["strict_accuracy"] == 1.0

    # List surfaces it with the at-a-glance chips.
    listing = (await client.get("/api/bench/runs")).json()
    assert any(r["id"] == run_id for r in listing)
    row = next(r for r in listing if r["id"] == run_id)
    assert row["system"] == "sources_only"
    assert row["strict_accuracy"] == 1.0


@pytest.mark.asyncio
async def test_run_invalid_system_400(
    app_with_db: tuple[httpx.AsyncClient, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = app_with_db
    monkeypatch.setattr("vesta.api.bench.make_judge_llm", lambda _s, _m: (None, None))
    resp = await client.post(
        "/api/bench/run", json={"systems": ["does_not_exist"], "models": ["m"]}
    )
    assert resp.status_code == 400


# ── GET /runs/{id} detail ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_detail(app_with_db: tuple[httpx.AsyncClient, Any]) -> None:
    client, app = app_with_db
    run_id = await _seed_run(app, system="sources_only", verdicts={"q1": "correct"})
    resp = await client.get(f"/api/bench/runs/{run_id}")
    assert resp.status_code == 200
    detail = resp.json()
    assert detail["system"] == "sources_only"
    assert detail["metrics"]["answer"]["strict_accuracy"] == 1.0
    assert detail["metrics"]["reference"]["headroom_realised"] == 1.0
    assert detail["metrics"]["attribution"]["correct_source_found"] == 1
    assert detail["metrics"]["by_capability"]["lookup"]["strict_accuracy"] == 1.0


@pytest.mark.asyncio
async def test_run_detail_404(app_with_db: tuple[httpx.AsyncClient, Any]) -> None:
    client, _ = app_with_db
    assert (await client.get("/api/bench/runs/999999")).status_code == 404


# ── GET /runs/{id}/results (paginated + filterable, never trace_json) ──────


@pytest.mark.asyncio
async def test_run_results_paginated_and_filtered(
    app_with_db: tuple[httpx.AsyncClient, Any],
) -> None:
    client, app = app_with_db
    run_id = await _seed_run(
        app, system="sources_only", verdicts={"q1": "correct", "q2": "incorrect", "q3": "correct"}
    )
    # Default page: all rows ordered by question_id.
    page = (await client.get(f"/api/bench/runs/{run_id}/results")).json()
    assert page["total"] == 3
    assert [r["question_id"] for r in page["items"]] == ["q1", "q2", "q3"]
    # trace_json must never be present (trap 10).
    assert "trace_json" not in page["items"][0]
    assert "trace" not in page["items"][0]

    # Filter by verdict.
    correct = (
        await client.get(f"/api/bench/runs/{run_id}/results", params={"verdict": "correct"})
    ).json()
    assert correct["total"] == 2

    # Pagination.
    page2 = (
        await client.get(f"/api/bench/runs/{run_id}/results", params={"offset": 1, "limit": 1})
    ).json()
    assert page2["total"] == 3
    assert [r["question_id"] for r in page2["items"]] == ["q2"]

    # Attribution-cell filter: correct_source_missed → none (all rows have rank 1).
    missed = (
        await client.get(
            f"/api/bench/runs/{run_id}/results", params={"attribution": "correct_source_missed"}
        )
    ).json()
    assert missed["total"] == 0

    # Unknown attribution cell → 400.
    bad = await client.get(f"/api/bench/runs/{run_id}/results", params={"attribution": "nope"})
    assert bad.status_code == 400


# ── GET /compare — four buckets + shared denominator ────────────────────────


@pytest.mark.asyncio
async def test_compare_four_buckets(app_with_db: tuple[httpx.AsyncClient, Any]) -> None:
    client, app = app_with_db
    # q1: a correct → b wrong (broken); q2: a wrong → b correct (fixed);
    # q3: both correct; q4: both wrong; q5: only in b.
    run_a = await _seed_run(
        app,
        system="system-a",
        verdicts={"q1": "correct", "q2": "incorrect", "q3": "correct", "q4": "incorrect"},
    )
    run_b = await _seed_run(
        app,
        system="system-b",
        verdicts={
            "q1": "incorrect",
            "q2": "correct",
            "q3": "correct",
            "q4": "incorrect",
            "q5": "correct",
        },
    )
    resp = await client.get("/api/bench/compare", params={"runs": f"{run_a},{run_b}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["runs"] == [run_a, run_b]
    assert len(body["pairs"]) == 1
    pair = body["pairs"][0]
    assert pair["run_a"] == run_a
    assert pair["run_b"] == run_b
    assert pair["shared_denominator"] == 4
    assert pair["broken"] == ["q1"]
    assert pair["fixed"] == ["q2"]
    assert pair["both_correct"] == ["q3"]
    assert pair["both_wrong"] == ["q4"]
    assert pair["only_b"] == ["q5"]


@pytest.mark.asyncio
async def test_compare_requires_at_least_two_runs(
    app_with_db: tuple[httpx.AsyncClient, Any],
) -> None:
    client, app = app_with_db
    run_id = await _seed_run(app, system="system-a", verdicts={"q1": "correct"})
    assert (await client.get("/api/bench/compare", params={"runs": str(run_id)})).status_code == 400
    assert (await client.get("/api/bench/compare", params={"runs": "999,1000"})).status_code == 404


# ── GET /dataset ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dataset_info(app_with_db: tuple[httpx.AsyncClient, Any]) -> None:
    client, _ = app_with_db
    data_dir = Path(os.environ["data.dir"])
    dataset_path = data_dir / "ds.json"
    dataset_path.write_text(
        json.dumps(
            {
                "name": "ds",
                "version": 1,
                "questions": [
                    {
                        "id": "a",
                        "question": "Qa",
                        "capability": "lookup",
                        "difficulty": "easy",
                        "slice": "core",
                        "expected_behavior": "answer",
                        "answer": "A",
                        "sources": [{"zim": "z", "article_title": "T", "article_path": "T"}],
                        "oracle": {"model": "m", "verdict": "correct"},
                        "closed_book": {"model": "m", "verdict": "incorrect"},
                    },
                    {
                        "id": "b",
                        "question": "Qb",
                        "capability": "multi_hop_cross_article",
                        "difficulty": "hard",
                        "slice": "cross",
                        "expected_behavior": "answer",
                        "answer": "B",
                        "sources": [{"zim": "z", "article_title": "U", "article_path": "U"}],
                        "oracle": {"model": "m", "verdict": "incorrect"},
                        "closed_book": {"model": "m", "verdict": "incorrect"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    resp = await client.get("/api/bench/dataset", params={"path": str(dataset_path)})
    assert resp.status_code == 200
    info = resp.json()
    assert info["name"] == "ds"
    assert info["total"] == 2
    assert info["by_capability"]["lookup"] == 1
    assert info["by_difficulty"]["easy"] == 1
    assert info["by_slice"]["core"] == 1
    # ceiling = 1/2 correct oracle; floor = 0/2.
    assert info["ceiling"]["correct"] == 1
    assert info["ceiling"]["score"] == 0.5
    assert info["floor"]["score"] == 0.0


# ── DELETE /runs/{id} cascade ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_run_cascades(app_with_db: tuple[httpx.AsyncClient, Any]) -> None:
    client, app = app_with_db
    run_id = await _seed_run(
        app, system="sources_only", verdicts={"q1": "correct", "q2": "incorrect"}
    )

    resp = await client.delete(f"/api/bench/runs/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == run_id

    # Run gone.
    assert (await client.get(f"/api/bench/runs/{run_id}")).status_code == 404
    # Per-question rows cascaded (FK ON DELETE CASCADE).
    from vesta.api.bench import SqliteBenchStore

    store = SqliteBenchStore(app.state.vesta.db)
    assert await store.list_question_results(run_id) == []
    # Re-delete → 404.
    assert (await client.delete(f"/api/bench/runs/{run_id}")).status_code == 404


# ── Driver event reduction (adapted from the old InProcessAnswerDriver) ─────


class _FakeAppState:
    """Minimal stand-in for AppState — the driver only passes it through to
    ``iter_answer_events``, which we monkeypatch."""

    def __init__(self) -> None:
        self.db = None
        self.runner = None
        self.registry = None
        self.encoders = None
        self.gateway = None
        self.supervisor = None


def _make_events(answer: str, paths: tuple[str, ...], abstain: bool = False) -> list[object]:
    events: list[object] = []
    from vesta.retrieval.contracts import SourceCard

    cards = tuple(
        SourceCard(zim_id=1, path=p, title="", snippet="", breadcrumb="", score=0.5, source="")
        for p in paths
    )
    events.append(SourcesEvent(cards=cards))
    if abstain:
        events.append(StatusEvent(phase="abstaining", detail="no candidates"))
        events.append(TokenEvent(text="No passage in your archives closely matches this query."))
    else:
        events.append(StatusEvent(phase="generating", detail=""))
        for word in answer.split():
            events.append(TokenEvent(text=word + " "))
    events.append(TraceEvent(trace={"stages": [{"name": "answer", "duration_ms": 100}]}))
    events.append(DoneEvent())
    return events


@pytest.mark.asyncio
async def test_driver_reduces_events_to_output(monkeypatch: pytest.MonkeyPatch) -> None:
    state: Any = _FakeAppState()

    async def _fake_iter(_state, q, scope, profile, strategy, **kw):
        for ev in _make_events("The answer is 42", ("Correct_Article", "Other")):
            yield ev

    monkeypatch.setattr("vesta.api.bench.iter_answer_events", _fake_iter)
    out = await bench._drive_answer_events(
        state, "test question", profile=None, scope=None, strategy="sources_only"
    )
    assert out.answer_text == "The answer is 42 "
    assert out.retrieved_paths == ("Correct_Article", "Other")
    assert out.abstained is False
    assert out.error is None


@pytest.mark.asyncio
async def test_driver_prefers_citations_answer_text(monkeypatch: pytest.MonkeyPatch) -> None:
    state: Any = _FakeAppState()

    async def _fake_iter(_state, q, scope, profile, strategy, **kw):
        yield SourcesEvent(cards=())
        yield TokenEvent(text="The battle happened in 1066 [7].")
        yield CitationsEvent(spans=(), answer_text="The battle happened in 1066 [1].")
        yield TraceEvent(trace={"stages": []})
        yield DoneEvent()

    monkeypatch.setattr("vesta.api.bench.iter_answer_events", _fake_iter)
    out = await bench._drive_answer_events(
        state, "test", profile=None, scope=None, strategy="sources_only"
    )
    assert out.answer_text == "The battle happened in 1066 [1]."


@pytest.mark.asyncio
async def test_driver_detects_abstention(monkeypatch: pytest.MonkeyPatch) -> None:
    from vesta.answer.abstention import ABSTENTION_NO_MATCH

    state: Any = _FakeAppState()

    async def _fake_iter(_state, q, scope, profile, strategy, **kw):
        for ev in _make_events(ABSTENTION_NO_MATCH, (), abstain=True):
            yield ev

    monkeypatch.setattr("vesta.api.bench.iter_answer_events", _fake_iter)
    out = await bench._drive_answer_events(
        state, "test", profile=None, scope=None, strategy="sources_only"
    )
    assert out.abstained is True


@pytest.mark.asyncio
async def test_driver_captures_error(monkeypatch: pytest.MonkeyPatch) -> None:
    state: Any = _FakeAppState()

    async def _fake_iter(_state, q, scope, profile, strategy, **kw):
        yield SourcesEvent(cards=())
        yield ErrorEvent(code="budget_exhausted", message="ran out", recoverable=True)
        yield DoneEvent()

    monkeypatch.setattr("vesta.api.bench.iter_answer_events", _fake_iter)
    out = await bench._drive_answer_events(
        state, "test", profile=None, scope=None, strategy="sources_only"
    )
    assert out.error is not None
    assert "budget_exhausted" in out.error


# ── GatewayJudgeLLM ─────────────────────────────────────────────────────────


class _FakeGateway:
    def __init__(self, text: str) -> None:
        self._text = text
        self.last_kwargs: dict[str, object] = {}

    async def chat_once(
        self, messages, *, model, temperature, max_tokens, enable_thinking=None, timeout=None
    ):
        self.last_kwargs = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "enable_thinking": enable_thinking,
        }
        from vesta.inference.gateway import ChatResult

        return ChatResult(text=self._text, finish_reason="stop", latency_ms=10.0)

    async def chat_stream(
        self, messages, *, model, temperature, max_tokens, enable_thinking=None, timeout=None
    ):
        self.last_kwargs = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "enable_thinking": enable_thinking,
        }
        from vesta.inference.gateway import ChatDelta

        yield ChatDelta(text=self._text, finish_reason="stop")
        yield ChatDelta(
            text="", finish_reason=None, input_tokens=1, output_tokens=1, total_tokens=2
        )


@pytest.mark.asyncio
async def test_gateway_judge_forwards_enable_thinking_false() -> None:
    gw = _FakeGateway("correct | match")
    judge = bench.GatewayJudgeLLM(gw, "test-model", temperature=0.0, max_tokens=64)
    result = await judge.judge("Judge this: ...")
    assert result == "correct | match"
    assert gw.last_kwargs["enable_thinking"] is False
    assert gw.last_kwargs["temperature"] == 0.0
    assert gw.last_kwargs["max_tokens"] == 64


# ── make_judge_llm ──────────────────────────────────────────────────────────


def _configure_judge(values: dict[str, str]) -> None:
    from vesta import config

    config.configure(env=values, db_values={})


@pytest.mark.asyncio
async def test_make_judge_llm_reuses_main_gateway_when_no_judge_endpoint() -> None:
    from vesta import config

    _configure_judge({"eval.judge.model": "judge-model", "eval.judge.endpoint_url": ""})
    main_gw = _FakeGateway("correct | ok")
    state = _FakeAppState()
    state.gateway = main_gw
    try:
        judge, owned = bench.make_judge_llm(state, "judge-model")
        assert judge is not None
        assert owned is None
        assert await judge.judge("Judge: ...") == "correct | ok"
        assert main_gw.last_kwargs["model"] == "judge-model"
    finally:
        config.reset_for_test()


@pytest.mark.asyncio
async def test_make_judge_llm_no_judge_model_returns_none() -> None:
    from vesta import config

    _configure_judge({"eval.judge.model": ""})
    state = _FakeAppState()
    state.gateway = _FakeGateway("x")
    try:
        judge, owned = bench.make_judge_llm(state, "")
        assert judge is None
        assert owned is None
    finally:
        config.reset_for_test()


# ── _to_summary serializer ──────────────────────────────────────────────────


def test_to_summary_carries_score_chips() -> None:
    summary = bench._to_summary(_run_record(system="sources_only"))
    assert summary.system == "sources_only"
    assert summary.strict_accuracy == 1.0
    assert summary.headroom == 1.0
    assert summary.source_recall_at_10 == 1.0
    assert summary.trusted is True
