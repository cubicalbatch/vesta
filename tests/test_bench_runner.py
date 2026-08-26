"""Unified benchmark runner + store tests.

Offline: fakes for the SystemUnderTest and the JudgeLLM; a real (temp-file)
SqliteBenchStore for the persistence half. Covers the two-stage flow, matrix
expansion, cascade delete, judge cache, compare query, rejudge, and the
concurrency-invariance trap.
"""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from vesta.db.connection import Database
from vesta.db.migrations import run_migrations
from vesta.eval.bench_dataset import (
    BenchDataset,
    BenchQuestion,
    BenchSource,
    dataset_hash,
)
from vesta.eval.bench_runner import (
    BenchQuestionResult,
    BenchRunRecord,
    CompareResult,
    IncomparableRuns,
    QuestionOutput,
    _compute_metrics,
    _rebuild_scored,
    compare_runs,
    rejudge_run,
    run_benchmark,
)
from vesta.eval.bench_scoring import (
    JudgeOutcome,
    Verdict,
    aggregate_answer_metrics,
    judge_cache_key,
    render_rubric,
    score_question,
)

# ── Fixtures ──────────────────────────────────────────────────────────────


def _q(
    qid: str,
    *,
    capability: str = "lookup",
    answer: str = "42",
    sources: bool = True,
    expected_behavior: str = "answer",
) -> BenchQuestion:
    srcs = (
        (
            BenchSource(
                zim="wikipedia_en_top_nopic_2026-06.zim",
                article_title="A",
                article_path="A",
            ),
        )
        if sources
        else ()
    )
    return BenchQuestion(
        id=qid,
        question=f"Question {qid}?",
        capability=capability,
        difficulty="easy",
        slice="core",
        expected_behavior=expected_behavior,
        answer=answer,
        sources=srcs,
    )


def _dataset(qs: tuple[BenchQuestion, ...]) -> BenchDataset:
    return BenchDataset(
        name="vesta_test",
        version=1,
        questions=qs,
        hash=dataset_hash(qs),
    )


class FakeSUT:
    """A controllable SystemUnderTest."""

    name = "fake"
    answer_model = "model-a"
    profile_name = "profile-x"
    profile_hash = "phash"

    def __init__(
        self, answers: dict[str, str] | None = None, call_count: list[int] | None = None
    ) -> None:
        self._answers = answers or {}
        self._calls = call_count if call_count is not None else []

    async def run_one(self, q: BenchQuestion) -> QuestionOutput:
        self._calls.append(q.id)
        ans = self._answers.get(q.id, "generic-answer")
        return QuestionOutput(
            answer_text=ans,
            retrieved_paths=("A", "B", "C"),
            abstained=False,
            error=None,
            trace={"stages": [{"name": "answer", "component": "sources_only"}]},
            resolved_strategy="fake",
            rounds=1,
        )


class FailSUT:
    """A SUT that fails one question."""

    name = "fail"
    answer_model = "model-a"
    profile_name = "profile-x"
    profile_hash = "phash"

    def __init__(self, fail_id: str, call_count: list[int] | None = None) -> None:
        self._fail_id = fail_id
        self._calls = call_count if call_count is not None else []

    async def run_one(self, q: BenchQuestion) -> QuestionOutput:
        self._calls.append(q.id)
        if q.id == self._fail_id:
            raise RuntimeError("boom")
        return QuestionOutput(
            answer_text="ok",
            retrieved_paths=("A",),
            abstained=False,
            error=None,
            trace={},
            resolved_strategy="fail",
        )


class GateBoomSUT:
    """A SUT that fails fatally once a sibling cell has started."""

    name = "boom"
    answer_model = "model-a"
    profile_name = "profile-x"
    profile_hash = "phash"

    def __init__(self, started: asyncio.Event) -> None:
        self._started = started

    async def run_one(self, q: BenchQuestion) -> QuestionOutput:
        await self._started.wait()
        raise RuntimeError("boom")


class SlowSUT:
    """A SUT that answers slowly and records how far it got."""

    name = "slow"
    answer_model = "model-a"
    profile_name = "profile-x"
    profile_hash = "phash"
    generates_answers = False

    def __init__(self, started: asyncio.Event, calls: list[str]) -> None:
        self._started = started
        self.calls = calls

    async def run_one(self, q: BenchQuestion) -> QuestionOutput:
        await asyncio.sleep(0.05)
        self.calls.append(q.id)
        if not self._started.is_set():
            self._started.set()
        return QuestionOutput(
            answer_text="slow",
            retrieved_paths=("A",),
            abstained=False,
            error=None,
            trace={},
            resolved_strategy="slow",
        )


class FakeJudge:
    """A deterministic JudgeLLM: correct iff the answer says 'yes'."""

    def __init__(self, calls: list[str] | None = None) -> None:
        self._calls = calls if calls is not None else []

    async def judge(self, prompt: str) -> str:
        self._calls.append(prompt)
        if "generic-answer" in prompt or "ok" in prompt or "yes" in prompt:
            verdict = "correct"
        else:
            verdict = "incorrect"
        return json.dumps({"verdict": verdict, "reason": "r", "abstained": False})


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = Database(str(tmp_path / "test.db"), busy_timeout_ms=1000)
    await database.start()
    async with database.write() as conn:
        await run_migrations(conn)
    yield database
    await database.stop()


@pytest.fixture
def store(db: Database):
    from vesta.api.bench import SqliteBenchStore

    return SqliteBenchStore(db)


def _make_run_record() -> BenchRunRecord:
    return BenchRunRecord(
        run_group="group-1",
        label="test run",
        started_at="2026-01-01T00:00:00+00:00",
        status="running",
        dataset_name="vesta_test",
        dataset_hash="abc",
        subset_hash="",
        system="fake",
        profile_name="profile-x",
        profile_hash="phash",
        answer_model="model-a",
        judge_model="judge-b",
    )


# ── Store round-trip + cascade delete ───────────────────────────────────────


@pytest.mark.asyncio
async def test_store_round_trip(store) -> None:
    rec = _make_run_record()
    run_id = await store.insert_run(rec)
    assert run_id > 0

    got = await store.get_run(run_id)
    assert got is not None
    assert got.run_group == "group-1"
    assert got.system == "fake"
    assert got.status == "running"

    rows = await store.list_runs()
    assert len(rows) == 1
    assert rows[0].id == run_id

    # Insert a question row; update pending → correct.
    qrow = BenchQuestionResult(
        run_id=run_id,
        question_id="q1",
        capability="lookup",
        difficulty="easy",
        question_text="Q1",
        expected_answer="42",
        answer_text="42",
        abstained=False,
        verdict=Verdict.PENDING.value,
        retrieved_paths=("A",),
        source_hit_rank=1,
        source_coverage=1.0,
    )
    await store.insert_question_result(run_id, qrow)
    qrows = await store.list_question_results(run_id)
    assert len(qrows) == 1
    assert qrows[0].verdict == Verdict.PENDING.value

    updated = BenchQuestionResult(
        **{
            **qrow.__dict__,
            "verdict": Verdict.CORRECT.value,
            "verdict_reason": "r",
        }
    )
    ok = await store.update_question_result(run_id, "q1", updated)
    assert ok
    reloaded = await store.list_question_results(run_id)
    assert reloaded[0].verdict == Verdict.CORRECT.value

    pending = await store.list_pending_results(run_id)
    assert pending == []


@pytest.mark.asyncio
async def test_cascade_delete(store) -> None:
    run_id = await store.insert_run(_make_run_record())
    qrow = BenchQuestionResult(
        run_id=run_id,
        question_id="q1",
        capability="lookup",
        difficulty="easy",
        question_text="Q",
        expected_answer="A",
        answer_text="A",
        abstained=False,
        verdict=Verdict.PENDING.value,
    )
    await store.insert_question_result(run_id, qrow)
    assert await store.list_question_results(run_id)

    ok = await store.delete_run(run_id)
    assert ok
    assert await store.get_run(run_id) is None
    assert await store.list_question_results(run_id) == []


@pytest.mark.asyncio
async def test_abort_reason_and_judge_flag_survive_round_trip(store) -> None:
    """AUDIT_0824: abort_reason and judge_shares_endpoint live in config_json
    (no dedicated columns) and must survive the persistence round-trip."""
    rec = replace(_make_run_record(), config_json={"judge_shares_endpoint": True})
    run_id = await store.insert_run(rec)

    got = await store.get_run(run_id)
    assert got is not None
    assert got.judge_shares_endpoint is True

    # mark_aborted stashes the reason; get_run must lift it back.
    assert await store.mark_aborted(run_id, "process restarted mid-run") is True
    aborted = await store.get_run(run_id)
    assert aborted is not None
    assert aborted.status == "aborted"
    assert aborted.abort_reason == "process restarted mid-run"

    # The failed-cell path (status='failed', reason stashed in config_json)
    # keeps its reason through a reload.
    failed_rec = replace(
        rec,
        status="failed",
        abort_reason="ValueError: boom",
        config_json={**rec.config_json, "abort_reason": "ValueError: boom"},
    )
    await store.update_run(run_id, failed_rec)
    failed = await store.get_run(run_id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.abort_reason == "ValueError: boom"


@pytest.mark.asyncio
async def test_failed_cell_persists_abort_reason(store) -> None:
    """A cell that dies on a fatal error writes status='failed' and its reason
    must survive a reload (stashed in config_json by the failed-cell path)."""

    def _boom(update: object) -> None:
        raise RuntimeError("boom")

    qs = (_q("q1"), _q("q2"))
    ds = _dataset(qs)
    with pytest.raises(RuntimeError, match="boom"):
        await run_benchmark(
            dataset=ds,
            questions=qs,
            systems=[FakeSUT()],
            store=store,
            judge=FakeJudge(),
            judge_model="judge-b",
            run_group="grp-fail",
            progress=_boom,
        )
    runs = await store.list_runs()
    assert len(runs) == 1
    assert runs[0].status == "failed"
    assert runs[0].abort_reason == "RuntimeError: boom"


@pytest.mark.asyncio
async def test_all_errored_cell_not_stamped_complete(store) -> None:
    """AUDIT_0824 M12: a cell where every question failed on a missing LLM
    gateway must land as status='failed', never 'complete' — a complete row of
    empty answers would pass later comparisons as clean."""

    class NoGatewaySUT:
        name = "nogw"
        answer_model = "model-a"
        profile_name = "profile-x"
        profile_hash = "phash"

        async def run_one(self, q: BenchQuestion) -> QuestionOutput:
            return QuestionOutput(
                answer_text="",
                retrieved_paths=(),
                abstained=True,
                error="no LLM gateway configured",
                trace={},
            )

    qs = (_q("q1"), _q("q2"))
    records = await run_benchmark(
        dataset=_dataset(qs),
        questions=qs,
        systems=[NoGatewaySUT()],
        store=store,
        judge=FakeJudge(),
        judge_model="judge-b",
    )
    rec = records[0]
    assert rec.status == "failed"
    assert rec.abort_reason is not None
    assert "all 2 questions errored" in rec.abort_reason
    assert "no LLM gateway configured" in rec.abort_reason
    # The failure (and its reason) survives a reload from the DB.
    got = await store.get_run(rec.id)
    assert got is not None
    assert got.status == "failed"
    assert got.abort_reason is not None and "all 2 questions errored" in got.abort_reason


@pytest.mark.asyncio
async def test_partial_error_cell_still_completes(store) -> None:
    """AUDIT_0824 M12: the all-errored guard must not overreach — one failing
    question among healthy ones stays a normal completed run."""
    qs = (_q("q1"), _q("q2"))
    records = await run_benchmark(
        dataset=_dataset(qs),
        questions=qs,
        systems=[FailSUT("q1")],
        store=store,
        judge=FakeJudge(),
        judge_model="judge-b",
    )
    assert records[0].status == "complete"


@pytest.mark.asyncio
async def test_first_failed_cell_cancels_sibling_cells(store) -> None:
    """AUDIT_0824 M3: one failed cell must not abandon the matrix to orphaned
    background tasks. Sibling cells are cancelled, and after the caller marks
    them aborted they must never flip back to complete."""
    qs = (_q("q1"), _q("q2"), _q("q3"))
    ds = _dataset(qs)
    started = asyncio.Event()
    slow_calls: list[str] = []

    def _boom_cell_progress(update: object) -> None:
        # A fatal cell-level error must escape _run_cell's per-question
        # catch, so fail the boom cell through its progress hook.
        if getattr(update, "system", "") == "boom":
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await run_benchmark(
            dataset=ds,
            questions=qs,
            systems=[GateBoomSUT(started), SlowSUT(started, slow_calls)],
            store=store,
            judge=FakeJudge(),
            judge_model="judge-b",
            run_group="grp-cascade",
            max_concurrent=2,
            progress=_boom_cell_progress,
        )

    # Mirror api/bench._run_to_completion: on the propagated failure every
    # still-running row is marked aborted.
    runs = {r.system: r for r in await store.list_runs()}
    assert set(runs) == {"boom", "slow"}
    assert runs["boom"].status == "failed"
    assert runs["boom"].abort_reason == "RuntimeError: boom"
    assert await store.mark_aborted(runs["slow"].id, "RuntimeError: boom") is True

    # Give any (hypothetical) orphaned sibling ample time to finish its
    # questions and rewrite its row; it must stay aborted.
    await asyncio.sleep(0.25)
    slow = await store.get_run(runs["slow"].id)
    assert slow is not None
    assert slow.status == "aborted"
    assert slow.abort_reason == "RuntimeError: boom"
    # The sibling was cancelled mid-flight, not left running to completion:
    # it answered exactly q1 before the failure landed and never progressed.
    assert slow_calls == ["q1"]


# ── Judge cache hit/miss ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_judge_cache_hit_miss(store) -> None:
    q = _q("q1")
    rubric = render_rubric(question=q, model_answer="42", abstained=False)
    key = judge_cache_key(rubric, q.id, "42", "judge-b")

    assert await store.judge_cache_get(key) is None  # miss

    from vesta.eval.bench_scoring import JudgeOutcome

    await store.judge_cache_put(key, JudgeOutcome(verdict=Verdict.CORRECT, reason="r"))
    cached = await store.judge_cache_get(key)
    assert cached is not None
    assert cached.verdict == Verdict.CORRECT
    assert cached.reason == "r"

    # GT edit → rendered rubric changes → different key → miss.
    q2 = BenchQuestion(
        id=q.id,
        question=q.question,
        capability=q.capability,
        difficulty=q.difficulty,
        slice="core",
        expected_behavior="answer",
        answer="CHANGED_GT",
        sources=q.sources,
    )
    key2 = judge_cache_key(
        render_rubric(question=q2, model_answer="42", abstained=False), q2.id, "42", "judge-b"
    )
    assert key2 != key
    assert await store.judge_cache_get(key2) is None


# ── Compare query ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compare_runs(store) -> None:
    def _insert(run_id: int, verdicts: dict[str, str]) -> None:
        pass

    rec_a = _make_run_record()
    run_a = await store.insert_run(rec_a)
    rec_b = BenchRunRecord(
        run_group="group-1",
        label="b",
        started_at="2026-01-01T00:00:00+00:00",
        status="running",
        dataset_name="vesta_test",
        dataset_hash="abc",
        subset_hash="",
        system="fake",
        profile_name="profile-x",
        profile_hash="phash",
        answer_model="model-a",
        judge_model="judge-b",
    )
    run_b = await store.insert_run(rec_b)

    def _row(run: int, qid: str, verdict: str, shr: int | None = None) -> BenchQuestionResult:
        return BenchQuestionResult(
            run_id=run,
            question_id=qid,
            capability="lookup",
            difficulty="easy",
            question_text=qid,
            expected_answer="A",
            answer_text="A",
            abstained=False,
            verdict=verdict,
            source_hit_rank=shr,
            source_coverage=0.5,
        )

    # q1: a correct → b wrong (broken regression)
    # q2: a wrong → b correct (fixed improvement)
    # q3: both correct
    # q4: both wrong
    # q5: only in b
    for qid, v in [
        ("q1", Verdict.CORRECT.value),
        ("q2", Verdict.INCORRECT.value),
        ("q3", Verdict.CORRECT.value),
        ("q4", Verdict.INCORRECT.value),
    ]:
        await store.insert_question_result(run_a, _row(run_a, qid, v, shr=1))
    for qid, v in [
        ("q1", Verdict.INCORRECT.value),
        ("q2", Verdict.CORRECT.value),
        ("q3", Verdict.CORRECT.value),
        ("q4", Verdict.INCORRECT.value),
        ("q5", Verdict.CORRECT.value),
    ]:
        await store.insert_question_result(run_b, _row(run_b, qid, v, shr=2))

    cmp = await compare_runs(store, run_a, run_b)
    assert isinstance(cmp, CompareResult)
    assert cmp.broken == ("q1",)
    assert cmp.fixed == ("q2",)
    assert cmp.both_correct == ("q3",)
    assert cmp.both_wrong == ("q4",)
    assert cmp.only_b == ("q5",)
    assert cmp.only_a == ()
    assert cmp.shared_denominator == 4


@pytest.mark.asyncio
async def test_compare_runs_refuses_dataset_mismatch(store) -> None:
    """Different dataset_hash → refused, never a silent misleading diff."""
    rec = _make_run_record()
    run_a = await store.insert_run(rec)
    run_b = await store.insert_run(replace(rec, label="b", dataset_hash="ffffff"))
    with pytest.raises(IncomparableRuns, match="dataset mismatch"):
        await compare_runs(store, run_a, run_b)


@pytest.mark.asyncio
async def test_compare_runs_refuses_subset_mismatch(store) -> None:
    """A filtered run (subset_hash set) is not comparable to a full run."""
    rec = _make_run_record()
    run_a = await store.insert_run(rec)
    run_b = await store.insert_run(replace(rec, label="b", subset_hash="deadbeef"))
    with pytest.raises(IncomparableRuns, match="subset mismatch"):
        await compare_runs(store, run_a, run_b)


@pytest.mark.asyncio
async def test_compare_runs_allows_profile_mismatch(store) -> None:
    """A/B across profiles on the same dataset stays allowed (the point of bench)."""
    rec = _make_run_record()
    run_a = await store.insert_run(rec)
    run_b = await store.insert_run(
        replace(rec, label="b", profile_name="hybrid", profile_hash="other")
    )
    cmp = await compare_runs(store, run_a, run_b)
    assert cmp.shared_denominator == 0


@pytest.mark.asyncio
async def test_compare_runs_refuses_degradation_mismatch(store) -> None:
    """AUDIT_0824 N36: a run whose traces recorded capability drops is flagged
    degraded in metrics_json; comparing it against a clean run must be refused,
    the same posture as eval's degraded-vs-clean guard."""
    rec = _make_run_record()
    clean_id = await store.insert_run(rec)
    degraded_id = await store.insert_run(
        replace(
            rec,
            label="degraded",
            metrics_json={
                "degraded": True,
                "degraded_components": ["candidate_source/vector_knn"],
            },
        )
    )
    with pytest.raises(IncomparableRuns, match="degradation mismatch"):
        await compare_runs(store, clean_id, degraded_id)


@pytest.mark.asyncio
async def test_compare_runs_allows_two_degraded_runs(store) -> None:
    """Both arms equally degraded → flags agree → comparison proceeds."""
    rec = replace(
        _make_run_record(),
        metrics_json={"degraded": True, "degraded_components": ["static_scorer"]},
    )
    run_a = await store.insert_run(rec)
    run_b = await store.insert_run(replace(rec, label="b"))
    cmp = await compare_runs(store, run_a, run_b)
    assert cmp.shared_denominator == 0


# ── Two-stage + rejudge ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_two_stage_and_rejudge(store) -> None:
    qs = (_q("q1"), _q("q2"), _q("q3"))
    ds = _dataset(qs)
    sut_calls: list[str] = []
    sut = FakeSUT(call_count=sut_calls)
    judge = FakeJudge()

    records = await run_benchmark(
        dataset=ds,
        questions=qs,
        systems=[sut],
        store=store,
        judge=judge,
        judge_model="judge-b",
        run_group="grp",
    )
    assert len(records) == 1
    rec = records[0]
    assert rec.status == "complete"
    assert len(sut_calls) == 3  # pipeline ran 3 questions once

    rows = await store.list_question_results(rec.id)
    assert len(rows) == 3
    # All judged (none pending).
    for r in rows:
        assert r.verdict != Verdict.PENDING.value
        assert r.verdict in (Verdict.CORRECT.value, Verdict.INCORRECT.value)
        assert r.source_hit_rank == 1  # retrieved ("A","B","C"), gold "A"
    pending = await store.list_pending_results(rec.id)
    assert pending == []

    # Rejudge a fresh (pending) run without pipeline work.
    # Simulate an aborted run: only stage-1 rows persisted as pending.
    run_b = await store.insert_run(_make_run_record())
    for q in qs:
        await store.insert_question_result(
            run_b,
            BenchQuestionResult(
                run_id=run_b,
                question_id=q.id,
                capability=q.capability,
                difficulty=q.difficulty,
                question_text=q.question,
                expected_answer=q.answer,
                answer_text="generic-answer",
                abstained=False,
                verdict=Verdict.PENDING.value,
                retrieved_paths=("A",),
                source_hit_rank=1,
            ),
        )
    calls_before = len(sut_calls)
    qmap = {q.id: q for q in qs}
    graded = await rejudge_run(store, judge, "judge-b", run_b, questions=qmap, judge_concurrency=4)
    assert graded == 3
    assert len(sut_calls) == calls_before  # no pipeline work
    pending_b = await store.list_pending_results(run_b)
    assert pending_b == []


@pytest.mark.asyncio
async def test_rejudge_preserves_rounds_and_latency(store) -> None:
    """Rejudge must not zero the stored rounds/latency columns or the run's
    recomputed ``source.latency`` metrics."""
    qs = (_q("q1"), _q("q2"))
    judge = FakeJudge()
    run_id = await store.insert_run(_make_run_record())
    for i, q in enumerate(qs):
        await store.insert_question_result(
            run_id,
            BenchQuestionResult(
                run_id=run_id,
                question_id=q.id,
                capability=q.capability,
                difficulty=q.difficulty,
                question_text=q.question,
                expected_answer=q.answer,
                answer_text="generic-answer",
                abstained=False,
                verdict=Verdict.PENDING.value,
                retrieved_paths=("A",),
                source_hit_rank=1,
                rounds=3,
                latency_ms=1234.5 + i,
            ),
        )

    graded = await rejudge_run(
        store,
        judge,
        "judge-b",
        run_id,
        questions={q.id: q for q in qs},
        judge_concurrency=2,
    )
    assert graded == 2

    rows = {r.question_id: r for r in await store.list_question_results(run_id)}
    assert rows["q1"].verdict == Verdict.CORRECT.value
    assert rows["q1"].rounds == 3
    assert rows["q1"].latency_ms == 1234.5
    assert rows["q2"].rounds == 3
    assert rows["q2"].latency_ms == 1235.5

    rec = await store.get_run(run_id)
    assert rec is not None
    lat = rec.metrics_json["source"]["latency"]
    assert isinstance(lat, dict)
    assert lat["n"] == 2
    assert lat["p50"] > 0


def test_rebuild_scored_matches_live_abstention_source() -> None:
    """AUDIT_0824 M17: given identical rows + verdicts, the rejudge rebuild and
    the live scoring path must agree on abstention metrics — both read the
    HARNESS ``abstained`` flag, never the judge's echoed field."""
    q = _q("a", expected_behavior="answer")  # in-corpus; harness refused
    row = BenchQuestionResult(
        run_id=1,
        question_id=q.id,
        capability=q.capability,
        difficulty=q.difficulty,
        question_text=q.question,
        expected_answer=q.answer,
        answer_text="",
        abstained=True,
        verdict=Verdict.CORRECT.value,
        verdict_reason="r",
    )
    rebuilt = _rebuild_scored([row], {q.id: q}, "jm")
    # The live path scores the same shape from outputs[q.id].abstained.
    live = score_question(
        q,
        row.retrieved_paths,
        JudgeOutcome(verdict=Verdict.CORRECT, reason="r"),
        abstained=row.abstained,
    )
    assert rebuilt[0].abstained is True
    assert aggregate_answer_metrics(rebuilt) == aggregate_answer_metrics([live])
    assert aggregate_answer_metrics(rebuilt).over_refusal == 1.0


# ── Concurrency invariance ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrency_invariance(store) -> None:
    """Same stored answers judged at judge_concurrency 1 and 8 → identical verdicts."""
    qs = tuple(_q(f"q{i}") for i in range(10))
    ds = _dataset(qs)
    sut = FakeSUT(answers={q.id: str(i) for i, q in enumerate(qs)})
    judge = FakeJudge()

    rec1 = await run_benchmark(
        dataset=ds,
        questions=qs,
        systems=[sut],
        store=store,
        judge=judge,
        judge_model="judge-b",
        run_group="g1",
        judge_concurrency=1,
    )
    rec8 = await run_benchmark(
        dataset=ds,
        questions=qs,
        systems=[FakeSUT(answers={q.id: str(i) for i, q in enumerate(qs)})],
        store=store,
        judge=judge,
        judge_model="judge-b",
        run_group="g8",
        judge_concurrency=8,
    )
    rows1 = await store.list_question_results(rec1[0].id)
    rows8 = await store.list_question_results(rec8[0].id)
    v1 = {r.question_id: r.verdict for r in rows1}
    v8 = {r.question_id: r.verdict for r in rows8}
    assert v1 == v8

    # Timing blocks legitimately vary run-to-run (latency metrics); every
    # verdict-derived aggregate must still be byte-identical.
    def _stable(m: dict[str, object]) -> dict[str, object]:
        src = dict(m["source"])  # type: ignore[arg-type]
        src.pop("latency", None)
        src.pop("latency_by_stage", None)
        return {**m, "source": src}  # type: ignore[dict-item]

    assert _stable(rec1[0].metrics_json) == _stable(rec8[0].metrics_json)


# ── End-to-end fake SUT + judge ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_benchmark_full(store) -> None:
    """Fake SUT + fake judge over a tiny dataset: metrics assemble correctly."""
    qs_list = [
        _q("q1", capability="lookup", answer="A1"),
        _q("q2", capability="multi_hop_cross_article", answer="A2", expected_behavior="answer"),
    ]
    # Give q2 an oracle verdict so reference points compute.
    from dataclasses import replace as _replace

    q2 = _replace(
        qs_list[1], oracle={"model": "model-a", "verdict": "correct", "checked_at": "now"}
    )
    qs = (qs_list[0], q2)
    ds = _dataset(qs)
    sut = FakeSUT(answers={"q1": "yes", "q2": "yes"})
    judge = FakeJudge()

    progress: list[str] = []
    records = await run_benchmark(
        dataset=ds,
        questions=qs,
        systems=[sut],
        store=store,
        judge=judge,
        judge_model="judge-b",
        run_group="grp-full",
        progress=lambda p: progress.append(p.stage),
    )
    assert len(records) == 1
    rec = records[0]
    m = rec.metrics_json
    # Both answered "yes" → judged correct by FakeJudge.
    assert m["answer"]["strict_accuracy"] == 1.0
    assert m["answer"]["weighted_accuracy"] == 1.0
    assert m["source"]["recall_at_10"] == 1.0
    assert m["source"]["source_coverage"] == 1.0
    # Reference counts only the oracle-bearing subset (q2; q1 has no oracle):
    # ceiling 1/1, system 1/1, floor 0. total is the full run size.
    assert m["reference"]["system"] == 1
    assert m["reference"]["total"] == 2
    assert m["reference"]["reference_n"] == 1
    assert m["by_capability"]["lookup"]["strict_accuracy"] == 1.0
    assert "pipeline" in progress and "judging" in progress and "complete" in progress


class DegradedSUT(FakeSUT):
    """A SUT whose pipeline traces record a capability drop (VECTORS unmet)."""

    async def run_one(self, q: BenchQuestion) -> QuestionOutput:
        out = await super().run_one(q)
        return replace(
            out,
            trace={
                "stages": [{"name": "answer", "component": "sources_only"}],
                "degradations": [
                    {
                        "component": "candidate_source/vector_knn",
                        "missing": "VECTORS",
                        "reason": "capability VECTORS not available",
                    }
                ],
            },
        )


def test_compute_metrics_derives_degraded_from_traces() -> None:
    """AUDIT_0824 N36: metrics_json carries the degradation signal reduced
    from cell traces, exactly like the eval twin's _reduce_metrics."""
    deg_trace = {
        "stages": [{"name": "answer", "component": "sources_only"}],
        "degradations": [{"component": "candidate_source/vector_knn", "missing": "VECTORS"}],
    }
    m = _compute_metrics([], answer_model="m", traces=[deg_trace])
    assert m["degraded"] is True
    assert m["degraded_components"] == ["candidate_source/vector_knn"]

    clean = _compute_metrics([], answer_model="m", traces=[{"stages": []}])
    assert clean["degraded"] is False
    assert clean["degraded_components"] == []


@pytest.mark.asyncio
async def test_capability_drop_flags_run_and_blocks_compare(store) -> None:
    """AUDIT_0824 N36 regression: a bench cell whose profile silently dropped
    VECTORS persists metrics_json.degraded=True, and compare_runs then refuses
    it against a clean run instead of producing a misleading diff."""
    qs = (_q("q1"), _q("q2"))
    ds = _dataset(qs)
    degraded = await run_benchmark(
        dataset=ds,
        questions=qs,
        systems=[DegradedSUT()],
        store=store,
        judge=FakeJudge(),
        judge_model="judge-b",
        run_group="grp-degraded",
    )
    m = degraded[0].metrics_json
    assert m["degraded"] is True
    assert m["degraded_components"] == ["candidate_source/vector_knn"]

    clean = await run_benchmark(
        dataset=ds,
        questions=qs,
        systems=[FakeSUT()],
        store=store,
        judge=FakeJudge(),
        judge_model="judge-b",
        run_group="grp-clean",
    )
    assert clean[0].metrics_json["degraded"] is False

    with pytest.raises(IncomparableRuns, match="degradation mismatch"):
        await compare_runs(store, clean[0].id, degraded[0].id)


class TokenSUT:
    """A SUT that reports per-question token usage."""

    name = "token_fake"
    answer_model = "model-a"
    profile_name = "profile-x"
    profile_hash = "phash"

    async def run_one(self, q: BenchQuestion) -> QuestionOutput:
        return QuestionOutput(
            answer_text="yes",
            retrieved_paths=("A",),
            abstained=False,
            error=None,
            trace={"stages": [{"name": "answer", "component": "sources_only"}]},
            resolved_strategy="token_fake",
            rounds=1,
            input_tokens=100,
            output_tokens=25,
        )


@pytest.mark.asyncio
async def test_token_tracking_through_run(store) -> None:
    """Token counts flow from SUT → question rows → metrics."""
    qs = (_q("q1"), _q("q2"))
    ds = _dataset(qs)
    sut = TokenSUT()
    judge = FakeJudge()
    records = await run_benchmark(
        dataset=ds,
        questions=qs,
        systems=[sut],
        store=store,
        judge=judge,
        judge_model="judge-b",
        run_group="grp-tokens",
    )
    rec = records[0]
    # Per-question rows carry the token counts.
    rows = await store.list_question_results(rec.id)
    for row in rows:
        assert row.input_tokens == 100
        assert row.output_tokens == 25
    # Metrics include the token summary.
    tok = rec.metrics_json.get("tokens", {}).get("answer", {})
    assert tok["total_input"] == 200  # 2 questions x 100
    assert tok["total_output"] == 50  # 2 questions x 25
    assert tok["total"] == 250
    assert tok["p50"] == 125  # each question is 125 total → median 125


# ── Boundary (import) ───────────────────────────────────────────────────────


def test_bench_runner_imports_only_retrieval_config() -> None:
    """eval/bench_runner.py may import only retrieval + config (+ eval siblings)."""
    import ast

    src = Path(__file__).resolve().parent.parent / "src" / "vesta" / "eval" / "bench_runner.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                seen.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            seen.add(mod.split(".")[0])
    internal = {
        s
        for s in seen
        if s
        in {
            "db",
            "zim",
            "inference",
            "answer",
            "api",
            "vectors",
            "encoders",
            "index",
            "catalog",
            "jobs",
        }
    }
    assert internal == set(), f"bench_runner imports forbidden packages: {internal}"


# ── Judge concurrency clamp ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retrieval_only_zero_llm_calls() -> None:
    """retrieval_only runs the pipeline only — the LLM gateway is never touched."""
    from types import SimpleNamespace

    from vesta.api.bench import RetrievalOnlySystem

    class BoomGateway:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(
                f"gateway.{name} called during retrieval_only — zero LLM calls expected"
            )

    q = _q("q1")
    state = SimpleNamespace(registry=None, encoders=None, gateway=BoomGateway(), db=None)
    system = RetrievalOnlySystem(state)
    out = await system.run_one(q)
    # With no archives the pipeline raises NoCandidatesError → abstained, no LLM.
    assert out.abstained is True
    assert out.answer_text == ""


@pytest.mark.asyncio
async def test_judge_concurrency_clamp() -> None:
    from vesta.eval.bench_runner import resolve_judge_concurrency

    c, shares = resolve_judge_concurrency(
        4, answer_endpoint="http://x:1234/v1", judge_endpoint="http://x:1234/v1"
    )
    assert c == 1 and shares
    c2, shares2 = resolve_judge_concurrency(
        4, answer_endpoint="http://x:1234/v1", judge_endpoint="http://y:1234/v1"
    )
    assert c2 == 4 and not shares2


# ── Startup trace pruning (main.py lifespan → prune_stale_bench_traces) ─────


def _trace_qrow(run_id: int, trace: dict[str, object] | None) -> BenchQuestionResult:
    return BenchQuestionResult(
        run_id=run_id,
        question_id="q1",
        capability="lookup",
        difficulty="easy",
        question_text="Q",
        expected_answer="A",
        answer_text="A",
        abstained=False,
        verdict=Verdict.CORRECT.value,
        trace=trace,
    )


@pytest.mark.asyncio
async def test_old_run_trace_is_pruned(db, store) -> None:
    from vesta.api.bench import prune_stale_bench_traces

    run_id = await store.insert_run(_make_run_record())
    await store.insert_question_result(run_id, _trace_qrow(run_id, {"version": 1}))

    pruned = await prune_stale_bench_traces(db, 30)
    assert pruned == 1
    rows = await store.list_question_results(run_id)
    assert rows[0].trace is None
    # Verdict + answer text survive the prune — only the raw evidence goes.
    assert rows[0].answer_text == "A"
    assert rows[0].verdict == Verdict.CORRECT.value


@pytest.mark.asyncio
async def test_recent_run_trace_is_kept(db, store) -> None:
    import datetime as _dt
    from dataclasses import replace

    from vesta.api.bench import prune_stale_bench_traces

    rec = replace(_make_run_record(), started_at=_dt.datetime.now(_dt.UTC).isoformat())
    run_id = await store.insert_run(rec)
    await store.insert_question_result(run_id, _trace_qrow(run_id, {"version": 1}))

    assert await prune_stale_bench_traces(db, 30) == 0
    rows = await store.list_question_results(run_id)
    assert rows[0].trace == {"version": 1}


@pytest.mark.asyncio
async def test_retention_zero_is_noop(db, store) -> None:
    from vesta.api.bench import prune_stale_bench_traces

    run_id = await store.insert_run(_make_run_record())
    await store.insert_question_result(run_id, _trace_qrow(run_id, {"version": 1}))

    assert await prune_stale_bench_traces(db, 0) == 0
    rows = await store.list_question_results(run_id)
    assert rows[0].trace == {"version": 1}
