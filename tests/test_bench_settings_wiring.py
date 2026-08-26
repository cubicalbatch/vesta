"""AUDIT_0824 B3 regression: the UI-exposed ``bench.*`` knobs are honored.

Each of these settings describes real behavior but was historically bypassed —
consumers read the descriptor's ``.default`` instead of the resolved value, so
a DB-set (or env-set) value silently did nothing. These tests prove a DB-set
value now changes observable behavior:

- ``bench.max_concurrent`` / ``bench.repeats`` / ``bench.judge.concurrency``
  reach ``run_benchmark`` when the caller does not pass them explicitly
  (pinned into the run's ``config_json``, so the proof is persisted).
- ``bench.judge.cache`` gates the judge cache on the rejudge path.
- ``bench.judge.retries`` controls parse-failure retry attempts.
- ``bench.calibration_min_correlation`` decides ``trusted`` for a given rho.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vesta import config
from vesta.db.connection import Database
from vesta.db.migrations import run_migrations
from vesta.eval.bench_dataset import BenchDataset, BenchQuestion, dataset_hash
from vesta.eval.bench_runner import (
    BenchQuestionResult,
    BenchRunRecord,
    QuestionOutput,
    rejudge_run,
    run_benchmark,
)
from vesta.eval.bench_scoring import (
    JudgeOutcome,
    Verdict,
    judge_cache_key,
    render_rubric,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


class FakeSUT:
    name = "fake"
    answer_model = "model-a"
    profile_name = "profile-x"
    profile_hash = "phash"

    async def run_one(self, q: BenchQuestion) -> QuestionOutput:
        return QuestionOutput(
            answer_text="yes",
            retrieved_paths=("A",),
            abstained=False,
            error=None,
            trace={},
        )


class FakeJudge:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def judge(self, prompt: str) -> str:
        self.calls.append(prompt)
        return json.dumps({"verdict": "correct", "reason": "r", "abstained": False})


class GarbageJudge(FakeJudge):
    async def judge(self, prompt: str) -> str:
        self.calls.append(prompt)
        return "not json at all"


def _q(qid: str) -> BenchQuestion:
    return BenchQuestion(
        id=qid,
        question=f"Question {qid}?",
        capability="lookup",
        difficulty="easy",
        slice="core",
        expected_behavior="answer",
        answer="42",
    )


def _dataset(qs: tuple[BenchQuestion, ...]) -> BenchDataset:
    return BenchDataset(name="vesta_test", version=1, questions=qs, hash=dataset_hash(qs))


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


async def _seed_pending_run(store: object, qs: tuple[BenchQuestion, ...], answer: str) -> int:
    """A stored run whose rows are all ``pending`` (the rejudge seam's input)."""
    run_id = await store.insert_run(_make_run_record())  # type: ignore[attr-defined]
    for q in qs:
        await store.insert_question_result(  # type: ignore[attr-defined]
            run_id,
            BenchQuestionResult(
                run_id=run_id,
                question_id=q.id,
                capability=q.capability,
                difficulty=q.difficulty,
                question_text=q.question,
                expected_answer=q.answer,
                answer_text=answer,
                abstained=False,
                verdict=Verdict.PENDING.value,
                retrieved_paths=("A",),
                source_hit_rank=1,
            ),
        )
    return run_id


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


@pytest.fixture
def resolver():
    config.configure(env={})
    yield config
    config.reset_for_test()


# ── The knobs ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_max_concurrent_from_settings(store, resolver) -> None:
    """DB-set bench.max_concurrent reaches run_benchmark without an explicit arg."""
    config.set_db_values({"bench.max_concurrent": "3"})
    recs = await run_benchmark(
        dataset=_dataset((_q("q1"),)),
        questions=(_q("q1"), _q("q2")),
        systems=[FakeSUT()],
        store=store,
        judge=FakeJudge(),
        judge_model="judge-b",
        run_group="grp-mc",
    )
    assert [r.config_json["max_concurrent"] for r in recs] == [3]


@pytest.mark.asyncio
async def test_repeats_from_settings(store, resolver) -> None:
    """DB-set bench.repeats expands the matrix without an explicit arg."""
    config.set_db_values({"bench.repeats": "2"})
    qs = (_q("q1"),)
    recs = await run_benchmark(
        dataset=_dataset(qs),
        questions=qs,
        systems=[FakeSUT()],
        store=store,
        judge=FakeJudge(),
        judge_model="judge-b",
        run_group="grp-rep",
    )
    assert len(recs) == 2
    assert sorted(r.config_json["repeat_index"] for r in recs) == [0, 1]


@pytest.mark.asyncio
async def test_judge_concurrency_from_settings(store, resolver) -> None:
    """DB-set bench.judge.concurrency reaches run_benchmark without an explicit arg."""
    config.set_db_values({"bench.judge.concurrency": "2"})
    recs = await run_benchmark(
        dataset=_dataset((_q("q1"),)),
        questions=(_q("q1"),),
        systems=[FakeSUT()],
        store=store,
        judge=FakeJudge(),
        judge_model="judge-b",
        run_group="grp-jc",
    )
    assert [r.config_json["judge_concurrency"] for r in recs] == [2]


@pytest.mark.asyncio
async def test_judge_cache_used_by_default(store) -> None:
    """Control: with the default (cache on), a cached verdict short-circuits the judge."""
    config.configure(env={})
    try:
        qs = (_q("q1"),)
        q = qs[0]
        run_id = await _seed_pending_run(store, qs, "generic-answer")
        prompt = render_rubric(question=q, model_answer="generic-answer", abstained=False)
        key = judge_cache_key(prompt, q.id, "generic-answer", "judge-b")
        await store.judge_cache_put(
            key, JudgeOutcome(verdict=Verdict.PARTIAL, reason="cached", judge_model="judge-b")
        )
        judge = FakeJudge()
        graded = await rejudge_run(store, judge, "judge-b", run_id, questions={q.id: q})
        assert graded == 1
        assert judge.calls == []  # cache hit — judge never called
        rows = await store.list_question_results(run_id)
        assert [r.verdict for r in rows] == [Verdict.PARTIAL.value]
    finally:
        config.reset_for_test()


@pytest.mark.asyncio
async def test_judge_cache_disabled_via_settings(store, resolver) -> None:
    """DB-set bench.judge.cache=false bypasses the cache: the judge recomputes."""
    config.set_db_values({"bench.judge.cache": "false"})
    qs = (_q("q1"),)
    q = qs[0]
    run_id = await _seed_pending_run(store, qs, "generic-answer")
    prompt = render_rubric(question=q, model_answer="generic-answer", abstained=False)
    key = judge_cache_key(prompt, q.id, "generic-answer", "judge-b")
    await store.judge_cache_put(
        key, JudgeOutcome(verdict=Verdict.PARTIAL, reason="cached", judge_model="judge-b")
    )
    judge = FakeJudge()
    graded = await rejudge_run(store, judge, "judge-b", run_id, questions={q.id: q})
    assert graded == 1
    assert len(judge.calls) == 1  # cache ignored — judge recomputed
    rows = await store.list_question_results(run_id)
    assert [r.verdict for r in rows] == [Verdict.CORRECT.value]


@pytest.mark.asyncio
async def test_judge_retries_from_settings(resolver) -> None:
    """DB-set bench.judge.retries controls parse-failure attempts (1 + retries)."""
    from vesta.eval.bench_scoring import judge_verdict

    q = _q("q1")
    config.set_db_values({"bench.judge.retries": "0"})
    judge = GarbageJudge()
    out = await judge_verdict(
        question=q, model_answer="x", abstained=False, judge=judge, judge_model="m"
    )
    assert out.verdict == Verdict.UNJUDGED
    assert len(judge.calls) == 1

    config.set_db_values({"bench.judge.retries": "3"})
    judge = GarbageJudge()
    out = await judge_verdict(
        question=q, model_answer="x", abstained=False, judge=judge, judge_model="m"
    )
    assert out.verdict == Verdict.UNJUDGED
    assert len(judge.calls) == 4


@pytest.mark.asyncio
async def test_calibration_min_correlation_controls_trusted(
    store, resolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DB-set bench.calibration_min_correlation flips ``trusted`` for a fixed rho."""

    async def _fixed_rho(*_args: object, **_kwargs: object) -> float:
        return 0.8

    monkeypatch.setattr("vesta.eval.bench_runner.measure_bench_calibration", _fixed_rho)
    qs = (_q("q1"),)
    kw = {
        "dataset": _dataset(qs),
        "questions": qs,
        "systems": [FakeSUT()],
        "store": store,
        "judge": FakeJudge(),
        "judge_model": "judge-b",
    }

    config.set_db_values({"bench.calibration_min_correlation": "0.9"})
    (rec,) = await run_benchmark(run_group="grp-trust-hi", **kw)
    assert rec.trusted is False

    config.set_db_values({"bench.calibration_min_correlation": "0.7"})
    (rec,) = await run_benchmark(run_group="grp-trust-lo", **kw)
    assert rec.trusted is True
