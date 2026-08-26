"""Unified benchmark runner — SystemUnderTest, two-stage execution, matrix.

The orchestration layer that drives a registered :class:`SystemUnderTest` over
a :class:`~vesta.eval.bench_dataset.BenchDataset` and persists per-question
results. It is the single entry point that replaces earlier measurement tools.

**Two-stage execution**: the pipeline stage runs questions one at
a time, writing each ``bench_question_results`` row as ``verdict='pending'``
the moment the answer completes (latency stopped then). A batch judging stage
then grades pending rows with N workers keyed by ``question_id``. Killing the
process mid-flight leaves completed answers as ``pending`` rows —
:func:`rejudge_run` grades them later without re-running the pipeline.

**Matrix expansion**: one invocation produces one ``run_group`` (uuid) + one
``bench_runs`` row per cell (system x profile x model). Cells execute
sequentially by default (``max_concurrent=1`` — shared inference endpoints
contaminate reported latency at N-wide pipeline concurrency).

**Concurrency invariance** (trap: same answers judged at judge_concurrency 1
and 8 must produce byte-identical verdicts): judge calls are stateless at
temperature 0, each grades one ``(question, answer)`` pair against its own
ground truth, and results are keyed by ``question_id`` — never by completion
order — so the stored rows and every aggregate are identical at any concurrency.

Boundary: this module imports ONLY ``vesta.retrieval`` + ``vesta.config`` (+ ``vesta.eval``
siblings, which are the same package and so do not count toward the dependency
cap). No ``db``/``zim``/``inference``/``answer``/``api``. All I/O (DB, LLM
judge, archive access) is injected via the :class:`BenchStore` /
:class:`SystemUnderTest` / :class:`~vesta.eval.answer_metrics.JudgeLLM`
Protocols defined here (or in eval siblings). The composition root
(``api/bench.py``) wires the real implementations.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from vesta.config.settings import setting
from vesta.eval.answer_metrics import JudgeLLM
from vesta.eval.bench_dataset import BenchDataset, BenchQuestion, subset_hash
from vesta.eval.bench_scoring import (
    BENCH_CALIBRATION_MIN_CORRELATION,
    BENCH_JUDGE_CACHE,
    BENCH_JUDGE_RETRIES,
    JudgeOutcome,
    ScoredQuestion,
    Verdict,
    aggregate_answer_metrics,
    aggregate_latency,
    aggregate_peak_context,
    aggregate_source_metrics,
    aggregate_token_usage,
    attribution_by_capability,
    attribution_matrix,
    judge_verdict,
    measure_bench_calibration,
    reference_points,
    score_question,
    source_coverage,
    source_hit_rank,
    stage_latency_breakdown,
)
from vesta.eval.runner import git_sha, machine_id, now_iso

# ── Settings ────────────────────────────────────────────────────────────────

BENCH_SYSTEMS = setting(
    "bench.systems",
    str,
    "agentic_pydantic",
    group="Benchmark",
    help="Comma-separated default system names for ``vesta bench run --system``.",
    hot=False,
)
BENCH_MAX_CONCURRENT = setting(
    "bench.max_concurrent",
    int,
    1,
    group="Benchmark",
    help="Pipeline question concurrency. Defaults to 1 — raising it contaminates "
    "the reported p50/p95 latency against a shared inference endpoint.",
    min=1,
    max=32,
    hot=False,
)
BENCH_JUDGE_CONCURRENCY = setting(
    "bench.judge.concurrency",
    int,
    4,
    group="Benchmark / Judge",
    help="Judge calls in flight during the batch judging stage. Clamped to 1 when "
    "the judge shares the answer endpoint.",
    min=1,
    max=32,
    hot=False,
)
BENCH_REPEATS = setting(
    "bench.repeats",
    int,
    1,
    group="Benchmark",
    help="Run each matrix cell N times for variance (mean/stdev on headline metrics).",
    min=1,
    max=10,
    hot=False,
)
BENCH_TRACE_RETENTION_DAYS = setting(
    "bench.trace_retention_days",
    int,
    30,
    group="Benchmark",
    help="Days to retain per-question trace_json before pruning (traces are a "
    "separate prunable column; verdicts + retrieval + answer text stay forever).",
    min=0,
    max=3650,
    hot=False,
)


def resolve_matrix_axes(
    systems: Sequence[str] | None,
    profiles: Sequence[str] | None,
    models: Sequence[str] | None,
    *,
    default_systems: str,
    default_model: str,
) -> tuple[list[str], list[str], list[str]]:
    """Resolve the systems / profiles / models matrix axes from (possibly
    empty) explicit selections. One copy of the defaults ladder shared by the
    API route and the CLI flag path so bench numbers stay comparable:
    ``bench.systems`` comma-split with an ``agentic_pydantic`` floor, the
    active/default profile (empty string), and ``inference.llm.model``
    filtered for emptiness. Model-PRESENCE validation stays with the caller —
    the two entry points fail differently (HTTP 400 vs ``SystemExit``) and the
    CLI additionally exempts ``retrieval_only``.
    """
    systems = list(systems or [s.strip() for s in default_systems.split(",") if s.strip()])
    systems = systems or ["agentic_pydantic"]
    profiles = list(profiles or [""])  # empty → active/default profile
    models = [m for m in (list(models or [default_model])) if m]
    return [s for s in systems if s], profiles, models


# ── QuestionOutput (the seam) ───────────────────────────────────────────────


@dataclass(frozen=True)
class QuestionOutput:
    """One system's output for one question.

    ``answer_text`` is the final generated answer (empty for ``retrieval_only``).
    ``retrieved_paths`` are the canonical article paths from the source cards —
    the input to every source metric. ``abstained`` is the harness decision.
    ``trace`` is the versioned pipeline trace (prunable on the row).
    ``citation_supported`` is the fraction of the answer supported by citations
    (from ``answer/citations.py``, recorded — not scored).
    """

    answer_text: str
    retrieved_paths: tuple[str, ...]
    abstained: bool
    error: str | None
    trace: dict[str, object]
    resolved_strategy: str = ""
    rounds: int = 0
    tool_calls: int = 0
    citation_supported: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class SystemUnderTest(Protocol):
    """One question in, one :class:`QuestionOutput` out.

    Profile, model, scope, and strategy are bound at construction by the
    composition root (``api/bench.py``). Adding a harness later = registering
    one class. Implementations SHOULD also carry ``answer_model``,
    ``profile_name``, ``profile_hash`` attributes so the runner can pin them on
    the run record (read via :func:`getattr`, duck-typed).
    """

    name: str

    async def run_one(self, q: BenchQuestion) -> QuestionOutput: ...


# ── Persistence Protocols (DB-free; the composition root wires aiosqlite) ───


class BenchStore(Protocol):
    """Persistence backend for benchmark runs (wired to ``bench_runs`` etc.).

    Defines the DB seam in ``eval/`` so the runner is DB-free. The concrete
    ``SqliteBenchStore`` lives in ``api/bench.py`` (it imports aiosqlite).
    """

    async def insert_run(self, record: BenchRunRecord) -> int: ...
    async def update_run(self, run_id: int, record: BenchRunRecord) -> bool: ...
    async def get_run(self, run_id: int) -> BenchRunRecord | None: ...
    async def list_runs(self, limit: int = 50) -> list[BenchRunRecord]: ...
    async def delete_run(self, run_id: int) -> bool: ...

    async def mark_aborted(self, run_id: int, reason: str) -> bool: ...
    async def insert_question_result(self, run_id: int, row: BenchQuestionResult) -> None: ...
    async def update_question_result(
        self, run_id: int, question_id: str, row: BenchQuestionResult
    ) -> bool: ...
    async def update_verdict(
        self,
        run_id: int,
        question_id: str,
        verdict: str,
        reason: str,
        sub_fact_coverage: float | None,
    ) -> bool: ...
    async def list_question_results(self, run_id: int) -> list[BenchQuestionResult]: ...
    async def list_pending_results(self, run_id: int) -> list[BenchQuestionResult]: ...

    async def judge_cache_get(self, key: str) -> JudgeOutcome | None: ...
    async def judge_cache_put(self, key: str, outcome: JudgeOutcome) -> None: ...

    async def prune_traces(self, older_than_days: int) -> int: ...
    async def reconcile_stale(self) -> int: ...


# ── Run record (the persisted row) ─────────────────────────────────────────


@dataclass(frozen=True)
class BenchRunRecord:
    """One persisted bench run: identity + the pins + metrics.

    One row per system x profile x model in a group. ``run_group`` (uuid) is the
    comparison unit. ``status`` is ``running`` during execution, ``complete`` on
    success, ``aborted`` on failure/process-death. ``metrics_json`` holds all
    aggregates (source, answer, reference points, attribution, per-capability).
    """

    run_group: str
    label: str
    started_at: str
    status: str  # running | complete | aborted
    dataset_name: str
    dataset_hash: str
    subset_hash: str
    system: str
    profile_name: str
    profile_hash: str
    answer_model: str
    judge_model: str
    config_json: dict[str, object] = field(default_factory=dict)
    metrics_json: dict[str, object] = field(default_factory=dict)
    id: int = 0
    finished_at: str | None = None
    scope: str = ""
    trusted: bool = False
    calibration: float | None = None
    judge_shares_endpoint: bool = False
    abort_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "run_group": self.run_group,
            "label": self.label,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "dataset_name": self.dataset_name,
            "dataset_hash": self.dataset_hash,
            "subset_hash": self.subset_hash,
            "system": self.system,
            "profile_name": self.profile_name,
            "profile_hash": self.profile_hash,
            "answer_model": self.answer_model,
            "judge_model": self.judge_model,
            "scope": self.scope,
            "trusted": self.trusted,
            "calibration": self.calibration,
            "judge_shares_endpoint": self.judge_shares_endpoint,
            "abort_reason": self.abort_reason,
            "config_json": dict(self.config_json),
            "metrics_json": dict(self.metrics_json),
        }


@dataclass(frozen=True)
class BenchQuestionResult:
    """One per-question row: the pinned question + system output + verdict."""

    run_id: int
    question_id: str
    capability: str
    difficulty: str
    question_text: str
    expected_answer: str
    answer_text: str
    abstained: bool
    verdict: str
    retrieved_paths: tuple[str, ...] = ()
    verdict_reason: str = ""
    source_hit_rank: int | None = None
    source_coverage: float = 0.0
    sub_fact_coverage: float | None = None
    rounds: int = 0
    latency_ms: float = 0.0
    error: str | None = None
    trace: dict[str, object] | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "question_id": self.question_id,
            "capability": self.capability,
            "difficulty": self.difficulty,
            "question_text": self.question_text,
            "expected_answer": self.expected_answer,
            "answer_text": self.answer_text,
            "abstained": self.abstained,
            "verdict": self.verdict,
            "verdict_reason": self.verdict_reason,
            "source_hit_rank": self.source_hit_rank,
            "source_coverage": self.source_coverage,
            "sub_fact_coverage": self.sub_fact_coverage,
            "retrieved_paths": list(self.retrieved_paths),
            "rounds": self.rounds,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "trace_json": dict(self.trace) if self.trace else None,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


# ── Compare (per-question diff across two runs) ────────────────────────────


@dataclass(frozen=True)
class CompareResult:
    """Per-question diff across two runs + aggregate deltas.

    ``fixed`` = A wrong → B correct. ``broken`` = A correct → B wrong. These
    are the buckets that catch regressions a mean can hide (trap 9).
    ``shared_denominator`` is the count of questions present in BOTH runs —
    ``--limit 5`` runs must never compare against full runs without this marker.
    """

    run_a: int
    run_b: int
    shared_denominator: int
    fixed: tuple[str, ...]
    broken: tuple[str, ...]
    both_correct: tuple[str, ...]
    both_wrong: tuple[str, ...]
    only_a: tuple[str, ...]
    only_b: tuple[str, ...]
    deltas: dict[str, float] = field(default_factory=dict)


# ── Progress callback ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProgressUpdate:
    """One progress tick: ``(system, stage, done, total, run_id)``."""

    system: str
    stage: str  # "pipeline" | "judging" | "complete" | "aborted"
    done: int
    total: int
    run_id: int = 0


# ── Helpers ────────────────────────────────────────────────────────────────


def _system_pin(system: Any, attr: str, default: str = "") -> str:
    """Read a metadata attribute from a SystemUnderTest (duck-typed)."""
    val = getattr(system, attr, default)
    return str(val) if val is not None else default


def _sub_fact_coverage(outcome: JudgeOutcome, q: BenchQuestion) -> float | None:
    """Fraction of declared sub-facts reported present by the judge (None if no sub-facts)."""
    if not q.sub_facts or not outcome.verdict.is_judged:
        return None
    present = outcome.sub_facts_present
    if not present:
        return 0.0
    return sum(1 for x in present if x) / len(present)


def _compute_metrics(
    scored: Sequence[ScoredQuestion],
    *,
    answer_model: str,
    traces: Sequence[Mapping[str, object]] = (),
    with_answers: bool = True,
) -> dict[str, object]:
    """Build the full metrics_json from scored questions.

    ``traces`` adds the per-stage latency breakdown.
    ``with_answers=False`` marks the answer/attribution blocks as skipped —
    the retrieval-only fast path produces no answers to judge.
    """
    source = aggregate_source_metrics(scored)
    answer = aggregate_answer_metrics(scored)
    refs = reference_points(scored, answer_model=answer_model)
    attribution = attribution_matrix(scored)
    by_cap = attribution_by_capability(scored)
    ans_tokens = aggregate_token_usage(
        scored, input_attr="answer_input_tokens", output_attr="answer_output_tokens"
    )
    return {
        "source": {
            "n": source.n,
            "recall_at_1": source.recall_at_1,
            "recall_at_5": source.recall_at_5,
            "recall_at_10": source.recall_at_10,
            "recall_at_20": source.recall_at_20,
            "source_coverage": source.mean_coverage,
            "source_mrr": source.mrr,
            "retrieved_precision": source.mean_precision,
            "latency": aggregate_latency(scored),
            "latency_by_stage": stage_latency_breakdown(traces),
        },
        "answer": {
            "n": answer.n,
            "strict_accuracy": answer.strict_accuracy,
            "weighted_accuracy": answer.weighted_accuracy,
            "sub_fact_coverage": answer.sub_fact_coverage,
            "abstention_correctness": answer.abstention_correctness,
            "over_refusal": answer.over_refusal,
            "hallucination_rate": answer.hallucination_rate,
            "unjudged": answer.unjudged,
            "complete": answer.complete,
            **({} if with_answers else {"skipped": True}),
        },
        "reference": {
            "ceiling": refs.ceiling,
            "system": refs.system,
            "floor": refs.floor,
            "total": refs.total,
            # Questions the counts above are taken over (oracle-bearing
            # subset); may be smaller than ``total`` on partially-referenced
            # datasets.
            "reference_n": refs.reference_n,
            "headroom_realised": refs.headroom_realised,
            "retrieval_regressions": refs.retrieval_regressions,
            "suppressed_reason": refs.suppressed_reason,
        },
        "attribution": {
            "correct_source_found": attribution.correct_source_found,
            "correct_source_missed": attribution.correct_source_missed,
            "failed_source_found": attribution.failed_source_found,
            "failed_source_missed": attribution.failed_source_missed,
        },
        "tokens": {
            "answer": {
                "total_input": ans_tokens.total_input,
                "total_output": ans_tokens.total_output,
                "total": ans_tokens.total,
                "p50": ans_tokens.p50,
                "p50_input": ans_tokens.p50_input,
                "p50_output": ans_tokens.p50_output,
            },
            # Measured peak single-request input-token
            # distribution from the agent runner's per-request meter — the
            # quantity a context window constrains. n=0 for pre-meter runs.
            "peak_context": aggregate_peak_context(traces),
        },
        "by_capability": {
            cap: {
                "n": cb.n,
                "source_recall_at_10": cb.source.recall_at_10,
                "source_coverage": cb.source.mean_coverage,
                "strict_accuracy": cb.answer.strict_accuracy,
                "weighted_accuracy": cb.answer.weighted_accuracy,
                "attribution": {
                    "correct_source_found": cb.attribution.correct_source_found,
                    "correct_source_missed": cb.attribution.correct_source_missed,
                    "failed_source_found": cb.attribution.failed_source_found,
                    "failed_source_missed": cb.attribution.failed_source_missed,
                },
            }
            for cap, cb in by_cap.items()
        },
    }


# ── Stage 2: batch judging ─────────────────────────────────────────────────


async def _judge_one(
    *,
    q: BenchQuestion,
    output: QuestionOutput,
    judge: JudgeLLM | None,
    judge_model: str,
    cache_enabled: bool,
    store: BenchStore,
    retries: int,
    semaphore: asyncio.Semaphore,
) -> JudgeOutcome:
    """Grade one answer (cache-aware, concurrency-bounded)."""
    async with semaphore:
        cache_get = store.judge_cache_get if cache_enabled else None
        cache_put = store.judge_cache_put if cache_enabled else None
        return await judge_verdict(
            question=q,
            model_answer=output.answer_text,
            abstained=output.abstained,
            judge=judge,
            judge_model=judge_model,
            retries=retries,
            cache_get=cache_get,
            cache_put=cache_put,
        )


async def _run_cell(  # noqa: PLR0912, PLR0915
    *,
    system: SystemUnderTest,
    questions: Sequence[BenchQuestion],
    store: BenchStore,
    judge: JudgeLLM | None,
    judge_model: str,
    run_group: str,
    label: str,
    scope: str,
    dataset: BenchDataset,
    subset_hash_val: str,
    config_snapshot: Mapping[str, object] | None,
    economy: str | None,
    context_profile: str | None,
    settings_set: Mapping[str, str] | None,
    judge_concurrency: int,
    judge_shares_endpoint: bool,
    max_concurrent: int,
    repeat_index: int,
    progress: Callable[[ProgressUpdate], None] | None,
    calibration: float | None = None,
    trusted: bool = False,
    level: int | None = None,
) -> BenchRunRecord:
    """Run one system over all questions (two-stage) and persist."""
    answer_model = _system_pin(system, "answer_model")
    profile_name = _system_pin(system, "profile_name")
    profile_hash = _system_pin(system, "profile_hash")
    sys_name = system.name
    full_label = f"{label}{' r' + str(repeat_index) if repeat_index else ''}".strip()
    started = now_iso()
    config: dict[str, object] = {
        "git_sha": git_sha(),
        "machine_id": machine_id(),
        "settings_snapshot": dict(config_snapshot) if config_snapshot else {},
        "judge_concurrency": judge_concurrency,
        "judge_shares_endpoint": judge_shares_endpoint,
        "max_concurrent": max_concurrent,
        "repeat_index": repeat_index,
        "system": sys_name,
    }
    if economy is not None:
        # `--economy` was forced on the CLI — record the forced value at the
        # top level so runs are comparable (the snapshot also carries it).
        config["economy"] = economy
    if context_profile is not None:
        # `--context-profile` was forced on the CLI — record
        # the forced window plan so profile runs are comparable (the snapshot
        # also carries it).
        config["context_profile"] = context_profile
    if settings_set:
        # `--set KEY=VALUE` (repeatable) forced these registered settings keys
        # for this run — record the applied pairs so
        # runs are comparable per override; the resolver also carries the
        # resolved values inside `settings_snapshot`, but this is the knob the
        # operator turned.
        config["settings_set"] = dict(settings_set)
    if level is not None:
        # The selected tier. `subset_hash` already pins the exact
        # question set; this records the *knob* the operator turned.
        config["level"] = level
    # A system that produces no answers by design (retrieval_only) never runs
    # the judge — its rows stay `pending` and the answer metrics are skipped.
    generates_answers = bool(getattr(system, "generates_answers", True))
    config["generates_answers"] = generates_answers
    context_passages = getattr(system, "context_passages", None)
    if context_passages is not None:
        # `--context-passages N` was set on the CLI: pin the
        # pre-seed sensitivity knob so replay runs are comparable per N.
        config["context_passages"] = context_passages
    record = BenchRunRecord(
        run_group=run_group,
        label=full_label,
        started_at=started,
        status="running",
        dataset_name=dataset.name,
        dataset_hash=dataset.hash,
        subset_hash=subset_hash_val,
        system=sys_name,
        profile_name=profile_name,
        profile_hash=profile_hash,
        answer_model=answer_model,
        judge_model=judge_model,
        scope=scope,
        config_json=config,
        judge_shares_endpoint=judge_shares_endpoint,
        calibration=calibration,
        trusted=trusted,
    )
    run_id = await store.insert_run(record)
    record = replace(record, id=run_id)

    cache_enabled = bool(BENCH_JUDGE_CACHE.default)
    retries = int(BENCH_JUDGE_RETRIES.default)
    n_q = len(questions)

    # keyed outputs for stage 2 (concurrency invariance: results keyed by qid)
    outputs: dict[str, QuestionOutput] = {}
    latencies: dict[str, float] = {}

    try:
        # ── Stage 1: pipeline (sequential) ───────────────────────────────
        for i, q in enumerate(questions):
            t0 = time.monotonic()
            try:
                output = await system.run_one(q)
            except Exception as exc:  # a single question failure is recorded, not fatal
                output = QuestionOutput(
                    answer_text="",
                    retrieved_paths=(),
                    abstained=False,
                    error=f"{type(exc).__name__}: {exc}",
                    trace={},
                )
            latency_ms = (time.monotonic() - t0) * 1000.0
            latencies[q.id] = latency_ms
            outputs[q.id] = output

            # Source metrics are deterministic — compute immediately.
            shr = source_hit_rank(output.retrieved_paths, q.sources)
            scov = source_coverage(output.retrieved_paths, q.sources)
            rounds = output.rounds

            row = BenchQuestionResult(
                run_id=run_id,
                question_id=q.id,
                capability=q.capability,
                difficulty=q.difficulty,
                question_text=q.question,
                expected_answer=q.answer,
                answer_text=output.answer_text,
                abstained=output.abstained,
                verdict=Verdict.PENDING.value,
                retrieved_paths=output.retrieved_paths,
                source_hit_rank=shr,
                source_coverage=scov,
                rounds=rounds,
                latency_ms=latency_ms,
                error=output.error,
                trace=output.trace or None,
                input_tokens=output.input_tokens,
                output_tokens=output.output_tokens,
            )
            await store.insert_question_result(run_id, row)

            if progress is not None:
                progress(
                    ProgressUpdate(
                        system=sys_name, stage="pipeline", done=i + 1, total=n_q, run_id=run_id
                    )
                )

        scored: list[ScoredQuestion] = []
        if generates_answers:
            sem = asyncio.Semaphore(max(1, judge_concurrency))
            tasks = [
                _judge_one(
                    q=q,
                    output=outputs[q.id],
                    judge=judge,
                    judge_model=judge_model,
                    cache_enabled=cache_enabled,
                    store=store,
                    retries=retries,
                    semaphore=sem,
                )
                for q in questions
            ]
            outcomes = await asyncio.gather(*tasks)

            for i, (q, outcome) in enumerate(zip(questions, outcomes, strict=True)):
                sfc = _sub_fact_coverage(outcome, q)
                await store.update_verdict(
                    run_id=run_id,
                    question_id=q.id,
                    verdict=outcome.verdict.value,
                    reason=outcome.reason,
                    sub_fact_coverage=sfc,
                )
                scored.append(
                    score_question(
                        question=q,
                        retrieved_paths=outputs[q.id].retrieved_paths,
                        outcome=outcome,
                        abstained=outputs[q.id].abstained,
                        answer_input_tokens=outputs[q.id].input_tokens,
                        answer_output_tokens=outputs[q.id].output_tokens,
                        latency_ms=latencies[q.id],
                    )
                )
                if progress is not None:
                    progress(
                        ProgressUpdate(
                            system=sys_name, stage="judging", done=i + 1, total=n_q, run_id=run_id
                        )
                    )
        else:
            for q in questions:
                pending_outcome = JudgeOutcome(
                    verdict=Verdict.PENDING,
                    reason="skipped (retrieval_only system)",
                )
                scored.append(
                    score_question(
                        question=q,
                        retrieved_paths=outputs[q.id].retrieved_paths,
                        outcome=pending_outcome,
                        abstained=outputs[q.id].abstained,
                        latency_ms=latencies[q.id],
                    )
                )

        all_traces = [outputs[q.id].trace for q in questions if outputs[q.id].trace]
        metrics = _compute_metrics(
            scored,
            answer_model=answer_model,
            traces=all_traces,
            with_answers=generates_answers,
        )
        ended = now_iso()
        final_record = replace(
            record,
            finished_at=ended,
            status="complete",
            metrics_json=metrics,
        )
        await store.update_run(run_id, final_record)
        if progress is not None:
            progress(
                ProgressUpdate(
                    system=sys_name, stage="complete", done=n_q, total=n_q, run_id=run_id
                )
            )
        return final_record

    except Exception as exc:
        ended = now_iso()
        reason = f"{type(exc).__name__}: {exc}"
        error_record = replace(
            record,
            finished_at=ended,
            status="failed",
            abort_reason=reason,
            # Stash in config_json too: there is no dedicated column, and
            # _row_to_run lifts the reason back from here on reload.
            config_json={**record.config_json, "abort_reason": reason},
        )
        await store.update_run(run_id, error_record)
        raise


async def run_benchmark(
    *,
    systems: Sequence[SystemUnderTest],
    questions: Sequence[BenchQuestion],
    store: BenchStore,
    judge: JudgeLLM | None,
    judge_model: str,
    scope: str = "",
    label: str = "",
    run_group: str = "",
    dataset: BenchDataset,
    config_snapshot: Mapping[str, object] | None = None,
    economy: str | None = None,
    context_profile: str | None = None,
    settings_set: Mapping[str, str] | None = None,
    judge_concurrency: int = int(BENCH_JUDGE_CONCURRENCY.default),
    judge_shares_endpoint: bool = False,
    max_concurrent: int = int(BENCH_MAX_CONCURRENT.default),
    repeats: int = int(BENCH_REPEATS.default),
    progress: Callable[[ProgressUpdate], None] | None = None,
    level: int | None = None,
) -> list[BenchRunRecord]:
    """Run a matrix of systems over a dataset, producing one run_group.

    One invocation produces one ``run_group`` (uuid) + one ``bench_runs`` row
    per cell (system x repeat). Cells run at most ``max_concurrent`` at a time
    (default 1 — shared inference endpoints contaminate reported latency at
    N-wide pipeline concurrency; raise it only for a dedicated endpoint).
    ``repeats > 1`` runs each cell N times for variance.

    Each cell uses two-stage execution: the pipeline writes
    ``pending`` rows immediately on answer completion; a batch judging pass
    with ``judge_concurrency`` workers grades them keyed by ``question_id``.
    """
    group = run_group or str(uuid.uuid4())
    sub_hash = subset_hash(list(questions))
    cells = [(repeat, system) for repeat in range(repeats) for system in systems]
    sem = asyncio.Semaphore(max(1, max_concurrent))

    calibration = await measure_bench_calibration(
        judge,
        judge_model,
        concurrency=judge_concurrency,
    )
    trusted = calibration is not None and calibration >= float(
        BENCH_CALIBRATION_MIN_CORRELATION.default
    )

    async def _run_cell_wrapped(repeat: int, system: SystemUnderTest) -> BenchRunRecord:
        async with sem:
            return await _run_cell(
                system=system,
                questions=questions,
                store=store,
                judge=judge,
                judge_model=judge_model,
                run_group=group,
                label=label,
                scope=scope,
                dataset=dataset,
                subset_hash_val=sub_hash,
                config_snapshot=config_snapshot,
                economy=economy,
                context_profile=context_profile,
                settings_set=settings_set,
                judge_concurrency=judge_concurrency,
                judge_shares_endpoint=judge_shares_endpoint,
                max_concurrent=max_concurrent,
                repeat_index=repeat if repeats > 1 else 0,
                progress=progress,
                calibration=calibration,
                trusted=trusted,
                level=level,
            )

    tasks = [asyncio.create_task(_run_cell_wrapped(r, s)) for r, s in cells]
    try:
        results = list(await asyncio.gather(*tasks))
    except BaseException:
        # A cell failure must not leave sibling cells running detached: after
        # the first exception the caller (api/bench._run_to_completion) marks
        # every still-running row aborted and reports the group finished — an
        # orphaned cell would keep burning LLM calls and later flip its row
        # from aborted back to complete. Cancel the outstanding cells and wait
        # for the cancellations to land before propagating.
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return results


async def rejudge_run(
    store: BenchStore,
    judge: JudgeLLM | None,
    judge_model: str,
    run_id: int,
    questions: Mapping[str, BenchQuestion] | None = None,
    *,
    judge_concurrency: int = int(BENCH_JUDGE_CONCURRENCY.default),
    progress: Callable[[ProgressUpdate], None] | None = None,
) -> int:
    """Re-grade ``pending`` rows on a stored run without re-running the pipeline.

    Grades each pending question via the judge (cache-aware). Questions whose
    full :class:`BenchQuestion` is unavailable (``questions`` is None or missing
    a qid) can only be graded from the judge cache — a cache miss leaves them
    ``pending``. This is the seam that lets a stored run be re-judged by a
    stronger model or recovered after an abort.

    Returns the number of rows graded (verdict moved off ``pending``).
    """
    pending = await store.list_pending_results(run_id)
    if not pending:
        return 0
    record = await store.get_run(run_id)
    answer_model = record.answer_model if record else ""
    if record is not None and not bool(record.config_json.get("generates_answers", True)):
        raise ValueError(f"run {run_id} ({record.system}) generates no answers — nothing to judge")
    cache_enabled = bool(BENCH_JUDGE_CACHE.default)
    retries = int(BENCH_JUDGE_RETRIES.default)
    sem = asyncio.Semaphore(max(1, judge_concurrency))
    sys_name = record.system if record else "unknown"
    n_p = len(pending)

    async def _rejudge(row: BenchQuestionResult) -> JudgeOutcome | None:
        q = questions.get(row.question_id) if questions else None
        if q is None:
            return None
        async with sem:
            return await judge_verdict(
                question=q,
                model_answer=row.answer_text,
                abstained=row.abstained,
                judge=judge,
                judge_model=judge_model,
                retries=retries,
                cache_get=store.judge_cache_get if cache_enabled else None,
                cache_put=store.judge_cache_put if cache_enabled else None,
            )

    outcomes_list = await asyncio.gather(*(_rejudge(r) for r in pending))
    graded = 0
    scored: list[ScoredQuestion] = []
    for row, outcome in zip(pending, outcomes_list, strict=True):
        q = questions.get(row.question_id) if questions else None
        if outcome is not None and q is not None:
            sfc = _sub_fact_coverage(outcome, q)
            updated = replace(
                row,
                verdict=outcome.verdict.value,
                verdict_reason=outcome.reason,
                sub_fact_coverage=sfc,
            )
            await store.update_question_result(run_id, row.question_id, updated)
            scored.append(
                score_question(
                    question=q,
                    retrieved_paths=row.retrieved_paths,
                    outcome=outcome,
                    abstained=row.abstained,
                )
            )
            graded += 1
        if progress is not None:
            progress(
                ProgressUpdate(
                    system=sys_name, stage="judging", done=graded, total=n_p, run_id=run_id
                )
            )

    if scored and record is not None:
        all_rows = await store.list_question_results(run_id)
        all_scored = _rebuild_scored(all_rows, questions or {}, judge_model)
        metrics = _compute_metrics(
            all_scored,
            answer_model=answer_model,
            traces=[r.trace for r in all_rows if r.trace],
        )
        final = replace(record, metrics_json=metrics)
        await store.update_run(run_id, final)
    return graded


def _rebuild_scored(
    rows: Sequence[BenchQuestionResult],
    questions: Mapping[str, BenchQuestion],
    judge_model: str,
) -> list[ScoredQuestion]:
    """Rebuild ScoredQuestion list from stored rows (for metrics recompute)."""
    out: list[ScoredQuestion] = []
    for row in rows:
        q = questions.get(row.question_id)
        if q is None:
            continue
        try:
            verdict = Verdict(row.verdict)
        except ValueError:
            continue
        sfp: tuple[bool, ...] = ()
        if q.sub_facts and row.sub_fact_coverage is not None:
            n_total = len(q.sub_facts)
            n_present = round(row.sub_fact_coverage * n_total)
            sfp = tuple(True for _ in range(n_present)) + tuple(
                False for _ in range(n_total - n_present)
            )
        outcome = JudgeOutcome(
            verdict=verdict,
            reason=row.verdict_reason,
            sub_facts_present=sfp,
            abstained=row.abstained,
            judge_model=judge_model,
        )
        out.append(
            score_question(
                question=q,
                retrieved_paths=row.retrieved_paths,
                outcome=outcome,
                abstained=row.abstained,
                answer_input_tokens=row.input_tokens,
                answer_output_tokens=row.output_tokens,
                latency_ms=row.latency_ms,
            )
        )
    return out


def _verdict_is_correct(v: str) -> bool:
    return v == Verdict.CORRECT.value


def _verdict_is_wrong(v: str) -> bool:
    return v in (Verdict.INCORRECT.value, Verdict.PARTIAL.value)


async def compare_runs(store: BenchStore, run_a: int, run_b: int) -> CompareResult:
    """Compare two runs: per-question buckets + aggregate deltas.

    ``fixed`` = A wrong → B correct (improvements). ``broken`` = A correct → B
    wrong (regressions). ``both_correct`` / ``both_wrong``. ``only_a`` /
    ``only_b`` = questions present in only one run (different subsets —
    ``shared_denominator`` is the intersection).
    """
    rows_a = {r.question_id: r for r in await store.list_question_results(run_a)}
    rows_b = {r.question_id: r for r in await store.list_question_results(run_b)}

    ids_a = set(rows_a)
    ids_b = set(rows_b)
    shared = ids_a & ids_b
    only_a = tuple(sorted(ids_a - ids_b))
    only_b = tuple(sorted(ids_b - ids_a))

    fixed: list[str] = []
    broken: list[str] = []
    both_correct: list[str] = []
    both_wrong: list[str] = []
    for qid in shared:
        va = rows_a[qid].verdict
        vb = rows_b[qid].verdict
        a_correct = _verdict_is_correct(va)
        b_correct = _verdict_is_correct(vb)
        if a_correct and b_correct:
            both_correct.append(qid)
        elif not a_correct and not b_correct:
            both_wrong.append(qid)
        elif not a_correct and b_correct:
            fixed.append(qid)
        else:
            broken.append(qid)

    deltas: dict[str, float] = {}
    if shared:
        n = len(shared)
        a_acc = sum(1 for qid in shared if _verdict_is_correct(rows_a[qid].verdict)) / n
        b_acc = sum(1 for qid in shared if _verdict_is_correct(rows_b[qid].verdict)) / n
        a_found = sum(1 for qid in shared if rows_a[qid].source_hit_rank is not None) / n
        b_found = sum(1 for qid in shared if rows_b[qid].source_hit_rank is not None) / n
        deltas["strict_accuracy"] = b_acc - a_acc
        deltas["source_recall"] = b_found - a_found

    return CompareResult(
        run_a=run_a,
        run_b=run_b,
        shared_denominator=len(shared),
        fixed=tuple(sorted(fixed)),
        broken=tuple(sorted(broken)),
        both_correct=tuple(sorted(both_correct)),
        both_wrong=tuple(sorted(both_wrong)),
        only_a=only_a,
        only_b=only_b,
        deltas=deltas,
    )


def resolve_judge_concurrency(
    requested: int,
    *,
    answer_endpoint: str,
    judge_endpoint: str,
) -> tuple[int, bool]:
    """Clamp judge concurrency to 1 when the judge shares the answer endpoint.

    Returns ``(concurrency, shares_endpoint)``. The clamp is cheap insurance:
    today the judge runs after the pipeline so a shared endpoint only slows
    judging, but the clamp keeps the latency story sound if judging is ever
    overlapped.
    """
    shares = bool(answer_endpoint and judge_endpoint) and answer_endpoint.rstrip(
        "/"
    ) == judge_endpoint.rstrip("/")
    concurrency = 1 if shares else max(1, requested)
    return concurrency, shares


__all__ = [
    "BENCH_JUDGE_CONCURRENCY",
    "BENCH_MAX_CONCURRENT",
    "BENCH_REPEATS",
    "BENCH_SYSTEMS",
    "BENCH_TRACE_RETENTION_DAYS",
    "BenchQuestionResult",
    "BenchRunRecord",
    "BenchStore",
    "CompareResult",
    "ProgressUpdate",
    "QuestionOutput",
    "SystemUnderTest",
    "compare_runs",
    "rejudge_run",
    "resolve_judge_concurrency",
    "run_benchmark",
]
