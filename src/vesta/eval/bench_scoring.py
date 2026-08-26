"""Unified benchmark scoring — source metrics, judge rubric, reference points,
failure attribution.

Two metric families that are NEVER mixed:

* **Source metrics** — deterministic, no LLM. "Did retrieval find the article the
  answer lives in?" A set-membership question; an LLM adds noise + cost. Computed
  from retrieved paths vs. the question's ``sources[]`` via
  :func:`vesta.eval.metrics.path_matches`.
* **Answer metrics** — always the LLM judge, via a structured rubric that returns
  strict JSON. NO lexical fallback exists in this module: a failed/unparseable
  judge yields ``Verdict.UNJUDGED``, the run is flagged incomplete, and the
  question is never counted correct.

Boundary: imports ONLY ``vesta.retrieval`` +
``vesta.config`` (+ ``vesta.eval`` siblings, which are the same package and so do
not count toward the dependency cap). No ``db``/``zim``/``inference``/``answer``/
``api``. The judge is the injected :class:`vesta.eval.answer_metrics.JudgeLLM`.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from statistics import mean

from vesta.config import get_or_default
from vesta.config.settings import setting
from vesta.eval.answer_metrics import JudgeLLM, compute_calibration_correlation
from vesta.eval.bench_dataset import BenchQuestion, BenchSource
from vesta.eval.metrics import path_matches

# Re-export so callers can measure judge calibration with the unified settings.
__all__ = [
    "BENCH_CALIBRATION_MIN_CORRELATION",
    "BENCH_CALIBRATION_PATH",
    "BENCH_JUDGE_CACHE",
    "BENCH_JUDGE_MAX_TOKENS",
    "BENCH_JUDGE_RETRIES",
    "BENCH_JUDGE_TEMPERATURE",
    "RUBRIC_PROMPT_VERSION",
    "AnswerMetrics",
    "AttributionMatrix",
    "CapabilityBreakdown",
    "JudgeOutcome",
    "ReferencePoints",
    "ScoredQuestion",
    "SourceMetrics",
    "TokenUsage",
    "Verdict",
    "aggregate_answer_metrics",
    "aggregate_latency",
    "aggregate_peak_context",
    "aggregate_source_metrics",
    "aggregate_token_usage",
    "attribution_by_capability",
    "attribution_matrix",
    "judge_cache_key",
    "judge_verdict",
    "measure_bench_calibration",
    "reference_points",
    "render_rubric",
    "retrieved_precision",
    "rubric_prompt_hash",
    "score_question",
    "source_coverage",
    "source_hit_rank",
    "source_mrr",
    "source_recall_at",
    "stage_latency_breakdown",
]

# ── Settings ────────────────────────────────────────────────────────────────

BENCH_JUDGE_TEMPERATURE = setting(
    "bench.judge.temperature",
    float,
    0.0,
    group="Benchmark / Judge",
    help="Judge temperature (0 = deterministic). Judging must be repeatable so a "
    "stored run re-judged by a stronger model is byte-identical at any concurrency.",
    min=0.0,
    max=2.0,
    hot=True,
)
BENCH_JUDGE_MAX_TOKENS = setting(
    "bench.judge.max_tokens",
    int,
    4096,
    group="Benchmark / Judge",
    help="Cap judge output. Reasoning models burn chain-of-thought tokens before "
    "the verdict. 4096 keeps headroom.",
    min=256,
    max=16384,
    hot=True,
)
BENCH_JUDGE_RETRIES = setting(
    "bench.judge.retries",
    int,
    1,
    group="Benchmark / Judge",
    help="Retries on a judge parse failure before the verdict becomes 'unjudged' "
    "(which marks the run incomplete). One retry covers transient garbling.",
    min=0,
    max=5,
    hot=True,
)
BENCH_JUDGE_CACHE = setting(
    "bench.judge.cache",
    bool,
    True,
    group="Benchmark / Judge",
    help="Cache judge verdicts keyed on the rendered rubric (which embeds ground "
    "truth) so a ground-truth fix invalidates its own cache entries "
    "and a retrieval-only change re-judges only changed answers.",
    hot=False,
)
BENCH_CALIBRATION_PATH = setting(
    "bench.calibration_path",
    str,
    "benchmarks/calibration_v1.json",
    group="Benchmark / Judge",
    help="Hand-scored calibration subset (25 items, all three verdicts + both "
    "abstention directions) the judge is validated against. Pearson < "
    "bench.calibration_min_correlation marks the run untrusted.",
    hot=False,
)
BENCH_CALIBRATION_MIN_CORRELATION = setting(
    "bench.calibration_min_correlation",
    float,
    0.7,
    group="Benchmark / Judge",
    help="Minimum Pearson correlation between judge and hand verdicts for the run "
    "to be trusted. Below this the headline reads 'untrusted'.",
    min=-1.0,
    max=1.0,
    hot=False,
)


# ── Verdict ─────────────────────────────────────────────────────────────────


class Verdict(StrEnum):
    """Judge verdict + the two non-judged states.

    ``CORRECT``/``PARTIAL``/``INCORRECT`` are the judge's three outcomes.
    ``PENDING`` = answered, not yet judged (the two-stage pipeline writes rows
    pending before the batch judge). ``UNJUDGED`` = the judge failed after retry;
    any ``UNJUDGED > 0`` marks the run incomplete and the question is NEVER
    counted correct.
    """

    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"
    PENDING = "pending"
    UNJUDGED = "unjudged"

    @property
    def is_judged(self) -> bool:
        """One of the three scored outcomes (not pending/unjudged)."""
        return self in (Verdict.CORRECT, Verdict.PARTIAL, Verdict.INCORRECT)

    def to_score(self) -> float:
        """Numeric for weighted accuracy + calibration correlation."""
        return {Verdict.CORRECT: 1.0, Verdict.PARTIAL: 0.5}.get(self, 0.0)


# ── Per-question scored record (the unit every aggregator consumes) ─────────


@dataclass(frozen=True)
class JudgeOutcome:
    """The judge's structured verdict on one answer."""

    verdict: Verdict
    reason: str = ""
    sub_facts_present: tuple[bool, ...] = ()
    abstained: bool = False
    judge_model: str = ""


@dataclass(frozen=True)
class ScoredQuestion:
    """One question's system output + judged verdict.

    Built by :func:`score_question` (or by the runner from a
    ``QuestionOutput`` + ``JudgeOutcome``). Every aggregator consumes this so
    the source / answer / reference / attribution / token computations share one
    record.
    """

    question: BenchQuestion
    retrieved_paths: tuple[str, ...]
    verdict: Verdict
    reason: str = ""
    abstained: bool = False
    sub_facts_present: tuple[bool, ...] = ()
    judge_model: str = ""
    error: str | None = None
    answer_input_tokens: int = 0
    answer_output_tokens: int = 0
    latency_ms: float = 0.0


def score_question(
    question: BenchQuestion,
    retrieved_paths: Sequence[str],
    outcome: JudgeOutcome,
    *,
    abstained: bool,
    answer_input_tokens: int = 0,
    answer_output_tokens: int = 0,
    latency_ms: float = 0.0,
) -> ScoredQuestion:
    """Bundle a question, its retrieved paths, and the judge outcome.

    ``abstained`` is the HARNESS decision (did the pipeline decline to answer?)
    and is the authoritative abstention source for the metrics — never the
    judge's echoed field, which defaults false on omission or judge failure.
    """
    return ScoredQuestion(
        question=question,
        retrieved_paths=tuple(retrieved_paths),
        verdict=outcome.verdict,
        reason=outcome.reason,
        abstained=abstained,
        sub_facts_present=outcome.sub_facts_present,
        judge_model=outcome.judge_model,
        answer_input_tokens=answer_input_tokens,
        answer_output_tokens=answer_output_tokens,
        latency_ms=latency_ms,
    )


# ── Reading metrics back ────────────────────────────────────────────────────


def metric_lookup(metrics: dict[str, object], path: str, *, flat_fallback: bool = False) -> object:
    """Look one metric up in a persisted ``metrics_json`` blob by dotted path
    (``answer.strict_accuracy``).

    ``flat_fallback`` also accepts the retired flat layout
    (``answer_strict_accuracy``): historical eval_runs rows predate the nested
    shape. Only the CLI report paths need it; API responses read nested rows
    only.
    """
    node: object = metrics
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            node = None
            break
        node = node[part]
    if node is not None:
        return node
    if flat_fallback:
        return metrics.get(path.replace(".", "_"))
    return None


# ── A. Source metrics (deterministic, no LLM) ───────────────────────────────


def _required_sources(sources: Sequence[BenchSource]) -> list[BenchSource]:
    return [s for s in sources if s.required]


def source_hit_rank(retrieved: Sequence[str], sources: Sequence[BenchSource]) -> int | None:
    """1-based rank of the first ``required`` source in ``retrieved``; ``None`` = miss.

    Path comparison reuses :func:`vesta.eval.metrics.path_matches` (ZIM
    path/title normalisation). Non-required (alternative) sources do not count.
    """
    required = [s.article_path for s in sources if s.required]
    for i, p in enumerate(retrieved):
        if any(path_matches(p, exp) for exp in required):
            return i + 1
    return None


def source_recall_at(retrieved: Sequence[str], sources: Sequence[BenchSource], k: int) -> bool:
    """True if the first required source is within the top-``k`` retrieved."""
    rank = source_hit_rank(retrieved, sources)
    return rank is not None and rank <= k


def source_coverage(retrieved: Sequence[str], sources: Sequence[BenchSource]) -> float:
    """Fraction of ``required`` sources present anywhere in ``retrieved``.

    The multi-hop metric: recall@k says "found one", coverage says "found them
    all". ``0.0`` when there are no required sources
    (out_of_corpus — undefined, excluded from the set mean).
    """
    required = _required_sources(sources)
    if not required:
        return 0.0
    hits = sum(1 for s in required if any(path_matches(p, s.article_path) for p in retrieved))
    return hits / len(required)


def source_mrr(retrieved: Sequence[str], sources: Sequence[BenchSource]) -> float:
    """Per-question reciprocal rank: ``1/rank`` of the first required source, ``0`` if miss.

    The set-level MRR is the mean of this over source-eligible questions.
    """
    rank = source_hit_rank(retrieved, sources)
    return 1.0 / rank if rank is not None else 0.0


def retrieved_precision(retrieved: Sequence[str], sources: Sequence[BenchSource]) -> float:
    """Fraction of retrieved cards that name a gold source.

    Guards against "retrieve 50 cards, hit by volume": a system that floods the
    context hits recall but tanks precision.
    """
    if not retrieved:
        return 0.0
    gold = [s.article_path for s in sources]  # all gold, required + alternative
    hits = sum(1 for p in retrieved if any(path_matches(p, exp) for exp in gold))
    return hits / len(retrieved)


@dataclass(frozen=True)
class SourceMetrics:
    """Set-level source metrics. The denominator ``n`` excludes out_of_corpus."""

    n: int  # source-eligible questions (>=1 required source)
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    mean_coverage: float
    mrr: float
    mean_precision: float


def _empty_source_metrics() -> SourceMetrics:
    return SourceMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def aggregate_source_metrics(results: Sequence[ScoredQuestion]) -> SourceMetrics:
    """Reduce per-question results into set-level source metrics.

    ``out_of_corpus`` questions (no required source) are EXCLUDED from the
    denominator — they have no gold source, so including them silently drags
    recall down.
    """
    eligible = [r for r in results if _required_sources(r.question.sources)]
    n = len(eligible)
    if n == 0:
        return _empty_source_metrics()

    def _frac(k: int) -> float:
        return (
            sum(1 for r in eligible if source_recall_at(r.retrieved_paths, r.question.sources, k))
            / n
        )

    return SourceMetrics(
        n=n,
        recall_at_1=_frac(1),
        recall_at_5=_frac(5),
        recall_at_10=_frac(10),
        recall_at_20=_frac(20),
        mean_coverage=mean(
            source_coverage(r.retrieved_paths, r.question.sources) for r in eligible
        ),
        mrr=mean(source_mrr(r.retrieved_paths, r.question.sources) for r in eligible),
        mean_precision=mean(
            retrieved_precision(r.retrieved_paths, r.question.sources) for r in eligible
        ),
    )


# ── Committed rubric prompt (structured JSON output; hashed) ────────────────
# The judge grades answer-vs-ground-truth ONLY. It is BLIND to retrieved context
# (we never pass it) and FORBIDDEN from its own parametric knowledge:
# a judge that sees the context starts grading faithfulness, a
# different metric, and a judge using its own memory grades its own style.

RUBRIC_PROMPT_VERSION = "16.1"

_SUB_FACT_HINT = (
    "For each SUB-FACT, report whether the model answer contains that specific "
    "fact (true/false), in declaration order."
)


def _render_rubric(
    *,
    question: str,
    known_answer: str,
    known_detail: str,
    model_answer: str,
    abstained: bool,
    expected_behavior: str,
    sub_facts: Sequence[str],
) -> str:
    answer_for_prompt = (
        "[the system abstained / gave no answer]"
        if abstained or not model_answer.strip()
        else model_answer
    )
    detail_line = f"\n(ground-truth detail: {known_detail})" if known_detail else ""
    if sub_facts:
        sub_block = "\nSUB-FACTS (each must be present for a complete answer):\n" + "\n".join(
            f"- {sf}" for sf in sub_facts
        )
        sub_instruction = (
            "\n\n"
            + _SUB_FACT_HINT
            + ' Return a "sub_facts_present" array of booleans, one per sub-fact in order.'
        )
    else:
        sub_block = ""
        sub_instruction = (
            '\n\nOmit "sub_facts_present" (or return an empty array) — this question '
            "has no sub-facts."
        )
    if expected_behavior == "abstain":
        behavior_line = (
            "This question is OUT OF CORPUS: the answer is provably absent from the "
            "scoped archives. A correct system ABSTAINS. If it abstained, verdict is "
            '"correct"; a confident invented answer is "incorrect".'
        )
    else:
        behavior_line = (
            "If the system abstained or gave no factual answer, the verdict is "
            '"incorrect" — the answer exists in the corpus.'
        )
    return (
        "You are a strict grader. Judge ONLY whether the MODEL ANSWER conveys the "
        "same essential fact(s) as the KNOWN CORRECT ANSWER.\n\n"
        "HARD RULES:\n"
        "1. Do NOT use your own parametric knowledge. Rule only on whether the model "
        "answer matches the provided ground truth.\n"
        "2. You are given NO retrieved context. Grade answer-vs-ground-truth only — "
        "do not infer what the system may have seen.\n"
        '3. "Based on the article…" phrasing, quoting, or markdown do NOT count '
        "against the answer.\n\n"
        f"QUESTION: {question}\n"
        f"KNOWN CORRECT ANSWER: {known_answer}"
        f"{detail_line}"
        f"{sub_block}\n\n"
        f"MODEL ANSWER TO GRADE: {answer_for_prompt}\n\n"
        f"{behavior_line}{sub_instruction}\n\n"
        "Verdict scale:\n"
        "- correct   : states every essential fact (entity / name / number / date), "
        "and every sub-fact where applicable.\n"
        "- partial   : the core answer is right but one required component or "
        "sub-fact is missing or wrong.\n"
        "- incorrect : core answer wrong, hallucinated, absent, or (for an "
        "in-corpus question) the system abstained.\n\n"
        "Respond with STRICT JSON only — no prose, no markdown fences:\n"
        "{\n"
        '  "verdict": "correct" | "partial" | "incorrect",\n'
        '  "reason": "one short sentence",\n'
        '  "sub_facts_present": [true, false],\n'
        '  "abstained": true | false\n'
        "}\n"
        'where "abstained" is true iff the model declined to answer.'
    )


def rubric_prompt_hash() -> str:
    """sha256 of the rendered rubric over placeholder inputs (a prompt change is
    detectable). Truncated to 16 hex, mirroring the ``rubric_prompt_hash``
    convention so a run's ``judge_prompt_hash`` pin is stable and comparable.

    Note: this is the TEMPLATE hash. The judge CACHE key hashes the
    fully RENDERED rubric (which embeds ground truth) so a ground-truth fix
    invalidates its own cache entries.
    """
    sample = _render_rubric(
        question="<question>",
        known_answer="<known_answer>",
        known_detail="<known_detail>",
        model_answer="<model_answer>",
        abstained=False,
        expected_behavior="answer",
        sub_facts=["<sub_fact_a>", "<sub_fact_b>"],
    )
    return hashlib.sha256(sample.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def render_rubric(
    *,
    question: BenchQuestion,
    model_answer: str,
    abstained: bool,
) -> str:
    """Public alias of :func:`_render_rubric` for a :class:`BenchQuestion`.

    The runner and the judge cache both need
    the fully-rendered prompt: the cache key hashes it (so a ground-truth fix
    invalidates its own entries), and re-judging a stored run rebuilds
    it from the pinned question.
    """
    return _render_rubric(
        question=question.question,
        known_answer=question.answer,
        known_detail=question.answer_detail,
        model_answer=model_answer,
        abstained=abstained,
        expected_behavior=question.expected_behavior,
        sub_facts=[sf.fact for sf in question.sub_facts],
    )


def judge_cache_key(
    rendered_rubric: str,
    question_id: str,
    model_answer: str,
    judge_model: str,
) -> str:
    """Cache key for ``bench_judge_cache``.

    The rendered rubric EMBEDS the ground truth (question text, known answer,
    detail, sub-facts) so editing the ground truth changes the key and
    invalidates the stale entry — a template-only hash would not. The
    ``question_id`` and ``model_answer`` are included explicitly as belt-and-
    suspenders (they are already embedded in the rubric).
    """
    raw = f"{rendered_rubric}\x1f{question_id}\x1f{model_answer}\x1f{judge_model}"
    return hashlib.sha256(raw.encode("utf-8"), usedforsecurity=False).hexdigest()


# ── Judge call (structured JSON; one retry, then unjudged; NO lexical path) ──


def _extract_json(raw: str) -> object | None:
    """Robustly extract the first JSON object from a judge response.

    Tolerates surrounding prose and ```json fences. Returns ``None`` when no
    parseable object is found.
    """
    text = raw.strip()
    if text.startswith("```"):
        # Drop the opening fence line; strip a trailing fence.
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        result: object = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return result


def _parse_judge_json(raw: str, judge_model: str) -> JudgeOutcome | None:
    """Parse a judge response into a :class:`JudgeOutcome`, or ``None`` (unparseable)."""
    obj = _extract_json(raw)
    if not isinstance(obj, Mapping):
        return None
    token = str(obj.get("verdict") or "").strip().lower()
    try:
        verdict = Verdict(token)  # only correct/partial/incorrect parse here
    except ValueError:
        return None
    reason = str(obj.get("reason") or "")[:300]
    sfp_raw = obj.get("sub_facts_present")
    sub_facts_present = tuple(bool(x) for x in sfp_raw) if isinstance(sfp_raw, list) else ()
    abstained = bool(obj.get("abstained", False))
    return JudgeOutcome(
        verdict=verdict,
        reason=reason,
        sub_facts_present=sub_facts_present,
        abstained=abstained,
        judge_model=judge_model,
    )


def _unjudged(judge_model: str, reason: str) -> JudgeOutcome:
    return JudgeOutcome(verdict=Verdict.UNJUDGED, reason=reason, judge_model=judge_model)


async def judge_verdict(
    *,
    question: BenchQuestion,
    model_answer: str,
    abstained: bool,
    judge: JudgeLLM | None,
    judge_model: str,
    retries: int | None = None,
    cache_get: Callable[[str], Awaitable[JudgeOutcome | None]] | None = None,
    cache_put: Callable[[str, JudgeOutcome], Awaitable[None]] | None = None,
) -> JudgeOutcome:
    """Grade one answer via the structured rubric.

    Calls the judge once, parses strict JSON, one retry on parse failure
    (configurable via ``retries``), then ``Verdict.UNJUDGED``. No judge / empty
    model / judge exception ⇒ ``UNJUDGED``. There is NO lexical fallback
    anywhere in this module: an unjudged question is never
    counted correct and marks the run incomplete.

    Cache hooks: when ``cache_get`` / ``cache_put`` are provided (by the
    runner, wired to the ``bench_judge_cache`` table), a hit short-circuits the
    judge call entirely and a judged result is stored. Only the three scored
    verdicts are cached — ``UNJUDGED`` is never cached so a re-judge retries.
    """
    if judge is None or not judge_model:
        return _unjudged(judge_model, "no judge configured")
    if retries is None:
        retries = int(get_or_default(BENCH_JUDGE_RETRIES))
    prompt = render_rubric(question=question, model_answer=model_answer, abstained=abstained)
    if cache_get is not None:
        key = judge_cache_key(prompt, question.id, model_answer, judge_model)
        try:
            cached = await cache_get(key)
        except Exception:
            cached = None
        if cached is not None:
            return cached
    attempts = 1 + max(0, retries)
    last_err = ""
    result: JudgeOutcome | None = None
    for _ in range(attempts):
        try:
            raw = await judge.judge(prompt)
        except Exception as exc:  # judge endpoint failure — not a parse problem
            last_err = f"judge error: {exc!r}"
            continue
        parsed = _parse_judge_json(raw, judge_model)
        if parsed is not None:
            result = parsed
            break
        last_err = "unparseable judge output"
    if result is None:
        result = _unjudged(judge_model, last_err)
    # Only cache the three scored verdicts — unjudged must be retriable.
    if result.verdict.is_judged and cache_put is not None:
        with contextlib.suppress(Exception):
            await cache_put(
                judge_cache_key(prompt, question.id, model_answer, judge_model),
                result,
            )
    return result


def _hand_score(verdict: str) -> float:
    """Map a calibration item's hand verdict to the Verdict.to_score() scale."""
    v = verdict.strip().lower()
    if v == Verdict.CORRECT.value:
        return 1.0
    if v == Verdict.PARTIAL.value:
        return 0.5
    return 0.0


async def measure_bench_calibration(
    judge: JudgeLLM | None,
    judge_model: str,
    calibration_path: str | None = None,
    *,
    retries: int | None = None,
    concurrency: int = 1,
) -> float | None:
    """Judge-vs-hand Pearson correlation over the calibration subset.

    Loads the hand-scored items at ``calibration_path`` (default
    ``BENCH_CALIBRATION_PATH``), judges each item's FIXED ``model_answer``
    against its ground truth via :func:`judge_verdict`, and returns the Pearson
    correlation between the judge's verdict scores and the hand scores.

    Returns ``None`` when there is no judge, no judge model, no readable
    calibration file, or fewer than 2 scored items — *unmeasured*, not
    untrusted. A correlation below ``BENCH_CALIBRATION_MIN_CORRELATION`` marks
    the run untrusted. Calibration is a judge property (one model grades the
    fixed subset), so it is measured ONCE per run group and applied to every
    cell — not re-measured per system.
    """
    if judge is None or not judge_model:
        return None
    path = calibration_path or str(get_or_default(BENCH_CALIBRATION_PATH))
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return None

    async def _grade(it: Mapping[str, object]) -> tuple[float, float] | None:
        q_text = str(it.get("question", ""))
        known = str(it.get("known_answer", ""))
        if not q_text or not known:
            return None
        question = BenchQuestion(
            id="calibration",
            question=q_text,
            capability="lookup",
            difficulty="easy",
            slice="core",
            expected_behavior="answer",
            answer=known,
            answer_detail=str(it.get("known_detail", "")),
        )
        model_answer = str(it.get("model_answer", ""))
        outcome = await judge_verdict(
            question=question,
            model_answer=model_answer,
            abstained=not model_answer.strip(),
            judge=judge,
            judge_model=judge_model,
            retries=retries,
        )
        return _hand_score(str(it.get("hand_verdict", ""))), outcome.verdict.to_score()

    sem = asyncio.Semaphore(max(1, concurrency))

    async def _bound(it: Mapping[str, object]) -> tuple[float, float] | None:
        async with sem:
            return await _grade(it)

    pairs = [
        p for p in await asyncio.gather(*(_bound(it) for it in items if isinstance(it, dict))) if p
    ]
    if len(pairs) < 2:
        return None
    hand_scores = [h for h, _ in pairs]
    judge_scores = [j for _, j in pairs]
    return compute_calibration_correlation(hand_scores, judge_scores)


# ── B. Answer metrics (always the LLM judge) ────────────────────────────────


@dataclass(frozen=True)
class AnswerMetrics:
    """Set-level answer metrics. ``complete`` is False when any verdict is unjudged."""

    n: int
    strict_accuracy: float
    weighted_accuracy: float
    sub_fact_coverage: float
    abstention_correctness: float
    over_refusal: float
    hallucination_rate: float
    unjudged: int
    complete: bool


def _empty_answer_metrics() -> AnswerMetrics:
    return AnswerMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, True)


def aggregate_answer_metrics(results: Sequence[ScoredQuestion]) -> AnswerMetrics:
    """Reduce per-question results into set-level answer metrics.

    Accuracy denominators are the full set ``n``; ``unjudged`` questions are in
    ``n`` but never in ``correct``, so an incomplete run's accuracy is honestly
    lower and flagged via ``complete=False``. ``sub_fact_coverage`` is
    judge-derived (mean of ``sub_facts_present`` over judged questions that HAVE
    sub-facts), NOT a substring check.
    """
    n = len(results)
    if n == 0:
        return _empty_answer_metrics()
    correct = sum(1 for r in results if r.verdict == Verdict.CORRECT)
    partial = sum(1 for r in results if r.verdict == Verdict.PARTIAL)
    unjudged = sum(1 for r in results if r.verdict == Verdict.UNJUDGED)

    # sub_fact_coverage: judge-derived, over judged questions with sub-facts.
    sf_vals: list[float] = []
    for r in results:
        if not r.question.sub_facts or not r.verdict.is_judged:
            continue
        present = r.sub_facts_present
        if not present:  # judge omitted the array on a sub-fact question
            sf_vals.append(0.0)
        else:
            sf_vals.append(sum(1 for x in present if x) / len(present))
    sub_fact_cov = mean(sf_vals) if sf_vals else 0.0

    answer_q = [r for r in results if r.question.expected_behavior == "answer"]
    abstain_q = [r for r in results if r.question.expected_behavior == "abstain"]
    # over_refusal: abstained on a question that should have been answered.
    over_refusal = sum(1 for r in answer_q if r.abstained) / len(answer_q) if answer_q else 0.0
    # hallucination_rate: answered confidently on an out_of_corpus question.
    hallucination_rate = (
        sum(1 for r in abstain_q if not r.abstained) / len(abstain_q) if abstain_q else 0.0
    )
    # abstention_correctness: did the abstain DECISION match expectation? Combines
    # both directions — reduces to "fraction that abstained" on a pure-abstain set
    # and to "1 - over_refusal" on a pure-answer set.
    decided_ok = sum(1 for r in abstain_q if r.abstained) + sum(
        1 for r in answer_q if not r.abstained
    )
    abstention_correctness = decided_ok / n if n else 0.0

    return AnswerMetrics(
        n=n,
        strict_accuracy=correct / n,
        weighted_accuracy=(correct + 0.5 * partial) / n,
        sub_fact_coverage=sub_fact_cov,
        abstention_correctness=abstention_correctness,
        over_refusal=over_refusal,
        hallucination_rate=hallucination_rate,
        unjudged=unjudged,
        complete=(unjudged == 0),
    )


# ── C. Three reference points — ceiling / system / floor ────────────────────


@dataclass(frozen=True)
class ReferencePoints:
    """ceiling=oracle, floor=closed_book, system=this run.

    All four counts are taken over the same *oracle-bearing* subset of
    questions (``reference_n``); ``total`` is the full run size. Computing
    them over different subsets would let ``system`` outrun ``ceiling`` and
    push ``headroom_realised`` above 1 on partially-referenced datasets.

    ``headroom_realised`` is the headline: of the accuracy this model could
    achieve, how much did retrieval + agent deliver? Suppressed (``None``)
    when no oracle reference exists or when the oracle was verified against
    a different model than the run's answer model. It can legitimately
    exceed 1 only if the run answers an oracle-incorrect question correctly.
    """

    ceiling: int
    system: int
    floor: int
    total: int
    reference_n: int
    headroom_realised: float | None
    retrieval_regressions: int
    suppressed_reason: str = ""


def _is_correct_verdict(v: object) -> bool:
    return isinstance(v, str) and v.strip().lower() == Verdict.CORRECT.value


def reference_points(results: Sequence[ScoredQuestion], *, answer_model: str) -> ReferencePoints:
    """Compute ceiling/system/floor counts + headroom_realised.

    Ceiling, system, floor, and ``retrieval_regressions`` are all computed
    over the oracle-bearing subset so the headroom fraction compares like
    with like; with a fully-referenced dataset that subset is every question.
    ``retrieval_regressions`` counts questions the model got right
    closed-book but wrong with retrieval — context poisoning, invisible in
    a mean.
    """
    total = len(results)
    referenced = [r for r in results if r.question.oracle]
    n_reference = len(referenced)
    ceiling = sum(1 for r in referenced if _is_correct_verdict(r.question.oracle.get("verdict")))
    floor = sum(1 for r in referenced if _is_correct_verdict(r.question.closed_book.get("verdict")))
    system = sum(1 for r in referenced if r.verdict == Verdict.CORRECT)
    regressions = sum(
        1
        for r in referenced
        if _is_correct_verdict(r.question.closed_book.get("verdict"))
        and r.verdict != Verdict.CORRECT
    )

    oracle_models = {
        str(r.question.oracle["model"])
        for r in results
        if r.question.oracle and "model" in r.question.oracle
    }
    suppressed_reason = ""
    headroom: float | None
    if not oracle_models:
        suppressed_reason = "no oracle reference points recorded"
        headroom = None
    elif answer_model and oracle_models != {answer_model}:
        suppressed_reason = (
            f"oracle model(s) {sorted(oracle_models)} != answer model {answer_model!r}"
        )
        headroom = None
    else:
        denom = ceiling - floor
        if denom > 0:
            headroom = (system - floor) / denom
        elif denom == 0:
            headroom = 1.0 if system >= ceiling else 0.0
        else:  # floor > ceiling: data anomaly
            suppressed_reason = "floor exceeds ceiling (data anomaly)"
            headroom = None

    return ReferencePoints(
        ceiling=ceiling,
        system=system,
        floor=floor,
        total=total,
        reference_n=n_reference,
        headroom_realised=headroom,
        retrieval_regressions=regressions,
        suppressed_reason=suppressed_reason,
    )


# ── D. Failure attribution — the 2x2 ───────────────────────────────────────


@dataclass(frozen=True)
class AttributionMatrix:
    """{correct, partial/incorrect} x {source_found, source_missed} + lucky cell.

    The lucky cell (correct + source missed) is a dataset-quality signal: a
    question that lands there repeatedly across models is answered from
    parametric memory, not retrieval, and should be requalified or retired.
    """

    correct_source_found: int = 0
    correct_source_missed: int = 0  # 🍀 lucky
    failed_source_found: int = 0  # ❌ answer-layer failure
    failed_source_missed: int = 0  # ❌ retrieval failure


def attribution_matrix(results: Sequence[ScoredQuestion]) -> AttributionMatrix:
    """Assign each judged, source-eligible question to a 2x2 cell.

    Unjudged/pending questions (no verdict) and out_of_corpus questions (no gold
    source) are excluded — they are captured by ``unjudged`` /
    ``hallucination_rate`` instead.
    """
    m = AttributionMatrix()
    for r in results:
        if not r.verdict.is_judged:
            continue
        if not _required_sources(r.question.sources):
            continue  # out_of_corpus: no source axis
        found = source_hit_rank(r.retrieved_paths, r.question.sources) is not None
        if r.verdict == Verdict.CORRECT:
            if found:
                m = _replace_cell(m, correct_source_found=m.correct_source_found + 1)
            else:
                m = _replace_cell(m, correct_source_missed=m.correct_source_missed + 1)
        elif found:
            m = _replace_cell(m, failed_source_found=m.failed_source_found + 1)
        else:
            m = _replace_cell(m, failed_source_missed=m.failed_source_missed + 1)
    return m


def _replace_cell(m: AttributionMatrix, **changes: int) -> AttributionMatrix:
    return replace(m, **changes)


@dataclass(frozen=True)
class CapabilityBreakdown:
    """Per-capability source + answer metrics + attribution + counts."""

    capability: str
    n: int
    source: SourceMetrics = field(default_factory=_empty_source_metrics)
    answer: AnswerMetrics = field(default_factory=_empty_answer_metrics)
    attribution: AttributionMatrix = field(default_factory=AttributionMatrix)


def attribution_by_capability(
    results: Sequence[ScoredQuestion],
) -> dict[str, CapabilityBreakdown]:
    """Group results by ``question.capability`` and reduce each group.

    Per-capability attribution answers "which capability broke" independently of
    "how hard was what broke" (the difficulty axis).
    """
    groups: dict[str, list[ScoredQuestion]] = {}
    for r in results:
        groups.setdefault(r.question.capability, []).append(r)
    out: dict[str, CapabilityBreakdown] = {}
    for cap, group in groups.items():
        out[cap] = CapabilityBreakdown(
            capability=cap,
            n=len(group),
            source=aggregate_source_metrics(group),
            answer=aggregate_answer_metrics(group),
            attribution=attribution_matrix(group),
        )
    return out


# ── E. Token usage (per-question LLM cost) ─────────────────────────────────


@dataclass(frozen=True)
class TokenUsage:
    """Set-level token usage for one cost category (answer or judge).

    ``total`` / ``p50`` are the headline numbers. ``p50`` is the median
    per-question total (input + output); ``p50_input`` / ``p50_output`` are the
    per-direction medians. All stay 0 for systems that make no LLM calls
    (``retrieval_only``) or when the endpoint does not report usage.
    """

    n: int
    total_input: int
    total_output: int
    total: int
    p50: int
    p50_input: int
    p50_output: int


def _empty_token_usage() -> TokenUsage:
    return TokenUsage(0, 0, 0, 0, 0, 0, 0)


def _median(sorted_vals: list[int]) -> int:
    """Integer median of a non-empty sorted list."""
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) // 2


def aggregate_token_usage(
    results: Sequence[ScoredQuestion],
    *,
    input_attr: str,
    output_attr: str,
) -> TokenUsage:
    """Reduce per-question token counts into set-level usage.

    ``input_attr`` / ``output_attr`` name the :class:`ScoredQuestion` fields to
    sum (``answer_input_tokens`` / ``judge_input_tokens`` etc.). Returns zeros
    when the set is empty or every question is 0 (no-LLM systems, endpoints
    that do not report usage).
    """
    if not results:
        return _empty_token_usage()
    pairs = [(int(getattr(r, input_attr, 0)), int(getattr(r, output_attr, 0))) for r in results]
    ins = sorted(a for a, _ in pairs)
    outs = sorted(b for _, b in pairs)
    totals = sorted(a + b for a, b in pairs)
    return TokenUsage(
        n=len(results),
        total_input=sum(ins),
        total_output=sum(outs),
        total=sum(ins) + sum(outs),
        p50=_median(totals),
        p50_input=_median(ins),
        p50_output=_median(outs),
    )


def aggregate_peak_context(traces: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Measured peak single-request input-token distribution.

    Reduces the per-question trace's request-accounting fields
    (``peak_input_tokens`` / ``requests`` / ``overflow_fallbacks``, written by
    the agent runner's meter) into the distribution a context window actually
    constrains: percentiles of the largest single prefilled request per
    question, plus how many questions would overflow the 8k/16k windows and
    how often the context-overflow fallback fired. Replaces
    chars-per-token estimates with endpoint-measured numbers. ``n`` is 0 for
    runs whose traces predate the meter (or systems that never call the
    model) — the honest-empty convention of the other aggregates.
    """
    peaks: list[int] = []
    requests_total = 0
    overflow_fallbacks = 0
    for t in traces:
        peak = t.get("peak_input_tokens")
        if isinstance(peak, int) and peak >= 0:
            peaks.append(peak)
        reqs = t.get("requests")
        if isinstance(reqs, int) and reqs >= 0:
            requests_total += reqs
        fallbacks = t.get("overflow_fallbacks")
        if isinstance(fallbacks, int) and fallbacks >= 0:
            overflow_fallbacks += fallbacks
    if not peaks:
        return {"n": 0}
    peaks.sort()
    return {
        "n": len(peaks),
        "p50": _pct(peaks, 50),
        "p90": _pct(peaks, 90),
        "p95": _pct(peaks, 95),
        "max": peaks[-1],
        "over_8192": sum(1 for p in peaks if p > 8192),
        "over_16384": sum(1 for p in peaks if p > 16384),
        "requests_total": requests_total,
        "overflow_fallbacks": overflow_fallbacks,
    }


# ── F. Latency (per-question + per-stage, no LLM) ───────────────────────────


def _pct(sorted_vals: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile of a pre-sorted non-empty sequence."""
    if not sorted_vals:
        return 0.0
    idx = max(0, min(len(sorted_vals) - 1, round(pct / 100.0 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def aggregate_latency(results: Sequence[ScoredQuestion]) -> dict[str, object]:
    """Per-question pipeline latency percentiles (ms) from ``latency_ms``.

    Zeros when nothing reported latency — the same honest-empty convention as
    the token aggregates.
    """
    vals = sorted(r.latency_ms for r in results if r.latency_ms > 0)
    if not vals:
        return {"n": 0, "p50": 0.0, "p95": 0.0, "mean": 0.0}
    return {
        "n": len(vals),
        "p50": _pct(vals, 50),
        "p95": _pct(vals, 95),
        "mean": sum(vals) / len(vals),
    }


def stage_latency_breakdown(
    traces: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Mean/p95 duration per pipeline stage across a run's traces.

    Reads the always-on ``Trace.to_dict()`` shape (``stages[].name`` /
    ``duration_ms``). Stage records — not per-question sums — are the unit: a
    profile with N concurrent candidate sources logs N ``candidate_source``
    records per question, and their wall-clock overlap means a per-question sum
    would overstate latency. ``count`` is the number of stage records averaged.
    """
    per: dict[str, list[float]] = {}
    for t in traces:
        stages = t.get("stages")
        if not isinstance(stages, Sequence):
            continue
        for st in stages:
            if not isinstance(st, Mapping):
                continue
            name = st.get("name")
            dur = st.get("duration_ms")
            if not isinstance(name, str) or not isinstance(dur, int | float):
                continue
            per.setdefault(name, []).append(float(dur))
    return {
        name: {
            "count": len(durs),
            "mean_ms": sum(durs) / len(durs),
            "p95_ms": _pct(sorted(durs), 95),
        }
        for name, durs in sorted(per.items())
    }
