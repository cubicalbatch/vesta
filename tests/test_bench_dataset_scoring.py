"""Tests for the unified benchmark dataset + scoring.

Covers: loader round-trip + validation errors naming the slug, hash stability,
hash insensitivity to oracle/closed_book/provenance/tags edits, filter
correctness, every source metric, out_of_corpus exclusion from source
denominators, unjudged-never-correct + run-incomplete, three-reference headroom
math, attribution 2x2 cell assignment, and judge-derived sub_fact_coverage.

No live LLM: a fake JudgeLLM drives the structured-rubric path deterministically.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from vesta.eval import bench_scoring
from vesta.eval.bench_dataset import (
    BenchQuestion,
    BenchSource,
    SubFact,
    dataset_hash,
    filter,
    load_bench_dataset,
    subset_hash,
)
from vesta.eval.bench_scoring import (
    RUBRIC_PROMPT_VERSION,
    JudgeOutcome,
    ScoredQuestion,
    SourceMetrics,
    Verdict,
    _parse_judge_json,
    _unjudged,
    aggregate_answer_metrics,
    aggregate_peak_context,
    aggregate_source_metrics,
    aggregate_token_usage,
    attribution_by_capability,
    attribution_matrix,
    judge_cache_key,
    judge_verdict,
    measure_bench_calibration,
    reference_points,
    retrieved_precision,
    rubric_prompt_hash,
    score_question,
    source_coverage,
    source_hit_rank,
    source_mrr,
    source_recall_at,
)

BENCH = Path(__file__).resolve().parent.parent / "benchmarks"


# ── Fixture builders ────────────────────────────────────────────────────────


def _src(path: str, *, required: bool = True, title: str = "T", zim: str = "z") -> BenchSource:
    return BenchSource(zim=zim, article_title=title, article_path=path, required=required)


def _q(
    qid: str,
    *,
    sources: tuple[BenchSource, ...] = (),
    expected_behavior: str = "answer",
    capability: str = "buried_fact",
    difficulty: str = "medium",
    slice: str = "core",
    sub_facts: tuple[SubFact, ...] = (),
    answer: str = "A",
    question: str | None = None,
    answer_detail: str = "",
    oracle: Mapping[str, object] | None = None,
    closed_book: Mapping[str, object] | None = None,
    tags: tuple[str, ...] = (),
    level: int = 3,
    status: str = "active",
) -> BenchQuestion:
    from types import MappingProxyType

    return BenchQuestion(
        id=qid,
        question=question or f"{qid}?",
        capability=capability,
        difficulty=difficulty,
        slice=slice,
        expected_behavior=expected_behavior,
        answer=answer,
        answer_detail=answer_detail,
        sources=sources,
        sub_facts=sub_facts,
        tags=tags,
        level=level,
        oracle=MappingProxyType(dict(oracle)) if oracle else MappingProxyType({}),
        closed_book=MappingProxyType(dict(closed_book)) if closed_book else MappingProxyType({}),
        status=status,
    )


def _scored(
    q: BenchQuestion,
    *,
    retrieved: tuple[str, ...] = (),
    verdict: Verdict = Verdict.CORRECT,
    abstained: bool = False,
    sub_facts_present: tuple[bool, ...] = (),
) -> ScoredQuestion:
    return ScoredQuestion(
        question=q,
        retrieved_paths=retrieved,
        verdict=verdict,
        abstained=abstained,
        sub_facts_present=sub_facts_present,
        judge_model="fake-judge",
    )


def _dataset_json(tmp_path: Path, questions: list[dict[str, object]]) -> Path:
    p = tmp_path / "ds.json"
    p.write_text(
        json.dumps(
            {
                "name": "test_bench",
                "version": 1,
                "generated": "2026-08-07",
                "archives": [{"zim": "z"}],
                "questions": questions,
            }
        )
    )
    return p


def _min_question(qid: str = "wiki-0001", **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": qid,
        "question": f"{qid}?",
        "capability": "buried_fact",
        "difficulty": "medium",
        "slice": "core",
        "expected_behavior": "answer",
        "answer": "42",
        "answer_detail": "",
        "sources": [
            {"zim": "z", "article_title": "T", "article_path": "Article", "required": True}
        ],
        "status": "active",
    }
    base.update(overrides)
    return base


# ── Loader round-trip + validation ──────────────────────────────────────────


def test_loader_round_trip(tmp_path: Path) -> None:
    p = _dataset_json(
        tmp_path,
        [
            _min_question("wiki-0001"),
            _min_question("wiki-0002", answer="other"),
        ],
    )
    ds = load_bench_dataset(str(p))
    assert ds.name == "test_bench"
    assert ds.version == 1
    assert len(ds) == 2
    assert ds.hash != ""
    assert ds.questions[0].sources[0].article_path == "Article"


def test_loader_tolerant_of_extra_fields(tmp_path: Path) -> None:
    p = _dataset_json(tmp_path, [_min_question(extra_audit="keep me")])  # type: ignore[arg-type]
    ds = load_bench_dataset(str(p))
    assert len(ds) == 1  # extra field tolerated, not rejected


def test_loader_validation_names_the_offending_slug(tmp_path: Path) -> None:
    bad = _min_question("wiki-0042")
    del bad["answer"]  # required field removed
    p = _dataset_json(tmp_path, [bad])
    with pytest.raises(ValueError, match="wiki-0042"):
        load_bench_dataset(str(p))


def test_loader_validation_missing_id(tmp_path: Path) -> None:
    p = _dataset_json(tmp_path, [_min_question()])
    # Remove the id at the JSON level.
    raw = json.loads(p.read_text())
    del raw["questions"][0]["id"]
    p.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="id"):
        load_bench_dataset(str(p))


def test_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    p = _dataset_json(tmp_path, [_min_question("wiki-0001"), _min_question("wiki-0001")])
    with pytest.raises(ValueError, match="duplicate"):
        load_bench_dataset(str(p))


def test_loader_rejects_answer_question_with_no_sources(tmp_path: Path) -> None:
    bad = _min_question("wiki-0009")
    bad["sources"] = []
    p = _dataset_json(tmp_path, [bad])
    with pytest.raises(ValueError, match="wiki-0009"):
        load_bench_dataset(str(p))


def test_loader_allows_abstain_question_with_no_sources(tmp_path: Path) -> None:
    """out_of_corpus questions have no gold source — abstain + empty sources is valid."""
    q = _min_question("ooc-0001")
    q["expected_behavior"] = "abstain"
    q["sources"] = []
    p = _dataset_json(tmp_path, [q])
    ds = load_bench_dataset(str(p))
    assert ds.questions[0].expected_behavior == "abstain"
    assert ds.questions[0].sources == ()


def test_loads_shipped_dataset() -> None:
    """The shipped vesta_bench_v2.json loads cleanly (200 Wikipedia-only questions)."""
    p = BENCH / "vesta_bench_v2.json"
    if not p.exists():
        pytest.skip("vesta_bench_v2.json not built yet")
    ds = load_bench_dataset(str(p))
    assert len(ds) == 200
    assert ds.hash != ""


# ── Hash stability + insensitivity ──────────────────────────────────────────


def test_hash_is_stable_for_same_content(tmp_path: Path) -> None:
    p = _dataset_json(tmp_path, [_min_question("wiki-0001"), _min_question("wiki-0002")])
    assert load_bench_dataset(str(p)).hash == load_bench_dataset(str(p)).hash


def test_hash_changes_on_question_edit(tmp_path: Path) -> None:
    p = _dataset_json(tmp_path, [_min_question("wiki-0001")])
    h1 = load_bench_dataset(str(p)).hash
    raw = json.loads(p.read_text())
    raw["questions"][0]["answer"] = "different"
    p.write_text(json.dumps(raw))
    h2 = load_bench_dataset(str(p)).hash
    assert h1 != h2


def test_hash_independent_of_question_order() -> None:
    """Hash is over id order, so question array order must not matter."""
    a = _q("b")
    b = _q("a")
    assert dataset_hash([a, b]) == dataset_hash([b, a])


def test_hash_insensitive_to_oracle_closed_book_provenance_tags(tmp_path: Path) -> None:
    """Re-verifying the oracle / editing provenance / tags must NOT change the
    dataset identity."""
    p = _dataset_json(tmp_path, [_min_question("wiki-0001")])
    h1 = load_bench_dataset(str(p)).hash
    raw = json.loads(p.read_text())
    raw["questions"][0].update(
        {
            "oracle": {"model": "m", "verdict": "correct"},
            "closed_book": {"model": "m", "verdict": "incorrect"},
            "provenance": {"reviewed_by": "x"},
            "tags": ["physics"],
        }
    )
    p.write_text(json.dumps(raw))
    h2 = load_bench_dataset(str(p)).hash
    assert h1 == h2


def test_hash_sensitive_to_sources_and_sub_facts(tmp_path: Path) -> None:
    p = _dataset_json(tmp_path, [_min_question("wiki-0001")])
    h1 = load_bench_dataset(str(p)).hash
    raw = json.loads(p.read_text())
    raw["questions"][0]["sources"][0]["article_path"] = "DifferentArticle"
    p.write_text(json.dumps(raw))
    assert load_bench_dataset(str(p)).hash != h1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("capability", "lookup"),
        ("difficulty", "hard"),
        ("level", 2),
        ("status", "quarantined"),
        ("answer_detail", "extra detail"),
    ],
)
def test_hash_sensitive_to_measurement_fields(field: str, value: object) -> None:
    """Every field that can change a measured score must change the dataset
    identity: capability/difficulty/level/status gate filtering and
    attribution, answer_detail is embedded in every judge prompt."""
    base = _q("a")
    assert dataset_hash([base]) != dataset_hash([replace(base, **{field: value})])


def test_hash_sensitive_to_source_required_flag() -> None:
    """``required:false`` flips the SourceMetrics denominators and the
    attribution 2x2 — two datasets differing only there must not compare equal
    under the B7 comparability guard."""
    a = _q("a", sources=(_src("P"),))
    b = _q("a", sources=(_src("P", required=False),))
    assert dataset_hash([a]) != dataset_hash([b])


def test_hash_insensitive_to_source_and_sub_fact_order() -> None:
    """Source / sub_fact order is presentation, not identity."""
    a = _q(
        "a",
        sources=(_src("P1"), _src("P2")),
        sub_facts=(SubFact(fact="f1"), SubFact(fact="f2")),
    )
    b = _q(
        "a",
        sources=(_src("P2"), _src("P1")),
        sub_facts=(SubFact(fact="f2"), SubFact(fact="f1")),
    )
    assert dataset_hash([a]) == dataset_hash([b])


def test_hash_stable_across_loads(tmp_path: Path) -> None:
    """Same content → same hash on every load (dict ordering can never leak in)."""
    p = _dataset_json(
        tmp_path,
        [
            _min_question("wiki-0001"),
            _min_question("wiki-0002", level=2),
            _min_question("wiki-0003"),
        ],
    )
    hashes = {load_bench_dataset(str(p)).hash for _ in range(3)}
    assert len(hashes) == 1


def test_subset_hash_differs_from_full_hash() -> None:
    qs = [_q("a"), _q("b"), _q("c")]
    full = dataset_hash(qs)
    sub = subset_hash(qs[:1])
    assert full != sub
    assert len(full) == 16 and len(sub) == 16


# ── Filtering ───────────────────────────────────────────────────────────────


def test_filter_by_slice_capability_difficulty() -> None:
    qs = [
        _q("a", slice="core", capability="buried_fact", difficulty="easy"),
        _q("b", slice="cross", capability="procedural", difficulty="hard"),
        _q("c", slice="core", capability="lookup", difficulty="medium"),
    ]
    assert [x.id for x in filter(qs, slice="core")] == ["a", "c"]
    assert [x.id for x in filter(qs, capabilities=("procedural",))] == ["b"]
    assert [x.id for x in filter(qs, difficulties=("easy", "medium"))] == ["a", "c"]
    assert [x.id for x in filter(qs, slice="core", capabilities=("lookup",))] == ["c"]
    # No filters ⇒ everything, in original order.
    assert [x.id for x in filter(qs)] == ["a", "b", "c"]


def test_filter_by_level_is_cumulative() -> None:
    """level=L keeps q.level <= L — a higher tier is a strict superset."""
    qs = [
        _q("a", level=1),
        _q("b", level=2),
        _q("c", level=3),
    ]
    assert [x.id for x in filter(qs, level=1)] == ["a"]
    assert [x.id for x in filter(qs, level=2)] == ["a", "b"]
    assert [x.id for x in filter(qs, level=3)] == ["a", "b", "c"]
    assert [x.id for x in filter(qs)] == ["a", "b", "c"]  # no level filter ⇒ all


def test_loader_parses_level_and_defaults_to_3(tmp_path: Path) -> None:
    p = _dataset_json(
        tmp_path,
        [
            _min_question("wiki-0001", level=1),
            _min_question("wiki-0002"),  # absent ⇒ deepest tier
        ],
    )
    ds = load_bench_dataset(str(p))
    assert [q.level for q in ds.questions] == [1, 3]


def test_loader_rejects_invalid_level(tmp_path: Path) -> None:
    p = _dataset_json(tmp_path, [_min_question("wiki-0007", level=4)])
    with pytest.raises(ValueError, match=r"wiki-0007.*level"):
        load_bench_dataset(str(p))


def test_level_is_hash_sensitive(tmp_path: Path) -> None:
    """Re-tiering changes which questions a run executes, so it must change
    the dataset identity (AUDIT_0824 N34)."""
    p = _dataset_json(tmp_path, [_min_question("wiki-0001", level=1)])
    h1 = load_bench_dataset(str(p)).hash
    raw = json.loads(p.read_text())
    raw["questions"][0]["level"] = 2
    p.write_text(json.dumps(raw))
    h2 = load_bench_dataset(str(p)).hash
    assert h1 != h2


def test_shipped_dataset_level_stratification() -> None:
    """The shipped 200-question set: cumulative 50/100/200, all 8 capabilities
    present at every tier (even the smoke tier tests the full surface)."""
    p = BENCH / "vesta_bench_v2.json"
    if not p.exists():
        pytest.skip("vesta_bench_v2.json not built yet")
    ds = load_bench_dataset(str(p))
    for lvl, expected_n in ((1, 50), (2, 100), (3, 200)):
        sub = filter(ds.questions, level=lvl)
        assert len(sub) == expected_n
        assert len({q.capability for q in sub}) == 8


# ── Source metrics (deterministic, no LLM) ──────────────────────────────────


def test_source_hit_rank() -> None:
    srcs = (_src("Alpha"),)
    assert source_hit_rank(["Alpha"], srcs) == 1
    assert source_hit_rank(["Beta", "Alpha"], srcs) == 2
    assert source_hit_rank(["Beta", "Gamma"], srcs) is None  # miss


def test_source_hit_rank_only_required_sources_count() -> None:
    srcs = (_src("Alpha", required=False), _src("Beta"))
    # A non-required 'alternative' source does not count as a hit.
    assert source_hit_rank(["Alpha"], srcs) is None
    assert source_hit_rank(["Alpha", "Beta"], srcs) == 2


def test_source_recall_at() -> None:
    srcs = (_src("Alpha"),)
    assert source_recall_at(["Alpha"], srcs, 1) is True
    assert source_recall_at(["Beta", "Alpha"], srcs, 1) is False
    assert source_recall_at(["Beta", "Alpha"], srcs, 5) is True


def test_source_coverage_multi_hop() -> None:
    """coverage = fraction of required sources found (the multi-hop metric)."""
    srcs = (_src("Alpha"), _src("Beta"))
    assert source_coverage(["Alpha"], srcs) == 0.5  # one of two
    assert source_coverage(["Alpha", "Beta"], srcs) == 1.0
    assert source_coverage(["Gamma"], srcs) == 0.0


def test_source_coverage_no_required_sources_is_zero() -> None:
    assert source_coverage(["X"], (_src("A", required=False),)) == 0.0


def test_source_mrr() -> None:
    srcs = (_src("Alpha"),)
    assert source_mrr(["Alpha"], srcs) == 1.0
    assert source_mrr(["Beta", "Alpha"], srcs) == 0.5
    assert source_mrr(["Gamma"], srcs) == 0.0  # miss


def test_retrieved_precision() -> None:
    srcs = (_src("Alpha"), _src("Beta"))
    assert retrieved_precision(["Alpha", "Noise"], srcs) == pytest.approx(0.5)
    assert retrieved_precision(["Alpha", "Beta"], srcs) == 1.0
    assert retrieved_precision([], srcs) == 0.0


def test_aggregate_source_metrics() -> None:
    qs = [
        _q("a", sources=(_src("A"),)),
        _q("b", sources=(_src("B"),)),
    ]
    results = [
        _scored(qs[0], retrieved=("A",)),  # rank 1
        _scored(qs[1], retrieved=("Noise", "B")),  # rank 2
    ]
    m = aggregate_source_metrics(results)
    assert isinstance(m, SourceMetrics)
    assert m.n == 2
    assert m.recall_at_1 == 0.5  # only q0 in top-1
    assert m.recall_at_5 == 1.0
    assert m.mrr == pytest.approx(0.75)  # (1.0 + 0.5) / 2
    assert m.mean_coverage == 1.0
    assert m.mean_precision == pytest.approx(0.75)  # (1.0 + 0.5) / 2


def test_out_of_corpus_excluded_from_source_denominators() -> None:
    """Trap 7: out_of_corpus (no sources) must not drag source metrics down."""
    answer_q = _q("a", sources=(_src("A"),))
    ooc_q = _q("ooc", sources=(), expected_behavior="abstain")
    results = [
        _scored(answer_q, retrieved=("A",)),  # hit
        _scored(ooc_q, retrieved=("Whatever",)),  # no gold source
    ]
    m = aggregate_source_metrics(results)
    assert m.n == 1  # only the answerable question counts
    assert m.recall_at_1 == 1.0


def test_aggregate_source_metrics_empty_is_zero() -> None:
    assert aggregate_source_metrics([]).n == 0
    assert aggregate_source_metrics([]).recall_at_1 == 0.0


# ── Judge (structured JSON; no lexical path) ────────────────────────────────


class _FakeJudge:
    """A stub JudgeLLM that returns canned responses (structured JSON)."""

    def __init__(self, responses: list[str] | str) -> None:
        self._responses = list(responses) if isinstance(responses, list) else [responses]
        self.calls: list[str] = []

    async def judge(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._responses.pop(0) if self._responses else ""


def test_rubric_prompt_version_and_hash() -> None:
    assert RUBRIC_PROMPT_VERSION == "16.1"
    h = rubric_prompt_hash()
    assert len(h) == 16
    assert rubric_prompt_hash() == h  # stable


@pytest.mark.asyncio
async def test_judge_verdict_parses_strict_json() -> None:
    q = _q("a", sub_facts=(SubFact("1879"), SubFact("1905")))
    judge = _FakeJudge(
        '{"verdict": "correct", "reason": "matches", '
        '"sub_facts_present": [true, false], "abstained": false}'
    )
    out = await judge_verdict(
        question=q, model_answer="Born 1879.", abstained=False, judge=judge, judge_model="m"
    )
    assert out.verdict == Verdict.CORRECT
    assert out.reason == "matches"
    assert out.sub_facts_present == (True, False)
    assert out.abstained is False
    # The judge is blind to retrieved context (trap 5) + forbids parametric
    # knowledge (trap 3): assert both clauses appear in the rendered prompt.
    assert "Do NOT use your own parametric knowledge" in judge.calls[0]
    assert "given NO retrieved context" in judge.calls[0]


@pytest.mark.asyncio
async def test_judge_verdict_parses_fenced_json() -> None:
    q = _q("a")
    judge = _FakeJudge('```json\n{"verdict": "partial", "reason": "r"}\n```')
    out = await judge_verdict(
        question=q, model_answer="x", abstained=False, judge=judge, judge_model="m"
    )
    assert out.verdict == Verdict.PARTIAL


@pytest.mark.asyncio
async def test_judge_verdict_retries_once_then_unjudged() -> None:
    q = _q("a")
    judge = _FakeJudge(["garbage", "still garbage"])
    out = await judge_verdict(
        question=q, model_answer="x", abstained=False, judge=judge, judge_model="m", retries=1
    )
    assert out.verdict == Verdict.UNJUDGED
    assert len(judge.calls) == 2  # one attempt + one retry


@pytest.mark.asyncio
async def test_judge_verdict_recovers_on_retry() -> None:
    q = _q("a")
    judge = _FakeJudge(["garbage", '{"verdict": "correct", "reason": "r"}'])
    out = await judge_verdict(
        question=q, model_answer="x", abstained=False, judge=judge, judge_model="m", retries=1
    )
    assert out.verdict == Verdict.CORRECT
    assert len(judge.calls) == 2


@pytest.mark.asyncio
async def test_judge_verdict_exception_is_unjudged() -> None:
    class _Boom:
        async def judge(self, prompt: str) -> str:
            raise RuntimeError("endpoint down")

    q = _q("a")
    out = await judge_verdict(
        question=q, model_answer="x", abstained=False, judge=_Boom(), judge_model="m"
    )
    assert out.verdict == Verdict.UNJUDGED


@pytest.mark.asyncio
async def test_judge_verdict_no_judge_is_unjudged() -> None:
    q = _q("a")
    out = await judge_verdict(
        question=q, model_answer="x", abstained=False, judge=None, judge_model="m"
    )
    assert out.verdict == Verdict.UNJUDGED


class _ParamJudge(_FakeJudge):
    """A fake judge that carries the sampling/endpoint identity the real
    ``GatewayJudgeLLM`` exposes (read duck-typed by the cache key)."""

    def __init__(
        self,
        responses: list[str] | str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        endpoint: str = "",
    ) -> None:
        super().__init__(responses)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.endpoint = endpoint


def test_judge_cache_key_is_sampling_param_sensitive() -> None:
    """AUDIT_0824 N33: verdicts minted at one temperature / token budget /
    endpoint must never be served for another."""
    rubric = "rubric"
    base = judge_cache_key(rubric, "q1", "ans", "judge-b")
    assert base != judge_cache_key(rubric, "q1", "ans", "judge-b", temperature=0.0)
    assert base != judge_cache_key(rubric, "q1", "ans", "judge-b", temperature=0.0, max_tokens=4096)
    assert judge_cache_key(rubric, "q1", "ans", "judge-b", temperature=0.0) != judge_cache_key(
        rubric, "q1", "ans", "judge-b", temperature=1.5
    )
    assert judge_cache_key(
        rubric, "q1", "ans", "judge-b", temperature=0.0, max_tokens=4096
    ) != judge_cache_key(rubric, "q1", "ans", "judge-b", temperature=0.0, max_tokens=2048)
    assert judge_cache_key(rubric, "q1", "ans", "judge-b", temperature=0.0) != judge_cache_key(
        rubric, "q1", "ans", "judge-b", temperature=0.0, endpoint="http://127.0.0.1:8080/v1"
    )
    # Identical params ⇒ identical key (cache hit).
    assert judge_cache_key(
        rubric, "q1", "ans", "judge-b", temperature=0.7, max_tokens=512, endpoint="ep"
    ) == judge_cache_key(
        rubric, "q1", "ans", "judge-b", temperature=0.7, max_tokens=512, endpoint="ep"
    )


@pytest.mark.asyncio
async def test_judge_verdict_cache_rejudges_on_temperature_change() -> None:
    """Same rubric/qid/answer/model but a different judge temperature ⇒ cache
    miss: the second call re-judges instead of serving the stale verdict."""
    q = _q("a")
    cache: dict[str, JudgeOutcome] = {}

    async def cache_get(key: str) -> JudgeOutcome | None:
        return cache.get(key)

    async def cache_put(key: str, outcome: JudgeOutcome) -> None:
        cache[key] = outcome

    hot = _ParamJudge('{"verdict": "correct", "reason": "hot"}', temperature=1.5)
    out_hot = await judge_verdict(
        question=q,
        model_answer="x",
        abstained=False,
        judge=hot,
        judge_model="m",
        cache_get=cache_get,
        cache_put=cache_put,
    )
    assert out_hot.verdict == Verdict.CORRECT
    assert len(hot.calls) == 1  # judged live, then cached

    cold = _ParamJudge('{"verdict": "partial", "reason": "cold"}', temperature=0.0)
    out_cold = await judge_verdict(
        question=q,
        model_answer="x",
        abstained=False,
        judge=cold,
        judge_model="m",
        cache_get=cache_get,
        cache_put=cache_put,
    )
    assert out_cold.verdict == Verdict.PARTIAL
    assert len(cold.calls) == 1  # temperature changed ⇒ re-judged, not served

    warm = _ParamJudge('{"verdict": "incorrect", "reason": "never called"}', temperature=1.5)
    out_warm = await judge_verdict(
        question=q,
        model_answer="x",
        abstained=False,
        judge=warm,
        judge_model="m",
        cache_get=cache_get,
        cache_put=cache_put,
    )
    assert out_warm.verdict == Verdict.CORRECT
    assert len(warm.calls) == 0  # identical params back ⇒ cache hit


# ── Answer metrics ──────────────────────────────────────────────────────────


def test_answer_metrics_basic() -> None:
    qs = [_q("a"), _q("b"), _q("c")]
    results = [
        _scored(qs[0], verdict=Verdict.CORRECT),
        _scored(qs[1], verdict=Verdict.PARTIAL),
        _scored(qs[2], verdict=Verdict.INCORRECT),
    ]
    m = aggregate_answer_metrics(results)
    assert m.n == 3
    assert m.strict_accuracy == pytest.approx(1 / 3)
    assert m.weighted_accuracy == pytest.approx((1 + 0.5) / 3)
    assert m.unjudged == 0
    assert m.complete is True


def test_unjudged_never_counted_correct_and_flags_incomplete() -> None:
    """Trap 16: deleting the lexical net means a failed judge marks the run
    incomplete; the unjudged question is never counted correct."""
    qs = [_q("a"), _q("b")]
    results = [
        _scored(qs[0], verdict=Verdict.CORRECT),
        _scored(qs[1], verdict=Verdict.UNJUDGED),
    ]
    m = aggregate_answer_metrics(results)
    assert m.unjudged == 1
    assert m.complete is False
    # strict_accuracy = correct / n; unjudged is in n but not in correct.
    assert m.strict_accuracy == pytest.approx(0.5)


def test_sub_fact_coverage_is_judge_derived_not_substring() -> None:
    """The judge reports sub_facts_present; coverage is its mean, NOT a substring
    check (the lexical sub_fact_coverage path is gone)."""
    q = _q("a", sub_facts=(SubFact("1879"), SubFact("1905")))
    # The answer text does NOT contain "1879" or "1905" as substrings, but the
    # judge (correctly) reports one present — coverage must follow the judge.
    results = [
        _scored(q, verdict=Verdict.CORRECT, sub_facts_present=(True, False)),
    ]
    m = aggregate_answer_metrics(results)
    assert m.sub_fact_coverage == 0.5


def test_over_refusal_and_hallucination_and_abstention() -> None:
    answer_q = _q("a", sources=(_src("A"),), expected_behavior="answer")
    ooc_q = _q("ooc", sources=(), expected_behavior="abstain")
    results = [
        _scored(answer_q, abstained=True),  # over-refusal: should have answered
        _scored(ooc_q, abstained=False),  # hallucination: should have abstained
    ]
    m = aggregate_answer_metrics(results)
    assert m.over_refusal == 1.0  # the 1 answerable question was refused
    assert m.hallucination_rate == 1.0  # the 1 ooc question was answered
    assert m.abstention_correctness == 0.0  # neither decision was correct


def test_correct_abstention_on_out_of_corpus() -> None:
    ooc = _q("ooc", sources=(), expected_behavior="abstain")
    m = aggregate_answer_metrics([_scored(ooc, abstained=True)])
    assert m.hallucination_rate == 0.0
    assert m.abstention_correctness == 1.0


def test_answer_metrics_empty() -> None:
    m = aggregate_answer_metrics([])
    assert m.n == 0
    assert m.complete is True


# ── Three reference points (ceiling / system / floor) ───────────────────────


def test_reference_points_headroom_math() -> None:
    qs = [
        _q(
            "a",
            oracle={"model": "m", "verdict": "correct"},
            closed_book={"model": "m", "verdict": "incorrect"},
        ),
        _q(
            "b",
            oracle={"model": "m", "verdict": "correct"},
            closed_book={"model": "m", "verdict": "incorrect"},
        ),
        _q(
            "c",
            oracle={"model": "m", "verdict": "correct"},
            closed_book={"model": "m", "verdict": "correct"},
        ),
        _q(
            "d",
            oracle={"model": "m", "verdict": "incorrect"},
            closed_book={"model": "m", "verdict": "incorrect"},
        ),
    ]
    # system gets a, b correct (2); c wrong with retrieval though closed-book right → regression.
    results = [
        _scored(qs[0], verdict=Verdict.CORRECT),
        _scored(qs[1], verdict=Verdict.CORRECT),
        _scored(qs[2], verdict=Verdict.INCORRECT),
        _scored(qs[3], verdict=Verdict.INCORRECT),
    ]
    rp = reference_points(results, answer_model="m")
    assert rp.ceiling == 3  # oracle correct: a, b, c
    assert rp.floor == 1  # closed_book correct: c
    assert rp.system == 2  # a, b
    assert rp.headroom_realised == pytest.approx((2 - 1) / (3 - 1))  # 0.5
    assert rp.retrieval_regressions == 1  # c: closed-book right, retrieval wrong


def test_reference_points_partial_oracle_coverage_stays_within_ceiling() -> None:
    """AUDIT_0824 M16: with oracle on only a subset of questions, ``system``
    must be counted over that same subset. Counting it over the full run let
    headroom_realised exceed 1 (here 5/3 ≈ 1.67 before the fix)."""
    qs = [
        _q(
            f"o{i}",
            oracle={"model": "m", "verdict": "correct"},
            closed_book={"model": "m", "verdict": "incorrect"},
        )
        for i in range(3)
    ] + [_q(f"u{i}") for i in range(3)]  # no oracle block
    results = [
        _scored(qs[0], verdict=Verdict.CORRECT),
        _scored(qs[1], verdict=Verdict.CORRECT),
        _scored(qs[2], verdict=Verdict.INCORRECT),
        # Unreferenced questions: system correct, but they must not inflate
        # system beyond the oracle ceiling.
        _scored(qs[3], verdict=Verdict.CORRECT),
        _scored(qs[4], verdict=Verdict.CORRECT),
        _scored(qs[5], verdict=Verdict.CORRECT),
    ]
    rp = reference_points(results, answer_model="m")
    assert rp.total == 6
    assert rp.reference_n == 3
    assert rp.ceiling == 3
    assert rp.system == 2  # only the referenced subset counts
    assert rp.floor == 0
    assert rp.headroom_realised is not None
    assert rp.headroom_realised <= 1.0
    assert rp.headroom_realised == pytest.approx(2 / 3)


def test_reference_points_no_oracle_zeroes_reference_counts() -> None:
    """With zero oracle coverage headroom stays suppressed; ceiling/system/
    floor/regressions are all taken over the empty reference subset."""
    qs = [_q("a"), _q("b")]
    results = [
        _scored(qs[0], verdict=Verdict.CORRECT),
        _scored(qs[1], verdict=Verdict.INCORRECT),
    ]
    rp = reference_points(results, answer_model="m")
    assert rp.total == 2
    assert rp.reference_n == 0
    assert rp.ceiling == 0
    assert rp.system == 0
    assert rp.floor == 0
    assert rp.retrieval_regressions == 0
    assert rp.headroom_realised is None
    assert "no oracle" in rp.suppressed_reason


def test_reference_points_suppressed_on_model_mismatch() -> None:
    """Trap 15: an oracle verified against a different model says nothing about
    this run's headroom — headroom_realised is suppressed."""
    q = _q("a", oracle={"model": "other-model", "verdict": "correct"})
    rp = reference_points([_scored(q, verdict=Verdict.CORRECT)], answer_model="m")
    assert rp.headroom_realised is None
    assert "!=" in rp.suppressed_reason


def test_reference_points_suppressed_when_no_oracle() -> None:
    q = _q("a")  # no oracle block
    rp = reference_points([_scored(q, verdict=Verdict.CORRECT)], answer_model="m")
    assert rp.headroom_realised is None
    assert "no oracle" in rp.suppressed_reason


# ── Failure attribution 2x2 ─────────────────────────────────────────────────


def test_attribution_matrix_cells() -> None:
    found_src = (_src("Gold"),)
    qs = {
        "a": _q("a", sources=found_src),
        "b": _q("b", sources=found_src),
        "c": _q("c", sources=found_src),
        "d": _q("d", sources=found_src),
    }
    results = [
        _scored(qs["a"], retrieved=("Gold",), verdict=Verdict.CORRECT),  # correct+found ✅
        _scored(qs["b"], retrieved=("Noise",), verdict=Verdict.CORRECT),  # correct+missed 🍀
        _scored(
            qs["c"], retrieved=("Gold",), verdict=Verdict.INCORRECT
        ),  # fail+found (answer layer)
        _scored(
            qs["d"], retrieved=("Noise",), verdict=Verdict.INCORRECT
        ),  # fail+missed (retrieval)
    ]
    m = attribution_matrix(results)
    assert m.correct_source_found == 1
    assert m.correct_source_missed == 1
    assert m.failed_source_found == 1
    assert m.failed_source_missed == 1


def test_attribution_excludes_unjudged_and_out_of_corpus() -> None:
    ooc = _q("ooc", sources=(), expected_behavior="abstain")
    ans = _q("a", sources=(_src("Gold"),))
    results = [
        _scored(ans, retrieved=("Gold",), verdict=Verdict.UNJUDGED),  # unjudged → excluded
        _scored(ooc, verdict=Verdict.CORRECT),  # ooc: no source axis → excluded
    ]
    m = attribution_matrix(results)
    assert m.correct_source_found == 0
    assert m.failed_source_missed == 0


def test_attribution_by_capability() -> None:
    a = _q("a", sources=(_src("G"),), capability="buried_fact")
    b = _q("b", sources=(_src("H"),), capability="lookup")
    results = [
        _scored(a, retrieved=("G",), verdict=Verdict.CORRECT),
        _scored(b, retrieved=("Noise",), verdict=Verdict.INCORRECT),
    ]
    by = attribution_by_capability(results)
    assert set(by) == {"buried_fact", "lookup"}
    assert by["buried_fact"].attribution.correct_source_found == 1
    assert by["lookup"].attribution.failed_source_missed == 1
    assert by["buried_fact"].n == 1


# ── score_question wiring ───────────────────────────────────────────────────


def test_score_question_bundles_fields() -> None:
    q = _q("a", sub_facts=(SubFact("x"),))
    out = JudgeOutcome(
        verdict=Verdict.PARTIAL,
        reason="r",
        sub_facts_present=(True,),
        abstained=False,
        judge_model="jm",
    )
    sq = score_question(q, ("Path1", "Path2"), out, abstained=True)
    assert sq.question is q
    assert sq.retrieved_paths == ("Path1", "Path2")
    assert sq.verdict == Verdict.PARTIAL
    assert sq.abstained is True  # harness decision, not the judge echo (False)
    assert sq.sub_facts_present == (True,)
    assert sq.judge_model == "jm"


def test_score_question_abstention_follows_harness_not_judge_echo() -> None:
    """AUDIT_0824 M17: a judge verdict that OMITS ``abstained`` must not turn an
    over-refusal into a clean answer — the harness decision decides."""
    q = _q("a", sources=(_src("A"),), expected_behavior="answer")
    outcome = _parse_judge_json('{"verdict": "correct", "reason": "r"}', "jm")
    assert outcome is not None
    assert outcome.abstained is False  # echo defaults false on omission
    sq = score_question(q, ("A",), outcome, abstained=True)
    assert sq.abstained is True
    m = aggregate_answer_metrics([sq])
    assert m.over_refusal == 1.0
    assert m.abstention_correctness == 0.0


def test_unjudged_outcome_does_not_inflate_hallucination_rate() -> None:
    """AUDIT_0824 M17: judge failure on a correctly-abstaining out-of-corpus
    question must not count as a hallucination."""
    ooc = _q("ooc", expected_behavior="abstain")
    outcome = _unjudged("jm", "judge endpoint down")
    assert outcome.abstained is False  # UNJUDGED outcomes carry no echo
    m = aggregate_answer_metrics([score_question(ooc, (), outcome, abstained=True)])
    assert m.hallucination_rate == 0.0
    assert m.abstention_correctness == 1.0


# ── Token usage aggregation ─────────────────────────────────────────────────


def _sq_with_tokens(qid: str, in_tok: int, out_tok: int) -> ScoredQuestion:
    return ScoredQuestion(
        question=_q(qid),
        retrieved_paths=("A",),
        verdict=Verdict.CORRECT,
        answer_input_tokens=in_tok,
        answer_output_tokens=out_tok,
    )


def test_token_usage_totals_and_p50() -> None:
    results = [
        _sq_with_tokens("q1", 100, 20),  # total 120
        _sq_with_tokens("q2", 200, 40),  # total 240
        _sq_with_tokens("q3", 300, 60),  # total 360
    ]
    usage = aggregate_token_usage(
        results, input_attr="answer_input_tokens", output_attr="answer_output_tokens"
    )
    assert usage.n == 3
    assert usage.total_input == 600
    assert usage.total_output == 120
    assert usage.total == 720
    # sorted totals: [120, 240, 360] → median = 240
    assert usage.p50 == 240
    assert usage.p50_input == 200
    assert usage.p50_output == 40


def test_token_usage_even_count_p50() -> None:
    """Even count: median is the average of the two middle values."""
    results = [
        _sq_with_tokens("q1", 100, 0),  # total 100
        _sq_with_tokens("q2", 300, 0),  # total 300
    ]
    usage = aggregate_token_usage(
        results, input_attr="answer_input_tokens", output_attr="answer_output_tokens"
    )
    # sorted: [100, 300] → median = (100 + 300) // 2 = 200
    assert usage.p50 == 200


def test_token_usage_p50_uses_per_question_totals_when_rankings_disagree() -> None:
    """p50 is the median of per-question input+output totals.

    When input and output rankings disagree, summing two independently
    sorted lists yields per-rank sums, not per-question totals
    (AUDIT_0822 M10): q1=(100,0), q2=(1,100), q3=(1,0) has true totals
    [100, 101, 1] → median 100; rank-wise sums would give [1, 1, 200]
    → median 1.
    """
    results = [
        _sq_with_tokens("q1", 100, 0),  # total 100
        _sq_with_tokens("q2", 1, 100),  # total 101
        _sq_with_tokens("q3", 1, 0),  # total 1
    ]
    usage = aggregate_token_usage(
        results, input_attr="answer_input_tokens", output_attr="answer_output_tokens"
    )
    assert usage.p50 == 100
    assert usage.p50_input == 1
    assert usage.p50_output == 0
    assert usage.total_input == 102
    assert usage.total_output == 100
    assert usage.total == 202


def test_token_usage_p50_unchanged_when_rankings_agree() -> None:
    """When both directions sort in the same order, per-question totals and
    rank-wise sums coincide — p50 stays the median question total."""
    results = [
        _sq_with_tokens("q1", 10, 1),  # total 11
        _sq_with_tokens("q2", 20, 2),  # total 22
        _sq_with_tokens("q3", 30, 3),  # total 33
    ]
    usage = aggregate_token_usage(
        results, input_attr="answer_input_tokens", output_attr="answer_output_tokens"
    )
    assert usage.p50 == 22


def test_token_usage_empty_returns_zeros() -> None:
    usage = aggregate_token_usage(
        [], input_attr="answer_input_tokens", output_attr="answer_output_tokens"
    )
    assert usage.total == 0
    assert usage.p50 == 0


def test_peak_context_distribution_and_overflow_counts() -> None:
    """The peak single-request distribution reduces traces the
    way the bench report renders them — percentiles of the per-question max,
    counts over the 8k/16k windows, request totals, overflow fallbacks."""
    traces = [
        {"peak_input_tokens": 4_044, "requests": 1, "overflow_fallbacks": 0},
        {"peak_input_tokens": 9_000, "requests": 3, "overflow_fallbacks": 0},
        {"peak_input_tokens": 17_000, "requests": 5, "overflow_fallbacks": 1},
        {"peak_input_tokens": 20_748, "requests": 7, "overflow_fallbacks": 2},
    ]
    out = aggregate_peak_context(traces)
    assert out["n"] == 4
    # sorted peaks [4044, 9000, 17000, 20748], nearest-rank: p50 → 17000,
    # p90/p95 → 20748.
    assert out["p50"] == 17_000
    assert out["p90"] == 20_748
    assert out["p95"] == 20_748
    assert out["max"] == 20_748
    assert out["over_8192"] == 3
    assert out["over_16384"] == 2
    assert out["requests_total"] == 16
    assert out["overflow_fallbacks"] == 3


def test_peak_context_empty_and_pre_meter_traces() -> None:
    """No traces, or traces that predate the meter, report the honest-empty
    n=0 shape (never a KeyError, never fabricated zeros)."""
    assert aggregate_peak_context([]) == {"n": 0}
    assert aggregate_peak_context([{"stages": []}, {}, {"peak_input_tokens": None}]) == {"n": 0}


def test_peak_context_ignores_non_integer_garbage() -> None:
    out = aggregate_peak_context(
        [
            {"peak_input_tokens": "big", "requests": "many"},
            {"peak_input_tokens": 5_000, "requests": 1, "overflow_fallbacks": 0},
        ]
    )
    assert out["n"] == 1
    assert out["max"] == 5_000
    assert out["requests_total"] == 1


def test_score_question_carries_answer_tokens() -> None:
    q = _q("a")
    out = JudgeOutcome(verdict=Verdict.CORRECT, reason="r")
    sq = score_question(
        q,
        ("A",),
        out,
        abstained=False,
        answer_input_tokens=150,
        answer_output_tokens=30,
    )
    assert sq.answer_input_tokens == 150
    assert sq.answer_output_tokens == 30


# ── Judge calibration (rho ≥ 0.7 marks a run trusted) ───────────────────────


def _cal_file(tmp_path: Path, items: list[dict[str, str]]) -> str:
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"name": "t", "items": items}), encoding="utf-8")
    return str(p)


@pytest.mark.asyncio
async def test_calibration_perfect_agreement_is_trusted(tmp_path: Path) -> None:
    """A judge that agrees with every hand verdict → rho = 1.0 (≥ 0.7 ⇒ trusted)."""
    items = [
        {"question": "Q1", "known_answer": "A1", "model_answer": "m1", "hand_verdict": "correct"},
        {"question": "Q2", "known_answer": "A2", "model_answer": "m2", "hand_verdict": "incorrect"},
        {"question": "Q3", "known_answer": "A3", "model_answer": "m3", "hand_verdict": "correct"},
    ]

    class _AgreeingJudge:
        async def judge(self, prompt: str) -> str:
            # Echo back the hand verdict embedded in the known-answer line.
            for it in items:
                if it["known_answer"] in prompt:
                    return json.dumps({"verdict": it["hand_verdict"], "reason": "r"})
            return json.dumps({"verdict": "incorrect", "reason": "?"})

    rho = await measure_bench_calibration(_AgreeingJudge(), "judge-b", _cal_file(tmp_path, items))
    assert rho is not None
    assert rho >= 0.7


@pytest.mark.asyncio
async def test_calibration_disagreement_is_untrusted(tmp_path: Path) -> None:
    """A judge that inverts every verdict → strongly negative rho (< 0.7 ⇒ untrusted)."""
    items = [
        {"question": "Q1", "known_answer": "A1", "model_answer": "m1", "hand_verdict": "correct"},
        {"question": "Q2", "known_answer": "A2", "model_answer": "m2", "hand_verdict": "correct"},
        {"question": "Q3", "known_answer": "A3", "model_answer": "m3", "hand_verdict": "incorrect"},
        {"question": "Q4", "known_answer": "A4", "model_answer": "m4", "hand_verdict": "incorrect"},
    ]
    invert = {"correct": "incorrect", "incorrect": "correct"}

    class _InvertingJudge:
        async def judge(self, prompt: str) -> str:
            for it in items:
                if it["known_answer"] in prompt:
                    return json.dumps({"verdict": invert[it["hand_verdict"]], "reason": "r"})
            return json.dumps({"verdict": "correct", "reason": "?"})

    rho = await measure_bench_calibration(_InvertingJudge(), "judge-b", _cal_file(tmp_path, items))
    assert rho is not None
    assert rho < 0.7


@pytest.mark.asyncio
async def test_calibration_none_without_judge_or_file(tmp_path: Path) -> None:
    """No judge / no file / too few items → None (unmeasured, not untrusted)."""
    assert await measure_bench_calibration(None, "m", _cal_file(tmp_path, [])) is None
    assert await measure_bench_calibration(_FakeJudge(["x"]), "", "") is None
    assert (
        await measure_bench_calibration(
            _FakeJudge(['{"verdict": "correct", "reason": "r"}']),
            "m",
            str(tmp_path / "missing.json"),
        )
        is None
    )


def _cal_items() -> list[dict[str, str]]:
    return [
        {"question": "Q1", "known_answer": "A1", "model_answer": "m1", "hand_verdict": "correct"},
        {
            "question": "Q2",
            "known_answer": "A2",
            "model_answer": "m2",
            "hand_verdict": "incorrect",
        },
    ]


class _ValidJudge:
    """Always returns a parseable (if crude) verdict."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def judge(self, prompt: str) -> str:
        self.calls.append(prompt)
        return json.dumps({"verdict": "correct", "reason": "r"})


@pytest.mark.asyncio
async def test_calibration_honors_item_expected_behavior(tmp_path: Path) -> None:
    """An ``abstain`` item is graded under the abstention rubric, not the answer one.

    Without per-item ``expected_behavior`` the shipped out-of-corpus items are
    judged under the answer direction ("abstention ⇒ incorrect"), which biases
    rho down and can mark a good judge untrusted.
    """
    items = [
        {
            "question": "Q1",
            "known_answer": "A1",
            "model_answer": "m1",
            "hand_verdict": "incorrect",
        },
        {
            "question": "Q2",
            "known_answer": "The correct behavior is to abstain.",
            "model_answer": "I could not find that in the archives.",
            "hand_verdict": "correct",
            "expected_behavior": "abstain",
        },
    ]

    class _RubricAwareJudge:
        async def judge(self, prompt: str) -> str:
            # Correct only when the prompt carries the out-of-corpus directive.
            verdict = "correct" if "OUT OF CORPUS" in prompt else "incorrect"
            return json.dumps({"verdict": verdict, "reason": "r"})

    rho = await measure_bench_calibration(_RubricAwareJudge(), "j", _cal_file(tmp_path, items))
    assert rho == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_calibration_relative_path_anchors_to_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The repo-relative default resolves no matter the process working directory."""
    root = tmp_path / "root"
    (root / "benchmarks").mkdir(parents=True)
    (root / "benchmarks" / "cal.json").write_text(
        json.dumps({"name": "t", "items": _cal_items()}), encoding="utf-8"
    )
    monkeypatch.setattr(bench_scoring, "_PROJECT_ROOT", root)
    monkeypatch.chdir(tmp_path)  # anywhere but the project root

    judge = _ValidJudge()
    rho = await measure_bench_calibration(judge, "m", "benchmarks/cal.json")
    assert rho is not None
    assert len(judge.calls) == 2


@pytest.mark.asyncio
async def test_calibration_relative_path_falls_back_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A custom relative path that lives only under the CWD still resolves."""
    cwd = tmp_path / "cwd"
    (cwd / "benchmarks").mkdir(parents=True)
    (cwd / "benchmarks" / "cal.json").write_text(
        json.dumps({"name": "t", "items": _cal_items()}), encoding="utf-8"
    )
    monkeypatch.setattr(bench_scoring, "_PROJECT_ROOT", tmp_path / "no-such-root")
    monkeypatch.chdir(cwd)

    judge = _ValidJudge()
    rho = await measure_bench_calibration(judge, "m", "benchmarks/cal.json")
    assert rho is not None
    assert len(judge.calls) == 2
