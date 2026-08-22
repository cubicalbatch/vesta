"""Eval harness tests.

Covers: golden-set loading + verification, the metrics (recall/nDCG/MRR/latency/
degradation guard), the runner (evaluate/compare/sweep/persist with fakes), the
regression gate (must FAIL a degraded profile — the load-bearing DoD item),
calibration, and the answer-metrics scaffold. Uses injected fakes so the suite
is fast and needs no DB or ZIM.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import pytest

from vesta.eval.calibrate import (
    ConfidenceSample,
    ConfidenceThresholds,
    escalates,
    fit_thresholds,
)
from vesta.eval.golden import (
    SLICES,
    GoldenEntry,
    GoldenSet,
    load_full_set,
    load_set,
    verify_against_archive,
)
from vesta.eval.metrics import (
    aggregate,
    degradations_from_traces,
    latency_from_traces,
    mrr_for,
    ndcg_at,
    path_matches,
    recall_at,
    run_metrics_from_dict,
)
from vesta.eval.regression import evaluate as gate_evaluate
from vesta.eval.runner import (
    EvalStore,
    PipelineRunner,
    QueryResult,
    RunRecord,
    compare,
    evaluate_profile,
    parse_sweep,
    persist_run,
)
from vesta.retrieval.profiles import load_profile

# ── Fakes (the eval seams are injectable by design) ──────────────────────────


class FakeRunner(PipelineRunner):
    """Returns a scripted list of paths per query (ranked candidates)."""

    def __init__(
        self,
        paths_by_query: Mapping[str, Sequence[str]],
        traces: Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        self._paths = paths_by_query
        self._traces = traces or {}

    async def run(self, profile: Any, query: str) -> tuple[tuple[str, ...], dict[str, object]]:
        paths = tuple(self._paths.get(query, ()))
        trace = dict(self._traces.get(query, {"stages": [], "degradations": []}))
        return paths, trace


class FakeStore(EvalStore):
    """In-memory EvalStore for runner/persist/compare tests (no DB)."""

    def __init__(self) -> None:
        self._next = 1
        self._rows: dict[int, RunRecord] = {}

    async def insert_run(self, record: RunRecord) -> int:
        rid = self._next
        self._next += 1
        self._rows[rid] = replace(record, id=rid)
        return rid

    async def update_run(self, run_id: int, record: RunRecord) -> bool:
        if run_id not in self._rows:
            return False
        self._rows[run_id] = replace(record, id=run_id)
        return True

    async def get_run(self, run_id: int) -> RunRecord | None:
        return self._rows.get(run_id)

    async def list_runs(self, limit: int = 50) -> list[RunRecord]:
        return list(self._rows.values())[-limit:]


def _golden(entries: list[GoldenEntry]) -> GoldenSet:
    return GoldenSet(
        name="test",
        archive_path="test.zim",
        archive_checksum="abc123",
        entries=tuple(entries),
        hash="testhash",
    )


def _entry(
    qid: str, query: str, paths: list[str], fact: str = "x", slice_: str = "entity"
) -> GoldenEntry:
    return GoldenEntry(
        id=qid,
        query=query,
        slice=slice_,
        expected_paths=tuple(paths),
        expected_fact=fact,
        provenance="hand-written",
    )


def _trace_with_stage(name: str, dur_ms: float) -> dict[str, object]:
    return {
        "stages": [
            {"name": name, "duration_ms": dur_ms, "params": {}, "inputs": {}, "outputs": {}}
        ],
        "degradations": [],
    }


def _trace_degraded(comp: str) -> dict[str, object]:
    return {"stages": [], "degradations": [{"component": comp, "missing": "x", "reason": "r"}]}


# ── Golden set ───────────────────────────────────────────────────────────────


class TestGoldenSet:
    def test_load_full_set_has_all_slices(self) -> None:
        gs = load_full_set()
        # owner decision: the original 6 slices at 10 queries each (60), plus
        # the `reformulation` slice at 20 (within the recommended
        # "15-25 judged queries") — 80 total.
        assert len(gs.entries) == 80, "owner decision: 60 original + 20 reformulation queries"
        by_slice = gs.by_slice()
        for sl in SLICES:
            expected = 20 if sl == "reformulation" else 10
            assert len(by_slice[sl]) == expected, f"slice {sl} should have {expected} entries"
        assert gs.hash, "golden set must be content-hashed"

    def test_load_fixture_subset(self) -> None:
        gs = load_set("fixture_subset")
        assert gs.name == "fixture_subset"
        assert gs.archive_path == "fixture"

    def test_verify_against_archive_passes(self) -> None:
        gs = _golden([_entry("e1", "q", ["Albert_Einstein"], fact="relativity")])
        assert (
            verify_against_archive(
                gs, lambda p: "the theory of relativity" if p == "Albert_Einstein" else None
            )
            == []
        )

    def test_verify_against_archive_flags_missing_path(self) -> None:
        gs = _golden([_entry("e1", "q", ["Does_Not_Exist"])])
        failures = verify_against_archive(gs, lambda p: None)
        assert any("no expected path resolves" in f for f in failures)

    def test_verify_against_archive_flags_missing_fact(self) -> None:
        gs = _golden([_entry("e1", "q", ["Earth"], fact="Martian colony")])
        failures = verify_against_archive(gs, lambda p: "Earth is a planet.")
        assert any("fact" in f for f in failures)

    def test_out_of_corpus_skipped_in_verification(self) -> None:
        gs = _golden([_entry("o1", "q", [], slice_="out_of_corpus")])
        assert verify_against_archive(gs, lambda p: None) == []


# ── Metrics ──────────────────────────────────────────────────────────────────


class TestMetrics:
    def test_recall_at_k_binary(self) -> None:
        assert recall_at(["A", "B", "C"], ["B"], 10) == 1.0
        assert recall_at(["A", "C", "D"], ["B"], 10) == 0.0
        assert recall_at(["A", "B"], ["B"], 1) == 0.0  # not in top-1

    def test_recall_empty_expected_is_zero(self) -> None:
        assert recall_at(["A"], [], 10) == 0.0  # out_of_corpus: abstention, not recall

    def test_ndcg_rank_based(self) -> None:
        import math

        # Hit at rank 1 ⇒ nDCG=1.0; rank 2 ⇒ 1/log2(3)≈0.631
        assert ndcg_at(["A"], ["A"], 10) == pytest.approx(1.0)
        assert ndcg_at(["X", "A"], ["A"], 10) == pytest.approx(1.0 / math.log2(3))
        assert ndcg_at(["A"], ["A"], 10) >= ndcg_at(["X", "A"], ["A"], 10)

    def test_mrr(self) -> None:
        assert mrr_for(["A", "B"], ["B"]) == 0.5
        assert mrr_for(["A"], ["B"]) == 0.0

    def test_path_matches_namespace_normalization(self) -> None:
        # Bare and A/-prefixed paths name the same article (pinned vs fixture scheme)
        assert path_matches("Albert_Einstein", "A/Albert_Einstein")
        assert path_matches("A/Earth", "Earth")

    def test_aggregate_means_across_queries(self) -> None:
        pairs = [(["A"], ["A"]), (["X", "B"], ["B"])]  # recall@1: 1.0, 0.0
        sm = aggregate(pairs)
        assert sm.recall_at_1 == 0.5
        assert sm.count == 2

    def test_latency_from_traces_percentiles(self) -> None:
        traces = [_trace_with_stage("candidate_source", float(i)) for i in [10, 20, 30, 40]]
        lat = latency_from_traces(traces)
        assert lat.stage_p50["candidate_source"] == pytest.approx(25.0)
        assert lat.total_p50 == pytest.approx(25.0)

    def test_degradations_collected_from_traces(self) -> None:
        traces = [
            _trace_degraded("passage_scorer/cross_encoder"),
            {"stages": [], "degradations": []},
        ]
        assert degradations_from_traces(traces) == ("passage_scorer/cross_encoder",)

    def test_run_metrics_round_trip(self) -> None:
        from vesta.eval.metrics import LatencyPercentiles, RunMetrics

        pairs = [(["A"], ["A"]), (["X"], ["A"])]
        sm = aggregate(pairs)
        metrics = RunMetrics(
            slices={"all": sm},
            latency_ms=LatencyPercentiles(),
            degraded=False,
            degraded_components=(),
            query_count=2,
        )
        d = metrics.to_dict()
        restored = run_metrics_from_dict(d)
        assert restored.query_count == 2
        assert restored.slice("all").recall_at_1 == pytest.approx(0.5)


# ── Runner: evaluate / persist / compare / sweep ─────────────────────────────


class TestRunner:
    @pytest.mark.asyncio
    async def test_evaluate_profile_collects_hits(self) -> None:
        gs = _golden(
            [
                _entry("e1", "einstein", ["Albert_Einstein"]),
                _entry("e2", "mars", ["Mars"]),
            ]
        )
        runner = FakeRunner({"einstein": ["Albert_Einstein"], "mars": ["Venus", "Mars"]})
        metrics, results = await evaluate_profile(
            load_profile("lexical") or _any_profile(), runner, gs
        )  # type: ignore[arg-type]
        assert len(results) == 2
        assert results[0].hit_rank == 1
        assert results[1].hit_rank == 2
        assert metrics.slice("all").recall_at_10 == 1.0
        assert metrics.query_count == 2

    @pytest.mark.asyncio
    async def test_persist_run_round_trips_through_store(self) -> None:
        gs = _golden([_entry("e1", "q", ["A"])])
        runner = FakeRunner({"q": ["A"]})
        profile = load_profile("lexical") or _any_profile()
        metrics, results = await evaluate_profile(profile, runner, gs)  # type: ignore[arg-type]
        store = FakeStore()
        rid = await persist_run(
            store,
            profile=profile,
            golden=gs,
            metrics=metrics,
            results=results,
            settings_snapshot={"x": 1},
            archive_path="t.zim",
            archive_checksum="abc",
        )
        record = await store.get_run(rid)
        assert record is not None
        assert record.profile_name == profile.name
        assert record.archive_checksum == "abc"
        assert record.metrics.query_count == 1

    @pytest.mark.asyncio
    async def test_compare_reports_win_loss(self) -> None:
        gs = _golden([_entry("e1", "q1", ["A"]), _entry("e2", "q2", ["B"])])
        # Baseline: both hit at rank 1. Candidate: q1 lost (miss), q2 unchanged.
        base_runner = FakeRunner({"q1": ["A"], "q2": ["B"]})
        cand_runner = FakeRunner({"q1": ["Z"], "q2": ["B"]})
        profile = _any_profile()
        bm, br = await evaluate_profile(profile, base_runner, gs)
        cm, cr = await evaluate_profile(profile, cand_runner, gs)
        baseline = _record(profile, gs, bm, br)
        candidate = _record(profile, gs, cm, cr)
        comp = compare(baseline, candidate)
        assert comp.losses == 1
        assert comp.wins == 0
        assert comp.unchanged == 1
        assert comp.degraded_guard is None

    @pytest.mark.asyncio
    async def test_compare_degradation_guard_blocks_without_force(self) -> None:
        gs = _golden([_entry("e1", "q", ["A"])])
        profile = _any_profile()
        clean_m, clean_r = await evaluate_profile(profile, FakeRunner({"q": ["A"]}), gs)
        # Candidate trace records a degradation (reranker dropped).
        degr = FakeRunner({"q": ["A"]}, traces={"q": _trace_degraded("scorer/cross_encoder")})
        deg_m, deg_r = await evaluate_profile(profile, degr, gs)
        baseline = _record(profile, gs, clean_m, clean_r)
        candidate = _record(profile, gs, deg_m, deg_r)
        assert candidate.metrics.degraded is True
        guarded = compare(baseline, candidate)
        assert guarded.degraded_guard is not None
        forced = compare(baseline, candidate, force=True)
        assert forced.degraded_guard is None  # --force overrides

    def test_parse_sweep_clones_profile_per_value(self) -> None:
        profile = load_profile("lexical")
        assert profile is not None
        points = parse_sweep("rrf.k=10,20,40", profile)
        assert [p.value for p in points] == ["10", "20", "40"]
        # Each clone carries the overridden k on the rrf fuser.
        for p in points:
            assert p.profile.fusion.params["k"] in (10, 20, 40)
        # The base profile is untouched.
        assert profile.fusion.params["k"] == 20  # lexical default

    def test_parse_sweep_rejects_bad_spec(self) -> None:
        profile = load_profile("lexical")
        assert profile is not None
        with pytest.raises(ValueError):
            parse_sweep("no_equals_sign", profile)
        with pytest.raises(ValueError):
            parse_sweep("nodothere=1,2", profile)


def _any_profile() -> Any:
    p = load_profile("lexical")
    assert p is not None
    return p


def _record(profile: Any, gs: GoldenSet, metrics: Any, results: Sequence[QueryResult]) -> RunRecord:
    return RunRecord(
        id=0,
        started_at="t",
        profile_name=profile.name,
        profile_hash=profile.hash,
        profile_yaml="",
        golden_hash=gs.hash,
        archive_path=gs.archive_path,
        archive_checksum=gs.archive_checksum,
        settings_snapshot={},
        git_sha="",
        machine_id="m",
        metrics=metrics,
        per_query=tuple(r.to_dict() for r in results),
        notes="",
    )


# ── Regression gate: must FAIL a degraded profile (the load-bearing DoD) ─────


class TestRegressionGate:
    def test_gate_passes_on_improvement(self) -> None:
        baseline = _metrics_record(recall10=0.5)
        candidate = _metrics_record(recall10=0.6)
        decision = gate_evaluate(baseline, candidate, epsilon=0.02)
        assert decision.passed is True

    def test_gate_passes_within_epsilon(self) -> None:
        baseline = _metrics_record(recall10=0.70)
        candidate = _metrics_record(recall10=0.69)
        decision = gate_evaluate(baseline, candidate, epsilon=0.02)
        assert decision.passed is True  # 0.01 drop < 0.02 epsilon

    def test_gate_FAILS_on_regression_past_epsilon(self) -> None:
        """The DoD: a deliberately degraded profile must fail the gate."""
        baseline = _metrics_record(recall10=0.70)
        candidate = _metrics_record(recall10=0.60)  # 0.10 drop >> 0.02 epsilon
        decision = gate_evaluate(baseline, candidate, epsilon=0.02)
        assert decision.passed is False
        assert "dropped" in decision.reason


def _metrics_record(*, recall10: float) -> RunRecord:
    from vesta.eval.metrics import LatencyPercentiles, RunMetrics, SliceMetrics

    sm = SliceMetrics(
        count=10,
        recall_at_1=recall10,
        recall_at_5=recall10,
        recall_at_10=recall10,
        recall_at_20=recall10,
        ndcg_at_10=recall10,
        mrr=recall10,
    )
    metrics = RunMetrics(
        slices={"all": sm},
        latency_ms=LatencyPercentiles(),
        degraded=False,
        degraded_components=(),
        query_count=10,
    )
    return RunRecord(
        id=0,
        started_at="",
        profile_name="p",
        profile_hash="h",
        profile_yaml="",
        golden_hash="g",
        archive_path="",
        archive_checksum="",
        settings_snapshot={},
        git_sha="",
        machine_id="m",
        metrics=metrics,
        per_query=(),
        notes="",
    )


# ── Calibration ─────────────────────────────────────────────────────────────────────────────


class TestCalibration:
    def test_escalates_on_low_density(self) -> None:
        s = ConfidenceSample(
            slice="paraphrase",
            top_score=None,
            score_dropoff=None,
            density=0.1,
            agreement=0.0,
            hit=False,
        )
        t = ConfidenceThresholds(top_score=0.3, score_dropoff=0.5, density=0.5, agreement=0.0)
        assert escalates(s, t) is True

    def test_does_not_escalate_on_strong_density(self) -> None:
        s = ConfidenceSample(
            slice="entity", top_score=None, score_dropoff=None, density=0.9, agreement=0.0, hit=True
        )
        t = ConfidenceThresholds(top_score=0.3, score_dropoff=0.5, density=0.5, agreement=0.0)
        assert escalates(s, t) is False

    def test_fit_thresholds_reports_achieved_rho(self) -> None:
        # 10 samples: 3 low-density (escalate), 7 high-density (don't).
        samples = [
            ConfidenceSample(
                slice="entity",
                top_score=None,
                score_dropoff=None,
                density=0.9 if i >= 3 else 0.1,
                agreement=0.0,
                hit=i >= 3,
            )
            for i in range(10)
        ]
        result = fit_thresholds(samples, target_rho=0.25)
        assert result.sample_count == 10
        assert 0.0 <= result.achieved_rho <= 1.0
        assert result.thresholds.density in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


# ── Bench (hardware is fast; encoder rows are deferred) ──────────────────────


class TestBench:
    def test_gemm_ceiling_measures_positive(self) -> None:
        from vesta.eval.bench import hardware

        result = hardware.measure_gemm_ceiling(n=256, repeats=2)
        assert result.value > 0.0
        assert result.verdict == "replaces"

    def test_memory_bandwidth_measures_positive(self) -> None:
        from vesta.eval.bench import hardware

        result = hardware.measure_memory_bandwidth(size_mb=64, repeats=2)
        assert result.value > 0.0

    def test_encoder_rows_marked_deferred(self) -> None:
        from vesta.eval.bench import encoder

        for row in encoder.deferred_rows():
            d = row.to_row()
            assert d["verdict"] == "deferred — encoder runtime absent"
            assert d["value"] is None
            assert d["projected"] is not None
