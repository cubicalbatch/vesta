"""Retrieval metrics: recall@k, nDCG@10, MRR, per-slice, latency, degradation guard.

The single source of truth for "did this change help". Two things
shape this module:

* **Ranks come from the retrieval result, not the trace.** A run produces an
  ordered list of candidate article paths (the source cards); the metrics ask
  "is the expected article in the top-k". Scores are absent for lexical sources
  (python-libzim exposes neither score nor snippet), so nDCG/MRR use
  *rank position*, not score magnitude — the only ranking signal at depth 0.
* **The degradation guard is a metric, not a log entry.** A run whose traces
  contain capability drops is ``degraded``; comparing a degraded run against a
  clean one requires ``--force`` (to prevent concluding "the reranker helped" when the
  reranker wasn't loaded). The guard makes that comparison impossible by accident.

Latency percentiles are pulled from the trace (the trace is
the only source of truth for timing — no separate instrumentation). Each trace
records per-stage ``duration_ms``; here we reduce across the golden set.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

# ── Path matching ────────────────────────────────────────────────────────────


def _normalize_path(path: str) -> str:
    """Normalize a ZIM entry path for comparison.

    The pinned archive uses the new namespace scheme (bare paths like
    ``Albert_Einstein``); the tiny fixture uses the legacy ``A/...`` scheme. Both
    ``Albert_Einstein`` and ``A/Albert_Einstein`` resolve to the same article,
    so comparison strips a leading ``A/`` namespace and any URL-encoding quirks
    (``%27`` → ``'``) before matching.
    """
    p = path.strip()
    if p.startswith("A/"):
        p = p[2:]
    return p


def path_matches(retrieved: str, expected: str) -> bool:
    """True when a retrieved path names the same article as an expected one."""
    return _normalize_path(retrieved) == _normalize_path(expected)


def _hit_rank(retrieved_paths: Sequence[str], expected: Sequence[str]) -> int | None:
    """Rank (1-based) of the first retrieved path that matches any expected path.

    ``None`` when none of the expected paths appear in the retrieved list. For
    ``out_of_corpus`` queries (empty ``expected``) the rank is ``None`` — there
    is no correct article, so recall/MRR are computed on abstention instead.
    """
    norm_expected = {_normalize_path(e) for e in expected}
    for i, p in enumerate(retrieved_paths):
        if _normalize_path(p) in norm_expected:
            return i + 1
    return None


# ── Set-level metrics ────────────────────────────────────────────────────────


def recall_at(retrieved_paths: Sequence[str], expected: Sequence[str], k: int) -> float:
    """1.0 if any expected article is in the top-k retrieved, else 0.0.

    Binary recall at k ("correct article in candidates"). At depth 0 the
    expected set is small (usually one article), so this reduces to "is the one
    right article in the top-k". Returns 0.0 for ``out_of_corpus`` (empty
    expected) — those queries are scored by abstention accuracy, not recall.
    """
    if not expected:
        return 0.0
    topk = retrieved_paths[:k]
    for e in expected:
        if any(path_matches(r, e) for r in topk):
            return 1.0
    return 0.0


def mrr_for(retrieved_paths: Sequence[str], expected: Sequence[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of the first hit, 0 if no hit."""
    rank = _hit_rank(retrieved_paths, expected)
    return 1.0 / rank if rank is not None else 0.0


def ndcg_at(retrieved_paths: Sequence[str], expected: Sequence[str], k: int) -> float:
    """nDCG@k using rank-based graded relevance.

    No scores are available at depth 0, so relevance is a step function
    of rank: a hit at rank ``r`` contributes ``1/log2(r+1)``. The ideal DCG
    places the single relevant doc at rank 1 (``1/log2(2) = 1``), so nDCG =
    DCG/IDCG. This is the standard rank-only nDCG used when scores are absent.
    """
    if not expected:
        return 0.0
    rank = _hit_rank(retrieved_paths[:k], expected)
    if rank is None:
        return 0.0
    dcg = 1.0 / math.log2(rank + 1)
    idcg = 1.0 / math.log2(2)  # ideal: relevant doc at rank 1
    return dcg / idcg if idcg > 0 else 0.0


@dataclass(frozen=True)
class SliceMetrics:
    """Metrics for one slice (or the ``all`` aggregate)."""

    count: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    ndcg_at_10: float
    mrr: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "recall@1": self.recall_at_1,
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "recall@20": self.recall_at_20,
            "ndcg@10": self.ndcg_at_10,
            "mrr": self.mrr,
        }


def _f(d: Mapping[str, object], key: str) -> float:
    v = d.get(key)
    return float(v) if isinstance(v, int | float) else 0.0


def _i(d: Mapping[str, object], key: str) -> int:
    v = d.get(key)
    return int(v) if isinstance(v, int | float) else 0


def slice_from_dict(d: Mapping[str, object]) -> SliceMetrics:
    return SliceMetrics(
        count=_i(d, "count"),
        recall_at_1=_f(d, "recall@1"),
        recall_at_5=_f(d, "recall@5"),
        recall_at_10=_f(d, "recall@10"),
        recall_at_20=_f(d, "recall@20"),
        ndcg_at_10=_f(d, "ndcg@10"),
        mrr=_f(d, "mrr"),
    )


def aggregate(retrieved_per_query: Sequence[tuple[Sequence[str], Sequence[str]]]) -> SliceMetrics:
    """Reduce ``(retrieved_paths, expected_paths)`` pairs into one SliceMetrics.

    The mean of each metric across the queries. Empty input yields zeros
    (count 0) — a degenerate but well-defined result.
    """
    n = len(retrieved_per_query)
    if n == 0:
        return SliceMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    r1 = sum(recall_at(r, e, 1) for r, e in retrieved_per_query) / n
    r5 = sum(recall_at(r, e, 5) for r, e in retrieved_per_query) / n
    r10 = sum(recall_at(r, e, 10) for r, e in retrieved_per_query) / n
    r20 = sum(recall_at(r, e, 20) for r, e in retrieved_per_query) / n
    nd = sum(ndcg_at(r, e, 10) for r, e in retrieved_per_query) / n
    mrr = sum(mrr_for(r, e) for r, e in retrieved_per_query) / n
    return SliceMetrics(n, r1, r5, r10, r20, nd, mrr)


@dataclass(frozen=True)
class RunMetrics:
    """Full metrics for one eval run: per-slice + aggregate + latency + guard."""

    slices: Mapping[str, SliceMetrics]
    latency_ms: LatencyPercentiles
    degraded: bool
    degraded_components: tuple[str, ...]
    query_count: int

    def slice(self, name: str) -> SliceMetrics:
        return self.slices.get(name) or SliceMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "query_count": self.query_count,
            "degraded": self.degraded,
            "degraded_components": list(self.degraded_components),
            "slices": {k: v.to_dict() for k, v in self.slices.items()},
            "latency_ms": self.latency_ms.to_dict(),
        }


def latency_from_dict(d: Mapping[str, object]) -> LatencyPercentiles:
    raw50 = d.get("stage_p50_ms")
    raw95 = d.get("stage_p95_ms")
    p50 = dict(raw50) if isinstance(raw50, dict) else {}
    p95 = dict(raw95) if isinstance(raw95, dict) else {}
    return LatencyPercentiles(
        stage_p50={str(k): float(v) for k, v in p50.items() if isinstance(v, int | float)},
        stage_p95={str(k): float(v) for k, v in p95.items() if isinstance(v, int | float)},
        total_p50=_f(d, "total_p50_ms"),
        total_p95=_f(d, "total_p95_ms"),
    )


def run_metrics_from_dict(d: Mapping[str, object]) -> RunMetrics:
    """Reconstruct a RunMetrics from its persisted dict shape."""
    raw_slices = d.get("slices")
    slices_map = raw_slices if isinstance(raw_slices, dict) else {}
    slices: dict[str, SliceMetrics] = {}
    for name, sm in slices_map.items():
        if isinstance(sm, dict):
            slices[str(name)] = slice_from_dict(sm)
    raw_latency = d.get("latency_ms")
    latency = (
        latency_from_dict(raw_latency) if isinstance(raw_latency, dict) else LatencyPercentiles()
    )
    raw_degs = d.get("degraded_components")
    degs = tuple(str(x) for x in raw_degs) if isinstance(raw_degs, list) else ()
    return RunMetrics(
        slices=slices,
        latency_ms=latency,
        degraded=bool(d.get("degraded")),
        degraded_components=degs,
        query_count=_i(d, "query_count"),
    )


# ── Latency percentiles (from the trace — the only source of truth) ──────────


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile of a list of durations (ms).

    ``pct`` in [0,100]. Returns 0.0 for empty input (no stages of that name ran).
    """
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (pct / 100.0) * (len(s) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return s[int(rank)]
    frac = rank - lo
    return s[lo] + (s[hi] - s[lo]) * frac


@dataclass(frozen=True)
class LatencyPercentiles:
    """Per-stage p50/p95 latency in ms, reduced across a run's traces.

    Stage names match the trace's stage ``name`` field (``candidate_source``,
    ``fuser``, ``passage_builder``, ``passage_scorer``, ``context_assembler``).
    ``total`` is the wall-clock of the whole pipeline per query. The trace is
    the single source of truth: no timing is recorded here.
    """

    stage_p50: Mapping[str, float] = field(default_factory=dict)
    stage_p95: Mapping[str, float] = field(default_factory=dict)
    total_p50: float = 0.0
    total_p95: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_p50_ms": dict(self.stage_p50),
            "stage_p95_ms": dict(self.stage_p95),
            "total_p50_ms": self.total_p50,
            "total_p95_ms": self.total_p95,
        }


def latency_from_traces(
    traces: Sequence[Mapping[str, object]],
) -> LatencyPercentiles:
    """Reduce per-stage ``duration_ms`` across a run's traces into percentiles.

    Each trace is ``Trace.to_dict()``: ``{"stages": [{"name","duration_ms"}, …]}``.
    The total per query is the sum of its stage durations (the trace does not
    record a wall-clock envelope; summing stages is the faithful reconstruction).
    """
    by_stage: dict[str, list[float]] = {}
    totals: list[float] = []
    for tr in traces:
        raw_stages = tr.get("stages")
        stages: Sequence[Mapping[str, object]] = raw_stages if isinstance(raw_stages, list) else []
        total = 0.0
        for st in stages:
            name = str(st.get("name") or "")
            dur = st.get("duration_ms")
            if not isinstance(dur, int | float) or name == "":
                continue
            by_stage.setdefault(name, []).append(float(dur))
            total += float(dur)
        totals.append(total)
    p50 = {n: percentile(v, 50) for n, v in by_stage.items()}
    p95 = {n: percentile(v, 95) for n, v in by_stage.items()}
    return LatencyPercentiles(
        stage_p50=p50,
        stage_p95=p95,
        total_p50=percentile(totals, 50),
        total_p95=percentile(totals, 95),
    )


def degradations_from_traces(traces: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    """Collect every dropped component across a run's traces (the guard signal)."""
    out: list[str] = []
    for tr in traces:
        raw_degs = tr.get("degradations")
        degs: Sequence[Mapping[str, object]] = raw_degs if isinstance(raw_degs, list) else []
        for d in degs:
            comp = str(d.get("component") or "")
            if comp and comp not in out:
                out.append(comp)
    return tuple(out)


__all__ = [
    "LatencyPercentiles",
    "RunMetrics",
    "SliceMetrics",
    "aggregate",
    "degradations_from_traces",
    "latency_from_dict",
    "latency_from_traces",
    "mrr_for",
    "ndcg_at",
    "path_matches",
    "percentile",
    "recall_at",
    "run_metrics_from_dict",
    "slice_from_dict",
]
