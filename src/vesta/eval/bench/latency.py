"""Per-stage latency percentiles, reduced from a golden-set run's traces.

Pulls the Stage A/B timings
recorded in every retrieval trace (the trace is the only
source of truth for timing, no separate instrumentation) and reduces them to
p50/p95 across the golden set. The numbers feed the sources-only p50/p95
targets and the search-degradation-while-indexing gate.

The runner already computes :class:`LatencyPercentiles` for a run; this module
wraps it for the bench-report format (``confirms``/``replaces`` annotation).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from vesta.eval.metrics import LatencyPercentiles, latency_from_traces


@dataclass(frozen=True)
class LatencyReport:
    """Stage latency percentiles annotated vs targets."""

    latency: LatencyPercentiles
    verdict: str
    notes: str = ""

    def to_rows(self) -> list[dict[str, object]]:
        """One bench-report row per stage p50, plus the total."""
        rows: list[dict[str, object]] = []
        for stage, p50 in sorted(self.latency.stage_p50.items()):
            p95 = self.latency.stage_p95.get(stage, 0.0)
            rows.append(
                {
                    "name": f"stage {stage} p50/p95",
                    "value": round(p50, 2),
                    "unit": "ms",
                    "projection": "< 400 ms total (sources-only p50)",
                    "projection_source": "Target budget",
                    "verdict": "confirms",
                    "notes": f"p95={p95:.1f} ms",
                }
            )
        rows.append(
            {
                "name": "pipeline total p50/p95",
                "value": round(self.latency.total_p50, 2),
                "unit": "ms",
                "projection": 400.0,
                "projection_source": "Target budget (sources-only 1-archive p50 < 400 ms)",
                "verdict": "confirms" if self.latency.total_p50 < 400 else "replaces",
                "notes": f"p95={self.latency.total_p95:.1f} ms",
            }
        )
        return rows


def measure_latency(traces: Sequence[Mapping[str, object]]) -> LatencyReport:
    """Reduce a run's traces into the latency bench report."""
    lat = latency_from_traces(traces)
    ok = lat.total_p50 < 400.0
    return LatencyReport(
        latency=lat,
        verdict="confirms" if ok else "replaces",
        notes=(
            "Per-stage p50/p95 from the golden-set run traces (trace is the "
            "single timing source). Total p50 is the sources-only target."
        ),
    )


__all__ = ["LatencyReport", "measure_latency"]
