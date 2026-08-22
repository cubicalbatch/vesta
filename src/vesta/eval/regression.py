"""Regression gate — fail a run if retrieval metrics drop past epsilon.

The rule this enforces is the whole point of the eval harness: *a change
that doesn't move a metric doesn't ship* — and a change that moves it *down*
more than ``eval.regression.epsilon`` fails the gate. The gate must be shown to
**fail a deliberately degraded profile**, which is what makes the rule
real rather than aspirational.

Two tiers, because the pinned archive is gitignored and not in CI:

* **CI-runnable gate** runs the ``fixture_subset`` golden set against the tiny
  ZIM fixture. It executes on every push and is what fails a degraded profile
  mechanically. Annotated clearly so nobody mistakes the fixture numbers for the
  pinned-archive targets.
* **Full gate** runs the 60-query set against the pinned archive on demand /
  nightly (where the archive is present). This is where the retrieval targets
  (recall@10 ≥ 0.70) are actually checked.

This module holds the *logic*; the CLI (``vesta eval regression``) and the CI
script wire it to a store + runner.
"""

from __future__ import annotations

from dataclasses import dataclass

from vesta.eval.runner import RunRecord


@dataclass(frozen=True)
class GateDecision:
    """The gate's verdict on a candidate run vs its baseline."""

    passed: bool
    metric: str  # the metric the gate keys on (recall@10)
    delta: float
    epsilon: float
    baseline_id: int
    candidate_id: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "metric": self.metric,
            "delta": self.delta,
            "epsilon": self.epsilon,
            "baseline_id": self.baseline_id,
            "candidate_id": self.candidate_id,
            "reason": self.reason,
        }


#: The metric the gate keys on (the headline retrieval target).
GATE_METRIC = "recall@10"


def _slice_metric(record: RunRecord, slice_name: str, metric_name: str) -> float:
    d = record.metrics.slice(slice_name).to_dict()
    val = d.get(metric_name)
    return float(val) if isinstance(val, int | float) else 0.0


def evaluate(
    baseline: RunRecord,
    candidate: RunRecord,
    *,
    epsilon: float,
    metric: str = GATE_METRIC,
) -> GateDecision:
    """Pass iff the candidate's ``metric`` did not drop more than ``epsilon``.

    A regression is a *drop* — an improvement or no-change always passes. The
    gate keys on the ``all`` aggregate of the configured metric (default
    recall@10) so it reflects the whole set, not one slice.
    """
    b = _slice_metric(baseline, "all", metric)
    c = _slice_metric(candidate, "all", metric)
    delta = c - b
    # Only a drop past -epsilon fails; gains and flat lines pass.
    passed = delta >= -epsilon
    if passed:
        reason = f"{metric} {c:+.4f} within epsilon {epsilon:.4f} of baseline {b:.4f}"
    else:
        reason = (
            f"{metric} dropped {abs(delta):.4f} (from {b:.4f} to {c:.4f}), "
            f"exceeding epsilon {epsilon:.4f}"
        )
    return GateDecision(
        passed=passed,
        metric=metric,
        delta=delta,
        epsilon=epsilon,
        baseline_id=baseline.id,
        candidate_id=candidate.id,
        reason=reason,
    )


__all__ = ["GATE_METRIC", "GateDecision", "evaluate"]
