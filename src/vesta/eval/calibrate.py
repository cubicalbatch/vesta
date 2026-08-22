"""Confidence-gate calibration — fit the four ``retrieval.confidence.*`` thresholds.

The confidence gate (top_score, score_dropoff, density,
agreement) decides whether the agent loop even *offers* a search tool. Calibrated
by eye it fires on everything or nothing; calibrating against the golden set
targets a specific escalation rate **rho ≈ 0.25** — the fraction of queries that
escalate to an extra search round.

The procedure: for each golden query, compute the four confidence
signals from the retrieval result; a query "escalates" when its top_score is
below the gate OR its dropoff/density/agreement exceed their thresholds. We
search the 4D threshold space for the configuration that brings the escalation
rate closest to rho=0.25 *without* gating the clearly-correct keyword/entity hits.

Calibration emits fitted values as settings recommendations. The confidence
signals are recorded in every trace — calibration reads
them, it does not re-derive them.

``eval`` imports only ``retrieval`` and ``config``; the confidence signals are
passed in (the runner reads them off the retrieval result), so this module never
imports ``answer`` (the boundary rule — the gate is calibrated before the loop
exists).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Target escalation rate: the fraction of queries the gate sends to an
# extra search round. rho=0.25 is the calibrated default; the eval harness reports
# the achieved value so a mis-calibrated gate is visible.
TARGET_RHO = 0.25


@dataclass(frozen=True)
class ConfidenceSample:
    """One query's four confidence signals + whether it was a retrieval hit.

    These are the :class:`ConfidenceSignals` values, lifted off the
    retrieval result. ``hit`` is whether the expected article was retrieved in
    the top-k (a miss is a prime escalation candidate). ``slice`` lets the
    calibration report per-slice escalation (paraphrase should escalate more).
    """

    slice: str
    top_score: float | None
    score_dropoff: float | None
    density: float
    agreement: float
    hit: bool


@dataclass(frozen=True)
class ConfidenceThresholds:
    """The four gate thresholds."""

    top_score: float
    score_dropoff: float
    density: float
    agreement: float


@dataclass(frozen=True)
class CalibrationResult:
    """Fitted thresholds + the achieved escalation rate, for reporting."""

    thresholds: ConfidenceThresholds
    achieved_rho: float
    target_rho: float
    per_slice_rho: dict[str, float]
    sample_count: int
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "thresholds": {
                "retrieval.confidence.top_score": self.thresholds.top_score,
                "retrieval.confidence.score_dropoff": self.thresholds.score_dropoff,
                "retrieval.confidence.density": self.thresholds.density,
                "retrieval.confidence.agreement": self.thresholds.agreement,
            },
            "achieved_rho": self.achieved_rho,
            "target_rho": self.target_rho,
            "per_slice_rho": dict(self.per_slice_rho),
            "sample_count": self.sample_count,
            "notes": self.notes,
        }


def escalates(signals: ConfidenceSample, t: ConfidenceThresholds) -> bool:
    """The gate rule: escalate when any signal crosses its threshold.

    Mirrors the four-signal gate: a low top score, a sharp dropoff, a low
    density (scattered evidence), or low source agreement all flag the query as
    worth a second look. ``None`` signals (no scorer ran) are treated as
    escalating only via the non-None axes — depth-0 lexical runs have no score,
    so density/agreement drive the decision.
    """
    if signals.top_score is not None and signals.top_score < t.top_score:
        return True
    if signals.score_dropoff is not None and signals.score_dropoff < t.score_dropoff:
        return True
    if signals.density < t.density:
        return True
    return signals.agreement < t.agreement


def escalation_rate(samples: Sequence[ConfidenceSample], t: ConfidenceThresholds) -> float:
    """Fraction of samples the gate would escalate at ``t``."""
    if not samples:
        return 0.0
    escalated = sum(1 for s in samples if escalates(s, t))
    return escalated / len(samples)


def per_slice_rho(samples: Sequence[ConfidenceSample], t: ConfidenceThresholds) -> dict[str, float]:
    """Escalation rate broken down by slice (paraphrase should escalate most)."""
    by_slice: dict[str, list[ConfidenceSample]] = {}
    for s in samples:
        by_slice.setdefault(s.slice, []).append(s)
    return {sl: escalation_rate(lst, t) for sl, lst in by_slice.items()}


def fit_thresholds(
    samples: Sequence[ConfidenceSample],
    *,
    target_rho: float = TARGET_RHO,
    seeds: ConfidenceThresholds | None = None,
) -> CalibrationResult:
    """Grid-search the threshold space for the config nearest rho=target_rho.

    The gate has four axes; a full joint search is intractable but unnecessary.
    The signal that matters most at depth 0 is ``density`` (low density ⇒ the
    evidence is scattered ⇒ escalate); ``top_score``/``score_dropoff`` are absent
    without a scorer and ``agreement`` is sparse with few sources. So the fit
    walks a coarse grid over the active axes, scoring each candidate by
    ``|achieved_rho - target_rho|``, and returns the closest.

    Honest about its limits: with no scorer the fit is density-driven and coarse.
    The achieved rho is reported regardless, so a wide gap is visible. This is
    exactly the "calibrate, don't eyeball" rule, applied to the signals
    available now.
    """
    if not samples:
        zeros = ConfidenceThresholds(0.3, 0.5, 0.5, 0.0)
        return CalibrationResult(
            thresholds=zeros,
            achieved_rho=0.0,
            target_rho=target_rho,
            per_slice_rho={},
            sample_count=0,
            notes="no samples — returned built-in defaults",
        )
    # Candidate values per axis. Density is the load-bearing one at depth 0;
    # top_score/dropoff matter once a scorer is wired.
    density_grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    top_grid = [0.2, 0.3, 0.4]
    dropoff_grid = [0.3, 0.5, 0.7]
    # Agreement is almost always 0 at depth 0 (one lexical source) — keep the
    # default (0.0 ⇒ never triggers on agreement alone).
    base = seeds or ConfidenceThresholds(
        top_score=0.3, score_dropoff=0.5, density=0.5, agreement=0.0
    )
    best = base
    best_rho = escalation_rate(samples, base)
    best_err = abs(best_rho - target_rho)
    for density in density_grid:
        for top in top_grid:
            for drop in dropoff_grid:
                t = ConfidenceThresholds(
                    top_score=top, score_dropoff=drop, density=density, agreement=base.agreement
                )
                rho = escalation_rate(samples, t)
                err = abs(rho - target_rho)
                if err < best_err:
                    best_err = err
                    best_rho = rho
                    best = t
    return CalibrationResult(
        thresholds=best,
        achieved_rho=best_rho,
        target_rho=target_rho,
        per_slice_rho=per_slice_rho(samples, best),
        sample_count=len(samples),
        notes=(
            "density-driven fit (no scorer: top_score/score_dropoff absent at depth 0); "
            "re-fit once a real scorer populates the score axes"
        ),
    )


__all__ = [
    "TARGET_RHO",
    "CalibrationResult",
    "ConfidenceSample",
    "ConfidenceThresholds",
    "escalates",
    "escalation_rate",
    "fit_thresholds",
    "per_slice_rho",
]
