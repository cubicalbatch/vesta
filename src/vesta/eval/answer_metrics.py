"""Answer-quality judge Protocol + calibration correlation.

``eval/`` stays free of ``inference`` imports: the LLM judge is injected as a
:class:`JudgeLLM` Protocol by the composition root (CLI/API) — this module
never imports ``inference`` (the boundary rule: eval imports only retrieval +
config). :func:`compute_calibration_correlation` scores judge-vs-hand agreement
on a calibration subset; a correlation < 0.7 means the judge is not yet
trustworthy ("an LLM-judge without calibration is a random number generator").
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class JudgeLLM(Protocol):
    """An LLM judge for Snippet-F1/Doc-F1 scoring.

    Injected by the composition root (CLI/API) with the real gateway; tests
    inject a stub. Defining the Protocol here keeps ``eval`` from importing
    ``inference`` (boundary rule: eval imports only retrieval + config).
    """

    async def judge(self, prompt: str) -> str:
        """Score the prompt and return the judge's text response."""
        ...


def compute_calibration_correlation(
    hand_scores: Sequence[float], judge_scores: Sequence[float]
) -> float:
    """Pearson correlation between hand and judge scores.

    Returns 0.0 for empty/mismatched inputs. A correlation < 0.7 means the judge
    is not yet trustworthy.
    """
    n = min(len(hand_scores), len(judge_scores))
    if n < 2:
        return 0.0
    h = list(hand_scores[:n])
    j = list(judge_scores[:n])
    mh = sum(h) / n
    mj = sum(j) / n
    num = sum((h[i] - mh) * (j[i] - mj) for i in range(n))
    dh = (sum((x - mh) ** 2 for x in h)) ** 0.5
    dj = (sum((x - mj) ** 2 for x in j)) ** 0.5
    if dh == 0 or dj == 0:
        return 0.0
    return float(num / (dh * dj))


__all__ = [
    "JudgeLLM",
    "compute_calibration_correlation",
]
