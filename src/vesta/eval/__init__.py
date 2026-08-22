"""Evaluation & benchmarks.

The measurement layer that every retrieval change reports into. Two
kinds of measurement share one reporting format and one consumer:

* **Quality** — golden set, recall@k, nDCG@10, MRR, per-slice breakdowns, A/B
  deltas with per-query win/loss (``metrics``, ``runner``, ``golden``).
* **Speed** — hardware ceiling, extraction throughput, stage latency, encoder
  rows (``bench``).

Boundary: ``eval`` imports only ``retrieval`` and
``config``. Archive access (ZIM reads) and persistence (the ``eval_runs`` table)
are injected through Protocols defined in ``runner`` — so this package never
imports ``zim`` or ``db``, and stays within the ≤2 dependency cap. The CLI
(``vesta.cli``) and the API router (``api.eval``) are the composition roots that
wire the real DB-backed store and archive registry; the tests wire fakes.
"""

from __future__ import annotations

from vesta.eval import answer_metrics, bench, calibrate, golden, metrics, regression, runner

__all__ = [
    "answer_metrics",
    "bench",
    "calibrate",
    "golden",
    "metrics",
    "regression",
    "runner",
]
