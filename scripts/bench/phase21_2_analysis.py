#!/usr/bin/env python3
"""Window-profile bench analysis (runs vs run 48 baseline).

For each run id: peak-request distribution (S1), overflow fallbacks (S4),
one-shot rate under BOTH definitions (S5 — reported first), judged strict
(all-150 and judged-only), sub-fact coverage, over-refusal, cumulative input
token mean/p95 (S3), request-count distribution, preseed_dropped rate (D4),
window-ledger blocks ("budget reached" steering strings in stage outputs),
and latency mean/p95. DB opened mode=ro; nothing is written to the DB.

Usage:
    uv run python scripts/bench/phase21_2_analysis.py 48 59 60 61
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys

SCORED = ("correct", "partial", "incorrect")
#: The D5 window ledger / insert caps' shared steering string prefix — each
#: occurrence in a stage output is one blocked (rejected or latched) call.
_STEERING_PREFIX = "You have already gathered substantial source material"


def percentile(xs: list[float], q: float) -> float:
    """Nearest-rank percentile — the repo's bench convention
    (``eval.bench_scoring._pct``), so numbers are like-for-like with
    the bench report's peak-context section."""
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = max(0, min(len(s) - 1, round(q * (len(s) - 1))))
    return float(s[idx])


def analyze_run(db: sqlite3.Connection, run_id: int) -> dict[str, object]:
    run = db.execute(
        "SELECT label, subset_hash, answer_model, judge_model, status,"
        " config_json, metrics_json FROM bench_runs WHERE id=?",
        (run_id,),
    ).fetchone()
    if run is None:
        raise SystemExit(f"run {run_id} not found")
    cfg = json.loads(run["config_json"])
    rows = db.execute(
        "SELECT question_id, verdict, abstained, input_tokens, output_tokens,"
        " latency_ms, rounds, sub_fact_coverage, trace_json"
        " FROM bench_question_results WHERE run_id=?",
        (run_id,),
    ).fetchall()

    peaks: list[int] = []
    requests: list[int] = []
    overflows = 0
    oneshot_req = 0
    oneshot_stages = 0
    preseed_dropped_q = 0
    preseed_dropped_total = 0
    ledger_blocks = 0
    ledger_block_qs = 0
    for r in rows:
        t = json.loads(r["trace_json"]) if r["trace_json"] else {}
        peaks.append(int(t.get("peak_input_tokens") or 0))
        n_req = int(t.get("requests") or 0)
        requests.append(n_req)
        overflows += int(t.get("overflow_fallbacks") or 0)
        if n_req == 1:
            oneshot_req += 1
        names = [s.get("name") for s in (t.get("stages") or [])]
        if names and all(n == "pre_seed" for n in names):
            oneshot_stages += 1
        b = t.get("budget") or {}
        d = int(b.get("preseed_dropped") or 0)
        if d > 0:
            preseed_dropped_q += 1
            preseed_dropped_total += d
        q_blocks = 0
        for s in t.get("stages") or []:
            for v in (s.get("outputs") or {}).values():
                if isinstance(v, str) and v.startswith(_STEERING_PREFIX):
                    q_blocks += 1
        if q_blocks:
            ledger_blocks += q_blocks
            ledger_block_qs += 1

    n = len(rows)
    judged = [r for r in rows if r["verdict"] in SCORED]
    correct = sum(1 for r in rows if r["verdict"] == "correct")
    abst_wrong = sum(1 for r in rows if r["abstained"] and r["verdict"] == "incorrect")
    toks = [r["input_tokens"] for r in rows]
    lat = [r["latency_ms"] for r in rows]
    sub = [r["sub_fact_coverage"] for r in judged if r["sub_fact_coverage"] is not None]
    peak_p50 = percentile([float(x) for x in peaks], 0.50)
    metrics_n = int((json.loads(run["metrics_json"]) or {}).get("answer", {}).get("n", 0))
    budget_windows = {
        int((json.loads(r["trace_json"]) or {}).get("budget", {}).get("window_tokens") or 0)
        for r in rows
        if r["trace_json"]
    }
    return {
        "run": run_id,
        "label": run["label"],
        "status": run["status"],
        "subset_hash": run["subset_hash"],
        "context_profile": cfg.get("context_profile"),
        "models": f"{run['answer_model']} / {run['judge_model']}",
        "n_questions": n,
        # S5 first, both definitions
        "oneshot_requests_eq1": f"{oneshot_req}/{n}",
        "oneshot_stages_preseed": f"{oneshot_stages}/{n}",
        # S1 peak distribution
        "peak_p50": round(peak_p50),
        "peak_p90": round(percentile([float(x) for x in peaks], 0.90)),
        "peak_p95": round(percentile([float(x) for x in peaks], 0.95)),
        "peak_max": max(peaks) if peaks else 0,
        "window_tokens_in_traces": sorted(budget_windows),
        "over_8192": sum(1 for p in peaks if p > 8192),
        "over_16384": sum(1 for p in peaks if p > 16384),
        # S4
        "overflow_fallbacks_sum": overflows,
        # S2
        "strict_all150": round(correct / n, 4) if n else 0.0,
        "correct_partial_incorrect_unjudged": "/".join(
            str(sum(1 for r in rows if r["verdict"] == v))
            for v in ("correct", "partial", "incorrect", "unjudged")
        ),
        "judged_n": len(judged),
        "strict_judged": round(sum(1 for r in judged if r["verdict"] == "correct") / len(judged), 4)
        if judged
        else 0.0,
        "subfact_mean": round(sum(sub) / len(sub), 4) if sub else None,
        "over_refusal": f"{abst_wrong}/{n}",
        # S3
        "input_tokens_mean": round(sum(toks) / n) if n else 0,
        "input_tokens_p95": round(percentile([float(x) for x in toks], 0.95)),
        "input_tokens_sum": sum(toks),
        "output_tokens_mean": round(sum(r["output_tokens"] for r in rows) / n) if n else 0,
        # requests distribution
        "requests_mean": round(sum(requests) / n, 2) if n else 0.0,
        "requests_max": max(requests) if requests else 0,
        "requests_hist": {k: sum(1 for x in requests if x == k) for k in sorted(set(requests))},
        # D4 / D5 audit
        "preseed_dropped_questions": f"{preseed_dropped_q}/{n}",
        "preseed_dropped_total": preseed_dropped_total,
        "ledger_block_questions": f"{ledger_block_qs}/{n}",
        "ledger_blocks_total": ledger_blocks,
        # latency
        "latency_mean_s": round(sum(lat) / n / 1000, 1) if n else 0.0,
        "latency_p95_s": round(percentile([float(x) for x in lat], 0.95) / 1000, 1),
        "_metrics_answer_n": metrics_n,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_ids", nargs="+", type=int)
    ap.add_argument("--json-out", default=None, help="optional artifact path")
    args = ap.parse_args()

    db = sqlite3.connect("file:data/vesta.db?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    out = [analyze_run(db, rid) for rid in args.run_ids]
    for r in out:
        print(
            f"== run {r['run']} '{r['label']}' profile={r['context_profile']}"
            f" subset={r['subset_hash']} status={r['status']}"
        )
        for k, v in r.items():
            if k.startswith("_") or k in (
                "run",
                "label",
                "status",
                "subset_hash",
                "context_profile",
            ):
                continue
            print(f"  {k:34} {v}")
    if args.json_out:
        Path = __import__("pathlib").Path
        Path(args.json_out).write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
