#!/usr/bin/env python3
"""The defaults matrix (2 models x 3 profiles).

Builds on ``phase21_2_analysis.analyze_run`` (S1-S5 columns, unchanged
definitions) and adds these cells:

* lever firings (the levers shipped OFF): counts of
  ``round_cap_fires``, ``compact_reask.fired``, ``aged_requests`` keys in
  ``trace_json`` (runs that predate the levers simply lack the keys);
* per-capability strict (judged-only) from rows joined to the dataset;
* paired correct-flips vs the same model's ``full`` cell (discordant pairs,
  exact sign test) — the within-model S2 read;
* tool-question core (the same model's full-run multi-request set): correct
  count on that subset per profile, plus read-article stage counts.

DB opened mode=ro; nothing written. Regenerates the artifact, never
hand-edited.

Usage:
    uv run python scripts/bench/phase21_5_analysis.py --json-out \
        benchmarks/phase21_5_matrix.json 59 62 64 <lfm-full> <lfm-16k> <lfm-8kfpw>
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase21_2_analysis import analyze_run

SCORED = ("correct", "partial", "incorrect")
DATASET = "benchmarks/vesta_bench_v2.json"


def sign_test_p(minus: int, plus: int) -> float:
    """Two-sided exact sign test on discordant pairs."""
    n = minus + plus
    if n == 0:
        return 1.0
    k = min(minus, plus)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * tail)


def rows(db: sqlite3.Connection, run_id: int) -> dict[str, sqlite3.Row]:
    return {
        r["question_id"]: r
        for r in db.execute(
            "SELECT question_id, verdict, abstained, input_tokens, trace_json"
            " FROM bench_question_results WHERE run_id=?",
            (run_id,),
        )
    }


def lever_firings(trace: dict) -> dict[str, int]:
    t = json.loads(trace) if trace else {}
    return {
        "round_cap_fires": int(t.get("round_cap_fires") or 0),
        "compact_reask_fired": 1 if (t.get("compact_reask") or {}).get("fired") else 0,
        "aged_requests": int(t.get("aged_requests") or 0),
    }


def trace_totals(
    rr: dict[str, sqlite3.Row], caps: dict[str, str]
) -> tuple[dict[str, int], int, dict[str, str]]:
    levers = {"round_cap_fires": 0, "compact_reask_fired": 0, "aged_requests": 0}
    by_cap: dict[str, list[int]] = {}
    reads = 0
    for qid, row in rr.items():
        t = json.loads(row["trace_json"]) if row["trace_json"] else {}
        for k, v in lever_firings(row["trace_json"]).items():
            levers[k] += v
        reads += sum(1 for s in t.get("stages") or [] if s.get("name") == "read_article")
        if row["verdict"] in SCORED:
            by_cap.setdefault(caps[qid], []).append(1 if row["verdict"] == "correct" else 0)
    return levers, reads, {c: f"{sum(v)}/{len(v)}" for c, v in sorted(by_cap.items())}


def vs_full(full: dict[str, sqlite3.Row], this: dict[str, sqlite3.Row]) -> dict[str, object]:
    """Paired correct-flips (exact sign test) + tool-core decomposition."""
    minus = plus = 0
    core = [
        qid
        for qid, row in full.items()
        if (json.loads(row["trace_json"]) or {}).get("requests", 0) > 1
    ]
    for qid, a in full.items():
        b = this.get(qid)
        if b and a["verdict"] in SCORED and b["verdict"] in SCORED:
            ca, cb = a["verdict"] == "correct", b["verdict"] == "correct"
            if ca and not cb:
                minus += 1
            elif cb and not ca:
                plus += 1
    return {
        "paired_vs_full": f"-{minus}/+{plus} (p={sign_test_p(minus, plus):.1e})",
        "tool_core": {
            "core_n": len(core),
            "correct_full": f"{sum(1 for q in core if full[q]['verdict'] == 'correct')}/{len(core)}",
            "correct_this": f"{sum(1 for q in core if this[q]['verdict'] == 'correct')}/{len(core)}",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_ids", nargs="+", type=int)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    with open(DATASET) as f:
        caps = {q["id"]: q["capability"] for q in json.load(f)["questions"]}
    db = sqlite3.connect("file:data/vesta.db?mode=ro", uri=True)
    db.row_factory = sqlite3.Row

    base = [analyze_run(db, rid) for rid in args.run_ids]
    per_run = {r["run"]: rows(db, r["run"]) for r in base}
    full_by_model = {
        r["models"].split(" /")[0]: r["run"] for r in base if r["context_profile"] == "full"
    }

    out: list[dict[str, object]] = []
    for r in base:
        rid = r["run"]
        levers, reads, cap_strict = trace_totals(per_run[rid], caps)
        cell: dict[str, object] = dict(r)
        cell["lever_firings_total"] = levers
        cell["read_article_stages"] = reads
        cell["per_capability_strict_judged"] = cap_strict
        full_rid = full_by_model.get(r["models"].split(" /")[0])
        if full_rid and full_rid != rid:
            cell.update(vs_full(per_run[full_rid], per_run[rid]))
        out.append(cell)

    for c in out:
        print(f"== run {c['run']} '{c['label']}' profile={c['context_profile']} {c['models']}")
        for k in (
            "lever_firings_total",
            "read_article_stages",
            "per_capability_strict_judged",
            "paired_vs_full",
            "tool_core",
        ):
            if k in c:
                print(f"  {k:30} {c[k]}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=1) + "\n")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
