#!/usr/bin/env python3
"""Pre-seed sensitivity analysis over frozen-context replays.

Reads the `answer_only --from-context --context-passages N` replay runs plus
the shared retrieval snapshot, and reports:

- judged-only strict accuracy / sub-fact coverage / over-refusal per N
  (recomputed from rows so the denominator excludes `unjudged`; the stored
  metrics_json uses the all-n convention and is printed for cross-check),
- paired flips against the N=6 control (same retrieval, thinner pre-seed —
  any score delta is synthesis, not search),
- gold-in-top-N per N from the snapshot (does the thinner pre-seed still
  CONTAIN the answer), using the same `path_matches` normalisation as
  `source_hit_rank`.

Usage:
    python3 scripts/bench/phase21_1a_analysis.py SNAP.json RUN_N6 RUN_N4 RUN_N3 RUN_N2 \
        [--db data/vesta.db] [--out benchmarks/phase21_1a_preseed_sensitivity.json]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Mapping
from pathlib import Path

from vesta.eval.bench_dataset import BenchQuestion, load_bench_dataset
from vesta.eval.metrics import path_matches

SCORED = ("correct", "partial", "incorrect")


def run_rows(db: sqlite3.Connection, run_id: int) -> dict[str, sqlite3.Row]:
    return {
        r["question_id"]: r
        for r in db.execute(
            "SELECT question_id, verdict, abstained, sub_fact_coverage, expected_answer, "
            "answer_text, input_tokens, output_tokens, source_hit_rank, trace_json "
            "FROM bench_question_results WHERE run_id=?",
            (run_id,),
        ).fetchall()
    }


def judged_metrics(rows: dict[str, sqlite3.Row]) -> dict[str, object]:
    judged = [r for r in rows.values() if r["verdict"] in SCORED]
    correct = sum(1 for r in judged if r["verdict"] == "correct")
    subs = [r["sub_fact_coverage"] for r in judged if r["sub_fact_coverage"] is not None]
    # over-refusal: abstained on a question whose expected behavior is `answer`
    refusals = sum(
        1
        for r in rows.values()
        if r["abstained"] and r["verdict"] in SCORED and _expected_answer(r) == "answer"
    )
    return {
        "n_total": len(rows),
        "n_judged": len(judged),
        "n_unjudged": sum(1 for r in rows.values() if r["verdict"] == "unjudged"),
        "correct": correct,
        "partial": sum(1 for r in judged if r["verdict"] == "partial"),
        "incorrect": sum(1 for r in judged if r["verdict"] == "incorrect"),
        "strict_judged_only_pct": round(100 * correct / len(judged), 1) if judged else None,
        "sub_fact_coverage_judged_pct": (round(100 * sum(subs) / len(subs), 1) if subs else None),
        "over_refusal_count": refusals,
        "mean_input_tokens": round(
            sum(r["input_tokens"] or 0 for r in rows.values()) / len(rows), 1
        )
        if rows
        else None,
    }


_EXPECTED: dict[str, str] = {}


def _expected_answer(r: sqlite3.Row) -> str:
    """expected_behavior from the pinned dataset (row does not carry it)."""
    return _EXPECTED.get(r["question_id"], "answer")


def paired(control: dict[str, sqlite3.Row], other: dict[str, sqlite3.Row]) -> dict[str, object]:
    """Per-question flips between the control (N=6) and another N."""
    common = [q for q in control if q in other]
    down = sum(
        1
        for q in common
        if control[q]["verdict"] == "correct" and other[q]["verdict"] in ("partial", "incorrect")
    )
    up = sum(
        1
        for q in common
        if control[q]["verdict"] in ("partial", "incorrect") and other[q]["verdict"] == "correct"
    )
    return {
        "common": len(common),
        "correct_to_notcorrect": down,
        "notcorrect_to_correct": up,
        "verdict_disagreements": sum(
            1 for q in common if control[q]["verdict"] != other[q]["verdict"]
        ),
    }


def gold_in_topn(
    snapshot: dict[str, object], questions: Mapping[str, BenchQuestion]
) -> dict[str, object]:
    """Per-N: does the top-N passage set contain a required gold source?"""
    qs_raw = snapshot.get("questions", {})
    qs = qs_raw if isinstance(qs_raw, Mapping) else {}
    out: dict[str, object] = {}
    for n in (6, 4, 3, 2):
        hits = total = 0
        for qid, meta in questions.items():
            entry = qs.get(qid)
            if not isinstance(entry, dict):
                continue
            passages_raw = entry.get("passages") or []
            passages = passages_raw if isinstance(passages_raw, list) else []
            top_paths = [str(p.get("path")) for p in passages[:n] if isinstance(p, dict)]
            required = [s.article_path for s in meta.sources if s.required]
            if not required:
                continue
            total += 1
            if any(any(path_matches(p, exp) for exp in required) for p in top_paths):
                hits += 1
        out[f"top_{n}"] = {
            "gold_in_cards": hits,
            "n": total,
            "pct": round(100 * hits / total, 1) if total else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot")
    ap.add_argument("run_ids", nargs="+", type=int, help="run ids in N order (6 4 3 2)")
    ap.add_argument("--db", default="data/vesta.db")
    ap.add_argument("--out", default="benchmarks/phase21_1a_preseed_sensitivity.json")
    args = ap.parse_args()

    db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    dataset = load_bench_dataset()
    for q in dataset.questions:
        _EXPECTED[q.id] = q.expected_behavior
    questions = {q.id: q for q in dataset.questions}

    runs: list[dict[str, object]] = []
    control_rows: dict[str, sqlite3.Row] | None = None
    for rid in args.run_ids:
        row = db.execute(
            "SELECT label, config_json, answer_model, judge_model, subset_hash FROM bench_runs "
            "WHERE id=?",
            (rid,),
        ).fetchone()
        cfg = json.loads(row["config_json"]) if row and row["config_json"] else {}
        n = cfg.get("context_passages")
        rows = run_rows(db, rid)
        entry: dict[str, object] = {
            "run_id": rid,
            "label": row["label"] if row else "",
            "context_passages": n,
            "answer_model": row["answer_model"] if row else "",
            "judge_model": row["judge_model"] if row else "",
            "subset_hash": row["subset_hash"] if row else "",
            **judged_metrics(rows),
        }
        if control_rows is None:
            control_rows = rows
        else:
            entry["paired_vs_control"] = paired(control_rows, rows)
        runs.append(entry)

    payload = {
        "analysis": "phase21-1a-preseed-sensitivity",
        "snapshot": str(args.snapshot),
        "snapshot_dataset": snapshot.get("dataset"),
        "method": {
            "runs": "answer_only --from-context (single request per question, no agent/tools)",
            "judged_only": "verdict in correct|partial|incorrect; unjudged stated separately",
            "over_refusal": "abstained==1 on expected_behavior=='answer' questions (judged)",
            "gold_in_topn": "top-N snapshot passage paths vs required sources via "
            "path_matches (same normalisation as source_hit_rank)",
        },
        "gold_in_topn": gold_in_topn(snapshot, questions),
        "runs": runs,
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    for r in runs:
        paired_txt = " (control)"
        pv = r.get("paired_vs_control")
        if isinstance(pv, dict):
            paired_txt = (
                f" | vs control: -{pv['correct_to_notcorrect']} +{pv['notcorrect_to_correct']}"
            )
        print(
            f"run {r['run_id']} N={r['context_passages']}: judged {r['n_judged']}/{r['n_total']} "
            f"strict {r['strict_judged_only_pct']}% sub-fact {r['sub_fact_coverage_judged_pct']}% "
            f"over-refusal {r['over_refusal_count']} mean-in {r['mean_input_tokens']}" + paired_txt
        )
    print("gold_in_topn:", json.dumps(payload["gold_in_topn"]))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
