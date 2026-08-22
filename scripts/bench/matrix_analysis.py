#!/usr/bin/env python3
"""Matrix Benchmark Analysis — 5 models x {8k, 16k} context profiles on Vesta Bench (200 Qs)."""

import json
import sqlite3

DB_PATH = "data/vesta.db"
DATASET_PATH = "benchmarks/vesta_bench_v2.json"

RUN_IDS = [80, 81, 82, 83, 84, 85, 86, 87, 88, 89]


def main():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    with open(DATASET_PATH) as f:
        ds = json.load(f)
    caps_by_qid = {q["id"]: q["capability"] for q in ds["questions"]}

    matrix = []

    for rid in RUN_IDS:
        r = conn.execute(
            "SELECT id, label, answer_model, judge_model, status, started_at, finished_at, "
            "config_json, metrics_json FROM bench_runs WHERE id=?",
            (rid,),
        ).fetchone()
        if not r:
            continue

        cfg = json.loads(r["config_json"])
        metrics = json.loads(r["metrics_json"]) if r["metrics_json"] else {}

        q_rows = conn.execute(
            "SELECT question_id, verdict, abstained, latency_ms, input_tokens, output_tokens, "
            "rounds, sub_fact_coverage, trace_json FROM bench_question_results WHERE run_id=?",
            (rid,),
        ).fetchall()

        n = len(q_rows)
        correct = sum(1 for q in q_rows if q["verdict"] == "correct")
        partial = sum(1 for q in q_rows if q["verdict"] == "partial")
        incorrect = sum(1 for q in q_rows if q["verdict"] == "incorrect")
        unjudged = sum(1 for q in q_rows if q["verdict"] == "unjudged")

        strict = correct / n if n else 0.0
        weighted = (correct + 0.5 * partial) / n if n else 0.0

        latencies = [q["latency_ms"] for q in q_rows if q["latency_ms"] is not None]
        total_latency_s = sum(latencies) / 1000.0 if latencies else 0.0
        avg_latency_s = total_latency_s / len(latencies) if latencies else 0.0

        in_toks = [q["input_tokens"] or 0 for q in q_rows]
        out_toks = [q["output_tokens"] or 0 for q in q_rows]
        total_in_tok = sum(in_toks)
        total_out_tok = sum(out_toks)
        total_tok = total_in_tok + total_out_tok
        p50_tok = (
            sorted(tin + tout for tin, tout in zip(in_toks, out_toks, strict=False))[n // 2]
            if n
            else 0
        )

        # Peak contexts & tool calls
        peaks = []
        tool_call_counts = []
        for q in q_rows:
            t = json.loads(q["trace_json"]) if q["trace_json"] else {}
            peaks.append(int(t.get("peak_input_tokens") or 0))
            tool_call_counts.append(int(t.get("tool_calls") or len(t.get("tool_calls_raw") or [])))

        # Capability breakdown
        cap_strict = {}
        for q in q_rows:
            cap = caps_by_qid.get(q["question_id"], "unknown")
            cap_strict.setdefault(cap, []).append(1 if q["verdict"] == "correct" else 0)

        cap_scores = {
            c: f"{sum(v)}/{len(v)} ({sum(v) / len(v):.1%})" for c, v in sorted(cap_strict.items())
        }

        matrix.append(
            {
                "run_id": rid,
                "model": r["answer_model"],
                "ctx_profile": cfg.get("context_profile"),
                "strict": strict,
                "weighted": weighted,
                "correct": correct,
                "partial": partial,
                "incorrect": incorrect,
                "unjudged": unjudged,
                "total_tokens": total_tok,
                "input_tokens": total_in_tok,
                "output_tokens": total_out_tok,
                "p50_tokens_per_q": p50_tok,
                "total_latency_s": round(total_latency_s, 1),
                "avg_latency_s": round(avg_latency_s, 2),
                "source_coverage": metrics.get("source", {}).get("source_coverage", 0.0),
                "recall_at_1": metrics.get("source", {}).get("recall_at_1", 0.0),
                "cap_scores": cap_scores,
            }
        )

    print(json.dumps(matrix, indent=2))


if __name__ == "__main__":
    main()
