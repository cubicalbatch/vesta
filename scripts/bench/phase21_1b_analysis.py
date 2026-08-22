#!/usr/bin/env python3
"""Do tool rounds earn their tokens? (offline trace analysis)

For the tool-round questions of a stored `agentic_pydantic` bench run: how often
does the final answer cite a card that first appeared AFTER a tool call?

Method (all offline, from `bench_question_results`):
- Tool-round question = `trace_json.stages` contains any `search`/`read_article`
  step beyond the leading `pre_seed`.
- Card numbering: `TurnResult.cards` is sorted by first-seen order (`agent_chat`
  `_TurnContext.turn_cards`, n assigned at discovery). The Round-0 pre-seed
  assigns n = 1..C where C = `stages[0].outputs.cards` (distinct articles in the
  pre-seed result). Cards discovered by the `search` tool get n > C — a card
  with n > C FIRST APPEARED after a tool call by construction.
- Final-answer citations = the `[n]` markers in `answer_text` (the agent prompt
  instructs bracketed citation numbers; `read_article`/`search` results tag
  every card with one). Filtered to 1..len(retrieved_paths).
- "cites_post_tool" = the answer cites at least one n > C.

Also recomputes the confound conditioning (gold-in-cards by <10k / ≥10k
cumulative input tokens) for the same runs, judged-only.

Usage:
    python3 scripts/bench/phase21_1b_analysis.py RUN_ID [RUN_ID ...] \
        [--db data/vesta.db] [--out benchmarks/phase21_1b_tool_round_citations.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
import sys
from pathlib import Path

CITE_RE = re.compile(r"\[(\d{1,2})\]")
TOOL_STAGES = ("search", "read_article")


def load_run(db: sqlite3.Connection, run_id: int) -> dict[str, sqlite3.Row]:
    rows = db.execute(
        "SELECT question_id, answer_text, verdict, source_hit_rank, input_tokens, "
        "output_tokens, retrieved_paths, trace_json, capability "
        "FROM bench_question_results WHERE run_id=?",
        (run_id,),
    ).fetchall()
    return {r["question_id"]: r for r in rows}


def stage_names(trace: dict) -> list[str]:
    return [str(s.get("name")) for s in trace.get("stages", [])]


def preseed_cards(trace: dict) -> int:
    stages = trace.get("stages", [])
    if not stages or stages[0].get("name") != "pre_seed":
        return 0
    return int(stages[0].get("outputs", {}).get("cards", 0) or 0)


def citations(answer: str, n_cards: int) -> set[int]:
    return {int(m) for m in CITE_RE.findall(answer) if 1 <= int(m) <= n_cards}


def analyze_run(db: sqlite3.Connection, run_id: int) -> dict[str, object]:
    rows = load_run(db, run_id)
    pin = db.execute(
        "SELECT label, system, answer_model, judge_model, subset_hash FROM bench_runs WHERE id=?",
        (run_id,),
    ).fetchone()
    judged = {q: r for q, r in rows.items() if r["verdict"] not in ("unjudged", "pending")}
    verdicts: dict[str, int] = {}
    for r in rows.values():
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1

    per_q: list[dict[str, object]] = []
    tool_q = search_q = 0
    cite_post = cite_post_search = 0
    no_citations = 0
    total_cites = post_cites = 0
    correct_by_group: dict[str, dict[str, int]] = {
        "cites_post_tool": {"n": 0, "correct": 0},
        "pre_seed_only": {"n": 0, "correct": 0},
    }
    for qid, r in sorted(rows.items()):
        trace = json.loads(r["trace_json"]) if r["trace_json"] else {}
        names = stage_names(trace)
        is_tool = any(n in TOOL_STAGES for n in names[1:] if names)
        paths = json.loads(r["retrieved_paths"]) if r["retrieved_paths"] else []
        if not is_tool:
            continue
        tool_q += 1
        has_search = any(n == "search" for n in names[1:])
        if has_search:
            search_q += 1
        c = preseed_cards(trace)
        cites = citations(r["answer_text"] or "", len(paths))
        post = {n for n in cites if n > c}
        total_cites += len(cites)
        post_cites += len(post)
        if not cites:
            no_citations += 1
        group = "cites_post_tool" if post else "pre_seed_only"
        if r["verdict"] == "correct":
            correct_by_group[group]["correct"] += 1
        if r["verdict"] in ("correct", "partial", "incorrect"):
            correct_by_group[group]["n"] += 1
        if post:
            cite_post += 1
            if has_search:
                cite_post_search += 1
        per_q.append(
            {
                "qid": qid,
                "capability": r["capability"],
                "stages": names,
                "preseed_cards": c,
                "n_cards": len(paths),
                "cited": sorted(cites),
                "cited_post_tool": sorted(post),
                "verdict": r["verdict"],
                "input_tokens": r["input_tokens"],
                "source_hit_rank": r["source_hit_rank"],
            }
        )

    # ── confound conditioning: gold-in-cards by turn-size split ──
    def cond(rowsel: dict[str, sqlite3.Row]) -> dict[str, object]:
        out: dict[str, object] = {}
        for label, lo, hi in (("<10k", 0, 10_000), (">=10k", 10_000, 10**9)):
            sel = {
                q: r
                for q, r in rowsel.items()
                if lo <= (r["input_tokens"] or 0) < hi and r["verdict"] != "unjudged"
            }
            gold = [r for r in sel.values() if r["source_hit_rank"] is not None]
            gold_correct = [r for r in gold if r["verdict"] == "correct"]
            ranks = [r["source_hit_rank"] for r in gold]
            out[label] = {
                "n": len(sel),
                "gold_in_cards_pct": round(100 * len(gold) / len(sel), 1) if sel else None,
                "mean_gold_rank": round(statistics.fmean(ranks), 1) if ranks else None,
                "correct_given_gold_pct": (
                    round(100 * len(gold_correct) / len(gold), 1) if gold else None
                ),
            }
        return out

    return {
        "run_id": run_id,
        "label": pin["label"] if pin else "",
        "system": pin["system"] if pin else "",
        "answer_model": pin["answer_model"] if pin else "",
        "judge_model": pin["judge_model"] if pin else "",
        "subset_hash": pin["subset_hash"] if pin else "",
        "verdicts": verdicts,
        "judged_n": len(judged),
        "total_questions": len(rows),
        "tool_round_questions": tool_q,
        "tool_round_with_search": search_q,
        "tool_round_read_only": tool_q - search_q,
        "answers_with_no_citation": no_citations,
        "cite_post_tool_questions": cite_post,
        "cite_post_tool_pct_of_tool_round": round(100 * cite_post / tool_q, 1) if tool_q else None,
        "cite_post_tool_pct_of_search_round": (
            round(100 * cite_post_search / search_q, 1) if search_q else None
        ),
        "total_citations": total_cites,
        "post_tool_citations": post_cites,
        "post_tool_citation_share_pct": (
            round(100 * post_cites / total_cites, 1) if total_cites else None
        ),
        "correct_rate_by_group": {
            k: {
                "judged_n": v["n"],
                "correct_pct": round(100 * v["correct"] / v["n"], 1) if v["n"] else None,
            }
            for k, v in correct_by_group.items()
        },
        "confound_conditioning": cond(rows),
        "per_question": per_q,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_ids", nargs="+", type=int)
    ap.add_argument("--db", default="data/vesta.db")
    ap.add_argument("--out", default="benchmarks/phase21_1b_tool_round_citations.json")
    args = ap.parse_args()

    db = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    runs = [analyze_run(db, rid) for rid in args.run_ids]
    payload = {
        "analysis": "phase21-1b-tool-round-citations",
        "method": {
            "tool_round": "stages beyond the leading pre_seed contain search/read_article",
            "preseed_card_count": "stages[0].outputs.cards of the pre_seed step",
            "citation": r"regex \[(\d{1,2})\] on answer_text, filtered to 1..len(retrieved_paths)",
            "post_tool_card": "cited card number n > preseed card count (cards numbered at "
            "first discovery; pre-seed cards occupy 1..C)",
            "confound": "judged-only; gold in cards = source_hit_rank IS NOT NULL; "
            "split on cumulative input_tokens at 10k",
        },
        "runs": runs,
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    for r in runs:
        print(
            f"run {r['run_id']} ({r['label']}): tool-round {r['tool_round_questions']}/"
            f"{r['total_questions']} (search {r['tool_round_with_search']}, read-only "
            f"{r['tool_round_read_only']}); answers citing >=1 post-tool card: "
            f"{r['cite_post_tool_questions']} ({r['cite_post_tool_pct_of_tool_round']}% of "
            f"tool-round, {r['cite_post_tool_pct_of_search_round']}% of search-round); "
            f"citation instances post-tool {r['post_tool_citations']}/{r['total_citations']} "
            f"({r['post_tool_citation_share_pct']}%)"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
