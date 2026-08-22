#!/usr/bin/env python3
"""Evidence-tier analysis — evidence tiers, tool-call counts, answer-shape rates.

LLM-free. Reads a retrieval context snapshot (``--snapshot``, written by
``vesta bench run --system retrieval_only --save-context``) plus one or more
persisted ``bench_runs`` ids, and prints:

* how many tool stages each run actually made (``rounds`` in
  ``bench_question_results`` is always 0 — trap 1);
* the evidence tiers A/B/C/D: is the gold answer's distinctive vocabulary in
  the pre-seed the model saw, deeper in the retrieved set, or nowhere;
* the answer-shape pathology rates (preface / ``archive-N`` leak / missing
  citation / shown arithmetic / claimed-not-found).

Usage::

    uv run python scripts/bench/phase22_evidence_tiers.py 80 88
    uv run python scripts/bench/phase22_evidence_tiers.py 90 --preseed-n 6
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
import statistics

DB_PATH = "data/vesta.db"
DATASET = "benchmarks/vesta_bench_v2.json"
SNAPSHOT = "benchmarks/context/hybrid_200_wikipedia.json"

_STOPWORDS = (
    "a an the and or of to in on at by for with from into is are was were be been being do "
    "does did have has had this that about"
)
STOP = frozenset(_STOPWORDS.split(" "))

#: Word characters plus the punctuation Wikipedia bodies use inside tokens
#: (apostrophes and dashes, straight and typographic).
WORD = re.compile("\\b[\\w.'\u2019\u2013-]+\\b")

PREFACE = re.compile(
    r"^\s*(the question asks|based on|according to the (provided )?(sources|excerpts|passages|"
    r"information)|after review|the information (about|regarding)|from the (provided )?sources|"
    r"looking at)",
    re.I,
)
NOTFOUND = re.compile(
    r"(not (present|available|found|mentioned|specified|contained)|cannot (determine|be "
    r"determined|find)|does not (specify|mention|contain)|no information|unable to "
    r"(determine|find))",
    re.I,
)
CALC = re.compile(
    r"(subtract|calculate|let'?s compute|step \d|therefore, the answer|=\s*-?\d)", re.I
)
ARCHIVE = re.compile(r"archive-\d")


def _toks(text: str) -> set[str]:
    out = set()
    for m in re.findall(WORD, text):
        low = m.lower().strip(".,'")
        if len(low) >= 3 and low not in STOP:
            out.add(low)
    return out


def _tier(question: dict, entry: dict, preseed_n: int, cap: int) -> str | None:
    """A/B/C/D by where the answer's distinctive vocabulary lives."""
    key = _toks(question.get("answer", "")) - _toks(question["question"])
    if not key:
        return None
    passages = entry.get("passages") or []
    preseed = "\n".join(p["text"][:cap] for p in passages[:preseed_n]).lower()
    everything = "\n".join(p["text"] for p in passages).lower()
    in_seed = sum(1 for k in key if k in preseed) / len(key)
    in_all = sum(1 for k in key if k in everything) / len(key)
    if in_seed >= 0.999:
        return "A. in the pre-seed already"
    if in_all >= 0.999:
        return "B. in the retrieved set, deeper passages"
    if in_all >= 0.7:
        return "C. mostly in the retrieved set"
    return "D. needs a full-article read or a new search"


def _report_run(conn, rid: int, rows: dict) -> None:
    """Print one run's verdict, tool-use, latency and answer-shape summary."""
    meta = conn.execute("SELECT answer_model, label FROM bench_runs WHERE id=?", (rid,)).fetchone()
    n = len(rows)
    verdicts = collections.Counter(r["verdict"] for r in rows.values())
    strict = verdicts["correct"] / n
    weighted = (verdicts["correct"] + 0.5 * verdicts["partial"]) / n

    tool_qs = 0
    requests = []
    pre_ms = []
    for r in rows.values():
        trace = json.loads(r["trace_json"] or "{}")
        stages = trace.get("stages") or []
        if any(s.get("name") != "pre_seed" for s in stages):
            tool_qs += 1
        requests.append(int(trace.get("requests") or 0))
        pre_ms.append(sum(s.get("duration_ms", 0) for s in stages if s.get("name") == "pre_seed"))

    shapes: collections.Counter = collections.Counter()
    for r in rows.values():
        text = r["answer_text"] or ""
        if PREFACE.search(text):
            shapes["preface"] += 1
        if ARCHIVE.search(text):
            shapes["archive-N leak"] += 1
        if not re.search(r"\[\d+\]", text):
            shapes["no [n] citation"] += 1
        if CALC.search(text):
            shapes["shows arithmetic"] += 1
        if NOTFOUND.search(text):
            shapes["claims not found"] += 1

    lat = [r["latency_ms"] or 0 for r in rows.values()]
    toks = [(r["input_tokens"] or 0) + (r["output_tokens"] or 0) for r in rows.values()]
    print(f"\n=== run {rid} — {meta['answer_model']} — {meta['label']}")
    print(f"  n={n}  strict={strict:.1%}  weighted={weighted:.1%}  {dict(verdicts)}")
    print(
        f"  questions with >=1 TOOL stage: {tool_qs}/{n}"
        f"   (requests mean {statistics.mean(requests):.2f}, max {max(requests)})"
    )
    print(
        f"  latency p50 {statistics.median(lat) / 1000:.1f}s"
        f"  (pre_seed p50 {statistics.median(pre_ms) / 1000:.1f}s)"
        f"   tokens p50 {statistics.median(toks):.0f}, total {sum(toks)}"
    )
    for k, v in shapes.most_common():
        print(f"    {k:22s} {v:4d} ({v / n:.0%})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=int, help="bench_runs ids to analyse")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--snapshot", default=SNAPSHOT)
    ap.add_argument("--preseed-n", type=int, default=6, help="answer.agent.preseed_passages")
    ap.add_argument(
        "--preseed-cap", type=int, default=2400, help="answer.agent.preseed_passage_max_chars"
    )
    args = ap.parse_args()

    with open(args.dataset) as fh:
        questions = {q["id"]: q for q in json.load(fh)["questions"]}
    with open(args.snapshot) as fh:
        snapshot = json.load(fh)["questions"]

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    tiers = {
        qid: t
        for qid, q in questions.items()
        if qid in snapshot
        and (t := _tier(q, snapshot[qid], args.preseed_n, args.preseed_cap)) is not None
    }

    per_run: dict[int, dict[str, dict]] = {}
    for rid in args.runs:
        rows = {
            r["question_id"]: dict(r)
            for r in conn.execute("SELECT * FROM bench_question_results WHERE run_id=?", (rid,))
        }
        per_run[rid] = rows
        if not rows:
            print(f"run {rid}: no question rows")
            continue
        _report_run(conn, rid, rows)

    print(f"\n=== evidence tiers (pre-seed = top-{args.preseed_n} x {args.preseed_cap} chars)")
    header = f"{'tier':42s} {'n':>4s}" + "".join(f"  run {r:<9d}" for r in args.runs)
    print(header)
    counts = collections.Counter(tiers.values())
    for tier in sorted(counts):
        ids = [qid for qid, t in tiers.items() if t == tier]
        cells = ""
        for rid in args.runs:
            rows = per_run.get(rid, {})
            # A run may cover only a level subset — score it on the questions
            # it actually ran, never on the full tier denominator.
            seen = [qid for qid in ids if qid in rows]
            got = sum(1 for qid in seen if rows[qid]["verdict"] == "correct")
            cells += (
                f"  {got:>3d}/{len(seen):<3d} {got / len(seen) if seen else 0:>4.0%}"
                if seen
                else f"  {'—':>12s}"
            )
        print(f"{tier:42s} {len(ids):4d}{cells}")


if __name__ == "__main__":
    main()
