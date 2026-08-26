#!/usr/bin/env python3
"""Bake judged oracle/closed_book blocks into the benchmark dataset.

Reads the machine-readable twin emitted by ``vesta bench verify``
(``benchmarks/verification/<date>-review.json``) and merges each question's
closed-book and oracle verdicts into ``vesta_bench_v2.json``.

The dataset content hash deliberately EXCLUDES ``oracle``/``closed_book`` (see
``src/vesta/eval/bench_dataset.py``), so baking reference points never
invalidates the comparability of pipeline runs. The blocks pin the *answer
model* that produced them — ``reference_points`` suppresses headroom when the
run's answer model differs from the baked ``oracle.model``.

Usage:
    uv run python scripts/bench_authoring/bake_reference.py \
        benchmarks/verification/2026-08-15-review.json \
        --dataset benchmarks/vesta_bench_v2.json [--write]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def bake(review: dict, questions: list[dict]) -> tuple[int, int]:
    """Merge review verdicts into the question list in place.

    Returns ``(baked, skipped)`` — skipped are questions the review judged
    ``unjudged`` (a judge failure must never overwrite a good reference point).
    """
    by_id = {q["id"]: q for q in questions}
    model = review.get("model", "")
    checked_at = review.get("generated", "")
    baked = skipped = 0
    for qid, entry in review.get("questions", {}).items():
        q = by_id.get(qid)
        if q is None:
            continue
        for block_name, src_key in (("closed_book", "closed_book"), ("oracle", "oracle")):
            src = entry.get(src_key, {})
            verdict = src.get("verdict")
            if verdict in (None, "", "unjudged"):
                skipped += 1
                continue
            q[block_name] = {
                "model": model,
                "verdict": verdict,
                "answer": src.get("answer", ""),
                "adjudicated_by": "vesta bench verify + bake_reference.py",
                "checked_at": checked_at,
                "reason": src.get("reason", ""),
            }
            baked += 1
    return baked, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("review", help="Path to benchmarks/verification/<date>-review.json")
    ap.add_argument(
        "--dataset",
        default="benchmarks/vesta_bench_v2.json",
        help="Dataset JSON to bake into (default benchmarks/vesta_bench_v2.json).",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="Write the dataset in place (default: dry-run, print a diff summary).",
    )
    args = ap.parse_args()

    review_path = Path(args.review)
    dataset_path = Path(args.dataset)
    try:
        review = json.loads(review_path.read_text(encoding="utf-8"))
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    questions = dataset.get("questions")
    if not isinstance(questions, list):
        print(f"error: {dataset_path} has no 'questions' array", file=sys.stderr)
        return 1

    baked, skipped = bake(review, questions)
    print(f"bake {review_path.name} → {dataset_path.name}: {baked} blocks, {skipped} skipped")
    if not args.write:
        print("dry-run — pass --write to persist.")
        return 0

    dataset_path.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {dataset_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
