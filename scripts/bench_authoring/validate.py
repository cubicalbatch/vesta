#!/usr/bin/env python3
"""Validate authored benchmark questions: schema + fact-presence in the ZIM.

For each question, for each source article, extract the article text and check
that every sub_fact (mapped by source_index) literally appears in it. This is
the net that catches fabrication: a fact that is not in the article is reported.

Usage: uv run python scripts/bench_authoring/validate.py [file.json ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from libzim.reader import Archive

from vesta.zim.extract import extract_article
from vesta.zim.reader import read_entry_sync

ZIM = "data/zims/wikipedia_en_top_nopic_2026-06.zim"

REQUIRED_KEYS = {
    "id",
    "question",
    "capability",
    "difficulty",
    "slice",
    "level",
    "tags",
    "expected_behavior",
    "answer",
    "answer_detail",
    "sources",
    "sub_facts",
    "provenance",
    "closed_book",
    "oracle",
    "status",
}


def extract(archive: Archive, path: str) -> str:
    raw = read_entry_sync(archive, path)
    if raw.is_redirect:
        return ""
    return extract_article(raw.content, path=path, title=raw.title).text


def normalize(s: str) -> str:
    return " ".join(s.lower().split())


def main() -> int:  # noqa: PLR0912, PLR0915 — one branch per validation check
    files = sys.argv[1:] or sorted(str(p) for p in Path("data/bench_logs/authoring").glob("*.json"))
    archive = Archive(ZIM)
    seen_ids: set[str] = set()
    total = 0
    failures: list[str] = []

    for f in files:
        raw_json = json.loads(Path(f).read_text(encoding="utf-8"))
        if isinstance(raw_json, dict) and "questions" in raw_json:
            data = raw_json["questions"]
        elif isinstance(raw_json, list):
            data = raw_json
        else:
            print(f"FAIL {f}: not a JSON array or benchmark object", file=sys.stderr)
            failures.append(f"{f}: not an array or benchmark object")
            continue
        for q in data:
            total += 1
            qid = q.get("id", "?")
            missing = REQUIRED_KEYS - set(q)
            if missing:
                failures.append(f"{qid}: missing keys {sorted(missing)}")
            if qid in seen_ids:
                failures.append(f"{qid}: duplicate id")
            seen_ids.add(qid)
            level = q.get("level")
            if level not in (1, 2, 3):
                failures.append(f"{qid}: invalid level {level!r} (must be 1, 2, or 3)")
            cap = q.get("capability")
            srcs = q.get("sources") or []
            exp_beh = q.get("expected_behavior", "answer")

            if cap in ("multi_hop_cross_article", "comparative"):
                expected_src = 2
                if len(srcs) != expected_src:
                    failures.append(f"{qid}: {len(srcs)} sources (expected {expected_src})")
            elif cap == "adversarial_abstention":
                if exp_beh != "abstain":
                    failures.append(
                        f"{qid}: expected_behavior must be 'abstain' for adversarial_abstention"
                    )
            else:
                expected_src = 1
                if len(srcs) != expected_src:
                    failures.append(f"{qid}: {len(srcs)} sources (expected {expected_src})")

            subs = q.get("sub_facts") or []
            if cap not in ("buried_fact", "adversarial_abstention") and len(subs) < 2:
                failures.append(f"{qid}: {len(subs)} sub_facts (need >=2)")
            if cap in ("multi_hop_cross_article", "comparative") and len(srcs) >= 2:
                src_indices = {sf.get("source_index") for sf in subs}
                if not {0, 1}.issubset(src_indices):
                    failures.append(
                        f"{qid}: sub_facts must cover all sources ({src_indices} vs {0, 1})"
                    )

            for i, sf in enumerate(subs):
                idx = sf.get("source_index")
                fact = sf.get("fact", "")
                if not isinstance(idx, int) or idx < 0 or idx >= len(srcs):
                    failures.append(f"{qid}: sub_fact[{i}] bad source_index {idx!r}")
                    continue
                if not fact:
                    failures.append(f"{qid}: sub_fact[{i}] empty fact")
                    continue
                path = srcs[idx].get("article_path", "")
                text = extract(archive, path)
                if not text:
                    failures.append(f"{qid}: could not extract {path!r}")
                    continue
                if normalize(fact) not in normalize(text):
                    failures.append(f"{qid}: fact {fact!r} NOT in {path!r}")

    print(f"validated {total} questions across {len(files)} files")
    if failures:
        print(f"{len(failures)} FAILURES:")
        for x in failures:
            print("  -", x)
        return 1
    print("all facts present, schema OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
