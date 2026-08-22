"""Zero-LLM counter-experiment: can entity-shaped title lookup find the gold article?

Round 0 hands `title_suggest` the whole stopword-stripped question as a *title
prefix* ("old napoleon became emperor"), which prefix-matches nothing. This
probe asks: if we instead fed it capitalized/distinctive spans lifted from the
raw question — no LLM, ~24ms per lookup — would the gold article surface?

Arms (all against `Archive.suggest` only; no pipeline, no ONNX):
  T0  whole stopword-stripped question as prefix   (what Round 0 does today)
  T1  capitalized noun-phrase spans from the raw question
  T2  T1 + distinctive single terms (>=4 chars, non-stopword)
"""

from __future__ import annotations

import asyncio
import json
import re
import string
import sys

sys.path.insert(0, "/home/loki/git/vesta/src")

from vesta.cli import _open_runtime
from vesta.zim.query import DEFAULT_STOPWORDS

SCRATCH = "/tmp/claude-1000/-home-loki-git-vesta/6db70c51-7bb8-4044-99f1-2147091128a9/scratchpad"
DATASET = "/home/loki/git/vesta/benchmarks/vesta_bench_v2.json"
ARCHIVE = "wikipedia_en_top_nopic_2026-06.zim"
STOP = frozenset(DEFAULT_STOPWORDS)
K = 10

# Capitalized runs not at sentence start, plus quoted spans.
_CAP_RUN = re.compile(
    r"\b([A-Z][\w'-]*(?:\s+(?:of|de|van|der|the)\s+[A-Z][\w'-]*|\s+[A-Z][\w'-]*)*)"
)


def norm(p: str) -> str:
    return p.strip().strip("/").replace(" ", "_").lower()


def stripped_question(q: str) -> str:
    toks = [w for w in q.lower().replace("?", " ").replace('"', " ").split() if w not in STOP]
    return " ".join(toks)


def cap_spans(q: str) -> list[str]:
    words = q.split()
    out: list[str] = []
    for m in _CAP_RUN.finditer(q):
        span = m.group(1).strip(string.punctuation + " ")
        # Drop a leading interrogative that only got capitalized by sentence start.
        if span and words and span.split()[0] == words[0].strip(string.punctuation):
            rest = " ".join(span.split()[1:])
            span = rest
        if len(span) >= 3 and span.lower() not in STOP:
            out.append(span)
    seen: set[str] = set()
    uniq = []
    for s in out:
        if s.lower() not in seen:
            seen.add(s.lower())
            uniq.append(s)
    return uniq[:4]


def distinctive(q: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for w in q.split():
        c = w.strip(string.punctuation)
        if len(c) >= 4 and c.lower() not in STOP and c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out[:5]


async def main() -> None:
    with open(DATASET, encoding="utf-8") as f:
        data = json.load(f)
    qs = [
        q
        for q in data["questions"]
        if q.get("expected_behavior") == "answer"
        and any(s.get("zim") == ARCHIVE and s.get("required") for s in q.get("sources", []))
    ]
    print(f"questions={len(qs)}", flush=True)

    async with _open_runtime("/home/loki/git/vesta/data") as state:
        all_a = state.registry.enabled()
        archives = [a for a in all_a if ARCHIVE in str(getattr(a, "filename", ""))]
        if not archives:
            print("archive not found; available:", [str(getattr(a, "filename", a)) for a in all_a])
            return
        arch = archives[0]

        async def hit(prefixes: list[str], gold: str) -> bool:
            for p in prefixes:
                try:
                    paths = await arch.suggest(p, K)
                except Exception:
                    continue
                if any(norm(x) == norm(gold) for x in paths):
                    return True
            return False

        tot = {"T0": 0, "T1": 0, "T2": 0}
        misses = []
        for n, q in enumerate(qs, 1):
            src = next(s for s in q["sources"] if s.get("zim") == ARCHIVE and s.get("required"))
            gold = src["article_path"]
            nl = q["question"]
            t0 = await hit([stripped_question(nl)], gold)
            caps = cap_spans(nl)
            t1 = await hit(caps, gold)
            t2 = t1 or await hit(distinctive(nl), gold)
            tot["T0"] += t0
            tot["T1"] += t1
            tot["T2"] += t2
            if not t2:
                misses.append((q["id"], gold, nl))
            if n % 25 == 0:
                print(f"  ...{n}/{len(qs)}", flush=True)

        n = len(qs)
        print(f"\n=== gold article in title_suggest top-{K} (zero LLM) ===")
        print(f"T0  whole question as prefix (today)      {tot['T0']}/{n} = {tot['T0'] / n:.2f}")
        print(f"T1  capitalized spans                     {tot['T1']}/{n} = {tot['T1'] / n:.2f}")
        print(f"T2  spans + distinctive terms             {tot['T2']}/{n} = {tot['T2'] / n:.2f}")
        print(f"\nstill missed by T2: {len(misses)}")
        for _i, (qid, gold, nl) in enumerate(misses[:12]):
            print(f"  - {qid} gold={gold} :: {nl[:70]}")
        with open(f"{SCRATCH}/titles.json", "w", encoding="utf-8") as f:
            json.dump({"totals": tot, "n": n, "misses": misses}, f, indent=1)


asyncio.run(main())
