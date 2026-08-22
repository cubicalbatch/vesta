"""Measure Round-0 retrieval on natural-language questions vs. shaped/dense queries.

Arms, all LLM-free, over bench v2 questions that name a gold Wikipedia article:
  A  raw NL question, `standard`  — exactly what Round 0 (`search_exact`) does today
  B  gold article title, `standard` — oracle ceiling for any query-shaping step
  D  raw NL question, `hybrid`   — the dense alternative to an LLM rewrite

Reports rank of the gold article path in `result.cards` per arm.
"""

from __future__ import annotations

import asyncio
import json
import sys

sys.path.insert(0, "/home/loki/git/vesta/src")

from vesta import config
from vesta.cli import _open_runtime
from vesta.config.capabilities import compute_capabilities
from vesta.retrieval.contracts import Scope as RetScope
from vesta.retrieval.pipeline import Deps, NoCandidatesError, run_pipeline
from vesta.retrieval.profiles import load_profile
from vesta.vectors import get_store

SCRATCH = "/tmp/claude-1000/-home-loki-git-vesta/6db70c51-7bb8-4044-99f1-2147091128a9/scratchpad"
DATASET = "/home/loki/git/vesta/benchmarks/vesta_bench_v2.json"
ARCHIVE = "wikipedia_en_top_nopic_2026-06.zim"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def norm(p: str) -> str:
    return p.strip().strip("/").replace(" ", "_").lower()


async def rank_of(state, profile, query: str, gold: str) -> tuple[int | None, float]:
    deps = Deps(
        archives=state.registry,
        settings=config.snapshot(),
        capabilities=compute_capabilities(),
        semaphore=asyncio.Semaphore(8),
        encoders=state.encoders,
        vectors=get_store(),
    )
    try:
        result = await run_pipeline(profile=profile, query=query, scope=RetScope(), deps=deps)
    except NoCandidatesError:
        return None, 0.0
    top = result.confidence.top_score if result.confidence else 0.0
    for i, c in enumerate(result.cards):
        if norm(c.path) == norm(gold):
            return i + 1, (top or 0.0)
    return None, (top or 0.0)


async def main() -> None:
    with open(DATASET, encoding="utf-8") as f:
        data = json.load(f)
    qs = [
        q
        for q in data["questions"]
        if q.get("expected_behavior") == "answer"
        and any(s.get("zim") == ARCHIVE and s.get("required") for s in q.get("sources", []))
    ][:LIMIT]
    print(f"questions={len(qs)}", flush=True)

    async with _open_runtime("/home/loki/git/vesta/data") as state:
        standard = load_profile("standard")
        hybrid = load_profile("hybrid")
        rows = []
        for n, q in enumerate(qs, 1):
            src = next(s for s in q["sources"] if s.get("zim") == ARCHIVE and s.get("required"))
            gold = src["article_path"]
            title = src.get("article_title") or gold
            nl = q["question"]

            a = await rank_of(state, standard, nl, gold)
            b = await rank_of(state, standard, title, gold)
            d = await rank_of(state, hybrid, nl, gold)
            rows.append(
                {"id": q["id"], "q": nl, "gold": gold, "title": title, "A": a, "B": b, "D": d}
            )
            print(f"[{n}/{len(qs)}] {q['id']} A={a[0]} B={b[0]} D={d[0]} :: {nl[:64]}", flush=True)
            with open(f"{SCRATCH}/round0.json", "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=1)

        def hits(key, k):
            return sum(1 for r in rows if r[key][0] is not None and r[key][0] <= k)

        n = len(rows)
        print("\n=== recall of the gold article among returned cards ===")
        print(f"{'arm':<34} {'@1':>6} {'@5':>6} {'@10':>6} {'any':>6}")
        for key, label in (
            ("A", "A  NL question / standard (today)"),
            ("B", "B  gold title / standard (oracle)"),
            ("D", "D  NL question / hybrid (dense)"),
        ):
            print(
                f"{label:<34} {hits(key, 1) / n:>6.2f} {hits(key, 5) / n:>6.2f} "
                f"{hits(key, 10) / n:>6.2f} {hits(key, 10**9) / n:>6.2f}"
            )

        for key, label in (("B", "oracle title"), ("D", "hybrid/dense")):
            fixed = [r for r in rows if r["A"][0] is None and r[key][0] is not None]
            lost = [r for r in rows if r["A"][0] is not None and r[key][0] is None]
            print(f"\n{label}: rescues {len(fixed)}/{n} NL misses, loses {len(lost)}/{n} NL hits")


asyncio.run(main())
