"""Join the pipeline probe with the title probe and report the partition."""

import json

SCRATCH = "/tmp/claude-1000/-home-loki-git-vesta/6db70c51-7bb8-4044-99f1-2147091128a9/scratchpad"
with open(f"{SCRATCH}/round0.json", encoding="utf-8") as f:
    rows = json.load(f)
with open(f"{SCRATCH}/titles.json", encoding="utf-8") as f:
    titles = json.load(f)
title_miss = {m[0] for m in titles["misses"]}

n = len(rows)
print(f"pipeline probe: {n}/150 questions complete\n")


def hits(key, k):
    return sum(1 for r in rows if r[key][0] is not None and r[key][0] <= k)


print("=== gold article among returned source cards ===")
print(f"{'arm':<36} {'@1':>6} {'@5':>6} {'@10':>6} {'any':>6}")
for key, label in (
    ("A", "A  NL question / standard (today)"),
    ("B", "B  gold title / standard (oracle)"),
    ("D", "D  NL question / hybrid (dense)"),
):
    print(
        f"{label:<36} {hits(key, 1) / n:>6.2f} {hits(key, 5) / n:>6.2f} "
        f"{hits(key, 10) / n:>6.2f} {hits(key, 10**9) / n:>6.2f}"
    )

for key, label in (("B", "oracle title"), ("D", "hybrid/dense")):
    fixed = [r for r in rows if r["A"][0] is None and r[key][0] is not None]
    lost = [r for r in rows if r["A"][0] is not None and r[key][0] is None]
    print(f"\n{label:<14} rescues {len(fixed):>3}/{n} NL misses, loses {len(lost):>3}/{n} NL hits")

# Partition the A-misses by whether a zero-LLM title lookup could have found them.
a_miss = [r for r in rows if r["A"][0] is None]
closable = [r for r in a_miss if r["id"] not in title_miss]
residual = [r for r in a_miss if r["id"] in title_miss]
print(f"\n=== of the {len(a_miss)} questions today's Round 0 misses entirely ===")
print(f"  {len(closable):>3} reachable by zero-LLM entity->title lookup")
print(f"  {len(residual):>3} NOT reachable lexically (title absent from the question)")
d_saves = [r for r in residual if r["D"][0] is not None]
print(f"      of those {len(residual)}, hybrid/dense rescues {len(d_saves)}")
for r in residual[:10]:
    print(f"       - {r['id']} gold={r['gold']} D={r['D'][0]} :: {r['q'][:60]}")
