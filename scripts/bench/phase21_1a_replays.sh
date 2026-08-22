#!/bin/sh
# Answer-only replays over the frozen snapshot, N ∈ {6,4,3,2}.
# One run at a time (single writer against data/vesta.db); pinned models.
# Usage: sh scripts/bench/phase21_1a_replays.sh [SNAPSHOT.json]
set -e
SNAP="${1:-benchmarks/context/phase21-1a-snapshot.json}"
for N in 6 4 3 2; do
  echo "=== replay N=$N ($(date -u +%H:%M:%S)) ==="
  uv run vesta bench run \
    --from-context "$SNAP" \
    --context-passages "$N" \
    --model qwen3.5-4b@q4_k_s \
    --endpoint http://desktop.onoz.cc:1234/v1 \
    --judge-model google/gemma-4-12b-qat \
    --level 3 \
    --capability buried_fact \
    --capability multi_fact_same_article \
    --capability multi_hop_cross_article \
    --label "phase21-1a-n$N" \
    --report both > "/tmp/phase21-1a-n$N.log" 2>&1
  tail -3 "/tmp/phase21-1a-n$N.log"
done
echo "=== all replays done ($(date -u +%H:%M:%S)) ==="
