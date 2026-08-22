#!/usr/bin/env bash
# Level-3 retrieval-only baselines.
#
# Two cells, back to back so the corpus state is identical:
#   A. all enabled archives (no --scope)  -> 1.07M vectors in the dense scan
#   B. wikipedia_en_top only              -> 628k vectors
#
# retrieval_only makes no LLM calls and needs no judge (cli.py:618-621), so this
# is pure pipeline latency + source metrics. Persisted to bench_runs for
# cross-run comparison; the label prefix is an env var so other runs can
# stamp their own labels with the same harness.
#
# Usage: [LABEL_PREFIX=l3-baseline] [REPEATS=2] ./scripts/vesta_l3_baseline.sh
set -euo pipefail

# Repo root derived from this script's own location, so it runs from any cwd.
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."

LABEL_PREFIX="${LABEL_PREFIX:-l3-baseline}"
REPEATS="${REPEATS:-2}"          # 2 gives run-to-run variance; set REPEATS=1 to halve the time
LEVEL=3
PROFILE=hybrid
WIKI_SCOPE=wikipedia_en_top_nopic_2026-06.zim

# ── Preflight ──────────────────────────────────────────────────────────────
# Two files in the working tree don't parse right now; `vesta bench` imports
# vesta.db, so it would die on a SyntaxError several minutes in.
echo "==> preflight: syntax check"
if ! .venv/bin/python -c '
import ast, pathlib, sys
bad = []
for p in pathlib.Path("src").rglob("*.py"):
    try:
        ast.parse(p.read_text())
    except SyntaxError as e:
        bad.append(f"{p}:{e.lineno}: {e.msg}")
if bad:
    print("ABORT — these files do not parse:", file=sys.stderr)
    for b in bad:
        print("  " + b, file=sys.stderr)
    sys.exit(1)
print("    ok, src/ parses clean")
'; then
    echo "Fix the files above, then re-run this script." >&2
    exit 1
fi

GIT_SHA="$(git rev-parse --short HEAD)"
DIRTY=""
git diff --quiet || DIRTY="-dirty"
STAMP="$(date +%Y%m%d-%H%M)"

# bench_runs.config_json does NOT record which archives were enabled (the schema
# comment claims "archive checksums" but the writer never populates them) — that
# is why historical runs 30 and 78 have byte-identical configs yet differ 15x in
# dense candidate count. Stamp the corpus state into the label so these runs stay
# interpretable after the LanceDB swap, and drop a sidecar snapshot next to them.
N_ARCH="$(sqlite3 data/vesta.db 'SELECT count(*) FROM zims WHERE enabled=1;')"
N_VEC="$(sqlite3 data/vesta.db 'SELECT count(*) FROM chunks;')"
# Index depth is the variable that matters most (depth 1 = 1 vector/article,
# depth 2 = ~7x that), so it goes in the label. Run this script once now for the
# depth-2 baseline, then again after re-indexing to depth 1 -- the depth-1 pair
# is the one the LanceDB gate compares against.
DEPTHS="$(sqlite3 data/vesta.db "SELECT group_concat(DISTINCT index_depth) FROM zims WHERE enabled=1 AND index_depth >= 1;")"
CORPUS="d${DEPTHS//,/+}-${N_ARCH}arch-${N_VEC}vec"
mkdir -p benchmarks/results
sqlite3 -json data/vesta.db \
  "SELECT z.id, z.name, z.filename, z.index_depth, z.index_status, count(c.id) AS chunks
     FROM zims z LEFT JOIN chunks c ON c.zim_id = z.id
    WHERE z.enabled = 1 GROUP BY z.id ORDER BY z.id;" \
  > "benchmarks/results/corpus-state-${STAMP}.json"
echo "==> corpus snapshot: benchmarks/results/corpus-state-${STAMP}.json (${CORPUS})"
echo "==> NOTE: depth in this run = ${DEPTHS}. Depth 1 is the production target."
echo "==> git ${GIT_SHA}${DIRTY}, repeats=${REPEATS}, level=${LEVEL}, profile=${PROFILE}, label-prefix=${LABEL_PREFIX}"
echo "==> archives enabled:"
sqlite3 -column data/vesta.db \
  "SELECT id, name, index_depth, index_status FROM zims WHERE enabled=1 ORDER BY id;"
echo "==> vectors: $(sqlite3 data/vesta.db 'SELECT count(*) FROM chunks;') chunks total"
echo

# ── A. all archives ────────────────────────────────────────────────────────
echo "==> [A] level-3 retrieval_only, ALL archives (unscoped)"
uv run vesta bench run \
    --system retrieval_only \
    --profile "${PROFILE}" \
    --level "${LEVEL}" \
    --repeats "${REPEATS}" \
    --concurrency 1 \
    --report both \
    --label "${LABEL_PREFIX}-l3-all-archives ${CORPUS} ${GIT_SHA}${DIRTY} ${STAMP}"
echo

# ── B. wikipedia only ──────────────────────────────────────────────────────
echo "==> [B] level-3 retrieval_only, wikipedia_en_top only"
uv run vesta bench run \
    --system retrieval_only \
    --profile "${PROFILE}" \
    --level "${LEVEL}" \
    --scope "${WIKI_SCOPE}" \
    --repeats "${REPEATS}" \
    --concurrency 1 \
    --report both \
    --label "${LABEL_PREFIX}-l3-wikipedia-only ${CORPUS} ${GIT_SHA}${DIRTY} ${STAMP}"
echo

# ── Summary: the numbers the comparison is graded against ──────────────────
echo "==> per-stage latency + dense candidate counts for the ${LABEL_PREFIX} runs"
.venv/bin/python - <<'PY'
import json, os, sqlite3, statistics
from collections import defaultdict

prefix = os.environ.get("LABEL_PREFIX", "l3-baseline")
c = sqlite3.connect("data/vesta.db")
runs = c.execute(
    "SELECT id, label FROM bench_runs WHERE label LIKE ? ORDER BY id",
    (f"{prefix}-l3-%",),
).fetchall()
if not runs:
    print(f"  (no {prefix}-l3-* runs found)")
for run_id, label in runs:
    rows = c.execute(
        "SELECT trace_json, latency_ms FROM bench_question_results "
        "WHERE run_id=? AND trace_json IS NOT NULL", (run_id,)
    ).fetchall()
    if not rows:
        print(f"\n  run {run_id} [{label}]: no traces")
        continue
    per, tot, dense = defaultdict(list), [], []
    for tj, lat in rows:
        try:
            t = json.loads(tj)
        except Exception:
            continue
        tot.append(lat)
        agg = defaultdict(float)
        for s in t.get("stages", []):
            agg[f"{s.get('name')}/{s.get('component')}"] += s.get("duration_ms") or 0
            if s.get("component") == "vector_knn":
                dense.append((s.get("outputs") or {}).get("candidate_count", 0))
        for k, v in agg.items():
            per[k].append(v)
    print(f"\n  run {run_id} [{label}]  n={len(rows)}  mean total={statistics.mean(tot):.0f} ms")
    if dense:
        print(f"    dense candidates: mean={statistics.mean(dense):.1f} "
              f"min={min(dense)} max={max(dense)}")
    for k, v in sorted(per.items(), key=lambda kv: -statistics.mean(kv[1]))[:6]:
        v = sorted(v)
        print(f"    {k:42s} p50={statistics.median(v):8.1f} ms  p95={v[int(len(v)*.95)]:8.1f} ms")
PY

echo
echo "==> done. Compare later with: uv run vesta bench compare <run_a> <run_b>"
