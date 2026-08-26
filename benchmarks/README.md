# Unified benchmark (`vesta bench`)

Vesta has **one benchmark** with **one dataset**, **one runner**, **one
persistence model**, and **one report**. Four older, overlapping measurement
tools and seven overlapping question files were collapsed into this single
surface so that "did this change help?" has one defensible answer.

Question authoring and dataset extension live in `scripts/bench_authoring/`.

## The dataset — `benchmarks/vesta_bench_v2.json`

A single capability-tagged question set (200 authored, verified questions
all from the pinned Wikipedia archive; the benchmark is Wikipedia-only). Every
question carries a stable slug id (not an ordinal), a `sources[]` block (one or
more gold articles), `sub_facts[]` for compositional questions, and —
critically — a `closed_book` block (the floor: the model with no context) and an
`oracle` block (the ceiling: the model given the gold articles), both baked in by
`vesta bench verify`.

Capabilities:
- **Algorithmic baseline (150 Qs):** `buried_fact` (50), `multi_fact_same_article` (50), `multi_hop_cross_article` (50).
- **Realistic scenarios (50 Qs):** `concept_lookup` (10, tip-of-the-tongue / vocabulary mismatch), `comparative` (10, true multi-entity comparison), `procedural` (10, how-to / emergency first aid), `complex_explanation` (10, multi-section synthesis), `adversarial_abstention` (10, false premise / ungrounded abstention).

Every question carries a `level: 1|2|3` tier; `--level L` keeps `q.level <= L`
(cumulative), and every tier is stratified across all 8 capabilities:

| Capability | L1 (smoke) | L2 (default) | L3 (release) |
|---|---:|---:|---:|
| `buried_fact` / `multi_fact_same_article` / `multi_hop_cross_article` | 10 each | 25 each | 50 each |
| the 5 scenario capabilities | 4 each | 5 each | 10 each |
| **Total** | **50** | **100** | **200** |

The dataset hash is a `sha256` over the questions' identity fields (`id |
question | answer | expected_behavior | sorted(source article_paths) |
sorted(sub_fact texts)`), deliberately EXCLUDING `oracle`/`closed_book`/`tags`/
`provenance` — so re-running the oracle pass never invalidates the
comparability of pipeline runs. A `--slice`/`--limit` run records both the
full-set hash and a `subset_hash` so filtered runs are never silently compared
to full runs.

## The one runner — `uv run vesta bench run`

The matrix-capable end-to-end benchmark. For each cell in
`systems × profiles × models` it runs the real answer pipeline (in-process,
the same `iter_answer_events` path the API uses), scores **retrieval
deterministically** (did search surface the gold articles?) and **answers with
an LLM judge** (never lexical — `_lexical_verdict` is deleted), and references
the run against the dataset's ceiling (`oracle`) and floor (`closed_book`).

Key flags:

```
--system NAME          repeatable; matrix axis (6 systems)
--profile NAME         repeatable; matrix axis
--model ID             repeatable; matrix axis (answer model)
--endpoint URL / --api-key      per-run LLM override
--judge-model / --judge-endpoint / --judge-api-key
--dataset PATH         default benchmarks/vesta_bench_v2.json
--slice core|all          default core ('all' = same set; Wikipedia-only)
--level 1|2|3          question tier (default 2): cumulative 50/100/200
--capability NAME      repeatable filter
--difficulty easy|medium|hard   repeatable filter
--limit N              smoke runs
--repeats N            variance measurement (default 1)
--concurrency N        pipeline questions in flight (default 1; keep it there —
                       above it, reported latency is contended)
--judge-concurrency N  judge calls in flight (default 4; clamped to 1 when the
                       judge shares the answer endpoint)
--scope ZIM            restrict retrieval scope
--label TEXT           human label on the run
--save-context PATH    (retrieval_only) dump retrieved passages for replay
--from-context PATH    (answer_only) replay a saved snapshot
--oracle-context       (answer_only) replay gold articles
--no-persist           report only
--report md|json|both  written to benchmarks/results/
--baseline RUN_ID      print the compare table at the end
```

Examples:

```sh
# Smoke test (first 2 questions, no persistence):
uv run vesta bench run --limit 2 --no-persist

# Matrix — two answer models, default systems, core slice:
uv run vesta bench run --model lmstudio/unsloth/qwen3.5-4b --model lmstudio/openai/gpt-oss-20b

# Retrieval-only (zero LLM calls — source metrics only), 50-question smoke tier:
uv run vesta bench run --system retrieval_only --level 1

# Override the judge for this run only:
uv run vesta bench run --judge-model gpt-4o --judge-endpoint https://api.openai.com/v1
```

The 6 registered systems (`SYSTEM_CLASSES`): `retrieval_only`, `sources_only`,
`agentic_pydantic`, `oracle`, `closed_book`, `answer_only`.

### Three decoupled modes

- **Retrieval-only** (`--system retrieval_only`) — the search + rerank pipeline
  only, **zero LLM calls** (the judge is bypassed). Source metrics
  (`recall@k`, `source_coverage`, `mrr`, `retrieved_precision`) plus per-question
  and per-stage latency. ~10 s for 200 questions.
- **Answer-only** (`--system answer_only`) — generation on **frozen context**,
  no live search. Feed it a snapshot (`--from-context snapshots/hybrid.json`)
  or gold articles (`--oracle-context`). Every model/prompt sees identical
  passages, so score deltas are attributable to synthesis, not ranking.
- **End-to-end** (`--system agentic_pydantic`, the default) — the full live
  pipeline + agent loop + judge.

### Snapshot → replay workflow

```sh
# 1. Freeze retrieval (10 s, zero LLM):
uv run vesta bench run --system retrieval_only --profile hybrid \
  --level 3 --save-context snapshots/hybrid_200.json

# 2. Replay against different answer models (0 s retrieval):
uv run vesta bench run --system answer_only --from-context snapshots/hybrid_200.json \
  --model qwen3.5-4b
uv run vesta bench run --system answer_only --from-context snapshots/hybrid_200.json \
  --model gpt-oss-20b
```

The snapshot pins the dataset hash + subset hash and the exact passage text, so
a replay proves it fed the same questions the retrieval run measured. Replaying
a `--level` higher than the snapshot's fails loudly (missing questions are never
silently fed empty context).

### Variance runs — N× in parallel against one endpoint

Run the full suite N times in parallel for variance, all pointed at one
endpoint with the same model as answer and judge. Launch one process per
repeat — `--repeats N` runs the cells *sequentially*; separate processes
overlap instead. Give each a distinct `--label` so the runs are easy to find
and compare:

```sh
for i in 1 2 3; do
  nohup uv run vesta bench run \
    --slice all \
    --scope wikipedia_en_top_nopic_2026-06.zim \
    --model qwen3.5-4b@q4_k_s \
    --endpoint http://desktop.onoz.cc:1234/v1 \
    --judge-model qwen3.5-4b@q4_k_s \
    --judge-endpoint http://desktop.onoz.cc:1234/v1 \
    --label "qwen3.5-4b@q4_k_s run $i/3" \
    > bench-run-$i.log 2>&1 &
done
```

Notes:

- `--scope` pins retrieval to the Wikipedia archive
  (`wikipedia_en_top_nopic_2026-06.zim`) so no other registered ZIM is
  consulted.
- The model id must match `/v1/models` on the endpoint exactly
  (e.g. `qwen3.5-4b@q4_k_s`).
- The judge shares the answer endpoint, so `--judge-concurrency` is auto-clamped
  to 1 (recorded as `judge_shares_endpoint` in `config_json`).
- Parallel processes share the endpoint's GPU, so per-question latency is
  contended. Never compare latency across these runs, and don't raise
  `--concurrency` unless the endpoint is dedicated.
- Judge calibration is measured once per process (25 fixed items); a same-model
  judge on a small model can land below the 0.7 trust threshold — check
  `uv run vesta bench list` / `uv run vesta bench show RUN_ID` for `untrusted`.
- Find the runs with `uv run vesta bench list`; a report lands in
  `benchmarks/results/` per run.

### Quick start — the full suite against one system

To run the **entire** question set (all 200 questions, the Wikipedia-only
syllabus) through a single system — e.g. the pydantic agentic search — one
ready-to-paste command:

```sh
uv run vesta bench run --system agentic_pydantic --level 3
```

What this does and the decisions it makes for you:

- `--system agentic_pydantic` selects the one system to benchmark. The default
  matrix is `agentic_pydantic`; pass `--system` (repeatable) instead when you
  only care about specific systems.
- `--level 3` selects the full 200-question release tier. The default is
  `--level 2` (100 questions); `--level 1` is the 50-question smoke tier. Every
  tier is stratified across all 8 capabilities. The benchmark is Wikipedia-only,
  so `core` and `all` are the same set; `--slice` exists for CLI compatibility.
  The legacy non-Wikipedia `cross` slice is retired.
- The benchmark queries only the pinned Wikipedia archive. Scope retrieval to
  it explicitly with `--scope wikipedia_en_top_nopic_2026-06.zim` (or the short
  archive id) to guarantee no other archive is consulted.
- The answer model and endpoint are taken from the configured
  `inference.llm.*` settings (no `--model`/`--endpoint` needed) — see the
  Settings table below. Override per-run with `--model`/`--endpoint`/`--api-key`
  if you don't want the configured defaults.
- The judge comes from `eval.judge.*` (`--judge-model`/`--judge-endpoint`/
  `--judge-api-key` to override).

A full run is long: 200 questions through the live answer pipeline plus an LLM
judge, expected to take on the order of half an hour. To sanity-check the config
before committing to the full run, smoke it first:

```sh
uv run vesta bench run --system agentic_pydantic --level 1 --limit 2 --no-persist
```

Then run the real thing as a long-lived background process (e.g. `hub start` in
this harness) rather than a foreground shell you might interrupt. Results land
in `bench_runs`/`bench_question_results` and a report in `benchmarks/results/`;
find the run with `uv run vesta bench list` and read its scorecard with
`uv run vesta bench show RUN_ID`.

### Scoring

- **Source metrics (deterministic, no LLM):** `source_hit_rank`, `source_recall@{1,5,10,20}`,
  `source_coverage` (the multi-hop "found them all" metric), `source_mrr`,
  `retrieved_precision`. `out_of_corpus` questions are excluded from these
  denominators (no gold source).
- **Answer metrics (always the LLM judge):** `strict_accuracy`, `weighted_accuracy`
  (`(correct + 0.5·partial)/n`), `sub_fact_coverage` (judge-derived),
  `abstention_correctness`, `over_refusal`, `hallucination_rate`, and the
  `unjudged` count (a judge failure marks the run incomplete — never counted
  correct).
- **Three reference points:** ceiling = `oracle`, floor = `closed_book`,
  system = this run. `headroom_realised = (system − floor)/(ceiling − floor)`
  (suppressed when the oracle model ≠ answer model).
- **Failure attribution 2×2:** `{correct, partial/incorrect} × {source_found,
  source_missed}`, plus a "lucky" cell (correct + source missed).
- **Answer-LLM token usage** (per question + suite): for every system that
  calls an LLM (`agentic_pydantic`, `oracle`,
  `closed_book`), the input/output token count of the **answering** model is
  recorded per question in `bench_question_results.input_tokens` /
  `output_tokens`. `retrieval_only` / `sources_only` make no LLM calls and stay
  at 0. The judge's tokens are deliberately **not** tracked — the metric is
  about the answering LLM's cost. The suite aggregates land in
  `metrics_json.tokens.answer`:

  | Field | Meaning |
  |---|---|
  | `total_input` / `total_output` | Sum over the entire question set. |
  | `total` | `total_input + total_output` — the suite's total tokens. |
  | `p50` | Median per-question total (`input + output`). |
  | `p50_input` / `p50_output` | Median per-direction counts. |

  Both the markdown report (`## Token usage (answer LLM)` table) and
  `vesta bench show RUN_ID` display the suite total and p50, plus per-question
  tokens. Token counts are best-effort: an endpoint that does not report usage
  (or a `--no-persist` historical run) records 0.

## `vesta bench verify` — the dataset admission gate

`vesta bench verify` runs three passes over the dataset and writes an
adjudication file to `benchmarks/verification/<date>-review.md`:

1. **Support check (no LLM)** — the answer's distinctive tokens appear in each
   required source's extracted article body. A failure means the ground truth
   is wrong.
2. **Closed-book pass** — the answer model, question only, no context. Judged.
   Populates `closed_book`.
3. **Oracle pass** — the answer model, full extracted text of every required
   source as context. Judged. Populates `oracle`.

Two derived checks gate the dataset: **ceiling ≥ 85%** on active questions and **floor
≤ 20%** on active questions (excluding `lookup`, the regression canary). A
question the answer model can't answer even with the gold article is
`quarantined`, not deleted.

## `vesta bench retrieval` — the golden-set retrieval gate

The former `vesta eval`. A frozen, LLM-free regression gate over the
hand-written golden-set YAML slices (`eval/golden/*.yaml`), persisting to
`eval_runs`. Kept separate from the unified benchmark on purpose — it encodes
retrieval intent (paraphrase, keyword, out-of-corpus) that the answer dataset
does not, and needs no LLM at all. `vesta eval` remains as a **deprecated
alias** printing a pointer to `vesta bench retrieval` (same flags, same
`eval_runs`).

```
uv run vesta bench retrieval [--profile P] [--golden full|fixture_subset]
  [--baseline B] [--sweep k=v1,v2] [--explain] [action run|verify-golden|regression]
```

### Dataset mode — the article-recall arms

`--dataset PATH` switches the same command to the **round-0 article-recall
measurement**: for every source-eligible dataset question (answer behaviour +
≥1 `required` source), run the pipeline under three fixed, zero-LLM arms and
report the rank of a gold article among the returned source cards:

| arm | query | profile |
|---|---|---|
| **A** | the natural-language question (what Round-0 `search_exact` fires today) | `standard` |
| **D** | the natural-language question | `hybrid` (dense) |
| **B** | the gold article's title (the oracle ceiling for any query-shaping step) | `standard` |

```sh
# The baseline methodology (the 150 algorithmic questions):
uv run vesta bench retrieval --dataset benchmarks/vesta_bench_v2.json \
  --capability buried_fact --capability multi_fact_same_article \
  --capability multi_hop_cross_article --data-dir data

# The current full selection (every source-eligible question):
uv run vesta bench retrieval --dataset benchmarks/vesta_bench_v2.json --data-dir data
```

Prints recall@1/@5/@10/any per arm plus the per-arm rescued/lost question ids,
and writes a JSON artifact (per-question per-arm ranks + retrieved paths, so a
later phase can name exactly which questions it moved) to
`benchmarks/results/<ts>-round0-article-recall.json` (`--out` to override).
Dataset mode is LLM-free and **never writes `eval_runs`/`bench_runs`**; the
profiles are pinned per arm (built-ins), so a flipped
`retrieval.active_profile` in the DB cannot contaminate a measurement. Dataset
mode takes the same `--level`/`--capability`/`--limit` filters as `bench run`;
`--profile`/`--sweep`/`--baseline` are golden-set flags and are rejected.

## `vesta bench hardware` — the encoder/extraction/latency harness

The former `vesta bench`. Measures hardware throughput (encoder, extraction,
latency, RAM) and writes a committed markdown verdict to `bench_results/`.
Untouched — just renamed for discoverability.

## `vesta bench rejudge` / `compare` / `list` / `show`

- `vesta bench rejudge RUN_ID` — re-grade a stored run's `pending` answers with
  no pipeline work (the runner wrote each answer as `pending` immediately on
  completion; a killed run leaves those pending, and rejudge completes them).
  It loads the run's dataset to render rubrics for judge-cache misses.
- `vesta bench compare A B` — per-question diff: aggregate deltas plus the
  buckets `fixed` / `broken` / `both_correct` / `both_wrong` / `unjudged`. The
  **broken** bucket is mandatory — never let a mean hide a regression. The
  `source_recall` delta's denominator matches the headline metric: abstain
  (out_of_corpus) questions are excluded. A pair where either side is
  `pending`/`unjudged` lands in **unjudged** — a judge failure must not read
  as a regression.
- `vesta bench list` / `vesta bench show RUN_ID` — list runs / show one run's
  scorecard.

## Judge & calibration

The judge is an LLM routed through the `eval.judge.*` endpoint trio. In this
deployment the **answer** model goes through the LiteLLM proxy
(`https://litellm.loki.onoz.cc/v1`) and the **judge** goes through the bifrost
gateway (`https://bifrost.loki.onoz.cc/v1`):

| Role | Setting | Value |
|---|---|---|
| Answer (results) | `inference.llm.endpoint_url` | `https://litellm.loki.onoz.cc/v1` |
| Answer (results) | `inference.llm.model` | `lmstudio/unsloth/qwen3.5-4b` |
| Answer (results) | `inference.llm.api_key` | `<litellm key>` |
| Judge | `eval.judge.endpoint_url` | `https://bifrost.loki.onoz.cc/v1` |
| Judge | `eval.judge.model` | `cline/cline-pass/deepseek-v4-flash` |
| Judge | `eval.judge.api_key` | `<key>` |

Both gateways need their own key. An empty `inference.llm.api_key` 401s the
answer pipeline (every answer comes back empty, 0 tokens); an empty or invalid
`eval.judge.api_key` shows as `unjudged`.

The `lmstudio/` prefix matters: LiteLLM's model groups are the deployed names
(`lmstudio/unsloth/qwen3.5-4b`), not the bare model id. A judge or answer model
that has **no healthy deployment** on the proxy fails with a 400
("no healthy deployments for this model") and every verdict lands `unjudged` —
check `/v1/models` on the proxy for the exact group names.

`enable_thinking=False` and `max_tokens ≥ 256` are mandatory (reasoning models
burn the token budget on reasoning tokens otherwise — Trap 5). Judging is
cache-aware: the cache key hashes the **rendered rubric** (which embeds the
ground truth), so a ground-truth fix invalidates its own cache entries (Trap 17).

Before a run's verdicts are trusted, the judge is validated against a
hand-scored calibration subset (`benchmarks/calibration_v1.json`, 25 items);
a Pearson correlation < 0.7 marks the run `untrusted`.

## Persistence

Runs persist to three tables (migration `0009`):

| Table | What it holds |
|---|---|
| `bench_runs` | One row per system × profile × model in a group; `run_group` (uuid) is the comparison unit. Pins: dataset hash, subset hash, profile hash, answer/judge model, settings snapshot, git SHA, machine id. |
| `bench_question_results` | Per-question rows (PK `(run_id, question_id)`), verdict ∈ pending\|correct\|partial\|incorrect\|unjudged, with trace (prunable), retrieval paths, answer text, and the answer-model's `input_tokens`/`output_tokens` (migration 0009 + 0010). |
| `bench_judge_cache` | Key = `sha256(rendered_rubric \| qid \| answer \| judge_model)`. |

Historical `answer_runs` rows were imported once into `bench_runs`
(`vesta bench run --import-old`, idempotent) and the old table is now
**read-only** — the unified numbers are the baseline from here on.

Runs left `running` by a dead process are marked `aborted` at startup (the CLI
reconciles them on open, mirroring `main.py`'s lifespan). For a run that dies
mid-flight, the `abort_reason` is recorded under `config_json['abort_reason']`.

Judge concurrency is clamped to 1 when the judge shares the answer endpoint, and
both `judge_shares_endpoint` and the pipeline `--concurrency` are recorded in
`config_json` so a contended run is never silently compared against a clean one.

## Settings

| Key | Default | Purpose |
|---|---|---|
| `bench.dataset` | `benchmarks/vesta_bench_v2.json` | Dataset path. |
| `bench.slice` | `core` | Default slice (Wikipedia-only; `core` = `all`). |
| `bench.max_concurrent` | `1` | Pipeline questions in flight. |
| `bench.judge.concurrency` | `4` | Judge calls in flight. |
| `bench.judge.temperature` | `0.0` | Judge temperature. |
| `bench.judge.max_tokens` | `4096` | Judge output cap (reasoning headroom). |
| `bench.judge.retries` | `1` | Judge retries before `unjudged`. |
| `bench.judge.cache` | `true` | Judge cache on/off. |
| `bench.calibration_path` | `benchmarks/calibration_v1.json` | Hand-scored calibration subset. |
| `bench.calibration_min_correlation` | `0.7` | Below ⇒ untrusted. |
| `bench.trace_retention_days` | `30` | Prune per-question trace blobs older than N days. |
| `inference.llm.endpoint_url` | `https://litellm.loki.onoz.cc/v1` | Answer-model gateway (LiteLLM proxy). |
| `inference.llm.model` | `lmstudio/unsloth/qwen3.5-4b` | Answer model id on the gateway. |
| `inference.llm.api_key` | `<litellm key>` | Answer-model gateway key (LiteLLM requires `sk-…`). |

## Architecture notes (for agents working on this)

- **Boundary rule (load-bearing).** `eval/bench_*.py` imports only `retrieval` +
  `config` (the ≤2 dep cap, enforced by `tests/test_boundaries.py`). All I/O
  (DB, archives, LLM gateway, answer pipeline) is injected via Protocols. The
  composition root (`api/bench.py`, `cli.py`) wires the real implementations.
- **The runner does not read endpoints.** The CLI calls
  `resolve_judge_concurrency(requested, answer_endpoint=…, judge_endpoint=…)`
  before `run_benchmark` and passes `judge_concurrency` + `judge_shares_endpoint`
  through.
- **No lexical answer scoring.** `eval/bench_scoring.py` has no lexical path;
  a judge failure yields `unjudged`, which makes the run incomplete, not wrong.
