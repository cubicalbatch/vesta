# Vesta Code Audit — 2026-08-24

**Scope:** full repo at `56aff0b` (tree clean) — `src/vesta/**` (38,310 LOC Python), `frontend/src/**` (14,295), `tests/**` (26,594), `scripts/**` (5,356), 15 migrations, CI/packaging.
**Method:** fourteen parallel deep-review passes (retrieval · agent-chat · answer · api-routers · bench-api · eval-domain · storage/index/jobs · zim+catalog · inference+encoders · cli+config · frontend · security · 0822-fix-regression · aux/tests-CI), then orchestrator re-verification. An upstream model rate-limit interrupted partway: the api-router, bench-api, and inference/encoder passes delivered complete reports, the security pass delivered a full summary, and the remainder were recovered from transcripts plus orchestrator-targeted verification at every cited site. Every **Major** below and every carried-over status verdict was independently re-verified against HEAD code before inclusion (bench persistence claims additionally reproduced empirically on a temp-file SQLite DB running the real migrations, marked **[emp]**).

**The headline correction:** the 0822 remediation (38 commits) covered the **Major, Performance, duplication, and dead-code tiers only**. The 0822 **Medium tier was never remediated** — of its ~55 items, roughly two thirds are still open at HEAD, several of them (B1, A1, A2, S1, S2, Z2) unchanged since v0.1. This audit therefore carries a verified per-item status table instead of re-arguing those findings.

Excluded by design (unchanged): no-auth trusted-LAN posture, off-by-default feature flags, `_WIDENED_CAPS`, the deprecated `vesta eval` alias, the deliberately reverted extract cache, and the documented 0822 follow-ups (PATCH-all-None 200, stale media/documents docstrings, word-set split).

---

## Executive summary

The 0822 remediation itself holds up: all ten Major fixes re-verified sound at their seams (M4 rank contract complete across all five emitters, M5 mirror truncation end-to-end correct, M6 migration wrapper structurally atomic, M7/M8 CLI arms correct, sanitizers survived a dedicated bypass hunt). The rot has moved to four places:

1. **Bench/eval persistence round-trips destroy data.** `vesta bench rejudge` rewrites every rejudged row's latency/rounds to zero and recomputes the run's headline latency percentiles as n=0 **[emp]**; `abort_reason`, `judge_shares_endpoint`, and eval `started_at` never survive their row mappers; failed and crashed eval runs report `done`.
2. **One failed bench cell abandons the matrix** — `asyncio.gather` without cancellation leaves sibling cells running detached, later flipping rows from `aborted` back to `complete` on a "failed" group; `DELETE /runs/{id}` during a run triggers the same cascade via an FK crash.
3. **Lifecycle seams the 0822 fixes didn't reach:** llama-server `stop()` races that leak the child (M3 incomplete), an index sidecar/wipe cross-arm hole (M8 incomplete), capability reseed gaps (S3 incomplete).
4. **Declared-but-dead controls:** the entire confidence-gate calibration feature (4 hot settings + `calibrate.py` + a CLI claim that the running app picks them up) is wired to nothing; six bench settings and two eval settings still read `.default`.

| Severity | Count |
|---|---|
| Major (new since 0822) | 8 |
| Medium (new since 0822) | 5 |
| 0822 Medium-tier items still open / partial | ~34 (status table below) |
| Dead-code items (new) | 10 |
| Duplication clusters (new/remaining) | 7 |

---

## Major findings (new)

### M1 · `bench rejudge` zeroes persisted latency metrics and rounds
`src/vesta/api/bench.py:1300-1335` (mapper) · `src/vesta/eval/bench_runner.py:899-937` (rejudge) · `src/vesta/api/bench.py:1121-1152` (rewrite)
`_row_to_result` omits `rounds` and `latency_ms` from the row→dataclass mapping — a row stored with rounds=3/latency=1234.5 reads back 0/0.0 **[emp]**. `rejudge_run` round-trips rows through this mapper twice with destructive effect: `update_question_result` rewrites the rejudged rows' columns with the zeros (original measurements lost from the DB), and `_rebuild_scored` builds every `ScoredQuestion` with `latency_ms=0`, so the recomputed `metrics_json` replaces the run's entire `source.latency` block with n=0/p50=0 (`bench_scoring.py:1086-1093` filters `>0`). After any rejudge, the run's headline latency percentiles are destroyed — wrong persisted numbers in the system of record. The DTO twin `_row_to_result_row` (bench.py:2275+) maps both fields, proving the omission is drift, not policy. **Fix:** map `rounds`/`latency_ms` in `_row_to_result`; add a round-trip test.

### M2 · `abort_reason` and `judge_shares_endpoint` never survive the persistence round-trip
`src/vesta/api/bench.py:1227-1272` · write side `bench.py:1080-1090` + `bench_runner.py:761-766`
`mark_aborted` stashes the reason in `config_json`, but `_row_to_run` never lifts it back: `get_run(...).abort_reason` returns `''` after a successful `mark_aborted` **[emp]** — the `BenchRunDetail.abort_reason` field (bench.py:1647, 2354) and the frontend "Aborted: …" panel are permanently empty. `_run_to_row` has no column for the field at all, so the failed-cell path (which sets `abort_reason` on a `status='failed'` row) loses the reason entirely, and `mark_aborted`'s `WHERE status='running'` then skips the already-failed row. Same mapper drops `judge_shares_endpoint` (True round-trips to False **[emp]**), making the detail field misleading. **Fix:** lift both from `config_json` in `_row_to_run`; stash the reason on the failed-cell path.

### M3 · One failed bench cell abandons the matrix to orphaned background tasks
`src/vesta/eval/bench_runner.py:846`
`run_benchmark` awaits `asyncio.gather(*cells)` without `return_exceptions=True` or cancellation. On the first cell exception the gather raises immediately but the remaining cell tasks keep executing detached; `_run_to_completion`'s handler (bench.py:1927-1935) marks every still-'running' cell aborted and pops `_tasks`, so the API reports the group finished while orphaned cells keep burning LLM calls and later overwrite their rows from `aborted` back to `complete` — mutating a "failed" group under the user. **Fix:** `gather(return_exceptions=True)` + cancel outstanding cells on first failure (or `asyncio.TaskGroup`).

### M4 · Deleting an in-flight bench run crashes its cells via FK and triggers the M3 cascade
`src/vesta/api/bench.py:2261-2269`
`delete_bench_run` has no in-flight guard. Deleting a run whose group is executing removes the `bench_runs` row (cascade wipes its question rows); the runner's next `insert_question_result` raises `sqlite3.IntegrityError` (foreign_keys ON **[emp]**), which becomes a cell-level exception and detonates the M3 cascade exactly. **Fix:** 409 when the run's group is in `_tasks`, mirroring the index-trigger guard.

### M5 · Eval run rows: `started_at` empty forever; failed/crashed runs report `done`
`src/vesta/api/eval.py:88-121` (UPDATE never sets `started_at`; placeholder inserts `''`) · `eval.py:276-291` (status) · `eval.py:348-366` (error path)
After a successful run the row reads `started_at=''` with a real `finished_at` **[emp]** — every API-created eval run renders '—' as its date; the frontend already documents this as a workaround (`frontend/src/lib/format.ts:36-38`). Worse, `EvalRunDetail.status` defaults to `"done"` and nothing ever derives `"error"` (the failure is recorded only inside `config.notes`; `EvalRunResponse` documents `running|done|error`), and there is no stale-placeholder reconciliation at startup — a crashed or failed eval run is indistinguishable from a legitimate all-zero retrieval run. **Fix:** include `started_at` in the UPDATE; persist an error status; reconcile placeholders in the startup sweep.

### M6 · The confidence-gate calibration feature is wired to nothing
`src/vesta/retrieval/__init__.py:92-187` (4 settings) · `src/vesta/eval/calibrate.py` (202 LOC) · `src/vesta/cli.py:2121-2122`
The four `retrieval.confidence.*` settings (hot=True, UI-exposed) have **zero readers** — no constant-name consumer and no string-key consumer anywhere in src/tests/scripts (verified both forms). `vesta bench calibrate` fits thresholds, writes them as settings, and the CLI comment claims "the running app picks them up immediately" — but no pipeline, agent-loop, or answer path reads them; the agent loop has no confidence gate at all. The whole knob cluster plus the calibration report is a placebo: tuning it changes nothing. **Fix:** either implement the gate the docstrings describe (calibrate.py:1-13) or delete the settings + calibrate surface; don't ship knobs that pretend to work.

### M7 · llama-server supervisor: `stop()` still races in-flight starts and leaks the child (0822-M3 incomplete)
`src/vesta/inference/local.py:241-255, 441-457` · rebind path `inference/__init__.py:400-405`
Three residual defects in the M3 fix: (a) `stop()` takes no `self._lock` while `_start_and_wait` assigns `self._proc` only *after* `create_subprocess_exec` (local.py:241) — a settings rebind between the exec await and the assignment reaps nothing (`_proc` still None) and the old supervisor is dropped with a live child holding port 8081; every later spawn dies on bind through all 5 retries. (b) `_abort_start` guards with `except Exception` (local.py:253) — a `CancelledError` delivered mid-exec bypasses the cleanup entirely, orphaning a spawned-but-unassigned child. (c) `ensure_running` awaits a live restart task with `suppress(Exception)` (local.py:205-208) while `stop()` cancels it without awaiting — the CancelledError propagates into the chat caller as a dropped connection. All three leave "local inference broken until the stray process is killed", the exact M3 symptom. **Fix:** lock (or `_stopping` flag) in `stop()`; `except BaseException: await self._abort_start(proc); raise`; suppress CancelledError in the restart-wait.

### M8 · Index sidecar/wipe cross-arm hole: API wipe + stale CLI sidecar ⇒ silent permanent holes (0822-M8 incomplete)
`src/vesta/api/zims.py:601-628` (delete_index) · `src/vesta/cli.py:2561-2585` (sidecar) · `src/vesta/index/job.py:214-227` (fresh branch)
The M8 fix unlinks the CLI sidecar only under CLI `--fresh`. The server arm has two wipe paths that never touch `data/.index_progress_<zim>.json`: `DELETE /api/zims/{id}/index` (depth→0, no pending-job guard either — S2) and a server-triggered depth change. After either, the CLI sidecar still says `{done_count: N, depth: d}`; a later plain `vesta index --depth d` matches the stale depth, "resumes" at N into the wiped store, finishes, and marks `complete` — articles 0..N permanently missing from a "complete" index. The server job's own `job.checkpoint({"done_count": 0})` (job.py:227) writes the *jobs-table* store; the two resume stores remain separate. **Fix:** delete_index and the API index-trigger's fresh branch must unlink the sidecar (or the CLI must validate the sidecar against `zims.index_status/index_depth` before trusting it).

---

## Medium findings (new)

- **N1 · Malformed or name-form `?scope=` silently searches all archives on `/api/search`.** `src/vesta/api/search.py:104-110` — `with suppress(ValueError)` drops the whole scope to None when any token is non-numeric; `/api/answer` fixed this exact bug class (`_parse_scope`: degrade to matches-nothing, resolve name tokens, warn — answer.py:368-405) but the search twin kept the retired pattern. The documented name-form scope works on one endpoint and silently unscopes the other. Reuse `_parse_scope`.
- **N2 · Job-type param validation still gates only `download_model`.** `src/vesta/api/jobs.py:49-85` — every other registered type accepts arbitrary params; `refresh_catalog` takes `params.url` verbatim into a server-side fetch with redirects followed (see carried A1 — this *widens* the SSRF surface to any job-submission client).
- **N3 · LLM API keys returned cleartext by the settings API.** `src/vesta/api/settings.py:88-93` — `GET /api/settings` returns every resolved setting incl. `inference.llm.api_key` / `eval.judge.api_key`; no redaction anywhere (repo-wide grep for redact/secret/mask: zero hits) and no `secret` marking in the schema, so the UI renders the key in plaintext too. Trusted-LAN mitigates exfiltration, not log/history/copy-paste leakage. (The security pass additionally reports keys persisted into run `config_json` snapshots and served back by run-detail endpoints — not independently re-verified.)
- **N4 · Free-form `golden_set` silently launches the full pinned-archive run.** `src/vesta/api/eval.py:202` + `eval/golden.py:254-258` — any unrecognized name falls back to the full set, so a typo (`"fixture_subsets"`) launches the expensive run instead of a 422 and pins it under the typo'd name. Validate against the known set.
- **N5 · `eval.archive.path` / `eval.archive.checksum` declared but unwired.** `src/vesta/eval/golden.py:129-145` — UI-exposed settings; every consumer (golden.py:223-224, api/eval.py:322/397, cli.py:1785/1953/2281) reads `.default`, so relocating the pinned archive is a silent no-op and CLI verify never finds it. Same class as the removed 0822 A4 `zim.dir` knob.

---

## 0822 Medium-tier items — verified status at HEAD

Fixed this cycle (verified): C3 (newest-window history, chat.py:157-159) · C5 (canonical reset ordering via merged recovery core, agent_chat.py:2531) · C6 (removed) · C7/C8 (P1 slim query + 0015 index) · R1 (curly-apostrophe escapes correct, title_entity_suggest.py:75-97) · R5 (P2) · Z4 (P5) · S4 (P3) · S6 (b66592f) · A3 · A4 · A5 (teardown verified clean) · B4 (prune wired at startup) · L1 (effective endpoint feeds the clamp, cli.py:769-774) · L3 (removed) · L9 (P4) · F2 (error rendering moved to dedicated block) · F4 (session aborts controller, session.svelte.ts:110-111) · F6 (P6) · F8 (DEV-gated) · I7 (bi-encoder reconfig correctly serialized).

**Still open** (verified at HEAD this cycle):

| 0822 ID | Current evidence |
|---|---|
| C1 · error+done after terminal event | `chat.py:363-368` serializer appends `DoneEvent`; persistence (`chat.py:234`) runs after `done` forwarded |
| C2 · partial answer lost (partial fix) | user turn now persists pre-stream (`chat.py:171`); non-LLM exceptions still skip the assistant row; docstring overclaims |
| C4 · conversations >500 404 | `chat.py:267-270` still 500-scan + linear search |
| I1 · GGUF resume no truncate/checksum | `inference/download.py:117,200` — M5's `os.truncate` went to the catalog twin only |
| I2 · idle-unload failure latches `error` | `runtime.py:481-485`, `_error` cleared only by rebuild |
| I3 · `source` flip never rebinds gateway | `inference/__init__.py:351-388` + `runtime.py:325-356`; source/endpoint absent from restart keys |
| I4 · envelope usage parsed as zeros | `gateway.py:159-167` getter still getattr-only |
| I5 · idle-unload can fire mid-generation | `runtime.py:476` — only load-in-flight guarded |
| I6 · `sleeping` unreachable, status stale | `runtime.py:444-470` never probes router |
| R2 · rewrite bypasses normalize | `conversational_rewrite.py:110-121` manual split, stale `is_keyword_query` |
| R3 · degradation unflagged (partial fix) | `pipeline.py:310-314` records a stage error but no `tr.degraded` → comparisons still unflagged |
| R4 · bare-path article identity (partial fix) | cards now `(zim_id,path)` (`_shared.py:126-134`) but `max_per_article`/`min_articles` still bare path (`:73-83`) |
| Z1 · no-space text collapses (partial fix) | `_HARD_SPLIT_MULT` cap added, but `_nearest_space_after` still ASCII-space-only with `len(text)` fallback (passages.py:196-205) |
| Z2 · UTF-8-only extraction | `extract.py:176,217,239,249` all `decode("utf-8","replace")` |
| Z3 · mislabeled hot rerank setting | `encoders/__init__.py:135-144` baked into cached cross-encoder |
| S1 · resume double-enqueues job rows | never remediated (no commit touches runner resume; runner accepts `queued` on resume) |
| S2 · delete_index lacks pending-job guard | `zims.py:601-628` — also feeds M8 |
| S3 · capability reseed gaps (partial fix) | reseeded at startup/CLI/delete_index only; archive delete/disable (`zims.py:295-296`) and job error arm (`job.py:176-179`) miss it |
| S5 · ETA inflated by midpoint | `estimate.py:220` still `(low+high)/2`, contradicting the module's own contract |
| A1 · SSRF, now widened | 6 fetch sites, `follow_redirects=True`, zero scheme/host checks; additionally reachable via `POST /api/jobs` `refresh_catalog` (N2) |
| A2 · invalid 206 from Range handling | `zim.py:112-146` never remediated — inverted Content-Range on `bytes=500-400`/past-EOF |
| B1 · gateway swap race | `bench.py:227-230,258` app-global swap, no in-flight guard; overlapping groups nest dead recorders and mis-attribute tokens |
| B2 · abort_reason persistence (read side) | see M2 |
| B3 · settings read `.default` | six remain: `bench.max_concurrent`, `bench.judge.concurrency`, `bench.repeats` (API path) + `bench.judge.cache`, `bench.judge.retries`, `bench.calibration_min_correlation` (runner) |
| B5 · unknown profile → lexical | `api/eval.py:294-307` fallback makes the 404 unreachable; same in `_profile_hash` (bench.py:1759-1764) and cli.py:1672-1678 |
| B6 · attribution drill-down ≠ matrix | `bench.py:2102-2107` still lacks the out-of-corpus exclusion |
| B7 · compare hash guards absent | `bench.py:2168-2207` + `bench_runner.py:987-1047` validate existence only |
| L2 · `--save-context` guard missing return | `cli.py:750-751` warns, runs the full matrix, exits 0 |
| L4 · `--import-old` not idempotent | `bench.py:1371-1445` blind `insert_run` per source row; the "(idempotent)" comment lies |
| L6 · CLI/API twins (partial fix) | pipeline/metric/matrix merged; the profile-fallback resolver is still x4 hand-rolled copies (see Duplication) |
| L8 · FAIL prints exit 0 (partial fix) | `--baseline` now returns 1; `bench verify` floors/ceilings still `return 0` (cli.py:1316-1324) |
| F1 · SSE silent-EOF (partial fix) | `sse.ts` now emits `transportErrorEvent` on fetch/non-2xx/stream errors, but a stream that *ends* without `done`/`error` still yields nothing and the residual buffer is discarded — the docstring's core promise is unimplemented |
| F3 · stale-response races | no abort/sequence guards in SearchPage/ArchiveBrowsePage/Reader (repo grep) |
| F5 · history popover error latch | `AskHistory.svelte:18-26` never clears `error` at fetch start |

Not re-verified this cycle (treat as open): L5 (degraded-boot persistence), L7 (models/hardware settings loading — no `_open_runtime` visible in `_cmd_models`), L10 (force-quit timing), F7 (citation-focus over-subscription).

---

## Dead code (new)

| Item | Location | Note |
|---|---|---|
| Confidence-gate knob cluster | `retrieval/__init__.py:92-187` + `eval/calibrate.py` | The M6 feature — settings + 202-LOC module + CLI wiring with zero runtime consumers |
| `recommend_preset` | `inference/models.py:108-112` | Zero callers repo-wide; also ignores its `ram_total_bytes` arg while the module docstring claims RAM-based picking |
| `ModelSpec.license` | `encoders/registry.py:48` | No reader; only a test fixture sets it |
| `POST /api/zims/{id}/index/estimate` | `api/zims.py:496-517` | Tests only — catalog cards embed server-computed estimates inline (b66592f); delete or declare script surface |
| `DELETE /api/zims/{id}/index` | `api/zims.py:601-628` | No frontend consumer (no `deleteIndex` anywhere); the SPA can deepen but never lower an index — also M8's wipe path |
| `GET /api/zims/{id}/index` | `api/zims.py:631-651` | Tests only; SPA reads index state from the list payload |
| `GET /api/catalog/{entry_id}` | `api/library.py:255-263` | Tests only; the SPA resolves entries from the list and passes entry_id in the download body |
| `GET /api/jobs/{id}` | `api/jobs.py:88-94` | Tests only; the UI consumes the SSE burst (list endpoint stays — documented) |
| `LlmStatusOut.hardware` | `api/models.py:84` | Serialized, never read (UI consumes `/api/system/hardware` instead) |
| `LivePipelineRunner.__init__(profile)` + system `_endpoint`/`_api_key` attrs | `api/eval.py:166-169`; `api/bench.py:639-641,722-724,817-819` | Accepted/stored, never read — the per-run endpoint/api_key args are inert pins for three of four systems |

(`ENCODERS_INDEX_INTRA_OP_THREADS` is missing from `encoders/__init__ __all__` while every sibling is exported — minor, likely an oversight rather than dead.)

## Duplicated code (drift-prone)

| Cluster | Locations | Already diverged? |
|---|---|---|
| Profile-fallback resolver | `api/answer.py:708-729` · `api/search.py:200-216` · `api/eval.py:294-307` · `cli.py:1672-1678` · hash twin `bench.py:1759-1764` | **Yes** — eval's copy makes its 404 dead code (B5); bench's pins the wrong hash; search's documents its fallback as deliberate |
| Bench result-row mappers | `_row_to_result` (bench.py:1300) vs `_row_to_result_row` (bench.py:2275) | **Yes** — the DTO maps rounds/latency_ms, the store copy drops both (root cause of M1) |
| Scope parsing | `search.py:104-110` vs `answer.py:368-405` | **Yes** — N1 |
| Preset→`ModelPresetOut` mapping | `models.py:176-189` vs `:265-277` | Identical today; two edit sites |
| Settings timestamp | `settings.py:27-29` `utc_now_iso` vs inline copy `setup.py:36` | Identical today |
| Media/document wire shapes | `zims.py:219-239` DTOs vs `search.py:245-316` hand-built dicts | Same keys by hand — a DTO field added on one side won't reach the other |
| `_msg` download progress formatter | `inference/download.py:210-214` vs `catalog/download.py:552-556` | **Yes** — same name/signature, MB-style vs bytes-style output |

---

## What checked out clean

Worth recording so future audits don't re-litigate: all ten 0822 Major fixes re-verified sound at HEAD, including the M4 rank contract across **all five** candidate emitters, M5's mirror truncation (re-sync before every attempt, `_MirrorFailure.written` carried forward), M6's per-migration `BEGIN IMMEDIATE` wrapper, and M7's lease module; `safe_gguf_basename`/`safe_zim_basename` survived a dedicated bypass hunt (unicode, `..` variants, absolute paths, backslashes, empty-after-sanitize, symlink resolve+parenthood; NUL passes the guard but fails closed at `open()`); the M9 markdown allow-list is browser-accurate and fail-closed (image tokens never render, raw HTML escaped, Shiki language allow-listed, no `{@html}` outside Shiki); SQL is parameterized everywhere traced (incl. int-coerced `PRAGMA busy_timeout`); spa.py/zim.py path traversal clean; stdlib XML fails closed on external entities (no XXE); no secrets in repo or workflows; P2's bigram cache is faithful to the naive selection; P7's hoists are real; CI runs the documented gates including pre-commit (config + dep present); `scripts/` imports nothing the dead-code sweep removed; regression-gate epsilon arithmetic is correct and fail-closed on judge parsing (`Verdict.UNJUDGED`, no lexical fallback); the vec0 store remains the healthiest subsystem.

## Suggested fix order

1. **Measurement integrity (before the next benchmarking session):** M1 + M2 (one mapper pass + round-trip test), M3 + M4 (TaskGroup + in-flight guard), M5, then carried B1/B5/B7 — the gate is only as good as these.
2. **Lifecycle:** M7 (supervisor locking), M8 + S2 + S3 (one index-transition helper: guard, wipe sidecar, reseed), N1 (reuse `_parse_scope`).
3. **Security posture:** N2 (per-type param validation + A1 allowlist), N3 (secret marking + redaction).
4. **Dead controls:** M6 (decide: implement or delete), B3's six settings, N5, N4.
5. **Batch:** dead-code table, duplication consolidation (start with the two already-diverged mapper pairs), then the remaining carried mediums in the table above.
