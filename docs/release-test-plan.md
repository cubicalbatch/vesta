# Vesta Release Test Plan — Full User Journey

Purpose: confirm the whole standard user journey works before shipping — Docker
Compose setup → first-run wizard → archive download + indexing → search → AI
answers on the local Qwen3.5 4B model → settings → persistence → offline. You (the
test agent) are acting as a normal user with a browser, a shell, and this file.
Follow the phases in order; each phase lists concrete steps, expected results,
and the evidence to capture. File a failure for anything that deviates, with
the evidence attached.

**How to test**: browser-first (the UI is the product), API-level (`curl`) to
confirm or debug. Screenshots + `docker compose logs` excerpts are the evidence
format. Test IDs (`P<n>.<m>`) must appear in the final report.

---

## 0. Product map (what exists, so you know what you're looking at)

Vesta is an offline knowledge appliance over Kiwix ZIM archives. One Docker
container: FastAPI backend + Svelte SPA, SQLite as system of record, ONNX
encoders for retrieval, a bundled `llama-server` for local LLM inference.

UI surfaces (served at `/`):

| Surface | Route | What it is |
|---|---|---|
| Search | `/` | One search box; **Use AI** toggle switches between ranked source cards (`GET /api/search`) and a streamed, cited answer (`POST /api/chat` SSE). URL state: `?q=&scope=&ai=1&c=<conversationId>`. |
| First-run wizard | `/welcome` | Two steps: choose archives → set up AI (download GGUF / remote endpoint / skip). Shown automatically when no archives exist and setup isn't complete. |
| Catalog | `/catalog` | Browse the Kiwix OPDS catalog, download archives (resumable), manage archives "On this machine" (enable/disable, index depth, delete, Browse). |
| Archive browse | `/archive/[zimId]` | Per-archive "look inside": random article, discover grid, archive-scoped search. Works at any index depth. |
| Settings | `/settings` | Tabs: **Settings** (Basic view + "All settings"; an **AI** section at the top with model management), **Jobs** (global job list with pause/resume/cancel). An **Advanced** tab (eval/benchmarks) exists only when `VESTA_ADVANCED_MENU` is set — verify it is hidden. |
| Reader overlay | (overlay) | Clicking a source card / citation opens the article in a sandboxed iframe over the ZIM passthrough. Esc closes. Prev/next across result cards. PDFs (document ZIMs) render in-app via pdf.js. |
| TopBar / StatusBar | (chrome) | Model chip (live LLM state, Load/Unload), job dot (links to Jobs tab), command palette (**Ctrl+K**), `/` focuses search. StatusBar bottom line: `offline · N archives · M articles · model · profile`. |

Key facts:

- **Ports**: compose maps host **5129 → container 8080**. App URL:
  `http://localhost:5129`.
- **Data lives in one bind-mounted directory** (zims/, models/, vesta.db).
  Compose binds `./data` (created on first run); the `docker-compose.e2e.yml`
  override points the bind at `$E2E` instead — read Phase 1 before starting;
  do a **fresh-directory** install so the first-run journey is honestly
  exercised.
- **Three retrieval profiles**: `lexical` (Xapian full-text only), `standard`
  (+ONNX static scoring), `hybrid` (default; + vector kNN **only when the
  archive has a semantic index**). Keyword search works immediately at depth 0;
  semantic matching needs the one-time index build (depth 1–3).
- **Local LLM preset**: exactly one ships — **Qwen3.5 4B (Q4_K_S)**
  (`qwen3.5-4b-q4_k_s`) — 2,590,430,368 bytes (~2.6 GB download), needs ≥4 GB
  RAM (~2.5 GB resident at the default 8k window), thinking toggle (off =
  fast direct answers; default off). It is the only preset, so the wizard
  preselects it regardless of RAM.
- **Internet is needed only for**: the Docker image build, the Kiwix catalog
  refresh + ZIM downloads, and the GGUF model download (HuggingFace). After
  setup, everything is local.
- Encoders (embed/static/rerank ONNX models) are **baked into the image** and
  symlinked into the data volume on first boot — no encoder downloads at runtime.

Host environment (already verified): Linux, Docker 29.x, 8 CPUs, ~38 GB RAM,
~196 GB free disk — ample for a 2.24 GB archive + the 4B model.

---

## 1. Phase 1 — Fresh install via Docker Compose

**Time budget**: first `--build` pulls npm/uv/llama.cpp/HF-model layers —
allow 10–25 minutes. Boot after build: seconds.

### P1.1 Pre-flight

1. From the repo root, confirm what the compose actually binds:
   ```sh
   docker compose config | sed -n '/volumes:/,/^[a-z]/p'
   ```
   Expected: `./data:/app/data` plain, or with the e2e override
   `${E2E:-/tmp/vesta-e2e}:/app/data`.
2. **Do a fresh-directory install** so the first-run journey is honestly
   exercised: the e2e override points the bind at an empty dir and also
   publishes the llama-server router on `127.0.0.1:8081` (used in Phase 11):
   ```sh
   export E2E=/tmp/vesta-e2e && mkdir -p "$E2E"
   docker compose -f docker-compose.yml -f docker-compose.e2e.yml up -d --build
   ```
3. Confirm nothing else squats on host port 5129.

### P1.2 Build & boot

1. Watch the build succeed (all 5 stages: SPA → Python deps → llama.cpp →
   baked encoder models → runtime).
2. `docker compose ps` → container **Up (healthy)**. The healthcheck polls
   `GET /health` inside the container; it must pass within ~1–2 min of start.
3. `docker compose logs vesta | tail -50` → structured JSON log lines, no
   tracebacks; a `vesta.started`-style lifespan line present.
4. **Evidence**: `docker compose ps` output + first 50 log lines.

### P1.3 Healthy empty state

1. `curl -s http://localhost:5129/health | jq` →
   - `"status": "ok"`, `components.database: "ok"`;
   - `setup_completed: false`;
   - `advanced_menu: false`;
   - all `capabilities` false (empty appliance is *healthy*).
2. Open **http://localhost:5129** in the browser → the SPA loads and
   **redirects to `/welcome`** (zero archives + setup incomplete).
3. Spot-check API surfaces on the empty state:
   ```sh
   curl -s http://localhost:5129/api/zims | jq            # {"archives":[]}
   curl -s http://localhost:5129/api/models/status | jq    # state "absent"
   curl -s http://localhost:5129/api/settings/schema | jq '.settings | length'
   curl -s http://localhost:5129/api/system/hardware | jq  # ram_total_bytes ~38GB, cpu_count 8
   curl -s http://localhost:5129/api/system/storage | jq   # free_bytes sane
   ```
   Schema length should be ~104 settings.
4. **Evidence**: `/health` JSON + a screenshot of the welcome page.

---

## 2. Phase 2 — First-run wizard (archive + local LLM)

**Time budget**: archive download depends on bandwidth (~0.18 GB should be
minutes); model download ~2.6 GB; index depth-1 of the small archive: minutes.

### P2.1 Wizard step 1 — choose archives

1. On `/welcome`, the catalog auto-refreshes on first mount (fresh boot has no
   cached OPDS feed). Watch the Jobs (dot in top bar → Jobs tab, or
   `GET /api/jobs`) — a `refresh_catalog` job runs and completes. If the feed
   was already cached, no refresh fires; either is fine.
2. Two featured cards appear: **wikipedia_en_100 (~0.18 GB, "Fastest
   start")** and **wikipedia_en_top (nopic, ~2.24 GB, "Most useful")**, plus a
   secondary checkbox list (wikivoyage, stackexchange, …). Sizes/article counts
   render once the live catalog matches.
3. Select **wikipedia_en_100** only (keep the test corpus small and fast).
   Verify the disk meter reflects free space.
4. Click **Continue** → downloads start (a `download_zim` job), and the page
   advances to step 2. The acquisition chain (download → register → index at
   depth 1) runs in the background and survives a page reload.
5. **Evidence**: screenshot of step 1 with the selection; job list showing
   `download_zim` running with progress/rate.

### P2.2 Wizard step 2 — local LLM

1. Step 2 offers three options: use a remote endpoint, **download a GGUF**,
   or skip. The recommendation preselects the only preset,
   **Qwen3.5 4B (Q4_K_S)** (~2.6 GB download, min 4 GB RAM — fine on this
   38 GB box).
2. Click **Download** → a `download_model` job runs with progress. On
   completion the preset card flips to "Downloaded", the runtime **preloads**
   the model (`GET /api/models/status`: `loading` → `loaded`), and the wizard
   shows the model becoming ready.
3. **Ask a test question** (canned: *"In one short sentence, what is
   Wikipedia?"*) → the answer streams token-by-token inline in the wizard.
   Note: the archive index may still be building — sources may be sparse; the
   streaming itself is what's under test.
4. Click **Finish** → `POST /api/setup/complete`, `/health` now reports
   `setup_completed: true`, and you land on `/` (Search), no longer redirected
   to `/welcome`.
5. **Evidence**: screenshot of the streamed test answer; `/health` after
   finish.

### P2.3 Acquisition chain completes

1. Poll `GET /api/zims` until the archive shows `index_status: "complete"` and
   `index_depth: 1` (watch the Jobs tab for the `index_zim` job's progress
   bar). Expected within minutes for ~5k articles on 8 CPUs.
2. StatusBar (bottom of Search page) now reads like
   `offline · 1 archive · ~5K articles · Qwen3.5 4B … · profile hybrid`.
3. **Evidence**: final `GET /api/zims` JSON + StatusBar screenshot.

---

## 3. Phase 3 — Keyword search (sources mode) and the reader

The **Use AI** toggle defaults OFF. Sources mode hits `GET /api/search`.

### P3.1 Basic search

1. On `/`, search for a topic certain to be in the top-100 corpus (e.g.
   `Albert Einstein` or `World War II`). Ranked source cards appear: title,
   snippet, score, source label (e.g. `xapian_fts`/`title_suggest`), breadcrumb.
2. The URL updates to `/?q=<query>` (replaceState). Reload → the search
   re-runs from the URL.
3. Search gibberish (`qqqzzzxyz`) → an explainable empty result (no crash, no
   fabricated cards).
4. Search a natural-language question (e.g. `who won the battle of hastings`)
   → still returns results (stopword stripping works — without it NL questions
   return 0 hits).
5. **Evidence**: screenshots of a populated result and the empty result.

### P3.2 Reader overlay

1. Click a source card → the reader overlay opens, article renders in the
   sandboxed iframe (no external requests). Esc closes. Prev/next arrows walk
   the result cards when opened from a result list.
2. Open the trace/detail affordance on a sources result (if present) — the
  trace renders (stages, timings). Every run has one; tracing is always on.
3. **Evidence**: screenshot of an open article + one trace view.

### P3.3 Keyboard & chrome

1. `/` focuses the search box; **Ctrl+K** opens the command palette; palette
   navigates (Search/Catalog/Settings entries) and Esc closes everything.
2. **Evidence**: screenshot of the command palette.

---

## 4. Phase 4 — AI answers with Qwen3.5 4B (citations, follow-ups)

**Time budget**: first AI answer includes model load + retrieval + CPU
generation — allow **up to 5 minutes** before calling it hung. Subsequent
answers faster.

### P4.1 First streamed answer

1. Toggle **Use AI** on (persisted in localStorage — reloading keeps the mode).
2. Ask a question grounded in the corpus, e.g.
   *`When and where was Albert Einstein born?`*
3. Expected stream shape (visible progressively in the UI):
   - source cards appear first;
   - a truthful status line (reading → generating);
   - the answer types out token-by-token;
   - citation markers `[n]` appear inline, linked to the source cards;
   - the turn ends with a trace + done (no error banner).
4. The URL becomes `/?ai=1&c=<conversationId>` once the first turn streams.
5. **Evidence**: screenshot mid-stream (partial text) and one of the completed
   answer with citations visible.

### P4.2 Citation click-through

1. Click a citation chip → the reader opens that source with the cited passage
   highlighted/scrolled-to; the provenance strip shows "cited as [n]".
2. **Evidence**: screenshot of the reader showing the highlight.

### P4.3 Follow-up turn (contextual)

1. The hero is now a follow-up box. Ask a referential follow-up, e.g.
   *`What did he win the Nobel Prize for?`*
2. Expected: the agent resolves the reference from conversation history. It may
   answer with **zero new source cards** (valid — follow-ups skip the round-0
   pre-seed) or search mid-turn if the fact is genuinely missing. Either way
   the turn completes with a normal citations + trace ending.
3. **Evidence**: screenshot of the follow-up answer; note whether sources
   events fired (both behaviors are legal).

### P4.4 Conversation persistence

1. Reload the page at `/?ai=1&c=<id>` → the whole thread restores from the API.
2. Open the history panel (clock icon near the hero) → the conversation is
   listed with a derived title; clicking it loads the thread; the trash icon
   deletes it and it disappears from the list (verify with
   `GET /api/conversations`).
3. Start a **new conversation** (plus icon) → fresh thread, old one untouched.
4. **Evidence**: history panel screenshot + `GET /api/conversations` before and
   after delete.

### P4.5 Trace panel on an AI turn

1. Open the trace view on a completed AI turn → it renders the agent summary
   (system, elapsed, tokens in/out, search/read calls) and the per-stage timing
   breakdown (pre_seed / agent_llm / search / read_article, with nested
   retrieval stages).
2. **Evidence**: screenshot.

---

## 5. Phase 5 — Model lifecycle (chip, load/unload, idle, context window)

All observable via the **ModelChip** in the TopBar (click it for a popover:
file, context, thinking, memory estimate, last used) and Settings → AI.

### P5.1 Chip states and manual load/unload

1. Chip shows green + "Qwen3.5 4B …" when loaded. Click the chip → popover
   details; context ≈ 8192 (8K tokens — the factory default), thinking off
   (default; Qwen has a working toggle), memory ≈ 2.5 GB.
2. **Unload** from the popover (or Settings → AI) → chip state changes
   (unloaded/stopped; "No AI"/idle styling).
3. Ask a question while unloaded → the model transparently reloads and the
   answer still streams (first tokens arrive slower). Chip goes loading →
   loaded during the turn.
4. **Evidence**: screenshots of loaded and unloaded chip states; timing note of
   the reload-answer turn.

### P5.2 Idle unload

1. In Settings → AI set **idle unload** to `60` seconds (immediate write, no
   Save button; "Saved — applies to your next question").
2. Ask one question, then wait ~75 s without activity. The chip should show
   sleeping/unloaded (llama-server frees memory; belt-and-braces app watchdog).
3. Ask again → answers fine (transparent reload). Restore idle unload to `900`
   afterwards.
4. **Evidence**: chip screenshot after idle; the successful post-idle answer.

### P5.3 Context window ("Answer speed & memory")

1. In Settings → AI, the **Answer speed & memory** composite shows
   Lean/Balanced/Thorough presets with RAM deltas. Selecting one writes both
   `answer.agent.context_profile` and `inference.local.context_size`; the copy
   explains the window applies after a model restart (Vesta restarts it
   itself).
2. Pick **Thorough** (32k) → ask a question → still answers (the runtime
   restarts llama-server with the new window); the chip popover now shows
   ~32K context.
3. Restore **Lean** afterwards (the factory reading: profile `auto` +
   8192 tokens).
4. **Evidence**: popover screenshots at 32K and back at 8K.

### P5.4 Model deletion (do this only when no further AI tests remain)

1. Settings → AI → installed models list shows the GGUF with a delete (trash)
   action. Delete it → status flips to absent; chip shows a "No AI" state that
   links to `/settings#ai`.
2. **Evidence**: `/api/models/status` JSON after delete.
   (Optional: re-download via the same preset to leave the box in a working
   AI state.)

---

## 6. Phase 6 — Indexing and jobs

### P6.1 Index job lifecycle on a second, larger archive

1. On `/catalog`, download **wikipedia_en_top (nopic, ~2.24 GB)** from the
   catalog (search box finds it). Watch resumable download progress with rate.
2. When the download finishes it auto-registers; the "On this machine" row
   offers **Build index** (uses `index.default_depth`, default depth). Trigger
   it — this build is long enough to exercise job controls.
3. Progress: the row shows a live percentage + rate (fed by the global job SSE
   stream). The header job dot is active during the run.
4. **Pause** the job (row control or Settings → Jobs) → progress halts, status
   paused. **Resume** → continues. **Cancel** is exercised in P6.2.
5. **Evidence**: screenshots of running/paused/resumed states with percentages.

### P6.2 Cancel and re-trigger; depth semantics

1. Cancel the in-flight index → job terminal, archive left at depth 0 (or
   partial), the row offers Build index again. Re-trigger at **depth 1** and
   let it run to completion this time (allow a long budget; poll
   `GET /api/zims` for `index_status: "complete"`, `index_depth: 1`).
2. After it's ready: `index_depth` label shows on the row; a fresh AI question
   now retrieves from the bigger corpus (noticeably better answers).
3. **Evidence**: final `GET /api/zims` for both archives.

### P6.3 Hybrid vs lexical actually changes the path

1. While the big archive is still at depth 0 (before P6.2 completes), run a
   sources search and open its trace → `vector_knn` is recorded as
   **capability-dropped** (no semantic index yet), lexical ran fine.
2. After indexing completes, search again → trace now includes `vector_knn`
   (hybrid profile) with no drop for it.
3. In Settings, switch `retrieval.active_profile` to **lexical**, save, search
   → results still fine; StatusBar profile label flips to `lexical`. Switch
   back to **hybrid**.
4. **Evidence**: the two trace screenshots + StatusBar.

---

## 7. Phase 7 — Catalog and archive management

### P7.1 Catalog browsing

1. `/catalog` → curated "Recommended" section renders (works offline; live
   sizes once the feed is cached). The browse list supports: text search,
   language filter, size buckets (≤2/10/50 GB), recommended-only, sort orders.
   Exercise at least search + one size bucket + language filter — results
   update (debounced).
2. **Refresh catalog** action → a `refresh_catalog` job runs and the list
   re-fetches when it lands.
3. **Evidence**: screenshot with filters applied.

### P7.2 Manual add paths

1. **Add from URL**: use the catalog's add-from-URL dialog with a small direct
   ZIM URL (e.g. a tiny Kiwix ZIM; skip if no convenient URL — then at minimum
   verify the dialog validates an empty/garbage URL without breaking).
2. **Scan folder**: `wget` (or copy) any small `.zim` into the bind-mounted
   `zims/` dir on the host (for the e2e override: `$E2E/zims/`), then use the
   wizard/catalog **Scan** action → "1 added" (or drop a file mid-test and
   reboot — startup also scans). The archive appears in "On this machine".
3. **Evidence**: `GET /api/zims` showing the scanned archive.

### P7.3 Archive row controls

1. **Enable/disable toggle**: disable the small archive → StatusBar count
   drops; a search scoped to all archives no longer returns its cards;
   re-enable.
2. **Scope chips**: on `/`, scope the search to one archive via the scope
   chips → results only from that archive (URL carries `&scope=<id>`).
3. **Delete with keep-file**: delete the scanned test archive with
   *keep file* → row disappears from the list but the `.zim` remains on disk;
   a Scan re-adds it. (Deleting with file removal is the destructive variant —
   only run it on the throwaway archive.)
4. **Evidence**: `GET /api/zims` after each mutation.

### P7.4 Archive browse page

1. From the catalog row, **Browse** → `/archive/[id]`: discover grid of sample
   articles renders; **Random article** opens one in the reader; the scoped
   search box returns cards limited to this archive.
2. **Evidence**: screenshot of the discover grid.

---

## 8. Phase 8 — Settings

Do **not** exhaustively test all ~104 knobs. Test the surfaces and a
representative sample.

### P8.1 Basic view and composites

1. `/settings` → Basic view shows a handful of fields plus: the **AI** section
   (top) and **Answer speed & memory** (already exercised in P5.3).
2. Toggle **Answer speed & memory** to another preset → save → `GET /api/settings`
   reflects both keys. Toggle back.
3. **Evidence**: `GET /api/settings` diff.

### P8.2 All-settings view, groups, validation

1. Toggle **All settings** → the full ~104 settings render grouped by section
   → subsection, each with help text; restart-required settings are marked
   (e.g. `server.host` not hot).
2. Make one valid change (e.g. `logging.level` → `DEBUG`) plus one **invalid**
   one (e.g. `logging.level` typo'd, or a number beyond bounds) → Save → 400
   surfaced inline: the bad field is highlighted, banner says nothing was
   written. Verify **no partial write** (`GET /api/settings` unchanged for the
   bad key).
3. **Discard** reverts unsaved edits.
4. Check hot-ness for real: with `logging.level=DEBUG` saved,
   `docker compose logs` subsequently shows DEBUG lines (hot — no restart).
   Reset to INFO.
5. **Evidence**: screenshot of the validation error; settings JSON before/after.

### P8.3 Jobs tab & Advanced gate

1. `/settings?tab=jobs` → job history with statuses/types/timestamps; the
   pause/resume/cancel buttons match Phase 6 behavior.
2. The **Advanced** tab is **absent** (no `VESTA_ADVANCED_MENU` in compose).
   Optional stretch: `docker compose` with `VESTA_ADVANCED_MENU=1` env → tab
   appears with eval/benchmarks views (do not run benchmarks; just render).
3. **Evidence**: screenshot of the Jobs tab; note Advanced absence.

---

## 9. Phase 9 — Persistence across restart

1. Note current state: archives + depths, settings (e.g. idle unload value,
  profile), conversations, model configured.
2. `docker compose restart` → container returns healthy; the bind-mounted data
   survives:
   - `GET /api/zims` identical;
   - `GET /api/settings` identical;
   - history panel still lists conversations; one thread still restores;
   - `setup_completed` still true (no bounce back to `/welcome`);
   - asking a question still works (model reloads on demand).
3. **Restart resilience (optional but valuable)**: restart the container while
   an index job is running → after boot the job runner resumes/records it; no
   stuck `running` state without a job (the catalog row must not show an
   orphaned 0% progress bar forever).
4. **Evidence**: before/after `GET /api/zims` + `GET /api/settings`.

---

## 10. Phase 10 — Offline verification (the core promise)

1. After setup is complete, prove no external dependency for the core loop:
   - In the browser (devtools Network tab, or request-blocking), run a sources
     search, an AI question, an article open, and a PDF/media open if present —
     **only same-origin requests** may occur. Any request to a CDN/external
     host is a release blocker (the SPA ships fully offline; CI greps the
     build for `https://`).
2. Stronger variant (optional, needs root): cut the container's external
   network (e.g. move it to an `--internal` docker network, or
   `sudo iptables`-block egress for its IP, or simply disconnect the host) →
   search + AI answer + reader still work end-to-end; `/catalog` degrades
   gracefully (catalog "unavailable" state, curated list still shown, no
   crash); model already on disk, no downloads attempted.
3. **Evidence**: network-tab screenshot (filter: blocked/external) + a
   completed offline AI answer.

---

## 11. Phase 11 — API smoke (confirmation layer)

Run these against the running fresh install and record status codes (all
straightforward 200s unless noted):

```sh
BASE=http://localhost:5129
curl -s $BASE/health | jq '.status, .setup_completed'
curl -s $BASE/api/zims | jq '.archives | length'
curl -s "$BASE/api/search?q=einstein" | jq '.cards | length'        # > 0
curl -s $BASE/api/settings/schema | jq '.settings | length'          # ~104
curl -s $BASE/api/models/presets | jq '.presets[].id'                # qwen3.5-4b-q4_k_s (the only preset)
curl -s $BASE/api/models/status | jq '.state'                        # loaded
curl -s $BASE/api/jobs | jq '.jobs[0] | {type,status}'
curl -s $BASE/api/conversations | jq 'length'
curl -s $BASE/api/system/storage | jq '.free_bytes'
curl -s $BASE/api/catalog/state | jq '.available'
```

SSE spot-checks (stream a few events each, then kill):

```sh
# Sources-only answer (frozen protocol: sources → status(sources_only) → trace → done)
curl -sN "$BASE/api/answer?q=einstein&strategy=sources_only" | head -20

# One chat turn: sources → status → token… → citations → trace → done
curl -sN -X POST $BASE/api/chat -H 'content-type: application/json' \
  -d '{"query":"When was Einstein born?"}' | grep -E '^(event|data)' | head -30
```

Optional loopback "remote" test (uses the e2e override's published router):
switch inference to remote with endpoint `http://127.0.0.1:8081/v1` (Settings →
AI → Remote), set model to `Qwen3.5-4B-Q4_K_S.gguf`, ask a question →
answers exactly like local. Switch back to local afterwards.

---

## Known limitations — do NOT file as failures

- **No authentication layer exists** — there is no password or auth anywhere in
  the backend. Exposing the port exposes the whole app; only test on trusted
  networks. Note it in the report as a known gap, not a test failure.
- Follow-up AI turns may legitimately show **zero** source cards (answers from
  conversation context).
- Documents-bundle ZIMs (`zimgit-*`) and media ZIMs are specialized; PDF
  in-app rendering only applies to documents-kind archives. If you didn't
  download one, mark those as not exercised rather than failed.
- First AI answer is slow (model load + CPU generation). Generous timeouts are
  expected behavior, not hangs.

## Out of scope for this pass

- Benchmark/eval tooling (`vesta bench`, `vesta eval`, judge calibration) —
  behind the Advanced gate; not part of the standard user journey.
- Exhaustive settings coverage (test the sample above, not all ~104 knobs).
- Multi-user/auth hardening, GPU (Vulkan) acceleration paths (no `/dev/dri`
  mapping in the default compose), non-Docker dev workflow (`./start.sh`,
  `./dev.sh`) unless specifically requested.

## Report format

Produce a table: `Test ID | Status (pass/fail/blocked/not-exercised) |
Evidence (screenshot path / curl output / log excerpt) | Notes`. End with:
overall verdict (ship / ship-blockers listed), the known-gap notes (no auth
password), and any observed timings worth recording (first answer, index rate,
model load).
