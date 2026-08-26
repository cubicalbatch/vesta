# SSE Answer Protocol

**Frozen wire protocol.** The production UI depends on this exact shape.
Changes are recorded as dated amendments in this file (below); the protocol
itself is contract-tested (`tests/test_sse_protocol*.py` + recorded fixtures).

> **2026-08-12 amendment (agent trace timing breakdown — additive, non-breaking).**
> The `POST /api/chat` agent's `trace` event gains a `stages` field: a per-step
> wall-clock breakdown of the turn (`pre_seed`, `agent_llm`, `search`,
> `read_article`), each `{name, component, duration_ms, params, inputs, outputs}`.
> `search`/`pre_seed` steps also carry a nested `stages` list — the retrieval
> pipeline's own per-stage timings (`candidate_source`, `static_pass`,
> `cross_encoder`, `fuser`, …) so a slow answer can be attributed to its
> encoder/rerank/search cost. Purely additive: clients reading the existing flat
> fields (`elapsed_ms`, `total_tokens`, `search_calls`, …) are unaffected.
> **2026-08-09 amendment (agent-chat consolidation — additive, non-breaking).**
>`POST /api/chat` now drives the streaming **pydantic-ai agent** (the
>`agentic_pydantic` bench system's `iter_agent_turn_events`) instead of the
>old hand-rolled `agentic` loop. The event vocabulary is UNCHANGED — the agent
>emits the same `sources` → `status` → `token` → `citations` → `trace` → `done`
>sequence, plus a merge `sources` event when its tool rounds discover cards
>beyond Round 0. The `agentic` *strategy* is retired (the agent is the
>`/api/chat` engine, not a registered answer strategy); the sole registered
>strategy is `sources_only` (`GET /api/answer` is
>unchanged). `answer_reset` now also covers the agent's fallback/retry
>regenerations: a `UsageLimitExceeded` / context-overflow recovery or an
>abstention-retry emits `answer_reset` immediately before the single
>replacement `token` chunk. No new event type.
> **2026-08-09 amendment (context-aware follow-ups — additive, non-breaking).**
> A `POST /api/chat` follow-up turn (conversation
> history present, `answer.agent.contextual_followups` on by default) no longer
> runs the Round-0 pre-seed on the raw question. The agent resolves references
> from the conversation instead. Two consequences, both within the existing
> event vocabulary and ordering rules — no new event type, no client change:
> 1. **A follow-up may emit ZERO `sources` events.** When the answer follows
>    from facts already established in the conversation, the agent answers
>    directly with no retrieval (ordering rule 1 already says `sources` is first
>    *only "if there are any"*). The stream is `status(reading) →
>    status(generating) → token+ → citations(spans=[]) → trace → done`.
> 2. **The first `sources` event may arrive mid-stream, after a `status`.**
>    When the agent searches for a missing fact, the discovered cards surface
>    as `sources(merge=false)` (this turn's initial set) right after the
>    `searching` status and before the tokens — rather than as a Round-0 event
>    up front. Later discoveries still emit as the existing trailing
>    `sources(merge=true)` delta. The trace gains a boolean `followup` field so
>    a from-context turn (`followup:true`, `search_calls:0`) is distinguishable.
> Turn 1 (no history) is unchanged: it always pre-seeds, so `sources` remains
> the first event there. The reducer already handles both new shapes
> (`sources:[]` start; mid-stream `merge:false` after a status).
> **2026-08-01 amendment (sources merge — additive, non-breaking).** The
> agent chat path may emit a **second** `sources` event, carrying only the cards
> a tool round turned up beyond the initial sources. Every `sources` event now
> carries a `merge` field (`false` on the always-present first event; `true` on
> this new, optional, later one). Clients that don't know about `merge` are
> unaffected: `sources_only` never emits a second `sources` event, and the field
> is additive to the existing payload shape. See "Sources merge" below for the
> full contract.

> **2026-08-02 amendment (additive, non-breaking).** A new `answer_reset`
> event. The agent chat path (`POST /api/chat`) can regenerate the whole answer
> from scratch rather than extending it — a tool-call crash (`reason="fallback"`,
> retried without tools) or an over-refusal on relevant seed sources
> (`reason="abstention_retry"`, re-prompted for an answer). `answer_reset` is
> emitted immediately before the first `token` of such a regenerate, telling
> clients: **everything streamed so far for this turn is superseded — discard
> accumulated answer text and start fresh.** It carries one field, `reason` (a
> short machine tag, e.g. `"fallback"` — may be `""`). Clients that ignore
> `answer_reset` will concatenate the old and new answers (the exact bug this
> fixes) but will not crash — the event is purely additive to the stream.

> **2026-08-02 amendment (precision fix 2 — additive, non-breaking); wording
> corrected 2026-08-26.**
> `citations` gained an `answer_text` field. Inline `[n]` citation markers in
> the generated answer refer to CARD numbers (one per article): the agent is
> prompted with its cards numbered `[1]`..`[n]` (first-seen order), so a
> marker in the answer is already a valid card number — there is NO
> passage→card rewriting step (an earlier version of this amendment described
> a `renumber_inline_citations` pass that does not exist on the current path).
> Streaming makes any in-flight adjustment impossible; `answer_text`, when
> non-null, is simply the FINAL answer text — the exact text the
> `citations.spans` character offsets are computed against. Clients SHOULD
> replace their accumulated token text with `answer_text` when present (for
> both display and persistence); a client that ignores it loses nothing on
> today's path.

> **Evidence-first agent flow (additive — no new event).** The streaming agent
> (`POST /api/chat`) is evidence-first: the raw question is searched before the
> model commits to an answer, and the model may call `search` / `read_article`
> for a missing fact during the turn. Two client-visible consequences, both
> within the existing event vocabulary:
> 1. **No `answer_reset` fires on a normal turn.** Retrieval and any tool calls
>    resolve before the answer's `token` stream, so there is no partial answer
>    to erase and replace. `answer_reset` remains legal on a genuine regenerate
>    (the 2026-08-02 amendment above), so clients must keep handling it.
> 2. The merge `sources` event (`merge: true`, 2026-08-01 amendment) appears
>    when a tool round turns up new evidence beyond the initial sources. Clients
>    that already append on `merge` are unaffected.
> The event ordering contract is unchanged: `sources` (first) → `status` →
> `token` (the final answer, once) → [merge `sources`] → `citations` → `trace`
> → `done`.

## Endpoint

```
GET /api/answer?q=<query>&scope=<zim_ids>&profile=<name>&strategy=<name>
Content-Type: text/event-stream
```

Returns a Server-Sent Events stream. Progressive disclosure
is a **protocol property**: source cards go out immediately (~0.7 s), then
truthful intermediate status during the prefill gap, then tokens. The server is
responsible for there always being a next truthful thing to show.

## Event ordering

```
event: sources       data: {cards: [...]}                    # ~0.7s, FIRST
event: status        data: {phase: "reading", detail: "..."}  # zero or more
event: token         data: {text: "..."}                      # zero or more (streamed)
event: answer_reset  data: {reason: "..."}                    # zero or more, before a regenerate's first token
event: citations     data: {spans: [...], answer_text: "..."} # after tokens
event: trace         data: {...}                              # last data event
event: done          data: {}                                 # terminal success
```

Any event may be followed by an error:

```
event: error       data: {code, message, recoverable}       # terminal on fatal
```

### Ordering rules

1. **`sources` is always first** (if there are any). The source cards are the
   retrieval result — the user is already done if that's all they wanted. The
   agent chat path may emit a **second** `sources` event later, after any tool
   rounds and before `citations` — see "Sources merge" below.
2. **`status` events are truthful**: `"reading"` (evidence gathered, model
   loading/prefilling), `"generating"` (tokens streaming), `"abstaining"`
   (pre-gate fired), `"sources_only"` (no LLM path).
3. **`token` events stream incrementally** — never buffered.
4. **`answer_reset` (2026-08-02 amendment) precedes a regenerate's first
   `token`.** When a strategy replaces (not extends) the answer streamed so
   far, this event fires first. Clients discard all accumulated `token` text
   for this turn and start fresh. Zero or more per response; never emitted for
   a genuine continuation (the answer keeps growing, not restarting).
5. **`citations` comes after all tokens** — citation spans are synthesized
   from the full answer (~10 ms, zero extra tokens). Its `answer_text` field
   (2026-08-02 amendment), when present, is the authoritative final text.
6. **`trace` is the last data event** — the full retrieval + answer trace.
7. **`done` terminates the stream** on success. No more events after it.
8. **`error` may appear at any point** and terminates the stream. A partial
   answer with its citations already computed is better than nothing.

## Event payloads

### `sources`
```json
{
  "cards": [
    {
      "zim_id": 1,
      "path": "Battle_of_Hastings",
      "title": "Battle of Hastings",
      "snippet": "The Battle of Hastings was fought on 14 October 1066...",
      "breadcrumb": "Battle of Hastings > Aftermath",
      "score": 0.87,
      "source": "xapian_fts"
    }
  ],
  "merge": false
}
```
`merge`: `false` on the first `sources` event (always
present). See "Sources merge" below for the second, optional event.

### `status`
```json
{"phase": "reading", "detail": "6 sources"}
```
`phase` ∈ `"reading"` | `"searching"` | `"generating"` | `"abstaining"` | `"sources_only"`.
(`"searching"` — the agent chat path's mid-turn `search`/`read_article` tool
rounds — has been emitted and SPA-handled since the 2026-08-09
consolidation; listed here for completeness. No client change.)

### `token`
```json
{"text": "The Battle of"}
```
Incremental answer text. Concatenate all `token.text` values seen SINCE the
last `answer_reset` (or since the start of the stream, if none) for the full
answer.

### `answer_reset` (2026-08-02 amendment)
```json
{"reason": "fallback"}
```
Everything streamed so far for this turn (all `token.text` accumulated) is
superseded by the regenerate that follows. `reason` is a short machine tag for
the trace/dev console (`"fallback"` | `"abstention_retry"` | `""`) —
informational only, clients should treat any non-empty-or-empty value the
same way (discard and restart).

### `citations`
```json
{
  "spans": [
    {
      "answer_span": [0, 42],
      "card_id": 1,
      "passage_span": [120, 180],
      "score": 1.0
    }
  ],
  "answer_text": "The Battle of Hastings was fought in 1066 [1]."
}
```
- `answer_span`: `[start, end]` character offsets into `answer_text` (or, if
  `answer_text` is null, the concatenated answer per the `token` rule above).
- `card_id`: 0-based index into the `sources.cards` array. In `answer_text`,
  inline citation markers are `[card_id + 1]` — the agent is prompted with its
  cards numbered from 1 (first-seen order), so markers are card numbers by
  construction.
- `passage_span`: `[start, end]` character offsets into the source passage's
  text, or `null` if only document-level alignment was possible. Enables
  click-to-highlight.
- `answer_text` (2026-08-02 amendment, additive, nullable): the final answer
  text — the exact text `answer_span` offsets are computed against. `null`
  when the answer ended up empty. Clients SHOULD replace their accumulated
  token text with it (for both display and persistence) rather than slicing
  spans out of their own token concatenation.
- `score`: constant `1.0`. A marker is an authoritative card citation, not an
  n-gram alignment, so there is nothing to estimate; the retired
  `answer.citations.min_span_score` "weakly supported" treatment was removed
  by migration 0012 and clients MUST NOT key any UI off `score`.

**Citation validity is 100% by construction**: a citation to something not
retrieved cannot exist, because each marker is validated against the retrieved
cards and any `[n]` outside the valid card range is dropped rather than
relabeled.

### `trace`
```json
{
  "version": 1,
  "stages": [...],
  "degradations": [...],
  "profile": "lexical",
  "profile_hash": "abc123..."
}
```
The same trace structure as `GET /api/search`. The answer
stages (`abstention`, `answer`, `citations`) are appended to the retrieval stages.

### `error`
```json
{"code": "stream_error", "message": "...", "recoverable": true}
```
`code` ∈ `"no_llm"` | `"retrieval_failed"` | `"stream_error"` | `"fatal"` |
`"no_profile"` | `"budget_exhausted"` | `"unknown_event"`. `recoverable=true`
means killing `llama-server` mid-answer or a remote timeout — the next question
works. `"budget_exhausted"` (recoverable) means the model spent its whole token
budget before any answer token — seen with reasoning models, whose hidden
thinking consumes `max_tokens`; set `inference.llm.enable_thinking` to `false`
(where the endpoint supports the parameter), raise `answer.max_output_tokens`,
or use an instruct model.

## Sources-only mode

When no LLM is configured (or `?strategy=sources_only`), the stream is:
```
event: sources     data: {cards: [...]}
event: status      data: {phase: "sources_only", detail: "3 sources"}
event: trace       data: {...}
event: done        data: {}
```
No `token` or `citations` events. This is a first-class mode, not a fallback.

## Abstention

When a mechanical pre-gate fires, the stream is:
```
event: sources     data: {cards: [...]}
event: status      data: {phase: "reading", detail: "3 sources"}
event: status      data: {phase: "abstaining", detail: "pre-gate: top_score ..."}
event: token       data: {text: "No passage in your archives closely matches..."}
event: trace       data: {...}
event: done        data: {}
```
The harness-side abstention string is emitted as a single token. No LLM call was
made — zero hallucination risk.

## Sources merge

The agent chat path's tool-call rounds can turn up evidence beyond what the
initial search found.
When they do, a **second `sources` event** follows, with `merge: true`:

```
event: sources     data: {cards: [...], merge: false}   # Round 0, always first
event: status      data: {...}
event: token       data: {...}                           # zero or more
event: sources     data: {cards: [...], merge: true}      # OPTIONAL — new evidence only
event: citations   data: {spans: [...]}
event: trace       data: {...}
event: done        data: {}
```

Rules:

1. **The merge event, when present, carries ONLY the new cards** — not the
   full accumulated set. Its `cards` array is a *delta*, not a replacement.
2. **Clients append, never replace.** The merge event's cards continue the
   same 0-based numbering the first event started: if the first event had 2
   cards (indices 0-1), the merge event's cards are indices 2, 3, .... This is
   what keeps `citations.card_id` valid across both events — a citation
   grounded in recovered evidence can have `card_id: 2` and it resolves
   correctly as long as the client appended rather than rebuilt its card list.
3. **It appears after all tokens, before `citations`** — post-hoc citation
   alignment runs over the FULL merged evidence (Round 0 + everything the
   merge event added), so citations may reference either event's cards.
4. **At most one merge event per response.** Not repeated per round — the
   loop accumulates internally and emits one delta at the end.
5. **`sources_only` never emits a second `sources` event** — only the agent
   chat path (`POST /api/chat`) does.

A client that ignores `merge` entirely (treats every `sources` event as a
full replacement) will render an incomplete card list after a tool round
but will not crash — the field is additive and the first event's shape is
unchanged.

## Client guidance

- Start rendering source cards on the first `sources` event.
- If a second `sources` event arrives (`merge: true`), APPEND its cards to the
  existing list rather than replacing it.
- Show a determinate progress indicator during `status` events (the prefill gap
  on CPU is 10-25 s).
- Concatenate `token` events into the answer area. Never buffer.
- On `answer_reset`, clear the accumulated answer text and the rendered answer
  area, then keep concatenating subsequent `token` events into it (2026-08-02
  amendment — a regenerate is about to replace, not extend, what's shown).
- Render citations as clickable links after the answer completes. A click
  scrolls to the source card and highlights the `passage_span`. If
  `citations.answer_text` is present, replace the displayed/persisted answer
  with it before rendering citation link text (2026-08-02 amendment).
- On `error` with `recoverable=true`, show the error but keep the partial answer.
  The server auto-restarts the backend; the next question works.
