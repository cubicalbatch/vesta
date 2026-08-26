"""Answer pipeline — pluggable strategies that turn retrieval into a cited answer.

Same registry pattern: every swappable behaviour
is a registered Protocol implementation. Here the contract is
:class:`~vesta.answer.contracts.AnswerStrategy`. The single registered
implementation is ``sources_only`` (no generation; first-class). The LLM answer
path is the streaming pydantic-ai agent behind ``POST /api/chat``
(:mod:`vesta.api.agent_chat`), which is not a registered strategy.

``answer/`` depends on ``retrieval`` + ``inference`` (module
map — that's exactly 2 deps, at the cap). It must not import ``api/`` or
``index/`` (enforced by ``tests/test_boundaries.py``).

Registration happens by importing the impl module here — no plugin discovery
magic.
"""

from __future__ import annotations

from collections.abc import Container
from typing import TYPE_CHECKING

from vesta.config.capabilities import Capability
from vesta.config.settings import setting

if TYPE_CHECKING:
    from vesta.config.settings import SettingsSnapshot

# ── Settings ────────────────────────────────────────────────────────────────

ANSWER_STRATEGY = setting(
    "answer.strategy",
    str,
    "sources_only",
    group="Answer",
    help="Which answer strategy ``GET /api/answer`` runs. Only 'sources_only' "
    "(no LLM generation — emit source cards + trace) is registered; the LLM "
    "answer path is the pydantic-ai agent behind POST /api/chat.",
    choices=("sources_only",),
    hot=True,
)

# ── Multi-turn chat settings ────────────────────────────────────────────────
# Chat is a thin layer over the same machinery:
# history + conversational rewrite over the same pipeline. These knobs
# bound the history/context and trace growth that long conversations cause.

CHAT_HISTORY_MAX_TURNS = setting(
    "chat.history.max_turns",
    int,
    10,
    group="Chat",
    help="Maximum prior turns retained in the prompt for a multi-turn chat. "
    "Older turns are dropped from the prompt.",
    min=1,
    max=50,
    hot=True,
)
CHAT_TRACE_RETENTION_DAYS = setting(
    "chat.trace_retention_days",
    int,
    7,
    group="Chat",
    help="Days to keep messages.trace_json before pruning. "
    "Multi-turn conversations with full traces grow "
    "the DB; 0 ⇒ keep indefinitely.",
    min=0,
    hot=True,
)

# ── Agent chat follow-up setting ────────────────────────────────────────────
# `/api/chat` drives the streaming pydantic-ai agent. A follow-up turn (with
# conversation history) resolves references from the conversation and answers
# directly when the fact is already established, or searches with a
# self-contained query for a missing fact — instead of pre-seeding retrieval on
# the raw, decontextualized follow-up (the bug: "who died first" retrieved US
# presidents instead of resolving to Napoleon/Lafayette).

ANSWER_AGENT_CONTEXTUAL_FOLLOWUPS = setting(
    "answer.agent.contextual_followups",
    bool,
    True,
    group="Answer / Agent",
    help="When on (default), a follow-up chat turn with conversation history "
    "skips the Round-0 pre-seed and lets the agent resolve references from the "
    "conversation: it answers directly if the fact is already established, or "
    "searches with a self-contained, context-resolved query for a missing fact. "
    "Turn 1 (no history) always pre-seeds regardless. Off = the legacy "
    "always-pre-seed behaviour (kept for A/B measurement and rollback).",
    hot=True,
)

# ── Agent token-economy settings ────────────────────────────────────────────
# Local CPU inference is prefill-bound (measured live: 14k input tokens for
# 105 output tokens, 291 s on an i5-11320H — prefill is ~95% of wall time).
# These knobs bound what the agent feeds the model per turn. ``economy``
# gates the whole ladder: "auto" activates it only when the bound local
# runtime reports CPU-only hardware; "on"/"off" force it regardless (so a
# GPU benchmark can measure the strategy). Explicit user values for the
# individual knobs always win over the economy defaults.

ANSWER_AGENT_ECONOMY = setting(
    "answer.agent.economy",
    str,
    "auto",
    group="Answer / Agent",
    help="Token-economy mode for the agent answer path. 'auto' (default) "
    "applies the leaner context budget only when local inference is "
    "CPU-only; 'on' forces it (for A/B measurement on any hardware); "
    "'off' always uses the full-context defaults.",
    choices=("auto", "on", "off"),
    hot=True,
)


# ── Context-window profile ──────────────────────────────────────────────────
# The window the answer path plans against. "auto" derives the plan from the
# live local window (inference.local.context_size: ≤8192 → the 8k plan,
# ≤16384 → the 16k plan, else full); a remote endpoint has no window, so
# auto = full. Forcing "8k"/"16k" resolves that plan regardless of
# the runtime — and ACTIVATES window budgeting even where the hardware-gated
# economy is off — so the benchmark can A/B profiles on any hardware. "full"
# keeps uncapped behaviour. The 8k plan uses full system prompt + 2400-char
# passage cap; "8k-fullprompt" and "8k-fullprompt-wide" stay force-only bench
# axes (wide is values-identical to "8k").
ANSWER_AGENT_CONTEXT_PROFILE = setting(
    "answer.agent.context_profile",
    str,
    "auto",
    group="Answer / Agent",
    help="Context-window profile the agent plans its turn against. 'auto' "
    "follows the local inference window (8k/16k plan, else full; remote = "
    "full); '8k' forces the 8k plan — full system prompt + the 2400-char "
    "passage cap; '16k' forces the 16k plan; '8k-fullprompt' is the 8k plan with the "
    "narrower 1800-char cap; '8k-fullprompt-wide' is "
    "values-identical to '8k'; 'full' keeps "
    "uncapped behaviour.",
    choices=("auto", "8k", "8k-fullprompt", "8k-fullprompt-wide", "16k", "full"),
    hot=True,
)

# Token twin of tool_budget_chars: bounds cumulative
# tool-result inserts in ESTIMATED tokens. 0 = derive from the window
# arithmetic (window - output_reserve - system - question - pre-seed). The
# char knob keeps back-compat precedence: when both are set, chars win.
ANSWER_AGENT_TOOL_BUDGET_TOKENS = setting(
    "answer.agent.tool_budget_tokens",
    int,
    0,
    group="Answer / Agent",
    help="Turn-level cap on cumulative tool-result inserts, in estimated "
    "tokens (the token twin of tool_budget_chars; when both are set the "
    "char knob wins). 0 = derive from the context-window arithmetic.",
    min=0,
    hot=True,
)

# Escape hatch on the plan's output reserve: the tokens
# held back for the model's completion inside the window. 0 = the plan
# decides (8k: 1280, 16k: 1792).
ANSWER_AGENT_OUTPUT_RESERVE_TOKENS = setting(
    "answer.agent.output_reserve_tokens",
    int,
    0,
    group="Answer / Agent",
    help="Tokens reserved for the model's output inside the context window "
    "(the window covers prompt + completion). 0 = the profile's plan "
    "decides.",
    min=0,
    hot=True,
)
ANSWER_AGENT_PRESEED_PASSAGES = setting(
    "answer.agent.preseed_passages",
    int,
    6,
    group="Answer / Agent",
    help="How many passages (top-ranked first) the Round-0 pre-seed "
    "includes with full text. Default 6 = the previous hard-coded "
    "behaviour; lower to shrink turn-1 context.",
    min=1,
    max=12,
    hot=True,
)
ANSWER_AGENT_PRESEED_PASSAGE_MAX_CHARS = setting(
    "answer.agent.preseed_passage_max_chars",
    int,
    0,
    group="Answer / Agent",
    help="Per-passage character cap in the Round-0 pre-seed. 0 = no cap.",
    min=0,
    hot=True,
)
ANSWER_AGENT_PRESEED_SHOW_ARCHIVE_ID = setting(
    "answer.agent.preseed_show_archive_id",
    bool,
    False,
    group="Answer / Agent",
    help="Render the `(archive-N)` suffix on search/pre-seed source headers. "
    "The zim id is useless to the model (one archive per scoped turn) and "
    "smaller models may copy the token verbatim. Off drops the suffix, keeping "
    '`[n] "Title"`. Default false drops the token.',
    hot=True,
)
ANSWER_AGENT_PRESEED_ORDER = setting(
    "answer.agent.preseed_order",
    str,
    "idf",
    group="Answer / Agent",
    choices=("rank", "idf"),
    help="Order of the Round-0 pre-seed passages. 'idf' (default) re-orders by "
    "focus.py's IDF question-term score (stable, retrieval rank the tiebreak) "
    "before the preseed_passages slice; card numbering stays discovery-order. "
    "'rank' preserves raw retrieval rank. The compact search-tool branch always "
    "keeps rank order.",
    hot=True,
)
ANSWER_AGENT_COVERAGE_SEARCH = setting(
    "answer.agent.coverage_search",
    bool,
    True,
    group="Answer / Agent",
    help="After the Round-0 pre-seed search, extract the question's entity "
    "spans (the same extractor title_entity_suggest uses) and, for each span "
    "no retrieved passage mentions, fire one search_exact on that span alone "
    "and merge its passages into the pre-seed pool. "
    "Merged passages keep discovery-order card numbers. Costs one "
    "extra retrieval per fired span; default true.",
    hot=True,
)
ANSWER_AGENT_COVERAGE_SEARCH_MAX = setting(
    "answer.agent.coverage_search_max",
    int,
    1,
    group="Answer / Agent",
    help="How many UNCOVERED entity spans coverage_search may fire a second "
    "search for (strongest span first). 0 disables the searches even with "
    "coverage_search on. Bound: each span is one extra retrieval pass.",
    min=0,
    max=4,
    hot=True,
)
ANSWER_AGENT_READ_MAX_CHARS = setting(
    "answer.agent.read_max_chars",
    int,
    0,
    group="Answer / Agent",
    help="Character cap on read_article results, cut by the best-scoring "
    "window (never head-first). 0 = full article.",
    min=0,
    hot=True,
)
ANSWER_AGENT_MAX_OUTPUT_TOKENS = setting(
    "answer.agent.max_output_tokens",
    int,
    4096,
    group="Answer / Agent",
    help="Output token budget per agent run (replaces the hard-coded 4096).",
    min=256,
    hot=True,
)
ANSWER_AGENT_TOOL_BUDGET_CHARS = setting(
    "answer.agent.tool_budget_chars",
    int,
    0,
    group="Answer / Agent",
    help="Turn-level cap on cumulative tool-result characters (search results "
    "+ read_article text) inserted into the conversation. When a further "
    "call would exceed it, the tool returns a steering message instead. "
    "0 = no cap.",
    min=0,
    hot=True,
)
ANSWER_AGENT_AGE_TOOL_CHARS = setting(
    "answer.agent.age_tool_chars",
    int,
    -1,
    group="Answer / Agent",
    help="When >0, tool results older than the last two rounds are truncated "
    "to this many characters in requests sent to the model, with a stub note "
    "telling the agent it may re-call read_article [n] if needed. -1 = aging "
    "off (the default); 0 = derive an "
    "eviction budget from the turn's window ledger when a context-window "
    "profile is active.",
    min=-1,
    hot=True,
)

# ── Tail levers ─────────────────────────────────────────────────────────────
# Levers that make the tool-round tail cheap instead of absent, each inert
# unless a window plan is active. All ship OFF by default; the mechanisms
# stay behind these settings as explicit opt-ins.

ANSWER_AGENT_MAX_TOOL_ROUNDS = setting(
    "answer.agent.max_tool_rounds",
    int,
    -1,
    group="Answer / Agent",
    help="Cap on tool-call rounds (search + read_article executions) per "
    "turn. -1 = no extra cap (the default — the harness caps alone). "
    "0 = derive from the context-window ledger "
    "arithmetic when a profile is active (how many read-sized inserts the "
    "ledger admits). An explicit N > 0 binds on any profile.",
    min=-1,
    max=16,
    hot=True,
)

ANSWER_AGENT_COMPACT_REASK = setting(
    "answer.agent.compact_reask",
    str,
    "off",
    group="Answer / Agent",
    help="Compact-and-re-ask: when a windowed turn exhausts "
    "its tool rounds or ledger, or abstains despite gathered evidence, "
    "replace the steered answer with ONE fresh no-tool request over the "
    "system prompt + question + the best evidence gathered (pre-seed plus "
    "read excerpts, focused-windowed to fit). 'off' (the default); "
    "'auto' = on for windowed profiles, off at full; 'on' forces it on any "
    "profile (bench A/B).",
    choices=("auto", "on", "off"),
    hot=True,
)

ANSWER_AGENT_SEARCH_ENTRIES = setting(
    "answer.agent.search_entries",
    int,
    6,
    group="Answer / Agent",
    help="How many passages the compact 'search' tool result shows. Default 6 "
    "= the previous hard-coded behaviour; the CPU economy default is 5.",
    min=1,
    max=10,
    hot=True,
)
ANSWER_AGENT_SEARCH_SNIPPET_CHARS = setting(
    "answer.agent.search_snippet_chars",
    int,
    400,
    group="Answer / Agent",
    help="Per-passage snippet length in the compact 'search' tool result. "
    "Default 400 = the previous hard-coded behaviour; the CPU economy "
    "default is 350.",
    min=50,
    hot=True,
)
ANSWER_AGENT_COMPACT_PROMPT = setting(
    "answer.agent.compact_prompt",
    bool,
    False,
    group="Answer / Agent",
    help="Use the compact system prompt (~2.6k chars incl. the "
    "answer-immediately directive) under economy. Bench: -211k tokens on "
    "109/150 questions but +386k on 41 that destabilized into extra tool "
    "rounds; +1.6pp strict. Default off = full prompt (round-stable).",
    hot=True,
)
ANSWER_AGENT_ANSWER_CLEANUP = setting(
    "answer.agent.answer_cleanup",
    bool,
    True,
    group="Answer / Agent",
    help="Apply deterministic post-processing to generated answers: remove a "
    "narrow leading preface, archive markers, and trailing pseudo-citations.",
    hot=True,
)

# 'strong' appends a must-state clause to the Round-0 strong-evidence
# directive, gated on the seed's top cross-encoder score so
# ungrounded adversarial questions keep refusing.
ANSWER_AGENT_EVIDENCE_DIRECTIVE = setting(
    "answer.agent.evidence_directive",
    str,
    "strong",
    group="Answer / Agent",
    help="Strengthen the Round-0 strong-evidence directive: 'strong' (default) "
    "appends a must-state clause (a shown source stating the asked fact must "
    "be stated with a [n] citation), gated on evidence_directive_min_score so "
    "ungrounded adversarial questions still refuse. 'standard' keeps the "
    "unstrengthened directive.",
    choices=("standard", "strong"),
    hot=True,
)
ANSWER_AGENT_EVIDENCE_DIRECTIVE_MIN_SCORE = setting(
    "answer.agent.evidence_directive_min_score",
    float,
    0.85,
    group="Answer / Agent",
    help="Round-0 top cross-encoder score at which the 'strong' must-state "
    "clause fires. Default 0.85 sits above every adversarial top score in "
    "the pinned set (max 0.809) and below the strong-evidence p50 (0.995).",
    min=0.0,
    max=2.0,
    hot=True,
)

# ── Round-0 conditional reformulation ───────────────────────────────────────
#
# Note: ``answer.reformulate.enabled`` / ``.max_queries`` were retired with the
# agentic loop (migration 0011); these are keys for a different mechanism —
# one conditional LLM call after a *visibly failed* Round 0,
# not an always-on pre-retrieval transformation.

ANSWER_REFORMULATE_ENABLED = setting(
    "answer.reformulate.enabled",
    bool,
    False,
    group="Answer / Round 0",
    help="When Round-0 retrieval visibly fails (confidence.top_score below "
    "answer.reformulate.trigger_score), make ONE LLM call that names the "
    "article the fact would live in, re-search with it, and keep the better "
    "result. Healthy questions never fire it; on any failure the original "
    "result is returned untouched.",
    hot=True,
)
ANSWER_REFORMULATE_TRIGGER_SCORE = setting(
    "answer.reformulate.trigger_score",
    float,
    0.25,
    group="Answer / Round 0",
    help="Round-0 top_score below this fires the conditional reformulation "
    "(default 0.25 = the abstention floor).",
    min=0.0,
    max=1.0,
    hot=True,
)
ANSWER_REFORMULATE_MAX_QUERIES = setting(
    "answer.reformulate.max_queries",
    int,
    1,
    group="Answer / Round 0",
    help="Reformulated queries per trigger, one per line.",
    min=1,
    max=2,
    hot=True,
)
ANSWER_REFORMULATE_MAX_TOKENS = setting(
    "answer.reformulate.max_tokens",
    int,
    64,
    group="Answer / Round 0",
    help="Token budget for the reformulation call — enough for 1-2 short "
    "article names, deliberately too small for an essay.",
    min=16,
    max=256,
    hot=True,
)
ANSWER_REFORMULATE_PROMPT_VARIANT = setting(
    "answer.reformulate.prompt_variant",
    str,
    "exemplified",
    group="Answer / Round 0",
    help="The reformulation prompt arm: 'exemplified' (the bench-tuned "
    "'name the article this fact would live in' prompt with worked examples) "
    "or 'minimal' (the A/B control: no examples, conservative).",
    choices=("exemplified", "minimal"),
    hot=True,
)

# ── Strategy registration (import impl modules to populate the registry) ────
# No discovery magic; importing here registers them.

from vesta.answer import sources_only as _sources_only  # noqa: E402, F401


def resolve_strategy_name(
    configured: str,
    capabilities: Container[Capability],
    *,
    settings: SettingsSnapshot | None = None,
) -> str:
    """Auto-select a safe strategy when capabilities are unmet.

    ``configured`` is what the settings say. With only ``sources_only``
    registered (which requires no capability), there is no degradation ladder:
    every configured value passes through unchanged. The seam remains so a
    future capability-requiring strategy can re-introduce a degrade-don't-fail
    step here without touching callers.
    """
    return configured


def select_strategy(name: str) -> type:
    """Look up a registered answer strategy class by name. Returns the class."""
    from vesta.retrieval.registry import resolve

    cls = resolve("answer_strategy", name)
    if cls is None:
        raise RuntimeError(f"answer_strategy {name!r} not registered")
    return cls


__all__ = [
    "ANSWER_AGENT_ANSWER_CLEANUP",
    "ANSWER_AGENT_COVERAGE_SEARCH",
    "ANSWER_AGENT_COVERAGE_SEARCH_MAX",
    "ANSWER_AGENT_EVIDENCE_DIRECTIVE",
    "ANSWER_AGENT_EVIDENCE_DIRECTIVE_MIN_SCORE",
    "ANSWER_AGENT_PRESEED_ORDER",
    "ANSWER_AGENT_PRESEED_SHOW_ARCHIVE_ID",
    "ANSWER_REFORMULATE_ENABLED",
    "ANSWER_REFORMULATE_MAX_QUERIES",
    "ANSWER_REFORMULATE_MAX_TOKENS",
    "ANSWER_REFORMULATE_PROMPT_VARIANT",
    "ANSWER_REFORMULATE_TRIGGER_SCORE",
    "ANSWER_STRATEGY",
    "CHAT_HISTORY_MAX_TURNS",
    "CHAT_TRACE_RETENTION_DAYS",
    "resolve_strategy_name",
    "select_strategy",
]
