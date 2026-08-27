"""Agent chat API — a pydantic-ai tool-calling agent wired to Vesta's search.

A pydantic-ai ``Agent`` whose ``search`` / ``read_article`` tools run Vesta's real
retrieval pipeline against the loaded archives. It is the LLM answer path —
driven by the unified benchmark's ``agentic_pydantic`` system (``vesta bench run
--system agentic_pydantic``) and by the streaming ``/api/chat`` route.

The model, endpoint, API key, and thinking toggle come from the ``inference.llm.*``
settings — the same ones the Ask-with-AI answer path uses.

The standalone ``agent-chat`` HTTP route is removed; the reusable
turn runner (``run_one_turn`` / ``iter_agent_turn_events``) remains importable.
"""

from __future__ import annotations

import contextlib
import re
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from typing import Any, cast

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

from vesta.answer.citations import synthesize_citation_spans
from vesta.answer.contracts import (
    AnswerResetEvent,
    CitationsEvent,
    DoneEvent,
    ErrorEvent,
    SourcesEvent,
    StatusEvent,
    TokenEvent,
    TraceEvent,
)
from vesta.answer.economy import EconomyBudget, resolve_budget
from vesta.answer.tokens import estimate_tokens_for_chars
from vesta.answer.tools import SearchToolResult
from vesta.api.answer import _build_tool_runtime, _parse_scope, _resolve_profile
from vesta.api.state import AppState
from vesta.retrieval.contracts import SourceCard

SYSTEM_PROMPT = """You are a research assistant that answers questions using ONLY an offline \
archive (Wikipedia and similar reference sites), accessed through the `search` and \
`read_article` tools. You have no other knowledge sources — never answer from your own \
memory or training data.

Each question comes with INITIAL sources: numbered passages ([1], [2], ...) shown above the \
question. They are the archive's best matches for the question. Read them first, before \
deciding anything.

Tools:
- read_article(n): read the FULL text of source [n] (the number shown next to it). Use this \
when a source looks relevant but its excerpt does not contain the exact fact you need. This \
is usually a better next step than another search.
- search(query): keyword matching over passages. Returns short PASSAGES (excerpts), not full \
articles, ranked by matching words. Use it only when the initial sources and full reads do \
not surface a specific missing fact.

The search tool matches WORDS, not meaning — extra words copied from the question \
("age", "when", "how many", "born date", "why") do not help it find the right article; \
they just add noise that can drown out the one match that matters.

Query rules:
- Search for the ENTITY the question is about, plus at most one disambiguating word — never \
the fact you are trying to find. Ask yourself "what article would this live in", not "how \
do I rephrase the question".
  - Question: "How old was Napoleon when he became emperor?"
    - Good queries: "Napoleon", "Napoleon emperor", "Napoleon coronation"
    - Bad queries: "Napoleon age when became emperor", "Napoleon I born date crowned \
emperor date" — these are fact-shaped, not entity-shaped, and will match worse.
- If the results do not contain the answer, try a genuinely different angle: a related \
entity name, a narrower or broader term, or a synonym for the concept — not the same words \
padded with more fact-words.
- You get up to 3 search attempts. Prefer read_article on a promising source before \
spending another search.

When you find the answer, state it directly in 2-4 plain sentences. Ground each \
fact with its supporting detail from the source — the date, year, quantity, or name \
that establishes it — so the answer is verifiable. For a question about an age or \
duration, state the dates you used (e.g. "Einstein, born 1879, published special \
relativity in 1905 at about 26 [1]."). Cite every distinct fact with the bracket \
number of the source you used. Re-use the same [n] if it came from the same source. \
Do not also name the article in prose — the numbered source is shown to the user \
separately, alongside your answer. Do not open with a preface that restates where \
your answer came from — no "Based on the provided sources", "According to the \
sources", or similar. Every answer is drawn from the archive by design, so lead \
with the fact itself, not a mention of the sources.
- When the question asks for a specific name, term, drug, chemical, numeral, or \
entity, answer with that EXACT specific thing — not a related, broader, or similar \
concept. Re-read the question and verify your answer satisfies every qualifier it \
states before finalizing.

Do not refuse while a shown source is relevant. If a source looks relevant but its excerpt \
lacks the exact fact, call read_article on it first. Only say you could not find it after \
you have read the relevant sources in full and none contains the answer. Never guess.
"""

#: Economy-gated compact variant of :data:`SYSTEM_PROMPT` (~2.6k chars vs
#: 3.4k): every load-bearing rule survives — offline-only grounding,
#: initial-sources framing, both tool descriptions, words-not-meaning search,
#: entity-shaped queries, the search/read caps, the 2-4 sentence answer with
#: grounded detail, [n] citation + re-use, no-prose-source-name, no-preface,
#: exact-entity, don't-refuse-while-relevant, never guess — with the worked
#: examples and connective prose cut. Iteration-5 bench: without the explicit
#: "answer immediately — no tool call is needed" directive the compact swap
#: flipped 31 one-shot questions into a read_article (93→62 one-shots, input
#: 1.66M→1.90M); iteration-6 A/B on the flipped questions showed adding it
#: (plus the Einstein citation-format example) restores one-shots (input
#: 10.2k→4.3k, 35.2k→29.0k, 16.5k→4.2k). One-shot preservation is the
#: load-bearing property: the median question must stay a single request.
#: Selected by ``budget.compact_prompt`` (economy active); the
#: strong-evidence and follow-up directives append to it unchanged.
COMPACT_SYSTEM_PROMPT = """You are a research assistant that answers questions using ONLY an offline \
archive (Wikipedia and similar), via the `search` and `read_article` tools. You have no \
other knowledge sources — never answer from your own memory or training data.

Each question comes with INITIAL sources: numbered passages ([1], [2], ...) shown above \
the question — the archive's best matches. Read them first, before deciding anything.

Tools:
- read_article(n): the FULL text of source [n]. Use when a source looks relevant but \
its excerpt lacks the exact fact. Usually better than another search.
- search(query): keyword matching over passages (short excerpts, not full articles). \
Use only when the initial sources and full reads do not surface a specific missing fact.

Search matches WORDS, not meaning — extra words copied from the question ("age", \
"when", "how many") just add noise that drowns out the one match that matters.

Query rules:
- Search for the ENTITY the question is about, plus at most one disambiguating word — \
never the fact you are trying to find. Entity-shaped queries beat fact-shaped ones \
that copy the question's words.
- If the results miss, try a genuinely different angle: a related entity, a narrower \
or broader term, a synonym — not the same words padded with more fact-words.
- Up to 3 searches and 3 reads. Prefer read_article on a promising source before \
another search.

When you find the answer, state it directly in 2-4 plain sentences, grounding each \
fact with the date, year, quantity, or name that establishes it — e.g. "Einstein, \
born 1879, published special relativity in 1905 at about 26 [1]." For an age or \
duration, state the dates you used. Cite every distinct fact with the bracket number \
of the source you used. Re-use the same [n] if it came from the same source. If the \
initial sources' excerpts already contain the exact fact, answer immediately — no \
tool call is needed. Do not also name the article in prose — the numbered source is \
shown to the user separately. Do not open with a preface about where your answer \
came from; lead with the fact itself.
- When the question asks for a specific name, term, drug, chemical, numeral, or \
entity, answer with that EXACT specific thing — not a related, broader, or similar \
concept. Verify your answer satisfies every qualifier before finalizing.

Do not refuse while a shown source is relevant. If a source looks relevant but its \
excerpt lacks the exact fact, call read_article on it first. Only say you could not \
find it after you have read the relevant sources in full and none contains the \
answer. Never guess.
"""


_MAX_SEARCH_CALLS = 3

#: The harness caps below are intentionally module constants, not operator-facing
#: settings: they are deeply bench-tuned internal limits
#: (see PROGRESS.md agentic_pydantic R1-R4) — casually changing them moves bench
#: numbers, so they are deliberately not surfaced as ``answer.agent.*`` settings.

#: Harness cap on ``read_article`` calls per turn: a looping small model
#: (qwen3.5-4b) calls read_article repeatedly without converging; after this many
#: reads the tool returns a steering message forcing an answer, mirroring the
#: search-call cap (:data:`_MAX_SEARCH_CALLS`).
_MAX_READ_CALLS = 3

#: pydantic-ai model-request budget per ``agent.run`` — a backstop against a model
#: that re-requests without converging even after both tool caps exhaust. The tool
#: caps are the primary convergence force; this only bounds latency for the tail
#: (without it a looping 4b spins to pydantic-ai's default request_limit of 50).
_REQUEST_LIMIT = 8

#: Returned by the ``search``/``read_article`` tools when the turn's cumulative
#: tool-result character budget (``answer.agent.tool_budget_chars``) is
#: reached. Nothing else bounds cumulative inserts, and every tool round
#: re-prefills all of them — the measured tail (67.9k input on one question)
#: is 3 searches + 3 reads over 6 rounds, each round re-sending everything.
_TOOL_BUDGET_STEERING = (
    "You have already gathered substantial source material (budget reached). "
    "Do not call more tools; answer the question now from the sources above, "
    "with [n] citations."
)


#: Returned by the ``search``/``read_article`` tools when the turn's tool-round
#: cap (``answer.agent.max_tool_rounds`` — derived from the window
#: ledger unless set explicitly) is reached. Distinct text from
#: :data:`_TOOL_BUDGET_STEERING` so bench traces can count each lever's firings
#: separately from the ``tool_calls`` result previews.
_ROUND_CAP_STEERING = (
    "You have used every tool round this turn's context window permits. "
    "Do not call any more tools; answer the question now from the sources "
    "above, with [n] citations."
)


#: Appended to the system prompt when Round-0 retrieval returned sources, steering the
#: model away from refusal (over-refusal is the #1 answer-layer failure mode for this
#: harness — the standard agentic loop controls it with a mechanical pre-gate + score
#: floor; the agent analogue is a conditional directive + a retry, see ``run_one_turn``).
_STRONG_EVIDENCE_DIRECTIVE = (
    "\nThe initial sources above are the archive's best matches for this question and very "
    "likely contain the answer. Read them — and use read_article for the full text of any "
    "that look relevant — before considering any refusal, then answer with [n] citations."
)

#: Abstention calibration: appended to the strong-evidence
#: directive when ``answer.agent.evidence_directive='strong'`` and Round-0's
#: top cross-encoder score cleared ``evidence_directive_min_score`` — the
#: bundle refuses despite shown evidence (13 of 19 "not present" answers had
#: source_coverage 1.0), while the score gate keeps the genuinely-ungrounded
#: adversarial questions refusing. Like the directive it strengthens, it is
#: steering prose only: one sentence pair, no tool-teaching.
_STRONG_EVIDENCE_MUST_STATE_CLAUSE = (
    "\nIf a shown source states the fact the question asks about, you must state it "
    "with a [n] citation; refusing while that fact is shown is a failure."
)

#: A fresh no-tool retry is deliberately not the ordinary
#: pre-seed wrapper: it names the question and tells the model that the fact is
#: in the *focused* source blocks that follow.  Each retained block includes
#: its rendered ``[n]`` header, so every requested citation still resolves.
_P6_ABSTAIN_REASK_HEAD = (
    "The fact answering this question is in the focused sources below and must be stated. "
    "State it directly with [n] citations; do not refuse.\n\nQuestion: "
)
_P6_ABSTAIN_REASK_TAIL = "\n\nFocused sources:\n\n"

#: Headroom (chars) the window ledger adds when projecting the
#: next request from the last completed one: the assistant tool-call
#: message (call args + JSON envelope) plus the tool-return wrapper — wire
#: content ``_wire_chars`` cannot see. Deliberately modest: the 3.0
#: chars/token estimator already over-counts ~24 % on the mean
#: (measured calibration), which is the primary safety margin.
_WINDOW_LEDGER_SLACK_CHARS = 512


#: Window-safety slack: the per-card char reserve added ON TOP of
#: the read bound when pre-reserving prompt room for the preread blocks.
#: ``focused_view`` guarantees its budget only modulo the unconditionally
#: included spans (the article lead + the must-include snippet) and the
#: ``_MERGE_GAP_CHARS`` joins between them, so an excerpt can exceed its cap
#: by that slop; the block header and separators ride in the same reserve.
#: Erring high is the safe direction: the subtraction shrinks the pre-seed
#: fit, which only drops tail passages (evidence loss), never overflows.
_PREREAD_FIT_SLACK_CHARS = 1024

#: The wrapper around the pre-seed in the turn-1 user message
#: (named so the pre-flight fit counts exactly the chars the request will
#: carry — head + seed + tail + question).
_USER_MESSAGE_HEAD = "Initial sources for this question:\n\n"
_USER_MESSAGE_TAIL = "\n\nQuestion: "

#: Appended to the system prompt on a follow-up turn (conversation history present),
#: REPLACING the "initial sources" framing the base ``SYSTEM_PROMPT`` assumes. The
#: follow-up has no Round-0 pre-seed: the agent resolves references from the
#: conversation, answers directly when the fact is already established, or searches
#: with a self-contained query for a missing fact. Three lines
#: are load-bearing: "do not use [n] markers" on the from-context path (else a
#: dangling ``[1]`` survives in ``answer_text`` with no card to resolve it); "never
#: recall from memory" (the grounding invariant — a new fact MUST be retrieved, not
#: recalled, so the offline-first contract holds on the search path too); and "act,
#: do not ask to act" — a model that knows it must search will still ask the user
#: permission ("Would you like me to look that up?") instead of searching; this forces
#: the tool call so the turn ends with an answer, not a meta-question.
_FOLLOWUP_DIRECTIVE = (
    "\n\nYou are continuing an existing conversation shown in the message history. "
    "The user's new message may refer to people, things, or facts discussed in earlier "
    "turns — resolve those references from the history before answering.\n"
    "\nThis question does NOT come with initial sources.\n"
    "\nYou have NO knowledge of your own. Your parametric memory is NOT a source for "
    "facts — only the conversation history above and the archive (via tools) are. "
    "This is absolute, even for famous, obvious, or well-known facts.\n"
    "\n- If EVERY specific fact your answer needs is already stated verbatim in the "
    "conversation above, answer directly. Do not call any tool, and do not use [n] "
    "citation markers (there are no sources to cite).\n"
    "\n- Otherwise — if the answer needs ANY date, number, name, or detail that is NOT "
    "quoted in the conversation above — you do NOT know it yet, no matter how obvious it "
    "feels. You MUST call search or read_article to find it in the archive FIRST, then "
    "answer grounded in what the tools returned, with [n] citations. Stating an un-recalled "
    "fact from memory is a hallucination and is forbidden.\n"
    "\nNever ask the user for permission to search, and never reply with an offer or "
    'question about whether to look something up — no "Would you like me to look that '
    'up?", "Shall I search?", or "Do you want me to find out?". If the answer needs '
    "a fact that is not already quoted in the conversation above, call search or "
    "read_article and then answer. Act, do not ask to act.\n"
    "\nWhen you search, form a SELF-CONTAINED query: resolve pronouns and references into "
    'the specific entities from the conversation (e.g. for "who died first" after a turn '
    'about Napoleon and Lafayette, search "Napoleon death" and "Lafayette death", not '
    '"who died first").'
)


def _contextual_followups_enabled(sn: Any) -> bool:
    """Read the ``answer.agent.contextual_followups`` setting (default on).

    Lazily imported so ``agent_chat`` does not gain a module-load dependency on the
    answer settings module (mirrors :func:`_resolve_llm`'s lazy ``inference`` import).
    """
    from vesta.answer import ANSWER_AGENT_CONTEXTUAL_FOLLOWUPS

    if sn is None:
        return bool(ANSWER_AGENT_CONTEXTUAL_FOLLOWUPS.default)
    try:
        return bool(sn.get(ANSWER_AGENT_CONTEXTUAL_FOLLOWUPS))
    except Exception:
        return bool(ANSWER_AGENT_CONTEXTUAL_FOLLOWUPS.default)


def _setting_int(sn: Any, setting: Any) -> int:
    """Read one int ``answer.agent.*`` setting with its registered default as
    the fallback (    the :func:`_contextual_followups_enabled` recipe)."""
    if sn is None:
        return int(setting.default)
    try:
        return int(sn.get(setting))
    except Exception:
        return int(setting.default)


def _setting_str(sn: Any, setting: Any) -> str:
    """The str twin of :func:`_setting_int` (compact_reask's auto|on|off)."""
    if sn is None:
        return str(setting.default)
    try:
        return str(sn.get(setting))
    except Exception:
        return str(setting.default)


def _setting_bool(sn: Any, setting: Any) -> bool:
    """The bool twin of :func:`_setting_int` (snapshot values are already
    coerced/validated by the resolver, so a plain bool() is lossless)."""
    if sn is None:
        return bool(setting.default)
    try:
        return bool(sn.get(setting))
    except Exception:
        return bool(setting.default)


def _setting_float(sn: Any, setting: Any) -> float:
    """The float twin of :func:`_setting_int` (snapshot values are already
    coerced/validated by the resolver, so a plain float() is lossless)."""
    if sn is None:
        return float(setting.default)
    try:
        return float(sn.get(setting))
    except Exception:
        return float(setting.default)


_ANSWER_CLEANUP_ARCHIVE_RE = re.compile(r"[ \t]*(?:\(archive-\d+\)[ \t]*)+")
_ANSWER_CLEANUP_PREFACE_RE = re.compile(
    r"^\s*(?:"
    r"the question asks\b[^.!?\r\n]*[.!?]['\"]?"
    r"|based on (?:the )?(?:provided )?sources[.!?]['\"]?"
    r"|according to (?:the )?(?:provided )?sources[.!?]['\"]?"
    r")(?:[ \t]+|\r?\n+[ \t]*)(?P<remainder>\S[\s\S]*)$",
    re.IGNORECASE,
)
_ANSWER_CLEANUP_BARE_LINE_RE = re.compile(r"\s*\[\d+\]\s*")
_ANSWER_CLEANUP_PSEUDO_LINE_RE = re.compile(r'\s*\[\d+\]\s+"[^"\r\n]*"\s*')


def _cleanup_answer(answer: str) -> str:
    """Apply the opt-in, deterministic answer cleanup.

    The transformations are deliberately narrow and operate only on a leading
    preface sentence, literal archive markers, and complete trailing citation
    lines.  In particular, inline quotes and citation markers remain untouched.
    """

    def _archive_replacement(match: re.Match[str]) -> str:
        # Keep a separator when the marker had surrounding horizontal
        # whitespace, unless punctuation/newline already supplies the boundary.
        if not any(char in " \t" for char in match.group(0)):
            return ""
        after = match.string[match.end() :]
        if not after or after[0] in "\r\n,.;:!?)]":
            return ""
        return " "

    text = _ANSWER_CLEANUP_ARCHIVE_RE.sub(_archive_replacement, answer)
    text = text.strip()

    lines = text.splitlines()
    while lines and (
        _ANSWER_CLEANUP_BARE_LINE_RE.fullmatch(lines[-1]) is not None
        or _ANSWER_CLEANUP_PSEUDO_LINE_RE.fullmatch(lines[-1]) is not None
    ):
        lines.pop()
    text = "\n".join(lines).rstrip()

    match = _ANSWER_CLEANUP_PREFACE_RE.match(text)
    if match is not None:
        remainder = match.group("remainder").strip()
        remainder_content = re.sub(r"\[\d+\]", "", remainder)
        if (
            remainder_content.strip()
            and re.search(r"\w", remainder_content)
            and _ANSWER_CLEANUP_BARE_LINE_RE.fullmatch(remainder) is None
            and _ANSWER_CLEANUP_PSEUDO_LINE_RE.fullmatch(remainder) is None
        ):
            text = remainder

    return text.strip()


def _compact_reask_enabled(sn: Any, window_tokens: int) -> bool:
    """Resolve ``answer.agent.compact_reask`` — the economy's
    ``auto|on|off`` shape exactly: ``auto`` activates the lever only under a
    windowed profile (``window_tokens > 0``), ``on`` forces it on any profile
    (the bench A/B axis), ``off`` forces it off."""
    from vesta.answer import ANSWER_AGENT_COMPACT_REASK

    mode = _setting_str(sn, ANSWER_AGENT_COMPACT_REASK)
    if mode == "on":
        return True
    if mode == "off":
        return False
    return window_tokens > 0


#: Phrases that read as a refusal to answer. Single source of truth: the benchmark's
#: ``agentic_pydantic`` system (``api/bench.py``) imports :func:`looks_abstained` from here
#: instead of keeping its own copy, so the over-refusal stat and the retry gate agree.
_ABSTAIN_PATTERNS = [
    r"\bcould not find\b",
    r"\bcouldn't find\b",
    r"\bunable to find\b",
    r"\bno (?:relevant )?(?:information|passage|source)s? (?:were |was )?found\b",
    r"\bi don't have (?:that|this) information\b",
    r"\bnot (?:present|available) in (?:the|my) (?:available )?(?:archive|sources|corpus)\b",
    r"\bno mention\b",
    r"\bnot mentioned\b",
    r"\bdo(?:es)? not (?:contain|mention|include|provide)\b",
    r"\bcannot find\b",
    r"\bno information (?:about|on|regarding)\b",
]
_ABSTAIN_RE = re.compile("|".join(_ABSTAIN_PATTERNS), re.IGNORECASE)


def looks_abstained(answer_text: str) -> bool:
    """True when ``answer_text`` reads as a refusal to answer (or is empty)."""
    if not answer_text.strip():
        return True
    return bool(_ABSTAIN_RE.search(answer_text))


#: Body substrings (checked case-insensitively, together with "exceed") that identify a
#: context-window overflow across differently-worded provider error bodies (litellm
#: proxying LM Studio/llama.cpp, vLLM, ...). Requiring one of these AND "exceed" keeps
#: the match narrow: an unrelated 400 (malformed request, bad auth) won't contain either.
_CONTEXT_OVERFLOW_MARKERS = ("context size", "context length", "context window")


def _is_context_overflow_error(exc: ModelHTTPError) -> bool:
    """True when a ``ModelHTTPError`` is a context-window overflow, not some other 4xx/5xx.

    When the accumulated tool-call history exceeds the answer model's context window,
    the endpoint returns a hard 400 whose body says so, e.g. (litellm proxying LM
    Studio)::

        status_code: 400, body: {'message': "... 'Context size has been exceeded.' ..."}

    That case is recoverable the same way as :class:`UsageLimitExceeded`: a fresh
    no-tool fallback run against just the Round-0 pre-seed is far smaller than the
    blown-up history and fits comfortably. This helper is what lets ``run_one_turn``
    route ONLY that failure mode into the fallback — a malformed-request 400, a 401, or
    a provider 500 for an unrelated reason must stay loud and propagate, so the match is
    deliberately narrow: the status code must be 400 AND the body must contain both a
    context-size marker phrase and the word "exceed" (substring, so it also matches
    "exceeded").
    """
    if exc.status_code != 400:
        return False
    body_text = str(exc.body).lower()
    return "exceed" in body_text and any(
        marker in body_text for marker in _CONTEXT_OVERFLOW_MARKERS
    )


#: The directive half of :func:`_abstain_retry_prompt`, on its own.
#: Under a small window the full retry re-sends the whole
#: pre-seeded user message on top of a fresh history (~2x the turn-1 prompt,
#: which overflows an 8k window); the window-aware retry keeps the original
#: first request as history and sends ONLY this directive as the new prompt.
_ABSTAIN_RETRY_DIRECTIVE = (
    "You did not give an answer, but the initial sources above are relevant. "
    "Call read_article on the most relevant source number, then answer the "
    "question with [n] citations."
)


def _abstain_retry_prompt(user_message: str) -> str:
    """The abstention-retry prompt: the turn's own user message (question plus
    its Round-0 pre-seed) followed by the read-then-answer directive.

    The retry deliberately re-runs against the ORIGINAL pre-turn history, not
    the failed attempt's transcript — the failed transcript roughly doubles the
    retry's input tokens (the prefill-bound CPU case this exists for) while
    containing nothing the retry needs. Re-injecting the user message keeps the
    "initial sources above" reference in the directive true.
    """
    return f"{user_message}\n\n{_ABSTAIN_RETRY_DIRECTIVE}"


def _runtime_hardware() -> str | None:
    """The bound runtime's hardware tag (``"cpu"``/``"gpu"``), ``None`` if remote
    or no runtime is bound — the token-economy gate's hardware signal.

    Read via :func:`getattr` because ``LlmTarget.hardware`` lands concurrently
    (the field may not exist yet on a mid-merge tree); ``None``, its documented
    "remote/unknown" value, is the safe answer either way.
    """
    from vesta.inference import get_runtime

    runtime = get_runtime()
    if runtime is None:
        return None
    return getattr(runtime.target(), "hardware", None)


def _runtime_window_tokens(sn: Any) -> int | None:
    """The live context window in tokens for a LOCAL runtime, else ``None``.

    The window the answer path plans against. The source
    of truth for local-vs-remote is the bound runtime's ``target()`` — never
    a bare ``inference.local.context_size`` read when the source is remote,
    which would budget against a window no request uses. The local
    supervisor bakes exactly this setting into llama-server's ``c=``
    (inference/local.py), so on the local path the setting IS the window.
    """
    from vesta.inference import INFERENCE_LOCAL_CONTEXT_SIZE, get_runtime

    runtime = get_runtime()
    if runtime is None or runtime.target().source != "local":
        return None
    if sn is None:
        return int(INFERENCE_LOCAL_CONTEXT_SIZE.default)
    try:
        return int(sn.get(INFERENCE_LOCAL_CONTEXT_SIZE))
    except Exception:
        return int(INFERENCE_LOCAL_CONTEXT_SIZE.default)


def _resolve_llm(sn: Any) -> tuple[str, str, str, bool | None, str | None]:
    """Resolve the chat model + endpoint + API key + thinking toggle + hardware
    for a turn.

    The single source of truth is the bound LLM runtime's
    :meth:`~vesta.inference.runtime.LlmRuntime.target`: with
    ``inference.llm.source=local`` the endpoint is the supervisor's base URL
    and the model id the router-resolved one — never
    ``inference.llm.endpoint_url``. Only when no runtime is bound (an
    in-process caller that never ran the composition root's lifespan) does
    this fall back to reading the ``inference.llm.*`` settings directly.
    The fifth element, ``hardware``, feeds the token-economy gate
    (:func:`vesta.answer.economy.resolve_budget`); it is ``None`` whenever
    the source is remote or unknown.
    """
    from pathlib import Path

    from vesta.inference import get_runtime

    runtime = get_runtime()
    if runtime is not None:
        target = runtime.target()
        model_id = target.model_id
        if target.source == "local" and model_id.endswith(".gguf"):
            model_id = Path(model_id).stem
        # getattr bridge: LlmTarget.hardware lands in parallel (see
        # _runtime_hardware) — None is the "remote/unknown" contract value.
        return (
            model_id,
            target.base_url,
            target.api_key,
            target.enable_thinking,
            getattr(target, "hardware", None),
        )

    from vesta.inference import (
        INFERENCE_LLM_API_KEY,
        INFERENCE_LLM_ENABLE_THINKING,
        INFERENCE_LLM_ENDPOINT_URL,
        INFERENCE_LLM_MODEL,
        INFERENCE_LLM_SOURCE,
    )

    def _get(setting: Any, cast: type = str) -> Any:
        if sn is None:
            return setting.default
        try:
            return cast(sn.get(setting))
        except Exception:
            return setting.default

    model = str(_get(INFERENCE_LLM_MODEL))
    source = str(_get(INFERENCE_LLM_SOURCE))
    if source == "local" and model.endswith(".gguf"):
        model = Path(model).stem
    endpoint = str(_get(INFERENCE_LLM_ENDPOINT_URL))
    api_key = str(_get(INFERENCE_LLM_API_KEY))
    thinking: bool | None = None if sn is None else bool(_get(INFERENCE_LLM_ENABLE_THINKING, bool))
    return model, endpoint, api_key, thinking, None


#: Human message for a local runtime that cannot come up: the
#: user's fix is picking a model in Settings, not reading a stack trace. The
#: underlying runtime reason is appended in parentheses.
_LOCAL_RUNTIME_UNAVAILABLE = (
    "The local model runtime isn't available — open Settings → AI to pick a model"
)


async def _ensure_llm_ready(on_status: Callable[[str], None] | None = None) -> None:
    """Bring the bound LLM runtime to "ready to chat".

    No-op when no runtime is bound (or the source is remote — the runtime
    decides). Raises ``LlmRuntimeError`` / ``BinaryMissing`` /
    ``LlamaServerError`` when the local runtime cannot come up; streaming
    callers map that to a clean terminal SSE ``error`` event, ``run_one_turn``
    lets it propagate (a benchmark turn against a dead runtime should fail
    loudly, not hang on an endpoint that will never answer).
    """
    from vesta.inference import get_runtime

    runtime = get_runtime()
    if runtime is not None:
        await runtime.ensure_ready(on_status=on_status)


def _mark_llm_used() -> None:
    """Stamp the runtime's ``last_used`` on turn completion.

    Success or failure — the idle watchdog's unload/stop decisions key off
    this, so a turn that crashed mid-answer still counts as "in use".
    """
    from vesta.inference import get_runtime

    runtime = get_runtime()
    if runtime is not None:
        runtime.mark_used()


@contextlib.contextmanager
def _in_flight_generation() -> Iterator[None]:
    """Guard a turn against idle-unload in the bound LLM runtime (I5)."""
    from vesta.inference import get_runtime

    runtime = get_runtime()
    if runtime is not None and hasattr(runtime, "in_flight"):
        with runtime.in_flight():
            yield
    else:
        yield


class ToolCallLog(BaseModel):
    query: str
    result_preview: str


class SourceCardDTO(BaseModel):
    n: int
    zim_id: int
    path: str
    title: str
    snippet: str
    breadcrumb: str
    score: float | None
    source: str


# ── Reusable turn runner (shared by the unified benchmark's
#    ``agentic_pydantic`` system in api/bench.py) ────────────────────────────
#
# Extracted so the streaming runner and the benchmark can never silently
# diverge on the agent/system-prompt/citation-numbering logic — both call
# ``run_one_turn`` / ``iter_agent_turn_events`` directly.


@dataclass
class TurnResult:
    """One agent turn's outcome: answer text + citation-ordered source cards.

    ``cards`` is sorted by ``n`` (first-seen/citation order across every
    ``search`` call in the turn) — exactly the rank order a benchmark driver
    needs for ``retrieved_paths``. ``trace`` is the turn's trace
    dict (system/budget/stages) — the non-streaming analogue of the streaming
    path's ``TraceEvent`` payload, so benchmark drivers can merge it into
    their per-question ``trace_json``.
    """

    answer: str
    cards: list[SourceCardDTO] = field(default_factory=list)
    tool_calls: list[ToolCallLog] = field(default_factory=list)
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_ms: int = 0
    trace: dict[str, Any] = field(default_factory=dict)


def _make_model(model_id: str, endpoint: str, api_key: str) -> Any:
    """Build the pydantic-ai chat model for one turn.

    Seam extracted so tests can inject a ``FunctionModel`` stub by patching
    ``_make_model`` (the streaming runner and ``run_one_turn`` share it).

    The profile pins ``openai_chat_supports_max_completion_tokens=False``:
    pydantic-ai 2.x otherwise maps the ``max_tokens`` ModelSetting to the
    OpenAI ``max_completion_tokens`` request field, which llama-server /
    LM Studio / vLLM ignore — the output cap silently vanished from the wire
    (verified by request capture: the payload carried
    ``max_completion_tokens`` and ``max_tokens: null``). Every endpoint Vesta
    talks to is OpenAI-compatible server software that honors plain
    ``max_tokens``, so we send that. The explicit profile merges ON TOP of the
    name-inferred one (e.g. the qwen3.5 schema transformer is kept).
    """
    return OpenAIChatModel(
        model_id,
        provider=OpenAIProvider(base_url=endpoint, api_key=api_key or "not-set"),
        profile=OpenAIModelProfile(openai_chat_supports_max_completion_tokens=False),
    )


#: Appended to an aged (truncated) tool result, in the request copy only —
#: tells the agent the fact is still reachable by re-calling the tool.
_AGED_RESULT_NOTE = (
    "\n[older result truncated for context economy — re-call read_article [n] "
    "/ search if you still need it]"
)


def _age_rounds(messages: list[ModelMessage], age: int) -> list[ModelMessage]:
    """Deep-copy ``messages`` with tool results older than the last two rounds
    truncated to ``age`` chars (+ :data:`_AGED_RESULT_NOTE`).

    A "round" is a run of consecutive tool-result-bearing ``ModelRequest``\\ s
    (parallel tool results) bounded by assistant messages; rounds are counted
    from the END so the agent always sees its two most recent tool results in
    full. Strings at or under ``age`` — steering/dedup notes, short results —
    pass through untouched; the Round-0 pre-seed lives in the user prompt, not
    a tool result, so it is never aged. The input list and its part objects
    are never mutated: pydantic-ai's canonical history and traces stay
    full-size — only what crosses the wire shrinks.
    """
    if age <= 0:
        return messages
    aged = deepcopy(messages)

    def _has_tool_return(m: ModelMessage) -> bool:
        return isinstance(m, ModelRequest) and any(isinstance(p, ToolReturnPart) for p in m.parts)

    rounds: list[list[int]] = []
    for i, m in enumerate(aged):
        if _has_tool_return(m):
            if rounds and rounds[-1][-1] == i - 1:
                rounds[-1].append(i)
            else:
                rounds.append([i])
    for group in rounds[:-2]:
        for idx in group:
            msg = aged[idx]
            if not isinstance(msg, ModelRequest):
                continue
            msg.parts = [
                (
                    replace(p, content=p.content[:age] + _AGED_RESULT_NOTE)
                    if isinstance(p, ToolReturnPart)
                    and isinstance(p.content, str)
                    and len(p.content) > age
                    else p
                )
                for p in msg.parts
            ]
    return aged


@dataclass
class _AgingMeter:
    """Firing accounting: how many requests the
    aging wrapper actually TRIMMED and by how many characters — the per-turn
    firing count the bench counts. ``0/0`` when aging is off or no request
    carried a tool result old enough to evict."""

    requests: int = 0
    saved_chars: int = 0


class _AgedContextModel(Model[Any]):
    """Request-side context-aging wrapper (token economy, iteration 4).

    Measured tail anatomy: questions with 3+ tool rounds re-prefill OLD tool
    results at full size every round; keeping only the last two rounds full
    and truncating older ones to ``age_tool_chars`` models out to ~50% of the
    tail's input tokens. This wrapper delegates everything to the inner model
    but passes :func:`_age_rounds`-transformed copies to ``request`` /
    ``request_stream`` / ``count_tokens`` — pydantic-ai's canonical history is
    never mutated, so it stays faithful while the wire payload shrinks.

    ``stats`` (observability) accumulates what the trimming did
    per request; measurement only, never a behaviour change.
    """

    def __init__(
        self, inner: Model[Any], age_tool_chars: int, stats: _AgingMeter | None = None
    ) -> None:
        super().__init__()
        self._inner = inner
        self._age = age_tool_chars
        self._stats = stats

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def system(self) -> str:
        return self._inner.system

    @property
    def profile(self) -> ModelProfile:
        # Delegate: the inner model's profile carries the pinned
        # max_tokens mapping and the name-inferred schema transformer.
        return self._inner.profile

    def _aged(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        """The aged copy, plus the firing accounting (chars trimmed)."""
        aged = _age_rounds(messages, self._age)
        if self._stats is not None:
            saved = _wire_chars(messages) - _wire_chars(aged)
            if saved > 0:
                self._stats.requests += 1
                self._stats.saved_chars += saved
        return aged

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> Any:
        return await self._inner.request(
            self._aged(messages), model_settings, model_request_parameters
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        async with self._inner.request_stream(
            self._aged(messages), model_settings, model_request_parameters, run_context
        ) as stream:
            yield stream

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> Any:
        return await self._inner.count_tokens(
            self._aged(messages), model_settings, model_request_parameters
        )


@dataclass
class _RequestMeter:
    """Per-request usage accounting for one turn (measurement only).

    The window a context-size setting constrains is the LARGEST single request,
    not the cumulative prefill ``RunUsage`` reports. This meter records, per
    completed model request, the endpoint-reported input tokens and the
    request-side character count, so the trace can carry the measured peak.
    Requests that fail (e.g. the context-overflow 400) record nothing — they
    are counted by ``overflow_fallbacks`` on the runner instead.
    """

    requests: int = 0
    peak_input_tokens: int = 0
    #: One ``[chars, input_tokens]`` pair per completed request — the
    #: calibration ground truth (exact per-request pairs, vs. the turn-level
    #: cumulative ``input_tokens`` column) and the ledger's growth signal.
    request_log: list[list[int]] = field(default_factory=list)

    def record(self, chars: int, usage: RequestUsage) -> None:
        """Record one completed request's prompt size (chars) and usage."""
        input_tokens = usage.input_tokens or 0
        self.requests += 1
        self.request_log.append([chars, input_tokens])
        self.peak_input_tokens = max(self.peak_input_tokens, input_tokens)


def _wire_chars(messages: list[ModelMessage]) -> int:
    """Total prompt-side characters in a request (measurement only).

    Sums the string content of every message part (system/user text, tool
    returns, assistant text and tool-call args). Chat-template overhead (role
    markers, envelopes) is deliberately NOT counted: the estimator calibrated
    against this metric therefore errs high — the safe direction for a window
    check, where an under-estimate is a hard 400.
    """
    total = 0
    for message in messages:
        for part in getattr(message, "parts", []):
            content = getattr(part, "content", None)
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, str):
                        total += len(block)
                    else:
                        total += len(str(getattr(block, "text", "") or ""))
            args = getattr(part, "args", None)
            if isinstance(args, str):
                total += len(args)
            elif args is not None:
                total += len(str(args))
    return total


class _MeteredModel(Model[Any]):
    """Measurement-only model wrapper, the ``_AgedContextModel``
    precedent applied to usage accounting.

    Delegates everything untouched; after each completed ``request`` /
    ``request_stream`` it hands the endpoint-reported per-request usage to the
    turn's :class:`_RequestMeter`. Wrapped OUTSIDE the aging wrapper so it
    observes every request exactly once regardless of the economy. Zero
    behaviour change: messages, settings, and responses pass through as-is.
    """

    def __init__(self, inner: Model[Any], meter: _RequestMeter) -> None:
        super().__init__()
        self._inner = inner
        self._meter = meter

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def system(self) -> str:
        return self._inner.system

    @property
    def profile(self) -> ModelProfile:
        # Delegate: the inner model's profile carries the pinned max_tokens
        # mapping and any name-inferred schema transformer.
        return self._inner.profile

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        response = await self._inner.request(messages, model_settings, model_request_parameters)
        _mark_llm_used()
        self._meter.record(_wire_chars(messages), response.usage)
        return response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        async with self._inner.request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as stream:
            yield stream
        _mark_llm_used()
        # The consumer has finished the stream → its usage is final.
        self._meter.record(_wire_chars(messages), stream.usage)

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> Any:
        return await self._inner.count_tokens(messages, model_settings, model_request_parameters)


def _idf_ordered(passages: Any, question: str) -> tuple[Any, ...]:
    """``passages`` re-ordered by focus.py's IDF question-term score against
    ``question`` — stable, retrieval rank the tiebreak. Pure
    re-ordering: no I/O, no encoder."""
    from vesta.answer.focus import idf_passage_scores

    scores = idf_passage_scores(question, [sp.passage.text for sp in passages])
    order = sorted(range(len(passages)), key=lambda i: (-scores[i], i))
    return tuple(passages[i] for i in order)


#: The P5 detector intentionally recognizes only question-shaped boundaries.


@dataclass
class _TurnContext:
    """Per-turn state shared by :func:`run_one_turn` and
    :func:`iter_agent_turn_events`.

    Holds everything a turn needs after Round-0 pre-seed: the tool runtime, the
    citation-ordered card map, the tool-call log, the tool caps' counters, a
    status buffer (streaming path only — the tool closures append
    :class:`StatusEvent`\\ s here; ``run_one_turn`` ignores it), the pre-seeded
    message, the resolved model, and the turn start timestamp. :meth:`build_agent`
    constructs the pydantic-ai ``Agent`` with the ``search``/``read_article``
    tool closures closing over this state, so both runners can never silently
    diverge on the agent/system-prompt/tool-cap logic.
    """

    tool_runtime: Any
    turn_cards: dict[tuple[int, str], SourceCardDTO]
    calls: list[ToolCallLog]
    search_count: int
    read_count: int
    status_buf: list[object]
    seed_text: str
    seed_hit: bool
    sys_prompt: str
    user_message: str
    #: The raw turn question (``user_message`` embeds the pre-seed on turn 1) —
    #: used by the ``read_article`` tool's focused-view window scoring.
    question: str
    model: Any
    model_settings: ModelSettings
    #: Resolved token-economy budget (see :mod:`vesta.answer.economy`). Bounds
    #: the Round-0 pre-seed, the ``read_article`` window, and the output tokens.
    budget: EconomyBudget
    started: float
    model_id: str = ""
    #: Per-request usage accounting for this turn's model. The
    #: model wrapped by ``_build_turn`` records into it; the runners surface
    #: ``peak_input_tokens`` / ``requests`` / ``request_log`` in the trace.
    meter: _RequestMeter = field(default_factory=_RequestMeter)
    follow_up: bool = False
    #: Card keys already streamed to the client via a ``sources`` event. ``None``
    #: until the turn's first ``sources(merge=False)`` fires — set by the runner
    #: for a Round-0 pre-seed, or by ``_do_search`` live on a follow-up. Drives
    #: the trailing ``merge=True`` delta (everything discovered AFTER the first
    #: sources event) so cards are never double-counted.
    first_sources_keys: set[tuple[int, str]] | None = None
    #: Timed steps for the answer trace's per-step breakdown (pre_seed /
    #: agent_llm / search / read_article). Each is a dict mirroring the
    #: retrieval ``TraceStage`` shape (name/component/params/inputs/outputs/
    #: duration_ms) plus an optional nested ``stages`` list for search steps.
    steps: list[dict[str, Any]] = field(default_factory=list)
    #: Abstention-calibration record — the resolved
    #: ``answer.agent.evidence_directive`` mode, the seed's top cross-encoder
    #: score (``None`` on follow-ups and empty seeds), and whether the
    #: must-state clause was appended to the strong-evidence directive.
    evidence_directive_trace: dict[str, Any] = field(
        default_factory=lambda: {"mode": "strong", "top_score": None, "fired": False}
    )
    #: Tool-call dedup (token economy, tail questions): queries the ``search``
    #: tool has already run and sources ``read_article`` has already returned
    #: this turn. A temperature-0 greedy loop repeats IDENTICAL calls (same
    #: query, same source number), and every round re-prefills the whole
    #: transcript — an exact repeat returns a small steering string instead of
    #: hitting retrieval again and appending another multi-kb result.
    searched_queries: set[str] = field(default_factory=set)
    read_sources: set[int] = field(default_factory=set)
    #: Cumulative tool-result characters INSERTED this turn (compact search
    #: results + read_article results; steering/dedup strings and the Round-0
    #: pre-seed do not count). The tail explosion (67.9k input on one
    #: question) is unbounded cumulative inserts re-prefilled every round —
    #: ``budget.tool_budget_chars`` caps them.
    tool_chars: int = 0
    #: The same running total in ESTIMATED tokens: feeds the
    #: user-set ``answer.agent.tool_budget_tokens`` cap.
    tool_tokens: int = 0
    #: Passages dropped from the pre-seed TAIL so turn 1 fits
    #: ``window_tokens - output_reserve`` by construction (0 when no window
    #: is budgeted or nothing needed dropping).
    preseed_dropped: int = 0
    #: The fixed prompt-side char base (system + pre-seeded
    #: user message + pre-turn history) the window ledger projects the next
    #: request from when no completed request is on the meter yet.
    prompt_chars: int = 0
    #: Latched by the follow-up branch when its first request does NOT fit
    #: ``window_tokens - output_reserve`` — nothing there is shedable (the
    #: history is caller-owned), so the over-budget shape ships with this
    #: marker in the budget audit.
    first_request_over_budget: bool = False
    #: Latched by the overflow fallback when it dropped the pre-turn history
    #: because prompt + history would overflow the window plan again.
    fallback_history_dropped: bool = False
    #: Latched once a result is rejected for budget: further calls are
    #: steered away BEFORE executing (no more retrieval on a dead turn).
    tool_budget_exhausted: bool = False
    #: The turn's cap on tool-call rounds (0 = uncapped; the
    #: harness ``_MAX_SEARCH_CALLS``/``_MAX_READ_CALLS`` caps alone). Derived
    #: from the window ledger when a profile is active and the knob is 0.
    max_tool_rounds: int = 0
    #: How many tool calls the round cap steered away this turn — the
    #: persisted firing count (the steering strings themselves leave no
    #: ``tool_calls`` row).
    round_cap_fires: int = 0
    #: Tool calls that actually EXECUTED this turn (searches + reads) — the
    #: round-cap counter.
    tool_rounds: int = 0
    #: Latched when a tool call was steered away by the round cap (a
    #: compact-reask trigger).
    round_cap_fired: bool = False
    #: Compact-and-re-ask resolved on (auto under a windowed
    #: profile; on/off force it — ``answer.agent.compact_reask``).
    compact_reask: bool = False
    #: Deterministic answer cleanup, resolved once per turn.
    answer_cleanup: bool = True
    #: The existing evidence-directive score floor, retained
    #: with its Round-0 top score so the compact abstention retry can apply the
    #: same calibration without recomputing a different retrieval pool.
    abstention_floor: float = 0.85
    #: The ``read_article`` texts INSERTED this turn, in call
    #: order — ``(card n, capped text)``. The re-ask's evidence, alongside the
    #: pre-seed; only inserts the model actually saw are recorded (rejected
    #: reads are not evidence).
    read_excerpts: list[tuple[int, str]] = field(default_factory=list)
    #: What the aging wrapper actually trimmed this turn —
    #: the lever's firing count (0/0 when aging is off or never bit).
    aging: _AgingMeter = field(default_factory=_AgingMeter)

    #: Pre-seed shape knobs, resolved once in ``_build_turn``:
    preseed_order: str = "idf"
    preseed_show_archive_id: bool = False
    coverage_search: bool = True
    coverage_search_max: int = 1

    def _tool_budget_blocks(self, prospective_chars: int | None = None) -> bool:
        """True when the turn's tool-insert budget blocks this call.

        ``None`` probes whether the budget is already exhausted (checked
        BEFORE executing); an int probes whether that result — whose length
        is only known after retrieval — would push the running total over.
        An over-budget result is rejected whole, never partially inserted.

        Two caps share this probe/reject/latch shape:

        * the **window ledger** (active when
          ``budget.window_tokens > 0``): would this insert push the NEXT
          request past ``window - output_reserve``? Projected from the
          last completed request's exact wire size (the meter), so the
          pre-seed, history, aging and any re-ask are all accounted for,
          plus one round's envelope slack.
        * the cumulative insert caps — chars (``tool_budget_chars``, the
          legacy re-prefill multiplier cap) and, when the user set it,
          tokens (``tool_budget_tokens``; the char knob wins when both
          are set).
        """
        if self.tool_budget_exhausted:
            return True
        window = self.budget.window_tokens
        if window > 0 and prospective_chars is not None:
            limit = window - self.budget.output_reserve
            base = max(
                self.meter.request_log[-1][0] if self.meter.request_log else 0,
                self.prompt_chars,
            )
            projected = estimate_tokens_for_chars(
                base + prospective_chars + _WINDOW_LEDGER_SLACK_CHARS
            )
            if projected > limit:
                self.tool_budget_exhausted = True
                return True
        cap = self.budget.tool_budget_chars
        if cap > 0:
            if prospective_chars is not None and self.tool_chars + prospective_chars > cap:
                self.tool_budget_exhausted = True
                return True
        elif (
            self.budget.tool_budget_tokens > 0
            and prospective_chars is not None
            and self.tool_tokens + estimate_tokens_for_chars(prospective_chars)
            > self.budget.tool_budget_tokens
        ):
            # Token twin of the char cap (the char knob wins when both are
            # set). Applies the user-set knob; the per-turn DERIVED value
            # (filled in by _build_turn's window arithmetic) also lands here
            # as belt-and-braces — the D5 window ledger binds at least as
            # tightly against measured request sizes, so the stricter of the
            # two governs and neither can under-bound the window.
            self.tool_budget_exhausted = True
            return True
        return False

    def _round_cap_blocks(self) -> bool:
        """True when the round cap blocks this call (probed BEFORE
        executing, like the budget probe). Distinct from
        ``_tool_budget_blocks``: it never rejects an already-fetched result —
        it stops the NEXT call — and it latches ``round_cap_fired`` so the
        compact-reask trigger can see the cap bit this turn."""
        if self.max_tool_rounds <= 0:
            return False
        if self.tool_rounds >= self.max_tool_rounds:
            self.round_cap_fired = True
            self.round_cap_fires += 1
            return True
        return False

    def _capped_read(self, text: str, card: SourceCardDTO) -> str:
        """``read_article``'s cap: bound a fetched article by the
        best-scoring window at ``budget.read_max_chars``, NEVER a head cut —
        focus.py's ``focused_view`` guarantees the article lead plus
        IDF-scored question-term chunks within the budget. The
        retrieval-scored passage for this card — its snippet — is forced in
        via ``must_include_spans``; the span is re-derived by locating the
        snippet in the text because the tool runtime's text is the
        composition root's focused excerpt (elision markers invalidate the
        original passage char offsets). Since AUDIT_0824 N11 that excerpt's
        stage-1 window already forced this same snippet in (the seam's
        ``must_include``), so for articles beyond the stage-1 cap the
        ``find`` here succeeds instead of silently dropping the passage.
        This was extracted from the ``read_article`` tool closure so the
        mechanical pre-read makes the EXACT same call. ``read_max_chars`` 0
        (the registered default) or an article already under it passes the
        text through unchanged.
        """
        cap = self.budget.read_max_chars
        if cap and text and len(text) > cap:
            from vesta.answer.focus import focused_view

            probe = card.snippet.strip()[:200]
            idx = text.find(probe) if probe else -1
            spans: tuple[tuple[int, int], ...] = ((idx, idx + len(probe)),) if idx >= 0 else ()
            view = focused_view(
                text,
                self.question,
                cap,
                breadcrumb=card.title,
                must_include_spans=spans,
            )
            return view.excerpt
        return text

    def add_step(
        self,
        name: str,
        component: str,
        duration_ms: float,
        *,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        stages: list[dict[str, Any]] | None = None,
        at: int | None = None,
    ) -> None:
        """Append (or insert at ``at``) one timed step to the turn's breakdown."""
        step: dict[str, Any] = {
            "name": name,
            "component": component,
            "duration_ms": round(duration_ms, 1),
            "params": {},
            "inputs": dict(inputs or {}),
            "outputs": dict(outputs or {}),
        }
        if stages:
            step["stages"] = stages
        if at is None:
            self.steps.append(step)
        else:
            self.steps.insert(at, step)

    async def _do_search(  # noqa: PLR0915
        self,
        query: str,
        *,
        compact: bool = False,
        exact: bool = False,
        fit_chars: int | None = None,
    ) -> str:
        """Core search body shared by the Round-0 pre-seed and the ``search`` tool.

        ``compact`` (the follow-up ``search`` tool) returns just titles + short
        snippets so the model can pick what to ``read_article`` without dumping
        full passage text into the accumulating history; the Round-0 pre-seed
        keeps full passages (the model answers from them directly).

        ``exact`` (the Round-0 pre-seed only) routes through
        ``tool_runtime.search_exact`` instead of ``tool_runtime.search`` —
        bypassing the prefix-shortening/term-surfacing recovery ladder that
        ``search`` applies. That ladder is tuned for the model's own short,
        fact-shaped tool-call queries; run against the raw, full question it
        degrades a mid-sentence entity into a leading-stopword query (e.g.
        "In what year did the English minister and logician Isaac Watts
        publish..." shortens to "In what year", which scores high on
        irrelevant date articles and displaces the correct result). Round 0
        instead carries its own conditional recovery rung inside
        ``search_exact`` (one LLM call naming the article the
        fact would live in, only when the Round-0 result visibly failed — see
        ``api/answer.py::_maybe_reformulate_round0``). Falls back to
        ``search`` when the runtime doesn't provide ``search_exact`` (e.g.
        older test fakes).

        ``fit_chars`` (Round-0 pre-seed only): when set, the
        pre-seed drops whole passages from its tail until the rendered text
        fits this many characters — the pre-flight half of the window
        guarantee (turn 1 can never overflow). ``None`` (and every compact
        search) skips the fit: the window ledger governs tool-round
        inserts instead.
        """
        if self.tool_runtime is None:
            return "Search is unavailable — no archives are loaded."
        search_fn = (self.tool_runtime.search_exact if exact else None) or self.tool_runtime.search
        started = time.monotonic()
        result = await search_fn(query, "")
        duration_ms = (time.monotonic() - started) * 1000.0
        trace_dict = result.trace if isinstance(result, SearchToolResult) else None
        self.add_step(
            "pre_seed" if exact else "search",
            "search_exact" if exact else "search",
            duration_ms,
            inputs={"query": query},
            outputs={
                "passages": len(result.passages) if isinstance(result, SearchToolResult) else 0,
                "cards": len(result.cards) if isinstance(result, SearchToolResult) else 0,
            },
            stages=cast("list[dict[str, Any]] | None", (trace_dict or {}).get("stages")),
        )
        if not isinstance(result, SearchToolResult):
            self.calls.append(ToolCallLog(query=query, result_preview=result[:300]))
            return cast(str, result)

        # Coverage enrichment is Round-0-only.  It deliberately runs before
        # any selection/rendering and even when the first pool is empty: an
        # entity-title retry can be the only evidence returned.
        pool: list[Any] = list(result.passages)
        pool_cards: list[Any] = list(result.cards)
        pool, pool_cards = await self._coverage_search(
            query, pool, pool_cards, enabled=exact and self.coverage_search
        )
        if not pool:
            self.calls.append(ToolCallLog(query=query, result_preview=result.text[:300]))
            return result.text

        cards_by_key = {(c.zim_id, c.path): c for c in pool_cards}
        for key, sc in cards_by_key.items():
            if key not in self.turn_cards:
                self.turn_cards[key] = SourceCardDTO(
                    n=len(self.turn_cards) + 1,
                    zim_id=sc.zim_id,
                    path=sc.path,
                    title=sc.title,
                    snippet=sc.snippet,
                    breadcrumb=sc.breadcrumb,
                    score=sc.score,
                    source=sc.source,
                )
        # Follow-ups have no Round-0 ``sources`` event. Surface their first
        # discovery live, and latch exactly that set so the final merge delta
        # does not replay it. Turn 1's runner owns its initial event.
        if self.follow_up and self.first_sources_keys is None and self.turn_cards:
            self.status_buf.append(
                SourcesEvent(
                    cards=cast(
                        tuple[SourceCard, ...],
                        tuple(sorted(self.turn_cards.values(), key=lambda c: c.n)),
                    ),
                    merge=False,
                )
            )
            self.first_sources_keys = set(self.turn_cards)
        # Token economy: BOTH branches are shaped by the resolved budget —
        # the compact search-tool branch by search_entries x
        # search_snippet_chars (registered defaults 6x400 = the old
        # hard-coded shape; CPU economy 5x350), the Round-0 pre-seed branch
        # (full passages) by preseed_passages / preseed_passage_max_chars.
        if compact:
            shown = pool[: self.budget.search_entries]
        elif self.preseed_order == "idf":
            # Re-order the Round-0 pre-seed by focus.py's IDF
            # question-term score BEFORE the slice (+9/189 answer containment
            # at an identical char budget, measured). The compact search-tool
            # branch above keeps rank order, and card registration (the loop
            # over result.cards higher up) is untouched: cards stay numbered
            # by discovery order, so [n] never shifts — only the
            # ORDER the model reads passages in changes (no I/O,
            # no encoder).
            shown = list(_idf_ordered(pool, query)[: self.budget.preseed_passages])
        else:
            shown = pool[: self.budget.preseed_passages]
        cap = 0 if compact else self.budget.preseed_passage_max_chars

        def _render(passages: Any) -> str:
            lines = [f"Search returned {len(pool)} passages (showing top {len(passages)}):"]
            for sp in passages:
                p = sp.passage
                key = (p.zim_id, p.path)
                card = self.turn_cards.get(key)
                n = card.n if card else "?"
                title = card.title if card else p.breadcrumb.split(">")[0].strip()
                # The (archive-N) suffix is useless to the model
                # (one archive per scoped turn) and 26-30% of lfm2.5-1.2b
                # answers copy the token verbatim. One render path
                # gates both the compact and full branches.
                lines.append(
                    f'[{n}] "{title}"'
                    if not self.preseed_show_archive_id
                    else f'[{n}] "{title}" (archive-{p.zim_id})'
                )
                if compact:
                    snippet = p.text.replace("\n", " ")[: self.budget.search_snippet_chars]
                    lines.append(f"    {snippet}…")
                elif cap and len(p.text) > cap:
                    lines.append(p.text[: cap - 1].rstrip() + "…")
                else:
                    lines.append(p.text)
                lines.append("")
            return "\n".join(lines)

        text = _render(shown)
        if fit_chars is not None and not compact:
            # Pre-flight fit: drop WHOLE passages from the tail
            # until the pre-seed fits the window's evidence budget. Drop,
            # never truncate further — a half-passage is worse evidence
            # than one fewer passage, and ``preseed_passage_max_chars``
            # above is already the single-huge-chunk outlier guard. The
            # dropped passages' cards stay registered: the agent can still
            # read_article them, and citation numbering never shifts.
            dropped = 0
            while len(text) > fit_chars and shown:
                shown = shown[:-1]
                dropped += 1
                text = _render(shown)
            self.preseed_dropped = dropped
        # The recovery rungs inside the composition root's search callable
        # (term-surfacing, Round-0 reformulation visibility) append their
        # candidate-article titles only to ``result.text``. This branch
        # re-renders from passages instead of using that text, so without
        # this re-attach the surfaced articles never reach the model while
        # their retrieval latency is still paid.
        text += result.candidates_text
        self.calls.append(ToolCallLog(query=query, result_preview=text[:300]))
        return text

    async def _coverage_search(
        self,
        question: str,
        pool: list[Any],
        pool_cards: list[Any],
        *,
        enabled: bool,
    ) -> tuple[list[Any], list[Any]]:
        """Merge retrieval for uncovered title-entity spans into Round 0.

        The composition root deliberately imports the public extractor used by
        ``title_entity_suggest`` rather than maintaining a second entity
        heuristic. Coverage is against the complete first-result pool, before
        ordering, slicing, or window fitting; only cards are source-deduped,
        while passages retain their passage-level identity.
        """
        if not enabled:
            return pool, pool_cards

        from vesta.retrieval.impls.title_entity_suggest import extract_spans

        def normalize(text: str) -> str:
            return " ".join(text.split()).casefold()

        started = time.monotonic()
        spans = extract_spans(question)
        first_pool_texts = [normalize(scored.passage.text) for scored in pool]
        uncovered = [
            span
            for span in spans
            if all(normalize(span) not in passage_text for passage_text in first_pool_texts)
        ]
        search_exact = self.tool_runtime.search_exact or self.tool_runtime.search
        fired: list[str] = []
        seen_passages = {
            (
                scored.passage.zim_id,
                scored.passage.path,
                scored.passage.ordinal,
                scored.passage.char_start,
                scored.passage.char_end,
            )
            for scored in pool
        }
        seen_cards = {(card.zim_id, card.path) for card in pool_cards}
        added = 0
        for span in uncovered[: self.coverage_search_max]:
            fired.append(span)
            result = await search_exact(span, "")
            if not isinstance(result, SearchToolResult):
                continue
            for scored in result.passages:
                passage = scored.passage
                passage_key = (
                    passage.zim_id,
                    passage.path,
                    passage.ordinal,
                    passage.char_start,
                    passage.char_end,
                )
                if passage_key not in seen_passages:
                    seen_passages.add(passage_key)
                    pool.append(scored)
                    added += 1
            for card in result.cards:
                card_key = (card.zim_id, card.path)
                if card_key not in seen_cards:
                    seen_cards.add(card_key)
                    pool_cards.append(card)
        self.add_step(
            "coverage_search",
            "search_exact",
            (time.monotonic() - started) * 1000.0,
            inputs={"spans": fired, "max": self.coverage_search_max},
            outputs={"passages_added": added, "uncovered": len(uncovered)},
        )
        return pool, pool_cards

    def build_agent(self, *, with_tools: bool) -> Agent:
        """Build one pydantic-ai ``Agent`` for this turn.

        ``with_tools=True`` registers the ``search``/``read_article`` tool
        closures (harness-capped at :data:`_MAX_SEARCH_CALLS` /
        :data:`_MAX_READ_CALLS`); ``with_tools=False`` registers none (the
        no-tool fallback agent). The tool closures close over this context's
        counters, cards, call log, and (streaming path) ``status_buf``.
        """
        agent = Agent(self.model, system_prompt=self.sys_prompt, model_settings=self.model_settings)
        if not with_tools:
            return agent

        @agent.tool_plain
        async def search(query: str) -> str:
            """Search the offline archive for passages relevant to `query`.

            Returns ranked passages, each tagged with a bracketed citation
            number, and their source article title. Rewrite and retry with a
            different query if the results don't contain the answer.
            """
            self.status_buf.append(StatusEvent("searching", "Searching…"))
            # Tool-call dedup (token economy): a greedy temperature-0 loop
            # repeats IDENTICAL queries, and every round re-prefills everything.
            # An exact repeat returns a small steering string immediately — no
            # retrieval, no multi-kb result appended, no attempt consumed.
            if query in self.searched_queries:
                return (
                    f"You already searched for '{query}' — the results are above. "
                    "Do not repeat a search. Answer from the sources you have, "
                    "with [n] citations."
                )
            self.searched_queries.add(query)
            # Round cap: steer BEFORE executing once the turn's
            # window-derived (or user-set) tool-round allowance is spent.
            if self._round_cap_blocks():
                return _ROUND_CAP_STEERING
            # Turn-level tool-insert budget: once exhausted, steer BEFORE
            # executing (retrieval on a dead turn is pure waste).
            if self._tool_budget_blocks():
                return _TOOL_BUDGET_STEERING
            self.search_count += 1
            self.tool_rounds += 1
            if self.search_count > _MAX_SEARCH_CALLS:
                return (
                    "You have used all search attempts. Answer from the sources you "
                    "already have — use read_article on a relevant source number first."
                )
            text = await self._do_search(query, compact=True)
            # Budget check on the actual result length: an over-budget result
            # is rejected whole (never partially inserted) and the budget
            # latches so later calls steer before executing.
            if self._tool_budget_blocks(len(text)):
                return _TOOL_BUDGET_STEERING
            self.tool_chars += len(text)
            self.tool_tokens += estimate_tokens_for_chars(len(text))
            return text

        @agent.tool_plain
        async def read_article(n: int) -> str:  # noqa: PLR0911
            """Read the FULL text of source [n] (the bracket number shown next to it).

            Use this when a source looks relevant but its excerpt does not contain
            the exact fact you need.
            """
            self.status_buf.append(StatusEvent("searching", f"Reading source {n}…"))
            if self.tool_runtime is None:
                return "Read is unavailable — no archives are loaded."
            # Tool-call dedup (token economy), mirroring the search tool: an
            # exact repeat read returns a steering string, not the same
            # multi-kb article again.
            if n in self.read_sources:
                return (
                    f"You already read source [{n}] — its text is above. Do not "
                    "repeat reads. Answer now from the sources you have, with "
                    "[n] citations."
                )
            # Round cap, mirroring the search tool (the compact re-ask
            # triggers on this latch).
            if self._round_cap_blocks():
                return _ROUND_CAP_STEERING
            # Turn-level tool-insert budget, mirroring the search tool: once
            # exhausted, steer BEFORE executing the read.
            if self._tool_budget_blocks():
                return _TOOL_BUDGET_STEERING
            self.read_count += 1
            self.tool_rounds += 1
            if self.read_count > _MAX_READ_CALLS:
                return (
                    "You have read enough sources. Stop calling tools and answer the "
                    "question now from the sources you already have, with [n] citations."
                )
            card = next((c for c in self.turn_cards.values() if c.n == n), None)
            if card is None:
                available = ", ".join(
                    str(c.n) for c in sorted(self.turn_cards.values(), key=lambda c: c.n)
                )
                return f"No source [{n}]. Available source numbers: {available}."
            self.read_sources.add(n)
            started = time.monotonic()
            text = self._capped_read(
                await self.tool_runtime.read_article(
                    card.zim_id, card.path, must_include=card.snippet
                ),
                card,
            )
            # Budget check on the final (per-read-capped) length: a read that
            # would overflow is rejected whole — never a truncated partial —
            # and the budget latches so later calls steer before executing.
            if self._tool_budget_blocks(len(text or "")):
                return _TOOL_BUDGET_STEERING
            self.tool_chars += len(text or "")
            self.tool_tokens += estimate_tokens_for_chars(len(text or ""))
            # The inserted read is evidence — record it for the
            # compact re-ask (only what the model actually saw).
            self.read_excerpts.append((n, text or ""))
            self.add_step(
                "read_article",
                "read_article",
                (time.monotonic() - started) * 1000.0,
                inputs={"source": n, "title": card.title},
                outputs={"chars": len(text or "")},
            )
            self.calls.append(
                ToolCallLog(query=f"read_article([{n}] {card.title})", result_preview=text[:300])
            )
            return text or f"[source {n} is empty]"

        return agent


async def _build_turn(  # noqa: PLR0912, PLR0915
    state: AppState,
    sn: Any,
    question: str,
    *,
    model_id: str | None = None,
    endpoint: str | None = None,
    api_key: str = "",
    enable_thinking: bool | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    message_history: list[ModelMessage] | None = None,
    profile_override: str | None = None,
    scope: str | None = None,
) -> _TurnContext:
    """Resolve the profile/scope/tool-runtime and build a fully-populated
    :class:`_TurnContext` ready for :meth:`_TurnContext.build_agent`.

    Turn model: turn 1 (``message_history`` empty) runs the Round-0 pre-seed —
    retrieval on the raw question so the model starts from evidence. A follow-up
    (history present AND ``answer.agent.contextual_followups`` on) SKIPS the
    pre-seed: the agent resolves references from the conversation, answers
    directly or searches with a self-contained query. See :data:`_FOLLOWUP_DIRECTIVE`.

    ``model_id``/``endpoint``/``api_key``/``enable_thinking`` default to
    ``_resolve_llm(sn)`` when ``model_id`` is ``None`` (the streaming runner's
    case); :func:`run_one_turn` passes its explicit call-site values through so
    the benchmark's configured model is honoured unchanged.
    """
    hardware: str | None = None
    if model_id is None or endpoint is None:
        model_id, endpoint, api_key, enable_thinking, hardware = _resolve_llm(sn)
    else:
        # Explicit overrides (run_one_turn/benchmark): the model comes from the
        # call site, but the token-economy gate still needs the runtime's
        # hardware tag.
        hardware = _runtime_hardware()
    profile = _resolve_profile(profile_override)
    ret_scope = _parse_scope(scope, state.registry)
    tool_runtime = _build_tool_runtime(state, sn, ret_scope, profile, question)
    budget = resolve_budget(sn, hardware, _runtime_window_tokens(sn))

    # The model is built at the _make_model seam (so FunctionModel test stubs
    # flow through the same request-side transform the real model gets), but
    # the aging + meter wraps are applied at the END of this function:
    # the aging budget derives from the window ledger, which needs the
    # pre-seed — built below on ctx. The wrap order is unchanged (aging inner,
    # meter outermost).
    model: Any = _make_model(model_id, endpoint, api_key)
    # Measurement-only per-request accounting, wrapped outermost
    # so every request (aged or not, main run or fallback) is metered exactly
    # once. Zero behaviour change — the wrapper delegates untouched.
    meter = _RequestMeter()
    model_settings: ModelSettings = {"max_tokens": budget.max_output_tokens, "temperature": 0}
    if enable_thinking is not None:
        # Match the answer gateway's chat_template_kwargs forwarding:
        # Qwen3/3.5 reason by default and burn the whole output budget on hidden
        # reasoning_content unless explicitly disabled.
        model_settings["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": enable_thinking}
        }
    ctx = _TurnContext(
        tool_runtime=tool_runtime,
        turn_cards={},
        calls=[],
        search_count=0,
        read_count=0,
        status_buf=[],
        seed_text="",
        seed_hit=False,
        sys_prompt="",
        user_message="",
        question=question,
        model=model,
        model_id=model_id,
        model_settings=model_settings,
        budget=budget,
        started=time.monotonic(),
        meter=meter,
    )

    # Resolve the pre-seed mechanisms once per turn.
    from vesta.answer import (
        ANSWER_AGENT_ANSWER_CLEANUP,
        ANSWER_AGENT_COVERAGE_SEARCH,
        ANSWER_AGENT_COVERAGE_SEARCH_MAX,
        ANSWER_AGENT_EVIDENCE_DIRECTIVE,
        ANSWER_AGENT_EVIDENCE_DIRECTIVE_MIN_SCORE,
        ANSWER_AGENT_PRESEED_ORDER,
        ANSWER_AGENT_PRESEED_SHOW_ARCHIVE_ID,
    )

    ctx.preseed_order = _setting_str(sn, ANSWER_AGENT_PRESEED_ORDER)
    ctx.preseed_show_archive_id = _setting_bool(sn, ANSWER_AGENT_PRESEED_SHOW_ARCHIVE_ID)
    ctx.answer_cleanup = _setting_bool(sn, ANSWER_AGENT_ANSWER_CLEANUP)
    ctx.coverage_search = _setting_bool(sn, ANSWER_AGENT_COVERAGE_SEARCH)
    ctx.coverage_search_max = _setting_int(sn, ANSWER_AGENT_COVERAGE_SEARCH_MAX)

    # Abstention calibration. 'strong' appends the must-state
    # clause to the strong-evidence directive once Round-0's top cross-encoder
    # score clears the floor; 'standard' keeps the unstrengthened directive.
    evidence_mode = _setting_str(sn, ANSWER_AGENT_EVIDENCE_DIRECTIVE)
    evidence_min_score = _setting_float(sn, ANSWER_AGENT_EVIDENCE_DIRECTIVE_MIN_SCORE)
    ctx.evidence_directive_trace["mode"] = evidence_mode
    ctx.abstention_floor = evidence_min_score

    # ── Turn model ─────────────────────────────────────────────────
    # Turn 1 (no history) keeps the Round-0 pre-seed: run retrieval on the raw
    # question BEFORE the agent runs, so the model starts from evidence and the
    # "cards in ~1s" UX holds. A FOLLOW-UP (history present, setting on) skips
    # the pre-seed entirely: the agent resolves references from the conversation,
    # answers directly when the fact is already established, or searches with a
    # self-contained, context-resolved query for a missing fact.
    if system_prompt != SYSTEM_PROMPT:
        base_prompt = system_prompt
        strong_evidence_directive = _STRONG_EVIDENCE_DIRECTIVE
    elif budget.compact_prompt:
        base_prompt = COMPACT_SYSTEM_PROMPT
        strong_evidence_directive = _STRONG_EVIDENCE_DIRECTIVE
    else:
        base_prompt = SYSTEM_PROMPT
        strong_evidence_directive = _STRONG_EVIDENCE_DIRECTIVE
    ctx.follow_up = bool(message_history) and _contextual_followups_enabled(sn)
    if ctx.follow_up:
        ctx.sys_prompt = base_prompt + _FOLLOWUP_DIRECTIVE
        ctx.user_message = question
        # First-request fit check — the twin of turn-1's pre-seed fit below.
        # A follow-up carries system + directive + question + the FULL
        # pre-turn history, and none of it is shedable here (the history is
        # caller-owned), so an over-budget shape cannot be repaired — it is
        # latched into the budget audit instead (degrade-don't-fail).
        if budget.window_tokens > 0 and not _request_fits_window(
            ctx, ctx.user_message, message_history or []
        ):
            ctx.first_request_over_budget = True
    else:
        # ── Round-0 pre-seed: the model starts from evidence, not blind. ──
        # Under a window plan, bound the pre-seed so the FIRST
        # request fits ``window - output_reserve`` by construction.
        clause_reserve = len(_STRONG_EVIDENCE_MUST_STATE_CLAUSE) if evidence_mode == "strong" else 0
        # The legacy path (history present, contextual follow-ups off) re-sends
        # the whole pre-turn conversation on this first request too, so the fit
        # sheds pre-seed passages for it exactly like prompt/directive/question.
        seed_fit_chars: int | None = None
        if budget.window_tokens > 0:
            seed_fit_chars = max(
                0,
                3 * (budget.window_tokens - budget.output_reserve)
                - len(base_prompt)
                - len(strong_evidence_directive)
                - clause_reserve
                - len(_USER_MESSAGE_HEAD)
                - len(_USER_MESSAGE_TAIL)
                - len(question)
                - _wire_chars(message_history or []),
            )
        ctx.seed_text = await ctx._do_search(question, exact=True, fit_chars=seed_fit_chars)
        ctx.seed_hit = bool(ctx.turn_cards)
        directive = strong_evidence_directive
        if ctx.seed_hit:
            top_score = max(
                (card.score for card in ctx.turn_cards.values() if card.score is not None),
                default=0.0,
            )
            ctx.evidence_directive_trace["top_score"] = top_score
            if evidence_mode == "strong" and directive and top_score >= evidence_min_score:
                directive += _STRONG_EVIDENCE_MUST_STATE_CLAUSE
                ctx.evidence_directive_trace["fired"] = True
        ctx.sys_prompt = base_prompt + directive if ctx.seed_hit else base_prompt
        ctx.user_message = (
            f"{_USER_MESSAGE_HEAD}{ctx.seed_text}{_USER_MESSAGE_TAIL}{question}"
            if ctx.seed_hit
            else question
        )
    if budget.window_tokens > 0 and budget.tool_budget_tokens == 0:
        # Fill in the derived tool-insert allowance with the
        # per-turn window arithmetic — what the window leaves after the
        # (post-drop) pre-seed. Informational for enforcement (the
        # ledger already bounds the same quantity against MEASURED request
        # sizes), but the recorded number is the audit trail for how much
        # tool room the plan actually bought.
        budget = replace(
            budget,
            tool_budget_tokens=max(
                0,
                budget.window_tokens
                - budget.output_reserve
                - estimate_tokens_for_chars(
                    len(ctx.sys_prompt) + len(ctx.user_message) + _wire_chars(message_history or [])
                ),
            ),
        )
        ctx.budget = budget
    # ── Tail levers: (a)/(c) from the ledger; (b) needs only
    # the window. All inert unless a window plan is active (``full`` resolves
    # window_tokens=0 — today's behaviour, byte-identical).
    ctx.compact_reask = _compact_reask_enabled(sn, budget.window_tokens)
    budget = _resolve_tail_levers(sn, ctx, budget)
    # Deferred wraps (see the comment at the seam above): aging inner —
    # with its derivation live this turn, its trimming metered into
    # ctx.aging — then the meter outermost.
    if budget.age_tool_chars > 0:
        model = _AgedContextModel(model, budget.age_tool_chars, ctx.aging)
    ctx.model = _MeteredModel(model, meter)
    # The window ledger's fallback base when no request has
    # been metered yet.
    ctx.prompt_chars = (
        len(ctx.sys_prompt) + len(ctx.user_message) + _wire_chars(message_history or [])
    )
    return ctx


def _resolve_tail_levers(sn: Any, ctx: _TurnContext, budget: EconomyBudget) -> EconomyBudget:
    """Derive the round cap and the aging budget from the turn's
    window ledger. Called by :func:`_build_turn` after
    the ledger fill-in; returns the possibly-updated budget and sets
    ``ctx.max_tool_rounds``.

    All three levers LOST their A/Bs against run 64 at
    8k-fullprompt-wide (runs 72/73/74) and ship OFF by default; what follows
    documents the OPT-IN ladder each setting keeps.

    (a) Round cap from the window — how many read-sized inserts the ledger
    admits::

        rounds = tool_budget_tokens // estimate_tokens_for_chars(read_max_chars)

    floored at 1 (a windowed turn always keeps room for ONE evidence-gathering
    call — the cap bounds the tail, not the median turn) and never above the
    harness allowance (``_MAX_SEARCH_CALLS + _MAX_READ_CALLS``). The default
    -1 (and any value ≤ 0 the resolver leaves at 0) means no extra cap; an
    explicit 0 + an active window derives; the resolved value lands in
    ``trace.budget.max_tool_rounds``; an explicit user value > 0 wins on any
    profile; with no ``read_max_chars`` to size an insert there is nothing to
    derive (the D5 ledger still guards every insert).

    (c) Aging under a small window — an eviction policy derived from the same
    ledger: every aged result keeps an equal share of it, the per-insert
    allowance of a turn that spends its whole harness allowance::

        age_tool_chars = 3 * tool_budget_tokens // (_MAX_SEARCH_CALLS + _MAX_READ_CALLS)

    A user-set positive value already won the :func:`resolve_budget` ladder
    (an active plan defaults it to -1 = off, measured); ``-1`` is
    the explicit-off arm (aging off even under a windowed profile); an
    explicit ``0`` + window derives; without a window 0 stays off (today).
    """
    from vesta.answer import ANSWER_AGENT_MAX_TOOL_ROUNDS

    user_rounds = _setting_int(sn, ANSWER_AGENT_MAX_TOOL_ROUNDS)
    if user_rounds > 0:
        ctx.max_tool_rounds = user_rounds  # explicit wins, any profile
    elif (
        user_rounds == 0  # the derive-me opt-in (default -1 = off, run 72)
        and budget.window_tokens > 0
        and budget.read_max_chars > 0
        and budget.tool_budget_tokens > 0
    ):
        per_read = estimate_tokens_for_chars(budget.read_max_chars)
        ctx.max_tool_rounds = min(
            max(1, budget.tool_budget_tokens // per_read),
            _MAX_SEARCH_CALLS + _MAX_READ_CALLS,
        )
    if budget.window_tokens > 0 and budget.age_tool_chars == 0 and budget.tool_budget_tokens > 0:
        budget = replace(
            budget,
            age_tool_chars=(3 * budget.tool_budget_tokens) // (_MAX_SEARCH_CALLS + _MAX_READ_CALLS),
        )
        ctx.budget = budget
    if budget.age_tool_chars < 0:
        # -1 = explicitly off: normalize to 0 so the wire budget and the
        # trace never carry a negative cap.
        budget = replace(budget, age_tool_chars=0)
        ctx.budget = budget
    return budget


def _plan_abstain_retry(
    ctx: _TurnContext,
    message_history: list[ModelMessage] | None,
) -> tuple[str, list[ModelMessage] | None] | None:
    """The abstention retry's ``(prompt, history)``, or ``None`` to skip it.

    Today's retry re-sends the whole pre-seeded user message on a fresh
    history — roughly 2x the turn-1 prompt, which a small window cannot
    afford (an 8k plan whose pre-seed fits once would 400 on the duplicate).
    Under a window plan, first try the standard shape against the fit test;
    else re-run against the turn's ORIGINAL first request as history — the
    pre-turn history plus the turn's own user message, synthesized here
    because the streaming runner never tracks the run's message list. Only
    the directive is new, so nothing is duplicated and all evidence
    survives; if even that would not fit, skip the retry and keep the
    refusal (degrade-don't-fail — never trade S4's zero overflow fallbacks
    for one more retry).
    """
    prompt = _abstain_retry_prompt(ctx.user_message)
    history = message_history or None
    if ctx.budget.window_tokens <= 0 or _request_fits_window(ctx, prompt, history or []):
        return prompt, history
    dedup: list[ModelMessage] = list(message_history or [])
    dedup.append(ModelRequest(parts=[UserPromptPart(content=ctx.user_message)]))
    if _request_fits_window(ctx, _ABSTAIN_RETRY_DIRECTIVE, dedup):
        return _ABSTAIN_RETRY_DIRECTIVE, dedup
    return None


def _request_fits_window(ctx: _TurnContext, prompt: str, history: list[ModelMessage]) -> bool:
    """Would ``system + prompt + history`` fit ``window - output_reserve``?"""
    limit = ctx.budget.window_tokens - ctx.budget.output_reserve
    if limit <= 0:
        return False
    est = estimate_tokens_for_chars(
        len(ctx.sys_prompt) + len(prompt) + _wire_chars(history) + _WINDOW_LEDGER_SLACK_CHARS
    )
    return est <= limit


def _compact_reask_trigger(ctx: _TurnContext, answer: str) -> str | None:
    """Return the compact-reask trigger, with P6's calibrated abstention last.

    The existing ``round_cap`` and ``ledger`` paths deliberately remain first
    and unchanged.  P6 is narrower than the former generic ``turn_cards``
    abstention: it requires a Round-0 seed, the exact Round-0 top score P10
    recorded after ``_do_search``, and P10's existing score floor.  Live
    follow-up cards and cards gathered after Round 0 therefore cannot create a
    false-confidence retry.
    """
    if ctx.round_cap_fired:
        return "round_cap"
    if ctx.tool_budget_exhausted:
        return "ledger"
    top_score = ctx.evidence_directive_trace["top_score"]
    if (
        looks_abstained(answer)
        and ctx.seed_hit
        and top_score is not None
        and top_score >= ctx.abstention_floor
    ):
        return "abstain_p6"
    return None


def _p6_abstain_reask_message(ctx: _TurnContext) -> tuple[str, int] | None:  # noqa: PLR0912
    """Build P6's focused ``(message, evidence_chars)`` or degrade to ``None``.

    Only complete source blocks from the Round-0 rendered seed are eligible:
    each body is re-focused with :func:`~vesta.answer.focus.focused_view` and
    is emitted immediately after its original ``[n]`` header.  Thus no cited
    card header is orphaned from its evidence body, while source numbers retain
    the existing rendering exactly.

    The deterministic evidence cap is the stricter of the existing
    ``preseed_passage_max_chars`` cap (falling back to the established 2,400
    decompose cap when that unbounded setting is zero) and the exact window
    remainder.  With the 3-chars/token estimator, a request fits
    ``window_tokens - output_reserve`` iff its character count is at most
    ``3 * (window - reserve)``; the remainder subtracts the system prompt,
    directive/question wrapper, and question.  Retained headers, separators,
    and at least one body character are reserved first, then the remaining
    body capacity is divided evenly.  Each focused body is hard-bounded to
    that share, making the final request fit by construction despite
    ``focused_view``'s allowed merge/lead slop.
    """
    header_matches = list(
        re.finditer(r'(?m)^\[\d+\] "[^\n]*"(?: \(archive-\d+\))?$', ctx.seed_text)
    )
    candidates: list[tuple[str, str]] = []
    for index, match in enumerate(header_matches):
        body_end = (
            header_matches[index + 1].start()
            if index + 1 < len(header_matches)
            else len(ctx.seed_text)
        )
        body = ctx.seed_text[match.end() : body_end].strip()
        if body:
            candidates.append((match.group(), body))
    if not candidates:
        return None

    configured_cap = ctx.budget.preseed_passage_max_chars or 2_400
    fixed_chars = (
        len(ctx.sys_prompt)
        + len(_P6_ABSTAIN_REASK_HEAD)
        + len(ctx.question)
        + len(_P6_ABSTAIN_REASK_TAIL)
    )
    limit = ctx.budget.window_tokens - ctx.budget.output_reserve
    evidence_cap = configured_cap
    if limit > 0:
        window_cap = 3 * limit - fixed_chars
        if window_cap <= 0:
            return None
        evidence_cap = min(evidence_cap, window_cap)

    sep = "\n\n"
    retained: list[tuple[str, str]] = []
    reserved = 0
    for header, body in candidates:
        # Header + newline + one body character, plus an inter-source
        # separator after the first block: never emit a bare source header.
        minimum = len(header) + 2 + (len(sep) if retained else 0)
        if reserved + minimum > evidence_cap:
            break
        retained.append((header, body))
        reserved += minimum
    if not retained:
        return None

    header_chars = sum(len(header) + 1 for header, _ in retained) + len(sep) * (len(retained) - 1)
    body_share = max(1, (evidence_cap - header_chars) // len(retained))
    from vesta.answer.focus import focused_view

    blocks: list[str] = []
    for header, body in retained:
        excerpt = focused_view(body, ctx.question, body_share).excerpt[:body_share].strip()
        if not excerpt:
            excerpt = body[:body_share].strip()
        if excerpt:
            blocks.append(f"{header}\n{excerpt}")
    if not blocks:
        return None
    evidence = sep.join(blocks)
    message = f"{_P6_ABSTAIN_REASK_HEAD}{ctx.question}{_P6_ABSTAIN_REASK_TAIL}{evidence}"
    if limit > 0 and estimate_tokens_for_chars(len(ctx.sys_prompt) + len(message)) > limit:
        return None
    return message, len(evidence)


def _compact_reask_message(ctx: _TurnContext) -> str | None:
    """The fresh single request's user message, or ``None`` to skip.

    Shape: the turn-1 message with BETTER evidence — the pre-seed (already
    fitted) plus every inserted ``read_article`` excerpt, each tagged with
    its ``[n]`` card number so citations in the re-asked answer still resolve
    (the citing minority must be served). Read blocks are
    ``must_include`` spans of one :func:`~vesta.answer.focus.focused_view`
    window over the combined evidence — the reads are the deliberate
    evidence, the pre-seed fills the remaining room by question relevance —
    and the whole message is ``estimate_tokens_for_chars``-checked against
    ``window - output_reserve`` (the same composing char arithmetic as the
    D4 pre-flight fit). ``None`` (degrade-don't-fail, the
    ``_plan_abstain_retry`` precedent) when there is no evidence or nothing
    fits.
    """
    limit = ctx.budget.window_tokens - ctx.budget.output_reserve
    # A forced-on lever without a window (``compact_reask=on`` at ``full``,
    # the bench A/B arm) has nothing to fit against — the seed's D4 fit and
    # the per-read caps already bound the evidence, so no windowing applies.
    fit_chars: int | None = None
    if limit > 0:
        fit_chars = (
            3 * limit
            - len(ctx.sys_prompt)
            - len(_USER_MESSAGE_HEAD)
            - len(_USER_MESSAGE_TAIL)
            - len(ctx.question)
        )
        if fit_chars <= 0:
            return None
    blocks: list[tuple[str, bool]] = []  # (block, is a must-include read)
    if ctx.seed_text:
        blocks.append((ctx.seed_text, False))
    for n, text in ctx.read_excerpts:
        card = next((c for c in ctx.turn_cards.values() if c.n == n), None)
        header = f'[{n}] "{card.title}" (archive-{card.zim_id})' if card else f"[{n}]"
        blocks.append((f"{header}\n{text}", True))
    if not blocks:
        return None
    sep = "\n\n"
    evidence = sep.join(b for b, _ in blocks)
    if fit_chars is not None and len(evidence) > fit_chars:
        from vesta.answer.focus import focused_view

        spans: list[tuple[int, int]] = []
        pos = 0
        for block, must in blocks:
            if must:
                spans.append((pos, pos + len(block)))
            pos += len(block) + len(sep)
        evidence = focused_view(
            evidence, ctx.question, fit_chars, must_include_spans=tuple(spans)
        ).excerpt
        if len(evidence) > fit_chars:
            # focused_view may exceed its budget by merge-gap slop or when
            # the must-include reads alone overflow — trim that bounded
            # excess, never more.
            evidence = evidence[:fit_chars]
    message = f"{_USER_MESSAGE_HEAD}{evidence}{_USER_MESSAGE_TAIL}{ctx.question}"
    if limit > 0 and estimate_tokens_for_chars(len(ctx.sys_prompt) + len(message)) > limit:
        return None  # safety valve — unreachable under the composing fit
    return message


def _steered_alternative_est_chars(ctx: _TurnContext) -> int:
    """The wire size the STEERED alternative would carry: one
    more request on the transcript as it stood — the meter's last completed
    request (exact wire chars; the static prompt base before any request) —
    plus one steering message and the same envelope slack the window ledger
    projects. The re-ask's fresh request is priced against the token
    estimate of this in the trace (``compact_reask.steered_est_tokens`` vs
    the fresh request's measured ``input_tokens``), so the bench can state
    the saving per firing without a steered control arm."""
    base = ctx.meter.request_log[-1][0] if ctx.meter.request_log else ctx.prompt_chars
    steering = max(len(_ROUND_CAP_STEERING), len(_TOOL_BUDGET_STEERING))
    return base + steering + _WINDOW_LEDGER_SLACK_CHARS


def _budget_audit(ctx: _TurnContext) -> dict[str, Any]:
    """The trace's ``budget`` dict: the resolved plan plus the
    per-turn outcomes (``preseed_dropped``; ``max_tool_rounds`` when the
    cap is active) — the audit trail for how the window actually got spent.
    At ``full`` every extra key stays absent: the dict must remain exactly
    the locked reference's keys there (tests/test_agent_context_window.py's
    byte-identity trap)."""
    d = asdict(ctx.budget)
    d["preseed_dropped"] = ctx.preseed_dropped
    if ctx.first_request_over_budget:
        d["first_request_over_budget"] = True
    if ctx.fallback_history_dropped:
        d["fallback_history_dropped"] = True
    if ctx.max_tool_rounds > 0:
        d["max_tool_rounds"] = ctx.max_tool_rounds
    return d


@dataclass
class _TurnRecoveryState:
    """Mutable outcome state for the shared turn-recovery phase.

    Both drivers (:func:`run_one_turn` and :func:`iter_agent_turn_events`) run
    their first model attempt themselves (in-process ``agent.run`` vs streamed
    ``run_stream`` — deliberately different), classify a crash, then hand the
    outcome to :func:`_iter_recovery_events`, which mutates this state in place.
    ``reask`` is the compact-reask trace record both drivers embed in their
    trace payloads.
    """

    crashed: bool
    answer: str
    usage: RunUsage
    overflow_fallbacks: int
    reask: dict[str, Any] = field(default_factory=lambda: {"fired": False, "trigger": None})


async def _iter_recovery_events(  # noqa: PLR0912, PLR0915
    ctx: _TurnContext,
    st: _TurnRecoveryState,
    *,
    agent: Agent,
    message_history: list[ModelMessage] | None,
    stream_events: bool,
) -> AsyncIterator[object]:
    """The shared recovery core behind both turn drivers — THE canonical copy.

    Covers, in order: the context-overflow / request-cap fallback (a forced
    no-tool single-shot from the Round-0 pre-seed), the compact-reask lever
    (round-cap / ledger / calibrated-abstention triggers), and the abstention
    retry gate. ``stream_events=True`` (the SSE driver) additionally yields the
    protocol events each arm owes the client — ``AnswerResetEvent`` before the
    replacement text and a trailing ``TokenEvent`` — while ``False`` (the
    benchmark driver) yields nothing and only records outcomes. Everything
    else — crash classification of the FIRST attempt, usage folding, cleanup,
    trace assembly — stays in the drivers, whose observable outputs are
    unchanged by construction.
    """
    reask_fired = False
    p6_abstain_triggered = False
    if st.crashed:
        # No tools on the fallback: force a direct answer from the initial
        # sources. Keep the conversation when it fits the window plan — a
        # follow-up that overflows must not recover amnesic. When
        # prompt + history would overflow again, drop the history
        # deliberately (latched for the audit); without a window plan there
        # is nothing to estimate against, so today's history-less shape stands.
        fallback_history: list[ModelMessage] | None = message_history or None
        if (
            fallback_history is not None
            and ctx.budget.window_tokens > 0
            and not _request_fits_window(ctx, ctx.user_message, fallback_history)
        ):
            ctx.fallback_history_dropped = True
            fallback_history = None
        try:
            fb = await ctx.build_agent(with_tools=False).run(
                ctx.user_message,
                message_history=fallback_history,
                usage_limits=UsageLimits(request_limit=2),
            )
        except ModelHTTPError as exc:
            if not _is_context_overflow_error(exc):
                raise  # a real bug — stay loud
            # Even the pre-seed alone overflowed the context window. Degrade to an
            # empty-but-recorded answer rather than propagating a second crash — the
            # metered accounting already reflects everything actually spent, so the
            # counters stay honest even though no answer text was produced.
            st.answer = ""
            st.overflow_fallbacks += 1
        else:
            st.answer = fb.output
            st.usage = st.usage + fb.usage
            # Outcome-first ordering (AUDIT_0824 C5): the reset is owed only
            # when a replacement actually follows — a fallback that itself
            # overflows or produces nothing must not leave a dangling erase.
            if stream_events and st.answer:
                yield AnswerResetEvent(reason="fallback")
                yield TokenEvent(st.answer)
    else:
        if ctx.compact_reask:
            trigger = _compact_reask_trigger(ctx, st.answer)
            if trigger is not None:
                p6_abstain_triggered = trigger == "abstain_p6"
                st.reask["trigger"] = trigger
                # Price the choice up front (recorded even when the
                # re-ask then cannot run): what one more STEERED request
                # on this transcript would have cost, in estimate.
                st.reask["steered_est_tokens"] = estimate_tokens_for_chars(
                    _steered_alternative_est_chars(ctx)
                )
                p6_trace: dict[str, Any] | None = None
                if trigger == "abstain_p6":
                    p6_trace = {
                        "top_score": ctx.evidence_directive_trace["top_score"],
                        "floor": ctx.abstention_floor,
                        "focused_chars": 0,
                        "fired": False,
                    }
                    st.reask["p6_abstain"] = p6_trace
                    planned = _p6_abstain_reask_message(ctx)
                    if planned is None:
                        message = None
                    else:
                        message, p6_trace["focused_chars"] = planned
                else:
                    # The established round-cap / ledger request shape is
                    # intentionally untouched; P6 changes abstentions only.
                    message = _compact_reask_message(ctx)
                if message is None:
                    st.reask["reason"] = "fit"  # nothing fits — keep the answer we have
                else:
                    started_reask = time.monotonic()
                    try:
                        ra = await ctx.build_agent(with_tools=False).run(
                            message, usage_limits=UsageLimits(request_limit=2)
                        )
                    except UsageLimitExceeded:
                        pass  # keep the steered answer; don't crash
                    except ModelHTTPError as exc:
                        if not _is_context_overflow_error(exc):
                            raise  # a real bug — stay loud
                        # The estimate-checked re-ask overflowed anyway —
                        # counted, and the steered answer stands.
                        st.overflow_fallbacks += 1
                    else:
                        reask_fired = True
                        st.answer = ra.output
                        st.usage = st.usage + ra.usage
                        ra_in = ra.usage.input_tokens or 0
                        ra_out = ra.usage.output_tokens or 0
                        st.reask.update(
                            fired=True,
                            chars=len(message),
                            input_tokens=ra_in,
                            output_tokens=ra_out,
                        )
                        if p6_trace is not None:
                            p6_trace["fired"] = True
                        step_inputs: dict[str, Any] = {"trigger": trigger}
                        step_outputs: dict[str, Any] = {
                            "chars": len(message),
                            "input_tokens": ra_in,
                            "output_tokens": ra_out,
                            "answer_chars": len(st.answer),
                        }
                        if p6_trace is not None:
                            step_inputs.update(
                                top_score=p6_trace["top_score"], floor=p6_trace["floor"]
                            )
                            step_outputs["focused_chars"] = p6_trace["focused_chars"]
                        ctx.add_step(
                            "compact_reask",
                            "pydantic_ai",
                            (time.monotonic() - started_reask) * 1000.0,
                            inputs=step_inputs,
                            outputs=step_outputs,
                        )
                        # Outcome-first ordering (AUDIT_0824 C5), both
                        # triggers: the reset fires only when the replacement
                        # text follows — a re-ask that fails (usage cap /
                        # overflow) keeps the steered answer, so no erase.
                        if stream_events and st.answer:
                            yield AnswerResetEvent(reason="compact_reask")
                            yield TokenEvent(st.answer)
        if (
            not reask_fired
            and not p6_abstain_triggered
            and ctx.seed_hit
            and ctx.read_count == 0
            and looks_abstained(st.answer)
        ):
            # ── Abstention gate: if the model refused in Round 0 WITHOUT reading any
            #    source (read_count == 0) despite relevant evidence, retry once with
            #    explicit steering to read + answer. Gated on read_count so expensive
            #    spirals that already exhausted their reads aren't re-run (they keep
            #    their refusal). The cheap-refusal case is the fixable one. ──
            # Under a window plan the retry shape is chosen by
            # _plan_abstain_retry (fit the window, else dedup, else skip).
            # When the compact re-ask fired it REPLACES this
            # retry (one planned recovery channel, not two stacked).
            plan = _plan_abstain_retry(ctx, message_history)
            if plan is None:
                pass  # even the dedup retry would overflow — keep the refusal
            else:
                retry_prompt, retry_history = plan
                try:
                    retry = await agent.run(
                        retry_prompt,
                        # ORIGINAL pre-turn history, NOT the failed attempt's
                        # transcript — re-sending it roughly doubles the retry's
                        # input tokens and the failed attempt contributes nothing.
                        message_history=retry_history,
                        usage_limits=UsageLimits(request_limit=_REQUEST_LIMIT),
                    )
                    st.answer = retry.output
                    st.usage = st.usage + retry.usage
                    # Outcome-first ordering (AUDIT_0824 C5): the reset fires
                    # only when the replacement text follows — a skipped plan
                    # or a failed retry keeps the refusal, so no erase.
                    if stream_events and st.answer:
                        yield AnswerResetEvent(reason="abstention_retry")
                        yield TokenEvent(st.answer)
                except UsageLimitExceeded:
                    pass  # keep the original refusal; don't crash
                except ModelHTTPError as exc:
                    if not _is_context_overflow_error(exc):
                        raise  # a real bug — stay loud
                    # Context overflow on the retry too — keep the original
                    # refusal; still an overflow recovery, so it is counted.
                    st.overflow_fallbacks += 1


async def run_one_turn(
    state: AppState,
    sn: Any,
    question: str,
    *,
    model_id: str,
    endpoint: str,
    api_key: str = "",
    enable_thinking: bool | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    message_history: list[ModelMessage] | None = None,
    profile_override: str | None = None,
    scope: str | None = None,
) -> TurnResult:
    """Run one pydantic-ai agent turn against the real search tool, in-process.

    Evidence-first (Round 0):
    the raw question is run through the retrieval pipeline FIRST (via
    ``tool_runtime.search_exact`` — no prefix-shortening/term-surfacing recovery,
    since those are tuned for the model's own short tool-call queries, not a raw
    multi-clause question) and the passages are pre-seeded as the initial sources
    in the user message, so the model starts from evidence rather than searching
    blind. ``search`` (harness-capped at :data:`_MAX_SEARCH_CALLS`, and unlike the
    pre-seed still uses the shortening recovery ladder) and ``read_article`` remain
    available for missing facts and full reads. A harness-side abstention gate
    appends a no-refusal directive when Round-0 retrieval hit, and retries once if
    the model abstains despite evidence.

    ``scope`` follows the same convention as the CLI/bench ``--scope`` flag and the
    other benchmark systems (``RetrievalOnlySystem`` et al.) — a raw scope string
    parsed via :func:`~vesta.api.answer._parse_scope`; ``None``/empty means "every
    enabled archive" (unchanged default for callers that pass nothing).

    No FastAPI/HTTP involved — callable directly from a CLI script that opened its own
    runtime via ``cli._open_runtime``.
    """
    # The benchmark path warms the runtime too — no status callback
    # (there is no SSE stream here).
    await _ensure_llm_ready()
    with _in_flight_generation():
        try:
            ctx = await _build_turn(
                state,
                sn,
                question,
                model_id=model_id,
                endpoint=endpoint,
                api_key=api_key,
                enable_thinking=enable_thinking,
                system_prompt=system_prompt,
                message_history=message_history,
                profile_override=profile_override,
                # The scope was accepted and documented but never
                # forwarded — every bench pre-seed searched ALL enabled archives
                # (9 on the dev box: cross-archive kNN fanout + cold cluster
                # reads = the measured 7-14 s pre_seed vs ~2 s scoped, plus
                # foreign-archive cards in 6/50 run-90 questions). The streaming
                # path always passed it; only this driver omitted it.
                scope=scope,
            )
            agent = ctx.build_agent(with_tools=True)

            run_usage = RunUsage()
            answer: str = ""
            crashed = False
            # Count every context-overflow recovery (main run →
            # no-tool fallback; fallback re-ask itself overflowing; abstention
            # retry overflowing). Any nonzero value is a window-fit failure —
            # an invariant that must drive to zero.
            overflow_fallbacks = 0
            llm_started = time.monotonic()
            try:
                result = await agent.run(
                    ctx.user_message,
                    message_history=message_history or None,
                    usage_limits=UsageLimits(request_limit=_REQUEST_LIMIT),
                    usage=run_usage,
                )
                answer = result.output
            except UsageLimitExceeded:
                # The model looped to the request cap without converging (a greedy 4b can
                # repeat already-capped tool calls). Rather than crash with an empty answer,
                # fall through to a forced single-shot answer from the pre-seed evidence.
                # turn_cards (the initial sources) are preserved regardless of the crash.
                crashed = True
            except ModelHTTPError as exc:
                if not _is_context_overflow_error(exc):
                    raise  # a real bug (bad request, auth, provider 500) — stay loud
                # The accumulated message history exceeded the model's context window (a hard
                # 400 from the endpoint, not a pydantic-ai-side limit). Same recovery as
                # UsageLimitExceeded: fall through to a forced no-tool fallback against just
                # the Round-0 pre-seed, which is far smaller than the blown-up history.
                crashed = True
                overflow_fallbacks += 1

            # The model's inference wall time (thinking + tool rounds + token
            # generation), inserted right after the pre-seed step so the breakdown
            # reads pre_seed → agent_llm → tool calls. ``at`` places it before any
            # search/read_article steps already recorded during ``agent.run``
            # (those tool rounds happen *inside* this window).
            llm_ms = (time.monotonic() - llm_started) * 1000.0
            pos = 1 if ctx.steps and ctx.steps[0]["name"] == "pre_seed" else 0
            ctx.add_step(
                "agent_llm",
                "pydantic_ai",
                llm_ms,
                inputs={"input_tokens": run_usage.input_tokens or 0},
                outputs={
                    "output_tokens": run_usage.output_tokens or 0,
                    "answer_chars": len(answer),
                },
                at=pos,
            )

            st = _TurnRecoveryState(
                crashed=crashed,
                answer=answer,
                usage=run_usage,
                overflow_fallbacks=overflow_fallbacks,
            )
            async for _event in _iter_recovery_events(
                ctx,
                st,
                agent=agent,
                message_history=message_history,
                stream_events=False,
            ):
                pass  # pragma: no cover — the silent mode never yields

            # Mirror the recovery core's mutations of st.answer unconditionally —
            # the no-tool fallback / compact re-ask / abstention retry all write
            # st.answer, and with cleanup off the pre-recovery binding would
            # otherwise be returned (empty after a crash). Same rebind the
            # streaming driver performs at its recovery boundary.
            answer = _cleanup_answer(st.answer) if ctx.answer_cleanup else st.answer

            elapsed_ms = int((time.monotonic() - ctx.started) * 1000)
            cards = sorted(ctx.turn_cards.values(), key=lambda c: c.n)
            return TurnResult(
                answer=answer,
                cards=cards,
                tool_calls=ctx.calls,
                total_tokens=st.usage.total_tokens or 0,
                input_tokens=st.usage.input_tokens or 0,
                output_tokens=st.usage.output_tokens or 0,
                elapsed_ms=elapsed_ms,
                trace={
                    # Non-streaming analogue of the streaming path's TraceEvent
                    # payload, so benchmark drivers can merge budget/steps into
                    # their per-question trace_json.
                    "system": "agentic_pydantic",
                    "followup": ctx.follow_up,
                    "budget": _budget_audit(ctx),
                    "stages": ctx.steps,
                    # Per-request accounting (measurement only).
                    # ``peak_input_tokens`` is the largest single request the
                    # endpoint actually prefilled — the quantity a context window
                    # constrains, unlike the cumulative ``input_tokens`` above.
                    "peak_input_tokens": ctx.meter.peak_input_tokens,
                    "requests": ctx.meter.requests,
                    "overflow_fallbacks": st.overflow_fallbacks,
                    "request_log": ctx.meter.request_log,
                    # The compact-reask record — fired/trigger plus
                    # the per-firing token cost (a stage entry carries the same
                    # numbers with duration).
                    "compact_reask": st.reask,
                    # How many tool calls the round cap steered
                    # away (0 when no cap / nothing blocked).
                    "round_cap_fires": ctx.round_cap_fires,
                    # What the aging wrapper actually trimmed
                    # (0/0 when aging is off or never bit).
                    "aged_requests": ctx.aging.requests,
                    "age_saved_chars": ctx.aging.saved_chars,
                    # Whether the must-state clause fired.
                    "evidence_directive": ctx.evidence_directive_trace,
                },
            )
        finally:
            # Stamp last_used on turn completion, success or failure.
            _mark_llm_used()


async def iter_agent_turn_events(  # noqa: PLR0912, PLR0915
    state: AppState,
    sn: Any,
    question: str,
    *,
    message_history: list[ModelMessage] | None = None,
    profile_override: str | None = None,
    scope: str | None = None,
) -> AsyncIterator[object]:
    """Run one pydantic-agent turn, yielding the frozen SSE answer events
    (sources → status → token → [answer_reset → token] → citations → trace → done).

    Same evidence-first Round-0 pre-seed, tool caps, abstention gate, and recovery
    paths as :func:`run_one_turn` — but streamed. ``message_history`` carries prior
    conversation turns (reconstructed by ``/api/chat``). No FastAPI/HTTP; callable
    in-process (bench-friendly).

    Streaming is deliberately simple (no queue, no background task): pydantic-ai's
    tool rounds complete inside ``run_stream``'s ``__aenter__``, so the tool
    closures append their :class:`StatusEvent`\\ s to ``ctx.status_buf`` and we
    drain that list right after ``__aenter__`` returns, before streaming tokens
    via ``stream_text(delta=True, debounce_by=None)``.
    """
    ctx = await _build_turn(
        state,
        sn,
        question,
        profile_override=profile_override,
        scope=scope,
        message_history=message_history,
    )
    if ctx.follow_up:
        # Follow-up (history present): no Round-0 pre-seed, so no initial
        # ``sources`` event. The agent resolves references from the conversation
        # and either answers directly (zero ``sources`` events, protocol-legal)
        # or searches — the FIRST discovery then surfaces ``sources(merge=False)``
        # live from ``_do_search``. ``first_sources_keys`` stays ``None`` until then.
        yield StatusEvent("reading", detail="Considering your question…")
        #: Restored after a cold-load warm-up (see below) so the UI goes
        #: back to the truthful pre-warmup reading detail once the model
        #: is ready instead of staying stuck on ``Loading <model> into
        #: memory…``.
        pre_warmup_detail = "Considering your question…"
    else:
        # Turn 1: emit the Round-0 pre-seed cards first (always present, even when
        # empty — the existing contract), then a reading status.
        yield SourcesEvent(
            cards=cast(
                tuple[SourceCard, ...],
                tuple(sorted(ctx.turn_cards.values(), key=lambda c: c.n)),
            ),
            merge=False,
        )
        ctx.first_sources_keys = set(ctx.turn_cards.keys())
        yield StatusEvent("reading", detail=f"{len(ctx.turn_cards)} sources")
        pre_warmup_detail = f"{len(ctx.turn_cards)} sources"

    # ── Warm-up: bring the local runtime to ready BEFORE the
    # agent's first model call, so a cold model load surfaces as truthful
    # ``reading`` statuses instead of a silent multi-second gap. No new SSE
    # phase — the loading messages reuse ``reading`` (frozen protocol). A
    # runtime that cannot come up (missing binary, load failure, no matching
    # model id) is a clean terminal ``error`` event, never a 500ing stream.
    from vesta.inference.local import BinaryMissing, LlamaServerError
    from vesta.inference.runtime import LlmRuntimeError

    warm: list[StatusEvent] = []
    try:
        await _ensure_llm_ready(on_status=lambda msg: warm.append(StatusEvent("reading", msg)))
    except (LlmRuntimeError, BinaryMissing, LlamaServerError) as exc:
        yield ErrorEvent(
            code="no_llm",
            message=f"{_LOCAL_RUNTIME_UNAVAILABLE} ({exc})",
            recoverable=True,
        )
        # Protocol ordering rule 8: an error event terminates the stream —
        # no ``done`` after a terminal error.
        return
    for status in warm:
        yield status
    if warm:
        # Cold load: hand the UI back its pre-warmup reading status ("2
        # sources" / "Considering your question…"). Without this the client
        # keeps displaying ``Loading <model> into memory…`` through the whole
        # first inference gap — truthful statuses only, no new phase.
        yield StatusEvent("reading", detail=pre_warmup_detail)

    # Post-warmup sync: ensure ctx.model uses the live router model id.
    resolved_id, resolved_endpoint, resolved_key, _, _ = _resolve_llm(sn)
    if resolved_id and resolved_id != ctx.model_id:
        ctx.model_id = resolved_id
        model_inst: Any = _make_model(resolved_id, resolved_endpoint, resolved_key)
        if ctx.budget.age_tool_chars > 0:
            model_inst = _AgedContextModel(model_inst, ctx.budget.age_tool_chars, ctx.aging)
        ctx.model = _MeteredModel(model_inst, ctx.meter)

    with _in_flight_generation():
        try:
            agent = ctx.build_agent(with_tools=True)

            answer = ""
            usage = RunUsage()
            crashed = False
            #: Whether the ``status_buf`` drain below ran. A crash inside
            #: ``run_stream``'s ``__aenter__`` (a tool-round model request, e.g.
            #: UsageLimitExceeded / context overflow on a later round) skips the
            #: whole ``async with`` body — including that drain — so anything the
            #: tool closures buffered (notably a follow-up turn's latched first
            #: ``sources`` event) is still pending and must be emitted before the
            #: fallback answer cites those cards.
            buf_drained = False
            # Same overflow-recovery counter as run_one_turn.
            overflow_fallbacks = 0
            llm_started = time.monotonic()
            try:
                async with agent.run_stream(
                    ctx.user_message,
                    message_history=message_history or None,
                    usage_limits=UsageLimits(request_limit=_REQUEST_LIMIT),
                    # Threaded like run_one_turn so a UsageLimitExceeded /
                    # overflow mid-stream keeps whatever tokens the main run
                    # burned before crashing; recovery folds on top of it.
                    usage=usage,
                ) as result:
                    buf_drained = True
                    for s in ctx.status_buf:
                        yield s
                    yield StatusEvent("generating", "Thinking…")
                    async for delta in result.stream_text(delta=True, debounce_by=None):
                        answer += delta
                        _mark_llm_used()
                        yield TokenEvent(delta)
                    usage = result.usage
            except UsageLimitExceeded:
                crashed = True
            except ModelHTTPError as exc:
                if _is_context_overflow_error(exc):
                    crashed = True
                    overflow_fallbacks += 1
                else:
                    raise  # a real bug (bad request, auth, provider 500) — stay loud
            if crashed and not buf_drained:
                # The crash happened inside ``run_stream``'s ``__aenter__``, so the
                # buffered statuses — and, on a follow-up, the latched first
                # ``sources(merge=False)`` event — never reached the client (the
                # in-body drain above was skipped). Emit them now: the fallback
                # answer cites these cards, and the trailing merge delta excludes
                # exactly ``first_sources_keys``, so this is the client's only
                # copy. Ordering stays protocol-valid (statuses + sources before
                # any token).
                for event in ctx.status_buf:
                    yield event

            # The model's inference wall time (thinking + tool rounds + token
            # generation), inserted right after the pre-seed step so the breakdown
            # reads pre_seed → agent_llm → tool calls. ``at`` places it before any
            # search/read_article steps already recorded during ``run_stream``'s
            # ``__aenter__`` (those tool rounds happen *inside* this window).
            llm_ms = (time.monotonic() - llm_started) * 1000.0
            pos = 1 if ctx.steps and ctx.steps[0]["name"] == "pre_seed" else 0
            ctx.add_step(
                "agent_llm",
                "pydantic_ai",
                llm_ms,
                inputs={"input_tokens": usage.input_tokens or 0},
                outputs={"output_tokens": usage.output_tokens or 0, "answer_chars": len(answer)},
                at=pos,
            )

            st = _TurnRecoveryState(
                crashed=crashed,
                answer=answer,
                usage=usage,
                overflow_fallbacks=overflow_fallbacks,
            )
            async for event in _iter_recovery_events(
                ctx,
                st,
                agent=agent,
                message_history=message_history,
                stream_events=True,
            ):
                yield event
            answer = st.answer
            usage = st.usage
            overflow_fallbacks = st.overflow_fallbacks

            if ctx.answer_cleanup:
                cleaned_answer = _cleanup_answer(answer)
                if cleaned_answer != answer:
                    yield AnswerResetEvent(reason="cleanup")
                    yield TokenEvent(cleaned_answer)
                answer = cleaned_answer

            # Merge event: cards discovered AFTER the first ``sources`` event, delta-only,
            # with continuing 0-based numbering (docs/sse-protocol.md "Sources merge").
            # ``first_sources_keys`` is the set already streamed (pre-seed on turn 1, or
            # the first live discovery on a follow-up); everything else is a delta. On a
            # from-context follow-up that never searched, it stays ``None`` and no cards
            # exist anyway, so the delta is empty.
            already_sent = ctx.first_sources_keys or set()
            delta_cards = sorted(
                [c for k, c in ctx.turn_cards.items() if k not in already_sent],
                key=lambda c: c.n,
            )
            if delta_cards:
                yield SourcesEvent(
                    cards=cast(tuple[SourceCard, ...], tuple(delta_cards)),
                    merge=True,
                )

            if answer:
                spans = synthesize_citation_spans(answer, len(ctx.turn_cards))
                yield CitationsEvent(spans=tuple(spans), answer_text=answer or None)
            else:
                # Protocol ordering rule 8: an error event terminates the stream.
                # No trace/done may follow a terminal error.
                yield ErrorEvent(
                    code="budget_exhausted",
                    message="agent produced no answer",
                    recoverable=True,
                )
                return

            trace: dict[str, object] = {
                "system": "agentic_pydantic",
                "followup": ctx.follow_up,
                "elapsed_ms": int((time.monotonic() - ctx.started) * 1000),
                "total_tokens": usage.total_tokens or 0,
                "input_tokens": usage.input_tokens or 0,
                "output_tokens": usage.output_tokens or 0,
                "search_calls": ctx.search_count,
                "read_calls": ctx.read_count,
                "card_count": len(ctx.turn_cards),
                "stages": ctx.steps,
                # Per-request accounting (measurement only) — same
                # fields as run_one_turn's trace. ``peak_input_tokens`` is the
                # largest single prefilled request, not the cumulative total.
                "peak_input_tokens": ctx.meter.peak_input_tokens,
                "requests": ctx.meter.requests,
                "overflow_fallbacks": overflow_fallbacks,
                "request_log": ctx.meter.request_log,
                # The compact-reask record (same fields as
                # run_one_turn's trace).
                "compact_reask": st.reask,
                # Round-cap firings (same field as run_one_turn).
                "round_cap_fires": ctx.round_cap_fires,
                # What the aging wrapper actually trimmed (same
                # fields as run_one_turn's trace).
                "aged_requests": ctx.aging.requests,
                "age_saved_chars": ctx.aging.saved_chars,
                # Whether the must-state clause fired.
                "evidence_directive": ctx.evidence_directive_trace,
                # The resolved token-economy budget for this turn, so before/after
                # comparisons are observable per run (bench --economy A/B).
                "budget": _budget_audit(ctx),
            }
            yield TraceEvent(trace=trace)
            yield DoneEvent()
        finally:
            # Stamp last_used on turn completion, success or failure —
            # including a consumer abandoning the stream mid-answer.
            _mark_llm_used()


__all__ = [
    "TurnResult",
    "iter_agent_turn_events",
    "looks_abstained",
    "run_one_turn",
]
