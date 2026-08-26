"""Answer contracts — the Protocol, events, and context for answer strategies.

Mirrors the retrieval package's pattern: every swappable behaviour is a
registered Protocol implementation. Here the contract is :class:`AnswerStrategy`,
and the events it yields are the SSE protocol's domain model.

``answer/`` depends on ``retrieval`` + ``inference`` (exactly 2 deps, at the
module cap; do NOT exceed). It must not import ``api/`` or ``index/``
(enforced by ``tests/test_boundaries.py``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

from vesta.config.capabilities import Capability
from vesta.retrieval.contracts import RetrievalResult, SourceCard

if TYPE_CHECKING:
    from vesta.retrieval.trace import Trace


# ── Events (the SSE protocol's domain model, frozen here) ───────────────────


@dataclass(frozen=True)
class SourcesEvent:
    """Source cards from retrieval. The first ``sources`` event (~0.7 s, always
    ``merge=False``) is streamed before any LLM work so the user is already done
    if that's all they wanted (progressive disclosure — a protocol
    property, not a UI trick).

    ``merge`` (an additive protocol amendment — see ``docs/sse-protocol.md``)
    marks a second, later ``sources`` event carrying ONLY the new cards a
    recovery/tool round turned up beyond Round 0. Clients append rather than
    replace: the delta's cards continue the same 0-based numbering the first
    event started, so citation ``card_id``s stay valid across both events. Never
    set by ``sources_only`` — only the agent chat path (``api/agent_chat.py``)
    emits a second ``sources`` event, for delta cards a later tool round turns up.
    """

    cards: tuple[SourceCard, ...]
    merge: bool = False


@dataclass(frozen=True)
class StatusEvent:
    """Truthful intermediate status during the prefill gap.

    The server is responsible for there always being a next truthful thing to
    show. ``phase`` is one of: ``reading`` (evidence gathered, model loading),
    ``generating`` (tokens streaming), ``abstaining`` (no candidates)."""

    phase: str
    detail: str


@dataclass(frozen=True)
class TokenEvent:
    """One incremental answer token (never buffered)."""

    text: str


@dataclass(frozen=True)
class AnswerResetEvent:
    """Discard everything streamed so far for this turn — a fresh, REPLACEMENT
    generation is about to start (an additive protocol amendment, see
    ``docs/sse-protocol.md``'s ``answer_reset`` event).

    The agent chat path (``api/agent_chat.py``) regenerates the whole answer
    from scratch in two cases: a tool-call crash (``reason="fallback"`` —
    retries without tools) and an over-refusal on relevant seed sources
    (``reason="abstention_retry"`` — re-prompts for an answer). Every generated
    chunk still streams as ``TokenEvent``s (so the UI keeps showing something
    live), but a naive consumer that concatenates every ``TokenEvent.text`` it
    has ever seen would show the OLD answer followed by the NEW one, and
    citation offsets computed against the new (final) text would no longer line
    up with the concatenated text a client displays.

    Clients (SSE consumers, the dev console, conversation persistence) MUST
    discard all ``TokenEvent`` text accumulated so far for this turn on receipt
    and start accumulating fresh.

    ``reason`` is an optional short machine tag (e.g. ``"fallback"``,
    ``"abstention_retry"``) for the trace/dev console; empty string when the
    caller has nothing more specific to say.
    """

    reason: str = ""


@dataclass(frozen=True)
class CitationSpan:
    """A citation: a character span in the answer aligned to a span in a source.

    Citation validity is 100% by construction — a citation to something
    not retrieved cannot exist, because citations are derived by post-hoc span
    alignment against the retrieved passages.

    ``source_index`` is 0-based into the sources list (matching ``[1]``..``[n]``
    in the prompt, offset by one). ``passage_start``/``passage_end`` are character
    offsets into the source passage's text, enabling click-to-highlight.
    """

    answer_start: int
    answer_end: int
    source_index: int
    passage_start: int | None
    passage_end: int | None
    score: float


@dataclass(frozen=True)
class CitationsEvent:
    """Citation spans for the completed answer, emitted after the tokens.

    The agent prompts the model with cards numbered ``[1]``..``[n]`` (first-seen
    card order), so a ``[n]`` marker in the answer is already a valid card
    number; spans are synthesized from the markers
    (``citations.synthesize_citation_spans``), not aligned by n-gram overlap.
    Citation validity is structural: a marker can only reference a
    retrieved card.

    ``answer_text`` (an additive field, see ``docs/sse-protocol.md``): the FINAL
    answer text — the one the citation spans' character offsets are computed
    against. Streaming makes in-flight rewriting impossible, so when an
    ``answer_reset`` regeneration changes the text (a context-overflow recovery
    or an abstention-retry) this is the only place the corrected text is
    available; ``None`` when no citable spans were produced (e.g.
    ``sources_only``, which never emits this event). Clients SHOULD prefer this
    text over their own token-concatenation once received.
    """

    spans: tuple[CitationSpan, ...]
    answer_text: str | None = None


@dataclass(frozen=True)
class TraceEvent:
    """Emitted last. The full trace (versioned JSON)."""

    trace: dict[str, object]


@dataclass(frozen=True)
class ErrorEvent:
    """A recoverable or fatal error mid-stream.

    Killing ``llama-server`` mid-answer must produce this + an auto-restart."""

    code: str
    message: str
    recoverable: bool


@dataclass(frozen=True)
class DoneEvent:
    """Terminal success marker (no more events after this)."""


AnswerEvent = (
    SourcesEvent
    | StatusEvent
    | TokenEvent
    | AnswerResetEvent
    | CitationsEvent
    | TraceEvent
    | ErrorEvent
    | DoneEvent
)


# ── Context + deps ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AnswerContext:
    """Everything an answer strategy needs for one question.

    ``retrieval`` is the retrieval output (passages, cards, confidence, trace)
    — the sole input the registered strategy (``sources_only``) consumes.
    """

    retrieval: RetrievalResult


# ── Strategy Protocol ───────────────────────────────────────────────────────


class AnswerStrategy(Protocol):
    """Turn retrieved passages into a streamed, grounded, cited answer.

    The single registered implementation is ``sources_only`` (no generation).
    The LLM answer path is the streaming pydantic-ai agent behind
    ``POST /api/chat`` (``api/agent_chat.py``), which is not a registered
    strategy. The ``requires`` capability set drives automatic selection: with
    no ``Capability.LLM``, ``sources_only`` (which requires nothing) is
    auto-selected (degrade-don't-fail design).
    """

    requires: ClassVar[frozenset[Capability]]

    def answer(self, ctx: AnswerContext, tr: Trace) -> AsyncIterator[AnswerEvent]: ...


__all__ = [
    "AnswerContext",
    "AnswerEvent",
    "AnswerResetEvent",
    "AnswerStrategy",
    "CitationSpan",
    "CitationsEvent",
    "DoneEvent",
    "ErrorEvent",
    "SourcesEvent",
    "StatusEvent",
    "TokenEvent",
    "TraceEvent",
]
