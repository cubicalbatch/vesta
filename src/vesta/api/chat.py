"""Multi-turn chat API — ``POST /api/chat`` + conversation CRUD.

Chat is a **thin layer over the same machinery** as the single-turn
``GET /api/answer``: the same retrieval pipeline, the same answer strategy,
plus conversational rewrite and history. The
stream reuses :func:`vesta.api.answer.iter_answer_events` verbatim — the only
additions are conversation persistence (load history before, persist messages
after) and the ``X-Conversation-Id`` response header.

Why a header (not a new SSE event) for the conversation id: the frozen protocol
(docs/sse-protocol.md) makes ``sources`` always-first, so a metadata event ahead
of it would break ordering rule 1. A header carries request-scoped metadata
without touching the frozen event stream.

The JSON endpoints (``GET /api/conversations``, ``GET /api/conversations/{id}``,
``DELETE /api/conversations/{id}``) back a chat UI: list, reload history, remove.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, cast

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart

from vesta import config as app_config
from vesta.answer import CHAT_HISTORY_MAX_TURNS
from vesta.answer.contracts import (
    AnswerResetEvent,
    CitationsEvent,
    ErrorEvent,
    SourcesEvent,
    TokenEvent,
    TraceEvent,
)
from vesta.answer.conversation import build_history, derive_title
from vesta.api.agent_chat import iter_agent_turn_events
from vesta.api.answer import _card_to_dict, _serialize_event, iter_answer_events
from vesta.api.conversation_store import SqliteConversationStore, StoredConversation, StoredMessage
from vesta.api.state import AppState, app_state
from vesta.config.capabilities import Capability, compute_capabilities

router = APIRouter(prefix="/api", tags=["chat"])


# ── DTOs ────────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """``POST /api/chat`` body. Mirrors ``GET /api/answer``'s retrieval-relevant
    query params plus ``conversation_id`` (null ⇒ start a new conversation).
    (The answer strategy is not client-selectable here: this endpoint always
    runs the streaming agent.)"""

    query: str
    conversation_id: int | None = None
    scope: str | None = None
    profile: str | None = None


class ConversationSummary(BaseModel):
    id: int
    title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class MessageDetail(BaseModel):
    id: int
    role: str
    content: str | None = None
    sources: list[dict[str, Any]] | None = None
    trace: dict[str, Any] | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_ms: int | None = None
    created_at: str | None = None


class ConversationDetail(BaseModel):
    conversation: ConversationSummary
    messages: list[MessageDetail]


# ── POST /api/chat (SSE, multi-turn) ────────────────────────────────────────


@router.post("/chat")
async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
    """Stream a grounded, cited answer as SSE events, persisting the exchange.

    ``X-Conversation-Id`` is available immediately. History is loaded, the user
    turn is persisted, the answer is streamed, and the assistant turn is
    persisted once the stream ends — including a failed turn (even one that
    dies before the first token), which records the question plus whatever
    partial answer accumulated.
    """
    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    state: AppState = app_state(request)
    store = SqliteConversationStore(state.db)

    conversation_id = body.conversation_id
    if conversation_id is None:
        conversation_id = await store.create_conversation(derive_title(body.query))
    else:
        # Verify the conversation exists; 404 if not (don't silently create a
        # new one under a stranger's id). Slim read: only role/content, no
        # trace payloads deserialized just for a truthiness probe.
        existing = await store.list_recent_messages(conversation_id, limit=1)
        if not existing:
            async with state.db.read() as conn:
                cur = await conn.execute(
                    "SELECT 1 FROM conversations WHERE id=?", (conversation_id,)
                )
                row = await cur.fetchone()
            if row is None:
                raise HTTPException(
                    status_code=404, detail=f"conversation {conversation_id} not found"
                )

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for chunk in _run_chat_turn(state, store, conversation_id, body):
                yield chunk
        except Exception as exc:
            yield _serialize_event_error(exc)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "X-Conversation-Id": str(conversation_id),
        },
    )


def _absorb_turn_event(
    event: object,
    answer_parts: list[str],
    final_answer: str | None,
    source_cards: list[dict[str, Any]] | None,
    trace_dict: dict[str, object] | None,
) -> tuple[list[str], str | None, list[dict[str, Any]] | None, dict[str, object] | None]:
    """Accumulate one streamed event into the turn's persistence state."""
    if isinstance(event, TokenEvent):
        answer_parts.append(event.text)
    elif isinstance(event, AnswerResetEvent):
        # FIX 1: a fresh, REPLACEMENT generation is starting — discard
        # everything accumulated for this turn so far. Without this, a
        # regenerate (fallback/abstention_retry) doubles up with
        # whatever streamed before it in the persisted conversation turn.
        answer_parts = []
    elif isinstance(event, CitationsEvent):
        # FIX 2: the citations event carries the authoritative final text —
        # the answer after inline [n] passage-numbered citation markers were
        # rewritten to CARD numbers (which the UI displays). Persist THAT,
        # not the raw token concatenation, or the conversation history keeps
        # citing passage numbers the UI never shows. `answer_text` is null
        # only when no renumbering happened (then the token join is equal).
        if event.answer_text is not None:
            final_answer = event.answer_text
    elif isinstance(event, SourcesEvent):
        # docs/sse-protocol.md "Sources merge": a ``merge: true`` event
        # carries ONLY the delta cards discovered after the first event —
        # append them (continuing the same 0-based numbering), exactly like
        # the SPA reducer; a non-merge event is this turn's initial set and
        # replaces. Persisting only the last event's cards would drop every
        # Round-0 card from the system of record.
        if source_cards is None or not event.merge:
            source_cards = [_card_to_dict(c) for c in event.cards]
        else:
            source_cards.extend(_card_to_dict(c) for c in event.cards)
    elif isinstance(event, TraceEvent):
        trace_dict = event.trace
    return answer_parts, final_answer, source_cards, trace_dict


async def _run_chat_turn(
    state: Any,
    store: SqliteConversationStore,
    conversation_id: int,
    body: ChatRequest,
) -> AsyncIterator[str]:
    """One chat turn: load history → persist user msg → stream → persist assistant."""
    # 1. Load history BEFORE persisting this turn's user message, so the current
    #    query is excluded from its own context. Only user/assistant rows are
    #    conversation turns. Slim read: history consumes (role, content) only —
    #    full-row selects deserialized every stored trace payload per turn
    #    (AUDIT_0822 C8); the newest-window DESC+reverse keeps recent turns
    #    once the conversation outgrows 200 rows (C3).
    prior = await store.list_recent_messages(conversation_id, limit=200)
    pairs = tuple(
        (m.role, m.content or "") for m in prior if m.role in ("user", "assistant") and m.content
    )
    history = build_history(pairs, max_turns=_history_max_turns())

    # 2. Persist the user message now — it is known before streaming, so a
    #    mid-stream crash still records the question.
    await store.append_message(conversation_id, "user", body.query)

    # 3. Run the SAME pipeline the single-turn endpoint uses, threading history
    #    into both the conversational rewriter and the answer context. When an
    #    LLM is configured, POST /api/chat drives the streaming pydantic-ai agent;
    #    otherwise it degrades to the sources_only single-turn loop
    #    (test app + offline first-run).
    answer_parts: list[str] = []
    final_answer: str | None = None
    source_cards: list[dict[str, Any]] | None = None
    trace_dict: dict[str, object] | None = None
    started = time.monotonic()

    stream_failure: Exception | None = None
    event_gen: AsyncIterator[object] | None = None
    try:
        capabilities = compute_capabilities()
        try:
            sn = app_config.snapshot()
        except RuntimeError:
            sn = None

        if Capability.LLM in capabilities:
            model_history = _to_model_history(history)
            event_gen = iter_agent_turn_events(
                state,
                sn,
                body.query,
                message_history=model_history or None,
                profile_override=body.profile,
                scope=body.scope,
            )
        else:
            # No LLM: keep the sources_only degradation (test app + offline first-run).
            event_gen = iter_answer_events(
                state, body.query, body.scope, body.profile, None, history=history
            )
    except Exception as exc:
        # Setup died before a single event streamed (capability probe, history
        # reconstruction). Same contract as a mid-stream death: remember why,
        # let the persistence below write the (empty) assistant turn so the
        # question is not left dangling without an answer row, then re-raise.
        stream_failure = exc

    if event_gen is not None:
        async with contextlib.aclosing(cast(AsyncGenerator[object], event_gen)) as events:
            try:
                async for event in events:
                    yield _serialize_event(event)
                    if isinstance(event, ErrorEvent):
                        # Ordering rule 8 (docs/sse-protocol.md): an error event
                        # terminates the stream. Stop consuming upstream (a
                        # well-behaved emitter returns here anyway) so nothing can
                        # follow it on this layer's wire.
                        break
                    answer_parts, final_answer, source_cards, trace_dict = _absorb_turn_event(
                        event, answer_parts, final_answer, source_cards, trace_dict
                    )
            except Exception as exc:
                # The turn died mid-stream. Remember why, persist whatever exists
                # below, then re-raise so the wrapper emits the terminal error.
                stream_failure = exc

    # 4. Persist the assistant turn. tokens_in/out stay null — no tokenizer runs
    #    in the request path; the trace carries the full detail for analysis.
    latency_ms = int((time.monotonic() - started) * 1000)
    trace_json = _dumps(trace_dict) if trace_dict is not None else None
    sources_json = _dumps(source_cards) if source_cards is not None else None
    await store.append_message(
        conversation_id,
        "assistant",
        final_answer if final_answer is not None else "".join(answer_parts),
        sources_json=sources_json,
        trace_json=trace_json,
        latency_ms=latency_ms,
    )

    if stream_failure is not None:
        raise stream_failure


# ── Conversation CRUD (JSON) ────────────────────────────────────────────────


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
) -> list[ConversationSummary]:
    """List conversations, newest first."""
    state = app_state(request)
    store = SqliteConversationStore(state.db)
    return [_to_summary(c) for c in await store.list_conversations(limit)]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    request: Request,
    conversation_id: int,
    limit: int = Query(200, ge=1, le=2000),
) -> ConversationDetail:
    """Fetch one conversation and its messages (oldest first)."""
    state = app_state(request)
    store = SqliteConversationStore(state.db)
    conv = await store.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail=f"conversation {conversation_id} not found")
    messages = await store.list_messages(conversation_id, limit=limit)
    return ConversationDetail(
        conversation=_to_summary(conv),
        messages=[_to_message_detail(m) for m in messages],
    )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(request: Request, conversation_id: int) -> dict[str, bool]:
    """Delete a conversation and (via ON DELETE CASCADE) its messages."""
    state = app_state(request)
    store = SqliteConversationStore(state.db)
    deleted = await store.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"conversation {conversation_id} not found")
    return {"deleted": True}


# ── Helpers ─────────────────────────────────────────────────────────────────


def _to_model_history(history: tuple[tuple[str, str], ...]) -> list[ModelMessage]:
    """Reconstruct pydantic-ai ModelMessage history from stored (role, content) turns.

    Only user/assistant TEXT turns are replayed (decision 5: tool messages are never
    replayed — the agent re-searches each turn, exactly as the old loop did).
    """
    msgs: list[ModelMessage] = []
    for role, content in history:
        if role == "user":
            msgs.append(ModelRequest(parts=[UserPromptPart(content=content)]))
        elif role == "assistant":
            msgs.append(ModelResponse(parts=[TextPart(content=content)]))
    return msgs


def _history_max_turns() -> int:
    try:
        return max(1, int(app_config.get(CHAT_HISTORY_MAX_TURNS)))
    except Exception:
        return int(CHAT_HISTORY_MAX_TURNS.default)


def _to_summary(c: StoredConversation) -> ConversationSummary:
    return ConversationSummary(
        id=c.id, title=c.title, created_at=c.created_at, updated_at=c.updated_at
    )


def _to_message_detail(m: StoredMessage) -> MessageDetail:
    sources: list[dict[str, Any]] | None = None
    if m.sources_json:
        with contextlib.suppress(Exception):
            import json

            loaded = json.loads(m.sources_json)
            if isinstance(loaded, list):
                sources = loaded
    trace: dict[str, Any] | None = None
    if m.trace_json:
        with contextlib.suppress(Exception):
            import json

            loaded = json.loads(m.trace_json)
            if isinstance(loaded, dict):
                trace = loaded
    return MessageDetail(
        id=m.id,
        role=m.role,
        content=m.content,
        sources=sources,
        trace=trace,
        tokens_in=m.tokens_in,
        tokens_out=m.tokens_out,
        latency_ms=m.latency_ms,
        created_at=m.created_at,
    )


def _dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


def _serialize_event_error(exc: Exception) -> str:
    """Terminal ``error`` SSE event for a crashed turn.

    Ordering rule 8 (docs/sse-protocol.md): an error event terminates the
    stream — nothing, including ``done``, may follow it.
    """
    return _serialize_event(ErrorEvent(code="fatal", message=str(exc), recoverable=False))


__all__ = ["router"]
