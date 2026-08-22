"""Context-aware follow-up turns.

A follow-up chat turn (``message_history`` present) must NOT run the Round-0
pre-seed on the raw, decontextualized question — that was the bug ("who died
first" retrieved US presidents instead of resolving to Napoleon/Lafayette). It
must instead hand the conversation to the agent and let it answer from context
or search with a self-contained query.

These tests exercise the mechanism in-process (no HTTP) over the tiny-ZIM
fixture with a ``FunctionModel`` stub: the ``search``/``read_article`` tool
closures hit the real retrieval pipeline while the "model" is deterministic.
The "does the real model resolve context?" question is the manual Chrome repro.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from vesta.answer.contracts import (
    CitationsEvent,
    DoneEvent,
    SourcesEvent,
    StatusEvent,
    TokenEvent,
    TraceEvent,
)
from vesta.api import agent_chat
from vesta.api.agent_chat import _build_turn
from vesta.retrieval.contracts import ScoredPassage, SourceCard
from vesta.zim.types import Passage

# ── Fixture: a lightweight dummy state ──────────────────────────────────────


@pytest.fixture
def state() -> Any:
    """A lightweight dummy state object — _build_tool_runtime is patched."""
    return SimpleNamespace(registry=None, db=None)


@pytest.fixture(autouse=True)
def _no_llm_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detach the lifespan-bound LLM runtime: these tests stub the
    model itself, and the real runtime (no model configured under the test
    env pin) would fail the answer-path warm-up."""
    monkeypatch.setattr("vesta.inference.get_runtime", lambda: None)


def _scored(i: int, text: str) -> ScoredPassage:
    return ScoredPassage(
        passage=Passage(
            zim_id=1,
            path=f"a/{i}",
            ordinal=i,
            char_start=0,
            char_end=len(text),
            breadcrumb=f"Article {i} > Section",
            text=text,
            is_lead=False,
        ),
        score=10.0 - i,
        source_info="test",
    )


def _card(i: int, snippet: str) -> SourceCard:
    return SourceCard(
        zim_id=1,
        path=f"a/{i}",
        title=f"Article {i}",
        snippet=snippet,
        breadcrumb=f"Article {i} > Section",
        score=10.0 - i,
        source="test",
    )


class FakeToolRuntime:
    def __init__(
        self,
        passages: list[ScoredPassage] | None = None,
        cards: list[SourceCard] | None = None,
        article: str = "article",
    ):
        from vesta.answer.tools import SearchToolResult

        p = passages if passages is not None else [_scored(0, "Einstein text")]
        c = cards if cards is not None else [_card(0, "Einstein")]
        self._result = SearchToolResult(text="formatted", passages=tuple(p), cards=tuple(c))
        self._article = article
        self.search_calls: list[str] = []
        self.search_exact_calls: list[str] = []
        self.read_calls: list[tuple[int, str]] = []

    async def search(self, query: str, scope: str) -> Any:
        self.search_calls.append(query)
        return self._result

    async def search_exact(self, query: str, scope: str) -> Any:
        self.search_exact_calls.append(query)
        return self._result

    async def read_article(self, zim_id: int, path: str) -> str:
        self.read_calls.append((zim_id, path))
        return self._article


# ── FunctionModel stub helpers (mirror test_agent_stream.py) ────────────────


def _dummy_request(messages: list[ModelMessage], info: Any) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content="stub fallback answer")])


def _stub_model(
    *,
    tool_call: tuple[str, str] | None = None,
    text: str = "Lafayette was born first, in 1757.",
    seen: list[list[ModelMessage]] | None = None,
) -> FunctionModel:
    async def stream_fn(messages: list[ModelMessage], info: Any) -> AsyncIterator[Any]:
        if seen is not None:
            seen.append(list(messages))
        has_return = any(
            isinstance(m, ModelRequest)
            and any(getattr(p, "part_kind", None) == "tool-return" for p in m.parts)
            for m in messages
        )
        if has_return:
            yield text
        elif tool_call is not None:
            name, json_args = tool_call
            yield {0: DeltaToolCall(name=name, json_args=json_args)}
        else:
            yield text

    return FunctionModel(function=_dummy_request, stream_function=stream_fn)


def _prior_turns() -> list[ModelMessage]:
    """A reconstructed conversation: turn 1 established Lafayette & Napoleon."""
    return [
        ModelRequest(
            parts=[UserPromptPart(content="How long between Lafayette and Napoleon were born")]
        ),
        ModelResponse(
            parts=[
                TextPart(
                    content=(
                        "Lafayette was born on September 6, 1757 and Napoleon on "
                        "August 15, 1769 — about 12 years apart."
                    )
                )
            ]
        ),
    ]


async def _collect(state: Any, question: str, **kwargs: Any) -> list[Any]:
    return [ev async for ev in agent_chat.iter_agent_turn_events(state, None, question, **kwargs)]


# ── _build_turn branching (direct field assertions) ─────────────────────────


@pytest.mark.asyncio
async def test_build_turn_followup_skips_preseed(state: Any) -> None:
    """History present + setting default-on → no pre-seed, follow-up prompt."""
    ctx = await _build_turn(state, None, "Who died first", message_history=_prior_turns())

    assert ctx.follow_up is True
    assert ctx.seed_hit is False
    assert ctx.seed_text == ""
    assert ctx.turn_cards == {}  # no retrieval ran
    assert ctx.user_message == "Who died first"  # no "Initial sources" preamble
    # The follow-up directive is appended (replaces the "initial sources" framing).
    assert "continuing an existing conversation" in ctx.sys_prompt
    assert "SELF-CONTAINED" in ctx.sys_prompt
    # The follow-up must nudge the model to search instead of asking permission
    # (chat 295 regression: follow-up "how much should I water them" got
    # "Would you like me to look that up now?" with 0 search calls).
    assert "Act, do not ask to act" in ctx.sys_prompt


@pytest.mark.asyncio
async def test_build_turn_setting_off_legacy_preseed(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the setting off, a follow-up pre-seeds on the raw question (rollback)."""
    monkeypatch.setattr(agent_chat, "_build_tool_runtime", lambda *a, **k: FakeToolRuntime())
    monkeypatch.setattr(agent_chat, "_contextual_followups_enabled", lambda sn: False)
    ctx = await _build_turn(state, None, "Who died first", message_history=_prior_turns())

    assert ctx.follow_up is False
    # Legacy path: pre-seed ran on the bare "Who died first" → some retrieval result.
    assert "continuing an existing conversation" not in ctx.sys_prompt


# ── iter_agent_turn_events: from-context path (no tool call) ────────────────


@pytest.mark.asyncio
async def test_followup_answers_from_context_no_sources(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fact is in the conversation → the agent answers directly, no retrieval."""
    seen: list[list[ModelMessage]] = []
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: _stub_model(seen=seen))
    events = await _collect(state, "Who was born first", message_history=_prior_turns())

    # Zero sources events: the answer came from the conversation, not retrieval.
    assert not [e for e in events if isinstance(e, SourcesEvent)]

    # Event shape: status(reading) → status(generating) → token(s) → citations → trace → done.
    statuses = [e for e in events if isinstance(e, StatusEvent)]
    assert statuses[0].phase == "reading"
    assert "Considering" in statuses[0].detail
    assert any(s.phase == "generating" for s in statuses)

    tokens = "".join(t.text for t in events if isinstance(t, TokenEvent))
    assert "Lafayette" in tokens

    citations = [e for e in events if isinstance(e, CitationsEvent)]
    assert len(citations) == 1
    assert citations[0].spans == ()  # nothing to cite — no sources

    trace = next(e for e in events if isinstance(e, TraceEvent))
    assert trace.trace["followup"] is True
    assert trace.trace["search_calls"] == 0
    assert trace.trace["card_count"] == 0
    assert isinstance(events[-1], DoneEvent)

    # The agent saw the conversation history.
    assert seen, "the stub model was never invoked"
    seen_text = "".join(getattr(p, "content", "") for m in seen[0] for p in getattr(m, "parts", []))
    assert "Lafayette" in seen_text


# ── iter_agent_turn_events: search path (live sources) ─────────────────────


@pytest.mark.asyncio
async def test_followup_search_emits_live_sources_before_tokens(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fact is NOT in the conversation → the agent searches; cards appear live."""
    monkeypatch.setattr(agent_chat, "_build_tool_runtime", lambda *a, **k: FakeToolRuntime())
    seen: list[list[ModelMessage]] = []
    # The model resolves context and searches for the prior-turn entity (Einstein),
    # rather than the bare follow-up words.
    stub = _stub_model(
        tool_call=("search", '{"query": "Einstein"}'),
        text="Found it via a fresh search [1].",
        seen=seen,
    )
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: stub)
    events = await _collect(state, "tell me more about him", message_history=_prior_turns())

    sources = [e for e in events if isinstance(e, SourcesEvent)]
    # Exactly one sources event, merge=False (this turn's initial set, live-emitted).
    assert len(sources) == 1
    assert sources[0].merge is False
    assert sources[0].cards, "the searched card must surface"

    # The sources event precedes the first token (cards visible during search).
    first_token_idx = next(i for i, e in enumerate(events) if isinstance(e, TokenEvent))
    assert events.index(sources[0]) < first_token_idx

    # A searching status preceded the sources (truthful tool feedback).
    searching = [e for e in events if isinstance(e, StatusEvent) and e.phase == "searching"]
    assert searching
    assert events.index(searching[0]) < events.index(sources[0])

    # Citation resolves into the discovered card.
    citations = [e for e in events if isinstance(e, CitationsEvent)]
    assert len(citations) == 1
    assert citations[0].spans, "the [1] marker should resolve to the searched card"
    assert citations[0].spans[0].source_index == 0

    trace = next(e for e in events if isinstance(e, TraceEvent))
    assert trace.trace["followup"] is True
    assert trace.trace["search_calls"] >= 1
    assert trace.trace["card_count"] >= 1
