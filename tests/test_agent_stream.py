"""Streaming agent-turn runner tests.

``iter_agent_turn_events`` drives the same pydantic-ai agent as
``run_one_turn`` but yields the frozen SSE event vocabulary. These tests call
it **in-process** (no HTTP) against a real ``AppState`` over the tiny-ZIM
fixture, with the model injected as a ``FunctionModel`` stub via the
``_make_model`` seam — so the ``search``/``read_article`` tool closures hit the
real retrieval pipeline while the "model" is fully deterministic.

Contract under test (docs/sse-protocol.md): sources(merge=false) → status →
token → [answer_reset → token] → [sources(merge=true)] → citations → trace →
done.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

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
from vesta.api import agent_chat
from vesta.main import create_app

# ── Fixture: a real AppState over the tiny ZIM ──────────────────────────────


@pytest_asyncio.fixture
async def state(tmp_path: Path) -> AsyncIterator[Any]:
    """Build the app, drive its lifespan, and hand back ``app.state.vesta``.

    The lifespan scan registers the tiny-ZIM fixture (and probes the index /
    mines aliases), binding the registry the agent's search/read_article tools
    dispatch against — so a tool round genuinely hits the retrieval pipeline.
    """
    zims_dir = tmp_path / "zims"
    zims_dir.mkdir(parents=True, exist_ok=True)
    from fixtures.tiny_zim import build_tiny_zim

    build_tiny_zim(zims_dir / "tiny.zim")
    os.environ["data.dir"] = str(tmp_path)
    try:
        app = create_app()
        async with app.router.lifespan_context(app):
            yield app.state.vesta
    finally:
        os.environ.pop("data.dir", None)


@pytest.fixture(autouse=True)
def _no_llm_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detach the lifespan-bound LLM runtime for this module.

    The answer path warms the bound runtime; these tests stub
    the model itself, and the real runtime (source=local, no model configured
    under the test env pin) would fail the warm-up and turn every stream into
    an error event. ``None`` means "no warm-up; settings fallback".
    The runtime-owned tests at the bottom of this
    module re-bind a :class:`~fixtures.llm_runtime.FakeLlmRuntime` on top.
    """
    monkeypatch.setattr("vesta.inference.get_runtime", lambda: None)


# ── FunctionModel stub helpers ──────────────────────────────────────────────


def _dummy_request(messages: list[ModelMessage], info: Any) -> ModelResponse:
    """Non-streaming fallback used by the recovery/no-tool ``agent.run`` paths."""
    return ModelResponse(parts=[TextPart(content="stub fallback answer [1]")])


def _stub_model(
    *,
    tool_call: tuple[str, str] | None = None,
    text: str = "The Battle of Hastings was in 1066 [1].",
    crash_stream: bool = False,
    seen: list[list[ModelMessage]] | None = None,
) -> FunctionModel:
    """Build a ``FunctionModel`` whose stream drives the agent's tool loop.

    Round 0 (no ``tool-return`` part yet) either yields a ``DeltaToolCall``
    (``tool_call=(name, json_args)``) or streams ``text`` directly. Round 1
    (after a tool result is in the history) always streams ``text``. When
    ``crash_stream`` is set, round 0 raises ``UsageLimitExceeded`` — the
    ``__aenter__`` recovery the runner must handle. ``seen`` captures the
    ``messages`` of the first model call so tests can assert history threading.
    """

    async def stream_fn(messages: list[ModelMessage], info: Any) -> AsyncIterator[Any]:
        if seen is not None:
            seen.append(list(messages))
        if crash_stream:
            if False:  # make this an async generator so function.py can peek it
                yield None
            raise UsageLimitExceeded("request limit hit")
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


async def _collect(state: Any, question: str, **kwargs: Any) -> list[Any]:
    return [ev async for ev in agent_chat.iter_agent_turn_events(state, None, question, **kwargs)]


def _types(events: list[Any]) -> list[str]:
    return [type(e).__name__ for e in events]


# ── (a) happy path: text only, no tools ─────────────────────────────────────


@pytest.mark.asyncio
async def test_happy_path_text_only(state: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: _stub_model())
    events = await _collect(state, "Einstein")

    types = _types(events)
    assert types[0] == "SourcesEvent"
    assert isinstance(events[0], SourcesEvent)
    assert events[0].merge is False
    assert events[0].cards  # the Einstein pre-seed card

    # status(reading) → status(generating) → token(s)
    statuses = [e for e in events if isinstance(e, StatusEvent)]
    assert statuses[0].phase == "reading"
    assert statuses[1].phase == "generating"
    tokens = [e for e in events if isinstance(e, TokenEvent)]
    assert tokens
    assert "".join(t.text for t in tokens) == "The Battle of Hastings was in 1066 [1]."

    # citations after tokens, answer_text non-null, span resolves to card 0
    citations = [e for e in events if isinstance(e, CitationsEvent)]
    assert len(citations) == 1
    assert citations[0].answer_text is not None
    assert len(citations[0].spans) >= 1
    assert citations[0].spans[0].source_index == 0

    # trace last data event, done terminates
    assert isinstance(events[-2], TraceEvent)
    assert isinstance(events[-1], DoneEvent)
    assert events.index(next(e for e in events if isinstance(e, TraceEvent))) > events.index(
        next(e for e in events if isinstance(e, CitationsEvent))
    )


# ── (b) read_article tool round emits a searching status with the source ────


@pytest.mark.asyncio
async def test_read_article_round_emits_searching_status(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Round 0 calls read_article(1) on the pre-seeded Einstein card, round 1 answers.
    stub = _stub_model(tool_call=("read_article", '{"n": 1}'), text="Answer [1].")
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: stub)
    events = await _collect(state, "Einstein")

    searching = [e for e in events if isinstance(e, StatusEvent) and e.phase == "searching"]
    assert searching, "expected a searching status from the read_article tool round"
    assert any("1" in s.detail for s in searching)

    # The turn still completes cleanly.
    assert isinstance(events[-1], DoneEvent)
    tokens = "".join(t.text for t in events if isinstance(t, TokenEvent))
    assert tokens == "Answer [1]."


# ── (c) mid-turn discovery produces a merge sources event ───────────────────


@pytest.mark.asyncio
async def test_merge_sources_for_mid_turn_discovery(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Round-0 pre-seed finds nothing (the question is gibberish), then the model
    # calls search("Einstein"), which surfaces a card the seed did NOT. That is
    # a delta → merge sources event, continuing 0-based numbering.
    stub = _stub_model(tool_call=("search", '{"query": "Einstein"}'), text="Answer [1].")
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: stub)
    events = await _collect(state, "zzzzzznothing")

    sources = [e for e in events if isinstance(e, SourcesEvent)]
    seed = sources[0]
    assert seed.merge is False
    assert seed.cards == ()  # empty Round-0 seed

    merges = [e for e in sources if e.merge is True]
    assert len(merges) == 1
    delta_cards = merges[0].cards
    assert delta_cards, "expected the searched card in the merge event"
    # Continuing numbering: seed had 0 cards, so the first delta card is n=1.
    assert all(c.n > len(seed.cards) for c in delta_cards)
    assert [c.n for c in delta_cards] == sorted(c.n for c in delta_cards)

    # A citation resolves into the merged range (source_index 0 = first delta card).
    citations = [e for e in events if isinstance(e, CitationsEvent)]
    assert len(citations) == 1
    assert any(s.source_index < len(seed.cards) + len(delta_cards) for s in citations[0].spans)

    # Merge event sits after tokens, before citations.
    tokens = [e for e in events if isinstance(e, TokenEvent)]
    assert tokens
    merge_idx = events.index(merges[0])
    assert merge_idx > max(events.index(t) for t in tokens)
    assert merge_idx < events.index(citations[0])


# ── (d) UsageLimitExceeded fallback: answer_reset then a single token ───────


@pytest.mark.asyncio
async def test_usage_limit_fallback_emits_reset_then_single_token(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _stub_model(crash_stream=True)
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: stub)
    events = await _collect(state, "Einstein")

    resets = [e for e in events if isinstance(e, AnswerResetEvent)]
    assert len(resets) == 1
    assert resets[0].reason == "fallback"
    reset_idx = events.index(resets[0])

    tokens_after = [e for e in events[reset_idx + 1 :] if isinstance(e, TokenEvent)]
    assert len(tokens_after) == 1
    assert tokens_after[0].text  # the no-tool fallback answer

    assert isinstance(events[-1], DoneEvent)


# ── (e) empty retrieval still terminates cleanly ────────────────────────────


@pytest.mark.asyncio
async def test_empty_retrieval_terminates_cleanly(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _stub_model(text="nothing found here")
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: stub)
    events = await _collect(state, "zzzzzznothing")

    assert isinstance(events[0], SourcesEvent)
    assert events[0].cards == ()  # empty sources
    # No error event and the stream terminates with done — never crashes.
    assert not any(isinstance(e, ErrorEvent) for e in events)
    assert isinstance(events[-1], DoneEvent)


# ── (f) message_history is threaded into the model call ─────────────────────


@pytest.mark.asyncio
async def test_message_history_threaded(state: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[ModelMessage]] = []
    stub = _stub_model(text="follow-up answer [1]", seen=seen)
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: stub)

    prior = ModelRequest(parts=[UserPromptPart(content="prior turn")])
    events = await _collect(state, "Einstein", message_history=[prior])

    # The stub saw the prior turn in round 0's messages.
    assert seen, "the stub model was never invoked"
    first = seen[0]
    text = "".join(
        p.content
        for m in first
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, UserPromptPart)
    )
    assert "prior turn" in text

    # The turn itself still completes.
    assert isinstance(events[-1], DoneEvent)
    assert any(isinstance(e, TokenEvent) for e in events)


# ── The runtime-owned answer path ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resolve_llm_uses_runtime_target_not_settings(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(a) With ``source=local`` and a bound runtime, the agent's endpoint is
    the supervisor's base URL and the router-resolved model id — never the
    ``inference.llm.endpoint_url`` setting."""
    from fixtures.llm_runtime import FakeLlmRuntime

    fake = FakeLlmRuntime(base_url="http://supervisor.test:8081/v1", model_id="qwen3.5-4b")
    monkeypatch.setattr("vesta.inference.get_runtime", lambda: fake)

    captured: list[tuple[str, str, str]] = []

    def _capture(model_id: str, endpoint: str, api_key: str) -> object:
        captured.append((model_id, endpoint, api_key))
        return object()

    monkeypatch.setattr(agent_chat, "_make_model", _capture)
    os.environ["inference.llm.endpoint_url"] = "http://decoy.remote:1234/v1"
    try:
        await agent_chat._build_turn(state, None, "Einstein")
    finally:
        os.environ.pop("inference.llm.endpoint_url", None)

    assert captured == [("qwen3.5-4b", "http://supervisor.test:8081/v1", "local")]


@pytest.mark.asyncio
async def test_warmup_emits_reading_status_before_first_token(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(b) A cold local runtime surfaces its loading message as a ``reading``
    status before the first token (D13 — reuses ``reading``, no new phase),
    and the turn stamps ``mark_used`` on completion."""
    from fixtures.llm_runtime import FakeLlmRuntime

    fake = FakeLlmRuntime(status_messages=["Loading Qwen3.5 4B into memory…"])
    monkeypatch.setattr("vesta.inference.get_runtime", lambda: fake)
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: _stub_model())

    events = await _collect(state, "Einstein")

    loading_idx = [
        i for i, e in enumerate(events) if isinstance(e, StatusEvent) and "Loading" in e.detail
    ]
    assert loading_idx, "expected a reading status carrying the loading message"
    assert events[loading_idx[0]].phase == "reading"
    first_token = next(i for i, e in enumerate(events) if isinstance(e, TokenEvent))
    assert loading_idx[0] < first_token
    # Turn completed → the runtime's last_used was stamped (D4).
    assert fake.used >= 1
    assert isinstance(events[-1], DoneEvent)
