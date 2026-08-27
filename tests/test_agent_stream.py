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
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
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


_COUNT_DETAIL = re.compile(r"\d+ sources")


def _reading_statuses(events: list[Any]) -> list[StatusEvent]:
    return [e for e in events if isinstance(e, StatusEvent)]


@pytest.mark.asyncio
async def test_warmup_restores_reading_status_after_cold_load(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a cold load's ``Loading <model> into memory…`` statuses flush,
    the runner re-emits the pre-warmup reading detail (``N sources``). Without
    this the client sticks on the loading message through the whole first
    inference gap (it preserves explicit lifecycle details verbatim while the
    phase stays ``reading``) — users read it as a crash."""
    from fixtures.llm_runtime import FakeLlmRuntime

    fake = FakeLlmRuntime(status_messages=["Loading Qwen3.5 4B into memory…"])
    monkeypatch.setattr("vesta.inference.get_runtime", lambda: fake)
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: _stub_model())

    events = await _collect(state, "Einstein")
    statuses = _reading_statuses(events)

    loading_idx = next(i for i, s in enumerate(statuses) if "Loading" in s.detail)
    count_idxs = [i for i, s in enumerate(statuses) if _COUNT_DETAIL.fullmatch(s.detail)]
    # Initial "N sources" … Loading … restored "N sources" — same detail, same
    # phase, all before the first token.
    assert len(count_idxs) == 2
    assert count_idxs[0] < loading_idx < count_idxs[1]
    assert statuses[count_idxs[0]].detail == statuses[count_idxs[1]].detail
    assert statuses[count_idxs[1]].phase == "reading"
    first_token = next(i for i, e in enumerate(events) if isinstance(e, TokenEvent))
    assert count_idxs[0] < first_token and loading_idx < first_token


@pytest.mark.asyncio
async def test_warm_runtime_does_not_duplicate_reading_status(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A runtime that was already loaded reports no warm-up statuses, so no
    restore status either — the turn's only count detail stays the original."""
    from fixtures.llm_runtime import FakeLlmRuntime

    fake = FakeLlmRuntime()
    monkeypatch.setattr("vesta.inference.get_runtime", lambda: fake)
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: _stub_model())

    events = await _collect(state, "Einstein")
    statuses = _reading_statuses(events)
    details = [s.detail for s in statuses]
    assert not any("Loading" in d for d in details)
    assert sum(1 for d in details if _COUNT_DETAIL.fullmatch(d)) == 1


@pytest.mark.asyncio
async def test_streaming_turn_guards_in_flight_and_updates_used(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AUDIT_0824 I5: streaming agent turn tracks in-flight generation and stamps mark_used."""
    from fixtures.llm_runtime import FakeLlmRuntime

    fake = FakeLlmRuntime()
    monkeypatch.setattr("vesta.inference.get_runtime", lambda: fake)

    in_flight_during_stream: list[int] = []

    def _make_tracking_model(*args: Any, **kwargs: Any) -> Any:
        async def _stream(messages: list[ModelMessage], info: Any) -> AsyncIterator[str]:
            in_flight_during_stream.append(fake.in_flight_count)
            yield "Albert Einstein was a physicist [1]."

        return FunctionModel(
            function=lambda *a, **k: ModelResponse(
                parts=[TextPart(content="Albert Einstein was a physicist [1].")]
            ),
            stream_function=_stream,
        )

    monkeypatch.setattr(agent_chat, "_make_model", _make_tracking_model)

    events = await _collect(state, "Einstein")

    assert in_flight_during_stream == [1]
    assert fake.in_flight_count == 0
    assert fake.used >= 2
    assert isinstance(events[-1], DoneEvent)


# ── (g) UsageLimitExceeded keeps the main run's tokens in the trace ─────────


@pytest.mark.asyncio
async def test_usage_limit_crash_keeps_main_run_tokens(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The streaming driver threads a ``RunUsage`` accumulator into
    ``run_stream`` (like ``run_one_turn``), so a mid-run
    ``UsageLimitExceeded`` keeps whatever tokens the main run already burned.
    The trace then reports main-run + recovery tokens, not recovery alone."""

    async def burn_stream(messages: list[ModelMessage], info: Any) -> AsyncIterator[Any]:
        has_return = any(
            isinstance(m, ModelRequest)
            and any(getattr(p, "part_kind", None) == "tool-return" for p in m.parts)
            for m in messages
        )
        if not has_return:
            # Round 0 completes as a real billed request: a search tool call.
            yield {0: DeltaToolCall(name="search", json_args='{"query": "Einstein"}')}
            return
        if False:  # make this an async generator so function.py can peek it
            yield None
        raise UsageLimitExceeded("request limit hit")

    burn = FunctionModel(function=_dummy_request, stream_function=burn_stream)
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: burn)
    events = await _collect(state, "Einstein")

    # The crash recovered via the no-tool fallback (answer_reset → token).
    resets = [e for e in events if isinstance(e, AnswerResetEvent)]
    assert len(resets) == 1 and resets[0].reason == "fallback"

    trace = dict(next(e for e in events if isinstance(e, TraceEvent)).trace)

    # Control: identical recovery path, but the main run crashes before any
    # request completes — its trace holds recovery tokens only. The burn run
    # must report strictly more, proving the main-run usage was retained.
    instant = _stub_model(crash_stream=True)
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: instant)
    ctrl_events = await _collect(state, "Einstein")
    ctrl_trace = dict(next(e for e in ctrl_events if isinstance(e, TraceEvent)).trace)

    assert trace["total_tokens"] > ctrl_trace["total_tokens"]
    assert trace["input_tokens"] > 0
    assert trace["output_tokens"] > 0


# ── (h) follow-up crash after a search still delivers the latched sources ───


@pytest.mark.asyncio
async def test_followup_crash_after_search_still_emits_latched_sources(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A follow-up's FIRST ``sources`` event is buffered by the ``search`` tool
    closure and drained only inside ``run_stream``'s body. When a LATER model
    request crashes inside ``__aenter__`` (UsageLimitExceeded on the round
    after the search), the buffer must still reach the client before the
    fallback answer — which cites those cards — and the trailing merge delta
    must not replay them (docs/sse-protocol.md "Sources merge" numbering).
    """

    async def stream_fn(messages: list[ModelMessage], info: Any) -> AsyncIterator[Any]:
        has_return = any(
            isinstance(m, ModelRequest)
            and any(getattr(p, "part_kind", None) == "tool-return" for p in m.parts)
            for m in messages
        )
        if has_return:
            # Crash on the round AFTER the search executed — the __aenter__
            # path where the buffered events were never drained.
            raise UsageLimitExceeded("request limit hit")
        yield {0: DeltaToolCall(name="search", json_args='{"query": "Einstein"}')}

    stub = FunctionModel(function=_dummy_request, stream_function=stream_fn)
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: stub)

    history = [ModelRequest(parts=[UserPromptPart(content="Who was Einstein?")])]
    events = await _collect(state, "When was he born?", message_history=history)

    sources = [e for e in events if isinstance(e, SourcesEvent)]
    assert sources, "latched follow-up sources event lost on post-search crash"
    assert len(sources) == 1  # delivered once — the merge delta excludes it
    first = sources[0]
    assert first.merge is False
    assert first.cards

    src_idx = events.index(first)
    resets = [e for e in events if isinstance(e, AnswerResetEvent)]
    assert [r.reason for r in resets] == ["fallback"]
    reset_idx = events.index(resets[0])
    # Protocol ordering: sources before the fallback regenerate.
    assert src_idx < reset_idx
    assert all(not isinstance(e, TokenEvent) for e in events[:src_idx])

    tokens_after = [e for e in events[reset_idx + 1 :] if isinstance(e, TokenEvent)]
    assert len(tokens_after) == 1
    assert tokens_after[0].text == "stub fallback answer [1]"

    assert isinstance(events[-1], DoneEvent)


# ── (i) terminal errors end the stream (protocol ordering rule 8) ───────────


@pytest.mark.asyncio
async def test_empty_answer_ends_stream_at_budget_exhausted_error(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model that produces no answer text emits the ``budget_exhausted``
    error — and nothing after it: an error event terminates the stream, so no
    ``trace``/``done`` may follow (docs/sse-protocol.md ordering rule 8)."""
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: _stub_model(text=""))
    # An empty answer normally routes into the abstention-retry gate; keep it
    # so the turn reaches the budget_exhausted emitter with nothing to say.
    monkeypatch.setattr(agent_chat, "looks_abstained", lambda _answer: False)
    events = await _collect(state, "Einstein")

    errors = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(errors) == 1
    assert errors[0].code == "budget_exhausted"
    assert isinstance(events[-1], ErrorEvent)
    # No trace/done tail after the terminal error.
    assert not isinstance(events[-2], TraceEvent)
    assert sum(isinstance(e, DoneEvent) for e in events) == 0


@pytest.mark.asyncio
async def test_warmup_failure_ends_stream_at_no_llm_error(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A runtime that cannot come up emits one terminal ``no_llm`` error and
    the generator stops there — no ``done`` after the error (rule 8)."""
    from fixtures.llm_runtime import FakeLlmRuntime
    from vesta.inference.runtime import LlmRuntimeError

    fake = FakeLlmRuntime(error=LlmRuntimeError("no model matched"))
    monkeypatch.setattr("vesta.inference.get_runtime", lambda: fake)

    events = await _collect(state, "Einstein")

    errors = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(errors) == 1
    assert errors[0].code == "no_llm"
    assert isinstance(events[-1], ErrorEvent)
    assert sum(isinstance(e, DoneEvent) for e in events) == 0


# ── (j) answer_reset outcome-first ordering (AUDIT_0824 C5) ─────────────────


def _assert_no_dangling_reset(events: list[Any]) -> None:
    """Every ``answer_reset`` must be IMMEDIATELY followed by a replacement
    token — an erase with no replacement leaves the client with no answer
    text at all (the bug class AUDIT_0824 C5 fixes)."""
    for i, event in enumerate(events):
        if isinstance(event, AnswerResetEvent):
            assert i + 1 < len(events), f"dangling {event.reason} reset at stream end"
            assert isinstance(events[i + 1], TokenEvent), (
                f"{event.reason} reset not immediately followed by a token"
            )


def _overflow_request(messages: list[ModelMessage], info: Any) -> ModelResponse:
    """The non-streaming recovery paths see this as a context overflow."""
    raise ModelHTTPError(
        status_code=400,
        model_name="stub",
        body={"message": "Context size has been exceeded."},
    )


@pytest.mark.asyncio
async def test_failed_fallback_emits_no_dangling_reset(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The main run crashes AND the no-tool fallback itself overflows: the
    pre-fix ordering emitted ``answer_reset(fallback)`` up front and then
    nothing — the client erased its text for no replacement. The reset must
    not fire at all when the arm fails."""

    async def crash_stream(messages: list[ModelMessage], info: Any) -> AsyncIterator[Any]:
        if False:  # make this an async generator so function.py can peek it
            yield None
        raise UsageLimitExceeded("request limit hit")

    stub = FunctionModel(function=_overflow_request, stream_function=crash_stream)
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: stub)
    events = await _collect(state, "Einstein")

    assert [e for e in events if isinstance(e, AnswerResetEvent)] == []
    _assert_no_dangling_reset(events)
    # The empty final answer terminates the stream per rule 8.
    assert isinstance(events[-1], ErrorEvent)
    assert events[-1].code == "budget_exhausted"


def _reask_snapshot() -> Any:
    """Forced-on compact re-ask with a user-set cap of one tool round, so the
    second tool attempt trips the round cap and latches the trigger."""
    from vesta.config.settings import SettingsSnapshot, all_settings

    values: dict[str, object] = {s.key: s.default for s in all_settings().values()}
    values.update({"answer.agent.compact_reask": "on", "answer.agent.max_tool_rounds": 1})
    return SettingsSnapshot(values=values)


def _round_cap_reask_model(
    *,
    final_text: str = "The steered answer [1].",
    reask_fn: Any = None,
) -> FunctionModel:
    """Round 0 reads a card (consuming ``max_tool_rounds=1``), round 1
    attempts another call — steered by the round cap — and the next round
    streams the steered answer. ``reask_fn`` is what the recovery core's
    non-streaming re-ask request sees (default: the stub fallback answer)."""

    async def default_fn(messages: list[ModelMessage], info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="stub fallback answer [1]")])

    step = {"n": 0}

    async def stream_fn(messages: list[ModelMessage], info: Any) -> AsyncIterator[Any]:
        step["n"] += 1
        # Distinct cards: a repeat read would hit the dedup guard before the
        # round-cap probe and never latch the trigger.
        if step["n"] == 1:
            yield {0: DeltaToolCall(name="read_article", json_args='{"n": 1}')}
        elif step["n"] == 2:
            yield {0: DeltaToolCall(name="read_article", json_args='{"n": 2}')}
        else:
            yield final_text

    return FunctionModel(function=reask_fn or default_fn, stream_function=stream_fn)


@pytest.mark.asyncio
async def test_compact_reask_resets_only_before_replacement(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On success the ``compact_reask`` reset comes AFTER the steered tokens,
    immediately before the single replacement token — never up front."""

    def reask_fn(messages: list[ModelMessage], info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="The re-asked answer [1].")])

    stub = _round_cap_reask_model(final_text="The steered answer [1].", reask_fn=reask_fn)
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: stub)
    events = [
        ev async for ev in agent_chat.iter_agent_turn_events(state, _reask_snapshot(), "Einstein")
    ]

    resets = [e for e in events if isinstance(e, AnswerResetEvent) and e.reason == "compact_reask"]
    assert len(resets) == 1
    idx = events.index(resets[0])
    # Outcome-first: the steered tokens streamed BEFORE the reset…
    first_token_idx = min(i for i, e in enumerate(events) if isinstance(e, TokenEvent))
    assert idx > first_token_idx
    assert any("steered" in t.text for t in events[:idx] if isinstance(t, TokenEvent))
    # …and the reset is immediately followed by the single replacement token.
    assert isinstance(events[idx + 1], TokenEvent)
    assert events[idx + 1].text == "The re-asked answer [1]."
    trace = dict(next(e for e in events if isinstance(e, TraceEvent)).trace)
    assert trace["compact_reask"]["fired"] is True
    assert trace["compact_reask"]["trigger"] == "round_cap"
    _assert_no_dangling_reset(events)


@pytest.mark.asyncio
async def test_failed_compact_reask_emits_no_dangling_reset(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the re-ask itself fails (usage cap), the steered answer stands and
    NO reset may have been emitted — the old up-front erase left clients with
    an empty accumulator over a still-live answer."""

    def usage_exceeded(messages: list[ModelMessage], info: Any) -> ModelResponse:
        raise UsageLimitExceeded("request limit hit")

    stub = _round_cap_reask_model(reask_fn=usage_exceeded)
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: stub)
    events = [
        ev async for ev in agent_chat.iter_agent_turn_events(state, _reask_snapshot(), "Einstein")
    ]

    assert [e for e in events if isinstance(e, AnswerResetEvent)] == []
    tokens = "".join(t.text for t in events if isinstance(t, TokenEvent))
    assert "The steered answer" in tokens  # stands, un-erased
    trace = dict(next(e for e in events if isinstance(e, TraceEvent)).trace)
    assert trace["compact_reask"]["fired"] is False
    assert trace["compact_reask"]["trigger"] == "round_cap"
    _assert_no_dangling_reset(events)


def _abstain_model(*, retry_fn: Any) -> FunctionModel:
    """Round 0 streams a refusal; the abstention retry then runs the
    non-streaming ``retry_fn``."""

    async def stream_fn(messages: list[ModelMessage], info: Any) -> AsyncIterator[Any]:
        yield "I cannot find the answer in the provided sources."

    return FunctionModel(function=retry_fn, stream_function=stream_fn)


@pytest.mark.asyncio
async def test_abstention_retry_resets_only_before_replacement(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On success the ``abstention_retry`` reset comes AFTER the refusal
    tokens, immediately before the single replacement token."""

    def retry_fn(messages: list[ModelMessage], info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="The answer is 42 [1].")])

    monkeypatch.setattr(
        agent_chat, "_make_model", lambda *a, **k: _abstain_model(retry_fn=retry_fn)
    )
    events = await _collect(state, "Einstein")

    resets = [e for e in events if isinstance(e, AnswerResetEvent)]
    assert len(resets) == 1
    assert resets[0].reason == "abstention_retry"
    idx = events.index(resets[0])
    # The refusal streamed first; the reset precedes only the replacement.
    refusal_tokens = [e for e in events[:idx] if isinstance(e, TokenEvent)]
    assert any("cannot find" in t.text for t in refusal_tokens)
    assert isinstance(events[idx + 1], TokenEvent)
    assert events[idx + 1].text == "The answer is 42 [1]."
    _assert_no_dangling_reset(events)


@pytest.mark.asyncio
async def test_failed_abstention_retry_emits_no_dangling_reset(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the retry overflows too, the refusal stands and NO reset may have
    been emitted — plan-skip, usage-cap and overflow failures alike keep the
    original answer on every path."""

    monkeypatch.setattr(
        agent_chat, "_make_model", lambda *a, **k: _abstain_model(retry_fn=_overflow_request)
    )
    events = await _collect(state, "Einstein")

    assert [e for e in events if isinstance(e, AnswerResetEvent)] == []
    tokens = "".join(t.text for t in events if isinstance(t, TokenEvent))
    assert "cannot find" in tokens  # the refusal stands, un-erased
    trace = dict(next(e for e in events if isinstance(e, TraceEvent)).trace)
    assert trace["overflow_fallbacks"] == 1
    _assert_no_dangling_reset(events)


@pytest.mark.asyncio
async def test_agent_llm_step_timing_excludes_recovery_wall_time(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agent_llm stage records inference time of the main stream only;
    recovery wall time (e.g. abstention retry or compact reask) is not
    double-counted into agent_llm."""
    import time

    def slow_retry_fn(messages: list[ModelMessage], info: Any) -> ModelResponse:
        time.sleep(0.06)  # 60ms in retry
        return ModelResponse(parts=[TextPart(content="The answer is 42 [1].")])

    monkeypatch.setattr(
        agent_chat, "_make_model", lambda *a, **k: _abstain_model(retry_fn=slow_retry_fn)
    )
    events = await _collect(state, "Einstein")
    trace = dict(next(e for e in events if isinstance(e, TraceEvent)).trace)
    stages = trace["stages"]
    llm_step = next(s for s in stages if s["name"] == "agent_llm")
    # Main stream is instant (< 45ms), while retry added 60ms.
    # If retry was included, llm_step["duration_ms"] would be >= 60ms.
    assert llm_step["duration_ms"] < 50.0
    assert llm_step["outputs"]["answer_chars"] == len(
        "I cannot find the answer in the provided sources."
    )


@pytest.mark.asyncio
async def test_run_one_turn_records_agent_llm_step(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_one_turn records the agent_llm step in ctx.steps with accurate
    inputs/outputs, aligning batch and streaming traces."""
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: _stub_model())
    result = await agent_chat.run_one_turn(
        state,
        None,
        "Einstein",
        model_id="test",
        endpoint="http://test",
    )
    stages = result.trace["stages"]
    stage_names = [s["name"] for s in stages]
    assert "agent_llm" in stage_names
    llm_step = next(s for s in stages if s["name"] == "agent_llm")
    assert llm_step["component"] == "pydantic_ai"
    assert "input_tokens" in llm_step["inputs"]
    assert "output_tokens" in llm_step["outputs"]
    assert "answer_chars" in llm_step["outputs"]
