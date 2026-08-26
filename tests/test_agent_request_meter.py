"""Per-request accounting tests (meter → agent_chat trace).

Same in-process recipe as tests/test_agent_economy.py: the tool runtime is a
fake and the model is a ``FunctionModel`` stub swapped at the ``_make_model``
seam. The stub pins an explicit ``RequestUsage`` on every ``ModelResponse``,
so the meter's ``peak_input_tokens`` / ``requests`` / ``request_log`` are
asserted against known numbers rather than FunctionModel's estimates.

The meter is measurement-only: these tests double as the zero-behaviour-change
guard — the tool loop, fallback, and answer must behave exactly as before with
the ``_MeteredModel`` wrapper in place.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.usage import RequestUsage

from vesta.answer.contracts import TraceEvent
from vesta.answer.tools import SearchToolResult
from vesta.api import agent_chat
from vesta.main import create_app
from vesta.retrieval.contracts import ScoredPassage, SourceCard
from vesta.zim.types import Passage

# ── Fixtures (recipe: tests/test_agent_economy.py) ──────────────────────────


@pytest_asyncio.fixture
async def state(tmp_path: Path) -> AsyncIterator[Any]:
    """A real AppState over the tiny ZIM (same recipe as test_agent_stream)."""
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
    """Detach the lifespan-bound runtime: hardware is None, no warm-up."""
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
    """The tool-runtime surface _do_search/read_article dispatch to."""

    def __init__(self, passages: list[ScoredPassage], cards: list[SourceCard], article: str):
        self._result = SearchToolResult(
            text="formatted", passages=tuple(passages), cards=tuple(cards)
        )
        self._article = article
        self.read_calls: list[tuple[int, str]] = []
        self.must_includes: list[str] = []

    async def search(self, query: str, scope: str) -> SearchToolResult:
        return self._result

    async def search_exact(self, query: str, scope: str) -> SearchToolResult:
        return self._result

    async def read_article(self, zim_id: int, path: str, *, must_include: str = "") -> str:
        self.read_calls.append((zim_id, path))
        self.must_includes.append(must_include)
        return self._article


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, runtime: FakeToolRuntime) -> None:
    monkeypatch.setattr(agent_chat, "_build_tool_runtime", lambda *a, **k: runtime)


# ── Stub helpers ────────────────────────────────────────────────────────────


def _usage_fn_model(
    tool_calls: list[tuple[str, dict[str, Any]]], usages: list[RequestUsage]
) -> FunctionModel:
    """Non-streaming model: one scripted tool call per round then a final
    answer, with a pinned ``RequestUsage`` per response (round i gets
    ``usages[i]``; the final answer gets ``usages[-1]``)."""

    def fn(messages: list[ModelMessage], info: Any) -> ModelResponse:
        idx = sum(
            1
            for m in messages
            if isinstance(m, ModelRequest)
            and any(getattr(p, "part_kind", None) == "tool-return" for p in m.parts)
        )
        if idx < len(tool_calls):
            name, args = tool_calls[idx]
            return ModelResponse(parts=[ToolCallPart(tool_name=name, args=args)], usage=usages[idx])
        return ModelResponse(parts=[TextPart(content="Final answer [1].")], usage=usages[-1])

    async def stream_fn(messages: list[ModelMessage], info: Any) -> AsyncIterator[Any]:
        yield "unused"  # pragma: no cover — run_one_turn never streams

    return FunctionModel(function=fn, stream_function=stream_fn)


#: Four read rounds → 5 requests total (4 tool rounds + the final answer).
_READ_SCRIPT = [("read_article", {"n": i + 1}) for i in range(4)]


# ── run_one_turn: peak accounting through the fake-model seam ───────────────


async def test_peak_tokens_and_request_count_in_trace(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 5-request turn reports the scripted peak, request count, and a
    per-request log whose token column matches the scripted usages."""
    runtime = FakeToolRuntime(
        [_scored(i, f"passage {i}") for i in range(2)],
        [_card(i, f"snippet {i}") for i in range(2)],
        "full article body text",
    )
    _patch_runtime(monkeypatch, runtime)
    scripted = [
        RequestUsage(input_tokens=4_100, output_tokens=5),
        RequestUsage(input_tokens=9_800, output_tokens=5),
        RequestUsage(input_tokens=15_000, output_tokens=5),
        RequestUsage(input_tokens=8_200, output_tokens=5),
        RequestUsage(input_tokens=12_500, output_tokens=7),
    ]
    monkeypatch.setattr(
        agent_chat, "_make_model", lambda *a, **k: _usage_fn_model(_READ_SCRIPT, scripted)
    )

    result = await agent_chat.run_one_turn(
        state, None, "question", model_id="stub", endpoint="http://stub"
    )

    assert result.answer == "Final answer [1]."
    assert result.trace["requests"] == 5
    assert result.trace["peak_input_tokens"] == 15_000
    assert result.trace["overflow_fallbacks"] == 0
    log = result.trace["request_log"]
    assert [pair[1] for pair in log] == [4_100, 9_800, 15_000, 8_200, 12_500]
    # Char column: every request non-empty and growing while reads accumulate
    # (the re-prefill multiplier the window budget must reason about).
    chars = [pair[0] for pair in log]
    assert all(c > 0 for c in chars)
    assert chars == sorted(chars)
    # Consistency with the cumulative accounting: the peak is one request's
    # share of the cumulative total, never above it.
    assert result.input_tokens == sum(pair[1] for pair in log)
    assert result.trace["peak_input_tokens"] <= result.input_tokens


async def test_overflow_fallback_counted_and_fallback_metered(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A context-overflow 400 on the main run routes to the no-tool fallback:
    the recovery is counted, and the fallback's own request is metered (the
    failed 400 recorded nothing — it has no usage to report)."""
    runtime = FakeToolRuntime(
        [_scored(0, "relevant passage")],
        [_card(0, "relevant passage")],
        "article body",
    )
    _patch_runtime(monkeypatch, runtime)
    calls = {"n": 0}

    def _model(*a: Any, **k: Any) -> FunctionModel:
        def fn(messages: list[ModelMessage], info: Any) -> ModelResponse:
            calls["n"] += 1
            if calls["n"] == 1:
                raise ModelHTTPError(
                    status_code=400,
                    model_name="stub",
                    body={"message": "Context size has been exceeded."},
                )
            return ModelResponse(
                parts=[TextPart(content="fallback answer [1]")],
                usage=RequestUsage(input_tokens=2_500, output_tokens=9),
            )

        async def stream_fn(messages: list[ModelMessage], info: Any) -> AsyncIterator[Any]:
            yield "unused"  # pragma: no cover — run_one_turn never streams

        return FunctionModel(function=fn, stream_function=stream_fn)

    monkeypatch.setattr(agent_chat, "_make_model", _model)

    result = await agent_chat.run_one_turn(
        state, None, "question", model_id="stub", endpoint="http://stub"
    )

    assert result.answer == "fallback answer [1]"
    assert result.trace["overflow_fallbacks"] == 1
    assert result.trace["requests"] == 1
    assert result.trace["peak_input_tokens"] == 2_500
    assert len(result.trace["request_log"]) == 1
    assert result.trace["request_log"][0][1] == 2_500


async def test_unrelated_400_stays_loud_and_uncounted(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-overflow 400 propagates and is never counted as a fallback."""
    runtime = FakeToolRuntime([_scored(0, "p")], [_card(0, "p")], "body")
    _patch_runtime(monkeypatch, runtime)

    def _model(*a: Any, **k: Any) -> FunctionModel:
        def fn(messages: list[ModelMessage], info: Any) -> ModelResponse:
            raise ModelHTTPError(
                status_code=400, model_name="stub", body={"message": "bad request"}
            )

        async def stream_fn(messages: list[ModelMessage], info: Any) -> AsyncIterator[Any]:
            yield "unused"  # pragma: no cover

        return FunctionModel(function=fn, stream_function=stream_fn)

    monkeypatch.setattr(agent_chat, "_make_model", _model)
    with pytest.raises(ModelHTTPError):
        await agent_chat.run_one_turn(
            state, None, "question", model_id="stub", endpoint="http://stub"
        )


# ── Streaming twin: same fields on the TraceEvent ──────────────────────────


async def test_stream_trace_carries_meter_fields(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The streaming runner surfaces the same request-accounting fields: a
    one-tool-round stream is 2 requests, both recorded in the log."""
    runtime = FakeToolRuntime(
        [_scored(0, "relevant passage")],
        [_card(0, "relevant passage")],
        "article body text",
    )
    _patch_runtime(monkeypatch, runtime)

    def _model(*a: Any, **k: Any) -> FunctionModel:
        async def stream_fn(messages: list[ModelMessage], info: Any) -> AsyncIterator[Any]:
            has_return = any(
                isinstance(m, ModelRequest)
                and any(getattr(p, "part_kind", None) == "tool-return" for p in m.parts)
                for m in messages
            )
            if has_return:
                yield "The answer is 42 [1]."
            else:
                yield {0: DeltaToolCall(name="read_article", json_args='{"n": 1}')}

        def fn(messages: list[ModelMessage], info: Any) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content="unused [1].")])

        return FunctionModel(function=fn, stream_function=stream_fn)

    monkeypatch.setattr(agent_chat, "_make_model", _model)
    events = [ev async for ev in agent_chat.iter_agent_turn_events(state, None, "question")]
    traces = [e for e in events if isinstance(e, TraceEvent)]
    assert len(traces) == 1
    trace: dict[str, Any] = dict(traces[0].trace)
    assert trace["requests"] == 2
    assert trace["peak_input_tokens"] > 0
    assert trace["overflow_fallbacks"] == 0
    assert len(trace["request_log"]) == 2
    # Second request re-prefills the first plus the tool result.
    assert trace["request_log"][1][0] > trace["request_log"][0][0]


# ── _wire_chars: the char half of the calibration pair ─────────────────────


def test_wire_chars_counts_text_parts_only() -> None:
    """System/user text, tool-call args, and tool returns are counted; nothing
    else is (the template-overhead margin is the safety direction)."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart(content="sys " * 10)]),
        ModelRequest(parts=[UserPromptPart(content="user " * 20)]),
        ModelResponse(parts=[ToolCallPart(tool_name="read_article", args='{"n": 1}')]),
        ModelRequest(
            parts=[ToolReturnPart(tool_name="read_article", content="x" * 500, tool_call_id="t1")]
        ),
    ]
    assert agent_chat._wire_chars(messages) == 40 + 100 + len('{"n": 1}') + 500


def test_wire_chars_empty_is_zero() -> None:
    assert agent_chat._wire_chars([]) == 0
