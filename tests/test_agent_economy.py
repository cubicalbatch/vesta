"""Agent token-economy integration tests (economy → api/agent_chat).

In-process (no HTTP, no model server): the tool runtime is a fake returning
controllable passages/articles and the model is a ``FunctionModel`` stub via
the ``_make_model`` seam — same recipe as tests/test_agent_stream.py.

Covers:
* Round-0 pre-seed slicing + per-passage cap under the CPU economy budget;
* the ``read_article`` character cap selecting a score-aware window that
  contains the retrieval-scored passage (never a plain head cut);
* retry slimming: the abstention retry re-runs against the ORIGINAL history,
  dropping the failed attempt's transcript;
* the resolved budget recorded once per turn in the trace.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from vesta.answer.contracts import (
    AnswerResetEvent,
    CitationsEvent,
    TokenEvent,
    TraceEvent,
)
from vesta.answer.economy import CPU_ECONOMY_DEFAULTS
from vesta.answer.tools import SearchToolResult
from vesta.api import agent_chat
from vesta.config.settings import SettingsSnapshot, all_settings
from vesta.retrieval.contracts import ScoredPassage, SourceCard
from vesta.zim.types import Passage

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def state() -> Any:
    """A lightweight dummy state object — _build_tool_runtime is patched."""
    return SimpleNamespace(registry=None, db=None)


@pytest.fixture(autouse=True)
def _no_llm_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detach the lifespan-bound runtime: hardware is None, no warm-up."""
    monkeypatch.setattr("vesta.inference.get_runtime", lambda: None)


def make_snapshot(**overrides: Any) -> SettingsSnapshot:
    values: dict[str, object] = {s.key: s.default for s in all_settings().values()}
    values.update(overrides)
    return SettingsSnapshot(values=values)


#: Snapshot forcing the economy on — the iter3 control semantics (6 passages
#: / 6000-char outlier cap / 6000 read / 2048 output / 12k tool budget /
#: 6x400 search snippets / full prompt) regardless of the detached (None)
#: hardware.
ECONOMY_SN = make_snapshot(**{"answer.agent.economy": "on"})

#: The iter6 variant: economy on plus the opt-in compact prompt.
COMPACT_SN = make_snapshot(**{"answer.agent.economy": "on", "answer.agent.compact_prompt": True})


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
    """The tool-runtime surface _do_search/read_article dispatch to.

    Records what actually executed (for the tool-call dedup tests).
    """

    def __init__(
        self,
        passages: list[ScoredPassage],
        cards: list[SourceCard],
        article: str | dict[str, str],
    ):
        self._result = SearchToolResult(
            text="formatted", passages=tuple(passages), cards=tuple(cards)
        )
        self._article = article
        self.search_calls: list[str] = []
        self.search_exact_calls: list[str] = []
        self.read_calls: list[tuple[int, str]] = []
        self.must_includes: list[str] = []

    async def search(self, query: str, scope: str) -> SearchToolResult:
        self.search_calls.append(query)
        return self._result

    async def search_exact(self, query: str, scope: str) -> SearchToolResult:
        self.search_exact_calls.append(query)
        return self._result

    async def read_article(self, zim_id: int, path: str, *, must_include: str = "") -> str:
        self.read_calls.append((zim_id, path))
        self.must_includes.append(must_include)
        if isinstance(self._article, dict):
            return self._article[path]
        return self._article


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, runtime: FakeToolRuntime) -> None:
    monkeypatch.setattr(agent_chat, "_build_tool_runtime", lambda *a, **k: runtime)


# ── FunctionModel stub helpers (same recipe as test_agent_stream.py) ────────


def _dummy_request(messages: list[ModelMessage], info: Any) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content="stub fallback answer [1]")])


def _stub_model(
    seen: list[list[ModelMessage]] | None = None,
    text: str = "The Battle of Hastings was in 1066 [1].",
) -> FunctionModel:
    """Streams one direct answer; optionally records its request messages."""

    async def stream_fn(messages: list[ModelMessage], info: Any) -> AsyncIterator[Any]:
        if seen is not None:
            seen.append(list(messages))
        yield text

    return FunctionModel(function=_dummy_request, stream_function=stream_fn)


def _crash_then_fallback_model(fallback_text: str) -> FunctionModel:
    """First ``run`` raises UsageLimitExceeded; every later call answers directly."""
    calls = {"n": 0}

    def fn(messages: list[ModelMessage], info: Any) -> ModelResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            raise UsageLimitExceeded("request limit hit")
        return ModelResponse(parts=[TextPart(content=fallback_text)])

    async def stream_fn(messages: list[ModelMessage], info: Any) -> AsyncIterator[Any]:
        yield "unused"  # pragma: no cover — run_one_turn never streams

    return FunctionModel(function=fn, stream_function=stream_fn)


# ── Round-0 pre-seed: outlier-guard cap (passage set unchanged) ─────────────


async def test_preseed_guard_capped_under_economy(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    passages = [_scored(i, f"PASSAGE-{i} " + "x" * 7000) for i in range(8)]
    runtime = FakeToolRuntime(passages, [_card(i, "s") for i in range(8)], "article")
    _patch_runtime(monkeypatch, runtime)
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: _stub_model())

    ctx = await agent_chat._build_turn(state, ECONOMY_SN, "Einstein relativity")

    # Economy-on defaults reproduce the iter3 control exactly (compact off).
    assert ctx.budget == CPU_ECONOMY_DEFAULTS
    assert ctx.model_settings["max_tokens"] == CPU_ECONOMY_DEFAULTS.max_output_tokens
    # All 6 default passages survive (cutting the pre-seed starved gold facts
    # and pushed one-shot questions into thrashing tool loops in iteration 1);
    # only passages 7-8 are dropped.
    for i in range(6):
        assert f"PASSAGE-{i}" in ctx.seed_text
    for i in range(6, 8):
        assert f"PASSAGE-{i}" not in ctx.seed_text
    # The 6000-char cap is an outlier guard: 7k passages are elided to 6000.
    assert max(len(line) for line in ctx.seed_text.splitlines()) <= 6000
    assert "…" in ctx.seed_text


# ── read_article: capped to a score-aware window, never a head cut ──────────


async def test_read_article_window_contains_scored_passage_not_head_cut(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The fact sits ~9k chars in — beyond the 6000-char economy cap, so a
    # head cut would lose it. It is the card's snippet (the retrieval-scored
    # passage) and must survive via the focused-view must-include span.
    lead = "LEAD. " + "Generic filler about unrelated historical topics. " * 180
    needle = "The ultraviolet ANS1915 constant was measured precisely here."
    article = lead + " " + needle + " " + "Trailing filler. " * 300
    assert article.find(needle) > 6000 and len(article) > 6000

    runtime = FakeToolRuntime([_scored(0, needle)], [_card(0, needle)], article)
    _patch_runtime(monkeypatch, runtime)

    seen: list[list[ModelMessage]] = []

    def _model(*a: Any, **k: Any) -> FunctionModel:
        async def stream_fn(messages: list[ModelMessage], info: Any) -> AsyncIterator[Any]:
            seen.append(list(messages))
            has_return = any(
                getattr(m, "parts", None) is not None
                and any(getattr(p, "part_kind", None) == "tool-return" for p in m.parts)
                for m in messages
            )
            if has_return:
                yield "The constant was measured in 1915 [1]."
            else:
                yield {0: DeltaToolCall(name="read_article", json_args='{"n": 1}')}

        return FunctionModel(function=_dummy_request, stream_function=stream_fn)

    monkeypatch.setattr(agent_chat, "_make_model", _model)
    events = [
        ev
        async for ev in agent_chat.iter_agent_turn_events(state, ECONOMY_SN, "ultraviolet constant")
    ]

    assert not [e for e in events if type(e).__name__ == "ErrorEvent"]
    # Round 1 messages contain the read_article tool return: capped, and the
    # scored passage (the needle) survived — not a head cut.
    tool_text = str(seen[1])
    assert "ANS1915" in tool_text
    assert len(tool_text) < len(article)


async def test_read_article_threads_card_snippet_into_seam(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AUDIT_0824 N11: the read_article tool passes each card's retrieval
    snippet to the injected callable as ``must_include`` — that is the only
    channel through which the composition root's stage-1 focused window can
    guarantee the snippet survives its 32k elision."""
    needle = "The ANSN11 marker constant was recorded here."
    runtime = FakeToolRuntime([_scored(0, "p")], [_card(0, needle)], "article body")
    _patch_runtime(monkeypatch, runtime)

    def _model(*a: Any, **k: Any) -> FunctionModel:
        async def stream_fn(messages: list[ModelMessage], info: Any) -> AsyncIterator[Any]:
            has_return = any(
                getattr(m, "parts", None) is not None
                and any(getattr(p, "part_kind", None) == "tool-return" for p in m.parts)
                for m in messages
            )
            if has_return:
                yield "The constant is recorded [1]."
            else:
                yield {0: DeltaToolCall(name="read_article", json_args='{"n": 1}')}

        return FunctionModel(function=_dummy_request, stream_function=stream_fn)

    monkeypatch.setattr(agent_chat, "_make_model", _model)
    events = [
        ev
        async for ev in agent_chat.iter_agent_turn_events(state, ECONOMY_SN, "ultraviolet constant")
    ]

    assert not [e for e in events if type(e).__name__ == "ErrorEvent"]
    assert runtime.read_calls == [(1, "a/0")]
    assert runtime.must_includes == [needle]


async def test_stage1_focused_view_keeps_snippet_beyond_32k_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AUDIT_0824 N11: for an article longer than the 32k stage-1 cap whose
    best snippet lies beyond what question-term focusing naturally selects,
    threading the card snippet in via ``must_include`` keeps it in the excerpt
    (and without the thread the naive window really does drop it)."""
    from vesta.api.answer import _build_tool_runtime

    # Every chunk contains all question terms → identical IDF scores → the
    # greedy fill walks chunks in document order until the budget is spent,
    # so anything past ~32k chars is elided unless a must-include span forces
    # it. The needle carries none of the question terms and sits at ~45k.
    unit = "Common filler word about the record appears here again. " * 8
    lead = "LEAD section of the long record. "
    needle = "Zqxvv ANS2408 buried marker sentence with no overlap terms."
    article = lead + unit * 100 + needle + " " + unit * 20
    assert len(article) > 40_000
    assert article.find(needle) > 32_000

    class _FakeArchive:
        async def extract(self, path: str) -> Any:
            return SimpleNamespace(title="Long record", text=article)

    class _FakeRegistry:
        def get(self, zim_id: int) -> Any:
            return _FakeArchive()

    fake_state = SimpleNamespace(registry=_FakeRegistry(), db=None, gateway=None)
    monkeypatch.setattr("vesta.api.answer._build_reformulator", lambda *a, **k: None)
    rt = _build_tool_runtime(
        fake_state,
        ECONOMY_SN,
        None,
        None,
        "common filler word record",
    )
    assert rt is not None

    # Control: without the threaded snippet, the 32k focused window drops it.
    bare = await rt.read_article(1, "a/0")
    assert needle not in bare

    # With the card snippet threaded in, stage 1 forces the span in.
    threaded = await rt.read_article(1, "a/0", must_include=needle)
    assert needle in threaded


# ── Retry slimming ──────────────────────────────────────────────────────────


async def test_abstention_retry_gets_original_history_not_failed_transcript(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = FakeToolRuntime([_scored(0, "relevant passage")], [_card(0, "relevant passage")], "a")
    _patch_runtime(monkeypatch, runtime)

    run_calls: list[list[ModelMessage]] = []

    def _model(*a: Any, **k: Any) -> FunctionModel:
        async def stream_fn(messages: list[ModelMessage], info: Any) -> AsyncIterator[Any]:
            yield "I cannot find the answer in the provided sources."

        def fn(messages: list[ModelMessage], info: Any) -> ModelResponse:
            run_calls.append(list(messages))
            return ModelResponse(parts=[TextPart(content="The answer is 42 [1].")])

        return FunctionModel(function=fn, stream_function=stream_fn)

    monkeypatch.setattr(agent_chat, "_make_model", _model)
    events = [
        ev
        async for ev in agent_chat.iter_agent_turn_events(state, ECONOMY_SN, "what is the answer")
    ]

    # The retry ran exactly once, against the ORIGINAL (empty) history: no
    # assistant message from the failed attempt rides along.
    assert len(run_calls) == 1
    retry_messages = run_calls[0]
    assert not any(isinstance(m, ModelResponse) for m in retry_messages)
    assert not any("cannot find" in str(m) for m in retry_messages)
    # The retry prompt carries the pre-seed (dropping the transcript must not
    # drop the evidence) plus the directive.
    prompt = str(retry_messages)
    assert "Initial sources" in prompt
    assert "You did not give an answer" in prompt

    resets = [e for e in events if isinstance(e, AnswerResetEvent)]
    assert resets and resets[0].reason == "abstention_retry"

    # The resolved budget is recorded once per turn in the trace.
    trace = next(e for e in events if isinstance(e, TraceEvent)).trace
    assert trace["budget"] == {
        "preseed_passages": 6,
        "preseed_passage_max_chars": 6000,
        "read_max_chars": 6000,
        "max_output_tokens": 2048,
        "tool_budget_chars": 12000,
        "age_tool_chars": 0,
        "search_entries": 6,
        "search_snippet_chars": 400,
        "compact_prompt": False,
        # Window fields: economy-only turn (profile full) — zeros.
        "window_tokens": 0,
        "output_reserve": 0,
        "tool_budget_tokens": 0,
        "preseed_dropped": 0,
    }


async def test_run_one_turn_retry_drops_failed_transcript(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = FakeToolRuntime([_scored(0, "relevant passage")], [_card(0, "relevant passage")], "a")
    _patch_runtime(monkeypatch, runtime)

    run_calls: list[list[ModelMessage]] = []

    def _model(*a: Any, **k: Any) -> FunctionModel:
        def fn(messages: list[ModelMessage], info: Any) -> ModelResponse:
            run_calls.append(list(messages))
            if len(run_calls) == 1:  # first attempt: refuse
                return ModelResponse(parts=[TextPart(content="I cannot find it.")])
            return ModelResponse(parts=[TextPart(content="The answer is 42 [1].")])

        async def stream_fn(messages: list[ModelMessage], info: Any) -> AsyncIterator[Any]:
            yield "unused"

        return FunctionModel(function=fn, stream_function=stream_fn)

    monkeypatch.setattr(agent_chat, "_make_model", _model)
    result = await agent_chat.run_one_turn(
        state, ECONOMY_SN, "what is the answer", model_id="stub", endpoint="http://stub"
    )

    assert result.answer == "The answer is 42 [1]."
    assert len(run_calls) == 2
    # The retry's history excludes the failed attempt entirely (original
    # pre-turn history was empty → system + retry prompt only).
    assert not any(isinstance(m, ModelResponse) for m in run_calls[1])
    assert "Initial sources" in str(run_calls[1])

    # run_one_turn's trace carries the same budget/stages payload so the
    # benchmark path (bench trace_json) is observable too.
    assert result.trace["budget"] == {
        "preseed_passages": 6,
        "preseed_passage_max_chars": 6000,
        "read_max_chars": 6000,
        "max_output_tokens": 2048,
        "tool_budget_chars": 12000,
        "age_tool_chars": 0,
        "search_entries": 6,
        "search_snippet_chars": 400,
        "compact_prompt": False,
        # Window fields: economy-only turn (profile full) — zeros.
        "window_tokens": 0,
        "output_reserve": 0,
        "tool_budget_tokens": 0,
        "preseed_dropped": 0,
    }
    assert result.trace["stages"][0]["name"] == "pre_seed"


async def test_run_one_turn_forwards_scope_to_preseed(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: ``run_one_turn`` accepted ``scope`` but never
    forwarded it to ``_build_turn`` — every bench pre-seed searched ALL
    enabled archives (cross-archive kNN fanout + cold cluster reads = the
    measured 7-14 s ``pre_seed`` vs ~2 s scoped, plus foreign-archive cards
    in 6/50 run-90 questions) while the streaming path was correctly scoped.
    The scope string must reach ``_parse_scope`` and the parsed scope must
    reach ``_build_tool_runtime``."""
    seen_scopes: list[str | None] = []
    sentinel = object()
    seen_rt_args: list[tuple[Any, ...]] = []

    def _fake_parse(scope: str | None, registry: Any) -> object:
        seen_scopes.append(scope)
        return sentinel

    runtime = FakeToolRuntime([_scored(0, "relevant passage")], [_card(0, "relevant passage")], "a")

    def _fake_build_rt(*a: Any, **k: Any) -> FakeToolRuntime:
        seen_rt_args.append(a)
        return runtime

    monkeypatch.setattr(agent_chat, "_parse_scope", _fake_parse)
    monkeypatch.setattr(agent_chat, "_build_tool_runtime", _fake_build_rt)
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: _stub_model())

    result = await agent_chat.run_one_turn(
        state,
        ECONOMY_SN,
        "what is the answer",
        model_id="stub",
        endpoint="http://stub",
        scope="wikipedia_en_top_nopic_2026-06.zim",
    )
    assert seen_scopes == ["wikipedia_en_top_nopic_2026-06.zim"]
    # The PARSED scope reaches the tool-runtime wiring (3rd positional arg),
    # not just the parser.
    assert seen_rt_args[0][2] is sentinel
    assert result.trace["stages"][0]["name"] == "pre_seed"


# ── Tool-call dedup (greedy repeat loops) ───────────────────────────────────


def _repeat_then_answer_model(
    tool_calls: list[tuple[str, str]], seen: list[list[ModelMessage]]
) -> FunctionModel:
    """Emit the given tool calls one per model round, then a final answer."""

    async def stream_fn(messages: list[ModelMessage], info: Any) -> AsyncIterator[Any]:
        seen.append(list(messages))
        idx = sum(
            1
            for m in messages
            if getattr(m, "parts", None) is not None
            and any(getattr(p, "part_kind", None) == "tool-return" for p in m.parts)
        )
        if idx < len(tool_calls):
            name, json_args = tool_calls[idx]
            yield {0: DeltaToolCall(name=name, json_args=json_args)}
        else:
            yield "Final answer [1]."

    return FunctionModel(function=_dummy_request, stream_function=stream_fn)


async def test_identical_search_repeat_returns_steering_not_retrieval(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = FakeToolRuntime([_scored(0, "relevant passage")], [_card(0, "relevant passage")], "a")
    _patch_runtime(monkeypatch, runtime)
    seen: list[list[ModelMessage]] = []
    monkeypatch.setattr(
        agent_chat,
        "_make_model",
        lambda *a, **k: _repeat_then_answer_model(
            [("search", '{"query": "dup query"}'), ("search", '{"query": "dup query"}')], seen
        ),
    )

    events = [
        ev async for ev in agent_chat.iter_agent_turn_events(state, ECONOMY_SN, "some question")
    ]

    # Retrieval ran exactly once (plus the Round-0 pre-seed via search_exact);
    # the identical repeat hit the dedup guard instead.
    assert runtime.search_calls == ["dup query"]
    assert len(runtime.search_exact_calls) == 1
    # The repeat's tool result is the small steering string, in the transcript.
    assert "You already searched for 'dup query'" in str(seen[2])
    assert not [e for e in events if type(e).__name__ == "ErrorEvent"]


async def test_identical_read_repeat_returns_steering_not_article(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = FakeToolRuntime(
        [_scored(0, "relevant passage")], [_card(0, "relevant passage")], "a" * 100
    )
    _patch_runtime(monkeypatch, runtime)
    seen: list[list[ModelMessage]] = []
    monkeypatch.setattr(
        agent_chat,
        "_make_model",
        lambda *a, **k: _repeat_then_answer_model(
            [("read_article", '{"n": 1}'), ("read_article", '{"n": 1}')], seen
        ),
    )

    events = [
        ev async for ev in agent_chat.iter_agent_turn_events(state, ECONOMY_SN, "some question")
    ]

    # The article was extracted exactly once; the repeat was steered away.
    assert runtime.read_calls == [(1, "a/0")]
    assert "You already read source [1]" in str(seen[2])
    assert not [e for e in events if type(e).__name__ == "ErrorEvent"]


# ── max_tokens reaches the wire ─────────────────────────────────────────────


async def test_make_model_sends_max_tokens_in_request_body() -> None:
    """The output cap must land in the request: pydantic-ai 2.x maps the
    ``max_tokens`` ModelSetting to OpenAI's ``max_completion_tokens`` field by
    default, which llama-server/LM Studio/vLLM ignore — the cap silently
    vanished from the wire (captured live: ``max_tokens: null``).
    ``_make_model`` pins the profile so the request carries plain
    ``max_tokens``. Loopback-only (binds an ephemeral 127.0.0.1 port)."""
    from pydantic_ai import Agent
    from pydantic_ai.settings import ModelSettings

    captured: dict[str, Any] = {}

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        headers = b""
        while not headers.endswith(b"\r\n\r\n"):
            headers += await reader.read(1)
        length = 0
        for line in headers.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                length = int(line.split(b":")[1])
        captured["body"] = json.loads(await reader.read(length) or b"{}")
        resp = json.dumps(
            {
                "id": "1",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
            + str(len(resp)).encode()
            + b"\r\nConnection: close\r\n\r\n"
            + resp
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    model = agent_chat._make_model("qwen3.5-4b", f"http://127.0.0.1:{port}/v1", "")
    try:
        settings: ModelSettings = {"max_tokens": 2048, "temperature": 0}
        result = await Agent(model, system_prompt="s", model_settings=settings).run("hi")
        assert result.output == "ok"
    finally:
        server.close()
        await server.wait_closed()
        await model.client.close()

    assert captured["body"]["max_tokens"] == 2048
    assert captured["body"].get("max_completion_tokens") is None


# ── Turn-level tool-insert budget ───────────────────────────────────────────


def _two_source_runtime(articles: dict[str, str]) -> FakeToolRuntime:
    """A runtime with two cards (n=1 → a/0, n=2 → a/1) and per-path articles."""
    return FakeToolRuntime(
        [_scored(0, "first"), _scored(1, "second")],
        [_card(0, "first"), _card(1, "second")],
        articles,
    )


async def test_tool_budget_second_read_returns_steering(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two 7k reads against a 12k tool budget: the first inserts, the second
    would overflow and is rejected whole (steering, no partial read), and the
    turn latches so a third read steers BEFORE executing retrieval."""
    sn = make_snapshot(
        **{"answer.agent.economy": "on", "answer.agent.read_max_chars": 0}  # full 7k reads
    )
    articles = {
        "a/0": "ARTICLE-ONE " + "x" * 7000,
        "a/1": "ARTICLE-TWO " + "y" * 7000,
    }
    runtime = _two_source_runtime(articles)
    _patch_runtime(monkeypatch, runtime)
    seen: list[list[ModelMessage]] = []
    monkeypatch.setattr(
        agent_chat,
        "_make_model",
        lambda *a, **k: _repeat_then_answer_model(
            [
                ("read_article", '{"n": 1}'),
                ("read_article", '{"n": 2}'),
                ("read_article", '{"n": 2}'),
            ],
            seen,
        ),
    )

    events = [ev async for ev in agent_chat.iter_agent_turn_events(state, sn, "question")]

    # First read inserted; second executed retrieval but was rejected whole;
    # third steered BEFORE executing (still only 2 retrievals total).
    assert runtime.read_calls == [(1, "a/0"), (1, "a/1")]
    # The transcript holds the first article and the steering strings, never
    # the second article: cumulative inserts stay bounded.
    transcript = str(seen[-1])
    assert "ARTICLE-ONE" in transcript
    assert "ARTICLE-TWO" not in transcript
    assert "budget reached" in str(seen[2])
    assert "budget reached" in str(seen[3])
    assert not [e for e in events if type(e).__name__ == "ErrorEvent"]


# ── Request-side context aging ──────────────────────────────────────────────


def _aged_articles() -> dict[str, str]:
    """Four ~900-char articles whose tail marker sits beyond the 400-char
    aging cap — TAIL-i present in a request means that round stayed full."""
    return {f"a/{i}": f"HEAD-{i} " + chr(ord("a") + i) * 900 + f" TAIL-{i}" for i in range(4)}


def _aged_runtime() -> FakeToolRuntime:
    return FakeToolRuntime(
        [_scored(i, f"snippet {i}") for i in range(4)],
        [_card(i, f"snippet {i}") for i in range(4)],
        _aged_articles(),
    )


def _tool_script_fn_model(
    tool_calls: list[tuple[str, dict[str, Any]]], seen: list[list[ModelMessage]]
) -> FunctionModel:
    """Non-streaming model that emits one scripted tool call per round (the
    ``agent.run`` path run_one_turn drives), then a final text answer."""

    def fn(messages: list[ModelMessage], info: Any) -> ModelResponse:
        seen.append(list(messages))
        idx = sum(
            1
            for m in messages
            if isinstance(m, ModelRequest)
            and any(getattr(p, "part_kind", None) == "tool-return" for p in m.parts)
        )
        if idx < len(tool_calls):
            name, args = tool_calls[idx]
            return ModelResponse(parts=[ToolCallPart(tool_name=name, args=args)])
        return ModelResponse(parts=[TextPart(content="Final answer [1].")])

    async def stream_fn(messages: list[ModelMessage], info: Any) -> AsyncIterator[Any]:
        yield "unused"  # pragma: no cover — run_one_turn never streams

    return FunctionModel(function=fn, stream_function=stream_fn)


_READ_SCRIPT = [("read_article", {"n": i + 1}) for i in range(4)]


async def test_context_aging_truncates_old_rounds_only(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 4-read conversation: requests after two more rounds see the older
    results truncated to age_tool_chars (+ stub note); the last two rounds
    always stay full."""
    runtime = _aged_runtime()
    _patch_runtime(monkeypatch, runtime)
    seen: list[list[ModelMessage]] = []
    monkeypatch.setattr(
        agent_chat, "_make_model", lambda *a, **k: _tool_script_fn_model(_READ_SCRIPT, seen)
    )

    # Aging is opt-in since iteration 4 (bench net loss at the CPU default):
    # economy on alone no longer activates it.
    aging_sn = make_snapshot(**{"answer.agent.economy": "on", "answer.agent.age_tool_chars": 400})
    result = await agent_chat.run_one_turn(
        state, aging_sn, "question", model_id="stub", endpoint="http://stub"
    )

    assert result.answer == "Final answer [1]."
    # Request 3 (after reads 1+2): both rounds are the last two → full.
    assert "TAIL-0" in str(seen[2]) and "TAIL-1" in str(seen[2])
    # Request 4 (after read 3): round 1 aged, rounds 2-3 full.
    assert "TAIL-0" not in str(seen[3])
    assert "TAIL-1" in str(seen[3]) and "TAIL-2" in str(seen[3])
    # Request 5 (final): read 4 hit _MAX_READ_CALLS (steering string, ≤400
    # chars → untouched), so rounds are [art0, art1, art2, steering]: rounds
    # 1-2 aged, rounds 3-4 full.
    final_req = str(seen[4])
    assert "TAIL-0" not in final_req and "TAIL-1" not in final_req
    assert "TAIL-2" in final_req and "TAIL-3" not in final_req
    assert "truncated for context economy" in final_req


# ── Iteration 5: compact system prompt + search snippet shaping ────────────


async def test_compact_prompt_used_when_knob_set(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Economy on + answer.agent.compact_prompt=true (the iter6 variant): the
    model sees the compact system prompt, NOT the full prompt's Napoleon
    example."""
    runtime = FakeToolRuntime([_scored(0, "relevant")], [_card(0, "relevant")], "a")
    _patch_runtime(monkeypatch, runtime)
    seen: list[list[ModelMessage]] = []
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: _stub_model(seen=seen))

    events = [ev async for ev in agent_chat.iter_agent_turn_events(state, COMPACT_SN, "q")]

    first_prompt = str(seen[0])
    assert "Up to 3 searches and 3 reads" in first_prompt
    # Iteration-6 one-shot-preservation directive + citation-format example.
    assert "answer immediately — no tool call is needed" in first_prompt
    assert (
        "Einstein, born 1879, published special relativity in 1905 at about 26 [1]." in first_prompt
    )
    assert "Napoleon" not in first_prompt  # worked example cut
    assert "Good queries" not in first_prompt
    assert not [e for e in events if type(e).__name__ == "ErrorEvent"]


async def test_search_snippet_shaping_6x400_default_5x350_opt_in(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compact search results: the CPU economy default reverted to the
    round-stable 6 x 400 (iter3 control); 5 x 350 remains available by user
    override (the iter6 shape)."""
    passages = [_scored(i, f"PASSAGE-{i} " + "x" * 500) for i in range(8)]

    # ── economy on, defaults (== economy off shape) ──
    runtime = FakeToolRuntime(passages, [_card(i, "s") for i in range(8)], "a")
    _patch_runtime(monkeypatch, runtime)
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: _stub_model())
    ctx = await agent_chat._build_turn(state, ECONOMY_SN, "Einstein")
    text = await ctx._do_search("query", compact=True)

    assert "(showing top 6):" in text
    assert "PASSAGE-5" in text and "PASSAGE-6" not in text
    snippet_lines = [ln for ln in text.splitlines() if ln.startswith("    PASSAGE-")]
    assert len(snippet_lines) == 6
    longest = max(len(ln) for ln in snippet_lines)
    assert 4 + 400 <= longest <= 4 + 400 + 1

    # ── user override to the iter6 5 x 350 shape ──
    runtime = FakeToolRuntime(passages, [_card(i, "s") for i in range(8)], "a")
    _patch_runtime(monkeypatch, runtime)
    sn = make_snapshot(
        **{
            "answer.agent.economy": "on",
            "answer.agent.search_entries": 5,
            "answer.agent.search_snippet_chars": 350,
        }
    )
    ctx = await agent_chat._build_turn(state, sn, "Einstein")
    text = await ctx._do_search("query", compact=True)

    assert "(showing top 5):" in text
    assert "PASSAGE-4" in text and "PASSAGE-5" not in text
    snippet_lines = [ln for ln in text.splitlines() if ln.startswith("    PASSAGE-")]
    assert len(snippet_lines) == 5
    assert max(len(ln) for ln in snippet_lines) <= 4 + 350 + 1  # indent + snippet + ellipsis

    # ── economy off ──
    runtime = FakeToolRuntime(passages, [_card(i, "s") for i in range(8)], "a")
    _patch_runtime(monkeypatch, runtime)
    sn = make_snapshot(**{"answer.agent.economy": "off"})
    ctx = await agent_chat._build_turn(state, sn, "Einstein")
    text = await ctx._do_search("query", compact=True)

    assert "(showing top 6):" in text
    assert "PASSAGE-5" in text and "PASSAGE-6" not in text
    snippet_lines = [ln for ln in text.splitlines() if ln.startswith("    PASSAGE-")]
    assert len(snippet_lines) == 6
    longest = max(len(ln) for ln in snippet_lines)
    assert 4 + 400 <= longest <= 4 + 400 + 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "The question asks what happened. The battle happened in 1066 [1].",
            "The battle happened in 1066 [1].",
        ),
        (
            "Based on the provided sources. The battle happened in 1066 [1].",
            "The battle happened in 1066 [1].",
        ),
        (
            "According to the sources. The battle happened in 1066 [1].",
            "The battle happened in 1066 [1].",
        ),
        (
            "The battle happened in 1066 (archive-3) [1].",
            "The battle happened in 1066 [1].",
        ),
        (
            'The battle happened in 1066 [1].\n[2]\n[1] "Battle of Hastings"\n[2] "England"',
            "The battle happened in 1066 [1].",
        ),
    ],
)
def test_answer_cleanup_features(raw: str, expected: str) -> None:
    assert agent_chat._cleanup_answer(raw) == expected


def test_answer_cleanup_is_idempotent_and_preserves_inline_content() -> None:
    raw = (
        'The answer is based on the record "Based on the evidence" [1] '
        'and mentions [2] "Title" inline. (archive-4)'
    )
    cleaned = agent_chat._cleanup_answer(raw)
    assert cleaned == (
        'The answer is based on the record "Based on the evidence" [1] '
        'and mentions [2] "Title" inline.'
    )
    assert agent_chat._cleanup_answer(cleaned) == cleaned

    # A preface with no substantive remainder is a fact-bearing answer, not a
    # removable preface.
    preface_only = "Based on the provided sources, the answer is 42."
    assert agent_chat._cleanup_answer(preface_only) == preface_only

    factual_based_on = "Based on the provided sources, the rate is 100 [1]."
    assert agent_chat._cleanup_answer(factual_based_on) == factual_based_on


@pytest.mark.asyncio
async def test_run_one_turn_answer_cleanup_is_opt_in(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = FakeToolRuntime(
        [_scored(0, "relevant passage")],
        [_card(0, "relevant passage")],
        "a",
    )
    _patch_runtime(monkeypatch, runtime)
    raw = 'Based on the provided sources. The answer is 42 [1].\n[1] "Article"'

    def _model(*a: Any, **k: Any) -> FunctionModel:
        def fn(messages: list[ModelMessage], info: Any) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content=raw)])

        async def stream_fn(messages: list[ModelMessage], info: Any) -> AsyncIterator[Any]:
            yield raw

        return FunctionModel(function=fn, stream_function=stream_fn)

    monkeypatch.setattr(agent_chat, "_make_model", _model)
    sn = make_snapshot(**{"answer.agent.answer_cleanup": True})
    result = await agent_chat.run_one_turn(
        state,
        sn,
        "what is the answer",
        model_id="stub",
        endpoint="http://stub",
    )
    assert result.answer == "The answer is 42 [1]."


@pytest.mark.asyncio
async def test_run_one_turn_recovery_answer_survives_cleanup_off(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: with ``answer.agent.answer_cleanup=false``, a crashed main
    run must still return the no-tool fallback answer. The shared recovery core
    writes ``st.answer``; the driver must mirror it unconditionally (the
    streaming driver always rebinds) — not return the pre-recovery binding,
    which is empty after UsageLimitExceeded/overflow."""
    runtime = FakeToolRuntime(
        [_scored(0, "relevant passage")],
        [_card(0, "relevant passage")],
        "a",
    )
    _patch_runtime(monkeypatch, runtime)
    fallback = 'The fallback answer is 42 [1].\n[1] "Article"'
    monkeypatch.setattr(
        agent_chat, "_make_model", lambda *a, **k: _crash_then_fallback_model(fallback)
    )
    sn = make_snapshot(**{"answer.agent.answer_cleanup": False})
    result = await agent_chat.run_one_turn(
        state,
        sn,
        "what is the answer",
        model_id="stub",
        endpoint="http://stub",
    )
    assert result.answer == fallback


@pytest.mark.asyncio
async def test_run_one_turn_recovery_answer_cleaned_when_opt_in(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With cleanup enabled the recovered answer still passes through cleanup —
    unchanged behaviour on top of the unconditional rebind."""
    runtime = FakeToolRuntime(
        [_scored(0, "relevant passage")],
        [_card(0, "relevant passage")],
        "a",
    )
    _patch_runtime(monkeypatch, runtime)
    fallback = 'Based on the provided sources. The fallback answer is 42 [1].\n[1] "Article"'
    monkeypatch.setattr(
        agent_chat, "_make_model", lambda *a, **k: _crash_then_fallback_model(fallback)
    )
    sn = make_snapshot(**{"answer.agent.answer_cleanup": True})
    result = await agent_chat.run_one_turn(
        state,
        sn,
        "what is the answer",
        model_id="stub",
        endpoint="http://stub",
    )
    assert result.answer == "The fallback answer is 42 [1]."


@pytest.mark.asyncio
async def test_stream_answer_cleanup_resets_and_citations_use_cleaned_text(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = 'Based on the provided sources. The answer is 42 [1].\n[1] "Article"'
    monkeypatch.setattr(
        agent_chat,
        "_make_model",
        lambda *a, **k: _stub_model(text=raw),
    )
    sn = make_snapshot(**{"answer.agent.answer_cleanup": True})
    events = [ev async for ev in agent_chat.iter_agent_turn_events(state, sn, "Einstein")]

    cleanup_resets = [
        e for e in events if isinstance(e, AnswerResetEvent) and e.reason == "cleanup"
    ]
    assert len(cleanup_resets) == 1
    reset_index = events.index(cleanup_resets[0])
    assert events[reset_index + 1] == TokenEvent(text="The answer is 42 [1].")
    citations = [e for e in events if isinstance(e, CitationsEvent)]
    assert len(citations) == 1
    assert citations[0].answer_text == "The answer is 42 [1]."
    trace = next(e for e in events if isinstance(e, TraceEvent)).trace
    llm_step = next(step for step in trace["stages"] if step["name"] == "agent_llm")
    assert llm_step["outputs"]["answer_chars"] == len(raw)


@pytest.mark.asyncio
async def test_stream_answer_cleanup_disabled_is_byte_identical(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = 'Based on the provided sources. The answer is 42 [1].\n[1] "Article"'
    monkeypatch.setattr(
        agent_chat,
        "_make_model",
        lambda *a, **k: _stub_model(text=raw),
    )
    events = [
        ev
        async for ev in agent_chat.iter_agent_turn_events(
            state, make_snapshot(**{"answer.agent.answer_cleanup": False}), "Einstein"
        )
    ]

    assert not any(isinstance(e, AnswerResetEvent) for e in events)
    assert "".join(e.text for e in events if isinstance(e, TokenEvent)) == raw
