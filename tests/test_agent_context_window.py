"""Window-aware budget resolution (tests).

Covers the D1-D6 contract:
* ``resolve_budget``'s three-rung ladder with the context profile — forced
  plans activate window budgeting on ANY hardware (remote included), ``auto``
  maps the live local window onto a plan (budgeting against the REAL window),
  and ``full``/``auto``+remote is today's behaviour exactly;
* user-set knobs winning over plan-derived values (read_max_chars, the
  pre-seed cap, the output reserve, the token tool budget, max_output);
* D4: the pre-flight pre-seed fit — whole tail passages dropped until turn 1
  fits ``window - output_reserve`` by construction, cards stay registered;
* D5: the window ledger blocks exactly the insert that WOULD overflow (not
  the one after), latches, and steers the model to answer;
* the derived ``tool_budget_tokens`` (what the arithmetic leaves) recorded in
  the trace's ``budget``;
* the ``--context-profile`` CLI override (the ``--economy`` template);
* ``full`` byte-identity against the locked pre-change reference fixture
  (``tests/fixtures/phase21_full_reference.json``, tool_call_id masked).

In-process throughout (no HTTP, no model server): the tool runtime is a
fake and the model is a ``FunctionModel`` stub at the ``_make_model`` seam —
the tests/test_agent_economy.py recipe.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from vesta.answer.contracts import AnswerResetEvent, CitationsEvent, TokenEvent, TraceEvent
from vesta.answer.economy import CONTEXT_PLANS, EconomyBudget, resolve_budget
from vesta.answer.tokens import estimate_tokens, estimate_tokens_for_chars
from vesta.answer.tools import SearchToolResult
from vesta.api import agent_chat
from vesta.config.settings import SettingsSnapshot, all_settings
from vesta.retrieval.contracts import ScoredPassage, SourceCard
from vesta.zim.types import Passage

# ── Shared fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def state() -> Any:
    """A lightweight dummy state object — _build_tool_runtime is patched."""
    return SimpleNamespace(registry=None, db=None)


@pytest.fixture(autouse=True)
def _no_llm_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detach the lifespan-bound runtime: hardware None, window None (remote)."""
    monkeypatch.setattr("vesta.inference.get_runtime", lambda: None)


def make_snapshot(**overrides: Any) -> SettingsSnapshot:
    values: dict[str, object] = {s.key: s.default for s in all_settings().values()}
    values.update(overrides)
    return SettingsSnapshot(values=values)


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
        self.read_calls: list[tuple[int, str]] = []
        self.must_includes: list[str] = []

    async def search(self, query: str, scope: str) -> SearchToolResult:
        return self._result

    async def search_exact(self, query: str, scope: str) -> SearchToolResult:
        return self._result

    async def read_article(self, zim_id: int, path: str, *, must_include: str = "") -> str:
        self.read_calls.append((zim_id, path))
        self.must_includes.append(must_include)
        if isinstance(self._article, dict):
            return self._article[path]
        return self._article


class CapturingModel:
    """Deterministic non-streaming model: one scripted tool call per round,
    then a fixed answer. Records every request's messages verbatim."""

    def __init__(self, tool_calls: list[tuple[str, str]], answer: str):
        self.seen: list[list[ModelMessage]] = []
        calls = list(tool_calls)
        outer = self

        async def fn(messages: list[ModelMessage], info: Any) -> ModelResponse:
            outer.seen.append(list(messages))
            if calls:
                name, args = calls.pop(0)
                return ModelResponse(parts=[ToolCallPart(tool_name=name, args=args)])
            return ModelResponse(parts=[TextPart(content=answer)])

        self._model = FunctionModel(function=fn)

    @property
    def model(self) -> FunctionModel:
        return self._model


async def _run(
    state: Any,
    monkeypatch: pytest.MonkeyPatch,
    sn: SettingsSnapshot,
    model: CapturingModel,
    runtime: FakeToolRuntime,
    *,
    question: str = "Who fought at Hastings?",
) -> Any:
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: model.model)
    monkeypatch.setattr(agent_chat, "_build_tool_runtime", lambda *a, **k: runtime)
    return await agent_chat.run_one_turn(
        state, sn, question, model_id="fake-model", endpoint="http://fake", api_key="k"
    )


# ── D1/D2/D3: the resolve ladder ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("settings", "hw", "window", "expected_attrs"),
    [
        (
            {"answer.agent.context_profile": "8k"},
            None,
            None,
            {
                "window_tokens": 8_192,
                "output_reserve": 1_280,
                "preseed_passages": 6,
                "preseed_passage_max_chars": 2_400,
                "read_max_chars": 4_500,
                "compact_prompt": False,
                "tool_budget_chars": 0,
                "search_entries": 6,
                "search_snippet_chars": 400,
                "max_output_tokens": 1_280,
            },
        ),
        (
            {"answer.agent.context_profile": "8k-fullprompt"},
            None,
            None,
            {
                "window_tokens": 8_192,
                "output_reserve": 1_280,
                "preseed_passages": 6,
                "preseed_passage_max_chars": 1_800,
                "read_max_chars": 4_500,
                "max_output_tokens": 1_280,
                "compact_prompt": False,
            },
        ),
        (
            {"answer.agent.context_profile": "8k-fullprompt-wide"},
            None,
            None,
            {
                "window_tokens": 8_192,
                "output_reserve": 1_280,
                "preseed_passages": 6,
                "preseed_passage_max_chars": 2_400,
                "read_max_chars": 4_500,
                "max_output_tokens": 1_280,
                "compact_prompt": False,
            },
        ),
        (
            {
                "answer.agent.context_profile": "8k",
                "answer.agent.compact_prompt": True,
                "answer.agent.preseed_passage_max_chars": 1_800,
            },
            None,
            None,
            {
                "window_tokens": 8_192,
                "compact_prompt": True,
                "preseed_passage_max_chars": 1_800,
                "read_max_chars": 4_500,
            },
        ),
        (
            {"answer.agent.context_profile": "16k"},
            None,
            None,
            {
                "window_tokens": 16_384,
                "output_reserve": 1_792,
                "preseed_passage_max_chars": 2_400,
                "read_max_chars": 8_000,
                "compact_prompt": False,
                "max_output_tokens": 1_792,
            },
        ),
        (
            {},
            None,
            None,
            {
                "window_tokens": 0,
                "output_reserve": 0,
                "read_max_chars": 0,
                "preseed_passage_max_chars": 0,
                "tool_budget_chars": 0,
                "compact_prompt": False,
                "max_output_tokens": 4_096,
            },
        ),
        (
            {"answer.agent.context_profile": "full"},
            "cpu",
            8_192,
            {
                "window_tokens": 0,
                "read_max_chars": 6_000,
            },
        ),
        (
            {"answer.agent.context_profile": "8k", "answer.agent.read_max_chars": 2_000},
            None,
            None,
            {
                "read_max_chars": 2_000,
                "window_tokens": 8_192,
            },
        ),
        (
            {"answer.agent.context_profile": "8k", "answer.agent.preseed_passage_max_chars": 3_000},
            None,
            None,
            {
                "preseed_passage_max_chars": 3_000,
            },
        ),
        (
            {"answer.agent.context_profile": "8k", "answer.agent.output_reserve_tokens": 2_048},
            None,
            None,
            {
                "output_reserve": 2_048,
                "max_output_tokens": 1_280,
            },
        ),
        (
            {"answer.agent.context_profile": "8k", "answer.agent.tool_budget_tokens": 500},
            None,
            None,
            {
                "tool_budget_tokens": 500,
            },
        ),
        (
            {"answer.agent.context_profile": "8k", "answer.agent.max_output_tokens": 2_048},
            None,
            None,
            {
                "max_output_tokens": 2_048,
            },
        ),
        (
            {"answer.agent.context_profile": "16k", "answer.agent.compact_prompt": True},
            None,
            None,
            {
                "compact_prompt": True,
            },
        ),
        (
            {"answer.agent.economy": "on", "answer.agent.context_profile": "8k"},
            None,
            None,
            {
                "read_max_chars": 4_500,
                "tool_budget_chars": 12_000,
            },
        ),
    ],
)
def test_resolve_budget_knob_overrides(
    settings: dict[str, Any], hw: str | None, window: int | None, expected_attrs: dict[str, Any]
) -> None:
    b = resolve_budget(make_snapshot(**settings), hw, window)
    for attr, expected in expected_attrs.items():
        assert getattr(b, attr) == expected, f"{attr}: expected {expected}, got {getattr(b, attr)}"


def test_8k_plan_identical_to_8k_fullprompt_wide() -> None:
    """Drift guard: '8k-fullprompt-wide' is VALUES-IDENTICAL to the
    (redefined) '8k' — kept only as the force-name under which run 64 ran.
    If either plan ever moves without the other, this fails so the
    equivalence stated in the settings/CLI help cannot silently rot."""
    assert CONTEXT_PLANS["8k"] == CONTEXT_PLANS["8k-fullprompt-wide"]
    b_8k = resolve_budget(make_snapshot(**{"answer.agent.context_profile": "8k"}), None, None)
    b_wide = resolve_budget(
        make_snapshot(**{"answer.agent.context_profile": "8k-fullprompt-wide"}), None, None
    )
    assert b_8k == b_wide


def test_8k_fullprompt_wide_ledger_arithmetic() -> None:
    """The DEFAULT 8k plan's trade (run 64's, stated in estimator tokens):
    the 2400-char cap spends the window on pre-seed chars
    (6x2400 @3.0 c/t = 4.8k est) so the derived tool ledger collapses to
    well under one full read — the D5 latch is EXPECTED to bite tool-round
    questions hard; that is the trade, not a bug — while turn 1 still fits
    window - reserve with all 6 passages (D4 drops nothing: 4.8k << the
    ~5.9k evidence budget)."""
    plan = CONTEXT_PLANS["8k-fullprompt-wide"]
    seed_est = estimate_tokens_for_chars(plan.preseed_passages * plan.preseed_passage_max_chars)
    assert seed_est == 4_800  # 6 x 2400 chars at the 3.0 chars/token floor
    # The real prompt-side base: full system prompt + the pre-seed-hit
    # directive + the user-message framing + a representative question.
    prompt_est = estimate_tokens(
        agent_chat.SYSTEM_PROMPT
        + agent_chat._STRONG_EVIDENCE_DIRECTIVE
        + agent_chat._USER_MESSAGE_HEAD
        + agent_chat._USER_MESSAGE_TAIL
        + "x" * 150
    )
    ledger = plan.window_tokens - plan.output_reserve - seed_est - prompt_est
    # Thin but positive: every full read (read_max_chars/3 = 1500 est) blows
    # it, so tool rounds must run on partial reads / the latch's refusal.
    assert 500 <= ledger < 1_500
    # D4: turn 1 fits by construction — the seed, prompt and question all
    # sit inside window - output_reserve (the ~6.1k pre-seed-hit request).
    assert prompt_est + seed_est <= plan.window_tokens - plan.output_reserve


@pytest.mark.parametrize(
    ("window", "expect_window", "expect_cap"),
    [
        (8_192, 8_192, 2_400),  # exactly 8k → the 8k plan (fpw semantics)
        (4_096, 4_096, 2_400),  # smaller box: the 8k KNOBS, the REAL window
        (16_384, 16_384, 2_400),
        (12_288, 12_288, 2_400),  # between: 16k knobs, real window
        (32_768, 0, 6_000),  # above 16k → full (CPU economy active on cpu hw)
    ],
)
def test_auto_local_maps_window_onto_plan(window: int, expect_window: int, expect_cap: int) -> None:
    """auto on a local runtime budgets against the REAL window, never the
    plan's name — a 4096 box must not plan for 8192."""
    b = resolve_budget(make_snapshot(), "cpu", window)
    assert b.window_tokens == expect_window
    assert b.preseed_passage_max_chars == expect_cap
    if expect_window:
        assert b.output_reserve > 0
    else:
        assert b.output_reserve == 0


def test_forced_full_stays_full_even_local() -> None:
    b = resolve_budget(make_snapshot(**{"answer.agent.context_profile": "full"}), "cpu", 8_192)
    assert b.window_tokens == 0
    assert b.read_max_chars == 6_000  # the plain CPU-economy ladder, untouched


# ── D4: the pre-flight pre-seed fit ─────────────────────────────────────────


@pytest.mark.parametrize("profile", ["8k", "8k-fullprompt", "8k-fullprompt-wide", "16k"])
async def test_turn1_fits_window_at_every_profile(
    state: Any, monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    """Turn 1 (the pre-seeded request) fits window - output_reserve by
    construction, at both plans, with fat passages and several questions."""
    for question in ("What is the answer?", "Who wrote the chronicle of Umbra?", "x" * 400):
        runtime = FakeToolRuntime(
            [_scored(i, f"PASSAGE-{i} " + "filler text. " * 400) for i in range(6)],
            [_card(i, "snippet") for i in range(6)],
            "article body",
        )
        model = CapturingModel([], "The answer is forty-two [1].")
        result = await _run(
            state,
            monkeypatch,
            make_snapshot(**{"answer.agent.context_profile": profile}),
            model,
            runtime,
            question=question,
        )
        w = CONTEXT_PLANS[profile].window_tokens
        r = CONTEXT_PLANS[profile].output_reserve
        assert result.trace["budget"]["window_tokens"] == w
        first_wire = agent_chat._wire_chars(model.seen[0])
        assert estimate_tokens_for_chars(first_wire) <= w - r
        # The 1800/2400 per-passage caps keep the top-6 seed inside the fit
        # without dropping anything (the floor is 6 passages).
        assert result.trace["budget"]["preseed_dropped"] == 0


async def test_preseed_fit_drops_whole_tail_passages(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An uncap-able pre-seed (user-set cap 0 wins over the plan) overflows the
    8k fit → whole TAIL passages are dropped until it fits; the dropped
    passages' cards stay registered with stable numbering (D4)."""
    passages = [_scored(i, f"PASSAGE-{i} " + "dense filler. " * 380) for i in range(6)]
    runtime = FakeToolRuntime(passages, [_card(i, "s") for i in range(6)], "article")
    model = CapturingModel([], "The answer is forty-two [1].")
    result = await _run(
        state,
        monkeypatch,
        make_snapshot(
            **{
                "answer.agent.context_profile": "8k",
                "answer.agent.preseed_passage_max_chars": 6_000,
            }
        ),
        model,
        runtime,
        question="What is the answer?",
    )
    b = result.trace["budget"]
    assert b["preseed_dropped"] >= 1
    # Dropped, never truncated mid-passage: the surviving seed keeps whole
    # passages and still fits the window arithmetic.
    assert estimate_tokens_for_chars(agent_chat._wire_chars(model.seen[0])) <= (
        b["window_tokens"] - b["output_reserve"]
    )
    # All six cards survive with their original numbering — the agent can
    # still read_article what the fit dropped, citations never shift.
    assert [c.n for c in result.cards] == [1, 2, 3, 4, 5, 6]
    seed = result.tool_calls[0].result_preview
    assert "[1]" in seed  # head passages kept


# ── D5: the window ledger ───────────────────────────────────────────────────


def _ctx_with_window(window: int, reserve: int, prompt_chars: int) -> agent_chat._TurnContext:
    ctx = agent_chat._TurnContext(
        tool_runtime=None,
        turn_cards={},
        calls=[],
        search_count=0,
        read_count=0,
        status_buf=[],
        seed_text="",
        seed_hit=False,
        sys_prompt="",
        user_message="",
        question="q",
        model=None,
        model_settings={},
        budget=EconomyBudget(
            preseed_passages=6,
            preseed_passage_max_chars=0,
            read_max_chars=0,
            max_output_tokens=1280,
            tool_budget_chars=0,
            age_tool_chars=0,
            search_entries=6,
            search_snippet_chars=400,
            window_tokens=window,
            output_reserve=reserve,
        ),
        started=0.0,
    )
    ctx.prompt_chars = prompt_chars
    return ctx


def test_ledger_blocks_the_insert_that_would_overflow() -> None:
    """The projection is over THIS insert: a size that still fits passes, the
    size that would push the NEXT request past window - reserve is rejected
    whole — not the insert after it."""
    ctx = _ctx_with_window(8_192, 1_280, prompt_chars=20_000)
    # ceil((20000 + 100 + 512)/3) = 6871 ≤ 6912 → fits, no latch.
    assert ctx._tool_budget_blocks(100) is False
    assert ctx.tool_budget_exhausted is False
    # ceil((20000 + 300 + 512)/3) = 6938 > 6912 → THIS insert is blocked…
    assert ctx._tool_budget_blocks(300) is True
    # …and the rejection latches: later calls steer before executing.
    assert ctx.tool_budget_exhausted is True
    assert ctx._tool_budget_blocks() is True
    assert ctx._tool_budget_blocks(1) is True


def test_ledger_projects_from_metered_request_size() -> None:
    """Once a request is on the meter, the base is its exact wire size — the
    running transcript (pre-seed + inserts + history), not the static prompt."""
    ctx = _ctx_with_window(8_192, 1_280, prompt_chars=20_000)
    ctx.meter.record(25_000, agent_chat.RequestUsage(input_tokens=8_000))
    # ceil((25000 + 512 + 0)/3) = 8504 > 6912 → even a zero-length insert is
    # over: the transcript itself already fills the evidence budget.
    assert ctx._tool_budget_blocks(0) is True


async def test_ledger_blocks_second_read_end_to_end(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forced 8k, two read_article calls: the first fits and is inserted; the
    second WOULD push the next request past the window and comes back as the
    steering message instead — its article text never enters the transcript."""
    article = "LEAD. " + ("filler about unrelated topics. " * 128) + "The answer is 42."
    # Seed sized for the 8k values (full prompt + 2400 cap): at the
    # old 1800-cap dense seed the FIRST read already overflowed the
    # evidence budget; at ~1540 chars/passage read 1 fits and read 2 is the
    # one that would overflow — the boundary this test exists to pin.
    runtime = FakeToolRuntime(
        [_scored(i, f"PASSAGE-{i} " + "dense seed text. " * 90) for i in range(6)],
        [_card(i, "s") for i in range(6)],
        article,
    )
    model = CapturingModel(
        [("read_article", '{"n": 1}'), ("read_article", '{"n": 2}')],
        "Answer from the first read: 42 [1].",
    )
    await _run(
        state,
        monkeypatch,
        make_snapshot(
            **{
                "answer.agent.context_profile": "8k",
                # Tail levers pinned OFF so this test stays about D5 in
                # isolation: the derived round cap would otherwise stop read 2
                # before retrieval, and the compact re-ask would replace the
                # steered final request this test inspects.
                "answer.agent.max_tool_rounds": 6,
                "answer.agent.compact_reask": "off",
            }
        ),
        model,
        runtime,
        question="What is the answer?",
    )
    # Both reads executed retrieval (the size is only known post-retrieval)…
    assert len(runtime.read_calls) == 2
    # …but only the first was inserted: the last request carries one article
    # plus the steering message, never a second copy of the article.
    last = model.seen[-1]
    wire = str(last)
    assert wire.count("filler about unrelated topics.") >= 1  # first read present
    assert "budget reached" in wire  # the steering replacement for read 2
    # And every request the model saw fits the window in estimate.
    w, r = 8_192, 1_280
    for messages in model.seen:
        assert estimate_tokens_for_chars(agent_chat._wire_chars(messages)) <= w - r + 512


# ── D1: the derived tool budget + the trace audit trail ─────────────────────


async def test_derived_tool_budget_recorded(state: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeToolRuntime(
        [_scored(i, f"PASSAGE-{i} " + "filler. " * 100) for i in range(6)],
        [_card(i, "s") for i in range(6)],
        "article",
    )
    model = CapturingModel([], "42 [1].")
    result = await _run(
        state,
        monkeypatch,
        make_snapshot(**{"answer.agent.context_profile": "8k"}),
        model,
        runtime,
        question="What is the answer?",
    )
    b = result.trace["budget"]
    assert b["window_tokens"] == 8_192
    assert b["output_reserve"] == 1_280
    # The derived allowance = window - reserve - est(prompt): positive and
    # stated, so the bench can report how much tool room the plan bought.
    assert b["tool_budget_tokens"] > 0
    assert b["tool_budget_tokens"] == 8_192 - 1_280 - estimate_tokens_for_chars(
        agent_chat._wire_chars(model.seen[0])
    )
    assert "preseed_dropped" in b


async def test_user_tool_budget_tokens_not_overwritten_by_derivation(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A user-set token budget binds; the per-turn derivation only fills an
    unset (0) knob (ladder rung one)."""
    runtime = FakeToolRuntime(
        [_scored(i, f"PASSAGE-{i} short") for i in range(6)],
        [_card(i, "s") for i in range(6)],
        "article",
    )
    model = CapturingModel([], "42 [1].")
    result = await _run(
        state,
        monkeypatch,
        make_snapshot(
            **{
                "answer.agent.context_profile": "8k",
                "answer.agent.tool_budget_tokens": 500,
            }
        ),
        model,
        runtime,
        question="What is the answer?",
    )
    assert result.trace["budget"]["tool_budget_tokens"] == 500


# ── full byte-identity against the locked reference ────────────────────────


def _mask_ids(s: str) -> str:
    # pydantic-ai mints a random "pyd_ai_<hex>" tool_call_id per run — not
    # behaviour. Mask it so runs stay byte-comparable (capture-script recipe).
    return f"pyd_ai_<id:{len(s)}>" if s.startswith("pyd_ai_") else s


def _ser_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [_ser_content(c) for c in content]
    if isinstance(content, dict):
        return {k: _ser_content(v) for k, v in sorted(content.items())}
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return f"block:{type(content).__name__}:{text}"
    return f"obj:{type(content).__name__}"


def _ser_part(p: Any) -> list[Any]:
    out: list[Any] = [type(p).__name__]
    out.append(_ser_content(getattr(p, "content", None)))
    args = getattr(p, "args", None)
    if args is not None:
        out.append(("args", _ser_content(args)))
    for attr in ("tool_name", "tool_call_id"):
        v = getattr(p, attr, None)
        if isinstance(v, str):
            out.append((attr, _mask_ids(v)))
    return out


def _ser_messages(messages: list[ModelMessage]) -> list[list[Any]]:
    return [[type(m).__name__, [_ser_part(p) for p in getattr(m, "parts", [])]] for m in messages]


async def _reference_scenario(
    state: Any,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    tool_calls: list[tuple[str, str]],
    answer: str,
    *,
    question: str,
) -> dict[str, Any]:
    """Re-run one capture-script scenario and serialize it the same way."""
    passages = [
        _scored(i, f"PASSAGE-{i} says the answer is forty-two. " + "detail " * (i + 3))
        for i in range(4)
    ]
    runtime = FakeToolRuntime(
        passages,
        [_card(i, passages[i].passage.text[:60]) for i in range(4)],
        "LEAD. " + ("filler about unrelated topics. " * 260) + "The answer is forty-two.",
    )
    model = CapturingModel(tool_calls, answer)
    result = await _run(
        state, monkeypatch, make_snapshot(**overrides), model, runtime, question=question
    )
    return {
        "requests": [_ser_messages(m) for m in model.seen],
        "budget": dict(result.trace["budget"]),
        "answer": result.answer,
        "cards": [[c.n, c.zim_id, c.path, c.title, c.snippet] for c in result.cards],
        "tool_calls": [[t.query, t.result_preview] for t in result.tool_calls],
        "stage_names": [s["name"] for s in result.trace["stages"]],
        "requests_count": result.trace.get("requests"),
        "peak_input_tokens": result.trace.get("peak_input_tokens"),
    }


#: The context-window keys the budget legitimately gained after the
#: reference was locked; at `full` they must all resolve to exactly zero.
_NEW_BUDGET_KEYS = {
    "window_tokens": 0,
    "output_reserve": 0,
    "tool_budget_tokens": 0,
    "preseed_dropped": 0,
}

_REFERENCE_SCENARIOS: list[tuple[str, dict[str, Any], list[tuple[str, str]], str]] = [
    ("s1_defaults_oneshot", {}, [], "The answer is forty-two [1]."),
    (
        "s2_economy_read",
        {"answer.agent.economy": "on"},
        [("read_article", '{"n": 1}')],
        "Read confirms: forty-two [1].",
    ),
    (
        "s3_economy_compact",
        {"answer.agent.economy": "on", "answer.agent.compact_prompt": True},
        [],
        "Forty-two [1].",
    ),
]


@pytest.mark.parametrize(
    ("name", "overrides", "tool_calls", "answer"),
    _REFERENCE_SCENARIOS,
    ids=[s[0] for s in _REFERENCE_SCENARIOS],
)
async def test_full_profile_is_byte_identical_to_reference(
    state: Any,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    overrides: dict[str, Any],
    tool_calls: list[tuple[str, str]],
    answer: str,
) -> None:
    """`full` (and economy-only) behaviour is byte-identical to the locked
    pre-change reference: same requests, answers, cards, tool calls, stages —
    the control for every future measurement."""
    ref_path = Path(__file__).parent / "fixtures" / "phase21_full_reference.json"
    reference = json.loads(ref_path.read_text(encoding="utf-8"))[name]
    phase21_defaults = {
        "answer.agent.preseed_order": "rank",
        "answer.agent.preseed_show_archive_id": True,
        "answer.agent.coverage_search": False,
        "answer.agent.evidence_directive": "standard",
        "answer.agent.answer_cleanup": False,
    }
    current = await _reference_scenario(
        state,
        monkeypatch,
        {**phase21_defaults, **overrides},
        tool_calls,
        answer,
        question="What is the answer?",
    )

    # Byte-identity on everything the model and the user see. The reference
    # round-tripped through JSON (tuples become lists); normalize the same way.
    assert json.loads(json.dumps(current["requests"])) == reference["requests"]
    assert current["answer"] == reference["answer"]
    assert current["cards"] == reference["cards"]
    assert current["tool_calls"] == reference["tool_calls"]
    assert current["stage_names"] == reference["stage_names"]
    assert current["requests_count"] == reference["requests_count"]
    assert current["peak_input_tokens"] == reference["peak_input_tokens"]

    # The budget dict = the reference plus exactly the four new window keys,
    # all zero at full — nothing else drifted.
    ref_budget = dict(reference["budget"])
    assert current["budget"] == {**ref_budget, **_NEW_BUDGET_KEYS}


def test_settings_override_sets_effective_context_profile() -> None:
    """The override merged by `_open_runtime` becomes the snapshot's effective
    value — what agent_chat's resolve_budget reads, and what the run's
    settings_snapshot persists."""
    from vesta import config
    from vesta.answer import ANSWER_AGENT_CONTEXT_PROFILE

    config.configure()
    try:
        for forced in ("auto", "8k", "16k", "full"):
            config.set_db_values({ANSWER_AGENT_CONTEXT_PROFILE.key: forced})
            snap = config.snapshot()
            assert snap.get(ANSWER_AGENT_CONTEXT_PROFILE) == forced
            assert snap.values[ANSWER_AGENT_CONTEXT_PROFILE.key] == forced
    finally:
        config.reset_for_test()


def _dense_seed_runtime(article: str = "article body", n: int = 6) -> FakeToolRuntime:
    """The 8k-shaped FakeToolRuntime: n ~1540-char passages the 2400-char
    plan cap renders at full length, sized (values: full prompt +
    2400 cap) so the FIRST read insert fits the evidence budget and the
    second is the one the D5 ledger rejects — the boundary these
    tests pin."""
    passages = [_scored(i, f"PASSAGE-{i} " + "dense seed text. " * 90) for i in range(n)]
    return FakeToolRuntime(passages, [_card(i, "s") for i in range(n)], article)


async def test_phase3_levers_inert_at_full(state: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """At `full` with default knobs none of the tail levers fire: no round cap (the
    budget key stays absent — the byte-identity trap's dict shape), no
    compact re-ask, no aging — even on a multi-round turn, and the abstention
    case still takes TODAY's retry channel (the directive, not a re-ask)."""
    article = "LEAD. " + ("filler about unrelated topics. " * 128) + "The answer is 42."
    runtime = _dense_seed_runtime(article)
    model = CapturingModel([("read_article", '{"n": 1}')], "The answer is 42 [1].")
    result = await _run(
        state, monkeypatch, make_snapshot(), model, runtime, question="What is the answer?"
    )
    assert "max_tool_rounds" not in result.trace["budget"]
    assert result.trace["budget"]["age_tool_chars"] == 0
    assert result.trace["compact_reask"] == {"fired": False, "trigger": None}
    assert len(runtime.read_calls) == 1  # the harness read cap, not a round cap

    # Abstention at full: the OLD retry runs (directive present), the re-ask
    # does not.
    runtime2 = _dense_seed_runtime(article)
    model2 = CapturingModel([], "I could not find the answer.")
    result2 = await _run(
        state, monkeypatch, make_snapshot(), model2, runtime2, question="What is the answer?"
    )
    assert result2.trace["compact_reask"] == {"fired": False, "trigger": None}
    assert any("You did not give an answer" in str(m) for m in model2.seen)


async def test_phase3_levers_ship_off_at_windowed_defaults(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measured verdict, as a regression test: all three tail levers LOST
    their 8k-fullprompt-wide A/Bs against run 64 (runs 72/73/74), so pure
    DEFAULT knobs at a windowed profile reproduce run 64's shape — no round
    cap (the budget key stays absent), no aging, no re-ask, and an abstainer
    still takes the OLD retry channel. The opt-in values are the separate
    tests below."""
    article = "LEAD. " + ("filler about unrelated topics. " * 128) + "The answer is 42."
    runtime = _dense_seed_runtime(article)
    model = CapturingModel(
        [
            ("read_article", '{"n": 1}'),
            ("read_article", '{"n": 2}'),
            ("read_article", '{"n": 3}'),
        ],
        "The answer is 42 [1].",
    )
    result = await _run(
        state,
        monkeypatch,
        make_snapshot(**{"answer.agent.context_profile": "8k"}),
        model,
        runtime,
        question="What is the answer?",
    )
    assert "max_tool_rounds" not in result.trace["budget"]
    assert result.trace["budget"]["age_tool_chars"] == 0
    # Reads bounded only by the harness cap / the D5 ledger — never
    # round-cap steering (the "every tool round" string).
    assert len(runtime.read_calls) <= 3
    assert not any("every tool round" in str(m) for m in model.seen)
    assert result.trace["round_cap_fires"] == 0
    assert result.trace["aged_requests"] == 0

    # Abstention under the window: the OLD retry runs, the re-ask does not.
    runtime2 = _dense_seed_runtime(article)
    model2 = CapturingModel([], "I could not find the answer.")
    result2 = await _run(
        state,
        monkeypatch,
        make_snapshot(**{"answer.agent.context_profile": "8k"}),
        model2,
        runtime2,
        question="What is the answer?",
    )
    assert result2.trace["compact_reask"] == {"fired": False, "trigger": None}
    assert any("You did not give an answer" in str(m) for m in model2.seen)


async def test_round_cap_derived_from_window_and_enforced(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(a) Forced 8k with the levers at their explicit-on values (they ship
    OFF at defaults — runs 72/73/74): cap 0 = derive, so the cap is the
    ledger arithmetic — ``tool_budget_tokens // est(read_max_chars)``
    floored at 1 — recorded in trace.budget; once spent, further tool calls
    get the round-cap steering BEFORE executing, and the re-ask fires on
    the round_cap trigger."""
    article = "LEAD. " + ("filler about unrelated topics. " * 128) + "The answer is 42."
    runtime = _dense_seed_runtime(article)
    model = CapturingModel(
        [
            ("read_article", '{"n": 1}'),
            ("read_article", '{"n": 2}'),
            ("read_article", '{"n": 3}'),
        ],
        "The answer is 42 [1].",
    )
    result = await _run(
        state,
        monkeypatch,
        make_snapshot(
            **{
                "answer.agent.context_profile": "8k",
                "answer.agent.max_tool_rounds": 0,
                "answer.agent.compact_reask": "auto",
            }
        ),
        model,
        runtime,
        question="What is the answer?",
    )
    b = result.trace["budget"]
    ledger = b["tool_budget_tokens"]
    assert ledger > 0
    assert b["max_tool_rounds"] == min(max(1, ledger // 1_500), 6)  # read_max_chars=4500
    assert len(runtime.read_calls) == b["max_tool_rounds"]
    assert any("every tool round" in str(m) for m in model.seen)
    assert result.trace["round_cap_fires"] >= 1  # persisted firing count
    # The re-ask fired on the round_cap trigger with a fresh, windowed, citable ask.
    assert result.trace["compact_reask"]["fired"] is True
    assert result.trace["compact_reask"]["trigger"] == "round_cap"
    assert "p6_abstain" not in result.trace["compact_reask"]
    # The pricing key rides along on every trigger (the steered alternative
    # this firing replaced).
    assert "steered_est_tokens" in result.trace["compact_reask"]
    last = model.seen[-1]
    assert not [
        p for m in last for p in getattr(m, "parts", []) if type(p).__name__ == "ToolReturnPart"
    ]
    assert "Initial sources for this question:" in str(last)
    assert estimate_tokens_for_chars(agent_chat._wire_chars(last)) <= 8_192 - 1_280 + 512
    assert "compact_reask" in [s["name"] for s in result.trace["stages"]]
    # Citability: the re-asked answer's [1] still resolves to a real card.
    assert "[1]" in result.answer
    assert any(c.n == 1 for c in result.cards)


async def test_round_cap_user_set_wins_over_derivation(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(a) ladder: an explicit ``answer.agent.max_tool_rounds`` beats the
    window derivation at 8k."""
    article = "LEAD. " + ("filler about unrelated topics. " * 128) + "The answer is 42."
    runtime = _dense_seed_runtime(article)
    model = CapturingModel(
        [
            ("read_article", '{"n": 1}'),
            ("read_article", '{"n": 2}'),
            ("read_article", '{"n": 3}'),
        ],
        "The answer is 42 [1].",
    )
    result = await _run(
        state,
        monkeypatch,
        make_snapshot(
            **{
                "answer.agent.context_profile": "8k",
                "answer.agent.max_tool_rounds": 2,
                "answer.agent.compact_reask": "off",
            }
        ),
        model,
        runtime,
        question="What is the answer?",
    )
    assert result.trace["budget"]["max_tool_rounds"] == 2
    assert len(runtime.read_calls) == 2
    assert result.trace["compact_reask"] == {"fired": False, "trigger": None}


async def test_round_cap_user_set_applies_at_full(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(a) an explicit cap binds on any profile (the universal user-set-wins
    ladder); only the DEFAULT (0) is inert at full."""
    runtime = _dense_seed_runtime()
    model = CapturingModel(
        [("read_article", '{"n": 1}'), ("read_article", '{"n": 2}')],
        "The answer is 42 [1].",
    )
    result = await _run(
        state,
        monkeypatch,
        make_snapshot(**{"answer.agent.max_tool_rounds": 1}),
        model,
        runtime,
        question="What is the answer?",
    )
    assert result.trace["budget"]["max_tool_rounds"] == 1
    assert len(runtime.read_calls) == 1


def test_round_cap_not_derived_without_read_cap() -> None:
    """(a) unit: with no ``read_max_chars`` there is no insert size to derive
    from — no round cap (the D5 ledger still guards every insert). No plan
    ships ``read_max_chars=0`` under a window, so this arm is exercised
    directly against the resolver."""
    from dataclasses import replace

    ctx = _ctx_with_window(8_192, 1_280, prompt_chars=0)  # read_max_chars=0
    ctx.budget = replace(ctx.budget, tool_budget_tokens=2_000)
    agent_chat._resolve_tail_levers(make_snapshot(), ctx, ctx.budget)
    assert ctx.max_tool_rounds == 0


async def test_compact_reask_trigger_ledger(state: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """(b) the ledger-exhausted trigger: with the round cap pinned open, the
    read the D5 ledger rejects latches the budget → the re-ask fires with
    trigger 'ledger' on a fresh single request."""
    article = "LEAD. " + ("filler about unrelated topics. " * 128) + "The answer is 42."
    runtime = _dense_seed_runtime(article)
    model = CapturingModel(
        [("read_article", '{"n": 1}'), ("read_article", '{"n": 2}')],
        "The answer is 42 [1].",
    )
    result = await _run(
        state,
        monkeypatch,
        make_snapshot(
            **{
                "answer.agent.context_profile": "8k",
                "answer.agent.max_tool_rounds": 6,
                "answer.agent.compact_reask": "auto",
            }
        ),
        model,
        runtime,
        question="What is the answer?",
    )
    assert "budget reached" in str(model.seen[-2])  # the D5 latch did the blocking
    reask = result.trace["compact_reask"]
    assert reask["fired"] is True
    assert reask["trigger"] == "ledger"
    last = model.seen[-1]
    assert "p6_abstain" not in reask
    assert "Initial sources for this question:" in str(last)
    assert estimate_tokens_for_chars(agent_chat._wire_chars(last)) <= 8_192 - 1_280 + 512


async def test_p6_at_floor_reasks_once_with_focused_citable_sources(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At P10's exact score floor P6 replaces the old retry with one fresh,
    no-tool focused answer request; its source headers remain attached."""
    passages = [
        replace(_scored(i, f"PASSAGE-{i} " + "dense source evidence. " * 90), score=0.85)
        for i in range(2)
    ]
    cards = [replace(_card(i, "s"), score=0.85) for i in range(2)]
    runtime = FakeToolRuntime(passages, cards, "article")
    model = CapturingModel([], "I could not find the answer.")
    result = await _run(
        state,
        monkeypatch,
        make_snapshot(
            **{"answer.agent.context_profile": "8k", "answer.agent.compact_reask": "auto"}
        ),
        model,
        runtime,
        question="What is the answer?",
    )
    reask = result.trace["compact_reask"]
    assert reask["fired"] is True
    assert reask["trigger"] == "abstain_p6"
    assert reask["p6_abstain"] == {
        "top_score": 0.85,
        "floor": 0.85,
        "focused_chars": reask["p6_abstain"]["focused_chars"],
        "fired": True,
    }
    assert reask["p6_abstain"]["focused_chars"] > 0
    assert len(model.seen) == 2  # initial abstention + exactly one no-tool answer call
    first, last = (str(messages) for messages in model.seen)
    assert len(last) < len(first)  # never resend the ordinary full pre-seed
    assert "You did not give an answer" not in last
    assert "Initial sources for this question:" not in last
    assert "The fact answering this question is in the focused sources below" in last
    assert "Question: What is the answer?" in last
    assert '[1] "Article 0"' in last
    assert "PASSAGE-0" in last
    assert not any(
        type(part).__name__ == "ToolReturnPart"
        for message in model.seen[-1]
        for part in getattr(message, "parts", [])
    )
    assert runtime.read_calls == []


async def test_p6_below_floor_keeps_ordinary_abstention_retry(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A seed just below P10's floor must not create P6's focused re-ask."""
    runtime = FakeToolRuntime(
        [replace(_scored(0, "relevant evidence"), score=0.849)],
        [replace(_card(0, "relevant evidence"), score=0.849)],
        "article",
    )
    model = CapturingModel([], "I could not find the answer.")
    result = await _run(
        state,
        monkeypatch,
        make_snapshot(
            **{"answer.agent.context_profile": "8k", "answer.agent.compact_reask": "auto"}
        ),
        model,
        runtime,
        question="What is the answer?",
    )
    assert result.trace["compact_reask"] == {"fired": False, "trigger": None}
    assert len(model.seen) == 2
    assert any("You did not give an answer" in str(messages) for messages in model.seen)


async def test_p6_streaming_resets_to_focused_answer_with_citations_and_trace(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SSE path uses the same P6 prompt/trace, resets once, then emits the
    replacement cited answer rather than stacking the ordinary retry."""
    runtime = _dense_seed_runtime()
    stream_calls: list[list[ModelMessage]] = []
    reask_calls: list[list[ModelMessage]] = []

    def _model(*_args: Any, **_kwargs: Any) -> FunctionModel:
        async def stream_fn(messages: list[ModelMessage], _info: Any) -> AsyncIterator[Any]:
            stream_calls.append(list(messages))
            yield "I could not find the answer."

        def fn(messages: list[ModelMessage], _info: Any) -> ModelResponse:
            reask_calls.append(list(messages))
            return ModelResponse(parts=[TextPart(content="The answer is 42 [1].")])

        return FunctionModel(function=fn, stream_function=stream_fn)

    monkeypatch.setattr(agent_chat, "_build_tool_runtime", lambda *a, **k: runtime)
    monkeypatch.setattr(agent_chat, "_make_model", _model)
    monkeypatch.setattr(
        agent_chat, "_resolve_llm", lambda _sn: ("fake-model", "http://fake", "k", None, None)
    )
    events = [
        event
        async for event in agent_chat.iter_agent_turn_events(
            state,
            make_snapshot(
                **{"answer.agent.context_profile": "8k", "answer.agent.compact_reask": "auto"}
            ),
            "What is the answer?",
        )
    ]

    assert len(stream_calls) == 1
    assert len(reask_calls) == 1
    assert estimate_tokens_for_chars(agent_chat._wire_chars(reask_calls[0])) <= 8_192 - 1_280
    reask_wire = str(reask_calls[0])
    assert '[1] "Article 0"' in reask_wire
    assert not any(
        type(part).__name__ == "ToolReturnPart"
        for message in reask_calls[0]
        for part in getattr(message, "parts", [])
    )
    resets = [event for event in events if isinstance(event, AnswerResetEvent)]
    assert [event.reason for event in resets] == ["compact_reask"]
    tokens = [event.text for event in events if isinstance(event, TokenEvent)]
    assert tokens[-1] == "The answer is 42 [1]."
    citations = [event for event in events if isinstance(event, CitationsEvent)]
    assert citations and citations[-1].answer_text == "The answer is 42 [1]."
    trace = next(event for event in events if isinstance(event, TraceEvent)).trace
    assert trace["compact_reask"]["trigger"] == "abstain_p6"
    assert trace["compact_reask"]["fired"] is True
    assert trace["compact_reask"]["p6_abstain"]["fired"] is True
    assert trace["compact_reask"]["p6_abstain"]["focused_chars"] > 0


async def test_compact_reask_forced_on_at_full(state: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """(b) 'on' forces the lever on any profile (the bench A/B axis), the
    economy auto|on|off mirror."""
    runtime = _dense_seed_runtime()
    model = CapturingModel([], "I could not find the answer.")
    result = await _run(
        state,
        monkeypatch,
        make_snapshot(**{"answer.agent.compact_reask": "on"}),
        model,
        runtime,
        question="What is the answer?",
    )
    assert result.trace["compact_reask"]["fired"] is True


async def test_compact_reask_off_keeps_old_retry_at_8k(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(b) 'off' forces the lever off even under a windowed profile — today's
    abstention retry runs instead."""
    runtime = _dense_seed_runtime()
    model = CapturingModel([], "I could not find the answer.")
    result = await _run(
        state,
        monkeypatch,
        make_snapshot(
            **{"answer.agent.context_profile": "8k", "answer.agent.compact_reask": "off"}
        ),
        model,
        runtime,
        question="What is the answer?",
    )
    assert result.trace["compact_reask"] == {"fired": False, "trigger": None}
    assert any("You did not give an answer" in str(m) for m in model.seen)


def test_compact_reask_message_fits_and_keeps_read_citations() -> None:
    """(b) unit: the built message fits window - reserve under the estimator,
    keeps the pre-seed framing, and every read block survives the focused
    window as a must-include span — the citing minority stays served."""
    ctx = _ctx_with_window(8_192, 1_280, prompt_chars=0)
    ctx.sys_prompt = "system prompt filler. " * 80
    ctx.question = "What is the answer?"
    ctx.seed_text = "SEED. " + "seed filler text. " * 600
    ctx.turn_cards = {}
    for n in (2, 3):
        ctx.turn_cards[(1, f"a/{n}")] = agent_chat.SourceCardDTO(
            n=n,
            zim_id=1,
            path=f"a/{n}",
            title=f"Article {n}",
            snippet="s",
            breadcrumb="b",
            score=1.0,
            source="test",
        )
    ctx.read_excerpts = [
        (2, "READ2. " + "read two filler. " * 500),
        (3, "READ3. " + "read three filler. " * 500),
    ]
    message = agent_chat._compact_reask_message(ctx)
    assert message is not None
    assert estimate_tokens_for_chars(len(ctx.sys_prompt) + len(message)) <= 8_192 - 1_280
    assert message.startswith("Initial sources for this question:")
    assert message.endswith("Question: What is the answer?")
    assert "[2]" in message and "[3]" in message  # reads survived the windowing


def test_compact_reask_message_skips_without_evidence() -> None:
    """(b) unit: no seed and no reads → nothing to re-ask on → None (a
    follow-up turn that never gathered evidence keeps its answer)."""
    ctx = _ctx_with_window(8_192, 1_280, prompt_chars=0)
    ctx.sys_prompt = "sys"
    ctx.question = "q"
    ctx.seed_text = ""
    ctx.read_excerpts = []
    assert agent_chat._compact_reask_message(ctx) is None


async def test_age_derived_from_window_and_wrapped(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(c) an explicit 0 + window derives the eviction budget from the
    ledger — ``3 * tool_budget_tokens // (3 searches + 3 reads)`` — recorded
    in the budget audit and LIVE on the model chain this turn (aged inner,
    meter outermost). Aging ships OFF by default (run 74); 0 is the opt-in."""
    runtime = _dense_seed_runtime()
    model = CapturingModel([], "42 [1].")
    monkeypatch.setattr(agent_chat, "_build_tool_runtime", lambda *a, **k: runtime)
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: model._model)
    ctx = await agent_chat._build_turn(
        state,
        make_snapshot(**{"answer.agent.context_profile": "8k", "answer.agent.age_tool_chars": 0}),
        "What is the answer?",
        model_id="fake-model",
        endpoint="http://fake",
        api_key="k",
    )
    ledger = ctx.budget.tool_budget_tokens
    assert ledger > 0
    assert ctx.budget.age_tool_chars == (3 * ledger) // 6
    assert isinstance(ctx.model, agent_chat._MeteredModel)
    assert isinstance(ctx.model._inner, agent_chat._AgedContextModel)


async def test_age_ladder_explicit_wins_and_minus_one_off(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(c) ladder: an explicit positive value wins over the derivation, -1 is
    the off arm even under a window (and the shipped default since run 74),
    and without a window 0 stays today's aging-off."""
    runtime = _dense_seed_runtime()
    model = CapturingModel([], "42 [1].")
    result = await _run(
        state,
        monkeypatch,
        make_snapshot(
            **{"answer.agent.context_profile": "8k", "answer.agent.age_tool_chars": 5_000}
        ),
        model,
        runtime,
        question="What is the answer?",
    )
    assert result.trace["budget"]["age_tool_chars"] == 5_000

    runtime2 = _dense_seed_runtime()
    model2 = CapturingModel([], "42 [1].")
    result2 = await _run(
        state,
        monkeypatch,
        make_snapshot(**{"answer.agent.context_profile": "8k", "answer.agent.age_tool_chars": -1}),
        model2,
        runtime2,
        question="What is the answer?",
    )
    assert result2.trace["budget"]["age_tool_chars"] == 0

    runtime3 = _dense_seed_runtime()
    model3 = CapturingModel([], "42 [1].")
    result3 = await _run(
        state, monkeypatch, make_snapshot(), model3, runtime3, question="What is the answer?"
    )
    assert result3.trace["budget"]["age_tool_chars"] == 0


async def test_compact_reask_prices_steered_alternative(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(b) observability: when a trigger fires, the trace record carries the
    measured fresh-request cost AND the estimate of the steered alternative
    it replaced — one more request on the transcript as it stood (the
    meter's last wire size) plus one steering message plus the D5 slack —
    the per-firing price the bench reports without a steered control arm."""
    runtime = _dense_seed_runtime()
    model = CapturingModel([], "I could not find the answer.")
    result = await _run(
        state,
        monkeypatch,
        make_snapshot(
            **{"answer.agent.context_profile": "8k", "answer.agent.compact_reask": "auto"}
        ),
        model,
        runtime,
        question="What is the answer?",
    )
    reask = result.trace["compact_reask"]
    assert reask["fired"] is True
    assert reask["trigger"] == "abstain_p6"
    # The steered estimate = the transcript's last wire size (the abstain
    # turn's only main-loop request) + steering + slack, under the estimator.
    expected = estimate_tokens_for_chars(
        agent_chat._wire_chars(model.seen[0])
        + max(len(agent_chat._ROUND_CAP_STEERING), len(agent_chat._TOOL_BUDGET_STEERING))
        + agent_chat._WINDOW_LEDGER_SLACK_CHARS
    )
    assert reask["steered_est_tokens"] == expected
    # The fresh request is the cheaper shape: its own message estimates
    # below the steered alternative it replaced.
    assert estimate_tokens_for_chars(reask["chars"]) <= reask["steered_est_tokens"]


async def test_compact_reask_record_minimal_without_trigger(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(b) observability: no trigger → the record stays the minimal
    ``{fired, trigger}`` pair (no pricing keys) — the shape the inert-at-full
    trap already pins."""
    runtime = _dense_seed_runtime()
    model = CapturingModel([], "42 [1].")
    result = await _run(
        state,
        monkeypatch,
        make_snapshot(**{"answer.agent.context_profile": "8k"}),
        model,
        runtime,
        question="What is the answer?",
    )
    assert result.trace["compact_reask"] == {"fired": False, "trigger": None}


async def test_aging_firing_counted_in_trace(state: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """(c) observability: with the derived eviction budget live, a multi-read
    turn's later requests carry their oldest round truncated — the trace
    counts the aged requests and the chars the trimming saved (the firing
    evidence; 0/0 when aging is off, pinned by the inert-at-full
    test's ``age_tool_chars == 0``)."""
    article = "LEAD. " + ("filler about unrelated topics. " * 128) + "The answer is 42."
    runtime = _dense_seed_runtime(article)
    model = CapturingModel(
        [
            ("read_article", '{"n": 1}'),
            ("read_article", '{"n": 2}'),
            ("read_article", '{"n": 3}'),
        ],
        "The answer is 42 [1].",
    )
    result = await _run(
        state,
        monkeypatch,
        make_snapshot(
            **{
                "answer.agent.context_profile": "8k",
                "answer.agent.max_tool_rounds": 6,
                "answer.agent.compact_reask": "off",
                "answer.agent.age_tool_chars": 0,
            }
        ),
        model,
        runtime,
        question="What is the answer?",
    )
    b = result.trace["budget"]
    assert b["age_tool_chars"] == (3 * b["tool_budget_tokens"]) // 6
    # Three tool rounds exist; every request past the second round carries
    # the first round's article truncated to the derived budget.
    assert result.trace["aged_requests"] >= 1
    assert result.trace["age_saved_chars"] > 0


# ── N7: history-aware fit arithmetic ────────────────────────────────────────


def _history_of(*exchanges: tuple[str, str]) -> list[ModelMessage]:
    """A reconstructed conversation: one user/assistant pair per exchange."""
    out: list[ModelMessage] = []
    for q, a in exchanges:
        out.append(ModelRequest(parts=[UserPromptPart(content=q)]))
        out.append(ModelResponse(parts=[TextPart(content=a)]))
    return out


async def test_legacy_preseed_fit_sheds_for_history(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """History present with contextual follow-ups OFF (legacy path): the first
    request carries the whole conversation, so the pre-seed fit must shed tail
    passages for it too — the request still fits ``window - reserve``."""
    runtime = FakeToolRuntime(
        [_scored(i, f"PASSAGE-{i} " + "dense filler. " * 380) for i in range(6)],
        [_card(i, "s") for i in range(6)],
        "article",
    )
    model = CapturingModel([], "The answer is forty-two [1].")
    history = _history_of(("prior question " + "q" * 3000, "prior answer " + "a" * 2000))
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: model.model)
    monkeypatch.setattr(agent_chat, "_build_tool_runtime", lambda *a, **k: runtime)
    monkeypatch.setattr(agent_chat, "_contextual_followups_enabled", lambda sn: False)
    result = await agent_chat.run_one_turn(
        state,
        make_snapshot(**{"answer.agent.context_profile": "8k"}),
        "What is the answer?",
        model_id="fake-model",
        endpoint="http://fake",
        api_key="k",
        message_history=history,
    )
    b = result.trace["budget"]
    assert b["window_tokens"] == 8_192
    # The fitted first request INCLUDES the history and still fits.
    assert estimate_tokens_for_chars(agent_chat._wire_chars(model.seen[0])) <= (
        b["window_tokens"] - b["output_reserve"]
    )
    # The fit shed pre-seed passages to make room for the conversation.
    assert b["preseed_dropped"] >= 1
    assert "prior answer" in str(model.seen[0])


async def test_followup_first_request_fit_latched_when_over_budget(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The follow-up branch performs the first-request fit check (the twin of
    turn-1's pre-seed fit): a shape that fits stays unmarked; an over-budget
    one — nothing there is shedable — latches ``first_request_over_budget``
    into the budget audit."""
    runtime = FakeToolRuntime([], [], "article")
    monkeypatch.setattr(agent_chat, "_build_tool_runtime", lambda *a, **k: runtime)
    sn = make_snapshot(**{"answer.agent.context_profile": "8k"})

    small = _history_of(("short question", "short answer"))
    ctx = await agent_chat._build_turn(
        state,
        sn,
        "Who died first",
        model_id="m",
        endpoint="http://fake",
        api_key="k",
        message_history=small,
    )
    assert ctx.follow_up is True
    assert ctx.first_request_over_budget is False
    assert "first_request_over_budget" not in agent_chat._budget_audit(ctx)

    big = _history_of(("question " + "q" * 30_000, "answer " + "a" * 10_000))
    ctx = await agent_chat._build_turn(
        state,
        sn,
        "Who died first",
        model_id="m",
        endpoint="http://fake",
        api_key="k",
        message_history=big,
    )
    assert ctx.first_request_over_budget is True
    assert agent_chat._budget_audit(ctx)["first_request_over_budget"] is True


class OverflowOnceModel:
    """First request raises a context-overflow 400; every later one answers.
    Records every request's messages verbatim."""

    def __init__(self, answer: str):
        self.seen: list[list[ModelMessage]] = []
        outer = self

        async def fn(messages: list[ModelMessage], info: Any) -> ModelResponse:
            outer.seen.append(list(messages))
            if len(outer.seen) == 1:
                raise ModelHTTPError(
                    status_code=400,
                    model_name="fake-model",
                    body={"error": "Request exceeded the context window."},
                )
            return ModelResponse(parts=[TextPart(content=answer)])

        self.model = FunctionModel(function=fn)


async def _overflow_run(
    state: Any,
    monkeypatch: pytest.MonkeyPatch,
    history: list[ModelMessage],
) -> Any:
    runtime = FakeToolRuntime(
        [_scored(i, f"P-{i} small passage") for i in range(6)],
        [_card(i, "s") for i in range(6)],
        "article",
    )
    model = OverflowOnceModel("Recovered.")
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: model.model)
    monkeypatch.setattr(agent_chat, "_build_tool_runtime", lambda *a, **k: runtime)
    result = await agent_chat.run_one_turn(
        state,
        make_snapshot(**{"answer.agent.context_profile": "8k"}),
        "What is the answer?",
        model_id="fake-model",
        endpoint="http://fake",
        api_key="k",
        message_history=history,
    )
    return result, model


async def test_overflow_fallback_keeps_history_that_fits(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A history-bearing turn whose main run overflows recovers WITH its
    conversation: when prompt + history fit the window plan, the no-tool
    fallback re-sends the pre-turn history."""
    result, model = await _overflow_run(
        state, monkeypatch, _history_of(("prior question", "prior answer about Lafayette"))
    )
    assert result.trace["overflow_fallbacks"] == 1
    assert result.answer == "Recovered."
    assert len(model.seen) == 2
    # The fallback's request carries the pre-turn history AND the question.
    assert "prior answer about Lafayette" in str(model.seen[1])
    assert "What is the answer?" in str(model.seen[1])
    assert "fallback_history_dropped" not in result.trace["budget"]


async def test_overflow_fallback_drops_history_that_would_not_fit(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When prompt + history would overflow the window plan again, the
    fallback sheds the history deliberately and latches the marker."""
    result, model = await _overflow_run(
        state, monkeypatch, _history_of(("q " + "q" * 30_000, "a " + "a" * 10_000))
    )
    assert result.trace["overflow_fallbacks"] == 1
    assert len(model.seen) == 2
    # The fallback recovered WITHOUT the conversation (deliberate degrade).
    assert "qqqqq" not in str(model.seen[1])
    assert result.trace["budget"]["fallback_history_dropped"] is True
