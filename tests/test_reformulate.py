"""Conditional Round-0 reformulation.

Covers the three contracts:

- **S4 (healthy questions pay nothing):** above ``trigger_score`` the gateway
  is never touched.
- **Never-worse:** gateway exception, empty output, stagnant
  near-duplicate queries, or a non-improving re-search all return the original
  result untouched; a replacement must strictly beat the original top_score.
- **S5 (degrade-don't-fail):** no gateway / no model / no snapshot behaves
  exactly like today.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from vesta.answer.reformulate import (
    MINIMAL_REFORMULATE_SYSTEM_PROMPT,
    REFORMULATE_SYSTEM_PROMPT,
    GatewayReformulator,
    parse_reformulations,
)
from vesta.config.settings import SettingsSnapshot, all_settings
from vesta.inference.gateway import ChatMessage, ChatResult, NullGateway


def make_snapshot(**overrides: Any) -> SettingsSnapshot:
    """A snapshot holding every setting's registered default plus overrides."""
    values: dict[str, object] = {s.key: s.default for s in all_settings().values()}
    values.update(overrides)
    return SettingsSnapshot(values=values)


def _builder() -> Any:
    """`_build_reformulator` viewed as Any — tests pass SimpleNamespace states."""
    from vesta.api.answer import _build_reformulator

    return _build_reformulator


def _mrr() -> Any:
    """`_maybe_reformulate_round0` viewed as Any — tests pass duck-typed fakes."""
    from vesta.api.answer import _maybe_reformulate_round0

    return _maybe_reformulate_round0


BOND_Q = "At approximately what value of the Bond number does confined boiling become dominant?"


# ── parse_reformulations ─────────────────────────────────────────────────────


class TestParse:
    def test_strips_decorations_and_drops_noise(self) -> None:
        text = '1. Boiling\n- → Film boiling\nQ: ignored\n(bad: bond number)\n\n"Heat transfer"'
        assert parse_reformulations(text, limit=3, original=BOND_Q) == [
            "Boiling",
            "Film boiling",
            "Heat transfer",
        ]

    def test_strips_query_label_keeps_content(self) -> None:
        assert parse_reformulations("Query: Boiling", limit=1, original=BOND_Q) == ["Boiling"]

    def test_caps_at_limit_strongest_first(self) -> None:
        text = "Boiling\nFilm boiling\nNucleate boiling"
        assert parse_reformulations(text, limit=2, original=BOND_Q) == [
            "Boiling",
            "Film boiling",
        ]

    def test_near_duplicate_of_original_is_stagnation(self) -> None:
        """A re-echo of the question's own words re-runs the same failed
        AND-match — the plan's stagnation case; it must be dropped."""
        echo = "Bond number confined boiling become dominant"
        assert parse_reformulations(echo, limit=2, original=BOND_Q) == []

    def test_reordered_duplicate_dropped(self) -> None:
        """Same tokens reordered add nothing to a word matcher."""
        text = "Film boiling\nboiling film"
        assert parse_reformulations(text, limit=2, original=BOND_Q) == ["Film boiling"]

    def test_digit_leading_title_survives_numbering_stripped(self) -> None:
        text = "1. Boiling\nBoeing 747"
        assert parse_reformulations(text, limit=3, original=BOND_Q) == [
            "Boiling",
            "Boeing 747",
        ]

    def test_overlong_fact_shaped_line_dropped(self) -> None:
        text = "the exact value at which confined boiling becomes dominant regime"
        assert parse_reformulations(text, limit=2, original=BOND_Q) == []

    def test_blank_and_garbage_returns_empty(self) -> None:
        assert parse_reformulations("", limit=2, original=BOND_Q) == []
        assert parse_reformulations("(bad: everything)", limit=2, original=BOND_Q) == []


# ── Prompt arms (the A/B contract) ──────────────────────────────────────────


class TestPromptArms:
    def test_minimal_is_the_control_arm(self) -> None:
        """Conservative, zero examples, REWRITE_SYSTEM_PROMPT posture."""
        assert "Q:" not in MINIMAL_REFORMULATE_SYSTEM_PROMPT
        assert "Do not expand" in MINIMAL_REFORMULATE_SYSTEM_PROMPT
        assert len(MINIMAL_REFORMULATE_SYSTEM_PROMPT) < 400


# ── GatewayReformulator ──────────────────────────────────────────────────────


class _FakeGateway:
    def __init__(self, text: str = "Boiling") -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    async def chat_once(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 300,
        enable_thinking: bool | None = None,
        timeout: float | None = None,
    ) -> ChatResult:
        self.calls.append(
            {
                "messages": list(messages),
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "enable_thinking": enable_thinking,
            }
        )
        return ChatResult(text=self.text, finish_reason="stop", latency_ms=1.0)

    async def chat_stream(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError  # protocol completeness; never streamed here
        yield  # unreachable — makes this an async generator per the Protocol

    async def aclose(self) -> None:  # pragma: no cover
        """Protocol completeness; the reformulator never closes the gateway."""


class _RaisingGateway(_FakeGateway):
    async def chat_once(self, *a: Any, **k: Any) -> ChatResult:
        self.calls.append({})
        raise RuntimeError("gateway down")


class TestGatewayReformulator:
    @pytest.mark.asyncio
    async def test_maps_messages_and_posture(self) -> None:
        gw = _FakeGateway("Boiling")
        r = GatewayReformulator(gw, model="qwen3.5-4b@q4_k_s")
        out = await r.reformulate(BOND_Q, limit=1)
        assert out == ["Boiling"]
        assert len(gw.calls) == 1
        call = gw.calls[0]
        assert [m.role for m in call["messages"]] == ["system", "user"]
        assert call["messages"][0].content == REFORMULATE_SYSTEM_PROMPT
        assert call["messages"][1].content == BOND_Q
        # Hidden reasoning would burn the 64-token budget.
        assert call["enable_thinking"] is False
        assert call["temperature"] == 0.0
        assert call["max_tokens"] == 64
        assert call["model"] == "qwen3.5-4b@q4_k_s"

    @pytest.mark.asyncio
    async def test_minimal_prompt_arm_selectable(self) -> None:
        gw = _FakeGateway("Boiling")
        r = GatewayReformulator(gw, model="m", prompt=MINIMAL_REFORMULATE_SYSTEM_PROMPT)
        await r.reformulate(BOND_Q)
        assert gw.calls[0]["messages"][0].content == MINIMAL_REFORMULATE_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_gateway_failure_propagates_to_never_worse_caller(self) -> None:
        gw = _RaisingGateway()
        r = GatewayReformulator(gw, model="m")
        with pytest.raises(RuntimeError):
            await r.reformulate(BOND_Q)

    @pytest.mark.asyncio
    async def test_limit_forwards_to_parser(self) -> None:
        gw = _FakeGateway("Boiling\nFilm boiling\nNucleate boiling")
        r = GatewayReformulator(gw, model="m")
        assert await r.reformulate(BOND_Q, limit=2) == ["Boiling", "Film boiling"]


# ── _build_reformulator (composition-root selection) ────────────────────────


class TestBuildReformulator:
    def test_no_gateway_is_none(self) -> None:
        from types import SimpleNamespace

        assert _builder()(SimpleNamespace(gateway=None), make_snapshot()) is None

    def test_null_gateway_is_none(self) -> None:
        """The fresh-install / depth-0 reality: an LLM-shaped hole (S5)."""
        from types import SimpleNamespace

        assert _builder()(SimpleNamespace(gateway=NullGateway()), make_snapshot()) is None

    def test_default_arm_is_exemplified(self) -> None:
        from types import SimpleNamespace

        r = _builder()(SimpleNamespace(gateway=_FakeGateway()), make_snapshot())
        assert r is not None and r.prompt == REFORMULATE_SYSTEM_PROMPT

    def test_minimal_arm_and_tokens_from_settings(self) -> None:
        from types import SimpleNamespace

        sn = make_snapshot(
            **{
                "answer.reformulate.prompt_variant": "minimal",
                "answer.reformulate.max_tokens": 96,
            }
        )
        r = _builder()(SimpleNamespace(gateway=_FakeGateway()), sn)
        assert r is not None and r.prompt == MINIMAL_REFORMULATE_SYSTEM_PROMPT


# ── _maybe_reformulate_round0 (the never-worse / S4 contract) ───────────────


class _Conf:
    def __init__(self, top: float) -> None:
        self.top_score = top


@dataclass
class _Card:
    zim_id: int
    path: str
    title: str


@dataclass
class _InnerPassage:
    zim_id: int
    path: str


@dataclass
class _Passage:
    passage: _InnerPassage
    score: float = 0.0


@dataclass
class _Result:
    confidence: _Conf
    cards: list[_Card]
    passages: list[_Passage]

    @classmethod
    def of(cls, top: float, cards: list[_Card] | None = None) -> _Result:
        return cls(_Conf(top), cards or [], [])

    def passage_for(self, card: _Card) -> _Passage:
        sp = _Passage(_InnerPassage(card.zim_id, card.path))
        self.passages.append(sp)
        return sp


def _card(n: int, zim_id: int = 1) -> _Card:
    return _Card(zim_id, f"A/{n}", f"Article {n}")


class _FakeReformulator:
    def __init__(self, queries: list[str] | None = None, error: bool = False) -> None:
        self.queries = queries if queries is not None else ["Boiling"]
        self.error = error
        self.calls: list[tuple[str, int]] = []

    async def reformulate(self, question: str, *, limit: int = 1) -> list[str]:
        self.calls.append((question, limit))
        if self.error:
            raise RuntimeError("llm dead")
        return self.queries


@pytest.fixture()
def research(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Fake run_pipeline: records calls, pops per-test results off ``results``.

    A queued result is consumed per re-search; an empty queue yields the
    default "second search" (new cards 1 and 2, passage for card 1).
    """
    from types import SimpleNamespace

    import vesta.api.answer as mod

    calls: list[dict[str, Any]] = []
    results: list[Any] = []

    async def fake_run_pipeline(*, profile: Any, query: str, scope: Any, deps: Any) -> Any:
        calls.append({"query": query})
        if results:
            return results.pop(0)
        r = _Result.of(0.6, [_card(1), _card(2)])
        r.passage_for(_card(1))
        return r

    monkeypatch.setattr(mod, "run_pipeline", fake_run_pipeline)
    return SimpleNamespace(calls=calls, results=results)


@pytest.mark.asyncio
async def test_gate_never_fires_on_healthy_round0(research: Any) -> None:
    """S4: top_score at/above the trigger touches neither the gateway nor the
    pipeline — the phase's healthy-question latency proof."""
    result = _Result.of(0.5, [_card(0)])
    ref = _FakeReformulator()
    out, appended = await _mrr()(
        result,
        query=BOND_Q,
        profile="P",
        scope="S",
        deps="D",
        reformulator=ref,
        sn=make_snapshot(**{"answer.reformulate.enabled": True}),
    )
    assert out is result and appended == ()
    assert ref.calls == [] and research.calls == []


@pytest.mark.parametrize(
    ("ref_factory", "sn_factory", "queued_result"),
    [
        (_FakeReformulator, lambda: make_snapshot(**{"answer.reformulate.enabled": False}), None),
        (lambda: None, lambda: make_snapshot(**{"answer.reformulate.enabled": True}), None),
        (object, lambda: None, None),
        (
            lambda: _FakeReformulator(error=True),
            lambda: make_snapshot(**{"answer.reformulate.enabled": True}),
            None,
        ),
        (
            lambda: _FakeReformulator(queries=[]),
            lambda: make_snapshot(**{"answer.reformulate.enabled": True}),
            None,
        ),
        (
            lambda: _FakeReformulator(queries=["Boiling"]),
            lambda: make_snapshot(**{"answer.reformulate.enabled": True}),
            _Result.of(0.9, []),
        ),
    ],
)
@pytest.mark.asyncio
async def test_never_worse_fallbacks_keep_original(
    research: Any, ref_factory: Any, sn_factory: Any, queued_result: Any
) -> None:
    """Never-worse contract: disabled setting, missing reformulator or snapshot,
    gateway failure, stagnant output, or non-improving re-search all preserve original."""
    if queued_result is not None:
        research.results.append(queued_result)
    result = _Result.of(0.05, [_card(0)])
    out, appended = await _mrr()(
        result,
        query=BOND_Q,
        profile="P",
        scope="S",
        deps="D",
        reformulator=ref_factory(),
        sn=sn_factory(),
    )
    assert out is result and appended == ()


@pytest.mark.asyncio
async def test_fires_below_trigger_appends_new_cards(research: Any) -> None:
    """Union-append: base cards keep their order at the front, the re-search's
    NEW cards join at the tail, duplicates drop, and only the appended cards'
    passages join. Confidence stays the base query's (it describes the base
    retrieval; the appends are extra articles, not a re-score)."""
    base = _Result.of(0.05, [_card(0)])
    base.passage_for(_card(0))
    second = _Result.of(0.6, [_card(0), _card(1)])  # 0 already present → dedup
    second.passage_for(_card(0))
    second.passage_for(_card(1))
    research.results.append(second)
    ref = _FakeReformulator(queries=["Boiling"])
    out, appended = await _mrr()(
        base,
        query=BOND_Q,
        profile="P",
        scope="S",
        deps="D",
        reformulator=ref,
        sn=make_snapshot(
            **{
                "answer.reformulate.enabled": True,
                "answer.reformulate.trigger_score": 0.25,
            }
        ),
    )
    assert ref.calls == [(BOND_Q, 1)]
    assert [c["query"] for c in research.calls] == ["Boiling"]
    assert [c.path for c in out.cards] == ["A/0", "A/1"]
    assert [c.path for c in appended] == ["A/1"]
    assert [(p.passage.zim_id, p.passage.path) for p in out.passages] == [(1, "A/0"), (1, "A/1")]
    assert out.confidence is base.confidence  # untouched


@pytest.mark.asyncio
async def test_append_happens_even_when_research_scores_lower(research: Any) -> None:
    """The measured kill-shot of the plan's replace rule: a re-search can
    score LOWER than the base yet still carry a new relevant article — and a
    re-search scoring HIGHER against the wrong article must not displace the
    base. Union-append makes both safe (never-worse by construction)."""
    base = _Result.of(0.05, [_card(0)])
    research.results.append(_Result.of(0.03, [_card(1)]))
    out, appended = await _mrr()(
        base,
        query=BOND_Q,
        profile="P",
        scope="S",
        deps="D",
        reformulator=_FakeReformulator(queries=["Boiling"]),
        sn=make_snapshot(**{"answer.reformulate.enabled": True}),
    )
    assert [c.path for c in out.cards] == ["A/0", "A/1"]
    assert [c.path for c in appended] == ["A/1"]


@pytest.mark.asyncio
async def test_append_capped_at_max(research: Any) -> None:
    """A re-search's full card list floods nothing: at most
    ``_REFORM_MAX_APPEND`` new cards join the union."""
    from vesta.api.answer import _REFORM_MAX_APPEND

    base = _Result.of(0.05, [_card(0)])
    research.results.append(_Result.of(0.9, [_card(i) for i in range(_REFORM_MAX_APPEND + 7)]))
    out, appended = await _mrr()(
        base,
        query=BOND_Q,
        profile="P",
        scope="S",
        deps="D",
        reformulator=_FakeReformulator(queries=["Boiling"]),
        sn=make_snapshot(**{"answer.reformulate.enabled": True}),
    )
    assert len(appended) == _REFORM_MAX_APPEND
    assert len(out.cards) == 1 + _REFORM_MAX_APPEND
    assert out.cards[0].path == "A/0"  # base stays in front


@pytest.mark.asyncio
async def test_max_queries_two_runs_both_in_order(research: Any) -> None:
    ref = _FakeReformulator(queries=["Boiling", "Heat transfer"])
    first = _Result.of(0.6, [_card(1)])
    second = _Result.of(0.7, [_card(1), _card(2)])  # 1 already appended → dedup
    research.results.extend([first, second])
    out, appended = await _mrr()(
        _Result.of(0.05, [_card(0)]),
        query=BOND_Q,
        profile="P",
        scope="S",
        deps="D",
        reformulator=ref,
        sn=make_snapshot(
            **{"answer.reformulate.enabled": True, "answer.reformulate.max_queries": 2}
        ),
    )
    assert ref.calls == [(BOND_Q, 2)]
    assert [c["query"] for c in research.calls] == ["Boiling", "Heat transfer"]
    assert [c.path for c in out.cards] == ["A/0", "A/1", "A/2"]
    assert [c.path for c in appended] == ["A/1", "A/2"]
