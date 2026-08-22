"""Pre-seed mechanisms.

In-process (no HTTP, no model server): the tool runtime is a fake returning
controllable passages and the model is a ``FunctionModel`` stub — the same
recipe as tests/test_agent_economy.py.

Covers:
* P2 ``answer.agent.preseed_order=idf`` (default) re-orders the Round-0 pre-seed
  text by focus.py's IDF question-term score (stable, retrieval rank the tiebreak)
  BEFORE the ``preseed_passages`` slice — while card numbers stay discovery-order
  and the compact search-tool branch keeps rank order;
* P4a archive-id rendering (default False omits the (archive-N) token);
* 12-passage slice validation;
* P7 coverage-gated exact search (default True);
* P10 evidence directive calibration (default strong appends must-state clause
  at or above 0.85).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart

from vesta.answer.tokens import estimate_tokens_for_chars
from vesta.api import agent_chat
from vesta.config.settings import SettingsSnapshot, all_settings
from vesta.retrieval.contracts import ScoredPassage, SourceCard
from vesta.zim.types import Passage

# ── Fixtures (lightweight dummy state) ───────────────────────────────────────


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

    def __init__(self, passages: list[ScoredPassage], cards: list[SourceCard]):
        from vesta.answer.tools import SearchToolResult

        self._result = SearchToolResult(
            text="formatted", passages=tuple(passages), cards=tuple(cards)
        )
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
        return "article"


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, runtime: FakeToolRuntime) -> None:
    monkeypatch.setattr(agent_chat, "_build_tool_runtime", lambda *a, **k: runtime)


def _stub_model() -> Any:
    def _fn(messages: list[ModelMessage], info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="stub answer [1]")])

    from pydantic_ai.models.function import FunctionModel

    return FunctionModel(function=_fn)


#: The question whose terms drive the IDF scores below. "xenolith" appears in
#: exactly ONE passage (rank 7 of 8); the filler never contains it.
_QUESTION = "Where did the xenolith sample originate?"

_FILLER = "The garden had many colorful flowers blooming in springtime every year."


def _passages(n: int = 8) -> list[ScoredPassage]:
    passages = [_scored(i, f"PASSAGE-{i}: {_FILLER}") for i in range(n)]
    passages[6] = _scored(6, "PASSAGE-6: the xenolith sample was collected here")
    return passages


def _cards(n: int = 8) -> list[SourceCard]:
    return [_card(i, "snippet") for i in range(n)]


async def _seed(
    state: Any,
    monkeypatch: pytest.MonkeyPatch,
    sn: SettingsSnapshot,
    passages: list[ScoredPassage] | None = None,
) -> str:
    """Run the Round-0 pre-seed over the given (default: fixed) result set."""
    ps = _passages() if passages is None else passages
    _patch_runtime(monkeypatch, FakeToolRuntime(ps, _cards(len(ps))))
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: _stub_model())
    ctx = await agent_chat._build_turn(state, sn, _QUESTION)
    return ctx.seed_text


def _body(seed: str, n: int = 0) -> str:
    """The rendered body of pre-seed passage ``n``: the lines between its
    ``[n+1] "Title"`` header (whatever suffix it carries) and the next
    passage's header."""
    header = f'[{n + 1}] "Article {n}"'
    start = seed.index("\n", seed.index(header)) + 1
    end = seed.index(f"\n[{n + 2}] ")
    return seed[start:end].strip("\n")


async def test_default_idf_order_re_orders_text_but_not_card_numbers(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Registered default (idf): the rare-term passage (retrieval rank 7, outside
    the rank top-6 slice) is re-ordered INTO the pre-seed and rendered FIRST — but its
    card stays [7]: cards are numbered by discovery order over result.cards,
    so the citation contract never shifts."""
    seed = await _seed(state, monkeypatch, make_snapshot())

    # The gold passage entered the pre-seed and leads it.
    assert "PASSAGE-6" in seed
    assert seed.index("PASSAGE-6") < seed.index("PASSAGE-0")
    # The slice still respects preseed_passages (6): one rank filler dropped.
    assert "PASSAGE-5" not in seed
    assert "PASSAGE-4" in seed
    # Card numbers unchanged: discovery order — passage 6 is [7], not [1].
    assert '[7] "Article 6"' in seed
    assert '[1] "Article 0"' in seed
    assert '[1] "Article 6"' not in seed
    # Stable tiebreak: equal-scored passages keep retrieval rank order.
    assert seed.index("PASSAGE-0") < seed.index("PASSAGE-1")
    # P4a default: archive ids omitted
    assert "(archive-" not in seed


async def test_rank_order_override(state: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit preseed_order=rank restores retrieval rank order."""
    sn = make_snapshot(**{"answer.agent.preseed_order": "rank"})
    seed = await _seed(state, monkeypatch, sn)

    assert "PASSAGE-6" not in seed  # top-6 by rank; the rare-term passage is out
    assert seed.index("PASSAGE-0") < seed.index("PASSAGE-5")  # rank order


async def test_idf_order_leaves_compact_search_branch_alone(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The follow-up ``search`` tool (compact=True) always keeps rank order —
    idf re-orders only what the Round-0 pre-seed shows."""
    _patch_runtime(monkeypatch, FakeToolRuntime(_passages(), _cards()))
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: _stub_model())
    sn = make_snapshot()
    ctx = await agent_chat._build_turn(state, sn, _QUESTION)

    text = await ctx._do_search(_QUESTION, compact=True)

    assert text.index("PASSAGE-0") < text.index("PASSAGE-5")  # rank order
    assert "PASSAGE-6" not in text  # compact keeps the rank slice, not idf
    assert "(showing top 6):" in text  # compact budget, not the pre-seed's


# ── P4a: preseed_show_archive_id ────────────────────────────────────────────


async def test_archive_id_rendering(state: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default omits suffix everywhere; setting True includes (archive-N)."""
    _patch_runtime(monkeypatch, FakeToolRuntime(_passages(), _cards()))
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: _stub_model())
    sn = make_snapshot()
    ctx = await agent_chat._build_turn(state, sn, _QUESTION)

    assert "(archive-" not in ctx.seed_text
    assert '[1] "Article 0"' in ctx.seed_text  # the numbered-title shape stays

    compact = await ctx._do_search(_QUESTION, compact=True)
    assert "(archive-" not in compact
    assert '[1] "Article 0"' in compact

    sn_with_archive = make_snapshot(**{"answer.agent.preseed_show_archive_id": True})
    ctx_with_archive = await agent_chat._build_turn(state, sn_with_archive, _QUESTION)
    assert "(archive-1)" in ctx_with_archive.seed_text


def test_preseed_passages_bound_accepts_twelve() -> None:
    """The P3 sweep's 12-passage arm is forceable: ``--set
    answer.agent.preseed_passages=12`` validates through the exact
    ``resolve_value`` path the flag uses — and 13 is still out of bounds."""
    from vesta.answer import ANSWER_AGENT_PRESEED_PASSAGES
    from vesta.config.settings import resolve_value

    key = ANSWER_AGENT_PRESEED_PASSAGES.key
    assert resolve_value(ANSWER_AGENT_PRESEED_PASSAGES, db_values={key: "12"}, env={}) == 12
    with pytest.raises(ValueError, match=r"13 > max 12"):
        resolve_value(ANSWER_AGENT_PRESEED_PASSAGES, db_values={key: "13"}, env={})


async def test_preseed_shows_twelve_passages(state: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """12 flows through ``_do_search``'s slice: over a 14-passage result set
    the seed shows the top 12 (the plan's P3 12-passage arm)."""
    sn = make_snapshot(**{"answer.agent.preseed_passages": 12})
    seed = await _seed(state, monkeypatch, sn, passages=_passages(14))

    assert "(showing top 12):" in seed
    assert "PASSAGE-11" in seed
    assert "PASSAGE-13" not in seed


# ── P7: coverage-gated exact search ─────────────────────────────────────────


class CoverageRuntime(FakeToolRuntime):
    def __init__(
        self,
        initial: tuple[list[ScoredPassage], list[SourceCard]],
        results: dict[str, tuple[list[ScoredPassage], list[SourceCard]]],
    ) -> None:
        super().__init__(*initial)
        from vesta.answer.tools import SearchToolResult

        self._results = {
            query: SearchToolResult(text="formatted", passages=tuple(passages), cards=tuple(cards))
            for query, (passages, cards) in results.items()
        }

    async def search_exact(self, query: str, scope: str) -> Any:
        self.search_exact_calls.append(query)
        return self._results.get(query, self._result)


_COVERAGE_QUESTION = "Where did Ada Lovelace meet Grace Hopper?"


@pytest.mark.parametrize(
    (
        "coverage_on",
        "coverage_dict",
        "initial_text",
        "expected_searches",
        "expected_added",
        "expected_in_seed",
    ),
    [
        (
            False,
            {"Grace Hopper": ([_scored(1, "GRACE COVERAGE RESULT")], [_card(1, "Grace Hopper")])},
            "Ada Lovelace appears in the first result.",
            [_COVERAGE_QUESTION],
            None,
            False,
        ),
        (
            True,
            {},
            "Ada Lovelace and Grace Hopper are both mentioned here.",
            [_COVERAGE_QUESTION],
            0,
            False,
        ),
        (
            True,
            {"Grace Hopper": ([_scored(1, "GRACE COVERAGE RESULT")], [_card(1, "Grace Hopper")])},
            "Ada Lovelace appears in the first result.",
            [_COVERAGE_QUESTION, "Grace Hopper"],
            1,
            True,
        ),
    ],
)
async def test_coverage_search_behavior(  # noqa: PLR0917
    state: Any,
    monkeypatch: pytest.MonkeyPatch,
    coverage_on: bool,
    coverage_dict: dict[str, Any],
    initial_text: str,
    expected_searches: list[str],
    expected_added: int | None,
    expected_in_seed: bool,
) -> None:
    runtime = CoverageRuntime(
        ([_scored(0, initial_text)], [_card(0, "Ada Lovelace")]),
        coverage_dict,
    )
    _patch_runtime(monkeypatch, runtime)
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: _stub_model())

    ctx = await agent_chat._build_turn(
        state,
        make_snapshot(
            **{
                "answer.agent.coverage_search": coverage_on,
                "answer.agent.coverage_search_max": 1,
            }
        ),
        _COVERAGE_QUESTION,
    )

    assert runtime.search_exact_calls == expected_searches
    if expected_in_seed:
        assert "GRACE COVERAGE RESULT" in ctx.seed_text
        assert [(card.n, card.title) for card in ctx.turn_cards.values()] == [
            (1, "Article 0"),
            (2, "Article 1"),
        ]
    else:
        assert "GRACE COVERAGE RESULT" not in ctx.seed_text
    if expected_added is not None:
        step = next(step for step in ctx.steps if step["name"] == "coverage_search")
        assert step["outputs"]["passages_added"] == expected_added
    elif not coverage_on:
        assert [step["name"] for step in ctx.steps] == ["pre_seed"]


async def test_coverage_search_merges_before_the_existing_fit_tail_drop(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enriched passages use the normal order/slice/fit path; the 8k fitter
    still drops tail evidence rather than bypassing its window guarantee."""
    initial = [
        _scored(i, f"INITIAL-{i} " + ("Ada Lovelace " if i == 0 else "") + "filler. " * 1_200)
        for i in range(6)
    ]
    runtime = CoverageRuntime(
        (initial, [_card(i, "Ada Lovelace") for i in range(6)]),
        {
            "Grace Hopper": (
                [_scored(6, "TAIL COVERAGE " + "filler. " * 1_200)],
                [_card(6, "Grace Hopper")],
            )
        },
    )
    _patch_runtime(monkeypatch, runtime)
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: _stub_model())
    ctx = await agent_chat._build_turn(
        state,
        make_snapshot(
            **{
                "answer.agent.context_profile": "8k",
                "answer.agent.preseed_passages": 12,
                "answer.agent.preseed_passage_max_chars": 4_000,
                "answer.agent.coverage_search": True,
            }
        ),
        _COVERAGE_QUESTION,
    )

    assert ctx.preseed_dropped > 0
    assert "TAIL COVERAGE" not in ctx.seed_text
    assert [(card.n, card.title) for card in ctx.turn_cards.values()][-1] == (7, "Article 6")


# ── P10: strong-evidence must-state clause ──────────────────────────────────


def _weak_cards() -> list[SourceCard]:
    """Cards whose cross-encoder score (0.5) sits below the P10 floor (0.85),
    the shape of an adversarial question's best match."""
    return [
        SourceCard(
            zim_id=1,
            path=f"a/{i}",
            title=f"Article {i}",
            snippet="weak",
            breadcrumb=f"Article {i} > Section",
            score=0.5,
            source="test",
        )
        for i in range(8)
    ]


@pytest.mark.parametrize(
    (
        "mode",
        "use_weak_cards",
        "empty_seed",
        "expected_fired",
        "expected_top_score",
        "has_clause",
        "has_directive",
    ),
    [
        ("standard", False, False, False, 10.0, False, True),
        ("strong", False, False, True, 10.0, True, True),
        ("strong", True, False, False, 0.5, False, True),
        ("strong", False, True, False, None, False, False),
    ],
)
async def test_evidence_directive_modes(  # noqa: PLR0917
    state: Any,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    use_weak_cards: bool,
    empty_seed: bool,
    expected_fired: bool,
    expected_top_score: float | None,
    has_clause: bool,
    has_directive: bool,
) -> None:
    """Verifies evidence directive modes (standard vs strong) and threshold calibration."""
    if empty_seed:
        _patch_runtime(monkeypatch, FakeToolRuntime([], []))
    else:
        cards = _weak_cards() if use_weak_cards else _cards()
        _patch_runtime(monkeypatch, FakeToolRuntime(_passages(), cards))
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: _stub_model())

    ctx = await agent_chat._build_turn(
        state,
        make_snapshot(**{"answer.agent.evidence_directive": mode}),
        _QUESTION,
    )

    assert ctx.evidence_directive_trace == {
        "mode": mode,
        "top_score": expected_top_score,
        "fired": expected_fired,
    }
    if has_directive:
        assert agent_chat._STRONG_EVIDENCE_DIRECTIVE in ctx.sys_prompt
    else:
        assert agent_chat._STRONG_EVIDENCE_DIRECTIVE not in ctx.sys_prompt
    if has_clause:
        assert agent_chat._STRONG_EVIDENCE_MUST_STATE_CLAUSE in ctx.sys_prompt
    else:
        assert agent_chat._STRONG_EVIDENCE_MUST_STATE_CLAUSE not in ctx.sys_prompt


async def test_evidence_directive_strong_8k_turn1_still_fits(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The clause length is reserved before the pre-seed fit, so a
    deliberately fat 8k Round-0 request with the clause appended still fits
    the window by construction."""
    passages = [_scored(i, f"PASSAGE-{i} " + "seed filler. " * 800) for i in range(14)]
    _patch_runtime(monkeypatch, FakeToolRuntime(passages, _cards(len(passages))))
    monkeypatch.setattr(agent_chat, "_make_model", lambda *a, **k: _stub_model())
    ctx = await agent_chat._build_turn(
        state,
        make_snapshot(
            **{
                "answer.agent.context_profile": "8k",
                "answer.agent.preseed_passages": 12,
                "answer.agent.evidence_directive": "strong",
            }
        ),
        _QUESTION,
    )

    assert ctx.evidence_directive_trace["fired"] is True  # top_score 10.0 — clause appended
    assert agent_chat._STRONG_EVIDENCE_MUST_STATE_CLAUSE in ctx.sys_prompt
    assert ctx.preseed_dropped > 0
    assert estimate_tokens_for_chars(len(ctx.sys_prompt) + len(ctx.user_message)) <= (
        ctx.budget.window_tokens - ctx.budget.output_reserve
    )
