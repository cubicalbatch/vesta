"""Tests for the four ``context_assembler`` strategies:
``topk_budget``, ``section_window``, ``lead_boost``, ``diverse``."""

from __future__ import annotations

import random
from itertools import pairwise
from typing import Any

import pytest

from vesta.retrieval.assemblers._shared import (
    apply_budget,
    apply_ordering,
    build_result,
)
from vesta.retrieval.assemblers.diverse import Diverse
from vesta.retrieval.assemblers.lead_boost import LeadBoost
from vesta.retrieval.assemblers.section_window import SectionWindow
from vesta.retrieval.assemblers.topk_budget import TopKBudget
from vesta.retrieval.contracts import Budget, PreparedQuery, ScoredPassage
from vesta.retrieval.dedup import DEFAULT_THRESHOLD
from vesta.retrieval.registry import resolve
from vesta.zim.types import Passage


def _sp(
    text: str,
    *,
    path: str = "A",
    score: float = 1.0,
    ordinal: int = 0,
    breadcrumb: str = "",
    is_lead: bool = False,
    zim_id: int = 1,
) -> ScoredPassage:
    p = Passage(
        zim_id=zim_id,
        path=path,
        ordinal=ordinal,
        char_start=0,
        char_end=len(text),
        breadcrumb=breadcrumb,
        text=text,
        is_lead=is_lead,
    )
    return ScoredPassage(passage=p, score=score, source_info="x")


def _pq(text: str = "test query") -> PreparedQuery:
    return PreparedQuery(
        raw=text,
        terms=tuple(text.split()),
        text=text,
        aliases=(),
        is_keyword_query=True,
        rung="initial",
    )


_BIG_BUDGET = Budget(token_total=100_000, max_per_article=100)


def test_all_four_assemblers_registered() -> None:
    assert resolve("context_assembler", "topk_budget") is TopKBudget
    assert resolve("context_assembler", "section_window") is SectionWindow
    assert resolve("context_assembler", "lead_boost") is LeadBoost
    assert resolve("context_assembler", "diverse") is Diverse


@pytest.mark.parametrize("assembler_cls", [TopKBudget, SectionWindow, LeadBoost, Diverse])
def test_assemblers_handle_empty_input(assembler_cls: Any) -> None:
    a = assembler_cls(params=assembler_cls.Params())
    result = a.assemble([], _BIG_BUDGET, _pq(), tr=None)  # type: ignore[arg-type]
    assert result.passages == ()
    assert result.cards == ()
    assert result.confidence.top_score is None


# ── topk_budget ───────────────────────────────────────────────────────────────


def test_topk_budget_orders_by_score_descending() -> None:
    a = TopKBudget(params=TopKBudget.Params(max_per_article=5))
    passages = [
        _sp("low relevance text here", path="A", score=0.2),
        _sp("high relevance text here", path="B", score=0.9),
    ]
    result = a.assemble(passages, _BIG_BUDGET, _pq(), tr=None)  # type: ignore[arg-type]
    assert result.passages[0].passage.path == "B"


def test_topk_budget_enforces_per_article_cap() -> None:
    a = TopKBudget(params=TopKBudget.Params(max_per_article=1))
    passages = [
        _sp("first passage text about topic one here", path="A", ordinal=0, score=0.9),
        _sp("second passage text about topic two here", path="A", ordinal=1, score=0.8),
    ]
    result = a.assemble(passages, _BIG_BUDGET, _pq(), tr=None)  # type: ignore[arg-type]
    assert len(result.passages) == 1


def test_topk_budget_stops_at_token_budget() -> None:
    a = TopKBudget(params=TopKBudget.Params(budget_tokens=5, max_per_article=10, dedup="none"))
    passages = [
        _sp("one two three four five six seven eight", path="A", score=0.9),
        _sp("nine ten eleven twelve thirteen fourteen", path="B", score=0.8),
    ]
    result = a.assemble(passages, _BIG_BUDGET, _pq(), tr=None)  # type: ignore[arg-type]
    # first passage alone already exceeds the 5-token budget, but at least one
    # passage is always kept (an empty result would be worse than over-budget).
    assert len(result.passages) == 1


def test_topk_budget_edges_ordering_alternates() -> None:
    a = TopKBudget(params=TopKBudget.Params(max_per_article=10, ordering="edges"))
    passages = [
        _sp("passage number one here today", path="A", score=0.9),
        _sp("passage number two here today", path="B", score=0.8),
        _sp("passage number three here today", path="C", score=0.7),
        _sp("passage number four here today", path="D", score=0.6),
    ]
    result = a.assemble(passages, _BIG_BUDGET, _pq(), tr=None)  # type: ignore[arg-type]
    paths = [sp.passage.path for sp in result.passages]
    # rank1 first, rank2 last, rank3 second, rank4 second-to-last.
    assert paths == ["A", "C", "D", "B"]


def test_topk_budget_builds_cards_with_snippets() -> None:
    a = TopKBudget(params=TopKBudget.Params(max_per_article=5))
    passages = [_sp("some passage text about the target term here", path="A", score=0.9)]
    result = a.assemble(passages, _BIG_BUDGET, _pq("target"), tr=None)  # type: ignore[arg-type]
    assert len(result.cards) == 1
    assert result.cards[0].snippet


def test_topk_budget_score_floor_defaults_to_disabled() -> None:
    """``score_floor_abs``/``score_floor_rel`` default to ``None`` — behaviour
    must be byte-identical to before the floor was added when not opted in,
    even for a score-0.0000 tail passage the floor would otherwise prune."""
    a = TopKBudget(params=TopKBudget.Params(max_per_article=1, budget_tokens=100_000))
    passages = [
        _sp(f"article {i} passage text here today", path=str(i), score=0.0) for i in range(20)
    ]
    result = a.assemble(passages, _BIG_BUDGET, _pq(), tr=None)  # type: ignore[arg-type]
    assert len(result.passages) == 20  # nothing pruned — the floor is off


def test_topk_budget_score_floor_prunes_near_zero_tail() -> None:
    """The motivating bug: a small archive where the token budget never binds,
    so every candidate the funnel returned — including score-0.0000 tag pages
    — reaches the prompt. With the floor enabled, only the top few (plus the
    ``min_articles`` guarantee) survive."""
    a = TopKBudget(
        params=TopKBudget.Params(
            max_per_article=1,
            budget_tokens=100_000,
            score_floor_abs=0.02,
            score_floor_rel=0.02,
            min_articles=3,
        )
    )
    passages = [_sp("the target article text here today", path="target", score=1.14)]
    passages += [
        _sp(f"tag page {i} noise text here today", path=f"tag{i}", score=0.0) for i in range(30)
    ]
    result = a.assemble(passages, _BIG_BUDGET, _pq(), tr=None)  # type: ignore[arg-type]
    paths = [sp.passage.path for sp in result.passages]
    assert paths[0] == "target"
    # min_articles=3 guarantees at least 3 distinct articles survive even
    # though every tag page scores 0.0 (well below both the abs and rel
    # floor), but the flood of 30 near-zero tag pages is gone.
    assert len(result.passages) == 3
    assert len(result.passages) < len(passages)


# ── apply_budget score floor (FIX 4) ─────────────────────────────────────────
#
# ``apply_budget`` (retrieval/assemblers/_shared.py) is the shared selection
# core behind ``topk_budget`` and ``lead_boost``. Tested directly here so the
# floor/no-starvation/no-op-under-budget properties are pinned independent of
# any one assembler's other behaviour (boosting, ordering, etc).


def test_apply_budget_floor_is_noop_when_token_budget_binds_first() -> None:
    """On a large archive the token budget truncates the ranked list long
    before ``min_articles`` distinct articles are ever selected — so the
    floor's tail-pruning branch (which only activates once ``min_articles``
    is reached) never fires at all, and an aggressive floor changes nothing.
    ``min_articles=100`` here is deliberately unreachable within the tiny
    budget, isolating exactly that: the budget breaks the loop first, every
    time, so the floor's own guard is never the reason anything is excluded."""
    top = _sp("a" * 200, path="top", score=0.9)  # ~1 "token" (single long word)
    tail = [_sp(f"low score passage number {i} here", path=f"p{i}", score=0.01) for i in range(10)]
    ranked = [top, *tail]

    without_floor = apply_budget(ranked, token_budget=3, max_per_article=10, dedup_threshold=None)
    with_floor = apply_budget(
        ranked,
        token_budget=3,
        max_per_article=10,
        dedup_threshold=None,
        score_floor_abs=0.5,
        score_floor_rel=0.5,
        min_articles=100,  # unreachable — the budget always breaks first
    )
    assert without_floor == with_floor
    # Confirm the budget really is what bound (only the always-kept first
    # passage survives), so this is actually exercising the "budget binds
    # first" regime and not vacuously passing on an empty result either way.
    assert [sp.passage.path for sp in without_floor] == ["top"]


def test_apply_budget_min_articles_prevents_starvation() -> None:
    """Every candidate scores below the floor, and the token budget is
    generous (does not bind) — the floor must still stop pruning once
    ``min_articles`` distinct articles are selected, never starving the
    context down to zero or near-zero."""
    ranked = [
        _sp(f"near zero score passage {i} here today", path=f"p{i}", score=0.001) for i in range(10)
    ]
    selected = apply_budget(
        ranked,
        token_budget=100_000,
        max_per_article=1,
        dedup_threshold=None,
        score_floor_abs=0.5,  # every candidate (0.001) is far below this
        score_floor_rel=0.0,
        min_articles=8,
    )
    assert len(selected) == 8  # guaranteed floor, not starved to 0 or 1
    # Without the floor, all 10 would have survived — confirming it actually
    # pruned the tail beyond the guarantee, not merely no-opped.
    unfiltered = apply_budget(ranked, token_budget=100_000, max_per_article=1, dedup_threshold=None)
    assert len(unfiltered) == 10


def test_apply_budget_floor_uses_top_score_of_ranked_input() -> None:
    """The relative floor is computed from ``ranked[0].score`` — the top score
    of the INPUT, not recomputed against whatever ends up selected."""
    ranked = [
        _sp("top scoring passage here today", path="top", score=1.0),
        _sp(
            "mid scoring passage here today", path="mid", score=0.10
        ),  # 10% of top: at the rel floor
        _sp("low scoring passage here today", path="low", score=0.02),  # 2% of top: below it
    ]
    selected = apply_budget(
        ranked,
        token_budget=100_000,
        max_per_article=1,
        dedup_threshold=None,
        score_floor_abs=0.0,
        score_floor_rel=0.10,
        min_articles=1,
    )
    paths = [sp.passage.path for sp in selected]
    assert "top" in paths  # always kept — min_articles=1 already satisfied by it
    assert "mid" in paths  # 0.10 >= 0.10 * 1.0
    assert "low" not in paths  # 0.02 < 0.10 * 1.0


# ── section_window ────────────────────────────────────────────────────────────


def test_section_window_pulls_in_same_section_neighbours() -> None:
    a = SectionWindow(params=SectionWindow.Params(max_per_article=5, window=1))
    passages = [
        _sp(
            "winning sentence about the topic here",
            path="A",
            ordinal=5,
            score=0.9,
            breadcrumb="Art > Sec",
        ),
        _sp(
            "neighbour before the winner sentence",
            path="A",
            ordinal=4,
            score=0.1,
            breadcrumb="Art > Sec",
        ),
        _sp(
            "neighbour after the winner sentence",
            path="A",
            ordinal=6,
            score=0.1,
            breadcrumb="Art > Sec",
        ),
    ]
    result = a.assemble(passages, _BIG_BUDGET, _pq(), tr=None)  # type: ignore[arg-type]
    ordinals = sorted(sp.passage.ordinal for sp in result.passages)
    assert ordinals == [4, 5, 6]


def test_section_window_does_not_cross_section_boundary() -> None:
    # max_per_article=1 admits only the single top-scored winning section, so
    # the assertion isolates window EXPANSION (does it pull in ordinal 4?)
    # from independent-winner admission (would ordinal 4 also qualify on its
    # own?), which a looser cap would conflate.
    a = SectionWindow(params=SectionWindow.Params(max_per_article=1, window=1))
    passages = [
        _sp(
            "winning sentence about the topic here",
            path="A",
            ordinal=5,
            score=0.9,
            breadcrumb="Art > SecTwo",
        ),
        _sp(
            "different section neighbour sentence",
            path="A",
            ordinal=4,
            score=0.1,
            breadcrumb="Art > SecOne",
        ),
    ]
    result = a.assemble(passages, _BIG_BUDGET, _pq(), tr=None)  # type: ignore[arg-type]
    ordinals = [sp.passage.ordinal for sp in result.passages]
    assert ordinals == [5]  # neighbour from a different section excluded


def test_section_window_max_per_article_counts_winning_sections() -> None:
    a = SectionWindow(params=SectionWindow.Params(max_per_article=1, window=1))
    passages = [
        _sp("winner one text here today", path="A", ordinal=0, score=0.9, breadcrumb="Sec1"),
        _sp("winner two text here today", path="A", ordinal=10, score=0.8, breadcrumb="Sec2"),
    ]
    result = a.assemble(passages, _BIG_BUDGET, _pq(), tr=None)  # type: ignore[arg-type]
    # only the first winning section is admitted (max_per_article=1 SECTIONS).
    assert {sp.passage.ordinal for sp in result.passages} == {0}


def test_section_window_card_and_confidence_use_winning_passage_not_first_ordinal() -> None:
    """``expanded`` interleaves same-section neighbours in ordinal order, not
    score order — a lower-ordinal neighbour of the winner can be its FIRST
    occurrence in the sequence. Cards/confidence must not mistake that
    neighbour's score for the article's real (highest) score."""
    a = SectionWindow(params=SectionWindow.Params(max_per_article=5, window=1))
    passages = [
        _sp(
            "winning sentence about the topic here",
            path="A",
            ordinal=5,
            score=0.9,
            breadcrumb="Art > Sec",
        ),
        _sp(
            "neighbour before the winner sentence",
            path="A",
            ordinal=4,
            score=0.1,
            breadcrumb="Art > Sec",
        ),
        _sp(
            "neighbour after the winner sentence",
            path="A",
            ordinal=6,
            score=0.1,
            breadcrumb="Art > Sec",
        ),
    ]
    result = a.assemble(passages, _BIG_BUDGET, _pq(), tr=None)  # type: ignore[arg-type]
    # the expanded sequence really does put the low-score neighbour first —
    # otherwise this test would not be exercising the bug at all.
    assert result.passages[0].score == 0.1
    assert len(result.cards) == 1
    assert result.cards[0].score == 0.9
    assert result.confidence.top_score == 0.9
    # a real dropoff (winner stands well above its neighbours), not the ~1.0
    # a naive scores[0]-as-top computation would report.
    assert result.confidence.score_dropoff is not None
    assert result.confidence.score_dropoff < 0.2


# ── lead_boost ────────────────────────────────────────────────────────────────


def test_lead_boost_lets_lead_passage_win_a_close_race() -> None:
    a = LeadBoost(params=LeadBoost.Params(max_per_article=5, boost=0.5))
    passages = [
        _sp("lead paragraph summary text here", path="A", score=0.60, is_lead=True),
        _sp("body paragraph detail text here", path="B", score=0.65, is_lead=False),
    ]
    result = a.assemble(passages, _BIG_BUDGET, _pq(), tr=None)  # type: ignore[arg-type]
    assert result.passages[0].passage.path == "A"  # 0.60 * 1.5 = 0.90 > 0.65


def test_lead_boost_does_not_affect_non_lead_passages() -> None:
    a = LeadBoost(params=LeadBoost.Params(max_per_article=5, boost=0.5))
    passages = [_sp("body paragraph detail text here", path="B", score=0.65, is_lead=False)]
    result = a.assemble(passages, _BIG_BUDGET, _pq(), tr=None)  # type: ignore[arg-type]
    assert result.passages[0].score == 0.65


@pytest.mark.parametrize(
    ("guarantee_top_lead", "expect_lead_included"),
    [
        (True, True),
        (False, False),
    ],
)
def test_guarantee_top_lead_rescues_a_lead(
    guarantee_top_lead: bool, expect_lead_included: bool
) -> None:
    """Regression case: a lead passage scored far too low for any reasonable
    boost multiplier to rescue via ranking, but its article is the clear top
    match (live-observed, 2026-08-01, "Eminem" article). Can be disabled."""
    a = LeadBoost(
        params=LeadBoost.Params(
            max_per_article=2,
            budget_tokens=100_000,
            boost=0.15,
            guarantee_top_lead=guarantee_top_lead,
        )
    )
    top_article_passages = [
        _sp(f"body passage {i}", path="A", score=0.99 - i * 0.01, ordinal=i + 1) for i in range(5)
    ]
    lead = _sp("lead paragraph summary text here", path="A", score=0.10, ordinal=0, is_lead=True)
    other_article = [_sp("unrelated other article text", path="B", score=0.50, ordinal=0)]
    result = a.assemble([*top_article_passages, lead, *other_article], _BIG_BUDGET, _pq(), tr=None)  # type: ignore[arg-type]

    has_lead = any(sp.passage.path == "A" and sp.passage.ordinal == 0 for sp in result.passages)
    assert has_lead is expect_lead_included
    if guarantee_top_lead:
        # max_per_article=2 is still honoured for every OTHER article/case — the
        # guarantee added exactly one extra passage, not a general cap increase.
        a_count = sum(1 for sp in result.passages if sp.passage.path == "A")
        assert a_count == 3  # the normal top-2 body passages + the guaranteed lead


@pytest.mark.parametrize(
    ("k", "max_per_article", "boost", "expect_b_lead"),
    [
        (1, 1, 0.0, False),
        (3, 2, 0.15, True),
    ],
)
def test_guarantee_top_lead_k_scoping(
    k: int, max_per_article: int, boost: float, expect_b_lead: bool
) -> None:
    """With k=1 only top-1 article gets lead rescue; with default k=3 article #2's
    lead is also rescued."""
    a = LeadBoost(
        params=LeadBoost.Params(
            max_per_article=max_per_article,
            budget_tokens=100_000,
            boost=boost,
            guarantee_top_lead_k=k,
        )
    )
    if k == 1:
        top = [_sp("top article body", path="A", score=0.99, ordinal=0)]
        other_lead = _sp("other article lead", path="B", score=0.05, ordinal=0, is_lead=True)
        other_body = _sp("other article body", path="B", score=0.40, ordinal=1)
        result = a.assemble([*top, other_lead, other_body], _BIG_BUDGET, _pq(), tr=None)  # type: ignore[arg-type]
        paths_ordinals = {(sp.passage.path, sp.passage.ordinal) for sp in result.passages}
        assert ("B", 0) not in paths_ordinals
        assert ("B", 1) in paths_ordinals
    else:
        a_body = _sp("article one body passage", path="A", score=0.999, ordinal=1)
        a_lead = _sp("article one lead infobox junk", path="A", score=0.05, ordinal=0, is_lead=True)
        b_body = _sp("article two body passage", path="B", score=0.95, ordinal=1)
        b_lead = _sp(
            "article two lead with the actual fact here",
            path="B",
            score=0.27,
            ordinal=0,
            is_lead=True,
        )
        result = a.assemble([a_body, a_lead, b_body, b_lead], _BIG_BUDGET, _pq(), tr=None)  # type: ignore[arg-type]
        paths_ordinals = {(sp.passage.path, sp.passage.ordinal) for sp in result.passages}
        assert ("A", 0) in paths_ordinals
        assert ("B", 0) in paths_ordinals
        assert ("A", 1) in paths_ordinals
        assert ("B", 1) in paths_ordinals


@pytest.mark.parametrize(
    ("passages", "params", "expected_paths_ordinals", "expected_total"),
    [
        # No-op when lead already selected
        (
            [_sp("lead paragraph text", path="A", score=0.90, ordinal=0, is_lead=True)],
            LeadBoost.Params(max_per_article=5, budget_tokens=100_000, boost=0.15),
            {("A", 0)},
            1,
        ),
        # At most one extra lead per article
        (
            [
                _sp("the body passage", path="A", score=0.99, ordinal=2),
                _sp("lead text one", path="A", score=0.10, ordinal=0, is_lead=True),
                _sp("lead text two", path="A", score=0.08, ordinal=1, is_lead=True),
            ],
            LeadBoost.Params(
                max_per_article=1, budget_tokens=100_000, boost=0.0, guarantee_top_lead_k=3
            ),
            {("A", 2), ("A", 0)},
            2,
        ),
        # Bypasses score floor
        (
            [
                _sp("top article body passage here", path="A", score=0.99, ordinal=1),
                _sp("top article lead text here", path="A", score=0.05, ordinal=0, is_lead=True),
            ],
            LeadBoost.Params(
                max_per_article=1,
                budget_tokens=100_000,
                boost=0.0,
                guarantee_top_lead=True,
                guarantee_top_lead_k=1,
                score_floor_abs=0.5,
                score_floor_rel=0.0,
                min_articles=1,
            ),
            {("A", 1), ("A", 0)},
            2,
        ),
    ],
)
def test_guarantee_top_lead_invariants(
    passages: list[ScoredPassage],
    params: LeadBoost.Params,
    expected_paths_ordinals: set[tuple[str, int]],
    expected_total: int,
) -> None:
    a = LeadBoost(params=params)
    result = a.assemble(passages, _BIG_BUDGET, _pq(), tr=None)  # type: ignore[arg-type]
    paths_ordinals = {(sp.passage.path, sp.passage.ordinal) for sp in result.passages}
    assert paths_ordinals == expected_paths_ordinals
    assert len(result.passages) == expected_total


# ── exact_title_boost ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("breadcrumb", "fire_score", "other_score", "boost"),
    [
        ("Fire", 0.99, 1.10, 0.15),
        ("  Fire  ", 0.5, 0.9, 0.0),
        ("Fire > Fire", 0.99, 1.10, 0.15),
    ],
)
def test_exact_title_match_ranks_first(
    breadcrumb: str, fire_score: float, other_score: float, boost: float
) -> None:
    a = LeadBoost(params=LeadBoost.Params(max_per_article=5, budget_tokens=100_000, boost=boost))
    fire_lead = _sp(
        "fire is the rapid oxidation of a fuel",
        path="Fire",
        score=fire_score,
        ordinal=0,
        is_lead=True,
        breadcrumb=breadcrumb,
    )
    other = (
        _sp(
            "a fire department is a government agency",
            path="Fire_department",
            score=other_score,
            ordinal=0,
            is_lead=True,
            breadcrumb=(
                "Fire department" if ">" not in breadcrumb else "Fire department > Fire department"
            ),
        )
        if boost > 0
        else _sp("other text", path="Other", score=other_score, ordinal=0, breadcrumb="Other")
    )
    result = a.assemble([fire_lead, other], _BIG_BUDGET, _pq("fire"), tr=None)  # type: ignore[arg-type]
    assert result.passages[0].passage.path == "Fire"


@pytest.mark.parametrize(
    ("candidate", "expected_winner"),
    [
        (
            _sp(
                "a paragraph inside an unrelated article's Fire section",
                path="Other_Article",
                score=0.5,
                ordinal=3,
                is_lead=False,
                breadcrumb="Other Article > Fire",
            ),
            "Other",
        ),
        (
            _sp("body text", path="Fire", score=0.5, ordinal=1, is_lead=False, breadcrumb="Fire"),
            "Other",
        ),
        (
            _sp(
                "lead paragraph summary text here",
                path="A",
                score=0.60,
                is_lead=True,
                breadcrumb="Something Else",
            ),
            "A",
        ),
    ],
)
def test_exact_title_match_non_matching_cases(
    candidate: ScoredPassage, expected_winner: str
) -> None:
    boost = 0.15 if expected_winner == "A" else 0.0
    a = LeadBoost(params=LeadBoost.Params(max_per_article=5, budget_tokens=100_000, boost=boost))
    other = (
        _sp(
            "body paragraph detail text here",
            path="B",
            score=0.65,
            is_lead=False,
            breadcrumb="Other Article",
        )
        if expected_winner == "A"
        else _sp("other text", path="Other", score=0.9, ordinal=0, breadcrumb="Other")
    )
    result = a.assemble([candidate, other], _BIG_BUDGET, _pq("fire"), tr=None)  # type: ignore[arg-type]
    assert result.passages[0].passage.path == expected_winner


def test_exact_title_match_tie_break_prefers_higher_score() -> None:
    """Two archives can hold an identically-titled article (unscoped search)
    — the tie-break is simple: highest (already-boosted) score wins."""
    a = LeadBoost(params=LeadBoost.Params(max_per_article=5, budget_tokens=100_000, boost=0.0))
    weak = _sp(
        "weak fire lead",
        path="Fire",
        score=0.3,
        ordinal=0,
        is_lead=True,
        breadcrumb="Fire",
        zim_id=1,
    )
    strong = _sp(
        "strong fire lead",
        path="Fire",
        score=0.7,
        ordinal=0,
        is_lead=True,
        breadcrumb="Fire",
        zim_id=2,
    )
    other = _sp("other text", path="Other", score=0.9, ordinal=0, breadcrumb="Other")
    result = a.assemble([weak, strong, other], _BIG_BUDGET, _pq("fire"), tr=None)  # type: ignore[arg-type]
    assert result.passages[0].passage.zim_id == 2


def test_exact_title_match_can_be_disabled() -> None:
    a = LeadBoost(
        params=LeadBoost.Params(
            max_per_article=5, budget_tokens=100_000, boost=0.15, exact_title_boost=False
        )
    )
    fire_lead = _sp("fire text", path="Fire", score=0.5, ordinal=0, is_lead=True, breadcrumb="Fire")
    dept = _sp(
        "dept text", path="Fire_department", score=0.9, ordinal=0, breadcrumb="Fire department"
    )
    result = a.assemble([fire_lead, dept], _BIG_BUDGET, _pq("fire"), tr=None)  # type: ignore[arg-type]
    assert result.passages[0].passage.path == "Fire_department"


# ── diverse ───────────────────────────────────────────────────────────────────


def test_diverse_prefers_dissimilar_second_pick_over_near_duplicate() -> None:
    dup_text = "Albert Einstein developed the theory of general relativity in 1915 in Germany"
    a = Diverse(params=Diverse.Params(max_per_article=5, dedup="none", lambda_relevance=0.5))
    passages = [
        _sp(dup_text, path="A", score=0.95),
        _sp(dup_text, path="B", score=0.90),  # near-duplicate of the top pick
        _sp(
            "Quantum computing exploits superposition of qubits for parallelism",
            path="C",
            score=0.70,
        ),
    ]
    result = a.assemble(passages, _BIG_BUDGET, _pq(), tr=None)  # type: ignore[arg-type]
    paths = [sp.passage.path for sp in result.passages]
    assert paths[0] == "A"
    # C (dissimilar, lower raw score) should be preferred over B (near-dup of A)
    # for the second slot under MMR with a real diversity weight.
    assert paths[1] == "C"


def test_diverse_enforces_max_per_archive() -> None:
    a = Diverse(params=Diverse.Params(max_per_article=10, max_per_archive=1))
    passages = [
        _sp("first archive one passage text here", path="A", score=0.9, zim_id=1),
        _sp("second archive one passage text here", path="B", score=0.8, zim_id=1),
        _sp("third archive two passage text here", path="C", score=0.7, zim_id=2),
    ]
    result = a.assemble(passages, _BIG_BUDGET, _pq(), tr=None)  # type: ignore[arg-type]
    zim_ids = [sp.passage.zim_id for sp in result.passages]
    assert zim_ids.count(1) == 1
    assert 2 in zim_ids


# ── AUDIT_0822 P2: cached bigram/word-set paths vs naive references ──────────
#
# The near-dedup and MMR paths now precompute each passage's bigram/word set
# once instead of re-splitting per candidate x selected pair (AUDIT_0822 P2).
# Pure caching — selections must be IDENTICAL to the old logic. The reference
# functions below are that old logic, verbatim (re-tokenizing inside every
# comparison). Randomized corpora mix exact duplicates, near-duplicates around
# the threshold, plain passages, and short/empty edge shapes; quantized scores
# produce ties so both implementations' identical tie-breaking is exercised.


_VOCAB = [
    "time",
    "person",
    "year",
    "way",
    "day",
    "thing",
    "man",
    "world",
    "life",
    "hand",
    "part",
    "child",
    "eye",
    "woman",
    "place",
    "work",
    "week",
    "case",
    "point",
    "government",
    "company",
    "number",
    "group",
    "problem",
    "fact",
]


def _ref_bigrams(text: str) -> frozenset[tuple[str, str]]:
    return frozenset(pairwise(text.lower().split()))


def _ref_is_near_duplicate(
    candidate: ScoredPassage,
    selected: list[ScoredPassage],
    *,
    threshold: float,
) -> bool:
    cand_bg = _ref_bigrams(candidate.passage.text)
    if len(cand_bg) < 4:
        return False
    for other in selected:
        other_bg = _ref_bigrams(other.passage.text)
        if not other_bg:
            continue
        union = cand_bg | other_bg
        if not union:
            continue
        if len(cand_bg & other_bg) / len(union) > threshold:
            return True
    return False


def _ref_similarity(a_text: str, b_text: str) -> float:
    """Old string-based unigram Jaccard — both texts split on every call."""
    wa = frozenset(a_text.lower().split())
    wb = frozenset(b_text.lower().split())
    if not wa or not wb:
        return 0.0
    union = wa | wb
    return len(wa & wb) / len(union) if union else 0.0


def _ref_apply_budget(
    ranked: list[ScoredPassage],
    *,
    token_budget: int,
    max_per_article: int,
    dedup_threshold: float | None,
    score_floor_abs: float | None = None,
    score_floor_rel: float | None = None,
    min_articles: int = 8,
) -> list[ScoredPassage]:
    floor: float | None = None
    if score_floor_abs is not None or score_floor_rel is not None:
        top_score = ranked[0].score if ranked else 0.0
        abs_part = score_floor_abs if score_floor_abs is not None else 0.0
        rel_part = (score_floor_rel * top_score) if score_floor_rel is not None else 0.0
        floor = max(abs_part, rel_part)

    selected: list[ScoredPassage] = []
    per_article: dict[str, int] = {}
    tokens_used = 0
    for sp in ranked:
        key = sp.passage.path
        if per_article.get(key, 0) >= max_per_article:
            continue
        if dedup_threshold is not None and _ref_is_near_duplicate(
            sp, selected, threshold=dedup_threshold
        ):
            continue
        if (
            floor is not None
            and sp.score < floor
            and key not in per_article
            and len(per_article) >= min_articles
        ):
            continue
        passage_tokens = len(sp.passage.text.split())
        if passage_tokens + tokens_used > token_budget and selected:
            break
        selected.append(sp)
        per_article[key] = per_article.get(key, 0) + 1
        tokens_used += passage_tokens
    return selected


def _ref_diverse_assemble(
    scored: list[ScoredPassage],
    *,
    token_budget: int,
    max_per_article: int,
    max_per_archive: int,
    lambda_relevance: float,
    threshold: float | None,
) -> list[ScoredPassage]:
    pool = sorted(scored, key=lambda sp: sp.score, reverse=True)
    if not pool:
        return []
    max_score = pool[0].score or 1.0

    selected: list[ScoredPassage] = []
    per_article: dict[str, int] = {}
    per_archive: dict[int, int] = {}
    tokens_used = 0

    while pool:
        best: ScoredPassage | None = None
        best_mmr = float("-inf")
        for sp in pool:
            if per_article.get(sp.passage.path, 0) >= max_per_article:
                continue
            if per_archive.get(sp.passage.zim_id, 0) >= max_per_archive:
                continue
            relevance = (sp.score / max_score) if max_score else 0.0
            redundancy = max(
                (_ref_similarity(sp.passage.text, s.passage.text) for s in selected),
                default=0.0,
            )
            mmr = lambda_relevance * relevance - (1.0 - lambda_relevance) * redundancy
            if mmr > best_mmr:
                best_mmr = mmr
                best = sp
        if best is None:
            break
        pool.remove(best)
        if threshold is not None and _ref_is_near_duplicate(best, selected, threshold=threshold):
            continue
        passage_tokens = len(best.passage.text.split())
        if passage_tokens + tokens_used > token_budget and selected:
            break
        selected.append(best)
        per_article[best.passage.path] = per_article.get(best.passage.path, 0) + 1
        per_archive[best.passage.zim_id] = per_archive.get(best.passage.zim_id, 0) + 1
        tokens_used += passage_tokens
    return selected


def _ref_section_window_assemble(
    scored: list[ScoredPassage],
    *,
    token_budget: int,
    max_per_article: int,
    window: int,
    threshold: float | None,
) -> list[ScoredPassage]:
    by_key: dict[tuple[int, str, int], ScoredPassage] = {
        (sp.passage.zim_id, sp.passage.path, sp.passage.ordinal): sp for sp in scored
    }
    ranked = sorted(scored, key=lambda sp: sp.score, reverse=True)

    expanded: list[ScoredPassage] = []
    seen: set[tuple[int, str, int]] = set()
    per_article: dict[str, int] = {}
    tokens_used = 0

    for sp in ranked:
        key = (sp.passage.zim_id, sp.passage.path, sp.passage.ordinal)
        if key in seen:
            continue
        if per_article.get(sp.passage.path, 0) >= max_per_article:
            continue
        if threshold is not None and _ref_is_near_duplicate(sp, expanded, threshold=threshold):
            continue
        group = [sp]
        for delta in range(1, window + 1):
            for direction in (-1, 1):
                nk = (sp.passage.zim_id, sp.passage.path, sp.passage.ordinal + direction * delta)
                neighbour = by_key.get(nk)
                if neighbour is not None and neighbour.passage.breadcrumb == sp.passage.breadcrumb:
                    group.append(neighbour)
        group.sort(key=lambda g: g.passage.ordinal)
        group_tokens = sum(len(g.passage.text.split()) for g in group)
        if group_tokens + tokens_used > token_budget and expanded:
            break
        for g in group:
            gkey = (g.passage.zim_id, g.passage.path, g.passage.ordinal)
            if gkey in seen:
                continue
            seen.add(gkey)
            expanded.append(g)
        per_article[sp.passage.path] = per_article.get(sp.passage.path, 0) + 1
        tokens_used += group_tokens
    return expanded


def _rand_text(rng: random.Random, lo: int, hi: int) -> str:
    return " ".join(rng.choice(_VOCAB) for _ in range(rng.randint(lo, hi)))


def _mutate(rng: random.Random, text: str) -> str:
    """One or two word swaps — near-duplicate Jaccard around the threshold."""
    out = text.split()
    for _ in range(rng.choice([1, 2])):
        out[rng.randrange(len(out))] = rng.choice(_VOCAB)
    return " ".join(out)


def _random_corpus(rng: random.Random) -> list[ScoredPassage]:
    """Articles with several same-section passages each; texts are exact
    duplicates of the article base, mutations of it (near-duplicates), or
    plain draws. Quantized scores force rank ties. Edge shapes (empty/short
    text, a cross-archive exact duplicate) are appended deterministically so
    every seed exercises them."""
    scored: list[ScoredPassage] = []
    ordinals: dict[str, int] = {}

    def add(text: str, *, path: str, score: float, zim_id: int, breadcrumb: str) -> None:
        o = ordinals.get(path, 0)
        ordinals[path] = o + 1
        scored.append(
            _sp(text, path=path, score=score, ordinal=o, breadcrumb=breadcrumb, zim_id=zim_id)
        )

    for a in range(rng.randint(3, 7)):
        path = f"art{a}"
        zim_id = rng.choice([1, 2])
        base = _rand_text(rng, 8, 24)
        for k in range(rng.randint(1, 5)):
            roll = rng.random()
            if roll < 0.35:
                text = base  # exact duplicate
            elif roll < 0.65:
                text = _mutate(rng, base)  # near-duplicate
            else:
                text = _rand_text(rng, 2, 18)  # plain
            score = rng.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            add(text, path=path, score=score, zim_id=zim_id, breadcrumb=f"Art{a} > Sec{k // 2}")

    # Deterministic edge shapes, independent of the rng rolls.
    add("", path="edge_empty", score=0.5, zim_id=1, breadcrumb="Edge")
    add("tiny", path="edge_short", score=0.5, zim_id=1, breadcrumb="Edge")
    anchor = _rand_text(rng, 10, 20)
    add(anchor, path="edge_dup_a", score=0.5, zim_id=1, breadcrumb="Edge")
    add(anchor, path="edge_dup_b", score=0.5, zim_id=2, breadcrumb="Edge")

    rng.shuffle(scored)
    return scored


@pytest.mark.parametrize("seed", range(20))
def test_selections_identical_to_naive_reference_randomized(seed: int) -> None:
    rng = random.Random(1000 + seed)
    corpus = _random_corpus(rng)
    q = _pq()
    token_budget = rng.choice([40, 150, 500])
    max_per_article = rng.choice([1, 2])
    budget = Budget(token_total=100_000, max_per_article=100)
    min_articles = rng.choice([1, 8])

    # apply_budget — the shared selection core behind topk_budget AND lead_boost.
    for dedup_threshold in (DEFAULT_THRESHOLD, None):
        for floor_abs, floor_rel in ((None, None), (0.02, 0.02)):
            ranked = sorted(corpus, key=lambda sp: sp.score, reverse=True)
            got = apply_budget(
                ranked,
                token_budget=token_budget,
                max_per_article=max_per_article,
                dedup_threshold=dedup_threshold,
                score_floor_abs=floor_abs,
                score_floor_rel=floor_rel,
                min_articles=min_articles,
            )
            want = _ref_apply_budget(
                ranked,
                token_budget=token_budget,
                max_per_article=max_per_article,
                dedup_threshold=dedup_threshold,
                score_floor_abs=floor_abs,
                score_floor_rel=floor_rel,
                min_articles=min_articles,
            )
            assert got == want

    # topk_budget — full result (passages, cards, confidence), not just picks.
    ordering = rng.choice(["score_desc", "edges"])
    topk = TopKBudget(
        params=TopKBudget.Params(
            budget_tokens=token_budget,
            max_per_article=max_per_article,
            dedup="near_exact",
            ordering=ordering,
        )
    )
    got_result = topk.assemble(list(corpus), budget, q, tr=None)  # type: ignore[arg-type]
    ref_ranked = sorted(corpus, key=lambda sp: sp.score, reverse=True)
    ref_selected = _ref_apply_budget(
        ref_ranked,
        token_budget=token_budget,
        max_per_article=max_per_article,
        dedup_threshold=DEFAULT_THRESHOLD,
    )
    want_result = build_result(
        ref_selected,
        apply_ordering(ref_selected, ordering),
        q.terms,
        None,  # type: ignore[arg-type]
        source="topk_budget",
    )
    assert got_result == want_result

    # diverse — MMR grid over lambda, dedup, and archive cap.
    max_per_archive = rng.choice([1, 4])
    for lambda_relevance in (0.0, 0.7, 1.0):
        for dedup_on in (True, False):
            diverse = Diverse(
                params=Diverse.Params(
                    budget_tokens=token_budget,
                    max_per_article=max_per_article,
                    max_per_archive=max_per_archive,
                    dedup="near_exact" if dedup_on else "none",
                    lambda_relevance=lambda_relevance,
                )
            )
            got_result = diverse.assemble(list(corpus), budget, q, tr=None)  # type: ignore[arg-type]
            ref_selected = _ref_diverse_assemble(
                corpus,
                token_budget=token_budget,
                max_per_article=max_per_article,
                max_per_archive=max_per_archive,
                lambda_relevance=lambda_relevance,
                threshold=DEFAULT_THRESHOLD if dedup_on else None,
            )
            want_result = build_result(
                ref_selected,
                apply_ordering(ref_selected, "score_desc"),
                q.terms,
                None,  # type: ignore[arg-type]
                source="diverse",
            )
            assert got_result == want_result

    # section_window — window expansion feeds neighbours into later dedup checks.
    for window in (0, 1, 2):
        for dedup_on in (True, False):
            section_window = SectionWindow(
                params=SectionWindow.Params(
                    budget_tokens=token_budget,
                    max_per_article=max_per_article,
                    dedup="near_exact" if dedup_on else "none",
                    window=window,
                )
            )
            got_result = section_window.assemble(list(corpus), budget, q, tr=None)  # type: ignore[arg-type]
            ref_selected = _ref_section_window_assemble(
                corpus,
                token_budget=token_budget,
                max_per_article=max_per_article,
                window=window,
                threshold=DEFAULT_THRESHOLD if dedup_on else None,
            )
            want_result = build_result(
                ref_selected,
                apply_ordering(ref_selected, "score_desc"),
                q.terms,
                None,  # type: ignore[arg-type]
                source="section_window",
            )
            assert got_result == want_result
