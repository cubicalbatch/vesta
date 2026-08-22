"""Tests for ``answer/focus.py`` (iter 17): the focused article view that
replaces position-based truncation (head-first / best-passage-centred) at the
read sites in ``tools.py``/``api/answer.py``."""

from __future__ import annotations

from itertools import pairwise

import pytest

from vesta.answer.focus import (
    _chunk_spans,
    _idf_scores,
    focused_view,
)

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "into",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "doing",
        "this",
        "that",
        "these",
        "those",
        "how",
        "what",
        "why",
        "who",
        "when",
        "where",
        "which",
        "documented",
    ]
)

#: A sentence-punctuated filler block with no overlap against any question
#: used below — every chunk built from this scores 0 unless a fact is planted
#: inside it. Real sentences (periods + spaces) matter: ``focus._chunk_spans``
#: is sentence-aligned, so unpunctuated filler (e.g. ``"a" * N``) would not
#: exercise the real chunking path the way actual article text does.
_FILLER = "The garden had many colorful flowers blooming in springtime every year without fail. "


def _filler(approx_chars: int) -> str:
    reps = approx_chars // len(_FILLER) + 1
    return (_FILLER * reps)[:approx_chars]


class TestWholeArticlePassthrough:
    """The common case (per focus.py's docstring, the full-context baseline scores
    50/50 on the motivating model) — an article that already fits the budget
    is returned completely unchanged, not re-chunked or re-ordered."""

    @pytest.mark.parametrize(
        ("text", "budget", "expected_mode"),
        [
            (
                "A short article about nothing in particular. It has two sentences.",
                1000,
                "whole_article",
            ),
            ("x" * 500, 500, "whole_article"),
            ("This is one sentence. " * 30, 500, "focused"),
        ],
    )
    def test_whole_article_budget_boundaries(
        self, text: str, budget: int, expected_mode: str
    ) -> None:
        view = focused_view(text, "sentence", budget, stopwords=STOPWORDS)
        assert view.mode == expected_mode
        if expected_mode == "whole_article":
            assert view.excerpt == text

    def test_empty_text_returns_nothing(self) -> None:
        """The "helper returns nothing" signal a caller falls back on."""
        view = focused_view("", "some question", 1000, stopwords=STOPWORDS)
        assert view.spans == ()
        assert view.excerpt == ""
        assert view.chars_used == 0


class TestIdfScoring:
    """Unit coverage of the scoring primitive in isolation — the measured signal
    from focus.py's module docstring: IDF-weighted question-term overlap,
    computed across the article's OWN passages, not term frequency and not a
    corpus-wide index."""

    @pytest.mark.parametrize(
        ("passages", "question_terms", "winning_idx", "losing_indices"),
        [
            (
                [
                    "The composition of rocks varies across regions.",
                    "Another passage mentions composition in passing too.",
                    "This buried passage discusses xenolith composition in detail.",
                ],
                frozenset({"composition", "xenolith"}),
                2,
                [0, 1],
            ),
            (
                [
                    "composition composition composition composition composition rocks.",
                    "a brief note on xenolith mineral content.",
                ],
                frozenset({"xenolith"}),
                1,
                [0],
            ),
        ],
    )
    def test_idf_ranks_rare_over_common(
        self,
        passages: list[str],
        question_terms: frozenset[str],
        winning_idx: int,
        losing_indices: list[int],
    ) -> None:
        scores = _idf_scores(passages, question_terms, STOPWORDS)
        for idx in losing_indices:
            assert scores[winning_idx] > scores[idx]

    def test_no_matching_terms_scores_zero(self) -> None:
        scores = _idf_scores(["completely unrelated text."], frozenset({"xenolith"}), STOPWORDS)
        assert scores == [0.0]


class TestFocusedViewChunkSelection:
    """End-to-end ``focused_view`` behavior once budget forces chunking."""

    def test_idf_picks_the_buried_passage_over_the_high_scoring_but_wrong_one(self) -> None:
        """Simulates the measured failure mode directly (focus.py's module
        docstring point 1): a decoy region shares the question's common,
        low-information term but not its rare, discriminating one. The
        buried region — the ONLY place both terms co-occur — must win,
        exactly as the offline probe found for q1/q12/q23/q41/q46 (IDF
        ranks the gold passage 1st in 4/5 cases)."""
        decoy = (
            "This section discusses the general composition of rock formations "
            "found across many different regions of the world. " * 30
        )
        buried = (
            "Deep within this remote section, a rare xenolith composition was "
            "recorded by geologists working the site. " * 30
        )
        lead = _filler(2_000)
        text = lead + decoy + _filler(2_000) + buried + _filler(2_000)

        view = focused_view(text, "What is the xenolith composition?", 4_000, stopwords=STOPWORDS)

        assert view.mode == "focused"
        assert "xenolith" in view.excerpt.lower()
        # The decoy's distinguishing word count trick did not win it a slot
        # ahead of the buried passage: whichever non-lead chunk was chosen
        # first must be a "xenolith" chunk, not a decoy-only chunk.
        buried_start = text.index(buried)
        decoy_start = text.index(decoy)
        non_lead_spans = [s for s in view.spans if s[0] >= 2_000]
        assert non_lead_spans, "expected at least one non-lead chunk to be selected"
        closest_selected = min(non_lead_spans, key=lambda s: abs(s[0] - buried_start))
        assert abs(closest_selected[0] - buried_start) < abs(closest_selected[0] - decoy_start)

    def test_must_include_span_always_present_even_with_zero_score(self) -> None:
        """The caller's guaranteed span (e.g. the retrieval-best passage) must
        survive even when it shares no terms with the question at all — this
        is what makes call site 1's wiring a strict superset of iter 16's
        behaviour, never a regression."""
        irrelevant = _filler(1_500)
        rest = _filler(20_000)
        text = irrelevant + rest
        must_span = (0, len(irrelevant))

        view = focused_view(
            text,
            "a question with no lexical overlap whatsoever xyzzyplugh",
            2_000,
            stopwords=STOPWORDS,
            must_include_spans=[must_span],
        )

        assert any(s[0] <= must_span[0] and s[1] >= must_span[1] for s in view.spans)
        assert irrelevant[:80] in view.excerpt

    def test_spans_emitted_in_document_order_not_score_order(self) -> None:
        """A later, higher-scoring chunk and an earlier, lower-but-still-
        selected chunk must come back sorted by position, not by score —
        the excerpt has to read coherently front-to-back."""
        early_weak = "brief mention of xenolith here only once. " * 20
        late_strong = "xenolith xenolith basalt xenolith formation study reference notes. " * 20
        text = early_weak + _filler(3_000) + late_strong

        view = focused_view(text, "xenolith basalt formation", 4_000, stopwords=STOPWORDS)

        starts = [s[0] for s in view.spans]
        assert starts == sorted(starts)

    def test_budget_is_respected_when_selected_chunks_do_not_touch(self) -> None:
        """When no two selected spans are within the merge gap of each other
        (so nothing gets padded by a merge), the total excerpt size must not
        exceed the requested budget."""
        # Three well-separated "xenolith" chunks, spaced far enough apart
        # that none can be within the ~200-char merge gap of another once
        # selected — only 1-2 will fit inside a small budget.
        hit = "xenolith deposits were surveyed extensively across this area. " * 20
        text = _filler(2_000) + hit + _filler(5_000) + hit + _filler(5_000) + hit + _filler(2_000)
        budget = 3_000

        view = focused_view(text, "xenolith deposits", budget, stopwords=STOPWORDS)

        assert view.mode == "focused"
        assert view.chars_used <= budget

    def test_low_budget_still_guarantees_lead_and_must_include(self) -> None:
        """Guaranteed spans are exempt from the budget cap by design (the
        module docstring: "budgeted first", not "budget-limited") — a tiny
        budget still returns the lead + must_include span whole, even if
        their combined size exceeds the nominal budget."""
        text = _filler(20_000)
        must_span = (10_000, 10_000 + 1_500)

        view = focused_view(
            text, "irrelevant", 100, stopwords=STOPWORDS, must_include_spans=[must_span]
        )

        assert any(s[0] <= must_span[0] and s[1] >= must_span[1] for s in view.spans)
        assert view.spans[0][0] == 0  # the lead is also guaranteed


class TestChunkSpans:
    def test_chunks_cover_the_whole_text_without_gaps_or_overlap(self) -> None:
        text = ("One sentence here. Another one follows. " * 200).rstrip()
        spans = _chunk_spans(text, 100)
        assert spans[0][0] == 0
        assert spans[-1][1] == len(text)
        for (_, end), (next_start, _) in pairwise(spans):
            assert end == next_start

    def test_no_sentence_punctuation_still_returns_a_span(self) -> None:
        spans = _chunk_spans("word " * 500, 100)
        assert spans
        assert spans[0][0] == 0
        assert spans[-1][1] == len("word " * 500)
