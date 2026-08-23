"""Drift guard: ``answer/focus.py``'s chunking mirrors ``zim/passages.py``.

The two copies exist on purpose — ``answer/`` must not import ``zim/``
(module-boundary cap, and no ``ExtractedArticle``/``Section`` data exists on
the answer path; see ``answer/focus.py``'s docstring). But a copy without a
check drifts silently (it already did once, pre-Z1). These tests pin the
copies to identical behavior so any edit to one side that forgets the other
fails here, not in a benchmark.
"""

from __future__ import annotations

from vesta.answer import focus as focus_mod
from vesta.zim import passages as passages_mod

_TEXTS = [
    # Ordinary punctuated prose.
    "One. Two! Three? Four… Five; six. " * 12,
    # Newline-separated structure (list items / headings).
    "# Heading\n\n- item one\n- item two\n\nParagraph with a full stop inside it. "
    "And another sentence here.\n" * 8,
    # One giant unpunctuated run (exercises the hard-split path) plus normal tail.
    ("word " * 900) + ". Short tail sentence.",
    "",
]


def test_sentence_boundary_patterns_identical() -> None:
    assert focus_mod._SENTENCE_BOUNDARY.pattern == passages_mod._SENTENCE_BOUNDARY.pattern, (
        "sentence-boundary regex drifted between answer/focus.py and zim/passages.py"
    )


def test_tokens_per_word_calibration_identical() -> None:
    assert float(focus_mod._TOKENS_PER_WORD) == float(passages_mod.TOKENS_PER_WORD)
    assert focus_mod._HARD_SPLIT_MULT == passages_mod._HARD_SPLIT_MULT


def test_sentence_spans_identical() -> None:
    for text in _TEXTS:
        assert focus_mod._sentence_spans(text) == passages_mod._sentences(text)


def test_nearest_space_after_identical() -> None:
    for text in _TEXTS:
        for start in (0, 17, 250):
            for hint in (1, 50, 400):
                assert focus_mod._nearest_space_after(
                    text, start, hint
                ) == passages_mod._nearest_space_after(text, start, hint)


def test_chunk_spans_identical() -> None:
    for text in _TEXTS:
        for target_words in (1, 30, 307):  # 307 = the module's default target
            assert focus_mod._chunk_spans(text, target_words) == (
                passages_mod._sentence_aligned_spans(text, target_words)
            )
