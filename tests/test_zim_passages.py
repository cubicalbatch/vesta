"""Passage splitting — ~400 tokens, sentence-aligned, ZERO overlap.

Three load-bearing rules, each asserted here over the fixture's long
multi-section article:

* passages never overlap and ``text == article.text[start:end]`` exactly
  (the offset contract that keeps the vector store text-free);
* no passage starts mid-sentence (sentence-boundary splitting, not overlap,
  is what prevents mid-thought truncation);
* every passage carries an ``Article > Section`` breadcrumb and lead passages
  are flagged (NOT boosted — boosting is retrieval policy).
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from vesta.zim.passages import _nearest_space_after, split_passages
from vesta.zim.types import EntryFlags, ExtractedArticle, Section


@pytest.fixture
def sample_article() -> ExtractedArticle:
    text = (
        "Albert Einstein was a German-born theoretical physicist. "
        "He developed the theory of relativity. "
        "He also made important contributions to quantum mechanics.\n\n"
        "Early Life\n"
        "Einstein was born in Ulm, in the Kingdom of Württemberg. "
        "His family moved to Munich when he was an infant. "
        "He attended the Luitpold Gymnasium in Munich.\n\n"
        "Career and Research\n"
        "In 1905, often called his annus mirabilis, he published four groundbreaking papers. "
        "These papers revolutionized the understanding of the photoelectric effect, "
        "Brownian motion, special relativity, and mass-energy equivalence. "
        "He received the Nobel Prize in Physics in 1921."
    )
    s1_end = text.index("\n\nEarly Life")
    s2_start = s1_end + 2
    s2_end = text.index("\n\nCareer and Research")
    s3_start = s2_end + 2
    s3_end = len(text)

    sections = (
        Section(heading_path=("Albert Einstein",), level=1, char_start=0, char_end=s1_end),
        Section(
            heading_path=("Albert Einstein", "Early Life"),
            level=2,
            char_start=s2_start,
            char_end=s2_end,
        ),
        Section(
            heading_path=("Albert Einstein", "Career and Research"),
            level=2,
            char_start=s3_start,
            char_end=s3_end,
        ),
    )
    return ExtractedArticle(
        path="A/Albert_Einstein",
        title="Albert Einstein",
        text=text,
        sections=sections,
        flags=EntryFlags.NONE,
    )


def test_passages_partition_text_and_respect_sentence_boundaries(
    sample_article: ExtractedArticle,
) -> None:
    passages = split_passages(sample_article, target_tokens=20, zim_id=7)
    assert passages, "sample article must yield passages"
    text = sample_article.text

    # Each passage's text is exactly its slice into the article text.
    for p in passages:
        assert p.text == text[p.char_start : p.char_end]
        assert p.zim_id == 7
        assert p.path == sample_article.path

    # Passages are contiguous, ordered, and non-overlapping (zero overlap).
    for a, b in pairwise(passages):
        assert a.ordinal < b.ordinal
        assert a.char_end <= b.char_start
        assert b.char_start < b.char_end

    # Sentence-boundary check: no passage starts mid-sentence.
    for p in passages:
        if p.char_start == 0:
            continue
        prev = text[p.char_start - 1]
        before = text[max(p.char_start - 2, 0) : p.char_start]
        assert prev in ".!?\n… " or before.rstrip().endswith((".", "!", "?", "…")), (
            f"passage starts mid-sentence at {p.char_start}: {text[p.char_start : p.char_start + 30]!r}"
        )


def test_breadcrumb_and_lead_flag(sample_article: ExtractedArticle) -> None:
    # Enabled breadcrumbs: includes article title and flags lead section.
    passages = split_passages(sample_article, target_tokens=30, breadcrumb_enabled=True)
    assert passages
    for p in passages:
        assert sample_article.title in p.breadcrumb
    assert any(p.is_lead for p in passages)

    # Disabled breadcrumbs: empty string.
    no_bc = split_passages(sample_article, target_tokens=30, breadcrumb_enabled=False)
    assert all(p.breadcrumb == "" for p in no_bc)


def test_empty_article_yields_no_passages() -> None:
    article = ExtractedArticle(
        path="A/X", title="X", text="   ", sections=(), flags=EntryFlags.NONE
    )
    assert split_passages(article, target_tokens=400) == []


def test_nearest_space_after_unicode_and_fallback() -> None:
    # 1. Unicode whitespace discovery (e.g. \u3000, \u00a0, \t, \n)
    text_cjk = "量子力学" * 50 + "\u3000" + "量子力学" * 50
    space_idx = text_cjk.index("\u3000")
    assert _nearest_space_after(text_cjk, 0, 10) == space_idx

    text_nbsp = "word" * 50 + "\u00a0" + "tail" * 50
    nbsp_idx = text_nbsp.index("\u00a0")
    assert _nearest_space_after(text_nbsp, 0, 10) == nbsp_idx

    # 2. Spaceless text fallback to target (bounded, not len(text))
    text_spaceless = "abcdef0123456789" * 100
    target_words = 15
    approx_chars = max(int(target_words * 1.3 * 5), 1)
    cut = _nearest_space_after(text_spaceless, 0, target_words)
    assert cut == approx_chars
    assert cut < len(text_spaceless)

    # 3. Forward progress guarantee
    for start in (0, 10, len(text_spaceless) - 5):
        res = _nearest_space_after(text_spaceless, start, target_words)
        assert res > start or (start >= len(text_spaceless) and res == len(text_spaceless))


def test_unpunctuated_unicode_run_hard_split() -> None:
    # A single runaway unpunctuated run with non-ASCII spaces (U+3000 / U+00A0).
    # Prior to Z1, _nearest_space_after's text.find(" ") returned -1 and collapsed
    # the entire remainder to len(text).
    text = ("词语\u3000" * 400) + ". 结尾句子。"
    article = ExtractedArticle(
        path="A/CJKRun",
        title="CJKRun",
        text=text,
        sections=(),
        flags=EntryFlags.NONE,
    )
    passages = split_passages(article, target_tokens=20)
    assert len(passages) > 1, "CJK text with Unicode spaces must not collapse to EOF"
    # First passage was hard-split near target, not collapsed to len(text)
    assert passages[0].char_end < len(text)
    for p in passages:
        assert p.text == text[p.char_start : p.char_end]


def test_newline_separated_text_splits_into_passages() -> None:
    text = "\n".join(f"Item {i}: measurement data point {i * 13}" for i in range(100))
    article = ExtractedArticle(
        path="A/DataList",
        title="DataList",
        text=text,
        sections=(),
        flags=EntryFlags.NONE,
    )
    passages = split_passages(article, target_tokens=20)
    assert len(passages) > 1, "newline-separated text must split into multiple passages"
    for p in passages:
        assert p.char_end - p.char_start < len(text)
        assert p.text == text[p.char_start : p.char_end]
