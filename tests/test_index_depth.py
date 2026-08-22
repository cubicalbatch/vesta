"""Index-depth → chunk mapping tests.

``chunks_for_article`` is pure logic over ``ExtractedArticle``; these tests pin
the per-depth contract the indexer depends on:

* depth 0 / empty text → nothing (redirects skip naturally, no special case);
* depth 1 = title+lead only, one chunk, offsets up to the first H2;
* depth 2 = lead + one chunk per H2+ section, capped (~8) with tail folding;
* depth 3 = every ~400-token passage, with the encoder-facing text composed
  EXACTLY as Stage B composes it (``retrieval/scorers/_compose.py``) and char
  offsets that index into ``article.text`` honestly (no text in the
  vector table — the span is recovered by offset).
"""

from __future__ import annotations

from itertools import pairwise

from vesta.index.depth import (
    DEPTH2_MAX_SECTIONS,
    DEPTH2_SECTION_TARGET_TOKENS,
    chunks_for_article,
)
from vesta.zim.passages import TOKENS_PER_WORD
from vesta.zim.types import EntryFlags, EntryPath, ExtractedArticle, Section


def _article(
    text: str,
    sections: tuple[Section, ...] = (),
    *,
    title: str = "Test Article",
) -> ExtractedArticle:
    return ExtractedArticle(
        path=EntryPath("A/Test_Article"),
        title=title,
        text=text,
        sections=sections,
        flags=EntryFlags.NONE,
    )


def _sectioned_article(n_sections: int) -> ExtractedArticle:
    """Lead + ``n_sections`` H2 sections, contiguous offsets, known content."""
    lead = "This is the lead paragraph of the test article. " * 4
    parts: list[str] = [lead]
    sections: list[Section] = []
    cursor = len(lead)
    for i in range(n_sections):
        body = f"Body of section {i}. " * 10
        sections.append(
            Section(
                heading_path=("Test Article", f"Section {i}"),
                level=2,
                char_start=cursor,
                char_end=cursor + len(body),
            )
        )
        parts.append(body)
        cursor += len(body)
    return _article("".join(parts), tuple(sections))


# ── depth 0 / empty ─────────────────────────────────────────────────────────


def test_depth_zero_embeds_nothing() -> None:
    assert chunks_for_article(_sectioned_article(3), 0) == []


def test_empty_text_embeds_nothing() -> None:
    # A redirect/soft-redirect stub: extraction yields no text, so the indexer
    # skips it without a special case.
    assert chunks_for_article(_article("   \n  "), 1) == []
    assert chunks_for_article(_article(""), 3) == []


# ── depth 1 ──────────────────────────────────────────────────────────────────


def test_depth1_single_lead_chunk_to_first_h2() -> None:
    article = _sectioned_article(2)
    chunks = chunks_for_article(article, 1)
    assert len(chunks) == 1
    (c,) = chunks
    first_h2 = article.sections[0].char_start
    assert (c.char_start, c.char_end) == (0, first_h2)
    assert c.is_lead and c.ordinal == 0
    # The encoder sees the title prefix; the stored offsets recover only the
    # raw lead span.
    assert c.text.startswith("Test Article > ")
    assert c.text.removeprefix("Test Article > ") == article.text[0:first_h2]


def test_depth1_no_sections_whole_text_is_lead() -> None:
    article = _article("A short article with no headings at all.")
    chunks = chunks_for_article(article, 1)
    assert len(chunks) == 1
    assert (chunks[0].char_start, chunks[0].char_end) == (0, len(article.text))


def test_depth1_empty_title_not_a_bare_separator() -> None:
    article = _article("Lead text.", title="  ")
    (c,) = chunks_for_article(article, 1)
    assert c.text == "Lead text."


# ── depth 2 ──────────────────────────────────────────────────────────────────


def test_depth2_lead_plus_one_chunk_per_h2() -> None:
    article = _sectioned_article(3)
    chunks = chunks_for_article(article, 2)
    assert len(chunks) == 4
    assert chunks[0].is_lead
    for chunk, section in zip(chunks[1:], article.sections, strict=True):
        assert not chunk.is_lead
        # Sections shorter than DEPTH2_SECTION_TARGET_TOKENS stay one chunk.
        # The span starts at the section and covers all of its content; the
        # sentence-aligned end may drop trailing whitespace, never text.
        assert chunk.char_start == section.char_start
        assert chunk.char_end <= section.char_end
        assert article.text[chunk.char_end : section.char_end].strip() == ""
        assert article.text[chunk.char_start : chunk.char_end] in chunk.text
    # Ordinals are sequential across the whole article.
    assert [c.ordinal for c in chunks] == [0, 1, 2, 3]


def test_depth2_long_section_splits_instead_of_truncating() -> None:
    """A section past the target becomes several chunks, not one chunk whose
    tail the encoder silently truncates away.

    Before this, an oversized H2 produced ONE ChunkSpec spanning the whole
    section; the encoder cut it at ``max_tokens`` (512) while the chunk row's
    ``char_end`` still claimed the full span — so retrieval by offset returned
    text the vector had never seen.
    """
    lead = "Lead sentence. " * 3
    body = "Body sentence number four hundred. " * 120  # ~4 KB, far over target
    text = lead + body
    section = Section(("Test Article", "Long"), 2, len(lead), len(text))
    article = _article(text, (section,))

    chunks = chunks_for_article(article, 2)
    section_chunks = [c for c in chunks if not c.is_lead]

    assert len(section_chunks) > 1, "an oversized section must split"
    # Contiguous, starting at the section and covering all of its text.
    assert section_chunks[0].char_start == section.char_start
    assert all(a.char_end == b.char_start for a, b in pairwise(section_chunks))
    assert text[section_chunks[-1].char_end : section.char_end].strip() == ""
    # Each piece is near the target, so none relies on encoder truncation.
    for chunk in section_chunks:
        assert len(text[chunk.char_start : chunk.char_end].split()) <= 2 * (
            DEPTH2_SECTION_TARGET_TOKENS / TOKENS_PER_WORD
        )
        assert text[chunk.char_start : chunk.char_end] in chunk.text
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_depth2_skips_level1_sections() -> None:
    text = "intro" + "x" * 50
    sections = (Section(("Test",), 1, 5, len(text)),)
    article = _article(text, sections)
    chunks = chunks_for_article(article, 2)
    # Only the lead chunk: level-1 headings are not H2 divisions.
    assert len(chunks) == 1 and chunks[0].is_lead


def test_depth2_cap_folds_tail_into_last_section() -> None:
    n = DEPTH2_MAX_SECTIONS + 3
    article = _sectioned_article(n)
    chunks = chunks_for_article(article, 2)
    assert len(chunks) == 1 + DEPTH2_MAX_SECTIONS
    # The last admitted section's span is extended to the article end so the
    # tail (e.g. a giant References list) is folded in, not lost.
    last = chunks[-1]
    assert last.char_end == len(article.text)
    assert last.char_start == article.sections[DEPTH2_MAX_SECTIONS - 1].char_start


# ── depth 3 ──────────────────────────────────────────────────────────────────


def test_depth3_every_passage_with_stage_b_text_shape() -> None:
    # Long enough to force multiple ~400-token passages.
    lead = "The opening section introduces the topic at length. " * 60
    body = "A detailed body sentence with substantial content. " * 200
    text = lead + body
    sections = (Section(("Test Article", "Details"), 2, len(lead), len(text)),)
    article = _article(text, sections)
    chunks = chunks_for_article(article, 3)
    assert len(chunks) >= 2
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        # Offsets index into the raw article text honestly, and the
        # encoder-facing text ends with exactly that span — the breadcrumb is
        # a virtual prefix, composed as Stage B composes it.
        span = article.text[c.char_start : c.char_end]
        assert span.strip()
        assert c.text.endswith(span)
    # The lead passage is flagged.
    assert any(c.is_lead for c in chunks)
