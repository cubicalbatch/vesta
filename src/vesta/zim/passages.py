"""Passage splitting — ~400 tokens, sentence-aligned, zero overlap.

Three load-bearing rules:

* **Zero overlap.** Overlap gives no measurable benefit and costs proportional
  indexing time; the usual 10-20% is inherited folklore. Sentence-boundary
  splitting — not overlap — is what prevents mid-thought truncation.
* **Breadcrumb prepended to every passage** (``Article > Section > Subsection``).
  Matches the models' ``title + passage`` training distribution; a free
  deterministic approximation of Contextual Retrieval (-35% retrieval failure)
  with no per-chunk LLM call; and it feeds the entity name to lexical matching
  on every chunk.
* **Lead-section passages are flagged, not boosted.** ``is_lead`` is a label;
  boosting is retrieval policy handled in the retrieval layer.

``char_start``/``char_end`` index into :pyattr:`ExtractedArticle.text`, and
``text == article.text[char_start:char_end]`` — so a passage is recoverable from
the ZIM by offset and vector rows stay text-free.

Token sizing uses a fast word→token approximation: English averages ~1.3
tokens/word, so target_words ≈ target_tokens / 1.3.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from vesta.zim.types import ExtractedArticle, Passage, Section

#: Approximation factor: English averages ~1.3 tokens per word.
TOKENS_PER_WORD = 1.3

#: A passage may grow to this multiple of its target before we force-split at a
#: whitespace, so a single very long unpunctuated run can't become one giant chunk.
_HARD_SPLIT_MULT = 2.0

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+|\n+", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s", re.UNICODE)


def _sentences(text: str) -> list[tuple[int, int]]:
    """Split ``text`` into ``(start, end)`` sentence spans (non-overlapping)."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENTENCE_BOUNDARY.finditer(text):
        end = m.start()
        if end > start:
            spans.append((start, end))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans or [(0, len(text))]


def _section_for_offset(sections: Sequence[Section], offset: int) -> Section | None:
    """The last section whose ``char_start`` ≤ ``offset`` (sections are sorted)."""
    current: Section | None = None
    for s in sections:
        if s.char_start <= offset:
            current = s
        else:
            break
    return current


def _first_h2_start(sections: Sequence[Section]) -> int:
    """Char offset of the first H2+ heading; the lead is everything before it."""
    for s in sections:
        if s.level >= 2:
            return s.char_start
    # No H2 ⇒ there is no lead/content boundary; treat the whole article as
    # lead. A value past EOF makes every passage's char_start < it ⇒ is_lead.
    return 2**31


def _sentence_aligned_spans(text: str, target_words: int) -> list[tuple[int, int]]:
    """Accumulate whole sentences up to ~``target_words`` with zero overlap.

    A single runaway sentence longer than the hard-split multiple is cut at the
    nearest whitespace so one giant unpunctuated run can't become one chunk.
    """
    spans: list[tuple[int, int]] = []
    sentences = _sentences(text)
    if not sentences:
        return spans
    chunk_start = sentences[0][0]
    end_of_last = sentences[-1][1]
    word_count = 0
    for s_start, s_end in sentences:
        s_text = text[s_start:s_end]
        words = s_text.split()
        if word_count and word_count + len(words) > target_words:
            spans.append((chunk_start, s_start))
            chunk_start = s_start
            word_count = len(words)
        else:
            word_count += len(words)
        # Force-split a runaway sentence at a whitespace near the target.
        if word_count > target_words * _HARD_SPLIT_MULT and s_end > chunk_start:
            cut = _nearest_space_after(text, chunk_start, target_words)
            if cut > chunk_start:
                spans.append((chunk_start, cut))
                chunk_start = cut
                word_count = len(text[cut:s_end].split())
    if chunk_start < end_of_last:
        spans.append((chunk_start, end_of_last))
    return spans


def sentence_aligned_spans(text: str, *, target_tokens: int = 400) -> list[tuple[int, int]]:
    """Public wrapper over :func:`_sentence_aligned_spans` in TOKEN units.

    Exists for depth-2 indexing (``index/depth.py``), which splits an oversized
    H2 section the same way :func:`split_passages` splits an article, so no
    section is silently cut off at the encoder's truncation ceiling. Spans are
    offsets into ``text``; ``text[start:end]`` is the span exactly.
    """
    if not text.strip():
        return []
    target_words = max(int(target_tokens / TOKENS_PER_WORD), 1)
    return _sentence_aligned_spans(text, target_words)


def _build_passage(
    *,
    text: str,
    ordinal: int,
    c_start: int,
    c_end: int,
    article: ExtractedArticle,
    breadcrumb_enabled: bool,
    zim_id: int,
    lead_end: int,
) -> Passage:
    section = _section_for_offset(article.sections, c_start)
    heading_path = tuple(section.heading_path) if section is not None else ()
    if breadcrumb_enabled:
        crumbs = [article.title, *heading_path]
        breadcrumb = " > ".join(c for c in crumbs if c)
    else:
        breadcrumb = ""
    return Passage(
        zim_id=zim_id,
        path=article.path,
        ordinal=ordinal,
        char_start=c_start,
        char_end=c_end,
        breadcrumb=breadcrumb,
        text=text[c_start:c_end],
        is_lead=c_start < lead_end,
    )


def split_passages(
    article: ExtractedArticle,
    *,
    target_tokens: int = 400,
    sentence_aligned: bool = True,
    breadcrumb_enabled: bool = True,
    zim_id: int = 0,
) -> list[Passage]:
    """Split ``article`` into breadcrumbed, non-overlapping passages.

    Each passage's ``char_start``/``char_end`` index into ``article.text`` and
    ``text`` is exactly that slice. Passages never overlap.
    """
    text = article.text
    if not text.strip():
        return []
    target_words = max(int(target_tokens / TOKENS_PER_WORD), 1)
    lead_end = _first_h2_start(article.sections)
    spans = _sentence_aligned_spans(text, target_words)
    passages: list[Passage] = []
    for ordinal, (c_start, c_end) in enumerate(spans):
        if c_end <= c_start:
            continue
        passages.append(
            _build_passage(
                text=text,
                ordinal=ordinal,
                c_start=c_start,
                c_end=c_end,
                article=article,
                breadcrumb_enabled=breadcrumb_enabled,
                zim_id=zim_id,
                lead_end=lead_end,
            )
        )
    return passages


def _nearest_space_after(text: str, start: int, target_words_hint: int) -> int:
    """A whitespace offset near ``start + approx target char length``.

    Falls back to ``min(target, len(text))`` rather than ``len(text)`` when no
    whitespace exists at or after ``target``, so unpunctuated/spaceless text splits
    into target-sized chunks instead of collapsing to EOF. Forward progress is
    guaranteed (target > start).
    """
    approx_chars = max(int(target_words_hint * TOKENS_PER_WORD * 5), 1)  # ~5 chars/word
    target = start + approx_chars
    if target >= len(text):
        return len(text)
    m = _WHITESPACE_RE.search(text, target)
    if m is None:
        return target
    return m.start()


__all__ = ["TOKENS_PER_WORD", "sentence_aligned_spans", "split_passages"]
