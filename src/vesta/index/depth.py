"""Index-depth → chunk mapping.

Per-archive, the depth chooses how much of each article is embedded:

* **1 — Article**: ``title + lead`` → one vector (the lead is the text up to the
  first H2, or the whole short article).
* **2 — Sectioned**: depth-1 chunk PLUS one chunk per H2+ section
  (``Section.level >= 2``). Long articles get up to ~8 section vectors.
* **3 — Full chunk**: every ~400-token passage via :func:`vesta.zim.passages.split_passages`.

The chunk text is the breadcrumb-prefixed passage (title + section path + text),
matching the encoder's ``title + passage`` training distribution (contextual
retrieval approximation). ``char_start``/``char_end`` index into
``ExtractedArticle.text`` exactly, so the vector row stays text-free.

Pure logic over the :class:`~vesta.zim.types.ExtractedArticle` domain type; no
I/O, no encoder, no DB — unit-testable in microseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

from vesta.zim.passages import sentence_aligned_spans, split_passages
from vesta.zim.types import ExtractedArticle, Section

#: Cap on section vectors per article at depth 2 (~8). Sections past
#: this are folded into the last admitted section so a giant "References" list
#: doesn't dominate an article's vector budget.
DEPTH2_MAX_SECTIONS = 8

#: Target chunk size, in tokens, when a depth-2 section is too long to embed as
#: one vector. Matches depth 3's ``split_passages`` target so the two depths
#: agree on what "one chunk" means.
#:
#: Without this an H2 section became ONE chunk spanning the whole section, and
#: the encoder truncated it at ``max_tokens`` (512). Everything past that point
#: was never embedded, yet the chunk row's ``char_end`` still claimed the whole
#: section — so retrieval-by-offset returned text the vector had never seen. It
#: also forced every long section to the maximum padding width, which is the
#: most expensive shape the encoder can be handed.
DEPTH2_SECTION_TARGET_TOKENS = 400

#: Depth-1/2 also embed the title; depth-3 passages already carry a breadcrumb
#: prefix from ``split_passages`` so the title is already present.
LEAD_BREADCRUMB_SEP = " > "


@dataclass(frozen=True)
class ChunkSpec:
    """One unit to embed for one article.

    ``text`` is what the encoder sees (breadcrumb + span). ``char_start``/
    ``char_end`` index into :pyattr:`ExtractedArticle.text` and recover the span
    from the ZIM by offset (no text is stored in the vector table).
    ``ordinal`` is the per-article chunk order the indexer persists on the chunk
    row. ``is_lead`` flags the title+lead chunk so a future lead-boost policy can
    find it without re-deriving it.
    """

    text: str
    char_start: int
    char_end: int
    ordinal: int
    is_lead: bool


def _first_h2_start(sections: tuple[Section, ...]) -> int:
    """Char offset where the lead ends (first H2+) — mirrors passages._first_h2_start."""
    for s in sections:
        if s.level >= 2:
            return s.char_start
    return 2**31  # no H2 ⇒ whole article is lead


def _lead_text(article: ExtractedArticle) -> tuple[str, int, int]:
    """The title + lead span: text from 0 to the first H2.

    Returns ``(text_to_embed, char_start, char_end)``. The lead IS the title plus
    the intro; for a short article with no sections this is the whole text.
    """
    lead_end = _first_h2_start(article.sections)
    lead_end = min(lead_end, len(article.text))
    return article.text[:lead_end], 0, lead_end


def chunks_for_article(article: ExtractedArticle, depth: int) -> list[ChunkSpec]:
    """Map one extracted article to the chunks to embed at ``depth`` (1/2/3).

    Depth 0 embeds nothing (the caller never reaches here at depth 0). An article
    with empty text yields no chunks (a redirect/soft-redirect stub), so the
    indexer naturally skips it without a special case.

    ``char_start``/``char_end`` always satisfy
    ``spec.text``-breadcrumb-recovery ⊆ ``article.text[char_start:char_end]``
    (the span is exactly that slice; the breadcrumb is a prefix the encoder sees
    but the stored offsets point at the raw span).
    """
    if depth < 1 or not article.text.strip():
        return []

    chunks: list[ChunkSpec] = []

    if depth == 3:
        # Every ~400-token passage. ``Passage.text``/``Passage.breadcrumb`` are
        # kept apart so the char offsets stay honest (zim/types.py), so the
        # encoder-facing text is composed here EXACTLY as Stage B does it
        # (``retrieval/scorers/_compose.py::passage_text`` — mirrored, not
        # imported: ``index/`` must not import ``retrieval/``). The index and
        # the query-side scorer then see the same text shape.
        for ordinal, p in enumerate(
            split_passages(article, target_tokens=400, breadcrumb_enabled=True)
        ):
            chunks.append(
                ChunkSpec(
                    text=_compose(p.breadcrumb, p.text),
                    char_start=p.char_start,
                    char_end=p.char_end,
                    ordinal=ordinal,
                    is_lead=p.is_lead,
                )
            )
        return chunks

    # Depth 1/2: a title+lead chunk first.
    lead_text, lead_start, lead_end = _lead_text(article)
    if lead_text.strip():
        chunks.append(
            ChunkSpec(
                text=_with_title(article.title, lead_text),
                char_start=lead_start,
                char_end=lead_end,
                ordinal=0,
                is_lead=True,
            )
        )

    if depth == 2:
        # One vector per H2+ section, capped to avoid a giant references
        # list dominating. Sections are contiguous partitions of the text; level
        # >= 2 are the H2/H3/... divisions beneath the lead.
        admitted = 0
        for section in article.sections:
            if section.level < 2:
                continue
            span = article.text[section.char_start : section.char_end]
            if not span.strip():
                continue
            if admitted >= DEPTH2_MAX_SECTIONS:
                # Fold the tail into the last admitted chunk's span so it is not
                # lost — extend that chunk's char_end to the article end.
                if chunks:
                    last = chunks[-1]
                    chunks[-1] = ChunkSpec(
                        text=last.text,
                        char_start=last.char_start,
                        char_end=len(article.text),
                        ordinal=last.ordinal,
                        is_lead=last.is_lead,
                    )
                break
            # A section longer than the target becomes several chunks rather
            # than one truncated one (see DEPTH2_SECTION_TARGET_TOKENS). Spans
            # are relative to `span`, so shift them into article coordinates.
            for rel_start, rel_end in sentence_aligned_spans(
                span, target_tokens=DEPTH2_SECTION_TARGET_TOKENS
            ):
                piece = span[rel_start:rel_end]
                if not piece.strip():
                    continue
                chunks.append(
                    ChunkSpec(
                        text=_with_title(article.title, piece),
                        char_start=section.char_start + rel_start,
                        char_end=section.char_start + rel_end,
                        ordinal=len(chunks),
                        is_lead=False,
                    )
                )
            # The cap counts SECTIONS, not chunks — its job is to stop a giant
            # references list from dominating, not to re-truncate long prose.
            admitted += 1

    return chunks


def _compose(breadcrumb: str, text: str) -> str:
    """``"Article > Section  passage text"`` — mirrors
    ``retrieval/scorers/_compose.py::passage_text`` field-for-field (``index/``
    may not import ``retrieval/``). Two-space separator, exactly as Stage B
    composes it, so indexed vectors and query-time scorings share one shape.
    """
    if breadcrumb:
        return f"{breadcrumb}  {text}"
    return text


def _with_title(title: str, span: str) -> str:
    """Prefix a span with the article title (breadcrumb approximation).

    Empty titles (a non-MediaWiki fixture) are skipped so we don't embed a bare
    separator. The offsets still point at ``span`` (the title is a virtual prefix
    the encoder sees; the stored char_start/char_end recover only ``span``).
    """
    if not title.strip():
        return span
    return f"{title}{LEAD_BREADCRUMB_SEP}{span}"


__all__ = ["DEPTH2_MAX_SECTIONS", "ChunkSpec", "chunks_for_article"]
