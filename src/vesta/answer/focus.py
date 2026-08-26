"""Focused article view — decide WHICH part of a long article the model sees
(answer-layer, mechanical, no model/encoder call).

Iterations 12-16 fixed *which article* reaches the model. This module fixes a
different, later-discovered bug: even when the right article IS read in full,
every read site in ``tools.py``/``api/answer.py`` truncated it by POSITION —
head-first (``_PROBE_READ_CHARS``) or centred on the single highest-scoring
retrieval passage (``_select_escalation_window``, iter 16). Two measured facts
motivate a smarter selection:

1. **The cross-encoder is an ANTI-SIGNAL at intra-article granularity.**
   Scoring every passage of the target article against the question, the gold
   passage ranked 11/11, 8/9, 11/15, 13/14, 3/12 (i.e. near LAST) across five
   failing benchmark questions. Within one article every passage is on-topic,
   so a relevance model tuned to separate on-topic from off-topic cannot
   discriminate further. Do NOT rerank intra-article passages with the
   cross-encoder or an embedding — this module never calls one.
2. **IDF-weighted question-term overlap ranks the gold passage 1st in 4/5 of
   those cases (4th in the 5th).** This is the mechanical, free signal this
   module implements: score every passage by how many *rare* (low document-
   frequency-within-this-article) question terms it contains, weighted by
   ``log(1 + n_passages / (1 + df[term]))`` — a standard IDF form computed
   fresh per article (no corpus-wide index needed, no I/O).

Algorithm (validated by simulation against the 5 failing questions + a
47 832-char Louis_Pasteur control, budget=8 000): whole-article passthrough
when the article already fits; otherwise split into ~400-token sentence-
aligned chunks, score each by IDF-weighted question-term overlap, and
greedily take the highest-scoring chunks that fit the budget — while always
guaranteeing the article lead and any caller-supplied ``must_include_spans``.
Coverage at budget=8 000: q1/q12/q23/q41/q46 all ✓, Louis_Pasteur ✓. At
budget=4 000 it drops to 5/6 (q12 fails) — hence :data:`DEFAULT_FOCUS_BUDGET_CHARS`
stays 8 000, matching the iter-16 ``_ESCALATION_WINDOW_CHARS`` it replaces.

**Why this does not import ``vesta.zim.passages.split_passages``, despite the
obvious parallel:** two independent reasons, either one sufficient alone.
First, ``answer/``'s module-boundary cap is 3 internal-package deps
(``retrieval``, ``inference``, ``config`` — see ``answer/tools.py``'s
docstring and ``tests/test_boundaries.py``'s ``_WIDENED_CAPS``); importing
``zim`` would make it 4 and fail that test. Second, and more fundamentally,
none of this module's three call sites ever hold a real ``ExtractedArticle``
with HTML-derived ``Section`` data — ``ToolRuntime.read_article`` (the DI
seam ``answer/`` uses to stay ``zim``-free, see ``tools.py``) returns plain
``str``, so real per-passage heading breadcrumbs are architecturally
unavailable here regardless of the import cap. :func:`_chunk_spans` below is
therefore a small, self-contained mirror of ``zim/passages.py``'s sentence-
aligned, zero-overlap chunking rule (same boundary regex, same ~1.3
tokens/word calibration) — a second encoding of one rule,
not reimplementation drift, and it only ever runs over already-extracted
plain text, never touching indexing.

Pure functions only: no I/O, no settings lookups, no model or encoder calls;
stopwords are a caller-supplied parameter.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

#: Mirrors ``zim/passages.py``'s ``TOKENS_PER_WORD`` calibration: a
#: local constant, not an import, per the module docstring's boundary note.
_TOKENS_PER_WORD = 1.3
_TARGET_TOKENS = 400
_TARGET_WORDS = max(int(_TARGET_TOKENS / _TOKENS_PER_WORD), 1)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+|\n+", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s", re.UNICODE)


#: A chunk may grow to this multiple of its target before a hard whitespace
#: split, so one very long unpunctuated run can't become a single giant chunk.
_HARD_SPLIT_MULT = 2.0

#: Content-word tokenizer for IDF scoring: alnum runs, lowercased, 3+ chars,
#: minus the caller's stopwords — a term identity for counting, not a
#: search-query constructor.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_MIN_TERM_LEN = 3

#: Spans within this many chars of each other are merged into one contiguous
#: span before rendering, so the excerpt doesn't fragment into slivers
#: separated by a trivial elision.
_MERGE_GAP_CHARS = 200

#: Default char budget for a focused view. Matches ``_ESCALATION_WINDOW_CHARS``
#: (~2 000 tokens, under the answer model's documented context cliff). Measured
#: coverage (gold fact present in the excerpt) at this budget: q1 ✓, q12 ✓,
#: q23 ✓, q41 ✓, q46 ✓, plus a 47 832-char Louis_Pasteur control ✓. At
#: budget=4 000 coverage drops to 5/6 (q12 fails) — do not lower this without
#: re-simulating.
DEFAULT_FOCUS_BUDGET_CHARS = 8_000

_ELISION = "\n\n[...]\n\n"


@dataclass(frozen=True)
class FocusedView:
    """The chosen excerpt of one article, ready to inject as evidence.

    ``spans`` are ``(char_start, char_end)`` into the ORIGINAL ``text``, in
    document order, post-merge (via :func:`_merge_spans`) — so a caller that needs
    the outer bounds for a single ``Passage.char_start``/``char_end`` can use
    ``spans[0][0]``/``spans[-1][1]``. ``mode`` is ``"whole_article"`` (the
    article fit under budget, returned unchanged) or ``"focused"`` (budget
    exceeded — the excerpt is chunk selection + elision, per the module
    docstring's algorithm). ``chars_used`` is the actual excerpt char count
    (merged spans may include a small amount of gap text — see
    ``_MERGE_GAP_CHARS`` — so this can exceed ``budget`` by that much, and
    guaranteed spans, e.g. a large ``must_include_span``, can exceed it more).

    An EMPTY ``spans`` (only possible when ``text`` itself is empty) is the
    "helper returns nothing" signal callers use to fall back to their own
    absolute-cap head read.
    """

    excerpt: str
    spans: tuple[tuple[int, int], ...]
    mode: str
    chars_used: int


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split ``text`` into ``(start, end)`` sentence spans (non-overlapping).
    Verbatim mirror of ``zim/passages.py``'s ``_sentences`` — see module
    docstring for why this is a local copy, not an import."""
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


def _nearest_space_after(text: str, start: int, target_words_hint: int) -> int:
    """A whitespace offset near ``start + approx target char length``.

    Falls back to ``min(target, len(text))`` rather than ``len(text)`` when no
    whitespace exists at or after ``target``, so unpunctuated/spaceless text splits
    into target-sized chunks instead of collapsing to EOF. Forward progress is
    guaranteed (target > start).
    """
    approx_chars = max(int(target_words_hint * _TOKENS_PER_WORD * 5), 1)  # ~5 chars/word
    target = start + approx_chars
    if target >= len(text):
        return len(text)
    m = _WHITESPACE_RE.search(text, target)
    if m is None:
        return target
    return m.start()


def _chunk_spans(text: str, target_words: int) -> list[tuple[int, int]]:
    """Accumulate whole sentences up to ~``target_words`` per chunk (zero
    overlap). Mirrors ``zim/passages.py``'s ``_sentence_aligned_spans``."""
    spans: list[tuple[int, int]] = []
    sentences = _sentence_spans(text)
    if not sentences:
        return spans
    chunk_start = sentences[0][0]
    end_of_last = sentences[-1][1]
    word_count = 0
    for s_start, s_end in sentences:
        words = text[s_start:s_end].split()
        if word_count and word_count + len(words) > target_words:
            spans.append((chunk_start, s_start))
            chunk_start = s_start
            word_count = len(words)
        else:
            word_count += len(words)
        if word_count > target_words * _HARD_SPLIT_MULT and s_end > chunk_start:
            cut = _nearest_space_after(text, chunk_start, target_words)
            if cut > chunk_start:
                spans.append((chunk_start, cut))
                chunk_start = cut
                word_count = len(text[cut:s_end].split())
    if chunk_start < end_of_last:
        spans.append((chunk_start, end_of_last))
    return spans


def _content_terms(text: str, stopwords: frozenset[str]) -> list[str]:
    """Lowercase alnum tokens of 3+ chars, minus ``stopwords``."""
    terms: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        w = raw.lower()
        if len(w) >= _MIN_TERM_LEN and w not in stopwords:
            terms.append(w)
    return terms


def _idf_scores(
    passage_texts: Sequence[str], question_terms: frozenset[str], stopwords: frozenset[str]
) -> list[float]:
    """IDF-weighted question-term overlap, one score per passage (see module
    docstring point 2 — the measured signal that works). Document frequency
    is computed ACROSS THE ARTICLE'S OWN PASSAGES (no corpus-wide index):
    ``df[term]`` = how many of this article's own chunks contain it. Each
    question term present in a passage contributes
    ``log(1 + n_passages / (1 + df[term]))``; rarer-within-this-article terms
    (lower df) score higher, which is exactly what discriminates a buried
    specific fact from the generic filler surrounding it.
    """
    term_sets = [set(_content_terms(t, stopwords)) for t in passage_texts]
    n_passages = len(term_sets)
    df = {t: sum(1 for ts in term_sets if t in ts) for t in question_terms}
    scores: list[float] = []
    for ts in term_sets:
        score = sum(math.log(1 + n_passages / (1 + df[t])) for t in question_terms if t in ts)
        scores.append(score)
    return scores


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _merge_spans(spans: list[tuple[int, int]], *, gap: int) -> list[tuple[int, int]]:
    """Sort by start and fold spans within ``gap`` chars of each other into
    one contiguous span, so the excerpt reads coherently (document order,
    minimal fragmentation) instead of many tiny elided slivers."""
    if not spans:
        return []
    ordered = sorted(spans)
    merged = [ordered[0]]
    for s, e in ordered[1:]:
        last_s, last_e = merged[-1]
        if s - last_e <= gap:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def _render(text: str, spans: Sequence[tuple[int, int]], breadcrumb: str) -> str:
    """Join the chosen spans in document order, each prefixed with
    ``breadcrumb`` (the article title — the finest breadcrumb available here,
    see module docstring) and separated by an explicit elision marker so the
    model doesn't mistake a gap for continuous text."""
    parts = []
    for s, e in spans:
        chunk = text[s:e].strip()
        parts.append(f"{breadcrumb}\n{chunk}" if breadcrumb else chunk)
    return _ELISION.join(parts)


def focused_view(
    text: str,
    question: str,
    budget: int = DEFAULT_FOCUS_BUDGET_CHARS,
    *,
    breadcrumb: str = "",
    stopwords: frozenset[str] = frozenset(),
    must_include_spans: Sequence[tuple[int, int]] = (),
) -> FocusedView:
    """Pick which part(s) of ``text`` to show the model for ``question``.

    ``must_include_spans`` (typically a retrieval-selected passage's own
    ``char_start``/``char_end``) and the article lead are ALWAYS included,
    budgeted first — this is what makes a caller's use of this function a
    strict superset of "always show the lead"/"always show the passage
    retrieval already found", never a regression. The remaining budget is
    filled greedily by IDF-weighted question-term overlap (descending score,
    ``score <= 0`` skipped entirely), skipping a chunk that doesn't fit and
    continuing to the next rather than stopping (a later, smaller, highly-
    relevant chunk should not be starved by one big low-relevance chunk
    ahead of it in score order).

    Returns ``mode="whole_article"`` unchanged when ``text`` already fits
    ``budget`` — the common case, and the full-context baseline this
    algorithm must never make worse (measured: full-context scores 50/50 on
    the motivating model). An empty ``text`` returns an empty
    :class:`FocusedView` (``spans=()``) — the "returns nothing" signal a
    caller falls back to its own head-read for.
    """
    if not text:
        return FocusedView(excerpt="", spans=(), mode="whole_article", chars_used=0)

    if len(text) <= budget:
        return FocusedView(
            excerpt=text, spans=((0, len(text)),), mode="whole_article", chars_used=len(text)
        )

    chunk_spans = _chunk_spans(text, _TARGET_WORDS)
    if not chunk_spans:  # defensive — _sentence_spans always returns >= 1 span
        end = min(len(text), budget)
        return FocusedView(excerpt=text[:end], spans=((0, end),), mode="focused", chars_used=end)

    question_terms = frozenset(_content_terms(question, stopwords))
    if question_terms:
        chunk_texts = [text[s:e] for s, e in chunk_spans]
        scored_texts = [f"{breadcrumb} {t}" if breadcrumb else t for t in chunk_texts]
        scores = _idf_scores(scored_texts, question_terms, stopwords)
    else:
        scores = [0.0] * len(chunk_spans)

    # Guaranteed spans, budgeted first: the caller's must-include span(s) —
    # typically the passage retrieval already scored highest — then the
    # article lead (chunk 0). Order matters only for overlap dedup; both are
    # unconditionally included regardless of remaining budget.
    guaranteed: list[tuple[int, int]] = []
    budget_used = 0
    for raw_span in (*must_include_spans, chunk_spans[0]):
        s = max(0, min(len(text), raw_span[0]))
        e = max(0, min(len(text), raw_span[1]))
        if e <= s or any(_overlaps((s, e), g) for g in guaranteed):
            continue
        guaranteed.append((s, e))
        budget_used += e - s

    selected = list(guaranteed)
    remaining = max(0, budget - budget_used)
    order = sorted(range(len(chunk_spans)), key=lambda i: (-scores[i], i))
    for i in order:
        if scores[i] <= 0:
            continue
        span = chunk_spans[i]
        if any(_overlaps(span, g) for g in selected):
            continue
        length = span[1] - span[0]
        if length <= remaining:
            selected.append(span)
            remaining -= length
        # else: doesn't fit — skip and keep looking at the next-best chunk.

    merged = _merge_spans(selected, gap=_MERGE_GAP_CHARS)
    excerpt = _render(text, merged, breadcrumb)
    return FocusedView(
        excerpt=excerpt,
        spans=tuple(merged),
        mode="focused",
        chars_used=sum(e - s for s, e in merged),
    )


def idf_passage_scores(
    question: str,
    passage_texts: Sequence[str],
    *,
    stopwords: frozenset[str] = frozenset(),
) -> list[float]:
    """IDF-weighted question-term overlap of each passage against ``question``.

    The same measured signal :func:`focused_view` ranks
    article chunks by (module docstring point 2), exposed for callers that
    rank WHOLE passages instead of selecting spans within one article —
    document frequency is computed across ``passage_texts`` themselves, so a
    caller ranking a retrieval result set gets an IDF computed over that set.
    One score per passage, same order; a ``question`` with no content terms
    scores everything 0.0 (the "no signal — keep the caller's order" case).
    Stopwords are caller-supplied, exactly like :func:`focused_view`'s.
    """
    question_terms = frozenset(_content_terms(question, stopwords))
    if not question_terms:
        return [0.0] * len(passage_texts)
    return _idf_scores(list(passage_texts), question_terms, stopwords)


__all__ = [
    "DEFAULT_FOCUS_BUDGET_CHARS",
    "FocusedView",
    "focused_view",
    "idf_passage_scores",
]
