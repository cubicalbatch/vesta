"""Snippet generation.

Xapian's snippets are unavailable through python-libzim (no
``getSnippet``), so Vesta generates its own — a query-term-centred window over
the extracted passage text, which every assembler already has in hand at this
point (no extra ZIM read).

Deliberately simple: find the earliest query-term match and window around it,
extending to word boundaries. This is a UI affordance (what a source card shows
under the title), not a ranking signal — the actual ranking is Stage B's job.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Default snippet window, in characters. Long enough to show real context,
#: short enough that a source-card list stays scannable.
DEFAULT_WINDOW_CHARS = 220

#: Query terms shorter than this are almost always noise ("a", "of", "to") and
#: would anchor the snippet on the first word of the passage instead of
#: anything meaningful.
_MIN_TERM_LEN = 3


def build_snippet(
    text: str, terms: Sequence[str], *, window_chars: int = DEFAULT_WINDOW_CHARS
) -> str:
    """A snippet centred on the earliest query-term match in ``text``.

    Falls back to the leading ``window_chars`` of ``text`` when no term
    matches (still useful — it is the start of the passage, which usually
    reads as a topic sentence given breadcrumb-prefixed, sentence-aligned
    chunking). Windows extend to the nearest word boundary so a snippet
    never starts or ends mid-word; an ellipsis marks a truncated edge.
    """
    if not text:
        return ""
    if len(text) <= window_chars:
        return text.strip()

    lower = text.lower()
    match_at: int | None = None
    for term in terms:
        needle = term.lower().strip()
        if len(needle) < _MIN_TERM_LEN:
            continue
        idx = lower.find(needle)
        if idx >= 0 and (match_at is None or idx < match_at):
            match_at = idx

    if match_at is None:
        window = text[:window_chars].rsplit(" ", 1)[0]
        return f"{window}…"

    half = window_chars // 2
    start = max(0, match_at - half)
    end = min(len(text), start + window_chars)
    start = max(0, end - window_chars)  # re-clamp if the window hit the end

    if start > 0:
        boundary = text.find(" ", start)
        if 0 <= boundary < start + 20:
            start = boundary + 1
    if end < len(text):
        boundary = text.rfind(" ", start, end)
        if boundary > start:
            end = boundary

    body = text[start:end].strip()
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{body}{suffix}"


__all__ = ["DEFAULT_WINDOW_CHARS", "build_snippet"]
