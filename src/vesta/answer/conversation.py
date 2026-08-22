"""Multi-turn chat orchestration.

Chat is a **thin layer over the same machinery**: the same retrieval pipeline,
the same answer strategy, plus conversational rewrite and history. No second
retrieval path, no second prompt family. The composition root (``api/chat.py``)
loads prior messages from the DB, calls :func:`build_history` to bound them,
then threads the bounded history into the existing pipeline + answer context —
exactly what the single-turn endpoint already does, plus ``history``.

This module owns the *pure* history policy and title derivation so it is testable
without a DB. It must not import ``db`` (``answer/`` is at its 3-dep cap); it
receives history as ``(role, content)`` string pairs the API layer already
loaded, and returns the bounded result. ``truncate``/``summarize`` are settings
(``chat.history.strategy``), not code branches scattered through the request path.

The actual retrieval + strategy run is NOT done here — it is shared with
``GET /api/answer`` via ``iter_answer_events``. This is what keeps chat from
becoming a second product with its own retrieval quirks.
"""

from __future__ import annotations

from collections.abc import Sequence


def build_history(
    prior: Sequence[tuple[str, str]],
    *,
    max_turns: int,
    strategy: str = "truncate",
) -> tuple[tuple[str, str], ...]:
    """Bound ``prior`` conversation per ``chat.history.*`` policy.

    ``prior`` is ``(role, content)`` pairs straight from the ``messages`` table,
    oldest-first. ``max_turns`` caps how many of the most recent entries survive
    into the prompt so chat history is strictly bounded. A "turn"
    here is one stored message — a simple, predictable bound; the conversational
    rewriter applies its own smaller ``max_history_turns`` cap on top, so the
    rewrite prompt is bounded twice.

    ``strategy``:

    * ``"truncate"`` — keep the most recent ``max_turns`` entries (default).
    * ``"summarize"`` — behaves as truncate until an LLM summarizer lands. The
      setting makes the upgrade a config change, not a code change.
    """
    if max_turns <= 0 or not prior:
        return ()
    bounded = list(prior)[-max_turns:]
    # ``summarize`` currently == ``truncate``; an LLM summarizer may change this.
    return tuple(bounded)


def derive_title(text: str, *, max_len: int = 60) -> str:
    """Auto-title from the first user message, truncated to ``max_len``.

    Used when a new conversation has no title yet. Collapses whitespace and
    appends an ellipsis only when the text was actually truncated — a clean
    short question becomes its own title, a long one gets a clean cut.
    """
    clean = " ".join(text.split())
    if len(clean) <= max_len:
        return clean
    return clean[:max_len].rstrip() + "…"


__all__ = ["build_history", "derive_title"]
