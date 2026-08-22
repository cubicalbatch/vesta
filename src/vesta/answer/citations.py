"""Citation-span synthesis for the agent answer path.

The agent (``api/agent_chat.py``) prompts the model with cards numbered
``[1]``..``[n]`` (first-seen card order, 1-based), so a ``[n]`` marker in the
answer is already a valid card number — no passage→card renumbering or n-gram
span alignment is needed. :func:`synthesize_citation_spans` turns each marker
into a :class:`~vesta.answer.contracts.CitationSpan` for the ``citations`` SSE
event.

Grounded by construction: cards only ever come from ``search`` /
``read_article``, so a citation to something unretrieved cannot exist; a marker
outside the valid card range is dropped rather than relabelled.

This module is pure string processing — no domain objects, no imports from
``retrieval`` or ``inference``.
"""

from __future__ import annotations

import re

from vesta.answer.contracts import CitationSpan

#: Matches an inline citation marker: digits only inside brackets. Deliberately
#: narrow so it never touches non-citation bracketed text — e.g. ``focus.py``'s
#: ``"[...]"`` elision marker (no digits) or a stray ``"[1a]"`` (trailing
#: non-digit) simply do not match.
_INLINE_CITATION_RE = re.compile(r"\[(\d+)\]")


def synthesize_citation_spans(answer: str, card_count: int) -> list[CitationSpan]:
    """One :class:`CitationSpan` per ``[n]`` marker in ``answer``.

    The agent path prompts the model with cards numbered ``[1]``..``[card_count]``
    (``agent_chat.py`` — first-seen card order, 1-based), so the model's ``[n]``
    markers are already card numbers. The span's ``source_index`` (0-based, the
    ``sources.cards`` index) is therefore ``n - 1``.

    ``passage_start``/``passage_end`` are ``None`` (document-level alignment only
    — the click jumps to the reader for the whole article) and each span carries
    a trivial ``score`` of 1.0 (the marker is authoritative, not an n-gram
    estimate).

    A marker outside the valid card range (``0 <= n - 1 < card_count``) is
    dropped from the result — the model cited a card that was never retrieved, so
    there is nothing to align it to.
    """
    spans: list[CitationSpan] = []
    for match in _INLINE_CITATION_RE.finditer(answer):
        n = int(match.group(1))
        card_id = n - 1
        if not (0 <= card_id < card_count):
            continue
        spans.append(
            CitationSpan(
                answer_start=match.start(),
                answer_end=match.end(),
                source_index=card_id,
                passage_start=None,
                passage_end=None,
                score=1.0,
            )
        )
    return spans


__all__ = ["synthesize_citation_spans"]
