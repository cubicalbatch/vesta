"""Exact-title-match detection for lead passages — shared by ``static_pass``
(Stage B1, scorers/) and ``lead_boost`` (the context assembler, assemblers/).

Not a sibling pair in the strict sense ("shared logic
moves down into a function in the package, not imported between sibling
implementations" was written about same-directory implementations), but the
"find the matching lead" step is byte-identical in both, so it lives here once
rather than twice — the same pattern ``dedup.py``'s ``is_near_duplicate`` uses
for ``assemblers/_shared.py``. What each caller DOES with the match differs
(``static_pass`` appends it to survive a shortlist truncation it would
otherwise be cut by; ``lead_boost`` moves it to the front of an already-
selected context), so only this common core is factored out.

No candidate source, fuser, or scorer anywhere in the pipeline compares the
query string to an article's title (checked ``title_suggest``, ``xapian_fts``,
``lexical_overlap``, ``rrf``, the static embedder, the cross-encoder — none
special-case it), so a query like "fire" against an archive with an article
literally titled "Fire" gets no credit for that exact match on its own; this
helper is what both places that DO act on it share.

``breadcrumb`` is built as ``" > ".join([article.title, *heading_path])``
(``zim/passages.py``), so its FIRST segment is always exactly the article
title regardless of what follows — no separate title field is needed. Real ZIM
articles commonly render their own name as an ``<h1>`` at the top of the body,
which ``extract.py``'s heading walk turns into a section covering the lead, so
a real lead's ``heading_path`` is often ``(title,)`` rather than empty and its
breadcrumb is ``"Title > Title"``, not bare ``"Title"`` (confirmed against the
live "Fire"/"Earth"/etc. archives — every observed lead breadcrumb there is
doubled this way). Matching only the first segment handles both that common
case and the bare-title case the same way.
"""

from __future__ import annotations

from collections.abc import Sequence

from vesta.retrieval.contracts import ScoredPassage


def find_exact_title_lead(
    candidates: Sequence[ScoredPassage], query_text: str
) -> ScoredPassage | None:
    """Return the highest-scoring ``is_lead`` passage among ``candidates`` whose
    breadcrumb's first segment exactly matches ``query_text`` (case/whitespace
    -insensitive), or ``None`` if there is no such passage.

    More than one archive can hold an identically-titled article (e.g. an
    unscoped search) — ties are broken by highest score, a simple, sensible
    choice each caller would otherwise have to reimplement identically.
    """
    query_norm = query_text.strip().lower()
    if not query_norm:
        return None

    matches = [
        sp
        for sp in candidates
        if sp.passage.is_lead
        and sp.passage.breadcrumb.split(" > ", 1)[0].strip().lower() == query_norm
    ]
    if not matches:
        return None
    return max(matches, key=lambda sp: sp.score)


__all__ = ["find_exact_title_lead"]
