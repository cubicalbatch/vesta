"""Shared assembler helpers: shared logic moves down into a
plain function, never imported sibling-to-sibling.

Every assembler needs the same three things — budget/cap/dedup selection,
source-card construction with real snippets, and confidence-signal computation
— so they live once here instead of four times.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vesta.retrieval.contracts import ConfidenceSignals, RetrievalResult, ScoredPassage, SourceCard
from vesta.retrieval.dedup import DEFAULT_THRESHOLD, is_near_duplicate
from vesta.retrieval.snippets import build_snippet

if TYPE_CHECKING:
    from vesta.retrieval.trace import Trace


def apply_budget(
    ranked: Sequence[ScoredPassage],
    *,
    token_budget: int,
    max_per_article: int,
    dedup_threshold: float | None = DEFAULT_THRESHOLD,
    score_floor_abs: float | None = None,
    score_floor_rel: float | None = None,
    min_articles: int = 8,
) -> list[ScoredPassage]:
    """Walk ``ranked`` (best-first) applying the per-article cap, dedup, an
    optional score floor, and the token budget in one pass — the shared core
    of every assembler's selection step. ``dedup_threshold=None`` disables
    dedup entirely.

    The score floor (opt-in, keyword-only, behaviour-preserving default —
    both ``score_floor_abs``/``score_floor_rel`` default ``None``, meaning
    disabled) exists for the shape of bug a token budget alone never catches:
    on a SMALL archive the budget never binds, so every candidate the upstream
    funnel returned — including score-0.0000 tag/disambiguation pages — enters
    the assembled context and dilutes it (large-archive runs never see this,
    because the budget truncates the tail first; see the ``min_articles``
    no-starvation guarantee below and the "no-op when the budget binds first"
    behaviour this implies).

    When enabled, the floor is ``max(score_floor_abs, score_floor_rel *
    top_score)``, where ``top_score`` is ``ranked[0].score`` — the top score
    of the RANKED input, i.e. computed once, before any selection happens, not
    recomputed against whatever ends up selected. A passage that would START A
    NEW ARTICLE'S presence in ``selected`` and scores below the floor is
    skipped — but ONLY once ``min_articles`` distinct articles are already
    selected. This is a deliberate "prune the tail, never the body" policy:
    the floor cannot reduce the context below ``min_articles`` distinct
    articles no matter how low their scores are (never starve the context),
    and it never re-examines a passage from an article already admitted (that
    article's bar was already cleared by an earlier, higher-scoring passage of
    its own).
    """
    floor: float | None = None
    if score_floor_abs is not None or score_floor_rel is not None:
        top_score = ranked[0].score if ranked else 0.0
        abs_part = score_floor_abs if score_floor_abs is not None else 0.0
        rel_part = (score_floor_rel * top_score) if score_floor_rel is not None else 0.0
        floor = max(abs_part, rel_part)

    selected: list[ScoredPassage] = []
    per_article: dict[str, int] = {}
    tokens_used = 0
    for sp in ranked:
        key = sp.passage.path
        if per_article.get(key, 0) >= max_per_article:
            continue
        if dedup_threshold is not None and is_near_duplicate(
            sp, selected, threshold=dedup_threshold
        ):
            continue
        if (
            floor is not None
            and sp.score < floor
            and key not in per_article
            and len(per_article) >= min_articles
        ):
            continue
        passage_tokens = len(sp.passage.text.split())
        if passage_tokens + tokens_used > token_budget and selected:
            break
        selected.append(sp)
        per_article[key] = per_article.get(key, 0) + 1
        tokens_used += passage_tokens
    return selected


def apply_ordering(selected: Sequence[ScoredPassage], ordering: str) -> list[ScoredPassage]:
    """Arrange already-selected (best-first) passages for the prompt ("lost in
    the middle"). ``"score_desc"`` (default) keeps rank order as-is —
    simplest, and correct when the model attends to the start of the context.
    ``"edges"`` alternates strongest-first/strongest-last so the two passages a
    "lost in the middle" model is most likely to actually read are both
    high-relevance, at the cost of putting rank-2 last rather than second."""
    if ordering != "edges" or len(selected) <= 2:
        return list(selected)
    front: list[ScoredPassage] = []
    back: list[ScoredPassage] = []
    for i, sp in enumerate(selected):
        if i % 2 == 0:
            front.append(sp)
        else:
            back.insert(0, sp)
    return front + back


def build_cards(
    selected: Sequence[ScoredPassage], query_terms: Sequence[str], *, source: str
) -> tuple[SourceCard, ...]:
    """One :class:`SourceCard` per distinct article among ``selected``, ordered
    by each article's first appearance, built from that article's *highest*-
    scoring passage — not necessarily the first one encountered. ``selected``
    need not be score-sorted: ``section_window`` interleaves same-section
    neighbours in ordinal order, so an article's first-appearing passage there
    can be a low-scoring neighbour rather than the winning passage, which would
    otherwise leak into the card's displayed score/snippet/title."""
    best_by_key: dict[tuple[int, str], ScoredPassage] = {}
    order: list[tuple[int, str]] = []
    for sp in selected:
        key = (sp.passage.zim_id, sp.passage.path)
        if key not in best_by_key:
            order.append(key)
            best_by_key[key] = sp
        elif sp.score > best_by_key[key].score:
            best_by_key[key] = sp

    cards: list[SourceCard] = []
    for key in order:
        sp = best_by_key[key]
        title = sp.passage.breadcrumb.split(" > ")[0] if sp.passage.breadcrumb else sp.passage.path
        cards.append(
            SourceCard(
                zim_id=sp.passage.zim_id,
                path=sp.passage.path,
                title=title,
                snippet=build_snippet(sp.passage.text, query_terms),
                breadcrumb=sp.passage.breadcrumb,
                score=sp.score,
                source=source,
            )
        )
    return tuple(cards)


def compute_confidence(passages: Sequence[ScoredPassage]) -> ConfidenceSignals:
    """Confidence signals from the final scored passages (recorded, not acted
    on, until abstention gates consume them). The
    upstream pipeline overwrites ``agreement`` with the pre-fusion Jaccard
    signal, which no assembler can see (``pipeline.py``'s Step 6 comment).

    ``passages`` need not be score-sorted — ``section_window`` returns its
    winning passages interleaved with same-section neighbours in ordinal
    order, not score order, so the dropoff ratio is computed against a locally
    sorted copy rather than assuming ``passages[0]`` is the top score."""
    if not passages:
        return ConfidenceSignals(top_score=None, score_dropoff=None, density=0.0, agreement=0.0)

    scores = sorted((sp.score for sp in passages), reverse=True)
    top_score = scores[0]

    if len(scores) >= 2:
        n = min(5, len(scores))
        top = scores[0]
        score_dropoff = (scores[n - 1] / top) if top > 0 else None
    else:
        score_dropoff = None

    per_article: dict[str, int] = {}
    for sp in passages:
        per_article[sp.passage.path] = per_article.get(sp.passage.path, 0) + 1
    max_count = max(per_article.values()) if per_article else 0
    density = max_count / len(passages) if passages else 0.0

    return ConfidenceSignals(
        top_score=top_score, score_dropoff=score_dropoff, density=density, agreement=0.0
    )


def build_result(
    selected: Sequence[ScoredPassage],
    ordered: Sequence[ScoredPassage],
    query_terms: Sequence[str],
    tr: Trace,
    *,
    source: str,
) -> RetrievalResult:
    """Assemble the final :class:`RetrievalResult`. ``selected`` (score-desc)
    drives cards + confidence; ``ordered`` (post ``apply_ordering``) is what the
    LLM prompt actually sees — the two can differ ("lost in the middle")."""
    return RetrievalResult(
        passages=tuple(ordered),
        cards=build_cards(selected, query_terms, source=source),
        trace=tr,
        confidence=compute_confidence(selected),
    )


__all__ = ["apply_budget", "apply_ordering", "build_cards", "build_result", "compute_confidence"]
