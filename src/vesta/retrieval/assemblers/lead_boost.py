"""Lead-boost assembler — boost the lead passage of a strongly-matching article,
and (optionally) guarantee it a seat regardless of rank.

Registered as ``context_assembler`` ``lead_boost``. Wikipedia's lead paragraph
*is* a human-written summary, so this boosts the score of any already-shortlisted
passage flagged ``is_lead`` (``zim/passages.py``: "flagged, not boosted" at
extraction time — boosting is retrieval policy, and this is where it belongs)
rather than paying for a RAPTOR-style LLM summary pass over millions of
articles. **Zero extra cost**: no additional embedding, scoring, or ZIM read —
it only reweights passages Stage B already scored and retained. If a lead
passage did not survive Stage B's scorer chain (e.g. B1 shortlisted it out),
it is not resurrected here; this is a reweighting strategy, not a fetch.

**A percentage boost cannot rescue a lead passage the cross-encoder scored
very low.** Live-observed case (2026-08-01, real archive, "What was Eminem's
first album" over a 66 KB/17-passage article): the lead passage's raw score
was 0.27 against other passages scoring 0.95-0.999 — the article's infobox
metadata (name variants, birth date, occupations list) sits ahead of the
useful sentence in the same ~400-token chunk and reads nothing like natural
prose to a small reranker. No boost multiplier under roughly 3x closes a gap
that size, and a multiplier that large would distort ordinary queries where
the lead genuinely isn't the answer. ``guarantee_top_lead`` (default ``True``)
is a separate, bounded mechanism for exactly this shape of miss: for each of
the top-``guarantee_top_lead_k`` articles (by pre-boost score), if that
article has an ``is_lead`` passage among the Stage-B-scored candidates that
the normal budget/cap/dedup selection did NOT choose, it is added as one
extra passage — bypassing ``max_per_article`` and the token budget for that
one passage only (not the boost's score, which still applies to it and
everything else).

The guarantee was originally scoped to the single top-scoring article only,
but the motivating bug itself exposed that as too narrow: on "What was
Eminem's first album" the fact ("Infinite", 1996) lives in article **#2's**
lead (the Eminem article, score 0.999), while article **#1** (Encore, score
1.10) held the irrelevant guarantee slot. The default ``guarantee_top_lead_k
= 3`` covers the top three articles — bounded blast radius (at most three
extra passages, each from a distinct article), large enough to reach a
runner-up article whose lead the reranker buried. Lower it to 1 to reproduce
the original top-1-only behaviour.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from vesta.config.capabilities import Capability
from vesta.retrieval.assemblers._shared import apply_budget, apply_ordering, build_result
from vesta.retrieval.contracts import Budget, PreparedQuery, RetrievalResult, ScoredPassage
from vesta.retrieval.dedup import DEFAULT_THRESHOLD
from vesta.retrieval.registry import ComponentParams, register
from vesta.retrieval.title_match import find_exact_title_lead

if TYPE_CHECKING:
    from vesta.retrieval.trace import Trace
    from vesta.zim.registry import ArchiveRegistry


@register("context_assembler", "lead_boost")
class LeadBoost:
    """Boost lead-section passages, then select like ``topk_budget``."""

    requires: ClassVar[frozenset[Capability]] = frozenset()

    class Params(ComponentParams):
        budget_tokens: int = 2400
        max_per_article: int = 2
        dedup: str = "near_exact"
        ordering: str = "score_desc"
        #: Fractional score boost applied to lead passages (0.15 = +15%).
        boost: float = 0.15
        #: Master switch for the top-K lead guarantee (see below). When False,
        #: no lead is force-added regardless of ``guarantee_top_lead_k``.
        guarantee_top_lead: bool = True
        #: Guarantee the lead passage of each of the top-K articles (by
        #: pre-boost score) a seat even if it didn't survive the normal
        #: budget/cap/dedup selection (see module docstring — a percentage
        #: boost alone cannot rescue a passage the cross-encoder scored far
        #: out of contention). Bounded to at most K extra passages, each from
        #: a distinct top-scoring article. 1 reproduces the original top-1-only
        #: behaviour; the default 3 reaches a runner-up article whose lead the
        #: reranker buried (the motivating Eminem case, where the fact lived
        #: in article #2's lead).
        guarantee_top_lead_k: int = 3
        #: Score floor (see ``apply_budget``'s docstring) — ``max(
        #: score_floor_abs, score_floor_rel * top_score)``. Both default
        #: ``None`` (disabled — behaviour-preserving default). Applies only to
        #: the normal budget/cap/dedup selection below; a guaranteed lead
        #: (``guarantee_top_lead``) bypasses it exactly as it already bypasses
        #: ``max_per_article`` and the token budget — it is a deliberate,
        #: bounded rescue for a specific known miss shape, not ordinary
        #: tail content the floor is meant to prune.
        score_floor_abs: float | None = None
        score_floor_rel: float | None = None
        #: The floor never reduces the context below this many distinct
        #: articles, however low their scores — prune the tail, never starve
        #: the context.
        min_articles: int = 8
        #: Force the lead passage of an article whose title exactly matches
        #: the query text (case/whitespace-insensitive) to rank first,
        #: regardless of its Stage-B score. Unconditional — independent of
        #: ``guarantee_top_lead_k``'s top-K-by-pre-boost-score gate. Motivating
        #: case: searching "fire" against an archive with an article titled
        #: exactly "Fire" — narrower articles ("Fire department", "Fire
        #: engine") can legitimately out-score it under semantic similarity
        #: alone for a bare keyword query, so it never lands in the top-3 by
        #: score and the ordinary guarantee never triggers for it. No signal
        #: upstream of this assembler (FTS, title-suggest, lexical overlap,
        #: RRF, the cross-encoder) ever compares the query string to the
        #: article title, so this is the only place such a match can be acted
        #: on.
        exact_title_boost: bool = True

    def __init__(
        self,
        params: Params | None = None,
        archives: ArchiveRegistry | None = None,
    ) -> None:
        self._params = params or self.Params()
        self._archives = archives

    def assemble(
        self, scored: list[ScoredPassage], budget: Budget, q: PreparedQuery, tr: Trace
    ) -> RetrievalResult:
        token_budget = min(budget.token_total, self._params.budget_tokens)
        max_per_article = min(budget.max_per_article, self._params.max_per_article)
        threshold = DEFAULT_THRESHOLD if self._params.dedup == "near_exact" else None

        boosted = [
            ScoredPassage(
                passage=sp.passage,
                # Sign-safe: the boost must move a lead UP the ranking in any
                # score regime. ``score * (1 + boost)`` makes a NEGATIVE score
                # more negative — a penalty, not a boost — because raw
                # static-encoder cosines can be negative whenever the
                # cross-encoder stage is unmet/disabled (the sigmoid is what
                # normally rescales scores into (0, 1)). Adding
                # ``boost * |score|`` boosts toward zero instead; for
                # non-negative scores this is exactly the old multiplier.
                score=sp.score + abs(sp.score) * self._params.boost,
                source_info=f"{sp.source_info}+lead_boost",
            )
            if sp.passage.is_lead
            else sp
            for sp in scored
        ]
        ranked = sorted(boosted, key=lambda sp: sp.score, reverse=True)
        selected = apply_budget(
            ranked,
            token_budget=token_budget,
            max_per_article=max_per_article,
            dedup_threshold=threshold,
            score_floor_abs=self._params.score_floor_abs,
            score_floor_rel=self._params.score_floor_rel,
            min_articles=self._params.min_articles,
        )

        if self._params.guarantee_top_lead and scored:
            # Guaranteed leads are added below via `_guarantee_top_k_article_
            # leads`, which appends directly to `selected` — never re-entering
            # `apply_budget` — so they bypass the score floor along with the
            # cap/dedup/budget it already bypassed (module docstring: a
            # deliberate, bounded rescue, not ordinary tail content).
            # The "top articles" are determined from the PRE-boost scores, not
            # `ranked`: boosting a lead passage can push IT above an article
            # that's actually the strongest match (e.g. a modest article whose
            # own lead gets boosted past a longer, more relevant article's
            # un-boosted body passages), which would make the guarantee target
            # — and trivially no-op on — the wrong article. "Top article" means
            # "what Stage B found most relevant before this component reweighted
            # anything", matching the module docstring's framing.
            k = max(1, self._params.guarantee_top_lead_k)
            selected = _guarantee_top_k_article_leads(scored, ranked, selected, k=k)

        if self._params.exact_title_boost and scored:
            # Unconditional — not gated by `guarantee_top_lead_k`'s top-K gate.
            # An exact title match is direct evidence the article IS the
            # answer, not merely a semantic-similarity signal that has to
            # clear a score-based cutoff to be trusted.
            selected = _guarantee_exact_title_match(ranked, selected, q)

        ordered = apply_ordering(selected, self._params.ordering)
        return build_result(selected, ordered, q.terms, tr, source="lead_boost")


def _guarantee_top_k_article_leads(
    scored: list[ScoredPassage],
    ranked: list[ScoredPassage],
    selected: list[ScoredPassage],
    *,
    k: int,
) -> list[ScoredPassage]:
    """Guarantee the lead passage of each of the top-K articles (by PRE-boost
    max passage score) a seat, if that lead is among ``ranked`` but was not
    chosen by the normal budget/cap/dedup selection.

    Generalizes the original top-1 guarantee. The motivating case (2026-08-01,
    "What was Eminem's first album"): the fact ("Infinite", 1996) lived in
    article #2's lead (score 0.27, buried by infobox metadata), while article
    #1's lead was irrelevant — so a top-1 guarantee rescued the wrong article.
    Each guaranteed lead bypasses ``max_per_article`` for that one passage
    only; the blast radius is bounded to at most K extra passages, each from a
    distinct article.
    """
    # Top-K articles by PRE-boost max passage score (same rationale as the
    # original: boosting a lead can re-rank, so "top article" must be decided
    # before this component reweights anything).
    article_top: dict[tuple[int, str], float] = {}
    for sp in scored:
        key = (sp.passage.zim_id, sp.passage.path)
        if sp.score > article_top.get(key, -1.0):
            article_top[key] = sp.score
    top_keys = frozenset(
        key for key, _ in sorted(article_top.items(), key=lambda kv: kv[1], reverse=True)[:k]
    )

    already = {(sp.passage.zim_id, sp.passage.path, sp.passage.ordinal) for sp in selected}
    guaranteed_articles: set[tuple[int, str]] = set()
    extra: list[ScoredPassage] = []
    # `ranked` is boosted-score-sorted desc, so the first is_lead passage we
    # meet for each article is its best lead — one per article, at most K.
    for sp in ranked:
        key = (sp.passage.zim_id, sp.passage.path)
        if (
            key in top_keys
            and key not in guaranteed_articles
            and sp.passage.is_lead
            and (sp.passage.zim_id, sp.passage.path, sp.passage.ordinal) not in already
        ):
            extra.append(sp)
            guaranteed_articles.add(key)
    if not extra:
        return selected
    return sorted([*selected, *extra], key=lambda sp: sp.score, reverse=True)


def _guarantee_exact_title_match(
    ranked: list[ScoredPassage],
    selected: list[ScoredPassage],
    q: PreparedQuery,
) -> list[ScoredPassage]:
    """Force the lead passage of an article whose title exactly matches the
    query text to rank first, regardless of Stage-B score.

    Matching itself (breadcrumb-first-segment vs. normalized query text) is
    factored into :func:`vesta.retrieval.title_match.find_exact_title_lead`,
    shared with ``static_pass`` (Stage B1), which runs its own, earlier
    instance of this same guarantee so a lead this narrow a query would
    otherwise shortlist out of Stage B entirely survives long enough to reach
    here at all — see that module for the full rationale (no candidate
    source, fuser, or scorer anywhere in the pipeline compares the query
    string to an article's title on its own). This is a reweighting/
    reordering strategy like the rest of the file, not a fetch (module
    docstring): only a lead passage that already reached ``scored``/
    ``ranked`` can be acted on here.
    """
    winner = find_exact_title_lead(ranked, q.text)
    if winner is None:
        return selected

    winner_key = (winner.passage.zim_id, winner.passage.path, winner.passage.ordinal)
    rest = [
        sp
        for sp in selected
        if (sp.passage.zim_id, sp.passage.path, sp.passage.ordinal) != winner_key
    ]
    return [winner, *rest]


__all__ = ["LeadBoost"]
