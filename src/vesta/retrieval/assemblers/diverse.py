"""Diverse assembler — MMR-style selection, caps per-article and per-archive
concentration.

Registered as ``context_assembler`` ``diverse``. Standard Maximal Marginal
Relevance: greedily pick the passage maximizing
``lambda * relevance - (1 - lambda) * max_similarity_to_already_selected``,
so a highly-relevant-but-redundant passage loses to a slightly-less-relevant
one that actually adds new information. Similarity uses word-unigram Jaccard —
cheap and adequate for a diversity *penalty* (as opposed to
``dedup.is_near_duplicate``'s bigram Jaccard, tuned to be a strict near-exact
*equality* test).

``max_per_archive`` (on top of ``max_per_article``) is this assembler's second
axis: on a multi-archive search, MMR alone still lets one strong archive
dominate every slot, which defeats "diverse" as a strategy name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from vesta.config.capabilities import Capability
from vesta.retrieval.assemblers._shared import apply_ordering, build_result
from vesta.retrieval.contracts import Budget, PreparedQuery, RetrievalResult, ScoredPassage
from vesta.retrieval.dedup import DEFAULT_THRESHOLD, NearDuplicateGate
from vesta.retrieval.registry import ComponentParams, register

if TYPE_CHECKING:
    from vesta.retrieval.trace import Trace
    from vesta.zim.registry import ArchiveRegistry


def _word_set(text: str) -> frozenset[str]:
    return frozenset(text.lower().split())


def _similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """Word-unigram Jaccard over precomputed word sets (each text is split
    exactly once per ``assemble`` call, not once per candidate x selected pair)."""
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


@register("context_assembler", "diverse")
class Diverse:
    """MMR selection with per-article and per-archive caps."""

    requires: ClassVar[frozenset[Capability]] = frozenset()

    class Params(ComponentParams):
        budget_tokens: int = 2400
        max_per_article: int = 2
        max_per_archive: int = 4
        dedup: str = "near_exact"
        ordering: str = "score_desc"
        #: MMR trade-off: 1.0 = pure relevance (identical to topk_budget's
        #: ranking), 0.0 = pure diversity (ignores relevance entirely).
        lambda_relevance: float = 0.7

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

        pool = sorted(scored, key=lambda sp: sp.score, reverse=True)
        if not pool:
            return build_result([], [], q.terms, tr, source="diverse")
        max_score = pool[0].score
        min_score = pool[-1].score

        selected: list[ScoredPassage] = []
        selected_sets: list[frozenset[str]] = []
        dedup_gate = NearDuplicateGate(threshold) if threshold is not None else None
        per_article: dict[tuple[int, str], int] = {}
        per_archive: dict[int, int] = {}
        tokens_used = 0
        pool_sets = [_word_set(sp.passage.text) for sp in pool]

        while pool:
            best_idx: int | None = None
            best_mmr = float("-inf")
            for i, (sp, sp_set) in enumerate(zip(pool, pool_sets, strict=True)):
                article_key = (sp.passage.zim_id, sp.passage.path)
                if per_article.get(article_key, 0) >= max_per_article:
                    continue
                if per_archive.get(sp.passage.zim_id, 0) >= self._params.max_per_archive:
                    continue
                # Relevance must be monotone-increasing in the raw score in
                # any sign regime. ``score / max_score`` INVERTS when every
                # score is negative (the CROSS_ENCODER-unmet chain ends at
                # static_pass's raw cosines, which can be negative): the
                # worst passage then gets the largest quotient and MMR picks
                # it first. In that regime fall back to a min-max shift,
                # which maps best → 1.0 and worst → 0.0 regardless of sign.
                if max_score > 0.0:
                    relevance = sp.score / max_score
                else:
                    span = max_score - min_score
                    relevance = (sp.score - min_score) / span if span else 1.0
                redundancy = max(
                    (_similarity(sp_set, s_set) for s_set in selected_sets),
                    default=0.0,
                )
                mmr = (
                    self._params.lambda_relevance * relevance
                    - (1.0 - self._params.lambda_relevance) * redundancy
                )
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            if best_idx is None:
                break
            # Strict ``>`` above means the scan lands on the FIRST occurrence
            # of the winning MMR — exactly the element ``list.remove`` would
            # drop even among equal-valued passages.
            best = pool.pop(best_idx)
            best_set = pool_sets.pop(best_idx)
            if dedup_gate is not None and dedup_gate.is_near_duplicate(best):
                continue
            passage_tokens = len(best.passage.text.split())
            if passage_tokens + tokens_used > token_budget and selected:
                break
            selected.append(best)
            selected_sets.append(best_set)
            if dedup_gate is not None:
                dedup_gate.accept(best)
            best_article_key = (best.passage.zim_id, best.passage.path)
            per_article[best_article_key] = per_article.get(best_article_key, 0) + 1
            per_archive[best.passage.zim_id] = per_archive.get(best.passage.zim_id, 0) + 1
            tokens_used += passage_tokens

        ordered = apply_ordering(selected, self._params.ordering)
        return build_result(selected, ordered, q.terms, tr, source="diverse")


__all__ = ["Diverse"]
