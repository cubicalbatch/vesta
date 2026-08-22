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

from pydantic import BaseModel

from vesta.config.capabilities import Capability
from vesta.retrieval.assemblers._shared import apply_ordering, build_result
from vesta.retrieval.contracts import Budget, PreparedQuery, RetrievalResult, ScoredPassage
from vesta.retrieval.dedup import DEFAULT_THRESHOLD, is_near_duplicate
from vesta.retrieval.registry import register

if TYPE_CHECKING:
    from vesta.retrieval.trace import Trace
    from vesta.zim.registry import ArchiveRegistry


def _word_set(text: str) -> frozenset[str]:
    return frozenset(text.lower().split())


def _similarity(a: str, b: str) -> float:
    wa, wb = _word_set(a), _word_set(b)
    if not wa or not wb:
        return 0.0
    union = wa | wb
    return len(wa & wb) / len(union) if union else 0.0


@register("context_assembler", "diverse")
class Diverse:
    """MMR selection with per-article and per-archive caps."""

    requires: ClassVar[frozenset[Capability]] = frozenset()

    class Params(BaseModel):
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
        max_score = pool[0].score or 1.0

        selected: list[ScoredPassage] = []
        per_article: dict[str, int] = {}
        per_archive: dict[int, int] = {}
        tokens_used = 0

        while pool:
            best: ScoredPassage | None = None
            best_mmr = float("-inf")
            for sp in pool:
                if per_article.get(sp.passage.path, 0) >= max_per_article:
                    continue
                if per_archive.get(sp.passage.zim_id, 0) >= self._params.max_per_archive:
                    continue
                relevance = (sp.score / max_score) if max_score else 0.0
                redundancy = max(
                    (_similarity(sp.passage.text, s.passage.text) for s in selected),
                    default=0.0,
                )
                mmr = (
                    self._params.lambda_relevance * relevance
                    - (1.0 - self._params.lambda_relevance) * redundancy
                )
                if mmr > best_mmr:
                    best_mmr = mmr
                    best = sp
            if best is None:
                break
            pool.remove(best)
            if threshold is not None and is_near_duplicate(best, selected, threshold=threshold):
                continue
            passage_tokens = len(best.passage.text.split())
            if passage_tokens + tokens_used > token_budget and selected:
                break
            selected.append(best)
            per_article[best.passage.path] = per_article.get(best.passage.path, 0) + 1
            per_archive[best.passage.zim_id] = per_archive.get(best.passage.zim_id, 0) + 1
            tokens_used += passage_tokens

        ordered = apply_ordering(selected, self._params.ordering)
        return build_result(selected, ordered, q.terms, tr, source="diverse")


__all__ = ["Diverse"]
