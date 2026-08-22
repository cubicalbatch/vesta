"""Top-K budget assembler — select passages up to a token budget.

Registered as ``context_assembler`` ``topk_budget``. Sorts by score descending,
enforces a per-article cap, near-exact dedup, and a token budget — the simplest
of the four strategies and the default profile's assembler. Scores come from
Stage B's actual scorers, snippets
are query-term-centred (``retrieval/snippets.py``), and dedup/budget selection
are the shared ``assemblers/_shared.py`` helpers every strategy uses.

Speed: O(passages²) for dedup, capped by the passage builder's ``max_passages``
(and, when the scorer chain runs, by Stage B1's shortlist — typically ≤20).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

from vesta.config.capabilities import Capability
from vesta.retrieval.assemblers._shared import apply_budget, apply_ordering, build_result
from vesta.retrieval.contracts import Budget, PreparedQuery, RetrievalResult, ScoredPassage
from vesta.retrieval.dedup import DEFAULT_THRESHOLD
from vesta.retrieval.registry import register

if TYPE_CHECKING:
    from vesta.retrieval.trace import Trace
    from vesta.zim.registry import ArchiveRegistry


@register("context_assembler", "topk_budget")
class TopKBudget:
    """Top-K budget assembler — select passages by score, cap per article, dedup."""

    requires: ClassVar[frozenset[Capability]] = frozenset()

    class Params(BaseModel):
        budget_tokens: int = 2400
        max_per_article: int = 2
        dedup: str = "near_exact"  # "near_exact" or "none"
        #: Where the strongest evidence lands in the prompt ("lost in the
        #: middle"): "score_desc" (default) or "edges" (strongest passages at
        #: both ends of the context, weakest in the middle).
        ordering: str = "score_desc"
        #: Score floor (see ``apply_budget``'s docstring) — ``max(
        #: score_floor_abs, score_floor_rel * top_score)``. Both default
        #: ``None`` (disabled — behaviour-preserving default): on a large
        #: archive the token budget already prunes the low-score tail, so this
        #: only matters on a small archive where the budget never binds.
        score_floor_abs: float | None = None
        score_floor_rel: float | None = None
        #: The floor never reduces the context below this many distinct
        #: articles, however low their scores — prune the tail, never starve
        #: the context.
        min_articles: int = 8

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

        ranked = sorted(scored, key=lambda sp: sp.score, reverse=True)
        selected = apply_budget(
            ranked,
            token_budget=token_budget,
            max_per_article=max_per_article,
            dedup_threshold=threshold,
            score_floor_abs=self._params.score_floor_abs,
            score_floor_rel=self._params.score_floor_rel,
            min_articles=self._params.min_articles,
        )
        ordered = apply_ordering(selected, self._params.ordering)
        return build_result(selected, ordered, q.terms, tr, source="topk_budget")


__all__ = ["TopKBudget"]
