"""Section-window assembler — expand a winning passage to its section neighbours
(the "chunks vs whole articles" tradeoff).

Registered as ``context_assembler`` ``section_window``. The passage builder
(``candidate_articles.py``) already splits and includes *every* passage of each
candidate article, not just query-matched ones — so a winning passage's
neighbours are already sitting in ``scored`` (or were dropped by an earlier
Stage B scorer's shortlist, in which case they are simply unavailable; no extra
ZIM read happens here). Expanding to neighbours trades a wider per-winner token
cost for less risk of truncating a winning passage mid-thought, at the cost of
fewer distinct articles represented for the same budget.

``Params.max_per_article`` here caps *winning sections* per article (each
winner may pull in up to ``2 * window`` extra passages), not raw passage count
— a deliberate difference from ``topk_budget``, documented so a profile author
does not assume the two knobs mean the same thing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

from vesta.config.capabilities import Capability
from vesta.retrieval.assemblers._shared import apply_ordering, build_result
from vesta.retrieval.contracts import Budget, PreparedQuery, RetrievalResult, ScoredPassage
from vesta.retrieval.dedup import DEFAULT_THRESHOLD, NearDuplicateGate
from vesta.retrieval.registry import register

if TYPE_CHECKING:
    from vesta.retrieval.trace import Trace
    from vesta.zim.registry import ArchiveRegistry

_PassageKey = tuple[int, str, int]  # (zim_id, path, ordinal)


@register("context_assembler", "section_window")
class SectionWindow:
    """Expand each winning passage to its same-section ordinal neighbours."""

    requires: ClassVar[frozenset[Capability]] = frozenset()

    class Params(BaseModel):
        budget_tokens: int = 2400
        max_per_article: int = 2  # winning SECTIONS per article, see module docstring
        dedup: str = "near_exact"
        ordering: str = "score_desc"
        window: int = 1  # neighbour passages pulled in on each side

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

        by_key: dict[_PassageKey, ScoredPassage] = {
            (sp.passage.zim_id, sp.passage.path, sp.passage.ordinal): sp for sp in scored
        }
        ranked = sorted(scored, key=lambda sp: sp.score, reverse=True)

        expanded: list[ScoredPassage] = []
        seen: set[_PassageKey] = set()
        dedup_gate = NearDuplicateGate(threshold) if threshold is not None else None
        per_article: dict[str, int] = {}
        tokens_used = 0

        for sp in ranked:
            key = (sp.passage.zim_id, sp.passage.path, sp.passage.ordinal)
            if key in seen:
                continue
            if per_article.get(sp.passage.path, 0) >= max_per_article:
                continue
            if dedup_gate is not None and dedup_gate.is_near_duplicate(sp):
                continue
            group = self._window_group(sp, by_key)
            group_tokens = sum(len(g.passage.text.split()) for g in group)
            if group_tokens + tokens_used > token_budget and expanded:
                break
            for g in group:
                gkey = (g.passage.zim_id, g.passage.path, g.passage.ordinal)
                if gkey in seen:
                    continue
                seen.add(gkey)
                expanded.append(g)
                # neighbours are part of the selection later winners are
                # deduped against, so the gate tracks them too.
                if dedup_gate is not None:
                    dedup_gate.accept(g)
            per_article[sp.passage.path] = per_article.get(sp.passage.path, 0) + 1
            tokens_used += group_tokens

        ordered = apply_ordering(expanded, self._params.ordering)
        return build_result(expanded, ordered, q.terms, tr, source="section_window")

    def _window_group(
        self, sp: ScoredPassage, by_key: dict[_PassageKey, ScoredPassage]
    ) -> list[ScoredPassage]:
        """``sp`` plus up to ``window`` ordinal neighbours on each side that
        share its breadcrumb (same section) and are present in ``by_key``."""
        group = [sp]
        for delta in range(1, self._params.window + 1):
            for direction in (-1, 1):
                nk = (sp.passage.zim_id, sp.passage.path, sp.passage.ordinal + direction * delta)
                neighbour = by_key.get(nk)
                if neighbour is not None and neighbour.passage.breadcrumb == sp.passage.breadcrumb:
                    group.append(neighbour)
        group.sort(key=lambda g: g.passage.ordinal)
        return group


__all__ = ["SectionWindow"]
