"""Lexical-overlap passage scorer — trivial placeholder.

Registered as ``passage_scorer`` ``lexical_overlap``. Scores each passage by the
fraction of query terms found in the passage text. This is deliberately simple —
real embedding-based scorers (the static-embedder shortlist then
cross-encoder chain) supersede it. The scorer chain pattern means the profile's
``scorers`` order controls which scores are applied, with zero pipeline code
changes.

Speed: O(passages * terms), trivial on < 200 passages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from vesta.config.capabilities import Capability
from vesta.retrieval.contracts import PreparedQuery, ScoredPassage
from vesta.retrieval.registry import ComponentParams, register

if TYPE_CHECKING:
    from vesta.retrieval.trace import Trace


@register("passage_scorer", "lexical_overlap")
class LexicalOverlap:
    """Trivial lexical overlap scorer (placeholder)."""

    requires: ClassVar[frozenset[Capability]] = frozenset()

    class Params(ComponentParams):
        """No tunable params for the lexical overlap heuristic."""

    def __init__(self, params: Params | None = None) -> None:
        self._params = params or self.Params()

    async def score(
        self, passages: list[ScoredPassage], q: PreparedQuery, tr: Trace
    ) -> list[ScoredPassage]:
        """Score each passage by the fraction of query terms present in its text.

        Scores from the previous scorer in the chain are discarded — the caller
        may persist them in the trace if desired.
        """
        terms = {t.lower() for t in q.terms if len(t) > 1}
        if not terms:
            return passages  # no scoring without meaningful terms

        scored: list[ScoredPassage] = []
        for sp in passages:
            text = sp.passage.text.lower()
            # Count unique query terms found in the passage.
            matches = sum(1 for term in terms if term in text)
            score = matches / len(terms) if terms else 0.0
            scored.append(
                ScoredPassage(
                    passage=sp.passage,
                    score=score,
                    source_info="lexical_overlap",
                )
            )
        return scored
