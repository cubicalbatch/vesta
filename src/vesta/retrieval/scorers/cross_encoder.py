"""Stage B2 — cross-encoder rerank.

``ms-marco-MiniLM-L-6-v2`` int8, over the ~20 shortlisted passages -> ~90-180 ms.
Produces the cross-corpus-comparable score: python-libzim exposes no
scores and cross-archive rank fusion is actively harmful, so this is the only
place in the system a trustworthy cross-archive ranking exists.

Registered as ``passage_scorer`` ``cross_encoder``, requires ``CROSS_ENCODER``.
**Ships on but gated behind an A/B**: a 2026 benchmark found MiniLM-class rerankers
*degrading* nDCG on technical corpora. The A/B counterpart is a profile
that disables reranking — this scorer is measured against it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

from vesta.config.capabilities import Capability
from vesta.retrieval.contracts import PreparedQuery, ScoredPassage
from vesta.retrieval.registry import register
from vesta.retrieval.scorers._compose import passage_text

if TYPE_CHECKING:
    from vesta.encoders.manager import EncoderManager
    from vesta.retrieval.trace import Trace


@register("passage_scorer", "cross_encoder")
class CrossEncoderScorer:
    """Stage B2: cross-encoder rerank over the (already shortlisted) passages."""

    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.CROSS_ENCODER})

    class Params(BaseModel):
        #: The A/B toggle. False makes this scorer a
        #: passthrough without removing it from the profile — a quick way to
        #: disable reranking from a saved profile's param form without hand-
        #: editing YAML. The DoD's actual A/B is a profile edit
        #: (``standard`` vs ``no_rerank``); this is the same knob exposed a
        #: second way.
        enabled: bool = True
        #: Defensive bound (Traps: "an unbounded pool will hurt") in case this
        #: scorer runs without ``static_pass`` having shortlisted first — a
        #: 278M+ reranker over 200 passages is 4-7s; MiniLM-L6 over 200
        #: would still blow the 0.7s budget, so this caps it regardless of what
        #: ran before.
        candidates_max: int = 20

    def __init__(
        self,
        params: Params | None = None,
        encoders: EncoderManager | None = None,
    ) -> None:
        self._params = params or self.Params()
        self._encoders = encoders

    async def score(
        self, passages: list[ScoredPassage], q: PreparedQuery, tr: Trace
    ) -> list[ScoredPassage]:
        if not passages or not self._params.enabled or self._encoders is None:
            return passages
        encoder = await self._encoders.get_rerank()
        if encoder is None:
            return passages

        bounded = passages[: self._params.candidates_max]
        query_text = q.text or q.raw
        texts = [passage_text(sp.passage) for sp in bounded]

        scores = await encoder.score(query_text, texts)
        if not scores:
            return passages

        rescored = [
            ScoredPassage(passage=sp.passage, score=float(s), source_info="cross_encoder")
            for sp, s in zip(bounded, scores, strict=True)
        ]
        rescored.sort(key=lambda sp: sp.score, reverse=True)
        return rescored


__all__ = ["CrossEncoderScorer"]
