"""Stage B1 — static-model shortlist.

Scores **all** candidate passages (~200) with the Fast-tier static embedder
(``potion-retrieval-32M`` class) at 3000-6000 chunks/s -> ~50 ms, then keeps the
top ``shortlist`` (default 20) by cosine similarity. This is why Stage B fits
the 0.7 s source-card budget at all: scoring 200 passages with the bi-encoder
would cost ~1.7 s; with a cross-encoder it is far worse.

Registered as ``passage_scorer`` ``static_pass``, requires ``STATIC_ENCODER``.
When the capability is unmet (no static model on disk), the pipeline drops this
stage and the chain falls through to whatever scorer runs next — the
``lexical_overlap`` placeholder if a profile lists it, or the unscored
``"initial"`` 0.0 passthrough otherwise (degrade-don't-fail).

**A bare keyword query can shortlist its own best answer out of Stage B
entirely.** Live-observed case (2026-08-04, real archive, query "fire" over an
archive containing an article titled exactly "Fire"): the cosine-similarity
model ranks several of the article's own BODY sections, and other articles'
passages, above its broad, general LEAD paragraph — a short, keyword-dense
sentence embeds more similarly to a bare single-word query than a lead's
scene-setting prose does. The lead is cut by the ``shortlist`` truncation
below and never reaches ``cross_encoder`` (Stage B2), which only re-ranks
whatever survives here — so ``lead_boost``'s own exact-title guarantee (the
context assembler, downstream) has nothing to promote; a passage that never
reaches ``scored`` cannot be resurrected later (see that module's docstring).
``exact_title_guarantee`` (default ``True``) fixes this at the source: a lead
passage whose article title exactly matches the query text survives this
stage's truncation regardless of its cosine rank, so it is at least given to
the cross-encoder to judge on its merits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

from vesta.config.capabilities import Capability
from vesta.retrieval.contracts import PreparedQuery, ScoredPassage
from vesta.retrieval.registry import register
from vesta.retrieval.scorers._compose import passage_text
from vesta.retrieval.title_match import find_exact_title_lead

if TYPE_CHECKING:
    from vesta.encoders.manager import EncoderManager
    from vesta.retrieval.trace import Trace


@register("passage_scorer", "static_pass")
class StaticPass:
    """Stage B1: static-embedder cosine-similarity shortlist."""

    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.STATIC_ENCODER})

    class Params(BaseModel):
        #: Defensive bound on the incoming passage count before embedding
        #: (Traps: "an unbounded pool will hurt"; the passage builder already
        #: caps at ~200, this is a second, cheap bound in case a future
        #: builder does not).
        candidates_max: int = 200
        #: Output shortlist size fed to Stage B2.
        shortlist: int = 20
        #: Guarantee the lead passage of an article whose title exactly
        #: matches the query text (case/whitespace-insensitive) a seat in the
        #: returned shortlist, even if its cosine rank here would otherwise
        #: cut it (module docstring — a bare keyword query can rank a
        #: keyword-dense body section above the article's own broad lead).
        #: Bypasses ONLY this stage's ``shortlist`` truncation, as one extra
        #: passage appended (never evicting anything the shortlist already
        #: chose) — ``cross_encoder`` still scores it like any other
        #: candidate; this stage does not decide it's the answer, only that
        #: it deserves a look.
        exact_title_guarantee: bool = True

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
        if not passages or self._encoders is None:
            return passages
        encoder = await self._encoders.get_static()
        if encoder is None:
            return passages

        bounded = passages[: self._params.candidates_max]
        query_text = q.text or q.raw
        texts = [passage_text(sp.passage) for sp in bounded]

        qvec = await encoder.embed([query_text], kind="query")
        pvecs = await encoder.embed(texts, kind="passage")
        if qvec.shape[0] == 0 or pvecs.shape[0] == 0:
            return passages
        sims = (pvecs @ qvec[0]).astype(float)  # both L2-normalized -> cosine

        rescored = [
            ScoredPassage(passage=sp.passage, score=float(s), source_info="static_pass")
            for sp, s in zip(bounded, sims, strict=True)
        ]
        rescored.sort(key=lambda sp: sp.score, reverse=True)
        shortlisted = rescored[: self._params.shortlist]

        if self._params.exact_title_guarantee:
            shortlisted = _guarantee_exact_title_lead(rescored, shortlisted, q)

        return shortlisted


def _guarantee_exact_title_lead(
    rescored: list[ScoredPassage],
    shortlisted: list[ScoredPassage],
    q: PreparedQuery,
) -> list[ScoredPassage]:
    """Prepend the exact-title-match lead (see module docstring) to
    ``shortlisted`` if ``rescored`` has one and it didn't already make the
    cosine-similarity cut — never evicting anything the shortlist already
    chose, so the added cost is at most one extra passage for
    ``cross_encoder`` to score.

    Prepended, not appended: ``cross_encoder``'s own defensive
    ``candidates_max`` bound (its docstring: "in case this scorer runs
    without static_pass having shortlisted first") slices its input with a
    plain ``passages[:candidates_max]``, and in the live profile that bound
    equals ``shortlist`` exactly (40/40) — appending would hand back a
    41-item list whose LAST item (our guarantee) is immediately sliced off
    again by that bound before the cross-encoder ever sees it, silently
    undoing this fix. Prepending guarantees it survives any downstream
    front-truncating bound at least as large as ``shortlist``."""
    winner = find_exact_title_lead(rescored, q.text)
    if winner is None:
        return shortlisted

    winner_key = (winner.passage.zim_id, winner.passage.path, winner.passage.ordinal)
    already_present = any(
        (sp.passage.zim_id, sp.passage.path, sp.passage.ordinal) == winner_key for sp in shortlisted
    )
    if already_present:
        return shortlisted
    return [winner, *shortlisted]


__all__ = ["StaticPass"]
