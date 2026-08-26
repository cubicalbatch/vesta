"""Conversational rewrite preparer — LLM-assisted query rewrite for turn ≥ 2.

Registered as ``query_preparer`` ``conversational_rewrite``. Round 0 is never
agentic, and neither is turn 1: this preparer is a no-op until the
query carries conversation history. From turn 2 on it asks the model to fold the
prior turns into a standalone search query (resolving pronouns/context), plus a
last-turn-concatenation fallback. Measured at +0.043/+0.021 nDCG@5 — small but
real, and the only query transformation Vesta ships.

Minimalism here is measured, not accidental: HyDE (-5.43% vs modern embedders),
multi-query expansion, step-back prompting, keyword rewrites and decomposition all
*measured worse*. SemEval-2026 MTRAG found retrieval-oriented rewrites
"consistently hurt". Do not add another transformation without golden-set proof.

The LLM is reached through the injected :class:`~vesta.retrieval.contracts.QueryRewriter`
(see ``Deps.rewriter``), NOT by importing ``inference/`` — ``retrieval/`` must not
know an LLM exists. The composition root builds the
gateway-backed rewriter and passes it through the pipeline's ``TypeError`` ladder.
When no rewriter is injected the preparer degrades to a no-op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from vesta.config.capabilities import Capability
from vesta.retrieval.contracts import PreparedQuery, QueryRewriter
from vesta.retrieval.impls.normalize import is_keyword_query, normalize_query_terms
from vesta.retrieval.registry import ComponentParams, register

if TYPE_CHECKING:
    from vesta.retrieval.trace import Trace
    from vesta.zim.registry import ArchiveRegistry

#: System prompt for the rewrite. Consumed by the gateway-backed
#: :class:`~vesta.answer.rewriter.GatewayQueryRewriter` the composition root
#: injects; kept here so the preparer owns its own contract. Conservative by
#: design: output the query only, resolve context, do not expand or paraphrase.
REWRITE_SYSTEM_PROMPT = (
    "You rewrite a follow-up question into a standalone search query for an "
    "offline encyclopedia. Use the conversation history only to resolve "
    "pronouns, entities, and omitted context. Do not expand the query, do not "
    "answer it, and do not add words the user did not imply. "
    "Output ONLY the rewritten query — no preamble, no quotation marks, no "
    "explanation."
)


@register("query_preparer", "conversational_rewrite")
class ConversationalRewrite:
    """Rewrite a turn-≥2 follow-up into a standalone search query.

    ``requires = {Capability.LLM}`` — without an LLM the profile drops this
    preparer and records the drop. ``rewriter`` arrives via
    DI (``Deps.rewriter``); when it is ``None`` (capability dropped, or a
    composition root that did not wire multi-turn) the preparer is a no-op.
    """

    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.LLM})

    class Params(ComponentParams):
        """``max_history_turns`` caps how many recent history entries reach the
        rewriter (a turn here is one ``(role, content)`` entry; the most recent
        are kept). Bounding history keeps the rewrite prompt inside the latency
        budget."""

        max_history_turns: int = 4

    def __init__(
        self,
        params: Params | None = None,
        archives: ArchiveRegistry | None = None,
        *,
        rewriter: QueryRewriter | None = None,
    ) -> None:
        # ``archives`` is accepted (and unused) to fit the pipeline's DI ladder
        # top rung, which passes ``(params, archives, rewriter)`` together —
        # the same bundling convention ``vector_knn`` follows for ``vectors``.
        del archives
        self._params = params or self.Params()
        self._rewriter = rewriter

    async def prepare(self, q: PreparedQuery, tr: Trace) -> PreparedQuery:
        """Rewrite turn-≥2 follow-ups; pass through turn 1 and degraded runs.

        No-op when there is no history (turn 1) or no injected rewriter. On any
        LLM failure or empty result the preparer returns the query unchanged —
        a rewrite must never make retrieval worse.
        """
        del tr  # the pipeline records the preparer stage; this adds no sub-stage
        rewriter = self._rewriter
        if rewriter is None or not q.history:
            return q

        # Keep only the most recent entries so the rewrite prompt stays bounded.
        limit = max(0, self._params.max_history_turns)
        history = list(q.history[-limit:]) if limit else []

        try:
            rewritten = await rewriter.rewrite(q.text or q.raw, history)
        except Exception:
            # Never make things worse: a dead endpoint or parse error falls back
            # to the raw query (degrade-don't-fail).
            return q

        if not rewritten or not rewritten.strip():
            return q

        text = rewritten.strip().lower()
        terms = normalize_query_terms(text)
        if not terms:
            return q

        return PreparedQuery(
            raw=q.raw,
            terms=terms,
            text=text,
            aliases=q.aliases,
            is_keyword_query=is_keyword_query(text),
            rung=q.rung,
            history=q.history,
        )
