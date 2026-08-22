"""``sources_only`` answer strategy — no generation, first-class mode.

Default when no LLM is configured, and a first-class mode otherwise.
Emits source cards and the trace, then a ``done`` event — no token stream, no
citations. This is the fast path that works without any model configured:
"offline Wikipedia search is useful without an answer."

Registered as ``answer_strategy/sources_only``. ``requires`` is empty — it works
with zero capabilities.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

from vesta.answer.contracts import (
    AnswerContext,
    AnswerDeps,
    AnswerEvent,
    DoneEvent,
    SourcesEvent,
    TraceEvent,
)
from vesta.config.capabilities import Capability
from vesta.retrieval.registry import register

if TYPE_CHECKING:
    from vesta.retrieval.trace import Trace


@register("answer_strategy", "sources_only")
class SourcesOnlyStrategy:
    """No LLM: emit source cards + trace, then done."""

    requires: ClassVar[frozenset[Capability]] = frozenset()

    class Params(BaseModel):
        """No tunable params — this strategy just returns what retrieval found."""

    def __init__(self, params: Params | None = None, deps: AnswerDeps | None = None) -> None:
        self._params = params or self.Params()
        self._deps = deps

    async def answer(
        self,
        ctx: AnswerContext,
        deps: AnswerDeps | None = None,
        tr: Trace | None = None,
    ) -> AsyncIterator[AnswerEvent]:
        """Emit sources, then trace, then done. No LLM call."""
        from vesta.answer.contracts import StatusEvent

        # Sources first (progressive disclosure).
        yield SourcesEvent(cards=ctx.retrieval.cards)
        yield StatusEvent(phase="sources_only", detail=f"{len(ctx.retrieval.cards)} sources")
        # Trace last.
        yield TraceEvent(trace=tr.to_dict() if tr is not None else {})
        yield DoneEvent()


__all__ = ["SourcesOnlyStrategy"]
