"""Title/suggestion candidate source — the universal ZIM fallback index.

Registered as ``candidate_source`` ``title_suggest``. Calls ``Archive.suggest()``
on every enabled archive in the scope. The ``X/title/xapian`` index is present in
*every* archive tested, including those without a full-text index,
making this the universal fallback (the last rung of the query ladder).

``keyword_boost`` (1.5-2.0x): when the query is a bare keyword query (not a
natural-language question), run suggestion with a multiplier on the effective
limit. For question-shaped queries, suggestion still runs but at the base limit.

Requires no capabilities — the title index is always available.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, ClassVar

from vesta.config.capabilities import Capability
from vesta.retrieval.contracts import Candidate, PreparedQuery, Scope
from vesta.retrieval.registry import ComponentParams, register

if TYPE_CHECKING:
    from vesta.retrieval.trace import Trace
    from vesta.zim.registry import ArchiveRegistry
    from vesta.zim.types import Archive


@register("candidate_source", "title_suggest")
class TitleSuggest:
    """Title/suggestion index search — the universal fallback.

    No capability dependencies: the title index exists in every ZIM.
    ``keyword_boost`` applies only to bare-keyword queries (1.5-2.0x)
    when the user typed a lookup rather than a question).
    """

    requires: ClassVar[frozenset[Capability]] = frozenset()

    class Params(ComponentParams):
        limit: int = 20
        keyword_boost: float = 1.8

    def __init__(
        self, params: Params | None = None, archives: ArchiveRegistry | None = None
    ) -> None:
        self._params = params or self.Params()
        self._archives = archives

    async def find(self, q: PreparedQuery, scope: Scope, tr: Trace) -> list[Candidate]:
        """Run title-suggest on every enabled archive in scope.

        For bare keyword queries, the effective limit is multiplied by
        ``keyword_boost``. For question-shaped queries, the base limit
        applies.
        """
        if self._archives is None:
            return []

        from vesta.retrieval.impls._scope import archives_for_scope

        archives = await archives_for_scope(self._archives, scope)

        if not archives:
            return []

        prefix = " ".join(q.terms) if q.terms else q.raw.strip()
        if not prefix:
            return []

        eff_limit = (
            int(self._params.limit * self._params.keyword_boost)
            if q.is_keyword_query
            else self._params.limit
        )

        async def _suggest_one(archive: Archive) -> list[Candidate]:
            try:
                paths = await archive.suggest(prefix, eff_limit)
            except Exception:
                paths = []
            return [
                Candidate(
                    zim_id=archive.id,
                    path=p,
                    source="title_suggest",
                    rank=i,
                    score=None,
                )
                for i, p in enumerate(paths)
            ]

        results = await asyncio.gather(*(_suggest_one(a) for a in archives), return_exceptions=True)

        # ``_suggest_one`` enumerates each archive's hits from 0, so ranks are
        # already within their (zim_id, source) group — concatenating the
        # per-archive blocks preserves that. A running counter across archives
        # would offset every archive after the first and shrink its RRF mass.
        out: list[Candidate] = []
        for r in results:
            if isinstance(r, list):
                out.extend(r)

        return out
