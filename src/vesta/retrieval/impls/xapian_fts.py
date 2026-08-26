"""Xapian FTS candidate source — per-archive full-text search through libzim.

Registered as ``candidate_source`` ``xapian_fts``. Calls ``Archive.search()`` on
every enabled archive in the scope, returning ``Candidate`` objects with
``score=None`` (lexical — python-libzim exposes no scores). When
``fallback_ladder: true`` and search returns empty, the query-preprocessing ladder
from ``zim/query.py`` is tried (all-terms → stopword-stripped → OR-of-terms →
title).

Requires ``ZIM_FULLTEXT`` — if no enabled archive has a probed full-text index,
this source is dropped from the pipeline (recorded in the trace, degrade-don't-fail).

Returns ``list[Candidate]`` per the ``CandidateSource`` contract.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, ClassVar

from vesta.config.capabilities import Capability
from vesta.retrieval.contracts import Candidate, PreparedQuery, Scope
from vesta.retrieval.registry import ComponentParams, register

if TYPE_CHECKING:
    from vesta.retrieval.trace import Trace
    from vesta.zim.registry import ArchiveRegistry
    from vesta.zim.types import Archive


@register("candidate_source", "xapian_fts")
class XapianFTS:
    """Full-text search across enabled archives with optional query-ladder fallback.

    Requires ZIM_FULLTEXT (degrade-don't-fail): the pipeline checks this against
    the live capability set and drops the source if unmet, recording the drop in
    the trace. If the source is not dropped and a FTS-less archive is in scope,
    that archive simply returns no candidates — no error, no trace entry.
    """

    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.ZIM_FULLTEXT})

    class Params(ComponentParams):
        limit: int = 40
        fallback_ladder: bool = True

    def __init__(
        self, params: Params | None = None, archives: ArchiveRegistry | None = None
    ) -> None:
        self._params = params or self.Params()
        self._archives = archives

    async def find(self, q: PreparedQuery, scope: Scope, tr: Trace) -> list[Candidate]:
        """Search every enabled archive in scope; ladder-fallback on empty results.

        Archives without a full-text index are skipped silently (they return no
        candidates). Concurrency is not bounded here — the pipeline's central
        semaphore gates fan-out across all sources.
        """
        if self._archives is None:
            return []

        from vesta.retrieval.impls._scope import archives_for_scope

        archives = await archives_for_scope(self._archives, scope)

        if not archives:
            return []

        limit = self._params.limit
        fallback = self._params.fallback_ladder

        async def _search_one(archive: Archive, terms: tuple[str, ...]) -> list[Candidate]:
            try:
                paths = await archive.search(terms, limit)
            except Exception:
                paths = []
            return [
                Candidate(
                    zim_id=archive.id,
                    path=p,
                    source="xapian_fts",
                    rank=i,
                    score=None,  # lexical — no scores in python-libzim
                )
                for i, p in enumerate(paths)
            ]

        # Fire concurrent searches across all archives.
        results = await asyncio.gather(
            *(_search_one(a, q.terms) for a in archives), return_exceptions=True
        )

        out: list[Candidate] = []
        for r in results:
            if isinstance(r, list):
                out.extend(r)

        # If fallback ladder is enabled and no results, try the query preprocessing
        # ladder from zim/query.py.
        if fallback and not out and q.terms:
            for archive in archives:
                hits = await _run_ladder_for_archive(archive, q.raw, limit)
                out.extend(
                    Candidate(
                        zim_id=archive.id,
                        path=p,
                        source="xapian_fts",
                        rank=i,
                        score=None,
                    )
                    for i, p in enumerate(hits)
                )
                if out:
                    break

        # Normalize ranks within each ``(zim_id, source)`` group: the
        # Candidate contract scopes ``rank`` to its own group only, so a
        # global counter over the concatenated list would offset every archive
        # after the first and shrink its RRF mass.
        counts: dict[tuple[int, str], int] = {}
        for i, c in enumerate(out):
            key = (c.zim_id, c.source)
            rank = counts.get(key, 0)
            counts[key] = rank + 1
            out[i] = Candidate(
                zim_id=c.zim_id,
                path=c.path,
                source=c.source,
                rank=rank,
                score=c.score,
            )

        return out


async def _run_ladder_for_archive(
    archive: Any,  # Archive Protocol — untyped to avoid import cycle
    raw: str,
    limit: int,
) -> list[str]:
    """Run the query preprocessing ladder against one archive.

    Takes an untyped `archive` parameter to avoid import cycle — the caller
    passes the real Archive implementation.
    """
    from vesta.zim.query import DEFAULT_STOPWORDS, QueryPreparer

    qp = QueryPreparer.from_settings(
        stopword_stripping=True,
        stopword_list=",".join(DEFAULT_STOPWORDS),
        ladder_enabled=True,
    )

    async def _search(terms: tuple[str, ...], lim: int) -> list[str]:
        try:
            result: list[str] = list(await archive.search(terms, lim))
            return result
        except Exception:
            return []

    async def _suggest(prefix: str, lim: int) -> list[str]:
        try:
            result: list[str] = list(await archive.suggest(prefix, lim))
            return result
        except Exception:
            return []

    return await qp.execute(raw, _search, _suggest, limit=limit)
