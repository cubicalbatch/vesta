"""Alias/title resolution candidate source.

Registered as ``candidate_source`` ``alias_title_resolve``.
The live "How much nitrogen in the
air?" trace showed ``alias_expand`` correctly resolve the redirect-table alias
*Air → Atmosphere of Earth*, then get it appended to the FTS term list, where
libzim's ``parse_query`` makes every term mandatory — the alias made the
query MORE restrictive instead of helping. Separately, ``title_suggest`` joins
every query term into one prefix string, so a canonical article title buried in
``q.aliases`` never gets looked up *as a title*.

This source fixes both: it resolves ``q.aliases`` directly to article
candidates via the ZIM-layer exact target (``ArchiveRegistry.resolve_alias_targets``
— the non-lossy sibling of ``lookup_aliases``), and separately checks whether
the query terms, joined and title-cased, are an *exact* title match on any
enabled archive's suggest index. Both mechanisms bypass the AND-mandatory FTS
ladder and the joined-prefix problem entirely.

Requires no capabilities — like ``title_suggest``, the title index and the
alias table are always available (an empty result is a valid, complete result).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

from vesta.config.capabilities import Capability
from vesta.retrieval.contracts import Candidate, PreparedQuery, Scope
from vesta.retrieval.registry import register

if TYPE_CHECKING:
    from vesta.retrieval.trace import Trace
    from vesta.zim.registry import ArchiveRegistry
    from vesta.zim.types import Archive


@register("candidate_source", "alias_title_resolve")
class AliasTitleResolve:
    """Resolve alias hits and exact title matches directly to article candidates.

    No capability dependencies: like ``title_suggest``, the alias table and the
    title/suggest index exist in every registered ZIM, so this source
    is always available.
    """

    requires: ClassVar[frozenset[Capability]] = frozenset()

    class Params(BaseModel):
        limit: int = 5
        exact_title_match: bool = True

    def __init__(
        self, params: Params | None = None, archives: ArchiveRegistry | None = None
    ) -> None:
        self._params = params or self.Params()
        self._archives = archives

    async def find(self, q: PreparedQuery, scope: Scope, tr: Trace) -> list[Candidate]:
        """Alias-target resolution first, then exact-title-match fallback.

        Alias hits (from ``q.aliases`` — populated when ``alias_expand`` runs in
        ``title_hints``/``both`` mode) resolve directly via
        ``resolve_alias_targets``, which returns the exact ``(zim_id, path)`` pair
        — no AND-ladder, no lossy display-name round trip. Exact title matches
        catch the case where the query terms themselves, joined, ARE a title
        (e.g. a reformulated query that already names the target article) but
        ``title_suggest``'s joined-prefix search misses because other terms
        precede it in the query string.
        """
        if self._archives is None:
            return []

        from vesta.retrieval.impls._scope import archives_for_scope

        archives = await archives_for_scope(self._archives, scope)
        if not archives:
            return []

        out: list[Candidate] = []

        if q.aliases:
            zim_ids = frozenset(a.id for a in archives)
            pairs = await self._archives.resolve_alias_targets(
                list(q.aliases), zim_ids=zim_ids, max_aliases=self._params.limit
            )
            out.extend(
                Candidate(
                    zim_id=zim_id, path=target, source="alias_title_resolve", rank=0, score=None
                )
                for zim_id, target in pairs
            )

        if self._params.exact_title_match and q.terms:
            phrase = " ".join(q.terms)
            wanted = phrase.strip().lower()
            if wanted:
                for archive in archives:
                    out.extend(await self._exact_matches(archive, phrase, wanted))

        return _dedupe_and_rerank(out)

    async def _exact_matches(self, archive: Archive, phrase: str, wanted: str) -> list[Candidate]:
        """Suggest-index candidates whose basename, normalized, equals ``wanted``.

        Normalization (basename, underscores → spaces, lowercase) mirrors
        ``ArchiveRegistry.lookup_aliases``'s existing convention exactly — same
        transform, same reasoning: ZIM entry paths use underscores for spaces.
        """
        try:
            paths = await archive.suggest(phrase, 3)
        except Exception:
            return []
        matches: list[Candidate] = []
        for p in paths:
            name = p.rsplit("/", 1)[-1] if "/" in p else p
            name = name.replace("_", " ").strip().lower()
            if name == wanted:
                matches.append(
                    Candidate(
                        zim_id=archive.id, path=p, source="alias_title_resolve", rank=0, score=None
                    )
                )
        return matches


def _dedupe_and_rerank(candidates: list[Candidate]) -> list[Candidate]:
    """Dedupe by ``(zim_id, path)`` keeping the first occurrence, then re-rank."""
    seen: set[tuple[int, str]] = set()
    deduped: list[Candidate] = []
    for c in candidates:
        key = (c.zim_id, c.path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return [
        Candidate(zim_id=c.zim_id, path=c.path, source=c.source, rank=i, score=c.score)
        for i, c in enumerate(deduped)
    ]
