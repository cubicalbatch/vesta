"""Alias expand preparer — appends ZIM redirect-table aliases to query terms.

Registered as ``query_preparer`` ``alias_expand``. Asks the archive registry for
alias expansions of the user's terms (the redirect table
yields real acronym expansions like ``AFAICS → Internet_slang`` — free, offline,
per-corpus).

The lookup itself lives on the registry (``zim/``), not here: the alias table is
a ZIM-layer artefact, so retrieval never queries the DB directly
(``retrieval/`` depends on zim/config/encoders, not db).

``mode`` controls where a resolved alias
goes:

* ``"fts_terms"`` (default) — appended to ``q.terms``, exactly today's behaviour.
  But libzim's ``parse_query`` makes every term in ``q.terms`` mandatory
  (the every-term-AND trap): an alias appended here is not a hint, it is
  *another constraint*, which is precisely how the live "How much nitrogen in
  the air?" trace turned a correct alias resolution (Air → Atmosphere of Earth)
  into a query that returns nothing useful. Kept as the default anyway — no
  profile sets ``mode`` explicitly today, and changing the default would move
  every existing profile's measured eval numbers by stealth).
* ``"title_hints"`` — populates ``q.aliases`` only. This is the fix: an alias
  candidate source (``alias_title_resolve``) can look the term up as a title
  directly, without ever poisoning the AND-mandatory FTS term list.
* ``"both"`` — the union: alias terms land in both ``q.terms`` and ``q.aliases``.

Speed: one indexed lookup per query; capped by ``max_aliases``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import BaseModel

from vesta.config.capabilities import Capability
from vesta.retrieval.contracts import PreparedQuery
from vesta.retrieval.registry import register

if TYPE_CHECKING:
    from vesta.retrieval.trace import Trace
    from vesta.zim.registry import ArchiveRegistry


@register("query_preparer", "alias_expand")
class AliasExpand:
    """Look up aliases from the ZIM redirect table.

    Requires no capabilities — the registry always exposes alias lookup, and an
    empty result (no redirect table, no matches) is a valid, complete result.
    """

    requires: ClassVar[frozenset[Capability]] = frozenset()

    class Params(BaseModel):
        max_aliases: int = 3
        mode: Literal["fts_terms", "title_hints", "both"] = "fts_terms"

    def __init__(
        self, params: Params | None = None, archives: ArchiveRegistry | None = None
    ) -> None:
        self._params = params or self.Params()
        self._archives = archives

    async def prepare(self, q: PreparedQuery, tr: Trace) -> PreparedQuery:
        """Resolve up to ``max_aliases`` aliases and route them per ``mode``."""
        if not q.terms or self._archives is None:
            return q

        expansions = await self._archives.lookup_aliases(
            list(q.terms), max_aliases=self._params.max_aliases
        )

        existing = {t.lower() for t in q.terms}
        aliases: list[str] = []
        for term in expansions:
            if len(aliases) >= self._params.max_aliases:
                break
            if term.lower() not in existing:
                aliases.append(term)
                existing.add(term.lower())

        if not aliases:
            return q

        # q.aliases is populated in every mode (today's behaviour, unchanged).
        # q.terms only gets the alias appended for fts_terms/both — title_hints
        # withholds it from q.terms entirely, which is D3's fix: the term never
        # reaches the all-terms-mandatory FTS ladder, only alias_title_resolve's
        # direct title lookup.
        new_terms = (
            tuple(q.terms) + tuple(aliases)
            if self._params.mode in ("fts_terms", "both")
            else q.terms
        )
        new_aliases = tuple(aliases)

        return PreparedQuery(
            raw=q.raw,
            terms=new_terms,
            text=q.text,
            aliases=new_aliases,
            is_keyword_query=q.is_keyword_query,
            rung=q.rung,
            history=q.history,
        )
