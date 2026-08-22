"""Shared scope-resolution helper for candidate sources.

A retrieval :class:`~vesta.retrieval.contracts.Scope` carries two independent
selection axes — explicit ``zim_ids`` and user-facing ``corpus_labels``. Turning
either into the concrete list of archives to search is *not* source-specific
policy, so it lives here as a plain function rather than being duplicated in
every ``CandidateSource`` (the rule: shared logic moves down into a function in
the package, not imported between sibling implementations).

``corpus_labels`` are resolved to zim ids via the registry (which owns label→id
mapping) and intersected with any explicit ``zim_ids`` before the registry
filters to enabled archives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vesta.retrieval.contracts import Scope as RetScope

if TYPE_CHECKING:
    from vesta.zim.registry import ArchiveRegistry
    from vesta.zim.types import Archive


async def archives_for_scope(archives: ArchiveRegistry | None, scope: RetScope) -> list[Archive]:
    """Resolve a retrieval ``Scope`` to the enabled archives it covers.

    ``None`` archives → empty list. ``corpus_labels`` (if set) are resolved to
    zim ids and intersected with ``zim_ids`` (if also set); the registry then
    returns only enabled archives matching the effective id set.
    """
    from vesta.zim.types import Scope as ZimScope

    if archives is None:
        return []
    zim_ids = scope.zim_ids
    if scope.corpus_labels:
        labelled = await archives.ids_for_labels(scope.corpus_labels)
        zim_ids = labelled if zim_ids is None else (zim_ids & labelled)
    return archives.enabled(ZimScope(zim_ids=zim_ids))
