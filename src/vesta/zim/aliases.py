"""Alias dictionary mined from the archive's redirect table.

At registration time we walk the redirect table once: Simple Wikipedia yields
108,571 redirects → 62,911 targets (~1.7 aliases/article), including real
acronym expansions (``AFAICS → Internet_slang``). Free, offline, per-corpus —
expansion data available without querying an LLM. The registry persists the
result into the ``aliases`` table (migration 0002) and rebuilds it on
(re)registration.

Access pattern: ``entry.title`` is the *alias* (human readable, e.g.
``"AIM speak"``) and the redirect target's ``entry.path`` is the *canonical*
article. Aliases per target are capped and list-page targets skipped to avoid
junk clusters (248 TLD redirects → one ``List of`` article, measured).
"""

from __future__ import annotations

from libzim.reader import Archive as LibzimArchive

#: Cap aliases per canonical target — some targets accrete hundreds of redundant
#: redirects (e.g. 248 TLDs → ``List_of_Internet_top-level_domains``).
MAX_ALIASES_PER_TARGET = 30

#: Targets whose path looks like a list/index page are skipped — their aliases
#: are navigation, not synonymy.
_LIST_TARGET_PREFIXES = ("List_of_", "Lists_of_", "Index_of_", "Timeline_of_", "Glossary_of_")


def mine_aliases(
    archive: LibzimArchive,
    *,
    max_aliases_per_target: int = MAX_ALIASES_PER_TARGET,
) -> list[tuple[str, str]]:
    """Walk the redirect table → ``(source_title, target_path)`` pairs.

    Single-threaded scan over ``entry_count`` (~14 k entries/s — ~29 s for
    Simple Wikipedia's 400 k entries; the registry only does this for *newly*
    registered archives, so repeat scans stay cheap). Results are de-duplicated
    by source title; targets over the cap and list-page targets are skipped.
    """
    per_target_count: dict[str, int] = {}
    seen_sources: set[str] = set()
    out: list[tuple[str, str]] = []
    for i in range(archive.entry_count):
        entry = archive._get_entry_by_id(i)  # documented libzim iteration API
        if not entry.is_redirect:
            continue
        source = entry.title.strip()
        if not source or source in seen_sources:
            continue
        try:
            target = entry.get_redirect_entry().path
        except Exception:  # dangling redirect; skip it
            continue
        if any(target.startswith(p) for p in _LIST_TARGET_PREFIXES):
            continue
        count = per_target_count.get(target, 0)
        if count >= max_aliases_per_target:
            continue
        seen_sources.add(source)
        per_target_count[target] = count + 1
        out.append((source, target))
    return out


__all__ = ["MAX_ALIASES_PER_TARGET", "mine_aliases"]
