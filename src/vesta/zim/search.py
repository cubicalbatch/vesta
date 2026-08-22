"""Per-archive Xapian FTS + title/suggestion search over libzim.

Two things make this module's shape non-obvious:

1. **Returns PATHS ONLY.** python-libzim's ``SearchResultSet`` is typed
   ``Iterator[str]`` and yields path strings; ``getScore``/``getSnippet``/
   ``getTitle`` exist in C++ but are not wrapped. So the type system here
   forbids a score — no caller can assume ranks are comparable across archives.
   Snippet generation and cross-archive ranking are handled in retrieval.
2. **No cross-archive search in the binding.** ``Searcher`` takes one
   ``Archive`` (the C++ ``Searcher(vector<Archive>)`` is unexposed). Fan-out is
   therefore one ``Searcher`` per archive in Python, returning unmerged lists —
   merging is retrieval policy, never done here.

Pagination reuses one ``Search`` object (calling ``searcher.search()`` again
re-runs the query). Latency is 1-40 ms on multi-GB archives, so a few probes
per user query is well inside budget.
"""

from __future__ import annotations

from collections.abc import Sequence

from libzim.reader import Archive as LibzimArchive
from libzim.search import Query, Searcher
from libzim.suggestion import SuggestionSearcher

from vesta.zim.types import EntryPath


def fulltext_search(archive: LibzimArchive, terms: Sequence[str], limit: int) -> list[EntryPath]:
    """AND the terms against the archive's fulltext Xapian index.

    libzim's default operator is AND, so joining terms with a space is
    the correct way to express "all terms must match". Returns up to ``limit``
    paths in BM25 order. Raises ``NoFulltextIndex`` if the archive has none —
    callers (``Archive.search``) check first and return ``[]``.
    """
    if not archive.has_fulltext_index:
        raise NoFulltextIndex(archive.filename)
    query = " ".join(terms).strip()
    if not query:
        return []
    searcher = Searcher(archive)
    search = searcher.search(Query().set_query(query))
    # ``getResults(start, count)`` paginates on the SAME Search object —
    # re-running ``searcher.search()`` would re-execute the query.
    return list(search.getResults(0, max(limit, 0)))


def title_suggest(archive: LibzimArchive, prefix: str, limit: int) -> list[EntryPath]:
    """Prefix-match the separate ``X/title/xapian`` index.

    Present in every archive tested, including those with no fulltext index,
    making it the fallback (the last rung of the query ladder). Returns paths
    only — no scores. Can be slow (~24 ms) for short, high-frequency prefixes;
    debounce type-ahead.
    """
    prefix = prefix.strip()
    if not prefix:
        return []
    suggester = SuggestionSearcher(archive)
    result = suggester.suggest(prefix)
    return list(result.getResults(0, max(limit, 0)))


class NoFulltextIndex(RuntimeError):
    """Raised when fulltext search is attempted on an archive without an index.

    The catalog's ``_ftindex`` tag has frequent false negatives, so the
    registry PROBES ``has_fulltext_index`` at runtime; this error means the
    probe said False and the caller did not guard. ``Archive.search`` does.
    """


__all__ = ["NoFulltextIndex", "fulltext_search", "title_suggest"]
