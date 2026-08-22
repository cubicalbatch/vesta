"""Entry classification.

libzim gives no semantic class for an entry, so this module applies
heuristics for detecting non-article and special entry types:

* **Hard redirect** — ``entry.is_redirect`` (libzim property; cheap).
* **Soft redirect** — an HTML ``<meta http-equiv="refresh">`` page that is *not*
  flagged ``is_redirect`` (e.g. ~4,700 in Simple Wikipedia alone). Missing
  these puts thousands of ~280-byte near-duplicates in the index.
* **Disambiguation** — ``.mw-disambig``/``.dmbox``/``#disambigbox`` class, or a
  title ending in ``(disambiguation)``.
* **List page** — title prefix ``List of``/``Lists of``/``Timeline of``/
  ``Index of``/``Glossary of``.
* **Stub** — extracted text below ~1 200 chars (caller supplies the length).

Classification *writes* ``articles.flags``; *skipping* an article for indexing
is a downstream indexing decision, so bits here only label — they never silently drop.
"""

from __future__ import annotations

import re

from vesta.zim.types import EntryFlags, EntryPath

#: A ``<meta http-equiv="refresh" content="0; url=TARGET">`` soft redirect. The
#: URL attribute may be single- or double-quoted and may use ``./`` prefixes.
#: Case-insensitive, tolerant of whitespace. The URL value
#: itself may be bare (``url=A/Foo``), single-quoted (``url='A/Foo'``), or — the
#: youtube2zim/ted2zim shape — single-quoted INSIDE a double-quoted attribute
#: (``content="0;URL='index.html#/watch/...'"``); the three capture groups cover
#: those and the target picker selects the one that matched.
_SOFT_REDIRECT_RE = re.compile(
    rb"""<meta\s+http-equiv\s*=\s*["']?refresh["']?\s+content\s*=\s*["']\s*\d+\s*;\s*"""
    rb"""(?:url\s*=\s*)?(?:"([^"]*)"|'([^']*)'|([^"'\s>]+))""",
    re.IGNORECASE,
)

#: Titles that mark a list/index page rather than a topic article.
_LIST_PREFIXES = ("list of ", "lists of ", "timeline of ", "index of ", "glossary of ")

#: Below this many extracted characters an article is considered a stub.
STUB_CHAR_THRESHOLD = 1200

#: Soft-redirect pages are tiny HTML shells (~280 bytes). We only run
#: the regex on entries below this byte budget to avoid scanning real articles.
_SOFT_REDIRECT_SIZE_BUDGET = 600


def extract_soft_redirect_target(
    content: bytes, *, size_budget: int = _SOFT_REDIRECT_SIZE_BUDGET
) -> str | None:
    """Return the target path of a ``<meta refresh>`` soft redirect, else None.

    Only inspected for small payloads: real articles are never soft redirects
    and the regex should not scan a 1.4 MB article body. Strips a leading
    ``./`` so the result is a plain entry path.
    """
    if len(content) > size_budget:
        return None
    m = _SOFT_REDIRECT_RE.search(content)
    if m is None:
        return None
    # Three capture groups (double-quoted / single-quoted / bare URL); pick the
    # one that matched. ``or`` falls through empty matches so an empty quoted
    # URL degrades to "not a soft redirect" rather than a bare fallback.
    raw = m.group(1) or m.group(2) or m.group(3)
    if not raw:
        return None
    target = raw.decode("utf-8", "replace").strip()
    # ZIM soft redirects point at ``./Target`` or ``A/Target``; normalise the
    # ``./`` away but keep whatever namespace libzim uses for this archive.
    return target.lstrip("./").strip()


def is_soft_redirect(content: bytes) -> bool:
    """True if ``content`` is a ``<meta refresh>`` soft-redirect shell."""
    return extract_soft_redirect_target(content) is not None


def classify_entry(
    path: EntryPath,
    title: str,
    html: bytes | None,
    *,
    is_redirect: bool,
    char_len: int = 0,
) -> EntryFlags:
    """Classify one entry into a flag bitfield.

    ``html`` is the raw entry bytes (only the first :data:`_SOFT_REDIRECT_SIZE_BUDGET`
    bytes are inspected for soft redirects). ``char_len`` is the *extracted-text*
    length when known (stub detection); pass 0 to skip the stub check.
    """
    flags = EntryFlags.NONE
    if is_redirect:
        flags |= EntryFlags.REDIRECT
    if html is not None and is_soft_redirect(html):
        flags |= EntryFlags.SOFT_REDIRECT
    title_lower = title.strip().lower()
    if title_lower.endswith("(disambiguation)") or (
        html is not None and _has_disambiguation_marker(html)
    ):
        flags |= EntryFlags.DISAMBIGUATION
    if title_lower.startswith(_LIST_PREFIXES):
        flags |= EntryFlags.LIST
    if char_len and char_len < STUB_CHAR_THRESHOLD:
        flags |= EntryFlags.STUB
    return flags


_DISAMBIG_MARKERS = (b"mw-disambig", b"disambigbox", b"dmbox")


def _has_disambiguation_marker(html: bytes) -> bool:
    sample = html[:4096]
    return any(marker in sample for marker in _DISAMBIG_MARKERS)


__all__ = [
    "STUB_CHAR_THRESHOLD",
    "classify_entry",
    "extract_soft_redirect_target",
    "is_soft_redirect",
]
