"""Frozen domain types for the ZIM data layer.

These are the owned interfaces for the ZIM data layer. Three rules hold across
all of them:

* **Frozen dataclasses for domain objects.** Pydantic models live only at the
  API boundary (``api/``) and are mapped from these explicitly.
* **``char_start``/``char_end`` are non-negotiable.** They index into an
  article's extracted text so a passage is recoverable from the ZIM by offset —
  this allows vector rows to stay text-free while citation spans point at exact
  characters.
* **Search returns paths only.** python-libzim exposes neither ``getScore`` nor
  ``getSnippet``, so the type system forbids a score here — no caller can
  accidentally assume ranks are comparable across archives. Snippets and
  cross-archive ranking are handled in retrieval.

``zim/`` depends only on ``config`` and ``db`` — at most two internal packages,
enforced by ``tests/test_boundaries.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntFlag
from typing import Protocol, TypeAlias, runtime_checkable

#: A libzim entry path. Modern ZIMs strip the namespace, so this is e.g.
#: ``"Albert_Einstein"``; old-scheme fixtures keep ``"A/Foo"``. We mirror the
#: ZIM's internal path verbatim and never normalise it — that is what makes
#: ZIM-relative links resolve unrewritten.
EntryPath: TypeAlias = str


def strip_entry_prefix(path: str) -> str:
    """Strip ONE leading ``./`` or ``/`` from a ZIM entry reference.

    Deliberately not ``lstrip("./")``: that removes any leading run of ``.`` and
    ``/`` characters, silently mangling targets that legitimately begin with
    dots (``..hidden`` → ``hidden``). Exactly one leading separator goes.
    """
    if path.startswith("./"):
        return path[2:]
    if path.startswith("/"):
        return path[1:]
    return path


def entry_title(path: str) -> str:
    """Human-readable title text for a ZIM entry path: the basename with
    underscores turned into spaces, trimmed. The one convention shared by the
    redirect-alias lookup and the exact-title matchers (ZIM paths use
    underscores for spaces)."""
    name = path.rsplit("/", 1)[-1] if "/" in path else path
    return name.replace("_", " ").strip()


def entry_title_key(path: str) -> str:
    """Case-folded :func:`entry_title` — the comparable form for equality
    checks between a query phrase and an entry path."""
    return entry_title(path).lower()


class EntryFlags(IntFlag):
    """Per-article classification bits, stored in ``articles.flags``.

    Entry classification writes the bits; skipping an article for indexing is a
    downstream indexer decision, so a set bit here never silently drops content —
    it only labels it. Soft redirects are their own bit (``<meta refresh>``
    entries are not flagged ``is_redirect`` and would otherwise become thousands
    of near-duplicate junk entries in the index).
    """

    NONE = 0
    REDIRECT = 1 << 0  # hard redirect (libzim ``entry.is_redirect``)
    SOFT_REDIRECT = 1 << 1  # <meta http-equiv="refresh"> soft redirect
    DISAMBIGUATION = 1 << 2  # ".mw-disambig" / "(disambiguation)"
    LIST = 1 << 3  # "List of" / "Timeline of" / "Index of" / "Glossary of"
    STUB = 1 << 4  # extracted text below ~1 200 chars


@dataclass(frozen=True)
class RawEntry:
    """The raw bytes libzim holds for one path, plus redirect intelligence.

    The reader returns this so the HTTP layer can decide: serve the bytes, or
    issue a 302 for a (hard or soft) redirect. ``content`` is always copied with
    ``bytes()`` — ``item.content`` is a ``memoryview`` over a cache-managed
    buffer, and a cluster eviction mid-response would be a use-after-free.
    """

    path: EntryPath
    title: str
    mimetype: str
    content: bytes
    is_redirect: bool
    redirect_target: EntryPath | None
    #: Target of a ``<meta refresh>`` soft redirect, else ``None``. Sandboxed
    #: iframes without ``allow-scripts`` block meta refresh, so the HTTP layer
    #: converts these to a real 302.
    soft_redirect_target: EntryPath | None


@dataclass(frozen=True)
class Section:
    """One contiguous section of an article's extracted text.

    ``heading_path`` is the H1→H2→… breadcrumb (e.g.
    ``("Albert Einstein", "Early life")``); ``char_start``/``char_end`` index
    into :pyattr:`ExtractedArticle.text`. Passage breadcrumbs and section vectors
    are built from this structure.
    """

    heading_path: tuple[str, ...]
    level: int  # 1..6 of the heading that opened this section
    char_start: int
    char_end: int


@dataclass(frozen=True)
class ExtractedArticle:
    """Section-aware plain-text extraction of one ZIM article.

    ``text`` is the ``resiliparse`` ``main_content`` output (correct inline
    handling; ``selectolax`` shreds sentences, ``trafilatura`` is 13x slower).
    ``sections`` partition ``text`` so every char lives in exactly one section.
    ``flags`` carry classification (entries.py).
    """

    path: EntryPath
    title: str
    text: str
    sections: tuple[Section, ...]
    flags: EntryFlags


@dataclass(frozen=True)
class Passage:
    """One ~target_tokens chunk of an article.

    * ``char_start``/``char_end`` index into the article's extracted text
      (the passage is recoverable from the ZIM by offset, so vector rows store
      no text).
    * ``text == article.text[char_start:char_end]`` exactly; ``breadcrumb`` is
      the separate ``Article > Section`` prefix the caller composes with the
      text when embedding. Keeping them apart keeps the offsets honest.
    * ``is_lead`` flags lead-section passages — **flagged, not boosted**;
      boosting is retrieval policy handled in retrieval.
    """

    zim_id: int
    path: EntryPath
    ordinal: int
    char_start: int
    char_end: int
    breadcrumb: str
    text: str
    is_lead: bool


@dataclass(frozen=True)
class Scope:
    """Which archives a query runs over. Retrieval owns retrieval policy; this is
    the minimal selection the registry filters on. ``zim_ids`` ``None`` means
    "every enabled archive"; a set restricts to those ids (e.g. a corpus pick).
    """

    zim_ids: frozenset[int] | None = None


@dataclass(frozen=True)
class ScanResult:
    """Outcome of a discovery scan of the ZIM directory."""

    added: tuple[int, ...]  # newly registered archive ids
    updated: tuple[int, ...]  # known archives whose file/path changed
    missing: tuple[int, ...]  # archive ids whose file disappeared
    total: int  # archives currently held by the registry


@dataclass(frozen=True)
class MediaRef:
    """Playable-media assets for one browsable entry (media/SPA ZIMs).

    Resolved from the per-video ``application/json`` sidecar's ``videoPath`` /
    ``thumbnailPath`` / ``duration`` fields by ``zim/media.py``. The frontend
    renders a native ``<video poster=… src=…>`` from it (no sandbox relaxation).
    Paths are ZIM-relative (served via the path-preserving reader route).
    """

    video_path: str | None
    poster_path: str | None
    duration: int | None  # seconds


@dataclass(frozen=True)
class DocumentRef:
    """One browsable document for a documents-kind ZIM (nautiluszim).

    Resolved from the nautilus ``database.js`` manifest by ``zim/documents.py``.
    ``url`` is the path-preserving reader URL (``/api/zim/{zim_id}/{doc_path}``)
    so the frontend renders the PDF natively in a sandboxed iframe.
    """

    doc_path: EntryPath
    title: str | None
    description: str | None
    author: str | None
    doc_mime: str
    url: str


@runtime_checkable
class Archive(Protocol):
    """One open ZIM archive.

    ``search``/``suggest`` return **paths only** — python-libzim exposes no
    scores or snippets, so callers cannot assume ranks are comparable.
    Cross-archive merging is retrieval policy; ``zim/`` returns
    unmerged per-archive lists.
    """

    id: int
    uuid: str
    title: str
    language: str
    #: PROBED at runtime — the catalog's ``_ftindex`` tag has ~41% false
    #: negatives. Never trust catalog metadata for this.
    has_fulltext_index: bool
    #: From the ``Counter`` metadata's ``text/html`` value, NOT
    #: ``archive.article_count`` (which over-counts ~40% by including
    #: redirects).
    article_count: int

    async def search(self, terms: Sequence[str], limit: int) -> list[EntryPath]: ...

    async def suggest(self, prefix: str, limit: int) -> list[EntryPath]: ...

    async def read(self, path: EntryPath) -> RawEntry: ...

    async def extract(self, path: EntryPath) -> ExtractedArticle: ...

    async def text_entry_paths(self) -> list[EntryPath]:
        """Stable, de-duplicated list of indexable text-entry paths.

        Covers every text-bearing entry mimetype — ``text/html`` (articles) plus
        ``text/vtt`` / ``text/plain`` / ``text/markdown`` sidecars (the real text
        of media/SPA ZIMs whose ``text/html`` entries are redirect stubs). Entry-id
        order is deterministic across runs so an interrupted index resumes exactly
        where it left off."""
        ...

    async def main_path(self) -> EntryPath: ...

    async def random(self) -> EntryPath:
        """A random entry path (archive-browse "Random article" action).

        Backed by libzim's native random-entry function. Implementations should
        avoid landing on a redirect when practical (redirects carry no article
        text — ``extract()`` would return an empty ``text``), but must degrade
        rather than raise if repeated attempts keep landing on one. On
        articles-kind archives implementations may additionally skip
        soft-redirect shells and non-HTML entries so the pick carries real
        text; other kinds (media stubs feed the manifest card grid) must not."""
        ...


__all__ = [
    "Archive",
    "EntryFlags",
    "EntryPath",
    "ExtractedArticle",
    "MediaRef",
    "Passage",
    "RawEntry",
    "ScanResult",
    "Scope",
    "Section",
    "entry_title",
    "entry_title_key",
    "strip_entry_prefix",
]
