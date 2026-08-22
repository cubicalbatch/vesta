"""Reader passthrough primitives (the path-preserving route).

The reader serves ZIM entries **unmodified**: no HTML rewriting, assets and
articles alike. Two load-bearing rules:

* **Serve ``item.mimetype`` from libzim, never infer from the extension.**
  Wikipedia assets *named* ``.jpg``/``.png`` are frequently actually WebP
  (magic ``RIFF…WEBP``); libzim reports the true mimetype. Doing this right is
  what makes images render without ``allow-scripts``, preserving the
  sandbox security property.
* **Convert soft (``<meta refresh>``) redirects to a real 302.** A sandboxed
  iframe without ``allow-scripts`` blocks meta refresh, so the bounce must be
  server-side. Hard redirects are also a 302 so the browser's base URL stays
  correct for relative asset resolution.

This module owns the synchronous libzim read (``~0.8 ms``); the HTTP layer in
``api/zim.py`` owns caching headers, range support and the 302s. Reads copy with
``bytes()`` because ``item.content`` is a ``memoryview`` over a cache-managed
buffer — a cluster eviction mid-response would be a use-after-free.
"""

from __future__ import annotations

from typing import cast

from libzim.reader import Archive as LibzimArchive

from vesta.zim.entries import extract_soft_redirect_target
from vesta.zim.types import EntryPath, RawEntry


class EntryNotFound(KeyError):
    """The requested path does not exist in the archive."""


def read_entry_sync(archive: LibzimArchive, path: EntryPath) -> RawEntry:
    """Read one entry by path, resolving redirect intelligence.

    Returns redirect targets (hard and soft) without following them — the HTTP
    layer decides whether to 302. Raises :class:`EntryNotFound` if the path is
    absent (the caller maps that to a 404).

    A ``#fragment`` suffix (a section anchor — e.g. a soft-redirect target
    like ``./A/Foo#Bar``) is a URL fragment, never part of a ZIM entry path,
    so it is stripped before lookup.
    """
    lookup = path.partition("#")[0]
    if not lookup or not archive.has_entry_by_path(lookup):
        raise EntryNotFound(path)
    entry = archive.get_entry_by_path(lookup)
    is_redirect = bool(entry.is_redirect)
    redirect_target: EntryPath | None = None
    if is_redirect:
        # Hard redirect: get the target path; no body needed (HTTP 302s).
        try:
            redirect_target = entry.get_redirect_entry().path
        except Exception:  # dangling redirect; treat as not-found
            raise EntryNotFound(path) from None
        return RawEntry(
            path=entry.path,
            title=entry.title,
            mimetype="text/html",
            content=b"",
            is_redirect=True,
            redirect_target=redirect_target,
            soft_redirect_target=None,
        )

    item = entry.get_item()
    content = bytes(item.content)  # copy: see module docstring (memoryview alias)
    mimetype = item.mimetype or "application/octet-stream"
    soft_target: EntryPath | None = None
    if mimetype.startswith("text/html"):
        # Soft-redirect handling: only small HTML shells are meta-refresh
        # redirects; the detector budgets its regex scan accordingly.
        soft_target = extract_soft_redirect_target(content)
    return RawEntry(
        path=entry.path,
        title=entry.title,
        mimetype=mimetype,
        content=content,
        is_redirect=False,
        redirect_target=None,
        soft_redirect_target=soft_target,
    )


def main_entry_path(archive: LibzimArchive) -> EntryPath:
    """The archive's main page path (used for the bare ``/api/zim/{id}/`` route).

    ``archive.main_entry`` is a well-known entry whose own path may be
    ``mainPage``; if it is itself a redirect, follow once to the real article
    path so the route mirrors the ZIM exactly (relative links must
    resolve against the article's own directory).
    """
    entry = archive.main_entry
    if entry.is_redirect:
        return cast("EntryPath", entry.get_redirect_entry().path)
    return cast("EntryPath", entry.path)


__all__ = ["EntryNotFound", "main_entry_path", "read_entry_sync"]
