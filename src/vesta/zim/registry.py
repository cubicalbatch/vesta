"""Archive registry — open once, hold, expose by id.

The registry owns the lifetime of every open libzim ``Archive`` and the mapping
between ZIM files and ``zims`` table rows. Four design constraints drive its shape:

* **Probe ``has_fulltext_index`` at runtime.** The catalog's ``_ftindex`` tag
  has ~41% false negatives; never trust catalog metadata.
* **Count articles from ``Counter['text/html']``.** ``archive.article_count``
  over-counts ~40% because it includes redirects (394 563 vs 281 284 measured).
* **Raise the cluster cache at startup.** The default 16 MB is *global* across
  all open archives and thrashes on multi-archive fan-out.
* **Thread safety.** libzim reads go through a bounded pool; each search call
  creates its own ``Searcher`` (searching is not thread-safe upstream),
  and per-archive search coroutines serialize on an asyncio lock.
* **Interactive vs registration pools.** The bounded pool exists for
  interactive query-time work (reads/search/suggest/random/extract) and must
  never be occupied by registration-time batch mining: ``mine_aliases`` walks
  every entry (~29 s on Simple Wikipedia) and would pin a worker, measurably
  stalling article/media serving and search while a large ZIM registers.
  Heavy registration passes therefore run on their own small one-shot
  executor (:meth:`ArchiveRegistry._dispatch_registration`); the media and
  documents manifest builders isolate themselves via ``asyncio.to_thread``
  inside their modules.

``zim/`` depends on ``db`` and ``config`` only. Discovery scans ``./data/zims/``
at startup and on demand; a missing file marks its row ``missing`` rather than
crashing.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
from collections.abc import Collection, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial as _partial
from pathlib import Path
from typing import Any, cast

from libzim.reader import Archive as LibzimArchive
from libzim.reader import set_cluster_cache_max_size

from vesta.db.connection import Database
from vesta.zim import aliases as aliases_mod
from vesta.zim import documents as documents_mod
from vesta.zim import media as media_mod
from vesta.zim import search as search_mod
from vesta.zim.entries import _SOFT_REDIRECT_SIZE_BUDGET, classify_entry, is_soft_redirect
from vesta.zim.extract import extract_entry
from vesta.zim.reader import EntryNotFound, read_entry_sync
from vesta.zim.types import (
    Archive,
    EntryPath,
    ExtractedArticle,
    RawEntry,
    ScanResult,
    Scope,
)

_log = logging.getLogger(__name__)

#: Worker cap for registration-time batch mining. ``mine_aliases`` walks every
#: entry (~14 k entries/s — ~29 s for Simple Wikipedia's 400 k), so it must
#: never run on the interactive read/search pool: one large registration would
#: pin a worker there and measurably stall article/media serving and search
#: (the pool's real purpose is interactive query work — the bulk indexer
#: already isolates itself in its own niced spawn pool). Instead each batch
#: call gets its own small one-shot executor (:meth:`ArchiveRegistry.
#: _dispatch_registration`), created per call and shut down after — the same
#: single-shot shape as ``media.build_media_manifest``'s ``to_thread``, but
#: hard-capped at 2 so concurrent registrations cannot multiply libzim reader
#: threads. Two workers never exceed what the 4-worker interactive pool
#: already assumes about concurrent libzim entry reads.
_REGISTRATION_POOL_SIZE = 2


def raise_cluster_cache(mb: int) -> None:
    """Raise libzim's *global* cluster cache above the 16 MB default.

    The cache is shared across every open archive; 16 MB thrashes on multi-archive
    fan-out (warm reads are no faster than cold at the default).
    Nearly free, and the single easiest performance win available. Must be called
    once at startup, *before* archives are opened.
    """
    set_cluster_cache_max_size(int(mb) * 1024 * 1024)


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()


#: Text-bearing entry mimetypes the indexer harvests. ``text/html`` is the
#: article body for article ZIMs; the rest are the sidecar text formats media/SPA
#: ZIMs (youtube2zim, ted2zim, …) put their real content in — their ``text/html``
#: entries are redirect stubs. A tuple argument to ``str.startswith`` matches any.
_TEXT_ENTRY_MIMETYPES = ("text/html", "text/vtt", "text/plain", "text/markdown")


def _text_entry_paths_sync(archive: LibzimArchive) -> list[EntryPath]:
    """Stable list of indexable text-entry paths (entry-id order), skipping
    redirects and non-text entries. One pass over ``entry_count``.
    Covers HTML articles AND vtt/plain/markdown sidecars so media/SPA ZIMs
    yield real text instead of an empty index."""
    paths: list[EntryPath] = []
    for i in range(archive.entry_count):
        try:
            entry = archive._get_entry_by_id(i)  # documented libzim iteration API
        except Exception:  # a corrupt entry never aborts enumeration
            continue
        if entry.is_redirect:
            continue
        try:
            mimetype = str(entry.get_item().mimetype)
        except Exception:
            continue
        if mimetype.startswith(_TEXT_ENTRY_MIMETYPES):
            paths.append(cast("EntryPath", entry.path))
    return paths


async def _document_entry_paths(db: Database, zim_id: int) -> list[EntryPath]:
    """The indexable path set for a ``documents``-kind archive: the manifest's
    ``doc_path``s only (nautiluszim). The viewer shell + vendored
    sourcemaps/templates that also live under ``text/*`` are scaffolding, not
    content. Ordered by ``doc_path`` for a deterministic, resumable work order
    (the fmt bump to 3 invalidates any older positional checkpoint)."""
    async with db.read() as conn:
        cur = await conn.execute(
            "SELECT doc_path FROM article_documents WHERE zim_id=? ORDER BY doc_path",
            (zim_id,),
        )
        rows = await cur.fetchall()
    return [str(r["doc_path"]) for r in rows]


def _is_content_article(entry: Any) -> bool:
    """True when ``entry`` is a text/html article with a real body — not an
    asset, not a soft-redirect shell.

    Modern mwoffliner archives are dominated by non-content entries (the
    wikipedia top-100 ZIM draws ~74% hard redirects and ~24% ``#Section``
    fragment shells), so "give me a random article" must filter, not trust
    ``get_random_entry``. Size-gated: only small payloads are scanned for the
    meta-refresh marker.
    """
    try:
        item = entry.get_item()
    except RuntimeError:  # redirect entries have no item
        return False
    if not str(item.mimetype or "").startswith("text/html"):
        return False
    if int(item.size) > _SOFT_REDIRECT_SIZE_BUDGET:
        return True
    return not is_soft_redirect(bytes(item.content))


#: Draw budget for articles-only random picks. The real-article hit rate can
#: be ~2% on shell-heavy archives, but a draw is ~10 µs, so the budget is
#: total draws (not re-roll rounds) and stays well inside request latency.
#: Non-articles-only picks keep the historical 5-draw redirect re-roll.
_RANDOM_ARTICLE_DRAW_BUDGET = 400


def _random_entry_path_sync(archive: LibzimArchive, *, articles_only: bool = False) -> EntryPath:
    """One random entry path via libzim's native ``get_random_entry``.

    ``get_random_entry`` can land on any entry type, including redirects
    (which carry no article text). Re-rolls avoid that; with
    ``articles_only`` (browse "Random article"/discover on an articles-kind
    ZIM) it also skips soft-redirect shells and non-HTML entries so callers
    get a real article body. Best-effort, never raising: if the draw budget
    runs out on a shell-heavy archive, the last draw is returned anyway — a
    redirect or shell path is still a valid ``extract()`` input.
    """
    entry = archive.get_random_entry()
    draws = 1
    budget = _RANDOM_ARTICLE_DRAW_BUDGET if articles_only else 5
    while draws < budget and (
        entry.is_redirect or (articles_only and not _is_content_article(entry))
    ):
        entry = archive.get_random_entry()
        draws += 1
    return cast("EntryPath", entry.path)


def _parse_counter_html(archive: LibzimArchive) -> int:
    """Exact article count from ``Counter['text/html']``.

    ``archive.article_count`` over-counts ~40% (it includes redirects). The
    ``Counter`` metadata is a mimetype histogram present in every ZIM and read
    instantly; summing every ``text/html*`` key is exact.
    """
    try:
        raw = archive.get_metadata("Counter").decode("utf-8", "replace")
    except Exception:  # very old ZIMs may lack Counter
        return int(archive.article_count)
    total = 0
    for kv in raw.split(";"):
        if "=" not in kv:
            continue
        key, _, value = kv.rpartition("=")
        if key.startswith("text/html"):
            try:
                total += int(value)
            except ValueError:
                continue
    return total


def _counter_dict(archive: LibzimArchive) -> dict[str, int]:
    """Parse the ``Counter`` mimetype histogram into a ``{mime: count}`` dict.

    Used by :func:`_classify_kind` to detect media/SPA ZIMs (a ``video/`` or
    ``audio/`` entry ⇒ the archive carries playable media). Only keys that look
    like real mimetypes (contain ``/``) are kept, which drops the
    ``charset=…`` fragments produced by splitting parameterized entries such as
    ``text/html; charset=iso-8859-1=1``. Returns ``{}`` if the archive lacks a
    Counter (very old ZIMs).
    """
    try:
        raw = archive.get_metadata("Counter").decode("utf-8", "replace")
    except Exception:
        return {}
    out: dict[str, int] = {}
    for kv in raw.split(";"):
        if "=" not in kv:
            continue
        key, _, value = kv.rpartition("=")
        key = key.strip()
        if "/" not in key:  # drop non-mime fragments (charset params, etc.)
            continue
        try:
            out[key] = out.get(key, 0) + int(value)
        except ValueError:
            continue
    return out


def _classify_kind(
    tags: str, counter: dict[str, int], *, has_nautilus_manifest: bool = False
) -> str:
    """Classify a ZIM's content kind from general signals — never the scraper name.

    Returns one of ``"articles"`` (the default Vesta was built for), ``"media"``
    (playable media ZIMs: youtube2zim, ted2zim, …), ``"spa"`` (single-page-app
    ZIMs whose content is rendered client-side with no per-entry body text) or
    ``"documents"`` (nautiluszim document-library ZIMs whose browsable HTML is a
    viewer shell and whose real content is binary PDFs catalogued by a
    ``database.js`` manifest).

    Order of reliability: the Counter mime histogram is the strongest signal
    (``video/``/``audio/`` entries ⇒ media); the Kiwix ``Tags`` convention
    ``_videos:yes``/``_spa:yes`` is the fallback for archives whose media lives
    outside the ZIM or whose Counter under-reports; the nautilus ``database.js``
    manifest (probed content-side, never the scraper name) is the documents
    signal. Signal-driven by design so this generalises across current and
    future openZIM scrapers without a per-scraper allowlist.
    """
    if any(mime.startswith(("video/", "audio/")) for mime in counter):
        return "media"
    tags_lower = tags.lower()
    if "_videos:yes" in tags_lower:
        return "media"
    if "_spa:yes" in tags_lower:
        return "spa"
    # Content-based: a nautiluszim archive carries a ``database.js`` manifest
    # whose ``var DATABASE = [...]`` records reference the real PDFs. Probed by
    # ``documents.looks_like_nautilus_manifest`` (reads the entry, checks shape)
    # — defensive: a malformed/absent manifest falls through to ``"articles"``,
    # and a normal article ZIM (no ``database.js``) never matches.
    if has_nautilus_manifest:
        return "documents"
    return "articles"


def _meta(archive: LibzimArchive, key: str) -> str:
    try:
        return cast("str", archive.get_metadata(key).decode("utf-8", "replace"))
    except Exception:  # optional metadata may be absent
        return ""


def _corpus_label(name: str) -> str:
    """A user-facing scope name from the ZIM name (e.g. ``wikipedia``)."""
    return name.split("_", 1)[0] if name else ""


@dataclass(frozen=True)
class _Probe:
    """Metadata read once from a freshly opened archive (all cheap, blocking)."""

    uuid: str
    name: str
    title: str
    description: str
    language: str
    flavour: str
    publisher: str
    zim_date: str
    filename: str
    file_size: int
    article_count: int  # from Counter['text/html']
    media_count: int
    has_fulltext_index: bool
    kind: str  # "articles" | "media" | "spa" | "documents" — signal-driven
    scraper: str  # raw Scraper metadata (transparency only, never branched on)
    tags: str  # raw Tags metadata (transparency only, never branched on)


def _probe_archive(path: Path) -> tuple[LibzimArchive, _Probe]:
    """Open and probe one archive (blocking). Caller dispatches off the loop."""
    archive = LibzimArchive(str(path))
    scraper = _meta(archive, "Scraper")
    tags = _meta(archive, "Tags")
    counter = _counter_dict(archive)
    has_nautilus_manifest = documents_mod.looks_like_nautilus_manifest(archive)
    probe = _Probe(
        uuid=str(archive.uuid),
        name=_meta(archive, "Name"),
        title=_meta(archive, "Title") or path.stem,
        description=_meta(archive, "Description"),
        language=_meta(archive, "Language"),
        flavour=_meta(archive, "Flavour"),
        publisher=_meta(archive, "Publisher"),
        zim_date=_meta(archive, "Date"),
        filename=path.name,
        file_size=path.stat().st_size,
        article_count=_parse_counter_html(archive),
        media_count=int(archive.media_count),
        has_fulltext_index=bool(archive.has_fulltext_index),
        kind=_classify_kind(tags, counter, has_nautilus_manifest=has_nautilus_manifest),
        scraper=scraper,
        tags=tags,
    )
    return archive, probe


class LocalArchive:
    """One open ZIM archive implementing the :class:`Archive` Protocol.

    Blocking libzim work runs on the registry's bounded pool. Search
    coroutines serialize on ``_search_lock`` because libzim's search API is not
    thread-safe upstream — there is no parallelism to win anyway.
    """

    def __init__(
        self,
        archive: LibzimArchive,
        probe: _Probe,
        zim_id: int,
        pool: ThreadPoolExecutor,
        db: Database,
    ) -> None:
        self._lz = archive
        self._probe = probe
        self._db = db
        self.id = zim_id
        self.uuid = probe.uuid
        self.name = probe.name
        self.filename = probe.filename
        self.title = probe.title
        self.language = probe.language
        self.has_fulltext_index = probe.has_fulltext_index
        self.article_count = probe.article_count
        self._pool = pool
        self._search_lock = asyncio.Lock()

    async def _dispatch(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, _partial(fn, *args, **kwargs))

    async def search(self, terms: Sequence[str], limit: int) -> list[EntryPath]:
        if not self.has_fulltext_index:
            # No probed index: return nothing; the query ladder's title rung
            # (Archive.suggest) is the fallback.
            return []
        async with self._search_lock:
            return list(
                await self._dispatch(search_mod.fulltext_search, self._lz, tuple(terms), limit)
            )

    async def suggest(self, prefix: str, limit: int) -> list[EntryPath]:
        async with self._search_lock:
            return list(await self._dispatch(search_mod.title_suggest, self._lz, prefix, limit))

    async def read(self, path: EntryPath) -> RawEntry:
        return cast("RawEntry", await self._dispatch(read_entry_sync, self._lz, path))

    async def main_path(self) -> EntryPath:
        """Resolved main-page path (follows one redirect if the main entry is one).

        The reader serves the bare ``/api/zim/{id}/`` route at a slash-terminated
        URL so ZIM-relative assets resolve correctly; resolving the path
        here lets the route serve the real article's bytes there.
        """
        from vesta.zim.reader import main_entry_path

        return cast("EntryPath", await self._dispatch(main_entry_path, self._lz))

    async def random(self) -> EntryPath:
        """A random entry path, redirect-avoiding on a best-effort basis.

        Mirrors ``main_path()``'s dispatch pattern: the libzim call is
        blocking, so it runs on the read pool via ``_dispatch``.

        On articles-kind archives the pick also skips soft-redirect shells
        and non-HTML entries (see ``_random_entry_path_sync``) so browse
        "Random article"/discover cards carry real text. Other kinds keep
        plain redirect-avoiding behavior — media ZIMs' stub entries are the
        manifest-backed card grid's input, and re-rolling them away would
        starve it.
        """
        return cast(
            "EntryPath",
            await self._dispatch(
                _random_entry_path_sync,
                self._lz,
                articles_only=self._probe.kind == "articles",
            ),
        )

    async def text_entry_paths(self) -> list[EntryPath]:
        """Stable, de-duplicated list of indexable text-entry paths.

        Iterates ``entry_count`` once (the documented libzim iteration API),
        skipping hard redirects and any non-text entry (images, assets, binary
        media, metadata). Covers ``text/html`` articles AND ``text/vtt`` /
        ``text/plain`` / ``text/markdown`` sidecars so a media/SPA ZIM (whose
        ``text/html`` entries are redirect stubs) still yields real text. The
        result is the indexer's stable work order — entry-id order is
        deterministic across runs, so a checkpoint high-water mark resumes
        exactly where it left off.

        For ``documents``-kind archives (nautiluszim), the manifest's
        ``doc_path``s ARE the indexable content — the ``text/*`` viewer shell,
        vendored sourcemaps (``*.css.map``/``*.js.map``/``*.d.ts.map``) and
        Handlebars templates (``*.handlebars``) that also live under ``text/*``
        are scaffolding, not content. Reading the manifest rows returns exactly
        those ``doc_path``s and excludes the viewer junk that would otherwise
        pollute the index.
        """
        if self._probe.kind == "documents":
            return await _document_entry_paths(self._db, self.id)
        return [
            cast("EntryPath", p) for p in (await self._dispatch(_text_entry_paths_sync, self._lz))
        ]

    async def extract(self, path: EntryPath) -> ExtractedArticle:
        raw = await self._dispatch(read_entry_sync, self._lz, path)
        if raw.is_redirect or raw.soft_redirect_target is not None:
            # Redirects carry no article text; classify and return empty text so
            # callers never try to embed a ~280-byte redirect shell.
            flags = classify_entry(path, raw.title, raw.content, is_redirect=raw.is_redirect)
            return ExtractedArticle(path=path, title=raw.title, text="", sections=(), flags=flags)
        # Mimetype-aware: HTML → resiliparse, vtt/plain/markdown → plain text.
        # Runs identically at query time and index time (the worker also calls
        # extract_entry), so chunk offsets recover consistently off the live
        # mimetype with no schema column.
        article = await self._dispatch(extract_entry, raw)
        # Fold in redirect/disambiguation classification from the raw entry.
        extra = classify_entry(
            path, raw.title, raw.content, is_redirect=False, char_len=len(article.text)
        )
        merged = article.flags | extra
        return ExtractedArticle(
            path=article.path,
            title=article.title,
            text=article.text,
            sections=article.sections,
            flags=merged,
        )

    def close(self) -> None:
        # libzim Archive frees its fd on GC; drop the reference.
        self._lz = None


# ``Sequence``/``partial`` are imported at the top of the module.


class ArchiveRegistry:
    """Owns every open archive and the ``zims``/``aliases`` rows behind them."""

    def __init__(
        self,
        db: Database,
        zims_dir: Path,
        *,
        read_pool_size: int = 4,
        cluster_cache_mb: int = 256,
    ) -> None:
        self._db = db
        self._zims_dir = zims_dir
        self._pool = ThreadPoolExecutor(max_workers=max(read_pool_size, 1))
        self._cluster_cache_mb = cluster_cache_mb
        self._archives: dict[int, LocalArchive] = {}
        self._enabled: set[int] = set()
        # vec0 tables are NOT FK-cascaded, so archive removal must cascade
        # vectors explicitly. ``zim/`` cannot import ``vectors/``, so the
        # composition root (``main``) registers the cascade here.
        self._on_remove: list[Any] = []

    def add_on_remove(self, callback: Any) -> None:
        """Register an ``async def(zim_id) -> None`` invoked before the zims row
        is deleted (the vector-store cascade is wired here from ``main``)."""
        self._on_remove.append(callback)

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> ScanResult:
        """Raise the cluster cache once, then run an initial discovery scan."""
        await self._dispatch(raise_cluster_cache, self._cluster_cache_mb)
        return await self.rescan()

    async def stop(self) -> None:
        """Close every held archive and shut the read pool down."""
        for arc in self._archives.values():
            arc.close()
        self._archives.clear()
        self._enabled.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)

    async def _dispatch(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, _partial(fn, *args, **kwargs))

    async def _dispatch_registration(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Dispatch one heavy registration-time batch call OFF the interactive
        pool (see ``_REGISTRATION_POOL_SIZE`` for why it must never go there).

        A fresh bounded executor per call, shut down without waiting once the
        await settles. ``wait=False`` keeps today's cancellation semantics: a
        cancelled registration abandons its running mining future exactly like
        a cancelled ``_dispatch`` does on the shared pool — the thread finishes
        unobserved instead of blocking cancellation on a ~29 s shutdown join.
        """
        loop = asyncio.get_running_loop()
        pool = ThreadPoolExecutor(
            max_workers=_REGISTRATION_POOL_SIZE,
            thread_name_prefix="vesta-zim-register",
        )
        try:
            return await loop.run_in_executor(pool, _partial(fn, *args, **kwargs))
        finally:
            pool.shutdown(wait=False)

    # ── reads ──────────────────────────────────────────────────────────────

    def get(self, zim_id: int) -> Archive:
        arc = self._archives.get(zim_id)
        if arc is None:
            raise KeyError(f"no open archive for zim_id={zim_id}")
        return arc

    def enabled(self, scope: Scope | None = None) -> list[Archive]:
        ids = (
            self._enabled
            if scope is None or scope.zim_ids is None
            else (self._enabled & scope.zim_ids)
        )
        return [self._archives[i] for i in sorted(ids) if i in self._archives]

    def has_any_fulltext(self) -> bool:
        """Capability input: ≥1 enabled archive with a probed fulltext index."""
        return any(
            self._archives[i].has_fulltext_index for i in self._enabled if i in self._archives
        )

    async def lookup_aliases(self, terms: Sequence[str], *, max_aliases: int) -> list[str]:
        """Expand query terms via the redirect alias table.

        Returns up to ``max_aliases`` canonical article-name fragments derived
        from the redirect table (e.g. ``"afaics" → "Internet slang"``). Offline
        and per-corpus; an empty result is valid. Matching is case-insensitive
        because query terms arrive lowercased (the ``normalize`` preparer) while
        redirect titles are human-readable title-case.

        Owned by ``zim/`` (not ``retrieval/``): the alias table is a ZIM-layer
        artefact populated at registration, so retrieval never queries the DB
        directly.
        """
        if not terms or max_aliases <= 0 or not self._enabled:
            return []
        ids = tuple(self._enabled)
        lower_terms = [t.lower() for t in terms if t]
        if not lower_terms:
            return []
        id_ph = ",".join("?" for _ in ids)
        term_ph = ",".join("?" for _ in lower_terms)
        async with self._db.read() as conn:
            cur = await conn.execute(
                f"SELECT DISTINCT target FROM aliases "
                f"WHERE zim_id IN ({id_ph}) AND lower(source) IN ({term_ph}) "
                f"LIMIT ?",
                (*ids, *lower_terms, max_aliases),
            )
            rows = [str(r[0]) for r in await cur.fetchall()]
        out: list[str] = []
        seen: set[str] = set()
        for target in rows:
            name = target.rsplit("/", 1)[-1] if "/" in target else target
            name = name.replace("_", " ").strip()
            low = name.lower()
            if name and low not in seen:
                seen.add(low)
                out.append(name)
        return out

    async def resolve_alias_targets(
        self, terms: Sequence[str], *, zim_ids: Collection[int] | None = None, max_aliases: int
    ) -> list[tuple[int, str]]:
        """Resolve query terms to exact ``(zim_id, path)`` alias targets.

        ``lookup_aliases`` collapses ``target`` to a lossy display-name fragment
        (basename, underscores→spaces) for feeding the FTS term ladder — it drops
        ``zim_id`` entirely because that caller never needs it. This method exists
        because a candidate source that wants to build a candidate directly from
        an alias hit needs the exact entry path and the owning archive id, not a
        display string (alias hits are canonical titles; they resolve to article
        candidates directly, without diluting an AND query). Same query shape
        as ``lookup_aliases`` (case-insensitive ``source`` match, scoped to
        enabled archives, optionally narrowed further by the caller's ``zim_ids``),
        but returns both columns untouched.

        Degrades to ``[]`` on any degenerate input — no terms, ``max_aliases <= 0``,
        no enabled archives, or a ``zim_ids`` scope that excludes every enabled
        archive — never raises.
        """
        if not terms or max_aliases <= 0 or not self._enabled:
            return []
        ids = self._enabled if zim_ids is None else (self._enabled & set(zim_ids))
        if not ids:
            return []
        lower_terms = [t.lower() for t in terms if t]
        if not lower_terms:
            return []
        id_ph = ",".join("?" for _ in ids)
        term_ph = ",".join("?" for _ in lower_terms)
        async with self._db.read() as conn:
            cur = await conn.execute(
                f"SELECT DISTINCT zim_id, target FROM aliases "
                f"WHERE zim_id IN ({id_ph}) AND lower(source) IN ({term_ph}) "
                f"LIMIT ?",
                (*ids, *lower_terms, max_aliases),
            )
            rows = await cur.fetchall()
        return [(int(r[0]), str(r[1])) for r in rows]

    async def ids_for_labels(self, labels: Collection[str]) -> frozenset[int]:
        """Enabled archive ids whose ``corpus_label`` is in ``labels``.

        Backs the retrieval ``Scope.corpus_labels`` axis: a query may scope by
        user-facing label (e.g. ``{"wikipedia"}``) instead of raw zim ids.
        Resolution lives here so the zim layer owns label→id mapping.
        """
        if not labels or not self._enabled:
            return frozenset()
        ph = ",".join("?" for _ in labels)
        async with self._db.read() as conn:
            cur = await conn.execute(
                f"SELECT id FROM zims WHERE enabled=1 AND corpus_label IN ({ph})",
                tuple(labels),
            )
            rows = [int(r[0]) for r in await cur.fetchall()]
        return frozenset(rows) & self._enabled

    def resolve_scope_token(self, token: str) -> int | None:
        """Resolve one ``--scope``/``?scope=`` token to a ``zim_id``, or ``None``.

        Callers (``_parse_scope`` in ``api/answer.py``) already accept bare
        integer ids directly; this covers the documented archive-name form
        (``benchmarks/README.md``: ``--scope wikipedia_en_top_nopic_2026-06.zim``
        "or the short archive id"). Matches, case-insensitively, against the
        in-memory probe metadata cached at registration time — no DB round
        trip needed:

        * the ZIM's ``Name`` metadata (``zims.name``, e.g. ``wikipedia_en_top``)
        * the archive filename (``zims.filename``, e.g.
          ``wikipedia_en_top_nopic_2026-06.zim``)
        * that filename with the ``.zim`` suffix stripped

        Exact matching only, deliberately — no fuzzy/substring matching. A
        token that doesn't match any of these must come back ``None`` so the
        caller can fail loudly instead of silently treating the scope as
        unscoped (the bug this replaces: an unresolvable token used to widen
        silently to "every archive").
        """
        needle = token.strip().lower()
        if not needle:
            return None
        for zim_id, arc in self._archives.items():
            name = (arc.name or "").lower()
            filename = (arc.filename or "").lower()
            stem = filename[:-4] if filename.endswith(".zim") else filename
            if needle in (name, filename, stem):
                return zim_id
        return None

    # ── discovery ──────────────────────────────────────────────────────────

    async def rescan(self) -> ScanResult:
        """Scan ``zims_dir`` for ``*.zim``; register new, retire missing.

        Cheap probes (Counter, has_fulltext_index) refresh on every scan; the
        expensive alias mining runs only for *newly* registered archives, so a
        repeat scan over a large corpus stays fast.
        """
        self._zims_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(self._zims_dir.glob("*.zim"))
        file_by_uuid: dict[str, tuple[LibzimArchive, _Probe, Path]] = {}
        for path in files:
            try:
                archive, probe = await self._dispatch(_probe_archive, path)
            except Exception as exc:  # a corrupt/unreadable ZIM never crashes discovery
                _log.warning("zim.open_failed", extra={"path": str(path), "error": repr(exc)})
                continue
            file_by_uuid[probe.uuid] = (archive, probe, path)

        # Load existing rows by uuid (uuid survives renames).
        async with self._db.read() as conn, conn.execute("SELECT * FROM zims") as cur:
            rows = [dict(r) for r in await cur.fetchall()]
        row_by_uuid = {r["uuid"]: r for r in rows if r["uuid"]}

        added: list[int] = []
        updated: list[int] = []
        for uuid, (archive, probe, path) in file_by_uuid.items():
            existing = row_by_uuid.get(uuid)
            if existing is None:
                zim_id = await self._register_new(archive, probe, path)
                added.append(zim_id)
            else:
                zim_id = await self._refresh_existing(existing, archive, probe, path)
                if zim_id not in self._archives:
                    updated.append(zim_id)
                else:
                    # already open; nothing structural changed
                    pass
            row_by_uuid.pop(uuid, None)

        # Anything left in row_by_uuid had no file on disk → mark missing.
        missing: list[int] = []
        for _uuid, row in row_by_uuid.items():
            zim_id = int(row["id"])
            if zim_id in self._archives:
                self._archives[zim_id].close()
                self._archives.pop(zim_id, None)
                self._enabled.discard(zim_id)
            if row["status"] != "missing":
                await self._set_status(zim_id, "missing")
            missing.append(zim_id)

        return ScanResult(
            added=tuple(added),
            updated=tuple(updated),
            missing=tuple(missing),
            total=len(self._archives),
        )

    async def _register_new(self, archive: LibzimArchive, probe: _Probe, path: Path) -> int:
        """Insert a zims row, mine aliases, and hold the archive open."""
        now = _now_iso()
        async with self._db.write() as conn:
            cur = await conn.execute(
                "INSERT INTO zims(uuid, filename, path, name, title, description, "
                "language, flavour, publisher, zim_date, file_size, article_count, "
                "media_count, has_fulltext_index, corpus_label, kind, scraper, tags, "
                "enabled, status, index_depth, index_status, added_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,'known',0,'none',?)",
                (
                    probe.uuid,
                    probe.filename,
                    str(path),
                    probe.name,
                    probe.title,
                    probe.description,
                    probe.language,
                    probe.flavour,
                    probe.publisher,
                    probe.zim_date,
                    probe.file_size,
                    probe.article_count,
                    probe.media_count,
                    1 if probe.has_fulltext_index else 0,
                    _corpus_label(probe.name),
                    probe.kind,
                    probe.scraper,
                    probe.tags,
                    now,
                ),
            )
            zim_id = int(cur.lastrowid) if cur.lastrowid is not None else 0
        # Alias mining is the expensive step (~14 k entries/s) — only for
        # newly registered archives, and off the interactive read pool so a
        # large registration never starves serving (see _REGISTRATION_POOL_SIZE).
        pairs = await self._dispatch_registration(aliases_mod.mine_aliases, archive)
        await self._store_aliases(zim_id, pairs)
        # Media manifest: for media-kind archives, resolve each browsable
        # stub to its video/poster/duration from the per-video JSON sidecars so
        # the frontend can render a native <video>. No-op for article ZIMs.
        media_rows = 0
        if probe.kind == "media":
            try:
                media_rows = await media_mod.build_media_manifest(self._db, archive, zim_id)
            except Exception as exc:  # never let a bad sidecar block registration
                _log.warning("zim.media_manifestFailed zim_id=%d error=%r", zim_id, exc)
        # Documents manifest: for documents-kind archives (nautiluszim),
        # parse database.js into a (doc_path → title/description/author/mime)
        # catalog so the frontend can render a document library and the indexer
        # can title the PDFs. No-op for article ZIMs.
        doc_rows = 0
        if probe.kind == "documents":
            try:
                doc_rows = await documents_mod.build_documents_manifest(self._db, archive, zim_id)
            except Exception as exc:  # never let a bad manifest block registration
                _log.warning("zim.documents_manifestFailed zim_id=%d error=%r", zim_id, exc)
        local = LocalArchive(archive, probe, zim_id, self._pool, self._db)
        self._archives[zim_id] = local
        self._enabled.add(zim_id)
        _log.info(
            "zim.registered",
            extra={
                "zim_id": zim_id,
                "uuid": probe.uuid,
                "zim_name": probe.name,
                "articles": probe.article_count,
                "fulltext": probe.has_fulltext_index,
                "aliases": len(pairs),
                "kind": probe.kind,
                "media": media_rows,
                "documents": doc_rows,
            },
        )
        return zim_id

    async def _refresh_existing(
        self, row: dict[str, Any], archive: LibzimArchive, probe: _Probe, path: Path
    ) -> int:
        """Refresh cheap probes/path/status for a known archive and hold it open."""
        zim_id = int(row["id"])
        async with self._db.write() as conn:
            await conn.execute(
                "UPDATE zims SET filename=?, path=?, title=?, file_size=?, "
                "article_count=?, media_count=?, has_fulltext_index=?, kind=?, "
                "scraper=?, tags=?, status='known' WHERE id=?",
                (
                    probe.filename,
                    str(path),
                    probe.title,
                    probe.file_size,
                    probe.article_count,
                    probe.media_count,
                    1 if probe.has_fulltext_index else 0,
                    probe.kind,
                    probe.scraper,
                    probe.tags,
                    zim_id,
                ),
            )
        if zim_id not in self._archives:
            self._archives[zim_id] = LocalArchive(archive, probe, zim_id, self._pool, self._db)
            if int(row.get("enabled") or 0):
                self._enabled.add(zim_id)
        # Media manifest: an archive that became media-kind but was
        # registered before this feature has no manifest rows yet — build them
        # on refresh (idempotent: skipped once rows exist). Covers the live
        # upgrade path; new registrations build the manifest in _register_new.
        if probe.kind == "media" and not await self._has_media_manifest(zim_id):
            try:
                media_rows = await media_mod.build_media_manifest(self._db, archive, zim_id)
                if media_rows:
                    _log.info("zim.media_manifestRefreshed zim_id=%d rows=%d", zim_id, media_rows)
            except Exception as exc:  # never let a bad sidecar block the scan
                _log.warning("zim.media_manifestFailed zim_id=%d error=%r", zim_id, exc)
        # Documents manifest: same live-upgrade path as media — an
        # archive that newly classifies as documents but was registered before
        # this feature has no catalog rows yet. Build them on refresh
        # (idempotent: skipped once rows exist). New registrations build the
        # manifest in _register_new.
        if probe.kind == "documents" and not await self._has_documents_manifest(zim_id):
            try:
                doc_rows = await documents_mod.build_documents_manifest(self._db, archive, zim_id)
                if doc_rows:
                    _log.info("zim.documents_manifestRefreshed zim_id=%d rows=%d", zim_id, doc_rows)
            except Exception as exc:  # never let a bad manifest block the scan
                _log.warning("zim.documents_manifestFailed zim_id=%d error=%r", zim_id, exc)
        return zim_id

    async def _has_media_manifest(self, zim_id: int) -> bool:
        async with (
            self._db.read() as conn,
            conn.execute("SELECT 1 FROM article_media WHERE zim_id=? LIMIT 1", (zim_id,)) as cur,
        ):
            return await cur.fetchone() is not None

    async def _has_documents_manifest(self, zim_id: int) -> bool:
        async with (
            self._db.read() as conn,
            conn.execute(
                "SELECT 1 FROM article_documents WHERE zim_id=? LIMIT 1", (zim_id,)
            ) as cur,
        ):
            return await cur.fetchone() is not None

    async def _set_status(self, zim_id: int, status: str) -> None:
        async with self._db.write() as conn:
            await conn.execute("UPDATE zims SET status=? WHERE id=?", (status, zim_id))

    async def _store_aliases(self, zim_id: int, pairs: list[tuple[str, str]]) -> None:
        if not pairs:
            return
        async with self._db.write() as conn:
            await conn.execute("DELETE FROM aliases WHERE zim_id=?", (zim_id,))
            await conn.executemany(
                "INSERT INTO aliases(zim_id, source, target) VALUES(?,?,?)",
                [(zim_id, s, t) for s, t in pairs],
            )

    # ── management actions (backing the API) ───────────────────────────────

    async def set_enabled(self, zim_id: int, enabled: bool) -> bool:
        if zim_id not in self._archives:
            return False
        async with self._db.write() as conn:
            await conn.execute(
                "UPDATE zims SET enabled=? WHERE id=?",
                (1 if enabled else 0, zim_id),
            )
        if enabled:
            self._enabled.add(zim_id)
        else:
            self._enabled.discard(zim_id)
        return True

    async def set_corpus_label(self, zim_id: int, label: str) -> bool:
        if zim_id not in self._archives:
            return False
        async with self._db.write() as conn:
            await conn.execute(
                "UPDATE zims SET corpus_label=? WHERE id=?",
                (label, zim_id),
            )
        return True

    async def remove(self, zim_id: int, *, delete_file: bool = False) -> bool:
        """Drop an archive from the registry and cascade-delete its rows.

        When ``delete_file`` is true the underlying ``.zim`` file (and any stale
        ``.part``) is removed too ("delete the file and every DB reference").
        The default keeps the file so existing callers and the ``./data/zims/``
        rescan stay intact. Either way the DB cascade is total:
        zims row → articles/aliases/chunks/index_meta (FK CASCADE) + vectors
        (the ``_on_remove`` callback wired in ``main``, since vec0 isn't
        FK-cascaded).
        """
        if zim_id not in self._archives:
            return False
        # Read the file path before the row is gone (only if we'll delete it).
        path_to_delete: str | None = None
        if delete_file:
            async with (
                self._db.read() as conn,
                conn.execute("SELECT path FROM zims WHERE id=?", (zim_id,)) as cur,
            ):
                row = await cur.fetchone()
            path_to_delete = str(row["path"]) if row is not None and row["path"] else None
        # Cascade vectors before closing the archive / dropping the zims row.
        # Tolerated: a failing callback must not block archive removal.
        for cb in self._on_remove:
            try:
                await cb(zim_id)
            except Exception as exc:
                _log.warning(
                    "zim.remove_callback_failed", extra={"zim_id": zim_id, "error": repr(exc)}
                )
        self._archives[zim_id].close()
        self._archives.pop(zim_id, None)
        self._enabled.discard(zim_id)
        async with self._db.write() as conn:
            await conn.execute("DELETE FROM zims WHERE id=?", (zim_id,))
        if path_to_delete:
            self._unlink_archive(path_to_delete)
        return True

    @staticmethod
    def _unlink_archive(path_str: str) -> None:
        """Best-effort file removal; never raises (removal already succeeded)."""
        from pathlib import Path

        target = Path(path_str)
        for candidate in (target, Path(str(target) + ".part")):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:  # pragma: no cover - permission/etc, non-fatal
                _log.warning(
                    "zim.unlink_failed", extra={"path": str(candidate), "error": repr(exc)}
                )


__all__ = [
    "ArchiveRegistry",
    "EntryNotFound",
    "LocalArchive",
    "raise_cluster_cache",
]
