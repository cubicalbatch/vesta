"""Media manifest — resolve browsable entry → playable assets for media ZIMs.

For ``kind='media'`` archives (youtube2zim, ted2zim, …) the browsable
``text/html`` entries are meta-refresh stubs; the real playable asset paths live
only in the per-video ``application/json`` sidecars. This module reads those
sidecars and persists a ``(zim_id, entry_path) → (video_path, poster_path,
duration)`` mapping into ``article_media`` (migration 0008), so the frontend can
render a native ``<video>`` from a card's path without touching JSON or relaxing
the Reader sandbox.

Design rules:

* **Field-name-driven, not scraper-specific.** A JSON entry is treated as a
  media descriptor iff it has a ``videoPath`` field; ``thumbnailPath`` and
  ``duration`` are read by name. ted2zim uses the same schema.
* **Best-effort stub-path derivation.** The manifest is keyed by the browsable
  stub path a search/browse candidate carries. The stub ↔ JSON correspondence
  follows the youtube2zim ``videos/<slug>.json ↔ index/<slug>`` layout; a
  scraper with a different layout simply yields no manifest (graceful:
  title fallback still surfaces cards). Rows whose derived stub is absent from
  the archive are dropped so the manifest never points at a non-entry.
* **Independent of the semantic index.** Populated at registration, never
  touched by a re-index; FK-cascades with the zims row.

``zim/`` depends on ``db`` and ``config`` only.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vesta.zim.types import EntryPath, MediaRef

if TYPE_CHECKING:
    from libzim.reader import Archive as LibzimArchive

    from vesta.db.connection import Database

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MediaRecord:
    """One resolved media descriptor before it reaches ``article_media``."""

    entry_path: EntryPath  # the browsable stub path
    video_path: str | None
    poster_path: str | None
    duration: int | None  # seconds


#: ISO-8601 duration (``PnDTnHnMnS``). youtube2zim emits durations as strings
#: like ``PT10M26S``; ``P1DT2H3M4.5S`` also appears (ttml-era). The regex
#: captures days then the optional ``T`` time part.
_ISO_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?"
    r"(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?$"
)


def _coerce_duration(value: object) -> int | None:
    """JSON ``duration`` arrives as float seconds OR an ISO-8601 string (the
    youtube2zim shape). Returns whole seconds, or ``None`` if neither."""
    if isinstance(value, bool):  # bool is an int subclass — exclude explicitly
        return None
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str):
        m = _ISO_DURATION_RE.match(value.strip())
        if m:
            parts = m.groupdict()
            total = 0
            total += int(parts["days"] or 0) * 86400
            total += int(parts["hours"] or 0) * 3600
            total += int(parts["minutes"] or 0) * 60
            if parts["seconds"]:
                total += int(float(parts["seconds"]))
            return total if total > 0 else None
    return None


def _mine_media_sync(archive: LibzimArchive) -> list[MediaRecord]:
    """One pass over the archive's ``application/json`` entries, returning the
    media descriptors (those carrying a ``videoPath``). Blocking; the registry
    dispatches it on the read pool."""
    out: list[MediaRecord] = []
    for i in range(archive.entry_count):
        try:
            entry = archive._get_entry_by_id(i)
        except Exception:  # a corrupt entry never aborts the sweep
            continue
        try:
            mimetype = str(entry.get_item().mimetype)
        except Exception:
            continue
        if mimetype != "application/json":
            continue
        path = str(entry.path)
        try:
            data = json.loads(bytes(entry.get_item().content).decode("utf-8", "replace"))
        except Exception:  # a malformed JSON sidecar is skipped, never fatal
            continue
        if not isinstance(data, dict):
            continue
        video_path = data.get("videoPath")
        if not isinstance(video_path, str) or not video_path:
            continue  # not a per-video descriptor (channel.json / playlists / config)
        # Derive the browsable stub path: ``videos/<slug>.json`` → ``index/<slug>``.
        slug = path.rsplit("/", 1)[-1]
        if slug.endswith(".json"):
            slug = slug[: -len(".json")]
        stub = f"index/{slug}"
        # Only keep rows whose stub actually exists as an entry, so the manifest
        # never points a card at a non-entry. Cheap against libzim's path index.
        if not archive.has_entry_by_path(stub):
            continue
        poster = data.get("thumbnailPath")
        out.append(
            MediaRecord(
                entry_path=stub,
                video_path=video_path,
                poster_path=poster if isinstance(poster, str) else None,
                duration=_coerce_duration(data.get("duration")),
            )
        )
    return out


async def build_media_manifest(db: Database, archive: LibzimArchive, zim_id: int) -> int:
    """Mine an archive's media descriptors and persist them to ``article_media``.

    Wipes the archive's prior rows first so a re-registration is a clean
    refresh. Returns the row count. Owned by ``zim/`` (called from the registry
    at registration, gated on ``kind='media'``).
    """
    import asyncio

    # The blocking libzim sweep runs off the event loop via to_thread — this
    # module stays executor-agnostic (the registry owns the pool, but to_thread
    # is the simpler, sufficient choice for a one-shot registration pass that
    # runs outside the hot search/read path).
    records = await asyncio.to_thread(_mine_media_sync, archive)
    async with db.write() as conn:
        await conn.execute("DELETE FROM article_media WHERE zim_id=?", (zim_id,))
        if records:
            await conn.executemany(
                "INSERT INTO article_media(zim_id, entry_path, video_path, poster_path, duration) "
                "VALUES(?,?,?,?,?)",
                [(zim_id, r.entry_path, r.video_path, r.poster_path, r.duration) for r in records],
            )
    _log.info("zim.media_manifestBuilt zim_id=%d rows=%d", zim_id, len(records))
    return len(records)


async def fetch_media_for_paths(
    db: Database, zim_id: int, paths: list[EntryPath]
) -> dict[EntryPath, MediaRef]:
    """Batch-fetch ``MediaRef`` for the given entry paths in one archive.

    Returns ``{path: MediaRef}`` only for paths that have a manifest row; paths
    without one are simply absent (callers treat absence as "not a media
    entry"). Used by the API layer to enrich ArticleOut / search cards without a
    per-row round-trip.
    """
    if not paths:
        return {}
    # De-dup to avoid OR-list blowups; placeholders sized to the unique set.
    unique = list(dict.fromkeys(paths))
    placeholders = ",".join("?" for _ in unique)
    query = (
        "SELECT entry_path, video_path, poster_path, duration FROM article_media "
        f"WHERE zim_id=? AND entry_path IN ({placeholders})"
    )
    async with db.read() as conn:
        cur = await conn.execute(query, (zim_id, *unique))
        rows = await cur.fetchall()
    return {
        str(r["entry_path"]): MediaRef(
            video_path=r["video_path"] if r["video_path"] is not None else None,
            poster_path=r["poster_path"] if r["poster_path"] is not None else None,
            duration=int(r["duration"]) if r["duration"] is not None else None,
        )
        for r in rows
    }


__all__ = ["MediaRecord", "MediaRef", "build_media_manifest", "fetch_media_for_paths"]
