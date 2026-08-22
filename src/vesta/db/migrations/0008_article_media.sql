-- Migration 0008 — per-entry media manifest (video / poster / duration).
--
-- For media-kind ZIMs (youtube2zim, ted2zim, …) the browsable ``text/html``
-- entries are meta-refresh stubs; the playable asset paths live only in the
-- per-video ``application/json`` sidecars (``videoPath`` / ``thumbnailPath`` /
-- ``duration``). This table is the resolved mapping from a browsable entry path
-- to its media assets, so the frontend can render a native ``<video>`` without
-- touching JSON or relaxing the Reader sandbox.
--
-- Populated by ``zim/media.py`` at archive registration (gated on
-- ``zims.kind = 'media'``), field-name-driven (never scraper-specific). Keyed by
-- ``(zim_id, entry_path)`` — the stub path a search/browse candidate carries —
-- so a card or Reader lookup is one indexed equality read. FK-cascades with the
-- zims row; independent of the semantic index (a re-index never touches it).

CREATE TABLE article_media (
    zim_id      INTEGER NOT NULL REFERENCES zims(id) ON DELETE CASCADE,
    entry_path  TEXT    NOT NULL,   -- the browsable stub path (a candidate path)
    video_path  TEXT,                -- e.g. videos/<id>/video.webm
    poster_path TEXT,                -- e.g. videos/<id>/video.webp
    duration    INTEGER,             -- seconds
    UNIQUE(zim_id, entry_path)
);

CREATE INDEX article_media_zim ON article_media(zim_id);
