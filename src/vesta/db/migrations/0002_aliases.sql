-- Migration 0002 — alias dictionary.
--
-- Mines each archive's redirect table at registration time into an
-- acronym/alias map (108,571 redirects → 62,911 targets on
-- Simple Wikipedia, yielding real expansions like `AFAICS → Internet_slang`).
-- Free, offline, per-corpus. Rebuilt whenever an archive is (re)registered.
--
-- `source` is the redirect entry's *title* (human-readable alias, e.g. "AIM
-- speak"); `target` is the redirect target's *path* (the canonical article).
-- Forward-only: no down-migration. Cascade-deletes with the owning zims row.

CREATE TABLE aliases (
    zim_id  INTEGER NOT NULL REFERENCES zims(id) ON DELETE CASCADE,
    source  TEXT NOT NULL,            -- redirect entry title (the alias)
    target  TEXT NOT NULL,            -- redirect target entry path (canonical)
    PRIMARY KEY (zim_id, source)
);

CREATE INDEX aliases_target ON aliases(zim_id, target);
