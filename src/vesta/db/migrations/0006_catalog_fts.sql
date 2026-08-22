-- Migration 0006 — catalog full-text index: server-side FTS5 filtering over the
-- ~3 k catalog entries.
--
-- The ``catalog_entries`` table itself was created in 0001_init.sql; this adds a
-- full-text index so ``GET /api/catalog?q=…`` filters server-side and the client
-- never receives the full ~3 000-row list (virtualization is
-- unnecessary when the server filters).
--
-- Why a standalone FTS5 table (not external-content): ``catalog_entries.id`` is
-- the OPDS ``urn:uuid`` (TEXT primary key), but FTS5's rowid is an internal
-- integer, which makes external-content mapping awkward. A standalone table the
-- OPDS client repopulates on every refresh is simpler and refresh always
-- replaces the whole catalog anyway (defensive parse → truncate + bulk insert).
-- ``entry_id`` is UNINDEXED so it round-trips back to join on ``catalog_entries``.
--
-- FTS5 is built into the stdlib ``sqlite3`` module; no extension load needed
-- (unlike the vec0 table in 0004), so a static migration is safe here.

CREATE VIRTUAL TABLE catalog_fts USING fts5(
    name,
    title,
    description,
    tags,
    entry_id UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
);
