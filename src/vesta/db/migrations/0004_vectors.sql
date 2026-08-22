-- Migration 0004 — semantic-index foundation. Adds the chunk metadata table and
-- the per-archive embedder compat record used to refuse mismatched queries.
--
-- What is NOT here: the `vectors_d{N}` vec0 virtual tables. Those are created
-- lazily by `SqliteVecStore` (`vectors/sqlite_vec_store.py`), not in this
-- migration, for two reasons:
--
--   1. vec0 requires the `vec0` extension to be loaded on the connection *before*
--      `CREATE VIRTUAL TABLE` runs. The extension load is gated try/except in
--      `connection.py` (degrade-don't-fail: a build without the extension
--      stays healthy). If this static migration script issued
--      `CREATE VIRTUAL TABLE ... USING vec0(...)`, an app on a vec0-less build
--      could not start — the migration runner has no per-statement try/except.
--      Keeping vec0 DDL in the store (which checks availability first) preserves
--      that invariant: an empty ./data on a vec0-less box still reaches
--      vesta.ready.
--   2. `rescore(quantizer=bit, oversample=8)` is the *intended* default, but
--      no published sqlite-vec wheel (0.1.9 / 0.1.10a4) implements the `rescore`
--      table option yet; a self-compiled `-O3 -mavx2 -mfma` build owns
--      enabling it. The store attempts the rescore DDL first and falls back to a
--      plain flat index, so a rescore-capable build picks it up automatically —
--      something a static migration could not do.
--
-- The end state is identical to "the migration created vectors_d384": the store
-- eagerly creates `vectors_d384` at construction time when vec0 is available
-- (main.py lifespan), and creates other dims (e.g. vectors_d768) on first upsert.

-- ── Chunks: the uniform unit of indexing across depths 1/2/3 ────────────────
-- `id` is the vector id (== vec0 primary key). At depth 1 chunk↔article is 1:1,
-- honouring the "articles.id doubles as the vector id" principle in spirit
-- while generalising to depths 2/3 where one article yields many chunks. NO text
-- is stored; char_start/
-- char_end recover the span from the ZIM at query time.
-- `ordinal`/`depth` are owned by the indexer; the store's
-- upsert writes (id, zim_id, article_id, char_start, char_end) and leaves them
-- to the indexer (ON CONFLICT(id) preserves them on re-upsert).
CREATE TABLE chunks (
    id          INTEGER PRIMARY KEY,
    zim_id      INTEGER NOT NULL REFERENCES zims(id) ON DELETE CASCADE,
    article_id  INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    ordinal     INTEGER,                 -- per-article chunk order (indexer-owned)
    char_start  INTEGER,                 -- offset into ExtractedArticle.text
    char_end    INTEGER,
    depth       INTEGER,                 -- 1|2|3 (indexer-owned)
    UNIQUE(zim_id, article_id, ordinal)
);
CREATE INDEX chunks_zim ON chunks(zim_id);
CREATE INDEX chunks_article ON chunks(article_id);

-- ── Per-archive embedder compat record ─────────────────────────────────────
-- One row per indexed archive. A query whose embedder differs from the index's
-- embedder is REFUSED, not silently served: a mismatched embedder returns
-- plausible-looking garbage — the worst failure mode for a grounded-answer
-- product. The store reads this via
-- `describe(zim_id)`; the dense source compares it against the live
-- query embedder before searching.
CREATE TABLE index_meta (
    zim_id          INTEGER PRIMARY KEY REFERENCES zims(id) ON DELETE CASCADE,
    embedder_id     TEXT NOT NULL,        -- HF repo id the index was built with
    dim             INTEGER NOT NULL,     -- embedding dimensionality → vectors_d{dim}
    query_prefix    TEXT NOT NULL,        -- asymmetric-prefix guard
    passage_prefix  TEXT NOT NULL,
    pooling         TEXT NOT NULL,        -- "cls" | "mean"
    normalize       INTEGER NOT NULL      -- BOOLEAN; L2-normalized?
);
