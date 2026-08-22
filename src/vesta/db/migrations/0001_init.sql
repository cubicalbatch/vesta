-- Migration 0001 — initial schema.
-- The `vectors` vec0 virtual table needs the sqlite-vec extension loaded and
-- is therefore not part of this static migration.
--
-- Notes on types: SQLite has no native BOOLEAN; booleans are INTEGER 0/1.
-- Timestamps are TEXT in ISO-8601 (UTC). JSON columns are TEXT holding JSON.
-- `articles.id` doubles as the vector id to keep the join free.

-- ── Archives ──────────────────────────────────────────────────────────────
CREATE TABLE zims (
    id INTEGER PRIMARY KEY,
    uuid TEXT UNIQUE,                 -- ZIM's own UUID; survives renames
    filename TEXT,
    path TEXT,
    name TEXT,
    title TEXT,
    description TEXT,
    language TEXT,
    flavour TEXT,
    publisher TEXT,
    zim_date TEXT,
    file_size INTEGER,
    article_count INTEGER,            -- from Counter metadata 'text/html', NOT
                                      -- archive.article_count (~40% over-count)
    media_count INTEGER,
    has_fulltext_index INTEGER,       -- BOOLEAN; PROBED at runtime — the catalog's
                                      -- _ftindex tag has ~41% false negatives
    corpus_label TEXT,                -- user-facing scope name, e.g. "wikipedia"
    enabled INTEGER,                  -- BOOLEAN; included in searches
    status TEXT,                      -- known|downloading|ready|error|missing
    index_depth INTEGER,              -- 0..3
    index_status TEXT,                -- none|running|paused|complete|stale|error
    index_progress INTEGER,
    index_total INTEGER,
    embedding_model TEXT,
    embedding_dim INTEGER,
    added_at TEXT,
    downloaded_at TEXT,
    indexed_at TEXT
);

-- ── Articles: pointers only, never text ──────────────────────────────────
CREATE TABLE articles (
    id INTEGER PRIMARY KEY,           -- also the vector id
    zim_id INTEGER REFERENCES zims(id) ON DELETE CASCADE,
    entry_path TEXT,                  -- libzim entry path
    title TEXT,
    char_len INTEGER,
    n_sections INTEGER,
    flags INTEGER,                    -- bitfield: redirect|disambiguation|list|stub
    UNIQUE(zim_id, entry_path)
);
CREATE INDEX articles_zim ON articles(zim_id);

-- ── Jobs: resumable, restart-surviving ───────────────────────────────────
CREATE TABLE jobs (
    id INTEGER PRIMARY KEY,
    type TEXT,                        -- download_zim|index_zim|download_model|...
    target TEXT,                      -- zim id / model id / url
    params TEXT,                      -- JSON; immutable input params. Job params
                                      -- must survive a restart (resumable
                                      -- noop/download). Minimal, documented
                                      -- deviation from a fully normalized design.
    status TEXT,                      -- queued|running|paused|done|error|cancelled
    progress INTEGER,
    total INTEGER,
    checkpoint TEXT,                  -- JSON; resume cursor (byte offset / last id)
    message TEXT,
    error TEXT,
    rate REAL,
    eta_seconds INTEGER,
    created_at TEXT,
    updated_at TEXT,
    finished_at TEXT
);

-- ── Models & endpoints ───────────────────────────────────────────────────
CREATE TABLE models (
    id INTEGER PRIMARY KEY,
    role TEXT,                        -- llm|embed|rerank
    source TEXT,                      -- local|endpoint
    display_name TEXT,
    repo TEXT,
    filename TEXT,
    path TEXT,                        -- local (HF repo + gguf file)
    endpoint_url TEXT,
    endpoint_model TEXT,
    api_key_ref TEXT,                 -- remote
    context_length INTEGER,
    dim INTEGER,
    quant TEXT,
    size_bytes INTEGER,
    status TEXT,                      -- available|downloading|error
    active INTEGER                    -- BOOLEAN; at most one active per role
);

-- ── Catalog cache (from Kiwix OPDS) ──────────────────────────────────────
CREATE TABLE catalog_entries (
    id TEXT PRIMARY KEY,              -- OPDS entry id
    name TEXT,
    title TEXT,
    description TEXT,
    language TEXT,
    flavour TEXT,
    tags TEXT,
    size_bytes INTEGER,
    article_count INTEGER,
    url TEXT,
    illustration_url TEXT,
    zim_date TEXT,
    curated_rank INTEGER,             -- non-null ⇒ appears in "Recommended"
    fetched_at TEXT
);

-- ── Conversations ────────────────────────────────────────────────────────
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY,
    title TEXT,
    created_at TEXT,
    updated_at TEXT
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT,
    content TEXT,
    sources_json TEXT,                -- source cards as shown
    trace_json TEXT,                  -- retrieval trace, nullable/prunable
    tokens_in INTEGER,
    tokens_out INTEGER,
    latency_ms INTEGER,
    created_at TEXT
);

-- ── Settings: key/value, overrides env ──────────────────────────────────
CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT
);

-- ── Eval runs ────────────────────────────────────────────────────────────
CREATE TABLE eval_runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT,
    config_json TEXT,
    metrics_json TEXT
);
