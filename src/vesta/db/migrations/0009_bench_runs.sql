-- Migration 0009 — unified benchmark runs.
--
-- Replaces the answer_runs blob model with per-question rows so
-- comparing two runs question-by-question is a JOIN (not a JSON parse of two
-- megabyte blobs), and trace pruning is a column NULL (not a blob rewrite).
--
-- Three tables:
--   bench_runs             — one row per system × profile × model in a group
--   bench_question_results — per-question rows, verdict pending|correct|partial|
--                            incorrect|unjudged, trace_json prunable
--   bench_judge_cache      — keyed on sha256(rendered_rubric|qid|answer|
--                            judge_model); rendered rubric embeds ground truth
--                            so a GT fix invalidates its own entries (trap 17)
--
-- answer_runs is NOT dropped here: its rows are imported once into
-- the new tables and the old table left read-only for historical reference.

CREATE TABLE bench_runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_group      TEXT NOT NULL,        -- uuid; the comparison unit
    label          TEXT NOT NULL DEFAULT '',
    started_at     TEXT NOT NULL,
    finished_at    TEXT,
    status         TEXT NOT NULL,        -- running | complete | aborted
    dataset_name   TEXT NOT NULL,
    dataset_hash   TEXT NOT NULL,        -- full-set hash (GT-edit-insensitive)
    subset_hash    TEXT NOT NULL DEFAULT '',   -- when --slice/--limit applied
    system         TEXT NOT NULL,        -- agentic | agentic_pydantic | ...
    profile_name   TEXT NOT NULL,
    profile_hash   TEXT NOT NULL,
    answer_model   TEXT NOT NULL,
    judge_model    TEXT NOT NULL,
    scope          TEXT NOT NULL DEFAULT '',
    trusted        INTEGER NOT NULL DEFAULT 0,
    calibration    REAL,
    config_json    TEXT NOT NULL,        -- archive checksums, git sha, machine,
                                         -- settings snapshot, rubric hash, seed
    metrics_json   TEXT NOT NULL         -- all aggregates incl. per-capability
);
CREATE INDEX bench_runs_group   ON bench_runs(run_group);
CREATE INDEX bench_runs_started ON bench_runs(started_at);
CREATE INDEX bench_runs_dataset ON bench_runs(dataset_hash);

CREATE TABLE bench_question_results (
    run_id            INTEGER NOT NULL REFERENCES bench_runs(id) ON DELETE CASCADE,
    question_id       TEXT    NOT NULL,   -- the stable slug
    capability        TEXT    NOT NULL,
    difficulty        TEXT    NOT NULL,
    question_text     TEXT    NOT NULL,   -- pinned: dataset may drift
    expected_answer   TEXT    NOT NULL,   -- pinned
    answer_text       TEXT    NOT NULL,
    abstained         INTEGER NOT NULL,
    verdict           TEXT    NOT NULL,   -- pending|correct|partial|incorrect|unjudged
                                          --   'pending' = answered, not yet judged
    verdict_reason    TEXT    NOT NULL DEFAULT '',
    source_hit_rank   INTEGER,            -- NULL = miss
    source_coverage   REAL    NOT NULL DEFAULT 0,
    sub_fact_coverage REAL,               -- NULL when no sub-facts
    retrieved_paths   TEXT    NOT NULL,   -- json array
    rounds            INTEGER NOT NULL DEFAULT 0,
    latency_ms        REAL    NOT NULL DEFAULT 0,
    error             TEXT,
    trace_json        TEXT,               -- prunable
    PRIMARY KEY (run_id, question_id)
);
CREATE INDEX bench_question_results_q ON bench_question_results(question_id);

CREATE TABLE bench_judge_cache (
    key         TEXT PRIMARY KEY,         -- sha256(rendered_rubric|qid|answer|judge_model)
    verdict     TEXT NOT NULL,
    reason      TEXT NOT NULL,
    payload     TEXT NOT NULL,            -- full structured judge JSON
    created_at  TEXT NOT NULL
);
