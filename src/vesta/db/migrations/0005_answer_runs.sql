-- Migration 0005 — answer benchmark runs.
--
-- One row per end-to-end gap-questions benchmark run. Distinct from eval_runs:
-- that table's RunRecord models retrieval RunMetrics (recall/nDCG/MRR), which
-- does not fit the verdict counts + per-question answer results this harness
-- produces. A new table keeps the existing persistence contract untouched.
--
-- config_json pins everything that makes a run comparable (parallel to
-- eval_runs' pin discipline): profile, strategies, scope, resolved
-- tiers, settings snapshot, git_sha, machine_id, judge prompt hash.
-- results_json holds per-strategy aggregates + per-question verdicts/retrieval/
-- traces (traces are prunable, mirroring chat.trace_retention_days).

CREATE TABLE answer_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    dataset_hash  TEXT NOT NULL,      -- content hash of gap_questions.json used
    judge_model   TEXT NOT NULL,      -- "" = lexical-only (inconclusive)
    config_json   TEXT NOT NULL,      -- profile, strategies, scope, resolved
                                      --   tiers, settings snapshot, git_sha,
                                      --   machine_id, judge prompt hash
    results_json  TEXT NOT NULL       -- aggregates(per strategy) + per-question
);

CREATE INDEX IF NOT EXISTS answer_runs_started_at ON answer_runs(started_at);
CREATE INDEX IF NOT EXISTS answer_runs_dataset    ON answer_runs(dataset_hash);
