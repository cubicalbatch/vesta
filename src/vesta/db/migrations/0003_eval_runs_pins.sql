-- Migration 0003 — extend eval_runs with the five comparison pins.
--
-- 0001 shipped eval_runs(id, started_at, config_json, metrics_json). Runs must
-- be *comparable*: a run is meaningless without its profile
-- hash, settings snapshot, archive checksum, git SHA, and machine id pinned
-- alongside it. Rather than fragment
-- those across columns, they live inside the existing config_json blob (the
-- runner serializes them there); the columns below are the *indexed* facets the
-- runner queries by (latest run for a profile, runs on an archive) so a list of
-- runs does not require parsing JSON per row.
--
-- All additions are nullable + default NULL so existing rows (the empty set at
-- this point) and any earlier rows stay valid.

ALTER TABLE eval_runs ADD COLUMN profile_name TEXT;
ALTER TABLE eval_runs ADD COLUMN profile_hash TEXT;
ALTER TABLE eval_runs ADD COLUMN golden_hash TEXT;
ALTER TABLE eval_runs ADD COLUMN archive_checksum TEXT;
ALTER TABLE eval_runs ADD COLUMN git_sha TEXT;
ALTER TABLE eval_runs ADD COLUMN machine_id TEXT;
ALTER TABLE eval_runs ADD COLUMN finished_at TEXT;

CREATE INDEX IF NOT EXISTS eval_runs_profile_hash ON eval_runs(profile_hash);
CREATE INDEX IF NOT EXISTS eval_runs_started_at ON eval_runs(started_at);
