-- Migration 0010 — per-question answer-LLM token usage.
--
-- Records input/output token counts for the answering model's calls on each
-- question, so the benchmark report can show total tokens for the suite and a
-- P50 per question. Only the answering LLM is tracked — the judge is not.
--
-- Both columns default to 0 so existing rows and no-LLM systems
-- (retrieval_only) stay valid without backfill.

ALTER TABLE bench_question_results ADD COLUMN input_tokens INTEGER NOT NULL DEFAULT 0;
ALTER TABLE bench_question_results ADD COLUMN output_tokens INTEGER NOT NULL DEFAULT 0;
