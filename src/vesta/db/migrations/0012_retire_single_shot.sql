-- Retire the ``single_shot`` answer strategy.
--
-- The ``single_shot`` strategy (a single grounded round) is deleted;
-- the LLM answer path is the streaming pydantic-ai agent behind POST /api/chat,
-- and ``GET /api/answer`` now runs ``sources_only``. ``single_shot`` is no
-- longer a valid ``answer.strategy`` choice, so a stored value of
-- ``single_shot`` (or a residual ``agentic`` from before migration 0011) would
-- fail the resolver's ``choices`` validation on the next settings snapshot.
-- Reset it to the sole remaining choice, ``sources_only``.
--
-- The seven ``answer.*`` knobs below configured only ``single_shot``'s pipeline
-- (prompt version, output-token budget, abstention pre-gate floors, post-hoc
-- citation span score, LLM-context floor, prompt logging). With the strategy
-- gone they have no reader; drop any stored rows so the settings table holds
-- only live keys. ``answer.strategy`` itself is kept (valid choice: sources_only).

UPDATE settings SET value = 'sources_only'
 WHERE key = 'answer.strategy' AND value IN ('single_shot', 'agentic');

DELETE FROM settings WHERE key IN (
    'answer.prompt_version',
    'answer.max_output_tokens',
    'answer.abstention.density_floor',
    'answer.abstention.top_score_floor',
    'answer.citations.min_span_score',
    'answer.context.min_score',
    'answer.log_prompts'
);
