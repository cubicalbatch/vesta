-- Retire the old agentic engine.
--
-- The ``agentic`` answer strategy and its loop/gate/probe/reformulate settings
-- are deleted. Any rows still stored in the ``settings`` table for those keys
-- are stale: the resolver validates every stored value against the declared
-- ``choices``, so a leftover ``answer.strategy='agentic'`` (valid only before
-- this phase) would raise on the next settings snapshot. Drop the removed
-- keys and reset the strategy to its default when it holds the retired value.
--
-- ``answer.strategy`` is kept (its valid choices are now ``single_shot`` /
-- ``sources_only``); only a stored ``agentic`` value is repaired.

DELETE FROM settings WHERE key IN (
    'answer.loop.max_rounds',
    'answer.loop.tier_override',
    'answer.loop.trigger',
    'answer.loop.stagnation_overlap',
    'answer.loop.query_dedup_threshold',
    'answer.loop.read_on_weak_support',
    'answer.loop.gap_search',
    'answer.loop.gap_max_passes',
    'answer.loop.gap_max_queries',
    'answer.loop.pre_search',
    'answer.gate.rho_target',
    'answer.probe.enabled',
    'answer.probe.cache_ttl',
    'answer.probe.reformulation_axis',
    'answer.reformulate.enabled',
    'answer.reformulate.max_queries'
);

UPDATE settings SET value = 'single_shot'
 WHERE key = 'answer.strategy' AND value = 'agentic';