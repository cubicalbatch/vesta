// The Basic/All split
// that doc was retired in ffed058, so this file is the list's verbatim record
// — do not retype it from memory). A
// key here absent from the live schema is skipped silently by the page; a
// schema key not here renders under "All settings" with zero frontend work.
// `settings-basic.test.ts` asserts this array against the *live*
// `GET /api/settings/schema`, not a hardcoded copy of the schema, so drift
// between this file and the real backend is caught.
export const BASIC: string[] = [
	'inference.llm.source',
	'inference.llm.model',
	'inference.llm.endpoint_url',
	'inference.llm.api_key',
	'index.default_depth',
	'retrieval.active_profile',
	'inference.local.context_size',
	'inference.local.idle_unload_seconds',
	'inference.local.preload_on_ready',
	'logging.level'
];

// The two keys the "How thorough" composite writes. Neither is in BASIC — the
// composite control, not a raw field, is their Basic-view representation. They
// still render individually under "All settings" (never special-cased out of
// the generated form) and inline under the composite once it's in Custom mode.
export const THOROUGH_KEYS = [
	'retrieval.stage_b.shortlist',
	'retrieval.context.max_per_article'
] as const;

// The two keys the "Answer speed & memory" composite writes
// context_profile follows the THOROUGH_KEYS rule — the composite,
// not a raw field, is its Basic-view representation, and it still renders
// under "All settings". context_size is ALSO in BASIC above (it predates the
// composite and stays a raw field); both are schema-asserted in
// settings-basic.test.ts like THOROUGH_KEYS.
export const CONTEXT_KEYS = [
	'answer.agent.context_profile',
	'inference.local.context_size'
] as const;
