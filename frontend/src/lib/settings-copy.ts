import { humanizeKey } from './settings-groups';
// Copy overrides for the ~10 BASIC keys. The `help` string from the schema is
// the fallback; where an override below has better wording for a Basic key, it
// takes precedence and everything else falls through to `help`.
//
// The wording maps to the real schema keys (the field labels are illustrative
// of intent, not a 1:1 mirror of the live schema — e.g. the schema has
// session-timeout or auto-index settings not covered here, and vice versa).
// Voice: describe what happens to the user's machine, never the pipeline.
//
export interface SettingCopyOverride {
	label?: string;
	help?: string;
}

export const SETTING_COPY: Record<string, SettingCopyOverride> = {
	'inference.llm.source': {
		label: 'Where the model runs',
		help: 'On this machine keeps every question local. Another server sends the question (and the passages found for it) to an OpenAI-compatible endpoint you point at below.'
	},
	'inference.llm.model': {
		label: 'Answer model',
		help: 'Writes the answer from the passages Vesta retrieved. Bigger models write better and run slower.'
	},
	'inference.llm.endpoint_url': {
		label: 'Remote server address',
		help: 'Any OpenAI-compatible /v1/chat/completions endpoint. Only used when the model above runs on another server.'
	},
	'index.default_depth': {
		label: 'Default index depth for new archives',
		help: 'Keyword-only is fast and small. Adding meaning search roughly triples the disk and the time, and is what makes Ask work well.'
	},
	'retrieval.active_profile': {
		label: 'Retrieval profile',
		help: 'How Vesta searches archives for relevant passages. Hybrid combines keyword and meaning search; standard uses keyword search with two-pass ranking; lexical is fast keyword-only search.'
	},
	'inference.local.idle_unload_seconds': {
		label: 'Free model memory when idle',
		help: 'Unloads the answer model after this many quiet seconds and reloads it on the next question. Worth it on a small machine; costs a few seconds on the first question after a break. 0 disables unloading.'
	},
	'logging.level': {
		label: 'Log detail',
		help: 'How much detail Vesta writes to its log file. INFO is right for normal use; DEBUG is verbose and meant for troubleshooting.'
	},
	'inference.local.context_size': {
		label: 'Context window',
		help: 'How much text the model can hold at once while answering. Bigger windows cost more RAM — Settings → AI shows a live estimate.'
	},
	'inference.local.preload_on_ready': {
		label: 'Keep the model warm after download',
		help: 'Loads the model into memory the moment its download finishes, so the first question is fast. Turn off to save RAM on a small machine.'
	},
};

export function settingLabel(key: string, fallback: string): string {
	return SETTING_COPY[key]?.label ?? fallback;
}

export function settingHelp(key: string, fallback: string): string {
	return SETTING_COPY[key]?.help ?? fallback;
}


// The post-save message: settings changes rebuild the runtime in-process.
// `inference.*` keys that the schema still marks `hot: false` DO restart
// something — the supervised llama-server child — but the user never restarts
// anything, so they must report "applies on your next question", never "needs
// a restart". Only genuinely cold keys (none today, but the function must stay
// honest if one appears) keep the restart wording.
//
// `schemaByKey` is the live schema lookup the Settings page already builds;
// unknown keys (applied but absent from the schema) are treated as hot.
export function buildSaveMessage(
	appliedKeys: string[],
	schemaByKey: Record<string, { hot?: boolean }>
): string {
	const nonHot = appliedKeys.filter((k) => schemaByKey[k]?.hot === false);
	const restart = nonHot.filter((k) => !k.startsWith('inference.'));
	if (restart.length === 0) return 'Applies to your next question — no restart needed.';
	const names = restart.map((k) => settingLabel(k, humanizeKey(k))).join(', ');
	return `Applies to the next question — no restart needed, except ${names}, which need a restart to take effect.`;
}
