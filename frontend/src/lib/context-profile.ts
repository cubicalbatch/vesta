// The "Answer speed & memory" composite's brain.
// Pure, DOM-free, so vitest can pin the whole contract without mounting
// anything — same posture as chipView in stores/model.svelte.ts and the
// helpers in lib/bench.ts. The .svelte skin (ContextBudgetControl) stays
// dumb: presets, matching, writes and RAM labels all live here.
//
// One preset writes TWO keys — answer.agent.context_profile (what the answer
// path plans against) and inference.local.context_size (the window the local
// model actually runs) — because setting one without the other breaks the
// answer path: a small window under a fat budget doesn't make the answer
// path leaner, it makes it crash mid-turn into the no-tool fallback. Remote
// endpoints have no local window, so a preset writes only the profile
// there and the UI says why the window half is absent.
import type { LlmStatus } from './api/models';

export const CONTEXT_PROFILE_KEY = 'answer.agent.context_profile';
export const CONTEXT_SIZE_KEY = 'inference.local.context_size';

/** The two keys the composite writes, PROFILE-then-SIZE order. */
export const CONTEXT_KEYS = [CONTEXT_PROFILE_KEY, CONTEXT_SIZE_KEY] as const;

export type ContextPlanId = '8k' | '16k' | 'full';

/** What profile "auto" resolves to at a given window — mirrors the backend
 *  mapping in src/vesta/answer/__init__.py (≤8192 → 8k, ≤16384 → 16k, else
 *  full). Kept in lockstep by test. */
export function autoPlanFor(sizeTokens: number): ContextPlanId {
	if (sizeTokens <= 8192) return '8k';
	if (sizeTokens <= 16384) return '16k';
	return 'full';
}

export type ContextPresetId = 'lean' | 'balanced' | 'thorough';

export interface ContextPreset {
	id: ContextPresetId;
	/** Named for what the user feels, not the token count. */
	label: string;
	profile: ContextPlanId;
	/** inference.local.context_size the preset writes — local runtime only. */
	sizeTokens: number;
}

/** Lightest → heaviest, the same left-to-right convention as
 *  HowThoroughControl's Fast/Balanced/Thorough pills. */
export const CONTEXT_PRESETS: ContextPreset[] = [
	{ id: 'lean', label: 'Lean', profile: '8k', sizeTokens: 8192 },
	{ id: 'balanced', label: 'Balanced', profile: '16k', sizeTokens: 16384 },
	{ id: 'thorough', label: 'Thorough', profile: 'full', sizeTokens: 32768 }
];

const PLAN_IDS: ContextPlanId[] = ['8k', '16k', 'full'];

/** Which preset the live pair reads as, or 'custom' when neither matches —
 *  never mis-highlight a hand-set pair (e.g. context_size=4096 or
 *  profile=8k-fullprompt-wide must read as Custom, not as a near preset).
 *
 *  Local: BOTH keys must agree — the preset's plan (or "auto" resolving to
 *  it) AND the exact window. Remote: the window setting doesn't apply, so
 *  only the profile half is compared, and auto means full (backend D2). */
export function matchContextPreset(
	profileValue: string,
	sizeValue: string,
	isLocal: boolean
): ContextPresetId | 'custom' {
	const p = String(profileValue).trim();
	const n = Number(sizeValue);
	const size = Number.isFinite(n) && n > 0 ? Math.round(n) : -1;
	const plan: ContextPlanId | null =
		p === 'auto'
			? isLocal
				? size > 0
					? autoPlanFor(size)
					: null
				: 'full'
			: PLAN_IDS.includes(p as ContextPlanId)
				? (p as ContextPlanId)
				: null;
	if (plan === null) return 'custom';
	for (const preset of CONTEXT_PRESETS) {
		if (preset.profile !== plan) continue;
		if (!isLocal || size === preset.sizeTokens) return preset.id;
	}
	return 'custom';
}

/** The settings a preset click writes. Local: BOTH keys, never one without
 *  the other. Remote: the profile only — the window is the remote
 *  server's, not a local knob. The caller saves them in one PUT, so the pair
 *  lands atomically. */
export function contextPresetWrites(
	preset: ContextPreset,
	isLocal: boolean
): Record<string, string> {
	return isLocal
		? {
				[CONTEXT_PROFILE_KEY]: preset.profile,
				[CONTEXT_SIZE_KEY]: String(preset.sizeTokens)
			}
		: { [CONTEXT_PROFILE_KEY]: preset.profile };
}

// ── RAM labels (mirror of the estimator) ──────────────────
// estimate_ram_bytes = weights + ctx × kv/token; only the KV term moves with
// the preset, so that is what the pills show. The rate is back-derived from
// the live status the same way AiSection does it; null when those inputs are
// unavailable — callers degrade, never guess.

/** The live KV bytes/token, or null when the status can't back-derive one
 *  (no model / no estimate). Never falls back to a guessed constant here:
 *  a wrong rate would mislabel every option's memory. */
export function kvBytesPerToken(status: LlmStatus | null): number | null {
	if (!status) return null;
	if (status.estimated_ram_bytes > 0 && status.size_bytes > 0 && status.context_size > 0) {
		return Math.max((status.estimated_ram_bytes - status.size_bytes) / status.context_size, 1);
	}
	return null;
}

/** KV-cache bytes a window costs, or null without a usable rate. */
export function kvCacheBytes(sizeTokens: number, kvRate: number | null): number | null {
	if (kvRate == null || !(kvRate > 0) || !(sizeTokens > 0)) return null;
	return Math.round(sizeTokens * kvRate);
}

/** Signed KV delta (bytes) a draft window has vs the running one — positive
 *  means the draft SAVES that much memory. Null when not computable. */
export function kvDeltaBytes(
	draftSizeTokens: number,
	liveSizeTokens: number | null,
	kvRate: number | null
): number | null {
	if (kvRate == null || liveSizeTokens == null || !(liveSizeTokens > 0)) return null;
	if (!(draftSizeTokens > 0) || draftSizeTokens === liveSizeTokens) return null;
	return Math.round((liveSizeTokens - draftSizeTokens) * kvRate);
}

// ── Copy (exported so the wording is pinnable by test) ──────────────────────

/** inference.local.context_size is hot:false in the settings schema, so the
 *  control must not promise the window lands on the next question. What a
 *  change actually does is restart the local model — Vesta does that itself,
 *  in-process ("a hot context-window change rebinds a fresh
 *  runtime without a container restart"), so the user restarts nothing. */
export const CONTEXT_WINDOW_RESTART_COPY =
	'The budget applies to your next question; the window takes effect after a restart of the local model — Vesta restarts it for you.';

/** Remote endpoints have no local window — say why the size half is
 *  gone instead of silently dropping it. */
export const CONTEXT_REMOTE_COPY =
	'Answers come from another server, so the context window is set there — only the budget profile applies here.';

/** Graceful degrade when the estimator's inputs aren't available yet. */
export const CONTEXT_RAM_UNKNOWN_COPY = 'Memory per option appears once a model is installed.';
