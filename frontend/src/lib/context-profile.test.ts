// The "Answer speed & memory" composite's contract,
// pinned against the pure module — same posture as settings-copy.test.ts and
// stores/model.test.ts: no DOM, no component mount (the suite has no DOM
// environment; the .svelte skin only wires these functions to the draft).
//
// What must never regress:
//   • a preset writes BOTH keys locally — one without the other is the plan
//     bug the composite exists to prevent;
//   • remote presets write the profile only (remote has no context_size)
//     and never mis-highlight from the stored, inapplicable window;
//   • a hand-set pair reads as Custom, never as a near preset;
//   • the window half's copy says it applies after a restart (hot:false in
//     the schema), and the RAM labels come from the live estimator inputs.
import { describe, expect, it } from 'vitest';
import type { LlmStatus } from './api/models';
import {
	CONTEXT_KEYS,
	CONTEXT_PRESETS,
	CONTEXT_PROFILE_KEY,
	CONTEXT_REMOTE_COPY,
	CONTEXT_SIZE_KEY,
	CONTEXT_WINDOW_RESTART_COPY,
	autoPlanFor,
	contextPresetWrites,
	kvBytesPerToken,
	kvCacheBytes,
	kvDeltaBytes,
	matchContextPreset
} from './context-profile';

function status(overrides: Partial<LlmStatus>): LlmStatus {
	return {
		source: 'local',
		configured: true,
		installed: true,
		state: 'loaded',
		model_file: 'qwen.gguf',
		display_name: 'Qwen3.5-4B',
		model_id: 'qwen3.5-4b@q4_k_s',
		size_bytes: 2_600_000_000,
		context_size: 32768,
		thinking: false,
		thinking_supported: true,
		idle_unload_seconds: 900,
		seconds_since_last_use: 10,
		estimated_ram_bytes: 3_170_425_344, // size + 32768×17408 — the Qwen3.5-4B rate
		error: null,
		...overrides
	};
}

describe('preset table', () => {
	it('three presets, lightest → heaviest, mapping to full/32k, 16k/16k, 8k/8k', () => {
		expect(CONTEXT_PRESETS.map((p) => p.id)).toEqual(['lean', 'balanced', 'thorough']);
		expect(CONTEXT_PRESETS.map((p) => [p.profile, p.sizeTokens])).toEqual([
			['8k', 8192],
			['16k', 16384],
			['full', 32768]
		]);
		expect(new Set(CONTEXT_PRESETS.map((p) => p.label)).size).toBe(3);
	});

	it('CONTEXT_KEYS covers exactly the two settings the composite writes', () => {
		expect([...CONTEXT_KEYS]).toEqual([CONTEXT_PROFILE_KEY, CONTEXT_SIZE_KEY]);
	});
});

describe('round-trip: preset writes ↔ preset match', () => {
	it('local: each preset writes BOTH keys and re-reads as itself', () => {
		for (const preset of CONTEXT_PRESETS) {
			const writes = contextPresetWrites(preset, true);
			expect(writes).toEqual({
				[CONTEXT_PROFILE_KEY]: preset.profile,
				[CONTEXT_SIZE_KEY]: String(preset.sizeTokens)
			});
			expect(matchContextPreset(writes[CONTEXT_PROFILE_KEY], writes[CONTEXT_SIZE_KEY], true)).toBe(
				preset.id
			);
		}
	});

	it('remote: presets write the profile only and still re-read as themselves', () => {
		for (const preset of CONTEXT_PRESETS) {
			const writes = contextPresetWrites(preset, false);
			expect(Object.keys(writes)).toEqual([CONTEXT_PROFILE_KEY]);
			// The stored window is inapplicable remotely — matching must ignore it
			// (even a stale 4096 must not read as Custom).
			expect(matchContextPreset(writes[CONTEXT_PROFILE_KEY], '4096', false)).toBe(preset.id);
		}
	});
});

describe('current-state detection', () => {
	it('the factory default (auto + 8192) reads as Lean, not Custom', () => {
		expect(matchContextPreset('auto', '8192', true)).toBe('lean');
	});

	it('auto follows the window onto the matching plan', () => {
		expect(matchContextPreset('auto', '32768', true)).toBe('thorough');
		expect(matchContextPreset('auto', '16384', true)).toBe('balanced');
		expect(matchContextPreset('auto', '8192', true)).toBe('lean');
	});

	it('a hand-set window outside the presets reads as Custom', () => {
		expect(matchContextPreset('8k', '4096', true)).toBe('custom');
		expect(matchContextPreset('auto', '4096', true)).toBe('custom');
	});

	it('force-only bench profiles never match a preset', () => {
		expect(matchContextPreset('8k-fullprompt-wide', '8192', true)).toBe('custom');
		expect(matchContextPreset('8k-fullprompt', '8192', false)).toBe('custom');
	});

	it('a crossed pair (plan ≠ window) reads as Custom', () => {
		expect(matchContextPreset('full', '8192', true)).toBe('custom');
		expect(matchContextPreset('8k', '32768', true)).toBe('custom');
	});

	it('an unreadable window reads as Custom locally, and auto means full remotely', () => {
		expect(matchContextPreset('16k', '', true)).toBe('custom');
		expect(matchContextPreset('auto', '', false)).toBe('thorough');
	});
});

describe('autoPlanFor mirrors the backend mapping', () => {
	it('≤8192 → 8k, ≤16384 → 16k, else full', () => {
		expect(autoPlanFor(2048)).toBe('8k');
		expect(autoPlanFor(8192)).toBe('8k');
		expect(autoPlanFor(8193)).toBe('16k');
		expect(autoPlanFor(16384)).toBe('16k');
		expect(autoPlanFor(16385)).toBe('full');
		expect(autoPlanFor(131072)).toBe('full');
	});
});

describe('RAM labels', () => {
	it('back-derives the KV rate from the live status the way the estimator does', () => {
		// Qwen3.5-4B: 3 170 425 344 − 2 600 000 000 = 570 425 344 KV bytes
		// over 32 768 tokens = 17 408 B/token.
		expect(kvBytesPerToken(status({}))).toBeCloseTo(17408, 5);
	});

	it('computes the concrete KV MBs per option (570 MB @32k → 285 MB @16k → 143 MB @8k)', () => {
		const kv = 17408;
		expect(kvCacheBytes(32768, kv)).toBe(570_425_344);
		expect(kvCacheBytes(16384, kv)).toBe(285_212_672);
		expect(kvCacheBytes(8192, kv)).toBe(142_606_336);
	});

	it('the delta vs the running window is signed and null when not computable', () => {
		expect(kvDeltaBytes(8192, 32768, 17408)).toBe(427_819_008); // saves ~427 MB
		expect(kvDeltaBytes(32768, 8192, 17408)).toBe(-427_819_008); // costs ~427 MB
		expect(kvDeltaBytes(8192, 8192, 17408)).toBeNull(); // unchanged
		expect(kvDeltaBytes(8192, null, 17408)).toBeNull();
		expect(kvDeltaBytes(8192, 32768, null)).toBeNull();
	});

	it('degrades to null instead of guessing when the inputs are missing', () => {
		expect(kvBytesPerToken(null)).toBeNull();
		expect(
			kvBytesPerToken(status({ estimated_ram_bytes: 0, size_bytes: 0, context_size: 0 }))
		).toBeNull();
		expect(kvCacheBytes(8192, null)).toBeNull();
		expect(kvCacheBytes(0, 17408)).toBeNull();
	});
});

describe('copy', () => {
	it('the local note says the window half applies after a restart (hot:false key)', () => {
		expect(CONTEXT_WINDOW_RESTART_COPY).toContain('after a restart');
		expect(CONTEXT_WINDOW_RESTART_COPY).toContain('next question');
	});

	it('the remote note explains why the window half is absent', () => {
		expect(CONTEXT_REMOTE_COPY).toContain('context window');
		expect(CONTEXT_REMOTE_COPY).toContain('another server');
	});
});
