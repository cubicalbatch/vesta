import { describe, expect, it } from 'vitest';
import { chipView } from './model.svelte';
import type { LlmStatus } from '$lib/api/models';

function status(overrides: Partial<LlmStatus> = {}): LlmStatus {
	return {
		source: 'local',
		configured: true,
		installed: true,
		state: 'loaded',
		model_file: 'qwen3.5-4b.gguf',
		display_name: 'Qwen3.5 4B',
		model_id: 'qwen3.5-4b',
		size_bytes: 2_600_000_000,
		context_size: 8192,
		thinking: false,
		thinking_supported: true,
		idle_unload_seconds: 900,
		seconds_since_last_use: 180,
		estimated_ram_bytes: 3_100_000_000,
		error: null,
		...overrides
	};
}

describe('chipView — the chip state→label mapping', () => {
	it('before the first fetch renders a neutral placeholder', () => {
		expect(chipView(null)).toEqual({ dot: 'hollow', pulsing: false, label: '…', popover: false });
	});

	it('absent is grey hollow "No AI" and links out, no popover', () => {
		const v = chipView(status({ state: 'absent', configured: false, display_name: null }));
		expect(v.dot).toBe('hollow');
		expect(v.label).toBe('No AI');
		expect(v.popover).toBe(false);
	});

	it('unloaded is grey hollow "<name> · asleep" with a popover', () => {
		const v = chipView(status({ state: 'unloaded' }));
		expect(v.dot).toBe('hollow');
		expect(v.label).toBe('Qwen3.5 4B · asleep');
		expect(v.popover).toBe(true);
	});

	it('sleeping and stopped render like unloaded — weights not in memory', () => {
		for (const state of ['sleeping', 'stopped'] as const) {
			expect(chipView(status({ state })).label).toBe('Qwen3.5 4B · asleep');
		}
	});

	it('loading is amber, pulsing, "Loading <name>…"', () => {
		const v = chipView(status({ state: 'loading' }));
		expect(v.dot).toBe('amber');
		expect(v.pulsing).toBe(true);
		expect(v.label).toBe('Loading Qwen3.5 4B…');
		expect(v.popover).toBe(true);
	});

	it('loaded is green with the display name', () => {
		const v = chipView(status({ state: 'loaded' }));
		expect(v.dot).toBe('green');
		expect(v.pulsing).toBe(false);
		expect(v.label).toBe('Qwen3.5 4B');
		expect(v.popover).toBe(true);
	});

	it('error is red "AI unavailable" regardless of name', () => {
		const v = chipView(status({ state: 'error', error: 'binary missing' }));
		expect(v.dot).toBe('red');
		expect(v.label).toBe('AI unavailable');
		expect(v.popover).toBe(true);
	});

	it('falls back to "Model" when the backend has no display_name yet', () => {
		expect(chipView(status({ state: 'loaded', display_name: null })).label).toBe('Model');
		expect(chipView(status({ state: 'unloaded', display_name: null })).label).toBe('Model · asleep');
		expect(chipView(status({ state: 'loading', display_name: null })).label).toBe('Loading model…');
	});
});
