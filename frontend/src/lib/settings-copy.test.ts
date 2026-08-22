// buildSaveMessage: `inference.*` keys that the
// schema still marks `hot: false` rebuild the runtime in-process — the user
// never restarts anything, so they must report "applies on your next
// question", never "needs a restart". These tests pin that contract; a
// regression reads as the UI promising a restart Vesta no longer needs.
import { describe, expect, it } from 'vitest';
import { buildSaveMessage } from './settings-copy';

const schema = (hot: Record<string, boolean>): Record<string, { hot?: boolean }> => {
	const out: Record<string, { hot?: boolean }> = {};
	for (const [k, v] of Object.entries(hot)) out[k] = { hot: v };
	return out;
};

describe('buildSaveMessage', () => {
	it('inference.* non-hot keys never mention a restart', () => {
		const s = schema({
			'inference.local.context_size': false,
			'inference.local.idle_unload_seconds': false,
			'inference.llm.model': true
		});
		const msg = buildSaveMessage(
			['inference.local.context_size', 'inference.local.idle_unload_seconds', 'inference.llm.model'],
			s
		);
		expect(msg).toBe('Applies to your next question — no restart needed.');
		expect(msg).not.toContain('restart to take effect');
	});

	it('a non-inference non-hot key keeps the honest restart wording', () => {
		const s = schema({ 'index.default_depth': false });
		const msg = buildSaveMessage(['index.default_depth'], s);
		expect(msg).toContain('Default index depth for new archives');
		expect(msg).toContain('which need a restart to take effect');
	});

	it('mixed saves exempt only the inference keys from the restart list', () => {
		const s = schema({
			'inference.local.context_size': false,
			'index.default_depth': false
		});
		const msg = buildSaveMessage(['inference.local.context_size', 'index.default_depth'], s);
		expect(msg).toContain('except Default index depth for new archives, which need a restart');
		expect(msg).not.toContain('context');
	});

	it('hot-only saves keep the plain next-question message', () => {
		expect(buildSaveMessage(['inference.llm.model'], schema({ 'inference.llm.model': true }))).toBe(
			'Applies to your next question — no restart needed.'
		);
		expect(buildSaveMessage([], {})).toBe('Applies to your next question — no restart needed.');
	});
});
