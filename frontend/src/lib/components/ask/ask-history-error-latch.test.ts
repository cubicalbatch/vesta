// Error latch in the History popover's conversation list (AUDIT_0824 F5).
// AskHistory.svelte fetches /api/conversations inside an $effect each time the
// popover opens; a failed load used to set `error` and nothing ever cleared it,
// so every later open showed "failed to load history" even when the retry
// succeeded. The fix clears `error` at each fetch start.
//
// Component testing isn't set up in this repo (node-env vitest, no
// @testing-library/svelte — see client.test.ts), so this replicates the effect
// body line-for-line, as reader-article-effect.test.ts does for Reader.
import { describe, expect, it, vi, afterEach } from 'vitest';
import { conversationsApi } from '$lib/api/conversations';
import type { ConversationSummary } from '$lib/types';

let items: ConversationSummary[] = [];
let loading = false;
let error: string | null = null;

/** Verbatim mirror of AskHistory.svelte's history-loading $effect body. */
function historyEffect(open: boolean): void {
	if (!open) return;
	loading = true;
	error = null;
	conversationsApi
		.list(50)
		.then((res) => (items = res))
		.catch((err) => (error = err instanceof Error ? err.message : 'failed to load history'))
		.finally(() => (loading = false));
}

async function drainMicrotasks(): Promise<void> {
	for (let i = 0; i < 50; i++) await Promise.resolve();
}

afterEach(() => vi.unstubAllGlobals());

describe('AskHistory load-effect error latch', () => {
	it('clears a prior failure when a reopened popover loads successfully', async () => {
		vi.stubGlobal(
			'fetch',
			vi.fn().mockRejectedValueOnce(new Error('network down')).mockRejectedValueOnce(new TypeError('fetch failed'))
		);

		historyEffect(true);
		await drainMicrotasks();
		expect(loading).toBe(false);
		expect(error).toBe('network down');

		// Reopen while the backend is still down — error persists, correctly.
		historyEffect(true);
		await drainMicrotasks();
		expect(error).toBe('fetch failed');

		// Third open: backend recovered.
		const conv: ConversationSummary = {
			id: 7,
			title: 'what is a ZIM',
			created_at: '2026-08-26T00:00:00Z',
			updated_at: '2026-08-26T00:00:00Z'
		};
		vi.stubGlobal(
			'fetch',
			vi.fn().mockResolvedValue(new Response(JSON.stringify([conv]), { status: 200 }))
		);
		historyEffect(true);
		await drainMicrotasks();
		expect(error).toBeNull();
		expect(items).toEqual([conv]);
	});

	it('shows no stale error when a retry succeeds after a mid-session failure', async () => {
		let calls = 0;
		vi.stubGlobal(
			'fetch',
			vi.fn(() => {
				calls += 1;
				return calls === 1
					? Promise.reject(new Error('boom'))
					: Promise.resolve(new Response(JSON.stringify([]), { status: 200 }));
			})
		);

		historyEffect(true);
		await drainMicrotasks();
		expect(error).toBe('boom');

		historyEffect(true);
		await drainMicrotasks();
		expect(error).toBeNull();
		expect(loading).toBe(false);
	});
});
