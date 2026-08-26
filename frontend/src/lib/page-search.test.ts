// Stale-response races on the sources-search surfaces (AUDIT_0824 F3).
// SearchPage.svelte and ArchiveBrowsePage.svelte both ran the identical
// unguarded fetch-then-assign: submit query A (slow), edit, submit query B
// (fast) — when A's response landed after B's it clobbered result/lastQuery
// with passages for a query the user had already moved past.
import { describe, expect, it, vi, afterEach } from 'vitest';
import { PageSearch } from './page-search.svelte';
import type { SearchResponse } from '$lib/api/search';
// The fix is PageSearch's monotonic sequence token; these tests drive the real
// class with a deferred fetch mock so resolutions can be ordered at will.
function searchResponse(): SearchResponse {
	return {
		cards: [],
		trace: {} as SearchResponse['trace'],
		confidence: { top_score: null, score_dropoff: null, density: 0, agreement: 0 },
		profile: null,
		profile_hash: null
	};
}

/** Which query a resolved payload belongs to — threaded through the URL the
 *  api client builds (`/api/search?q=<tag>`). */
function payloadFor(tag: string): Response {
	return {
		ok: true,
		status: 200,
		json: async () => ({ ...searchResponse(), marker: tag })
	} as unknown as Response;
}

interface Pending {
	url: string;
	resolve: (r: Response) => void;
	reject: (e: unknown) => void;
}

/** Deferred fetch: calls queue here in start order; tests settle them out of
 *  order to simulate a slow earlier request racing a newer one. */
function stubDeferredFetch(): Pending[] {
	const pending: Pending[] = [];
	vi.stubGlobal(
		'fetch',
		vi.fn((input: string | URL | Request) => {
			return new Promise<Response>((resolve, reject) => {
				pending.push({ url: String(input), resolve, reject });
			});
		})
	);
	return pending;
}

/** Deterministically run pending microtasks — enough ticks for request()'s
 *  await chain (fetch → json → assignment) to settle (session.test.ts idiom). */
async function drainMicrotasks(): Promise<void> {
	for (let i = 0; i < 50; i++) await Promise.resolve();
}

afterEach(() => vi.unstubAllGlobals());

describe('PageSearch stale-response suppression', () => {
	it('a slower earlier response never clobbers the newer result', async () => {
		const pending = stubDeferredFetch();
		const ps = new PageSearch();

		void ps.run('slow query');
		void ps.run('fast query');

		// Newer request resolves first.
		pending[1].resolve(payloadFor('fast query'));
		await drainMicrotasks();

		expect(ps.loading).toBe(false);
		expect(ps.lastQuery).toBe('fast query');
		expect(ps.result && (ps.result as { marker?: string }).marker).toBe('fast query');

		// The stale response lands afterwards…
		pending[0].resolve(payloadFor('slow query'));
		await drainMicrotasks();

		// …and must be dropped wholesale: result AND lastQuery untouched,
		// loading left exactly as the newer run settled it.
		expect(ps.lastQuery).toBe('fast query');
		expect(ps.result && (ps.result as { marker?: string }).marker).toBe('fast query');
		expect(ps.loading).toBe(false);
	});

	it('a stale failure does not surface after a newer success', async () => {
		const pending = stubDeferredFetch();
		const ps = new PageSearch();

		void ps.run('old');
		void ps.run('new');
		pending[1].resolve(payloadFor('new'));
		await drainMicrotasks();
		expect(ps.error).toBeNull();

		pending[0].reject(new Error('boom'));
		await drainMicrotasks();

		expect(ps.error).toBeNull();
		expect(ps.result && (ps.result as { marker?: string }).marker).toBe('new');
		expect(ps.loading).toBe(false);
	});

	it('a newer failure wins over the older in-flight run', async () => {
		const pending = stubDeferredFetch();
		const ps = new PageSearch();

		void ps.run('first');
		void ps.run('second');
		pending[1].reject(new Error('second failed'));
		await drainMicrotasks();

		expect(ps.error).toBe('second failed');
		expect(ps.result).toBeNull();
		expect(ps.loading).toBe(false);

		// The older response landing late changes nothing.
		pending[0].resolve(payloadFor('first'));
		await drainMicrotasks();
		expect(ps.error).toBe('second failed');
		expect(ps.result).toBeNull();
	});

	it('clear() invalidates an in-flight run without starting a new one', async () => {
		const pending = stubDeferredFetch();
		const ps = new PageSearch();

		void ps.run('abandoned');
		ps.clear();

		expect(ps.loading).toBe(false);

		pending[0].resolve(payloadFor('abandoned'));
		await drainMicrotasks();

		// "New question" / external re-seed blanked the surface; the racing
		// response must not repaint it.
		expect(ps.result).toBeNull();
		expect(ps.lastQuery).toBe('');
		expect(ps.error).toBeNull();
		expect(ps.loading).toBe(false);
	});

	it('a lone run assigns normally and threads the scope param', async () => {
		const pending = stubDeferredFetch();
		const ps = new PageSearch();

		void ps.run('wikipedia', '3');
		expect(ps.loading).toBe(true);

		pending[0].resolve(payloadFor('wikipedia'));
		await drainMicrotasks();

		expect(pending[0].url).toContain('/api/search?');
		expect(pending[0].url).toContain('q=wikipedia');
		expect(pending[0].url).toContain('scope=3');
		expect(ps.loading).toBe(false);
		expect(ps.lastQuery).toBe('wikipedia');
	});
});
