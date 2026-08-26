// Stale-response race in the Reader's /api/article load (AUDIT_0824 F3).
// Reader.svelte fetches the outline+title for `currentPath` inside an $effect;
// next/prev card and the iframe navigation poll rerun that effect while the
// previous request may still be in flight — the slow old response used to
// clobber article/articleError for the entry now showing. The fix is the
// cleanup-cancelled flag (the same idiom as Reader's pdf.js effect).
//
// Component testing isn't set up in this repo (node-env vitest, no
// @testing-library/svelte — see client.test.ts), so this replicates the
// effect body line-for-line, with cleanup() standing in for Svelte's
// rerun-cleans-up-first semantics.
import { describe, expect, it, vi, afterEach } from 'vitest';
import { api } from '$lib/api/client';
import type { ArticleOut } from '$lib/types';

let article: ArticleOut | null = null;
let articleError: string | null = null;
let cleanup: (() => void) | null = null;

/** Verbatim mirror of Reader.svelte's article-loading $effect. */
function articleEffect(
	target: { zimId: number; path: string } | null,
	currentPath: string | null
): void {
	cleanup?.();
	cleanup = null;
	article = null;
	articleError = null;
	if (!target || !currentPath) return;
	let cancelled = false;
	api
		.get<ArticleOut>(`/api/article/${target.zimId}/${encodeURIComponent(currentPath)}`)
		.then((a) => {
			if (!cancelled) article = a;
		})
		.catch((err) => {
			if (!cancelled) articleError = err instanceof Error ? err.message : 'failed to load article';
		});
	cleanup = () => {
		cancelled = true;
	};
}

interface Pending {
	url: string;
	resolve: (r: Response) => void;
	reject: (e: unknown) => void;
}

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

function payloadFor(title: string): Response {
	return {
		ok: true,
		status: 200,
		json: async () =>
			({ zim_id: 1, path: 'p', title, sections: [], document: null }) as unknown as ArticleOut
	} as unknown as Response;
}

/** Drain request()'s await chain (fetch → json → assignment). */
async function drainMicrotasks(): Promise<void> {
	for (let i = 0; i < 50; i++) await Promise.resolve();
}

afterEach(() => vi.unstubAllGlobals());

describe('Reader article effect stale-response suppression', () => {
	it('a slow response for the old path never clobbers the new article', async () => {
		const pending = stubDeferredFetch();

		articleEffect({ zimId: 1, path: 'A.html' }, 'A.html');
		expect(pending[0].url).toBe('/api/article/1/A.html');

		// Navigate to the next card before A finished loading.
		articleEffect({ zimId: 1, path: 'B.html' }, 'B.html');
		expect(pending[1].url).toBe('/api/article/1/B.html');

		pending[1].resolve(payloadFor('Article B'));
		await drainMicrotasks();
		expect(article?.title).toBe('Article B');

		pending[0].resolve(payloadFor('Article A'));
		await drainMicrotasks();

		expect(article?.title).toBe('Article B');
		expect(articleError).toBeNull();
	});

	it('a stale failure does not surface after a newer success', async () => {
		const pending = stubDeferredFetch();

		articleEffect({ zimId: 1, path: 'old' }, 'old');
		articleEffect({ zimId: 1, path: 'new' }, 'new');
		pending[1].resolve(payloadFor('New'));
		await drainMicrotasks();

		pending[0].reject(new Error('old exploded'));
		await drainMicrotasks();

		expect(article?.title).toBe('New');
		expect(articleError).toBeNull();
	});

	it('a fresh failure still surfaces over the older in-flight run', async () => {
		const pending = stubDeferredFetch();

		articleEffect({ zimId: 1, path: 'one' }, 'one');
		articleEffect({ zimId: 2, path: 'two' }, 'two');
		pending[1].reject(new Error('two failed'));
		await drainMicrotasks();

		expect(article).toBeNull();
		expect(articleError).toBe('two failed');

		pending[0].resolve(payloadFor('One'));
		await drainMicrotasks();
		expect(article).toBeNull();
		expect(articleError).toBe('two failed');
	});

	it('closing the reader (null target) cancels the in-flight load too', async () => {
		const pending = stubDeferredFetch();

		articleEffect({ zimId: 1, path: 'x' }, 'x');
		articleEffect(null, null); // readerStore.close() → target null

		pending[0].resolve(payloadFor('Late'));
		await drainMicrotasks();

		expect(article).toBeNull();
		expect(articleError).toBeNull();
	});
});
