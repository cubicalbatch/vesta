import { afterEach, describe, expect, it, vi } from 'vitest';
import { JobsStore } from './jobs.svelte';

async function drainMicrotasks() {
	for (let i = 0; i < 50; i++) await Promise.resolve();
}

function openResponse(): Response {
	return new Response(new ReadableStream<Uint8Array>({
		start(controller) {
			controller.enqueue(new TextEncoder().encode('event: snapshot\ndata: {"id":1,"status":"running"}\n\n'));
		}
	}));
}

afterEach(() => {
	vi.unstubAllGlobals();
	vi.useRealTimers();
});

describe('JobsStore connection lifecycle', () => {
	it('backs off consecutive transport errors instead of treating them as successful connections', async () => {
		vi.useFakeTimers();
		const fetch = vi.fn(async () => new Response('unavailable', { status: 503 }));
		vi.stubGlobal('fetch', fetch);
		const store = new JobsStore();
		try {
			store.start();
			await drainMicrotasks();
			expect(fetch).toHaveBeenCalledTimes(1);
			expect(store.connected).toBe(false);
			await vi.advanceTimersByTimeAsync(1000);
			expect(fetch).toHaveBeenCalledTimes(2);
			await vi.advanceTimersByTimeAsync(1000);
			expect(fetch).toHaveBeenCalledTimes(2);
			await vi.advanceTimersByTimeAsync(1000);
			expect(fetch).toHaveBeenCalledTimes(3);
		} finally {
			store.stop();
		}
	});

	it('cancels the old reconnect timer before starting a fresh connection', async () => {
		vi.useFakeTimers();
		const fetch = vi.fn()
			.mockResolvedValueOnce(new Response('unavailable', { status: 503 }))
			.mockImplementation(async () => openResponse());
		vi.stubGlobal('fetch', fetch);
		const store = new JobsStore();
		try {
			store.start();
			await drainMicrotasks();
			store.stop();
			store.start();
			await drainMicrotasks();
			expect(store.connected).toBe(true);
			await vi.advanceTimersByTimeAsync(1000);
			expect(fetch).toHaveBeenCalledTimes(2);
			store.stop();
			expect(store.connected).toBe(false);
		} finally {
			store.stop();
		}
	});

	it('ignores an old connection that resolves after stop/start', async () => {
		vi.useFakeTimers();
		const older = Promise.withResolvers<Response>();
		const fetch = vi.fn().mockReturnValueOnce(older.promise).mockImplementation(async () => openResponse());
		vi.stubGlobal('fetch', fetch);
		const store = new JobsStore();
		try {
			store.start();
			store.stop();
			store.start();
			await drainMicrotasks();
			expect(store.connected).toBe(true);
			older.resolve(new Response('unavailable', { status: 503 }));
			await drainMicrotasks();
			expect(store.connected).toBe(true);
			await vi.advanceTimersByTimeAsync(15000);
			expect(fetch).toHaveBeenCalledTimes(2);
		} finally {
			store.stop();
		}
	});
});
