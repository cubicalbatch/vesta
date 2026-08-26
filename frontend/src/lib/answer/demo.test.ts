// Repeated "Ask a test question" clicks used to seed one persisted demo
// conversation per click (AUDIT_0824 N16): every POST /api/chat without a
// conversation_id creates one server-side. DemoAsk remembers the id the first
// stream returns (X-Conversation-Id) and continues THAT conversation on later
// clicks — at most one demo conversation per wizard visit.
//
// Component testing isn't set up in this repo (see client.test.ts), so the
// page's ask flow is exercised here against a real DemoAsk + streamChat.
import { describe, expect, it, vi, afterEach } from 'vitest';
import { DemoAsk } from './demo.svelte';

const DONE = 'event: done\ndata: {}\n\n';
const enc = new TextEncoder();

/** Response-like object whose body emits `done` and closes. */
function finiteSseResponse(conversationId?: string): Response {
	const body = new ReadableStream<Uint8Array>({
		start(controller) {
			controller.enqueue(enc.encode(DONE));
			controller.close();
		}
	});
	return {
		ok: true,
		status: 200,
		headers: new Headers(conversationId ? { 'X-Conversation-Id': conversationId } : {}),
		body
	} as unknown as Response;
}

/** Response-like object whose body stays open forever — like a live SSE
 *  answer — unless `signal` fires, which errors the stream the way a browser
 *  cancels a fetch: pending read() rejects with AbortError (client.test.ts
 *  idiom). */
function hangingSseResponse(signal?: AbortSignal | null): Response {
	const body = new ReadableStream<Uint8Array>({
		start(controller) {
			signal?.addEventListener('abort', () =>
				controller.error(new DOMException('aborted', 'AbortError'))
			);
		}
	});
	return {
		ok: true,
		status: 200,
		headers: new Headers(),
		body
	} as unknown as Response;
}

/** Deterministically run pending microtasks — enough ticks for consume() and
 *  the conversation-id resolution to settle (session.test.ts idiom). */
async function drainMicrotasks(): Promise<void> {
	for (let i = 0; i < 50; i++) await Promise.resolve();
}

afterEach(() => vi.unstubAllGlobals());

function requestBody(call: readonly unknown[]): Record<string, unknown> {
	const init = call[1] as RequestInit;
	return JSON.parse(String(init.body)) as Record<string, unknown>;
}

describe('DemoAsk reuses one demo conversation', () => {
	it('first ask posts no conversation_id; every later ask continues it', async () => {
		let n = 0;
		const fetchMock = vi.fn(
			async (_url: string | URL, _init?: RequestInit) =>
				finiteSseResponse(String(1000 + ++n))
		);
		vi.stubGlobal('fetch', fetchMock);
		const demo = new DemoAsk();
		try {
			for (let i = 0; i < 3; i++) {
				await demo.ask();
				await drainMicrotasks();
			}

			expect(fetchMock).toHaveBeenCalledTimes(3);
			const bodies = fetchMock.mock.calls.map(requestBody);
			expect(bodies[0].conversation_id).toBeUndefined();
			expect(bodies[1].conversation_id).toBe(1001);
			expect(bodies[2].conversation_id).toBe(1001);
		} finally {
			demo.dispose();
		}
	});

	it('a click while a stream is already running fires no second request', async () => {
		// Body stays open until disposed, so the first ask never finishes.
		const fetchMock = vi.fn(
			async (_url: string | URL, init?: RequestInit) => hangingSseResponse(init?.signal)
		);
		vi.stubGlobal('fetch', fetchMock);
		const demo = new DemoAsk();
		try {
			const first = demo.ask();
			await drainMicrotasks();
			await demo.ask(); // guarded by `running`
			expect(fetchMock).toHaveBeenCalledTimes(1);
			demo.dispose();
			await first;
		} finally {
			demo.dispose();
		}
	});

	it('dispose() aborts the in-flight request signal', async () => {
		let seenSignal: AbortSignal | null | undefined;
		const fetchMock = vi.fn(async (_url: string | URL, init?: RequestInit) => {
			seenSignal = init?.signal;
			return hangingSseResponse(init?.signal);
		});
		vi.stubGlobal('fetch', fetchMock);
		const demo = new DemoAsk();
		const run = demo.ask();
		await drainMicrotasks();
		demo.dispose();
		await run;
		expect(seenSignal?.aborted).toBe(true);
	});
});
