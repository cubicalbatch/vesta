// Abort-signal threading through streamChat (AUDIT_0822 P6). The welcome
// wizard's demo answer stream must die with the page: the controller created
// per run has to reach fetch, aborting mid-stream has to stop the reader loop
// cleanly (AbortError = clean stop, never an unhandled rejection or a
// synthetic transport error), and the page's teardown effect has to abort it.
// Component testing isn't set up in this repo (node-env vitest, no
// @testing-library/svelte, and routes/+page.svelte pulls the whole SvelteKit
// runtime), so the last point is covered by replicating the page's exact
// teardown wiring ($effect root + cleanup abort - the AnswerSession idiom)
// against the real client here.
import { describe, it, expect, vi, afterEach } from 'vitest';
import { streamChat } from './client';
import { AnswerSession } from './session.svelte';
import type { AnswerEvent } from '../types';

function abortError(): Error {
	return new DOMException('This operation was aborted.', 'AbortError');
}

const DETAIL_EVENT = 'event: detail\ndata: {"detail":"Reading your archives"}\n\n';
const DONE_EVENT = 'event: done\ndata: {}\n\n';
const enc = new TextEncoder();

/**
 * Response-like object whose body emits `chunks`, then stays open forever -
 * like a live SSE answer - unless `signal` fires, which errors the stream
 * exactly the way a browser cancels a fetch: the pending read() rejects with
 * AbortError.
 */
type FetchArgs = [input: string | URL | Request, init?: RequestInit];

function hangingSseResponse(chunks: string[], signal?: AbortSignal | null): Response {
	const bytes = chunks.map((c) => enc.encode(c));
	const body = new ReadableStream<Uint8Array>({
		start(controller) {
			for (const b of bytes) controller.enqueue(b);
			signal?.addEventListener('abort', () => controller.error(abortError()));
		}
	});
	return sseResponseLike(body);
}

/** Response-like object whose body emits `chunks` and closes. */
function finiteSseResponse(chunks: string[]): Response {
	const bytes = chunks.map((c) => enc.encode(c));
	const body = new ReadableStream<Uint8Array>({
		start(controller) {
			for (const b of bytes) controller.enqueue(b);
			controller.close();
		}
	});
	return sseResponseLike(body);
}

function sseResponseLike(body: ReadableStream<Uint8Array>): Response {
	return {
		ok: true,
		status: 200,
		headers: new Headers({ 'X-Conversation-Id': '7' }),
		body
	} as unknown as Response;
}

async function collect(events: AsyncGenerator<AnswerEvent>): Promise<AnswerEvent[]> {
	const out: AnswerEvent[] = [];
	for await (const e of events) out.push(e);
	return out;
}

function wait(ms: number): Promise<void> {
	return new Promise((r) => setTimeout(r, ms));
}

afterEach(() => vi.unstubAllGlobals());

describe('streamChat signal threading', () => {
	it('forwards the caller AbortSignal identity to fetch', async () => {
		const controller = new AbortController();
		const fetchMock = vi.fn(async (..._: FetchArgs) => finiteSseResponse([DONE_EVENT]));
		vi.stubGlobal('fetch', fetchMock);

		await collect(streamChat({ query: 'q' }, controller.signal).events);

		expect(fetchMock).toHaveBeenCalledOnce();
		expect(fetchMock.mock.calls[0][1]?.signal).toBe(controller.signal);
	});

	it('leaves signal undefined when called without one (default behavior unchanged)', async () => {
		const fetchMock = vi.fn(async (..._: FetchArgs) => finiteSseResponse([DONE_EVENT]));
		vi.stubGlobal('fetch', fetchMock);

		await collect(streamChat({ query: 'q' }).events);

		expect(fetchMock.mock.calls[0][1]?.signal).toBeUndefined();
	});

	it('stops cleanly when aborted before the response arrives: no events, no rejection', async () => {
		const controller = new AbortController();
		controller.abort(); // navigate away before headers ever come back
		const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
			if (init?.signal?.aborted) throw abortError(); // what a real fetch does
			return finiteSseResponse([DONE_EVENT]);
		});
		vi.stubGlobal('fetch', fetchMock);

		const chat = streamChat({ query: 'q' }, controller.signal);
		const [events, id] = await Promise.all([collect(chat.events), chat.conversationId]);

		expect(events).toEqual([]);
		expect(id).toBeNull();
	});

	it('stops cleanly when aborted mid-stream: no synthetic error event, conversationId settled', async () => {
		const controller = new AbortController();
		const fetchMock = vi.fn(async (_url: string, init?: RequestInit) =>
			hangingSseResponse([DETAIL_EVENT], init?.signal)
		);
		vi.stubGlobal('fetch', fetchMock);

		const chat = streamChat({ query: 'q' }, controller.signal);
		const events: AnswerEvent[] = [];
		const consumed = (async () => {
			for await (const e of chat.events) events.push(e);
		})();
		await wait(10); // let the first event land; the reader then blocks

		controller.abort(); // "navigation away" mid-answer
		await consumed; // must end on its own - an unhandled rejection fails the suite
		expect(await chat.conversationId).toBe('7');

		expect(events).toHaveLength(1);
		expect(events[0].event).toBe('detail');
	});
});

describe('welcome demo-stream teardown wiring', () => {
	// routes/welcome/+page.svelte cannot be mounted here: this repo has no
	// component-test setup (node-env vitest, no @testing-library/svelte) and the
	// page imports the SvelteKit runtime ($app/navigation et al.). So pin the
	// controller-lifecycle idiom the page mirrors line-for-line - per-run
	// AbortController handed to streamChat, torn down by a $effect cleanup that
	// calls abort() - via AnswerSession, the existing production class built on
	// exactly that wiring, driven here against the real streamChat client.
	it('teardown aborts the in-flight stream: fetch saw the aborted signal, loop ends cleanly', async () => {
		const signalsSeenByFetch: AbortSignal[] = [];
		const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
			signalsSeenByFetch.push(init?.signal as AbortSignal);			return hangingSseResponse([DETAIL_EVENT], init?.signal);
		});
		vi.stubGlobal('fetch', fetchMock);

		const session = new AnswerSession();
		session.startTurn('In one short sentence, what is Wikipedia?');
		await wait(10);

		session.dispose(); // what navigating away mid-demo triggers

		expect(signalsSeenByFetch[0]?.aborted).toBe(true); // fetch was holding the aborted signal

		// The consumer loop AnswerStream.consume runs must finish on its own -
		// an unhandled rejection would fail this suite - and record no failure.
		await wait(20);
		expect(session.liveTurn?.stream.state.error ?? null).toBeNull();
		expect(signalsSeenByFetch[0]?.aborted).toBe(true);
	});
});
