// Non-2xx responses used to reduce to a bare "HTTP 404" synthetic error,
// discarding the body's reason (AUDIT_0824 N15): deleting the currently-open
// conversation made every follow-up fail with an opaque "HTTP 404" while the
// server had said exactly why — `{"detail": "conversation 17 not found"}`.
// These pin describeHttpError's contract: the body's reason is folded into
// transportErrorEvent's message.
import { describe, expect, it, vi, afterEach } from 'vitest';
import { fetchEventStream } from './sse';

function jsonResponse(status: number, body: string): Response {
	return {
		ok: status >= 200 && status < 300,
		status,
		headers: new Headers(),
		body: new ReadableStream<Uint8Array>(),
		text: async () => body
	} as unknown as Response;
}

/** A 200 response whose body streams the given chunks then closes cleanly —
 *  the silent-EOF shape: the server never sends `done` or `error`. */
function sseResponse(chunks: string[]): Response {
	const encoder = new TextEncoder();
	let i = 0;
	return {
		ok: true,
		status: 200,
		headers: new Headers(),
		body: new ReadableStream<Uint8Array>({
			pull(controller) {
				if (i < chunks.length) controller.enqueue(encoder.encode(chunks[i++]));
				else controller.close();
			}
		})
	} as unknown as Response;
}

async function collectAll(response: Response): Promise<Array<{ event: string; data: unknown }>> {
	vi.stubGlobal('fetch', vi.fn(async () => response));
	const events: Array<{ event: string; data: unknown }> = [];
	for await (const wire of fetchEventStream('/api/chat', { method: 'POST', body: {} })) {
		events.push({ event: wire.event, data: JSON.parse(wire.data) });
	}
	return events;
}

async function collectError(response: Response): Promise<{ code: string; message: string }> {
	vi.stubGlobal('fetch', vi.fn(async () => response));
	let last: { code: string; message: string } | null = null;
	for await (const wire of fetchEventStream('/api/chat', { method: 'POST', body: {} })) {
		last = JSON.parse(wire.data);
	}
	if (!last) throw new Error('no synthetic error event yielded');
	return last;
}

afterEach(() => vi.unstubAllGlobals());

describe('fetchEventStream non-2xx reason propagation', () => {
	it('folds a FastAPI detail body into the error message', async () => {
		const err = await collectError(jsonResponse(404, '{"detail":"conversation 17 not found"}'));
		expect(err.code).toBe('stream_error');
		expect(err.message).toBe('HTTP 404: conversation 17 not found');
	});

	it('surfaces a plain-text body trimmed and raw', async () => {
		const err = await collectError(jsonResponse(500, '  boom \n'));
		expect(err.message).toBe('HTTP 500: boom');
	});

	it('falls back to the bare status on an empty body', async () => {
		const err = await collectError(jsonResponse(503, ''));
		expect(err.message).toBe('HTTP 503');
	});
});

describe('fetchEventStream silent EOF (AUDIT_0824 F1)', () => {
	it('emits no synthetic error when the stream ends with done', async () => {
		const events = await collectAll(
			sseResponse([
				'event: token\ndata: {"text":"hi"}\n\n',
				'event: done\ndata: {}\n\n'
			])
		);
		expect(events.map((e) => e.event)).toEqual(['token', 'done']);
	});

	it('ends a stream that stops without done with a transport error carrying residual text', async () => {
		// The last frame is unterminated (no blank line) — the parser must
		// dispatch it at EOF instead of discarding it, and the premature close
		// must surface as the one synthetic transport error.
		const events = await collectAll(
			sseResponse([
				'event: token\ndata: {"text":"The Battle o"}\n\n',
				'data: {"text":"f Hastings"}'
			])
		);
		// the residual frame carries no `event:` line, so it surfaces under the
		// SSE default event name — but its text is not lost
		expect(events.map((e) => e.event)).toEqual(['token', 'message', 'error']);
		expect(events[1].data).toEqual({ text: 'f Hastings' });
		const err = events[2].data as { code: string; message: string; recoverable: boolean };
		expect(err.code).toBe('stream_error');
		expect(typeof err.message).toBe('string');
		expect(err.message.length).toBeGreaterThan(0);
		expect(err.recoverable).toBe(true);
	});

	it('reports a transport error for a body that closes with no events at all', async () => {
		const events = await collectAll(sseResponse([]));
		expect(events).toHaveLength(1);
		expect(events[0].event).toBe('error');
		expect((events[0].data as { code: string }).code).toBe('stream_error');
	});
});
