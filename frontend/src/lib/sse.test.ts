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
