// One SSE client for the whole app, over `fetch` + `ReadableStream`. Never
// `EventSource` — `POST /api/chat` needs a body and `EventSource` is GET-only,
// and a second transport path is a second set of bugs
// Streaming transport.
//
// Used for: GET /api/answer, POST /api/chat, GET /api/jobs/stream,
// GET /api/jobs/{id}/stream.

export interface WireEvent {
	event: string;
	data: string;
}

/** Parses SSE frames independently of byte boundaries and line-ending style. */
export async function* parseEventStream(
	body: ReadableStream<Uint8Array>
): AsyncGenerator<WireEvent> {
	const reader = body.getReader();
	const decoder = new TextDecoder();
	let buffer = '';
	let ended = false;
	let previousCR = false;
	const normalizeLines = (text: string): string => {
		if (!text) return '';
		// A CR already ended the previous line. Its optional LF may arrive
		// in the next network chunk and must not become a second newline.
		const normalized = previousCR && text.startsWith('\n') ? text.slice(1) : text;
		previousCR = text.endsWith('\r');
		return normalized.replace(/\r\n?/g, '\n');
	};
	try {
		for (;;) {
			const { done, value } = await reader.read();
			if (done) {
				ended = true;
				// Preserve this client's residual-frame recovery at EOF; the
				// transport reports an incomplete answer separately below.
				buffer += normalizeLines(decoder.decode());
				if (buffer.trim()) {
					const parsed = parseBlock(buffer);
					if (parsed) yield parsed;
				}
				break;
			}
			buffer += normalizeLines(decoder.decode(value, { stream: true }));
			let sep: number;
			// CRLF, LF and CR have all been normalized before framing.
			while ((sep = buffer.indexOf('\n\n')) !== -1) {
				const block = buffer.slice(0, sep);
				buffer = buffer.slice(sep + 2);
				const parsed = parseBlock(block);
				if (parsed) yield parsed;
			}
		}
	} finally {
		// Releasing a lock alone leaves the response downloading when a
		// consumer breaks out before EOF (for example after stopping a turn).
		if (!ended) await reader.cancel().catch(() => {});
		reader.releaseLock();
	}
}

function parseBlock(block: string): WireEvent | null {
	let event = 'message';
	const dataLines: string[] = [];
	for (const rawLine of block.split('\n')) {
		const line = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine;
		const colon = line.indexOf(':');
		const field = colon < 0 ? line : line.slice(0, colon);
		let value = colon < 0 ? '' : line.slice(colon + 1);
		if (value.startsWith(' ')) value = value.slice(1);
		if (field === 'event') event = value || 'message';
		else if (field === 'data') dataLines.push(value);
	}
	if (dataLines.length === 0 && event === 'message') return null;
	return { event, data: dataLines.join('\n') };
}

export interface StreamOptions {
	signal?: AbortSignal;
	method?: 'GET' | 'POST';
	body?: unknown;
	headers?: Record<string, string>;
	/** Called with response headers as soon as they arrive, before the body streams. */
	onResponse?: (response: Response) => void;
}

/**
 * A synthetic wire event this client emits on transport-level failure (fetch
 * rejects, non-2xx status, or the stream ends without a `done`/terminal
 * `error` event) so every caller has exactly one error-handling path
 * Streaming transport.
 */
export function transportErrorEvent(message: string): WireEvent {
	return {
		event: 'error',
		data: JSON.stringify({ code: 'stream_error', message, recoverable: true })
	};
}

/** Non-2xx: fold the response body's reason into the synthetic error message
 *  so the UI shows e.g. "HTTP 404: conversation 17 not found" instead of a
 *  bare status. FastAPI errors are `{"detail": "..."}`; anything else is
 *  surfaced trimmed and raw. */
async function describeHttpError(response: Response): Promise<string> {
	let reason = '';
	try {
		const text = (await response.text()).trim();
		try {
			reason = (JSON.parse(text) as { detail?: string }).detail ?? text;
		} catch {
			reason = text;
		}
	} catch {
		// body unreadable — the bare status is all we have
	}
	return reason ? `HTTP ${response.status}: ${reason}` : `HTTP ${response.status}`;
}

/** Fetches an SSE endpoint and yields parsed wire events, GET or POST alike. */
export async function* fetchEventStream(
	url: string,
	options: StreamOptions = {}
): AsyncGenerator<WireEvent> {
	let response: Response;
	try {
		response = await fetch(url, {
			method: options.method ?? 'GET',
			signal: options.signal,
			headers: {
				Accept: 'text/event-stream',
				...(options.body ? { 'Content-Type': 'application/json' } : {}),
				...options.headers
			},
			body: options.body ? JSON.stringify(options.body) : undefined
		});
	} catch (err) {
		if ((err as Error)?.name === 'AbortError') return;
		yield transportErrorEvent(err instanceof Error ? err.message : 'network request failed');
		return;
	}

	options.onResponse?.(response);

	if (!response.ok || !response.body) {
		yield transportErrorEvent(await describeHttpError(response));
		return;
	}

	try {
		let finished = false;
		for await (const wire of parseEventStream(response.body)) {
			if (wire.event === 'done' || wire.event === 'error') finished = true;
			yield wire;
		}
		if (!finished) {
			// Silent EOF: the connection closed without a `done` or terminal
			// `error`. Surface it so callers see one error path instead of a
			// stream that just stops mid-answer; anything left in the parser's
			// buffer was dispatched above, right before this.
			yield transportErrorEvent('connection closed before the stream finished');
		}
	} catch (err) {
		if ((err as Error)?.name === 'AbortError') return;
		yield transportErrorEvent(err instanceof Error ? err.message : 'stream interrupted');
	}
}
