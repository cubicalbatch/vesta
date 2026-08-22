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

/** Parses the `event:`/`data:` SSE wire format out of a byte stream. A dozen lines, by design. */
export async function* parseEventStream(
	body: ReadableStream<Uint8Array>
): AsyncGenerator<WireEvent> {
	const reader = body.getReader();
	const decoder = new TextDecoder();
	let buffer = '';
	try {
		for (;;) {
			const { done, value } = await reader.read();
			if (done) break;
			buffer += decoder.decode(value, { stream: true });
			let sep: number;
			// SSE frames are separated by a blank line ("\n\n"); tolerate "\r\n\r\n".
			while ((sep = buffer.indexOf('\n\n')) !== -1) {
				const block = buffer.slice(0, sep);
				buffer = buffer.slice(sep + 2);
				const parsed = parseBlock(block);
				if (parsed) yield parsed;
			}
		}
	} finally {
		reader.releaseLock();
	}
}

function parseBlock(block: string): WireEvent | null {
	let event = 'message';
	const dataLines: string[] = [];
	for (const rawLine of block.split('\n')) {
		const line = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine;
		if (line.startsWith('event:')) event = line.slice(6).trim();
		else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
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
		yield transportErrorEvent(`HTTP ${response.status}`);
		return;
	}

	try {
		yield* parseEventStream(response.body);
	} catch (err) {
		if ((err as Error)?.name === 'AbortError') return;
		yield transportErrorEvent(err instanceof Error ? err.message : 'stream interrupted');
	}
}
