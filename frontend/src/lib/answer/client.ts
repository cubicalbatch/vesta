// Typed wrapper over lib/sse.ts for the answer protocol (docs/sse-protocol.md).
// `streamChat` drives POST /api/chat (multi-turn) and resolves the conversation
// id from the X-Conversation-Id response header — read BEFORE the body is
// consumed, since that's the only place a new conversation's id is obtainable
// Streaming transport.
import { fetchEventStream, type WireEvent } from '../sse';
import type { AnswerEvent } from '../types';

function toAnswerEvent(wire: WireEvent): AnswerEvent | null {
	try {
		const data = wire.data ? JSON.parse(wire.data) : {};
		return { event: wire.event, data } as AnswerEvent;
	} catch {
		return {
			event: 'error',
			data: { code: 'stream_error', message: 'malformed event payload', recoverable: true }
		};
	}
}

export interface ChatBody {
	query: string;
	conversation_id?: number;
	scope?: string;
	profile?: string;
}

export interface ChatStream {
	/** Resolves once the response headers arrive — before any event is yielded. */
	conversationId: Promise<string | null>;
	events: AsyncGenerator<AnswerEvent>;
}

export function streamChat(body: ChatBody, signal?: AbortSignal): ChatStream {
	let settled = false;
	let resolveId: (id: string | null) => void;
	const conversationId = new Promise<string | null>((resolve) => {
		resolveId = resolve;
	});

	async function* run(): AsyncGenerator<AnswerEvent> {
		try {
			for await (const wire of fetchEventStream('/api/chat', {
				method: 'POST',
				body,
				signal,
				onResponse: (response) => {
					settled = true;
					resolveId(response.headers.get('X-Conversation-Id'));
				}
			})) {
				const event = toAnswerEvent(wire);
				if (event) yield event;
			}
		} finally {
			// If the request never reached a response (fetch itself rejected),
			// the id promise must still settle so callers awaiting it don't hang.
			if (!settled) resolveId(null);
		}
	}

	return { conversationId, events: run() };
}
