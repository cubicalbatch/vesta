// Deleting the currently-open conversation used to strand the session
// (AUDIT_0824 N15): AskHistory only filtered its popover list, so the session
// kept conversationId and every follow-up POSTed a dead id → 404. The fix is
// SearchPage.conversationDeleted → newQuestion() → session.reset(). These pin
// the contract that makes that wiring sufficient: before reset() a turn
// carries the conversation id; after reset(), startTurn sends NO
// conversation_id — nothing can strand on a deleted id.
//
// Component testing isn't set up in this repo (see client.test.ts), so the
// SearchPage guard (`id === session.conversationId`) is exercised here against
// a real AnswerSession.
import { describe, expect, it, vi, afterEach, type Mock } from 'vitest';
import { AnswerSession } from './session.svelte';
import type { AnswerState } from './reducer';
import { conversationsApi } from '../api/conversations';
import type { ConversationDetail } from '../types';

vi.mock('../api/conversations', () => ({
	conversationsApi: { get: vi.fn() }
}));

const getDetail = conversationsApi.get as unknown as Mock;

function detailOf(pairs: Array<[string, string | null]>): ConversationDetail {
	return {
		conversation: { id: 1, title: null, created_at: null, updated_at: null },
		messages: pairs.map(([role, content], i) => ({
			id: i + 1,
			role,
			content,
			sources: null,
			trace: null,
			tokens_in: null,
			tokens_out: null,
			latency_ms: null,
			created_at: null
		}))
	};
}

/** Deferred promise so a test decides exactly when a history fetch resolves. */
function deferred<T>() {
	let resolve!: (value: T) => void;
	const promise = new Promise<T>((r) => (resolve = r));
	return { promise, resolve };
}

const DONE = 'event: done\ndata: {}\n\n';
const enc = new TextEncoder();

function finiteSseResponse(): Response {
	const body = new ReadableStream<Uint8Array>({
		start(controller) {
			controller.enqueue(enc.encode(DONE));
			controller.close();
		}
	});
	return {
		ok: true,
		status: 200,
		headers: new Headers({ 'X-Conversation-Id': '17' }),
		body
	} as unknown as Response;
}

/** Deterministically run pending microtasks — enough ticks for consume() to
 *  pull and apply everything the fake response has yielded (see
 *  answerStream.test.ts for the same idiom). */
async function drainMicrotasks(): Promise<void> {
	for (let i = 0; i < 50; i++) await Promise.resolve();
}

afterEach(() => vi.unstubAllGlobals());

describe('AnswerSession.reset strands nothing', () => {
	it('a follow-up before reset carries the conversation id', async () => {
		const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => finiteSseResponse());
		vi.stubGlobal('fetch', fetchMock);
		const session = new AnswerSession();
		try {
			session.conversationId = 17;
			session.startTurn('first');
			await drainMicrotasks();

			const body = JSON.parse(String(fetchMock.mock.calls[0][1]?.body)) as Record<string, unknown>;
			expect(body.query).toBe('first');
			expect(body.conversation_id).toBe(17);
		} finally {
			session.dispose();
		}
	});

	it('after reset(), startTurn posts no conversation_id and state is blank', async () => {
		const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => finiteSseResponse());
		vi.stubGlobal('fetch', fetchMock);
		const session = new AnswerSession();
		try {
			session.conversationId = 17;
			session.priorTurns = [
				{ query: 'old', state: { done: true } as AnswerState }
			];
			session.reset();

			expect(session.conversationId).toBeNull();
			expect(session.priorTurns).toEqual([]);
			expect(session.liveTurn).toBeNull();

			session.startTurn('next');
			await drainMicrotasks();

			const body = JSON.parse(String(fetchMock.mock.calls[0][1]?.body)) as Record<string, unknown>;
			expect(body.conversation_id).toBeUndefined();
		} finally {
			session.dispose();
		}
	});
});

describe('loadHistory stale-resolution guard (AUDIT_0824 F4)', () => {
	it('a normal load populates priorTurns', async () => {
		getDetail.mockResolvedValue(
			detailOf([
				['user', 'q1'],
				['assistant', 'a1']
			])
		);
		const session = new AnswerSession();
		try {
			await session.loadHistory(7);
			expect(session.priorTurns.map((t) => t.query)).toEqual(['q1']);
			expect(session.loadingHistory).toBe(false);
		} finally {
			session.dispose();
		}
	});

	it('reset() mid-flight drops the late resolution instead of merging two threads', async () => {
		const gate = deferred<ConversationDetail>();
		getDetail.mockReturnValue(gate.promise);
		const session = new AnswerSession();
		try {
			const pending = session.loadHistory(17);
			session.reset();
			gate.resolve(detailOf([['user', 'stale'], ['assistant', 'stale answer']]));
			await pending;

			expect(session.priorTurns).toEqual([]);
			expect(session.conversationId).toBeNull();
			expect(session.loadingHistory).toBe(false);
		} finally {
			session.dispose();
		}
	});

	it('a newer loadHistory wins; the slower older one is dropped entirely', async () => {
		const older = deferred<ConversationDetail>();
		const newer = deferred<ConversationDetail>();
		getDetail.mockReturnValueOnce(older.promise).mockReturnValueOnce(newer.promise);
		const session = new AnswerSession();
		try {
			const p1 = session.loadHistory(1);
			const p2 = session.loadHistory(2);
			newer.resolve(detailOf([['user', 'fresh'], ['assistant', 'fresh answer']]));
			await p2;
			expect(session.priorTurns.map((t) => t.query)).toEqual(['fresh']);

			older.resolve(detailOf([['user', 'stale'], ['assistant', 'stale answer']]));
			await p1;
			// The stale conversation's turns never merge into the fresh thread.
			expect(session.priorTurns.map((t) => t.query)).toEqual(['fresh']);
			expect(session.loadingHistory).toBe(false);
		} finally {
			session.dispose();
		}
	});
});
