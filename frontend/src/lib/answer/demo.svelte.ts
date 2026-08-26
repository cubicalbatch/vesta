// The welcome wizard's "Ask a test question" drives the real POST /api/chat,
// which creates and persists a conversation whenever none is named — every
// click used to seed another demo conversation into history (AUDIT_0824 N16).
// The fix is frontend-owned and API-honest (no backend UI-awareness): remember
// the conversation id the first stream returns via X-Conversation-Id and
// continue THAT conversation on every later click — at most one demo
// conversation per wizard visit, never one per click.
import { streamChat } from './client';
import { AnswerStream } from './answerStream.svelte';

export const TEST_QUESTION = 'In one short sentence, what is Wikipedia?';

export class DemoAsk {
	readonly stream = new AnswerStream();
	running = $state(false);
	private controller: AbortController | null = null;
	private conversationId: number | null = null;

	async ask(): Promise<void> {
		if (this.running) return;
		this.running = true;
		this.controller?.abort();
		this.controller = new AbortController();
		try {
			const chat = streamChat(
				this.conversationId === null
					? { query: TEST_QUESTION }
					: { query: TEST_QUESTION, conversation_id: this.conversationId },
				this.controller.signal
			);
			void chat.conversationId.then((id) => {
				if (id !== null && this.conversationId === null) this.conversationId = Number(id);
			});
			await this.stream.consume(chat.events);
		} finally {
			this.running = false;
		}
	}

	/** Aborted on page teardown so navigating away mid-demo cancels the fetch
	 *  instead of streaming tokens into a dead component. */
	dispose(): void {
		this.controller?.abort();
		this.stream.stop();
	}
}
