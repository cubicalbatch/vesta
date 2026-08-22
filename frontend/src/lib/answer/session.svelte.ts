// `AnswerSession` — the orchestration state for one AI conversation, lifted
// out of the old AskPage.svelte so a single hero box can drive it from outside
// the component.
//
// It owns: priorTurns, liveTurn, conversationId, loadingHistory; and the
// methods startTurn / loadHistory / abort / reset. Everything in AskPage.svelte
// lines 33-128 verbatim, MINUS the pushState call — that stays in the
// orchestrator so this class stays testable without a SvelteKit runtime (the
// "pushState stays in the component, not the session class").
//
// The session exists only in AI mode. POST /api/chat now drives the streaming
// pydantic-ai agent directly (no strategy field): the server runs the agent
// whenever an LLM is configured and degrades to sources_only otherwise.
import { AnswerStream } from './answerStream.svelte';
import { streamChat } from './client';
import { createAnswerState, type AnswerState } from './reducer';
import { conversationsApi } from '../api/conversations';
import type { MessageDetail, SourceCard } from '../types';

export interface PriorTurn {
	query: string;
	state: AnswerState;
}

export interface LiveTurn {
	query: string;
	stream: AnswerStream;
}

function messageToState(assistant: MessageDetail | undefined): AnswerState {
	const s = createAnswerState();
	if (!assistant) return s;
	s.sources = (assistant.sources ?? []) as SourceCard[];
	s.text = assistant.content ?? '';
	s.buffer = s.text;
	s.trace = assistant.trace ?? null;
	s.done = true;
	return s;
}

export class AnswerSession {
	priorTurns = $state<PriorTurn[]>([]);
	liveTurn = $state<LiveTurn | null>(null);
	conversationId = $state<number | null>(null);
	loadingHistory = $state(false);

	private controller: AbortController | null = null;
	private rootDispose: (() => void) | null = null;

	/** Called when a brand-new conversation gets its id, so the orchestrator
	 *  can do the shallow pushState. Set by SearchPage; never imported here. */
	onConversationId?: (id: number) => void;

	/** Scope (csv of zim ids) passed through to every turn's chat request. */
	scope?: string;

	constructor() {
		// The teardown "$effect(() => () => controller?.abort())" must survive
		// the move out of AskPage. A plain .svelte.ts class
		// has no component lifecycle, so host it in an effect root; its cleanup
		// fires when dispose() is called (the orchestrator's teardown).
		this.rootDispose = $effect.root(() => {
			$effect(() => {
				return () => this.controller?.abort();
			});
		});
	}

	/** Tear the session down: abort any in-flight stream. Called by the
	 *  orchestrator on unmount and on leaving AI mode. */
	dispose() {
		this.controller?.abort();
		this.rootDispose?.();
		this.rootDispose = null;
	}

	/** Abort the in-flight stream without dropping the conversation state —
	 *  used when toggling modes or starting a new question mid-stream. */
	abort() {
		this.controller?.abort();
		this.liveTurn?.stream.stop();
	}

	async loadHistory(id: number) {
		this.loadingHistory = true;
		try {
			const detail = await conversationsApi.get(id);
			const built: PriorTurn[] = [];
			for (let i = 0; i < detail.messages.length; i++) {
				const m = detail.messages[i];
				if (m.role !== 'user') continue;
				const next = detail.messages[i + 1];
				built.push({
					query: m.content ?? '',
					state: messageToState(next?.role === 'assistant' ? next : undefined)
				});
			}
			this.priorTurns = built;
		} catch {
			this.priorTurns = [];
		} finally {
			this.loadingHistory = false;
		}
	}

	startTurn(query: string, { asRegenerate = false }: { asRegenerate?: boolean } = {}) {
		if (this.liveTurn && !asRegenerate) {
			this.priorTurns = [...this.priorTurns, { query: this.liveTurn.query, state: this.liveTurn.stream.state }];
		}
		this.controller?.abort();
		this.controller = new AbortController();
		const stream = new AnswerStream();
		this.liveTurn = { query, stream };
		const chat = streamChat(
			{
				query,
				conversation_id: this.conversationId ?? undefined,
				scope: this.scope ?? undefined
				// No `strategy` field: /api/chat runs the pydantic-ai agent
				// whenever an LLM is present, and degrades to sources_only
				// otherwise.
			},
			this.controller.signal
		);
		chat.conversationId.then((id) => {
			if (id != null && this.conversationId == null) {
				this.conversationId = Number(id);
				// The orchestrator owns URL writes; we only announce the id.
				this.onConversationId?.(this.conversationId);
			}
		});
		stream.consume(chat.events);
	}

	/** Back to a blank question — "New question", and leaving AI mode. Aborts
	 *  any in-flight stream and drops all conversation state. */
	reset() {
		this.controller?.abort();
		this.conversationId = null;
		this.priorTurns = [];
		this.liveTurn = null;
		this.loadingHistory = false;
	}
}
