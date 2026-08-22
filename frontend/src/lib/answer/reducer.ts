// Pure reducer over the frozen SSE answer protocol (docs/sse-protocol.md).
// Framework-agnostic and unit-tested directly (reducer.test.ts) against the
// recorded fixtures plus synthetic streams for the three amendments
// that only fire on recovery paths (merge, answer_reset, answer_text).
//
// Five rules this reducer exists to get right:
// "The answer stream"):
//   1. `sources` merge:true APPENDS, continuing the same 0-based numbering.
//   2. `answer_reset` clears all accumulated token text and starts over.
//   3. `citations.answer_text`, when non-null, REPLACES the token concatenation
//      for display and persistence — it has [n] markers rewritten from passage
//      numbers to card numbers.
//   4. `error` with recoverable:true keeps the partial answer.
//   5. Abstention is a normal `status.phase === "abstaining"` outcome, not an error.

import type { AnswerError, AnswerEvent, AnswerPhase, CitationSpan, SourceCard, Trace } from '../types';

export interface AnswerState {
	sources: SourceCard[];
	phase: AnswerPhase | null;
	detail: string;
	/** Raw concatenation of `token.text` since the last `answer_reset` (or start). */
	buffer: string;
	/** What to render: `buffer`, or `citations.answer_text` once it arrives. */
	text: string;
	citations: CitationSpan[];
	trace: Trace | null;
	done: boolean;
	error: AnswerError | null;
}

export function createAnswerState(): AnswerState {
	return {
		sources: [],
		phase: null,
		detail: '',
		buffer: '',
		text: '',
		citations: [],
		trace: null,
		done: false,
		error: null
	};
}

/** Applies one SSE event to a state, returning a new state (never mutates). */
export function applyAnswerEvent(state: AnswerState, event: AnswerEvent): AnswerState {
	switch (event.event) {
		case 'sources': {
			const incoming = event.data.cards;
			if (event.data.merge) {
				// Delta only — append, continuing the existing 0-based numbering
				// (docs/sse-protocol.md "Sources merge" rule 2). Replacing here is
				// the single most likely correctness bug: it breaks every citation
				// grounded in recovered evidence.
				return {
					...state,
					sources: [...state.sources, ...incoming.map((c) => ({ ...c, recovered: true }))]
				};
			}
			return { ...state, sources: incoming };
		}

		case 'status':
			return { ...state, phase: event.data.phase, detail: event.data.detail };

		case 'token': {
			const buffer = state.buffer + event.data.text;
			return { ...state, buffer, text: buffer };
		}

		case 'answer_reset':
			// Everything streamed so far for this turn is superseded by the
			// regenerate that follows. Discard, don't append.
			return { ...state, buffer: '', text: '' };

		case 'citations': {
			const text = event.data.answer_text != null ? event.data.answer_text : state.buffer;
			return { ...state, citations: event.data.spans, text };
		}

		case 'trace':
			return { ...state, trace: event.data };

		case 'done':
			return { ...state, done: true };

		case 'error':
			// Recoverable or not, the partial answer + its citations stay exactly
			// as accumulated — an error never wipes `text`.
			return { ...state, error: event.data, done: true };

		default:
			return state;
	}
}

export function replayAnswerEvents(events: AnswerEvent[]): AnswerState {
	return events.reduce(applyAnswerEvent, createAnswerState());
}
