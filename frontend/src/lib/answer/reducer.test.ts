import { describe, expect, it } from 'vitest';
import { applyAnswerEvent, createAnswerState, replayAnswerEvents } from './reducer';
import type { AnswerEvent } from '../types';

import singleShotCited from '../dev-fixtures/recorded/single_shot_cited.json';
import sourcesOnly from '../dev-fixtures/recorded/sources_only.json';
import abstention from '../dev-fixtures/recorded/abstention.json';
import agenticRecovery from '../dev-fixtures/recorded/agentic_recovery.json';
import answerReset from '../dev-fixtures/synthetic/answer_reset.json';
import answerTextRewrite from '../dev-fixtures/synthetic/answer_text_rewrite.json';
import recoverableError from '../dev-fixtures/synthetic/recoverable_error.json';

function events(fixture: { events: unknown[] }): AnswerEvent[] {
	return fixture.events as AnswerEvent[];
}

describe('recorded fixtures (tests/fixtures/sse)', () => {
	it('single_shot_cited: streams tokens, ends done with one citation', () => {
		const state = replayAnswerEvents(events(singleShotCited));
		expect(state.done).toBe(true);
		expect(state.error).toBeNull();
		expect(state.sources).toHaveLength(2);
		expect(state.text).toBe(
			'The Battle of Hastings was fought on 14 October 1066 during the Norman conquest of England [1].'
		);
		expect(state.citations).toHaveLength(1);
		expect(state.citations[0].card_id).toBe(0);
	});

	it('sources_only: no tokens, phase sources_only, still a complete page', () => {
		const state = replayAnswerEvents(events(sourcesOnly));
		expect(state.done).toBe(true);
		expect(state.phase).toBe('sources_only');
		expect(state.text).toBe('');
		expect(state.sources).toHaveLength(1);
	});

	it('abstention: renders as a normal single-token outcome, not an error', () => {
		const state = replayAnswerEvents(events(abstention));
		expect(state.done).toBe(true);
		expect(state.error).toBeNull();
		expect(state.phase).toBe('abstaining');
		expect(state.sources).toHaveLength(1);
		expect(state.text).toContain('No passage in your archives closely matches');
	});

	it('agentic_recovery: merge:true APPENDS, continuing 0-based numbering', () => {
		const state = replayAnswerEvents(events(agenticRecovery));
		// Round 0 had 1 card (index 0); the merge event's card must land at index 1,
		// not replace the list down to 1 card of its own.
		expect(state.sources).toHaveLength(2);
		expect(state.sources[0].path).toBe('Air');
		expect(state.sources[0].recovered).toBeUndefined();
		expect(state.sources[1].path).toBe('Atmosphere_of_Earth');
		expect(state.sources[1].recovered).toBe(true);
		// The citation is grounded in the recovered card and must resolve into
		// the concatenated list, not against a replaced 1-card list.
		expect(state.citations[0].card_id).toBe(1);
		expect(state.sources[state.citations[0].card_id].path).toBe('Atmosphere_of_Earth');
	});
});

describe('synthetic streams (2026-08-02 amendments)', () => {
	it('answer_reset: discards the first draft, keeps only the regenerate', () => {
		const state = replayAnswerEvents(events(answerReset));
		expect(state.done).toBe(true);
		expect(state.text).toBe('Nothing in your library specifically covers a 2027 currency reform [1].');
		// The pre-reset draft must not survive as a prefix or suffix anywhere.
		expect(state.text).not.toContain('hyperinflation');
	});

	it('answer_reset via incremental apply never concatenates two drafts', () => {
		let state = createAnswerState();
		for (const e of events(answerReset)) {
			state = applyAnswerEvent(state, e);
		}
		expect(state.text.match(/currency reform/g)?.length).toBe(1);
	});

	it('answer_text: replaces the token concatenation for display, not just appends', () => {
		const state = replayAnswerEvents(events(answerTextRewrite));
		expect(state.buffer).toBe('The Eiffel Tower was completed in 1889 [3] and stands 330 metres tall [1].');
		// text (the render/persist target) must be the rewritten card-numbered
		// version, not the raw passage-numbered buffer.
		expect(state.text).toBe(
			'The Eiffel Tower was completed in 1889 [1] and stands 330 metres tall [2].'
		);
		expect(state.text).not.toBe(state.buffer);
	});

	it('recoverable error: keeps the partial answer and does not clear it', () => {
		const state = replayAnswerEvents(events(recoverableError));
		expect(state.error).not.toBeNull();
		expect(state.error?.recoverable).toBe(true);
		expect(state.text).toBe('France has a population of approximately');
		expect(state.done).toBe(true);
	});
});

describe('follow-up streams (context-aware follow-ups)', () => {
	it('from-context follow-up: status-first, zero sources events, still renders', () => {
		// A follow-up answered from the conversation: no retrieval, so no
		// `sources` event at all. The stream starts with `status`.
		const stream: AnswerEvent[] = [
			{ event: 'status', data: { phase: 'reading', detail: 'Considering your question…' } },
			{ event: 'status', data: { phase: 'generating', detail: 'Thinking…' } },
			{ event: 'token', data: { text: 'Lafayette was born first, in 1757.' } },
			{ event: 'citations', data: { spans: [], answer_text: 'Lafayette was born first, in 1757.' } },
			{ event: 'trace', data: { system: 'agentic_pydantic', followup: true, elapsed_ms: 1200, total_tokens: 0, input_tokens: 0, output_tokens: 0, search_calls: 0, read_calls: 0, card_count: 0 } },
			{ event: 'done', data: {} }
		];
		const state = replayAnswerEvents(stream);
		expect(state.done).toBe(true);
		expect(state.sources).toEqual([]); // no sources event → empty list, never undefined
		expect(state.phase).toBe('generating');
		expect(state.text).toBe('Lafayette was born first, in 1757.');
		expect(state.citations).toEqual([]);
	});

	it('follow-up search: mid-stream sources(merge:false) after a status lands correctly', () => {
		// The agent searched on a follow-up: the first (only) `sources` event
		// arrives AFTER a status, mid-stream, with merge:false. It must set the
		// card list (not append to nothing), and the citation must resolve.
		const stream: AnswerEvent[] = [
			{ event: 'status', data: { phase: 'reading', detail: 'Considering your question…' } },
			{ event: 'status', data: { phase: 'searching', detail: 'Searching…' } },
			{
				event: 'sources',
				data: {
					cards: [
						{ zim_id: 1, path: 'Napoleon', title: 'Napoleon', snippet: '', breadcrumb: 'Napoleon', score: 0.9, source: 'xapian_fts' }
					],
					merge: false
				}
			},
			{ event: 'status', data: { phase: 'generating', detail: 'Thinking…' } },
			{ event: 'token', data: { text: 'Napoleon died first, in 1821 [1].' } },
			{
				event: 'citations',
				data: {
					spans: [{ answer_span: [0, 9], card_id: 0, passage_span: null, score: 1.0 }],
					answer_text: 'Napoleon died first, in 1821 [1].'
				}
			},
			{ event: 'trace', data: { system: 'agentic_pydantic', followup: true, elapsed_ms: 1600, total_tokens: 0, input_tokens: 0, output_tokens: 0, search_calls: 1, read_calls: 0, card_count: 1 } },
			{ event: 'done', data: {} }
		];
		const state = replayAnswerEvents(stream);
		expect(state.done).toBe(true);
		expect(state.sources).toHaveLength(1);
		expect(state.sources[0].path).toBe('Napoleon');
		expect(state.citations[0].card_id).toBe(0);
		expect(state.sources[state.citations[0].card_id].path).toBe('Napoleon');
	});
});
