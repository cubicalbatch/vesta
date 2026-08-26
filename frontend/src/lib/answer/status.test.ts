import { describe, expect, it } from 'vitest';
import { formatTurnStatus } from './status';
import { createAnswerState, type AnswerState } from './reducer';
import type { SourceCard } from '../types';

function makeCard(zimId: number, path: string): SourceCard {
	return {
		zim_id: zimId,
		path,
		title: path,
		snippet: 'snippet',
		breadcrumb: path,
		score: 0.9,
		source: 'fts'
	};
}

describe('formatTurnStatus', () => {
	it('returns "Searching library…" when phase is null (initial turn start)', () => {
		const state = createAnswerState();
		expect(formatTurnStatus(state)).toBe('Searching library…');
	});

	it('returns "AI is reading 4 sources…" when in reading phase with 4 sources and "4 sources" detail', () => {
		const state: AnswerState = {
			...createAnswerState(),
			phase: 'reading',
			detail: '4 sources',
			sources: [
				makeCard(1, 'A'),
				makeCard(1, 'B'),
				makeCard(2, 'C'),
				makeCard(2, 'D')
			]
		};
		expect(formatTurnStatus(state)).toBe('AI is reading 4 sources…');
	});

	it('returns "AI is reading 1 source…" when in reading phase with 1 source', () => {
		const state: AnswerState = {
			...createAnswerState(),
			phase: 'reading',
			detail: '1 sources',
			sources: [makeCard(1, 'A')]
		};
		expect(formatTurnStatus(state)).toBe('AI is reading 1 source…');
	});

	it('returns "AI is reading…" when in reading phase with 0 sources and empty detail', () => {
		const state: AnswerState = {
			...createAnswerState(),
			phase: 'reading',
			detail: '',
			sources: []
		};
		expect(formatTurnStatus(state)).toBe('AI is reading…');
	});

	it('preserves warmup loading messages during reading phase', () => {
		const state: AnswerState = {
			...createAnswerState(),
			phase: 'reading',
			detail: 'Loading Qwen3.5 4B into memory…',
			sources: [makeCard(1, 'A')]
		};
		expect(formatTurnStatus(state)).toBe('Loading Qwen3.5 4B into memory…');
	});

	it('preserves contextual follow-up messages during reading phase', () => {
		const state: AnswerState = {
			...createAnswerState(),
			phase: 'reading',
			detail: 'Considering your question…',
			sources: []
		};
		expect(formatTurnStatus(state)).toBe('Considering your question…');
	});

	it('returns tool detail when searching (e.g. Reading source 1…)', () => {
		const state: AnswerState = {
			...createAnswerState(),
			phase: 'searching',
			detail: 'Reading source 1…'
		};
		expect(formatTurnStatus(state)).toBe('Reading source 1…');
	});

	it('returns fallback "Searching again…" when searching without detail', () => {
		const state: AnswerState = {
			...createAnswerState(),
			phase: 'searching',
			detail: ''
		};
		expect(formatTurnStatus(state)).toBe('Searching again…');
	});

	it('returns detail or "Generating…" during generating phase', () => {
		const stateWithDetail: AnswerState = {
			...createAnswerState(),
			phase: 'generating',
			detail: 'Thinking…'
		};
		expect(formatTurnStatus(stateWithDetail)).toBe('Thinking…');

		const stateWithoutDetail: AnswerState = {
			...createAnswerState(),
			phase: 'generating',
			detail: ''
		};
		expect(formatTurnStatus(stateWithoutDetail)).toBe('Generating…');
	});

	it('returns "Sources only" for sources_only phase', () => {
		const state: AnswerState = {
			...createAnswerState(),
			phase: 'sources_only',
			detail: '2 sources'
		};
		expect(formatTurnStatus(state)).toBe('Sources only');
	});

	it('returns completed summary when done', () => {
		const state: AnswerState = {
			...createAnswerState(),
			done: true,
			sources: [makeCard(1, 'A'), makeCard(2, 'B')]
		};
		expect(formatTurnStatus(state)).toBe('Answered · read 2 sources across 2 archives');
	});

	it('returns "Failed" for a terminal errored turn, never the answered summary', () => {
		const state: AnswerState = {
			...createAnswerState(),
			done: true,
			error: { code: 'no_llm', message: 'Model unavailable', recoverable: false }
		};
		const text = formatTurnStatus(state);
		expect(text).toBe('Failed');
		expect(text).not.toContain('Answered');
	});

	it('keeps the settled summary for an aborted turn (stop(): done without error)', () => {
		const state: AnswerState = {
			...createAnswerState(),
			done: true,
			text: 'Partial ans',
			buffer: 'Partial ans'
		};
		expect(formatTurnStatus(state)).toBe('Answered · read 0 sources across 0 archives');
	});
});
