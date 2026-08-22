import type { AnswerState } from './reducer';

/**
 * Derives user-facing status text for an answer turn, accurately reflecting
 * what the system is doing at each phase of the retrieval + answer pipeline.
 */
export function formatTurnStatus(state: AnswerState): string {
	if (state.error && !state.done) {
		return 'Error';
	}

	if (state.done) {
		const archives = new Set(state.sources.map((c) => c.zim_id)).size;
		return `Answered · read ${state.sources.length} sources across ${archives} archive${archives === 1 ? '' : 's'}`;
	}

	switch (state.phase) {
		case 'reading': {
			// If detail is an explicit lifecycle/warmup/follow-up message (e.g. "Loading Qwen3.5 4B into memory…",
			// "Considering your question…"), preserve it.
			const trimmed = state.detail?.trim() ?? '';
			if (
				trimmed &&
				!/^\d+\s+sources?$/i.test(trimmed) &&
				trimmed.toLowerCase() !== 'sources' &&
				trimmed.toLowerCase() !== 'reading'
			) {
				return trimmed;
			}
			const count = state.sources.length || (parseInt(trimmed, 10) || 0);
			if (count > 0) {
				return `AI is reading ${count} source${count === 1 ? '' : 's'}…`;
			}
			return 'AI is reading…';
		}
		case 'searching':
			return state.detail || 'Searching again…';
		case 'generating':
			return state.detail || 'Generating…';
		case 'abstaining':
			return state.detail || 'Checking your library…';
		case 'sources_only':
			return 'Sources only';
		default:
			return 'Searching library…';
	}
}
