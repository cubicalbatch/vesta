// Citation-focus effect over-subscription (AUDIT_0824 F7).
//
// AskTurn.svelte's focus $effect used to read `answerState.sources[id]`
// reactively. The reducer (applyAnswerEvent) returns a fresh AnswerState
// object on EVERY streamed event — including each `token` — so after one
// citation-chip click during a live stream, every subsequent token re-ran the
// effect: re-scrolling the sources aside and calling readerStore.open() again,
// which re-triggers the Reader drawer's article refetch / PDF / poll effects.
// The fix wraps all answerState reads in untrack(), making the effect purely
// edge-triggered on the store's focusToken.
//
// The harness mirrors that effect body under a faithful model of Svelte 5's
// dependency tracking (see citation-focus.harness.ts for why real runes can't
// run in vitest's node environment); these tests drive it with real reducer
// output to prove once-only focus under streaming.
import { afterEach, describe, expect, it } from 'vitest';
import { applyAnswerEvent, createAnswerState, type AnswerState } from '$lib/answer/reducer';
import type { SourceCard } from '$lib/types';
import { flush, makeAnswerState, mountCitationFocus, SimSources } from './citation-focus.harness';

const cleanups: Array<() => void> = [];
afterEach(() => {
	for (const c of cleanups) c();
	cleanups.length = 0;
});

function card(i: number): SourceCard {
	return {
		zim_id: 7,
		path: `wiki/Article_${i}`,
		title: `Article ${i}`,
		snippet: '…',
		breadcrumb: 'Wiki',
		score: 1 - i * 0.1,
		source: 'xapian_fts'
	};
}

function stateWithSources(): AnswerState {
	return applyAnswerEvent(createAnswerState(), {
		event: 'sources',
		data: { cards: [card(0), card(1)], merge: false }
	});
}

/** One streamed token: the reducer replaces the whole state object. */
function streamToken(state: AnswerState, text: string): AnswerState {
	return applyAnswerEvent(state, { event: 'token', data: { text } });
}

describe('citation-focus effect is edge-triggered (one open per click)', () => {
	it('streams tokens before any click without opening the reader', () => {
		const sources = new SimSources([]);
		const state = makeAnswerState({ sources: [] });
		const h = mountCitationFocus(sources, state);
		cleanups.push(h.eff.dispose);

		let s = stateWithSources();
		state.replaceWith(s);
		flush();
		for (let i = 0; i < 10; i++) {
			s = streamToken(s, ` token ${i}`);
			state.replaceWith(s);
			flush();
		}

		expect(h.log.opens).toHaveLength(0);
		expect(h.log.scrolls).toBe(0);
	});

	it('one chip click mid-stream opens exactly once despite ongoing tokens', () => {
		const spans = [
			{ answer_span: [0, 5] as [number, number], card_id: 1, passage_span: null, score: 0.9 }
		];
		const sources = new SimSources(spans);
		const state = makeAnswerState({ sources: [] });
		const h = mountCitationFocus(sources, state);
		cleanups.push(h.eff.dispose);

		let s = stateWithSources();
		state.replaceWith(s);
		flush();
		expect(h.log.opens).toHaveLength(0);

		// User clicks citation chip [2] (card index 1) while the answer streams.
		sources.focus(1);
		flush();
		expect(h.log.opens).toHaveLength(1);
		expect(h.log.opens[0]).toMatchObject({ zimId: 7, path: 'wiki/Article_1', cardIndex: 1 });
		expect(h.log.opens[0].spanScore).toBe(0.9);

		// The stream keeps replacing the whole state object per token — this is
		// the exact churn that used to reopen the Reader per token.
		for (let i = 0; i < 20; i++) {
			s = streamToken(s, ` more ${i} [1]`);
			state.replaceWith(s);
			flush();
		}
		expect(h.log.opens).toHaveLength(1);

		// citations + done events also replace state — still no re-open.
		s = applyAnswerEvent(s, { event: 'citations', data: { spans } });
		state.replaceWith(s);
		flush();
		s = applyAnswerEvent(s, { event: 'done', data: {} });
		state.replaceWith(s);
		flush();

		expect(h.log.opens).toHaveLength(1);
		expect(h.log.scrolls).toBe(1);
	});

	it('re-clicking the same chip focuses again (per-click, not global-once)', () => {
		const sources = new SimSources([]);
		const state = makeAnswerState({ sources: [] });
		const h = mountCitationFocus(sources, state);
		cleanups.push(h.eff.dispose);

		state.replaceWith(stateWithSources());
		flush();

		sources.focus(1);
		flush();
		sources.focus(1);
		flush();
		expect(h.log.opens).toHaveLength(2);
		expect(h.log.scrolls).toBe(2);

		let s = stateWithSources();
		for (let i = 0; i < 5; i++) {
			s = streamToken(s, ' x');
			state.replaceWith(s);
			flush();
		}
		expect(h.log.opens).toHaveLength(2);
	});

	it('focus consumed on a click whose card is missing is never applied retroactively', () => {
		const sources = new SimSources([]);
		const state = makeAnswerState({ sources: [card(0)] });
		const h = mountCitationFocus(sources, state);
		cleanups.push(h.eff.dispose);

		let s = applyAnswerEvent(createAnswerState(), {
			event: 'sources',
			data: { cards: [card(0)], merge: false }
		});
		state.replaceWith(s);
		flush();
		expect(h.log.opens).toHaveLength(0);

		// Chips can transiently resolve [n] against a sources list that doesn't
		// yet hold that card mid-stream; focus() fires but the card lookup
		// misses and the focus request is consumed — not held pending.
		sources.focus(1);
		flush();
		expect(h.log.opens).toHaveLength(0);

		for (let i = 0; i < 10; i++) {
			s = streamToken(s, ' t');
			state.replaceWith(s);
			flush();
		}
		s = applyAnswerEvent(s, { event: 'sources', data: { cards: [card(0), card(1)], merge: true } });
		state.replaceWith(s);
		flush();

		// Card 1 has now arrived via the merge event — but no NEW click means no
		// retroactive Reader open.
		expect(h.log.opens).toHaveLength(0);
	});
});
