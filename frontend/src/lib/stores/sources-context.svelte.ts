// Reactive source-card list, provided via Svelte context so CitationChip can
// resolve `[n]` -> card without prop-drilling `sources` through every markdown
// token level (paragraph -> inline -> citation). The payoff
// (research/frontend-stack.md "How the streaming answer component works"): a
// chip fills in by itself when its card arrives, with no re-parse of the
// answer and no server round trip.
import { getContext, setContext } from 'svelte';
import type { CitationSpan, SourceCard } from '../types';

const KEY = Symbol('answer-sources');

export class SourcesContext {
	list = $state<SourceCard[]>([]);
	citations = $state<CitationSpan[]>([]);
	/** Card index most recently clicked/focused via a citation chip. */
	focused = $state<number | null>(null);
	/** Bumped on every focus() so listeners can react even to the same id twice. */
	focusToken = $state(0);

	/** Best (highest-score) span grounding a given card, if any. */
	spanForCard(cardId: number): CitationSpan | undefined {
		return this.citations
			.filter((s) => s.card_id === cardId)
			.sort((a, b) => b.score - a.score)[0];
	}

	focus(cardId: number) {
		this.focused = cardId;
		this.focusToken += 1;
	}
}

export function provideSources(): SourcesContext {
	const ctx = new SourcesContext();
	setContext(KEY, ctx);
	return ctx;
}

export function useSources(): SourcesContext {
	const ctx = getContext<SourcesContext>(KEY);
	if (!ctx) throw new Error('useSources() called outside a <SourcesProvider>');
	return ctx;
}
