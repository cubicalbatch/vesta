// Reader-overlay open/target state (Component: Reader). Lives here, not in
// the Reader component, because the app shell's Esc handler needs to close it
// from anywhere, and any page (Ask, Search, Catalog) needs to be able to open
// it without owning the drawer itself.
import type { MediaOut, SourceCard } from '../types';

export interface ReaderTarget {
	zimId: number;
	path: string;
	title?: string;
	/** The answer's full card list, for "prev/next across the answer's cards". */
	cards?: SourceCard[];
	cardIndex?: number;
	passageSpan?: [number, number] | null;
	/** Set when opened via a citation chip — "cited as [n]" in the provenance strip. */
	citedAs?: number;
	/** 0008: media assets for a media-ZIM entry (native <video> instead of iframe). */
	media?: MediaOut | null;
}

class ReaderStore {
	target = $state<ReaderTarget | null>(null);

	get isOpen() {
		return this.target !== null;
	}

	open(target: ReaderTarget) {
		this.target = target;
	}

	close() {
		this.target = null;
	}

	next() {
		const t = this.target;
		if (!t?.cards || t.cardIndex == null) return;
		const i = (t.cardIndex + 1) % t.cards.length;
		const card = t.cards[i];
		this.target = {
			...t,
			zimId: card.zim_id,
			path: card.path,
			title: card.title,
			media: card.media,
			cardIndex: i,
			citedAs: undefined
		};
	}

	prev() {
		const t = this.target;
		if (!t?.cards || t.cardIndex == null) return;
		const i = (t.cardIndex - 1 + t.cards.length) % t.cards.length;
		const card = t.cards[i];
		this.target = {
			...t,
			zimId: card.zim_id,
			path: card.path,
			title: card.title,
			media: card.media,
			cardIndex: i,
			citedAs: undefined
		};
	}
}

export const readerStore = new ReaderStore();
