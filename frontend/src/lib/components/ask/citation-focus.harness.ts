// Test harness for AskTurn.svelte's citation-focus $effect (AUDIT_0824 F7).
//
// Component testing isn't set up in this repo (node-env vitest, no
// @testing-library/svelte — see reader-article-effect.test.ts), and vitest's
// node environment compiles .svelte.ts with Svelte's server generator, where
// $effect is inert. So real runes can't run here. Instead this module
// simulates Svelte 5's dependency-tracking contract in miniature:
//
//   - reads of tracked state inside a running effect record dependencies;
//   - simUntrack(fn) suspends recording for fn — exactly the semantics the
//     production fix relies on;
//   - bumping any cell a live effect depends on re-runs that effect;
//   - TrackedAnswerState.replaceWith models the reducer returning a
//     brand-new AnswerState per streamed SSE event by invalidating every cell.
//
// The effect body mirrors AskTurn.svelte line-for-line INCLUDING the untrack()
// call sites: if someone removes them (or adds an answerState read outside
// them), the tracker records the read, each streamed token re-runs the
// effect, and citation-focus.test.ts fails. Not imported by production code.
import type { CitationSpan, SourceCard } from '$lib/types';

// --- miniature reactive core -------------------------------------------------

export type Cell = { version: number };
type Effect = { deps: Set<Cell>; run(): void };

const liveEffects = new Set<Effect>();
const pending = new Set<Effect>();
let active: Effect | null = null;

function cell(): Cell {
	return { version: 0 };
}

/** Read a cell under Svelte's tracking rules (records a dep if inside an effect). */
function read(c: Cell): number {
	if (active) active.deps.add(c);
	return c.version;
}

/** Invalidate a cell; dirties live effects depending on it (flush() runs them). */
function bump(c: Cell): void {
	c.version += 1;
	for (const e of liveEffects) {
		if (e.deps.has(c)) pending.add(e);
	}
}

/**
 * Drains the scheduler like Svelte's microtask tick: every mutation burst
 * (one reducer event, one chip click) is followed by exactly one re-run of
 * each dirtied effect.
 */
export function flush(): void {
	while (pending.size > 0) {
		const queued = [...pending];
		pending.clear();
		for (const e of queued) e.run();
	}
}

export function simUntrack<T>(fn: () => T): T {
	const prev = active;
	active = null;
	try {
		return fn();
	} finally {
		active = prev;
	}
}

/** Handle to an effect registered via simulateEffect. */
export interface SimulatedEffect {
	runs(): number;
	/** Stop reacting (stand-in for $effect.root's teardown). */
	dispose(): void;
}

/**
 * Runs `body` as an $effect would: records its dependencies on first run and
 * re-runs it whenever any recorded cell is bumped afterwards.
 */
export function simulateEffect(body: () => void): SimulatedEffect {
	let runs = 0;
	const effect: Effect = { deps: new Set(), run: () => {} };
	const runner = () => {
		runs += 1;
		effect.deps = new Set();
		const prev = active;
		active = effect;
		try {
			body();
		} finally {
			active = prev;
		}
	};
	effect.run = runner;
	runner();
	liveEffects.add(effect);
	return {
		runs: () => runs,
		dispose: () => {
			liveEffects.delete(effect);
		}
	};
}

// --- simulated reactive state ------------------------------------------------

/** The tracked view of AnswerState the mirrored effect reads through. */
export interface TrackedAnswerState {
	readonly sources: SourceCard[];
	sourceAt(id: number): SourceCard | undefined;
	/** Models `state = applyAnswerEvent(state, event)` — whole-object swap. */
	replaceWith(next: { sources: SourceCard[] }): void;
}

/**
 * AnswerState stand-in with per-field tracking cells. The reducer swaps the
 * whole state object on every event; replaceWith invalidates every cell.
 */
export function makeAnswerState(initial: { sources: SourceCard[] }): TrackedAnswerState {
	const sourcesArrayCell = cell(); // reading .sources itself
	const sourceAtCells = new Map<number, Cell>(); // reading sources[id]
	let current = initial;

	return {
		get sources(): SourceCard[] {
			read(sourcesArrayCell);
			return current.sources;
		},
		sourceAt(id: number): SourceCard | undefined {
			let c = sourceAtCells.get(id);
			if (!c) {
				c = cell();
				sourceAtCells.set(id, c);
			}
			read(c);
			return current.sources[id];
		},
		replaceWith(next: { sources: SourceCard[] }): void {
			current = next;
			bump(sourcesArrayCell);
			for (const c of sourceAtCells.values()) bump(c);
		}
	};
}

/**
 * SourcesContext stand-in mirroring exactly the surface the effect touches:
 * focusToken / focused reads, spanForCard, and the edge-triggering focus().
 */
export class SimSources {
	focusTokenCell = cell();
	focusedCell = cell();
	private _focusToken = 0;
	private _focused: number | null = null;

	constructor(private spans: CitationSpan[]) {}

	get focusToken(): number {
		read(this.focusTokenCell);
		return this._focusToken;
	}

	get focused(): number | null {
		read(this.focusedCell);
		return this._focused;
	}

	spanForCard(cardId: number): CitationSpan | undefined {
		return this.spans.filter((s) => s.card_id === cardId).sort((a, b) => b.score - a.score)[0];
	}

	focus(cardId: number): void {
		this._focused = cardId;
		this._focusToken += 1;
		bump(this.focusTokenCell);
		bump(this.focusedCell);
	}
}

/** Minimal stand-in for the `sourcesEl` aside DOM node. */
const fakeSourcesEl = {
	querySelector(sel: string): { scrollIntoView(o: object): void; animate(k: object[], o: object): unknown } | null {
		return sel.startsWith('[data-card-slot=')
			? { scrollIntoView: () => {}, animate: () => ({}) }
			: null;
	}
};

export interface FocusLog {
	opens: Array<{
		zimId: number;
		path: string;
		cards: SourceCard[];
		cardIndex: number;
		spanScore: number | null;
	}>;
	scrolls: number;
}

/**
 * Mirrors AskTurn.svelte's citation-focus $effect verbatim (same reactive
 * reads, same untrack boundaries); readerStore.open is replaced by a log.
 */
export function mountCitationFocus(
	sources: SimSources,
	answerState: TrackedAnswerState
): { log: FocusLog; eff: SimulatedEffect } {
	const log: FocusLog = { opens: [], scrolls: 0 };
	const sourcesEl = fakeSourcesEl;

	const eff = simulateEffect(() => {
		// --- verbatim mirror of AskTurn's focus effect ---
		sources.focusToken;
		const id = sources.focused;
		if (id == null || !sourcesEl) return;
		const el = sourcesEl.querySelector(`[data-card-slot="${id}"]`);
		el?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
		el?.animate(
			[{ boxShadow: '0 0 0 3px var(--accent-ring)' }, { boxShadow: '0 0 0 0 transparent' }],
			{ duration: 900 }
		);
		log.scrolls += 1;
		const card = simUntrack(() => answerState.sourceAt(id));
		if (card) {
			const span = sources.spanForCard(id);
			simUntrack(() =>
				log.opens.push({
					zimId: card.zim_id,
					path: card.path,
					cards: answerState.sources,
					cardIndex: id,
					spanScore: span?.score ?? null
				})
			);
		}
	});

	return { log, eff };
}
