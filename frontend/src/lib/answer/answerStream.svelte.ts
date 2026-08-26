// Runes-backed wrapper around the pure reducer, driving a SourcesContext so
// CitationChip resolves reactively with no re-parse of the answer. `replay()`
// is what the Task 0 demo (and manual QA) use against recorded/synthetic
// fixtures; `consume()` drives the same class from the live SSE client
// (GET /api/answer or POST /api/chat) for the real Ask/Search pages.
import type { AnswerEvent } from '../types';
import { applyAnswerEvent, createAnswerState, type AnswerState } from './reducer';
import type { SourcesContext } from '../stores/sources-context.svelte';

export class AnswerStream {
	state = $state<AnswerState>(createAnswerState());
	private sources?: SourcesContext;
	private timer: ReturnType<typeof setTimeout> | undefined = undefined;
	private consuming = false;

	constructor(sources?: SourcesContext) {
		this.sources = sources;
	}

	private apply(event: AnswerEvent) {
		this.state = applyAnswerEvent(this.state, event);
		if (this.sources) {
			this.sources.list = this.state.sources;
			this.sources.citations = this.state.citations;
		}
	}

	reset() {
		clearTimeout(this.timer);
		this.timer = undefined;
		this.consuming = false;
		this.state = createAnswerState();
		if (this.sources) {
			this.sources.list = [];
			this.sources.citations = [];
		}
	}

	/** Replay a recorded/synthetic event array with a per-event delay, for demos and manual QA. */
	replay(events: AnswerEvent[], delayMs = 40) {
		this.reset();
		let i = 0;
		const step = () => {
			if (i >= events.length) {
				this.timer = undefined;
				return;
			}
			this.apply(events[i]);
			i += 1;
			this.timer = setTimeout(step, delayMs);
		};
		// Defer even the first tick: `apply()` reads-then-writes `this.state`,
		// and calling it synchronously from within a caller's `$effect` (as the
		// demo harness does) makes Svelte's dependency tracker see that effect
		// both read and write the same state in one flush -> infinite rerun
		// ("effect_update_depth_exceeded"). Scheduling every tick, including the
		// first, keeps every mutation outside any enclosing effect's sync scope.
		this.timer = setTimeout(step, delayMs);
	}

	/** Apply every event synchronously — for instant/no-flicker checks. */
	replayInstant(events: AnswerEvent[]) {
		this.reset();
		for (const event of events) this.apply(event);
	}

	/** Drive from a live AsyncGenerator (the SSE client) — the real Ask/Search path. */
	async consume(events: AsyncGenerator<AnswerEvent>) {
		this.reset();
		this.consuming = true;
		for await (const event of events) {
			if (!this.consuming) return; // stop() was called mid-stream
			// Same reasoning as replay(): never apply synchronously inside
			// whatever effect scope kicked off consume() — yield a tick first.
			await Promise.resolve();
			if (!this.consuming) return;
			this.apply(event);
		}
	}

	stop() {
		this.consuming = false;
		clearTimeout(this.timer);
		this.timer = undefined;
		// An aborted stream never receives the wire's `done` event, but the
		// turn must still end: without a terminal state the spinner, elapsed
		// counter and "Generating…" run forever and the follow-up input stays
		// disabled. Apply the same terminal shape the reducer gives `done`
		// (and `error`): keep whatever partial content arrived, set done.
		this.apply({ event: 'done', data: {} });
	}
}
