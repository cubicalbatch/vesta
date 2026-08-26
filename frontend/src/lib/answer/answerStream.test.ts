// Aborting a live stream must end the turn (AUDIT_0824 N14). SearchPage.setMode
// calls session.abort() — which stops the stream but deliberately keeps the
// session — so the restored turn previously sat on a non-terminal state
// forever: spinner + "Generating…" + ticking elapsed counter and a disabled
// follow-up input. These tests pin stop()'s contract: mark the turn terminal
// with whatever partial content arrived, exactly like the reducer treats a
// wire `done`.
//
// Component testing isn't set up in this repo (see client.test.ts), so the UI
// gates are asserted through the same derivations the components use:
// AskTurn.svelte reads `answerState.done` for the spinner, elapsed interval
// and follow-up input; formatTurnStatus is its status-text source.
import { describe, expect, it } from 'vitest';
import { AnswerStream } from './answerStream.svelte';
import { formatTurnStatus } from './status';
import type { AnswerEvent } from '../types';

/** Deterministically run pending microtasks — enough ticks for consume() to
 *  pull and apply everything the fake generator has yielded so far, with no
 *  wall-clock time involved. */
async function drainMicrotasks(): Promise<void> {
	for (let i = 0; i < 50; i++) await Promise.resolve();
}

/** Yields `pre`, then stays open until `signal` aborts — mirrors how
 *  fetchEventStream swallows AbortError and just ends the generator when the
 *  session's controller fires (the real AnswerSession.abort() pairing). */
async function* hangableStream(
	pre: AnswerEvent[],
	signal: AbortSignal
): AsyncGenerator<AnswerEvent> {
	for (const event of pre) {
		yield event;
		if (signal.aborted) return;
	}
	const { promise, resolve } = Promise.withResolvers<void>();
	signal.addEventListener('abort', () => resolve(), { once: true });
	await promise;
}

describe('AnswerStream.abort', () => {
	it('marks the turn terminal on abort and keeps the partial answer', async () => {
		const ac = new AbortController();
		const stream = new AnswerStream();
		const consumed = stream.consume(
			hangableStream(
				[
					{ event: 'status', data: { phase: 'generating', detail: '' } },
					{ event: 'token', data: { text: 'Partial ans' } }
				],
				ac.signal
			)
		);
		await drainMicrotasks();

		// Precondition: mid-stream the turn really is non-terminal ("Generating…" + spinner).
		expect(stream.state.done).toBe(false);
		expect(formatTurnStatus(stream.state)).toBe('Generating…');

		// The exact wiring of SearchPage.setMode → session.abort().
		ac.abort();
		stream.stop();
		await consumed;

		expect(stream.state.done).toBe(true);
		expect(stream.state.error).toBeNull();
		expect(stream.state.text).toBe('Partial ans');
		// Spinner gone, elapsed counter stopped, follow-up input re-enabled,
		// and the status pill reads as finished rather than in-flight.
		expect(formatTurnStatus(stream.state)).not.toContain('Generating');
	});

	it('ignores events that arrive after stop()', async () => {
		const gate = Promise.withResolvers<void>();
		async function* gatedStream(): AsyncGenerator<AnswerEvent> {
			yield { event: 'token', data: { text: 'kept' } };
			await gate.promise;
			yield { event: 'token', data: { text: ' dropped' } };
		}

		const stream = new AnswerStream();
		const consumed = stream.consume(gatedStream());
		await drainMicrotasks();

		stream.stop();
		gate.resolve();
		await consumed;

		expect(stream.state.text).toBe('kept');
		expect(stream.state.done).toBe(true);
	});

	it('is safe on a stream that was never started', () => {
		const stream = new AnswerStream();
		stream.stop();
		expect(stream.state.done).toBe(true);
	});
});
