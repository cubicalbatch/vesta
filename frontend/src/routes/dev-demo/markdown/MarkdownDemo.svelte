<script lang="ts">
	// Demonstration harness (dev-only — the +page.svelte wrapper only mounts
	// this under import.meta.env.DEV): replays recorded SSE fixtures plus
	// synthetic streams for the three amendments that only fire on recovery
	// paths, through the real reducer + Markdown component. Not linked from the
	// app nav — a throwaway proving ground, not a product page.
	import { provideSources } from '$lib/stores/sources-context.svelte';
	import { AnswerStream } from '$lib/answer/answerStream.svelte';
	import Markdown from '$lib/markdown/Markdown.svelte';
	import type { AnswerEvent } from '$lib/types';

	import singleShotCited from '$lib/dev-fixtures/recorded/single_shot_cited.json';
	import sourcesOnly from '$lib/dev-fixtures/recorded/sources_only.json';
	import abstention from '$lib/dev-fixtures/recorded/abstention.json';
	import agenticRecovery from '$lib/dev-fixtures/recorded/agentic_recovery.json';
	import answerReset from '$lib/dev-fixtures/synthetic/answer_reset.json';
	import answerTextRewrite from '$lib/dev-fixtures/synthetic/answer_text_rewrite.json';
	import recoverableError from '$lib/dev-fixtures/synthetic/recoverable_error.json';

	const FIXTURES: Record<string, { description: string; query: string; events: AnswerEvent[] }> = {
		single_shot_cited: singleShotCited as never,
		sources_only: sourcesOnly as never,
		abstention: abstention as never,
		agentic_recovery: agenticRecovery as never,
		answer_reset: answerReset as never,
		answer_text_rewrite: answerTextRewrite as never,
		recoverable_error: recoverableError as never
	};

	const sources = provideSources();
	const stream = new AnswerStream(sources);

	let selected = $state('single_shot_cited');
	let speed = $state(40);

	function run() {
		stream.replay(FIXTURES[selected].events, speed);
	}

	function runInstant() {
		stream.replayInstant(FIXTURES[selected].events);
	}

	$effect(() => {
		run();
		return () => stream.stop();
	});

	// Debug hook for manual/automated QA of this throwaway harness only —
	// stripped from production builds.
	if (import.meta.env.DEV) {
		$effect(() => {
			(window as unknown as Record<string, unknown>).__markdownDemo = { stream, FIXTURES };
		});
	}
</script>

<div class="mx-auto flex max-w-3xl flex-col gap-6 p-8">
	<h1 class="text-xl font-semibold">Task 0 — streaming markdown demo</h1>

	<div class="flex flex-wrap items-center gap-3 rounded-lg border border-border bg-surface p-4">
		<label class="flex items-center gap-2 text-sm">
			Fixture
			<select
				class="rounded-md border border-border bg-surface px-2 py-1"
				bind:value={selected}
				onchange={run}
				data-testid="fixture-select"
			>
				{#each Object.keys(FIXTURES) as key (key)}
					<option value={key}>{key}</option>
				{/each}
			</select>
		</label>
		<label class="flex items-center gap-2 text-sm">
			Delay (ms/event)
			<input
				type="number"
				class="w-20 rounded-md border border-border bg-surface px-2 py-1"
				bind:value={speed}
			/>
		</label>
		<button
			type="button"
			class="rounded-md bg-accent px-3 py-1 text-sm font-medium text-white"
			onclick={run}
			data-testid="replay">Replay</button
		>
		<button
			type="button"
			class="rounded-md border border-border px-3 py-1 text-sm"
			onclick={runInstant}
			data-testid="replay-instant">Instant</button
		>
		<span class="text-xs text-muted">{FIXTURES[selected].description}</span>
	</div>

	<div class="grid grid-cols-[1fr_260px] gap-6">
		<div class="flex flex-col gap-3">
			<div class="text-sm text-muted" data-testid="status-line">
				{#if stream.state.error}
					<span class="text-danger">error: {stream.state.error.code} — {stream.state.error.message}</span>
				{:else if stream.state.phase}
					phase: {stream.state.phase} — {stream.state.detail}
				{:else}
					idle
				{/if}
				{#if stream.state.done}<span class="ml-2 text-success">done</span>{/if}
			</div>

			<div
				class="min-h-32 rounded-lg border border-border bg-surface p-4"
				data-testid="answer-body"
			>
				<Markdown text={stream.state.text} done={stream.state.done} />
			</div>

			{#if stream.state.citations.length}
				<div class="text-xs text-muted" data-testid="citations-json">
					citations: {JSON.stringify(stream.state.citations)}
				</div>
			{/if}
		</div>

		<div class="flex flex-col gap-2" data-testid="source-rail">
			<h2 class="text-xs font-semibold uppercase tracking-wide text-muted">
				Sources ({sources.list.length})
			</h2>
			{#each sources.list as card, i (i)}
				<div
					class="rounded-md border border-border bg-surface p-2 text-xs"
					class:border-accent={sources.focused === i}
					data-testid="source-card"
					data-card-id={i}
				>
					<div class="font-medium">{i + 1}. {card.title}</div>
					<div class="text-muted">{card.source}{card.recovered ? ' · recovered' : ''}</div>
				</div>
			{/each}
		</div>
	</div>
</div>
