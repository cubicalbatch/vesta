<script lang="ts">
	// One turn of an Ask conversation: query echo, status line, sources
	// (one-list-two-homes — CSS repositions it, single markup path), the
	// streaming answer, answer meta, trace disclosure, follow-ups
	// (Render order).
	import type { AnswerState } from '$lib/answer/reducer';
	import { formatTurnStatus } from '$lib/answer/status';
	import type { SourceCard as SourceCardT } from '$lib/types';
	import { provideSources } from '$lib/stores/sources-context.svelte';
	import { readerStore } from '$lib/stores/reader.svelte';
	import { modelStore } from '$lib/stores/model.svelte';
	import Markdown from '$lib/markdown/Markdown.svelte';
	import SourceCard from '$lib/components/SourceCard.svelte';
	import TraceView from '$lib/components/TraceView.svelte';
	import ChevronDown from '@lucide/svelte/icons/chevron-down';
	import Copy from '@lucide/svelte/icons/copy';
	import RotateCw from '@lucide/svelte/icons/rotate-cw';

	let {
		query,
		answerState,
		live = false,
		collapsedByDefault = false,
		onRegenerate,
		onFollowUp
	}: {
		query: string;
		answerState: AnswerState;
		live?: boolean;
		collapsedByDefault?: boolean;
		onRegenerate?: () => void;
		onFollowUp?: (q: string) => void;
	} = $props();

	const sources = provideSources();
	let sourcesEl = $state<HTMLElement | null>(null);
	let sourcesCollapsed = $state(collapsedByDefault);

	$effect(() => {
		sources.list = answerState.sources;
		sources.citations = answerState.citations;
	});

	$effect(() => {
		// Focus token changes on every click, even re-clicking the same chip.
		sources.focusToken;
		const id = sources.focused;
		if (id == null || !sourcesEl) return;
		const el = sourcesEl.querySelector<HTMLElement>(`[data-card-slot="${id}"]`);
		el?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
		el?.animate(
			[{ boxShadow: '0 0 0 3px var(--accent-ring)' }, { boxShadow: '0 0 0 0 transparent' }],
			{ duration: 900 }
		);
		const card = answerState.sources[id];
		if (card) {
			const span = sources.spanForCard(id);
			readerStore.open({
				zimId: card.zim_id,
				path: card.path,
				title: card.title,
				cards: answerState.sources,
				cardIndex: id,
				passageSpan: span?.passage_span ?? null,
				citedAs: id + 1
			});
		}
	});

	function openCard(i: number) {
		sources.focus(i);
	}

	// Elapsed counter — truthful cover for the CPU prefill gap (10-25s).
	let elapsedMs = $state(0);
	$effect(() => {
		if (answerState.done) return;
		const start = Date.now();
		const id = setInterval(() => (elapsedMs = Date.now() - start), 500);
		return () => clearInterval(id);
	});

	const statusText = $derived(formatTurnStatus(answerState));
	const model = $derived(
		modelStore.status?.display_name || modelStore.status?.model_file || 'model'
	);
	const approxTokens = $derived(
		Math.round(answerState.text.split(/\s+/).filter(Boolean).length * 1.3)
	);

	function copyAnswer() {
		navigator.clipboard?.writeText(answerState.text);
	}

	let followUpValue = $state('');
	function submitFollowUp(e: SubmitEvent) {
		e.preventDefault();
		if (!followUpValue.trim() || !onFollowUp) return;
		onFollowUp(followUpValue.trim());
		followUpValue = '';
	}
</script>

<!--
  grid-rows-[auto_1fr]: the sources aside spans both rows (HEAD + BODY). With
  both rows left `auto`, the still-empty BODY row can't yet supply the
  aside's height on its own, so the grid inflates the HEAD row to help —
  then shrinks it back as the answer streams in and BODY grows, making the
  answer's top edge visibly climb upward. Marking BODY `1fr` excludes it from
  that spanning-item redistribution, so only its own content sizes it.
-->
<div
	class="mx-auto grid max-w-[var(--wide-max)] gap-x-8 min-[1180px]:grid-cols-[minmax(0,var(--content-max))_var(--rail-w)] min-[1180px]:grid-rows-[auto_1fr]"
>
	<!-- HEAD -->
	<div class="min-w-0 min-[1180px]:[grid-area:1/1]">
		<div class="mb-4">
			<div class="mb-1 text-xs font-semibold uppercase tracking-wide text-faint">You asked</div>
			<h1 class="font-display text-2xl font-bold tracking-tight text-ink">{query}</h1>
		</div>
		<span class="mb-5 inline-flex items-center gap-2 rounded-full bg-surface-muted px-3 py-1 text-sm text-muted">
			{#if !answerState.done && !answerState.error}
				<span class="size-3 shrink-0 animate-spin rounded-full border-2 border-border-strong border-t-accent"></span>
			{/if}
			{statusText}
			{#if !answerState.done && elapsedMs > 3000}<span class="font-mono text-xs">· {(elapsedMs / 1000).toFixed(0)}s</span>{/if}
		</span>
	</div>

	<!--
	  min-w-0: without it, this grid item's automatic minimum width is its
	  content's max-content size — and the mobile horizontal card strip below
	  is an unwrapped flex row, so that max-content size is the sum of every
	  card's width. HEAD/SOURCES/BODY share one grid column below 1180px, so
	  that inflated minimum stretches the whole page, not just this strip.
	-->
	<!-- SOURCES: one list, CSS repositions it -->
	<aside
		bind:this={sourcesEl}
		class="mb-5 min-w-0 min-[1180px]:sticky min-[1180px]:top-[calc(var(--topbar-h)+1.25rem)] min-[1180px]:mb-0 min-[1180px]:max-h-[calc(100vh-var(--topbar-h)-2.5rem)] min-[1180px]:overflow-y-auto min-[1180px]:[grid-area:1/2/3/2]"
		aria-label="Sources used for this answer"
	>
		{#if answerState.sources.length > 0}
			{#if collapsedByDefault}
				<button
					type="button"
					class="mb-2 flex items-center gap-1 text-xs font-semibold uppercase tracking-wide text-faint"
					onclick={() => (sourcesCollapsed = !sourcesCollapsed)}
				>
					Sources ({answerState.sources.length})
					<ChevronDown class="size-3 transition-transform {sourcesCollapsed ? '' : 'rotate-180'}" />
				</button>
			{:else}
				<div class="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-faint">
					<span class="rounded-full bg-accent-soft px-1.5 py-px text-accent-soft-text">{answerState.sources.length}</span>
					Sources
				</div>
			{/if}
			{#if !sourcesCollapsed}
				<div
					class="grid grid-cols-1 gap-3 min-[721px]:grid-cols-3 min-[1180px]:grid-cols-1 max-[720px]:flex max-[720px]:snap-x max-[720px]:overflow-x-auto max-[720px]:pb-1"
				>
					{#each answerState.sources as card, i (i)}
						<div data-card-slot={i} class="max-[720px]:w-64 max-[720px]:shrink-0 max-[720px]:snap-start">
							<SourceCard {card} rank={i + 1} compact onOpen={() => openCard(i)} />
						</div>
					{/each}
				</div>
			{/if}
		{:else if !answerState.done}
			<div class="grid grid-cols-1 gap-3 min-[721px]:grid-cols-3 min-[1180px]:grid-cols-1">
				{#each Array(3) as _, i (i)}
					<div class="h-24 animate-pulse rounded-lg bg-surface-muted"></div>
				{/each}
			</div>
		{/if}
	</aside>

	<!-- BODY -->
	<div class="min-w-0 min-[1180px]:[grid-area:2/1]">
		{#if answerState.error}
			<div class="mb-4 rounded-lg border border-danger/30 bg-danger-soft p-3 text-sm text-danger">
				{#if answerState.error.code === 'budget_exhausted'}
					The model spent its whole token budget before producing an answer. Try
					<a href="/settings" class="underline">disabling extended thinking or raising the output token budget</a>
					in Settings.
				{:else}
					{answerState.error.message}
				{/if}
				{#if answerState.error.recoverable}
					<span class="block text-xs text-muted">The connection recovers automatically - your next question will work.</span>
				{/if}
			</div>
		{/if}

		{#if answerState.phase === 'abstaining'}
			<div class="mb-2 rounded-md bg-surface-muted px-3 py-1 text-xs text-muted">Nothing in your library closely covers this - see the sources below.</div>
		{/if}

		<Markdown text={answerState.text} done={answerState.done} />

		{#if answerState.done && !answerState.error}
			<div class="mt-5 flex flex-wrap items-center gap-3 font-mono text-xs text-faint">
				<span>{model}</span>
				{#if answerState.citations.length > 0}<span>{answerState.citations.length} citations</span>{/if}
				<div class="ml-auto flex gap-1">
					<button type="button" class="inline-grid size-7 place-items-center rounded-md hover:bg-surface-muted" onclick={copyAnswer} title="Copy answer">
						<Copy class="size-3.5" />
					</button>
					{#if live && onRegenerate}
						<button type="button" class="inline-grid size-7 place-items-center rounded-md hover:bg-surface-muted" onclick={onRegenerate} title="Regenerate">
							<RotateCw class="size-3.5" />
						</button>
					{/if}
				</div>
			</div>
		{/if}

		{#if answerState.trace}
			<details class="mt-6 border-t border-border pt-4">
				<summary class="cursor-pointer select-none text-sm font-medium text-muted hover:text-ink">How this answer was built ▾</summary>
				<div class="mt-4 rounded-md border border-border bg-surface-sunken p-4">
					<TraceView trace={answerState.trace} />
				</div>
			</details>
		{/if}

		{#if live && onFollowUp}
			<form class="mt-6 flex gap-2" onsubmit={submitFollowUp}>
				<input
					type="text"
					class="flex-1 rounded-lg border border-border bg-surface px-4 py-2.5 text-sm outline-none focus:border-accent"
					placeholder="Ask a follow-up…"
					bind:value={followUpValue}
					disabled={!answerState.done}
				/>
				<button
					type="submit"
					class="rounded-lg bg-accent px-4 py-2.5 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-40"
					disabled={!answerState.done || !followUpValue.trim()}
				>
					Ask
				</button>
			</form>
		{/if}
	</div>
</div>
