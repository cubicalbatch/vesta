<script lang="ts">
	// The sources-mode result region, lifted verbatim from the old
	// routes/+page.svelte: count header, source cards, the mid-index honesty
	// note, and the trace disclosure
	// (results view).
	import type { SearchResponse } from '$lib/api/search';
	import type { SourceCard as SourceCardT } from '$lib/types';
	import SourceCard from '$lib/components/SourceCard.svelte';
	import TraceView from '$lib/components/TraceView.svelte';

	let {
		result,
		loading,
		error,
		lastQuery,
		midIndexArchive,
		midIndexPct,
		onOpenCard
	}: {
		result: SearchResponse | null;
		loading: boolean;
		error: string | null;
		lastQuery: string;
		midIndexArchive: { id: number; corpus_label: string | null; title: string | null } | null;
		midIndexPct: number | null;
		onOpenCard: (card: SourceCardT, i: number) => void;
	} = $props();
</script>

{#if loading}
	<div class="py-10 text-center text-sm text-muted">Searching…</div>
{:else if error}
	<div class="rounded-lg border border-danger/30 bg-danger-soft p-4 text-sm text-danger">{error}</div>
{:else if result}
	<section class="mt-5">
		<div class="mb-4 flex items-baseline justify-between gap-4">
			<h2 class="text-xl font-semibold tracking-tight text-ink">{result.cards.length} passage{result.cards.length === 1 ? '' : 's'}</h2>
			<span class="text-sm text-muted">
				{result.profile}
				{#if result.trace.stages.length}· {result.trace.stages.reduce((s, st) => s + st.duration_ms, 0).toFixed(0)}ms{/if}
			</span>
		</div>

		{#if result.cards.length === 0}
			<div class="rounded-lg border border-border bg-surface p-6 text-center">
				<p class="mb-1 text-sm text-ink">Nothing in your library matched "{lastQuery}".</p>
				<p class="text-sm text-muted">Try narrowing the scope, different wording, or adding an archive from the <a href="/catalog" class="text-accent underline">Catalog</a>.</p>
			</div>
		{:else}
			<div class="flex flex-col gap-3">
				{#each result.cards as card, i (i)}
				<SourceCard {card} rank={i + 1} queryForHighlight={lastQuery} onOpen={() => onOpenCard(card, i)} />
				{/each}
			</div>
		{/if}

		{#if midIndexArchive}
			<p class="mt-6 text-xs text-faint">
				{midIndexArchive.corpus_label ?? midIndexArchive.title}
				{midIndexPct != null ? `is ${midIndexPct}% indexed` : 'is still indexing'} - results from it are keyword-only until that finishes.
				<a href="/settings?tab=jobs" class="text-accent underline">See progress</a>
			</p>
		{/if}

		<details class="mt-6 border-t border-border pt-4">
			<summary class="cursor-pointer select-none text-sm font-medium text-muted hover:text-ink">How this search ran ▾</summary>
			<div class="mt-4 rounded-md border border-border bg-surface-sunken p-4">
				<TraceView trace={result.trace} />
			</div>
		</details>
	</section>
{/if}
