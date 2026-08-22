<script lang="ts">
	import type { SourceCard as SourceCardT } from '$lib/types';
	import { highlightSegments } from '$lib/highlight';
	import { zimsStore } from '$lib/stores/zims.svelte';

	let {
		card,
		rank,
		compact = false,
		highlighted = false,
		queryForHighlight = '',
		onOpen
	}: {
		card: SourceCardT;
		rank: number;
		compact?: boolean;
		highlighted?: boolean;
		queryForHighlight?: string;
		onOpen: () => void;
	} = $props();

	const archive = $derived(zimsStore.archives.find((a) => a.id === card.zim_id));
	const archiveLabel = $derived(archive?.corpus_label ?? archive?.title ?? archive?.name ?? 'archive');
	const segments = $derived(
		queryForHighlight ? highlightSegments(card.snippet, queryForHighlight) : [{ text: card.snippet, mark: false }]
	);
</script>

<div
	class="w-full rounded-lg border bg-surface text-left shadow-xs transition-colors hover:border-border-strong hover:shadow-sm {compact
		? 'p-3'
		: 'p-4'} {highlighted ? 'border-accent/45 shadow-[0_0_0_3px_var(--accent-ring)]' : 'border-border'}"
>
	<button type="button" onclick={onOpen} class="block w-full cursor-pointer text-left">
		{#if card.media?.poster_path}
			<img
				src={`/api/zim/${card.zim_id}/${card.media.poster_path}`}
				alt=""
				class="mb-3 size-full max-h-40 w-full rounded-md object-cover"
				loading="lazy"
			/>
		{/if}
		<span class="mb-2 flex items-center gap-3">
			<span
				class="inline-grid size-[22px] shrink-0 place-items-center rounded-full bg-surface-muted text-xs font-bold text-muted"
				>{rank}</span
			>
			<span class="min-w-0 flex-1 truncate font-semibold tracking-tight text-ink {compact ? 'text-sm' : 'text-base'}"
				>{card.title}</span
			>
			{#if card.recovered}
				<span class="shrink-0 rounded-full bg-accent-soft px-1.5 py-0.5 text-[10px] font-medium text-accent-soft-text"
					>recovered</span
				>
			{/if}
		</span>
		{#if card.breadcrumb && card.breadcrumb !== card.title}
			<span class="mb-2 block truncate text-xs text-faint">{card.breadcrumb}</span>
		{/if}
		<!-- `block` only in the unclamped case: `line-clamp-3` supplies its own
		     `display: -webkit-box`, and a competing `block` utility silently
		     defeats the clamp. -->
		<span class="text-muted {compact ? 'line-clamp-3 text-sm' : 'block text-sm'} leading-snug">
			{#each segments as seg, i (i)}
				{#if seg.mark}<mark class="rounded-sm bg-accent-soft px-0.5 text-accent-soft-text">{seg.text}</mark
					>{:else}{seg.text}{/if}
			{/each}
		</span>
	</button>
	<div class="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-xs text-faint">
		<span class="rounded-full bg-surface-muted px-2 py-0.5">{archiveLabel}</span>
		<div class="ml-auto flex shrink-0 items-center gap-2">
			<span class="h-[5px] w-[60px] overflow-hidden rounded-full bg-score-track">
				<span class="block h-full rounded-full bg-score-fill" style="width: {Math.min(100, Math.max(0, Math.round(card.score * 100)))}%"></span>
			</span>
			<span class="min-w-8 font-mono text-xs text-faint">{card.score.toFixed(2)}</span>
		</div>
	</div>
</div>
