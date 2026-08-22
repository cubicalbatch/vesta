<script lang="ts">
	import { useSources } from '../stores/sources-context.svelte';

	let { ids }: { ids: number[] } = $props();

	const sources = useSources();
	// ids are 1-based card numbers ([n] = card_id + 1) once citations.answer_text
	// has rewritten them; during live streaming they may transiently be raw
	// passage numbers until the citations event lands — that's fine, the chip
	// just shows a pending state until its card exists.
	const resolved = $derived(ids.map((id) => sources.list[id - 1]));
</script>

{#each ids as id, k (id)}
	{@const src = resolved[k]}
	{@const weak = sources.isWeak(id - 1)}
	<button
		type="button"
		class="citation-chip align-super mx-0.5 -translate-y-px rounded-sm border px-1 text-[11px] font-bold leading-none transition-colors data-[pending=true]:opacity-40 {weak
			? 'border-warning/25 bg-warning-soft text-warning'
			: 'border-accent/22 bg-accent-soft text-accent-soft-text hover:bg-accent hover:text-white'}"
		data-pending={!src}
		data-card-id={id - 1}
		title={src
			? weak
				? `${src.title} — weakly supported`
				: src.title
			: 'source not yet loaded'}
		onclick={() => src && sources.focus(id - 1)}
	>
		{id}
	</button>
{/each}
