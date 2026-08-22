<script lang="ts">
	// One row of the secondary checkbox list on /welcome step 1 (curated ranks
	// 10-16). The whole row is a <label>, so the checkbox, title, and description
	// are all one click target. Like FeaturedCard, the matched CatalogEntry wins
	// for size/article count/flavour; the curated entry fills in when the catalog
	// feed is unreachable.
	import type { CatalogEntry, CuratedEntry } from '$lib/types';
	import { formatBytes, formatCount } from '$lib/format';
	import FlavourPill from '$lib/components/catalog/FlavourPill.svelte';

	let {
		curated,
		entry,
		checked,
		onchange
	}: {
		curated: CuratedEntry;
		entry?: CatalogEntry;
		checked: boolean;
		onchange: (checked: boolean) => void;
	} = $props();

	const title = $derived(entry?.title || humanizeName(curated.name));
	const sizeLabel = $derived(entry ? formatBytes(entry.size_bytes) : curated.size_note);
	const count = $derived(entry?.article_count ?? curated.article_count);
	const flavour = $derived(entry?.flavour ?? flavourFromKey(curated.name));

	const FLAVOURS = new Set(['nopic', 'maxi', 'mini', 'all']);

	function flavourFromKey(key: string): string {
		const parts = key.split('_');
		const last = parts[parts.length - 1];
		return FLAVOURS.has(last) ? last : '';
	}

	const KNOWN: Record<string, string> = {
		wikipedia: 'Wikipedia',
		wikivoyage: 'Wikivoyage',
		mdwiki: 'MDWiki',
		appropedia: 'Appropedia'
	};
	function humanizeName(key: string): string {
		const parts = key.split('_').filter(Boolean);
		if (parts.length > 1 && new Set([...FLAVOURS, 'medicines']).has(parts[parts.length - 1])) {
			parts.pop();
		}
		return parts
			.filter((p) => !/^[a-z]{2}$/.test(p))
			.map((p) => KNOWN[p] ?? (p.charAt(0).toUpperCase() + p.slice(1)))
			.join(' ');
	}
</script>

<label
	class="flex cursor-pointer items-start gap-3 rounded-lg border p-3 transition-colors {checked
		? 'border-accent/40 bg-accent-soft'
		: 'border-border bg-surface hover:bg-surface-muted'}"
>
	<input
		type="checkbox"
		class="mt-0.5 size-4 shrink-0 accent-[var(--color-accent)]"
		{checked}
		onchange={(e) => onchange((e.currentTarget as HTMLInputElement).checked)}
	/>
	<div class="min-w-0 flex-1">
		<div class="flex flex-wrap items-center gap-2">
			<span class="text-sm font-medium text-ink">{title}</span>
			<FlavourPill {flavour} />
		</div>
		<p class="text-xs text-muted">{curated.description}</p>
		<div class="mt-0.5 text-xs text-faint">{sizeLabel} · {formatCount(count)} articles</div>
		{#if curated.warning}
			<p class="mt-0.5 text-xs text-warning">{curated.warning}</p>
		{/if}
	</div>
</label>
