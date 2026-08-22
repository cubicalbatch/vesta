<script lang="ts">
	// One of the two big featured picks on /welcome step 1 ("Fastest start" +
	// "Most useful"). A large, whole-card toggle: clicking it flips membership in
	// the page's `selectedKeys` set. The matched CatalogEntry supplies the real
	// size/article count when the live catalog is reachable; otherwise the
	// curated entry's own size_note/article_count render so an offline first boot
	// still shows a complete card (curated ships in-repo — "catalog outage
	// ≠ degraded first-run").
	import type { CatalogEntry, CuratedEntry } from '$lib/types';
	import { formatBytes, formatCount } from '$lib/format';
	import Check from '@lucide/svelte/icons/check';

	let {
		tag,
		subtitle,
		curated,
		entry,
		selected,
		onclick
	}: {
		tag: string;
		subtitle: string;
		curated: CuratedEntry;
		entry?: CatalogEntry;
		selected: boolean;
		onclick: () => void;
	} = $props();

	// Prefer the matched catalog row's title (authoritative Kiwix title); fall
	// back to a humanized read of the ZIM filename stem when the catalog feed is
	// unreachable and no row matched.
	const title = $derived(entry?.title || humanizeName(curated.name));
	const sizeLabel = $derived(entry ? formatBytes(entry.size_bytes) : curated.size_note);
	const count = $derived(entry?.article_count ?? curated.article_count);

	const KNOWN: Record<string, string> = {
		wikipedia: 'Wikipedia',
		wikivoyage: 'Wikivoyage',
		mdwiki: 'MDWiki',
		appropedia: 'Appropedia'
	};
	const FLAVOURS = new Set(['nopic', 'maxi', 'mini', 'all', 'medicines']);

	function humanizeName(key: string): string {
		const parts = key.split('_').filter(Boolean);
		if (parts.length > 1 && FLAVOURS.has(parts[parts.length - 1])) parts.pop();
		return parts
			.filter((p) => !/^[a-z]{2}$/.test(p)) // drop 2-letter language codes
			.map((p) => KNOWN[p] ?? (p.charAt(0).toUpperCase() + p.slice(1)))
			.join(' ');
	}
</script>

<button
	type="button"
	onclick={onclick}
	aria-pressed={selected}
	class="flex w-full flex-col gap-2.5 rounded-xl border p-5 text-left transition-colors {selected
		? 'border-accent bg-accent-soft ring-1 ring-[var(--accent-ring)]'
		: 'border-border bg-surface hover:bg-surface-muted'}"
>
	<span class="flex items-center justify-between gap-2">
		<span
			class="inline-block w-fit rounded-full bg-accent-soft px-2 py-0.5 text-xs font-medium text-accent-soft-text"
		>
			{tag}
		</span>
		{#if selected}
			<span class="inline-grid size-5 shrink-0 place-items-center rounded-full bg-accent text-white">
				<Check class="size-3.5" />
			</span>
		{/if}
	</span>

	<span class="block">
		<span class="block font-display text-xl font-bold tracking-tight text-ink">{title}</span>
		<span class="block text-sm text-muted">{subtitle}</span>
	</span>

	<span class="block text-sm text-muted">{curated.description}</span>

	{#if curated.warning}
		<span class="block text-xs text-warning">{curated.warning}</span>
	{/if}

	<span class="mt-1 block text-xs text-faint">{sizeLabel} · {formatCount(count)} articles</span>
</button>
