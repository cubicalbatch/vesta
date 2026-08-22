<script lang="ts">
	// The variant pill (maxi / nopic / mini) for an archive — the same Kiwix
	// "flavour" shown on both the download-search card and the installed row, so
	// the look stays in one place. Renders nothing when the archive has no
	// flavour (~45% of catalog entries — non-MediaWiki scrapers carry none).
	import { flavourDescription } from '$lib/flavour';

	let { flavour }: { flavour: string | null } = $props();

	const description = $derived(flavour ? flavourDescription(flavour) : null);
</script>

{#if flavour}
	<span
		class="shrink-0 rounded-full border border-border px-2 py-0.5 text-xs text-faint"
		title={description ?? undefined}
	>
		{flavour}
	</span>
{/if}
