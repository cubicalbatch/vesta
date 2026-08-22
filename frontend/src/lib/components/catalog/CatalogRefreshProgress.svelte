<script lang="ts">
	// Progress banner for an in-flight catalog refresh. The refresh_catalog job
	// is a single OPDS fetch reported as progress 0/1 → 1/1 (no byte-level
	// progress), so the bar is indeterminate while the fetch runs. `firstDownload`
	// swaps the copy to the "Downloading catalog… Please wait" first-run message
	// that makes an empty-cache auto-download legible instead of a blank page.
	let {
		firstDownload,
		message = null
	}: { firstDownload: boolean; message?: string | null } = $props();
</script>

<div class="rounded-lg border border-accent/30 bg-accent-soft p-4">
	<div class="mb-2 text-sm font-medium text-accent-soft-text">
		{firstDownload ? 'Downloading catalog…' : 'Refreshing catalog…'}
	</div>
	<div class="catalog-bar-indeterminate relative h-1.5 w-full overflow-hidden rounded-full bg-surface-muted">
		<span></span>
	</div>
	<p class="mt-2 text-xs text-muted">
		{#if firstDownload}
			Please wait — fetching the Kiwix archive list (a few seconds the first time).
		{:else}
			{message ?? 'Fetching the latest catalog from Kiwix.'}
		{/if}
	</p>
</div>
