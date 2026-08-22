<script lang="ts">
	// "Worth adding" browse card + the install cost line — the most valuable
	// single element on the page
	// "The install cost line").
	import type { CatalogEntry } from '$lib/types';
	import { formatBytes, formatDuration, formatZimDate } from '$lib/format';
	import { formatLanguageNames } from '$lib/languages';
	import { diskBytesForDepth, preDownloadIndexTimeRange } from '$lib/install-cost';
	import { medianDownloadRate } from '$lib/download-rate';
	import { settingsValuesStore } from '$lib/stores/settings.svelte';
	import { acquisitionManager } from '$lib/stores/acquisition.svelte';
	import FlavourPill from './FlavourPill.svelte';

	let { entry, installed }: { entry: CatalogEntry; installed: boolean } = $props();

	const depth = $derived(Math.max(1, Math.min(3, Number(settingsValuesStore.values['index.default_depth'] ?? 3))));
	const downloadRate = $derived(medianDownloadRate());
	const downloadSeconds = $derived(downloadRate ? entry.size_bytes / downloadRate : null);
	const indexRange = $derived(preDownloadIndexTimeRange(entry.article_count, depth));
	const diskNeeded = $derived(diskBytesForDepth(entry.size_bytes, entry.article_count, depth));
	const expensive = $derived(entry.size_bytes > 20 * 1024 * 1024 * 1024); // 20 GB — amber past this

	let starting = $state(false);
	let error = $state<string | null>(null);

	async function install() {
		starting = true;
		error = null;
		try {
			await acquisitionManager.start(entry.id, entry.name);
		} catch (err) {
			error = err instanceof Error ? err.message : 'failed to start download';
		} finally {
			starting = false;
		}
	}
</script>

<div class="flex flex-col gap-2 rounded-lg border border-border bg-surface p-4">
	<div class="flex items-start gap-2">
		<div class="min-w-0 flex-1">
			<div class="font-semibold text-ink">{entry.title}</div>
			<div class="mt-0.5 line-clamp-2 text-sm text-muted">{entry.description}</div>
		</div>
		{#if entry.flavour || entry.language}
			<div class="flex shrink-0 flex-wrap items-center justify-end gap-1">
				<FlavourPill flavour={entry.flavour} />
				{#if entry.language}
					<span class="rounded-full bg-surface-muted px-2 py-0.5 text-xs text-muted" title={entry.language}>
						{formatLanguageNames(entry.language)}
					</span>
				{/if}
			</div>
		{/if}
	</div>

	{#if entry.curated_warning}
		<div class="rounded-md bg-warning-soft px-2 py-1.5 text-xs text-warning">{entry.curated_warning}</div>
	{/if}

	<div class="rounded-md p-2 text-xs {expensive ? 'bg-warning-soft text-warning' : 'bg-surface-muted text-muted'}">
		<div class="flex flex-wrap gap-x-4 gap-y-1">
			<span>download {formatBytes(entry.size_bytes)}{downloadSeconds ? ` · ~${formatDuration(downloadSeconds)}` : ''}</span>
			<span>index ~{formatDuration(indexRange.low)}–{formatDuration(indexRange.high)}</span>
			<span>disk ~{formatBytes(diskNeeded)}</span>
		</div>
	</div>

	{#if error}<p class="text-xs text-danger">{error}</p>{/if}

	<div class="mt-1 flex items-center justify-between">
		<span class="text-xs text-faint">~{entry.article_count.toLocaleString()} articles{entry.zim_date ? ` · ${formatZimDate(entry.zim_date)}` : ''}</span>
		{#if installed}
			<span class="text-xs font-medium text-success">Installed</span>
		{:else if acquisitionManager.stageFor(entry.name)}
			<span class="text-xs font-medium text-accent">{acquisitionManager.stageFor(entry.name)}…</span>
		{:else}
			<button type="button" class="rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-white hover:bg-accent-hover disabled:opacity-40" disabled={starting} onclick={install}>
				Download
			</button>
		{/if}
	</div>
</div>
