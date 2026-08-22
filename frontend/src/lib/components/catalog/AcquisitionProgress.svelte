<script lang="ts">
	// One progress element for the whole download -> index chain, even though
	// it's two backend jobs ("The
	// acquisition chain": "Surface the whole chain as ONE progress element").
	import type { PendingAcquisition } from '$lib/stores/acquisition.svelte';
	import { acquisitionManager } from '$lib/stores/acquisition.svelte';
	import { jobsStore } from '$lib/stores/jobs.svelte';
	import { zimsApi } from '$lib/api/zims';

	let { entry }: { entry: PendingAcquisition } = $props();

	const stage = $derived(acquisitionManager.stageFor(entry.name));
	const downloadJob = $derived(jobsStore.jobs.get(entry.downloadJobId));
	const indexJob = $derived(entry.indexJobId ? jobsStore.jobs.get(entry.indexJobId) : undefined);

	const pct = $derived.by(() => {
		if (stage === 'downloading') {
			const j = downloadJob;
			return j?.total ? Math.round(((j.progress ?? 0) / j.total) * 50) : 0;
		}
		if (stage === 'registering') return 50;
		if (stage === 'indexing') {
			const j = indexJob;
			const inner = j?.total ? (j.progress ?? 0) / j.total : 0;
			return 50 + Math.round(inner * 50);
		}
		return 100;
	});

	async function retryRegistration() {
		await zimsApi.scan();
		acquisitionManager.reconcile();
	}
</script>

<div class="rounded-lg border border-border bg-surface p-3">
	<div class="mb-2 flex items-center justify-between text-sm">
		<span class="font-medium text-ink">{entry.name}</span>
		<span class="text-xs text-faint capitalize">{stage}</span>
	</div>
	<div class="h-1.5 overflow-hidden rounded-full bg-surface-muted">
		<div class="h-full rounded-full bg-accent transition-all" style="width: {pct}%"></div>
	</div>
	<div class="mt-1 flex items-center justify-between text-xs text-faint">
		<span>Downloading → Indexing → Ready</span>
		{#if stage === 'failed'}
			<span class="flex items-center gap-2 text-danger">
				registration failed
				<button type="button" class="underline" onclick={retryRegistration}>Scan for files</button>
			</span>
		{/if}
	</div>
</div>
