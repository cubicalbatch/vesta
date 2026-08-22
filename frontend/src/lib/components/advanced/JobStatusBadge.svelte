<script lang="ts">
	// Shared status pill for jobs, eval runs and benchmark runs — same six
	// JobStatus values plus the free-form "running"/"done"/"error" strings the
	// eval/benchmark rows carry (which are a subset of the same vocabulary).
	let { status }: { status: string } = $props();

	const styleFor = (s: string): string => {
		if (s === 'running' || s === 'queued') return 'bg-warning-soft text-warning';
		if (s === 'paused') return 'bg-surface-muted text-muted';
		if (s === 'done') return 'bg-success-soft text-success';
		if (s === 'error' || s === 'cancelled' || s === 'aborted') return 'bg-danger-soft text-danger';
		return 'bg-surface-muted text-muted';
	};
</script>

<span class="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium capitalize {styleFor(status)}">
	<span class="size-1.5 rounded-full bg-current"></span>
	{status || 'unknown'}
</span>
