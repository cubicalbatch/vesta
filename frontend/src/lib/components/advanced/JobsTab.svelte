<script lang="ts">
	// Jobs tab — default tab, destination of the header job dot
	// "Page: Advanced"). The global jobsStore is the only source of truth;
	// this component derives, it never fetches GET /api/jobs itself (that
	// would be a second, redundant source of the same data the SSE snapshot
	// already delivered).
	import { jobsStore } from '$lib/stores/jobs.svelte';
	import { zimsStore } from '$lib/stores/zims.svelte';
	import { jobLabel } from '$lib/job-label';
	import { TERMINAL_JOB_STATUSES } from '$lib/types';
	import { formatDuration } from '$lib/format';
	import RunningJobCard from './RunningJobCard.svelte';
	import JobStatusBadge from './JobStatusBadge.svelte';

	const recent = $derived(jobsStore.list.filter((j) => TERMINAL_JOB_STATUSES.has(j.status)));

	// Shares formatDuration with the ETA span in RunningJobCard so "5 min"-style
	// durations read the same convention everywhere on this page, instead of two
	// independent formatters (this one used to print "5m 30s").
	function duration(createdAt: string, finishedAt: string | null): string {
		if (!finishedAt) return '—';
		const ms = new Date(finishedAt).getTime() - new Date(createdAt).getTime();
		if (!Number.isFinite(ms) || ms < 0) return '—';
		return formatDuration(ms / 1000);
	}
</script>

<div>
	<div class="mb-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
		<h2 class="text-lg font-semibold text-ink">Running now</h2>
		<span class="text-xs text-faint">state lives on the server — safe to reload or close the tab</span>
	</div>

	{#if !jobsStore.connected}
		<p class="text-sm text-muted">Connecting to the job stream…</p>
	{:else if jobsStore.running.length === 0}
		<p class="text-sm text-muted">Nothing running.</p>
	{:else}
		<div class="flex flex-col gap-3">
			{#each jobsStore.running as job (job.id)}
				<RunningJobCard {job} />
			{/each}
		</div>
	{/if}

	<div class="mb-3 mt-8 flex items-center justify-between">
		<h2 class="text-lg font-semibold text-ink">Recent</h2>
		<span class="text-xs text-faint">{recent.length} finished</span>
	</div>

	{#if recent.length === 0}
		<p class="text-sm text-muted">No finished jobs yet.</p>
	{:else}
		<div class="max-h-[480px] overflow-y-auto overflow-x-auto rounded-lg border border-border">
			<table class="w-full min-w-[640px] text-left text-sm">
				<thead class="sticky top-0 bg-surface-muted text-xs uppercase tracking-wide text-faint">
					<tr>
						<th class="px-3 py-2 font-medium">Job</th>
						<th class="px-3 py-2 font-medium">Type</th>
						<th class="px-3 py-2 font-medium">Status</th>
						<th class="px-3 py-2 font-medium">Took</th>
						<th class="px-3 py-2 font-medium">Finished</th>
						<th class="px-3 py-2 font-medium">Detail</th>
					</tr>
				</thead>
				<tbody class="divide-y divide-border">
					{#each recent as job (job.id)}
						<tr>
							<td class="px-3 py-2 text-ink">{jobLabel(job, zimsStore.archives)} <span class="text-faint">#{job.id}</span></td>
							<td class="px-3 py-2 font-mono text-xs text-ink-2">{job.type}</td>
							<td class="px-3 py-2"><JobStatusBadge status={job.status} /></td>
							<td class="px-3 py-2 font-mono text-xs text-ink-2">{duration(job.created_at, job.finished_at)}</td>
							<td class="px-3 py-2 text-xs text-faint">{job.finished_at ? new Date(job.finished_at).toLocaleString() : '—'}</td>
							<td class="max-w-[280px] truncate px-3 py-2 text-xs text-faint" title={job.error ?? job.message ?? ''}>
								{job.error ?? job.message ?? '—'}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</div>
