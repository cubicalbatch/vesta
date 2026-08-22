<script lang="ts">
	// One card per non-terminal job.
	// Never derives job state from a control-call response beyond ack/409 —
	// the global SSE stream (jobsStore) is the only source of truth for what
	// the job is actually doing (gotcha #9 in the handoff: classify from the
	// stream, not from a one-off response).
	import type { JobRecord } from '$lib/types';
	import { zimsStore } from '$lib/stores/zims.svelte';
	import { jobLabel } from '$lib/job-label';
	import { formatBytes, formatDuration } from '$lib/format';
	import { jobsApi } from '$lib/api/jobs';
	import JobStatusBadge from './JobStatusBadge.svelte';

	let { job }: { job: JobRecord } = $props();

	let busyAction = $state<'pause' | 'resume' | 'cancel' | null>(null);
	let actionError = $state<string | null>(null);

	const label = $derived(jobLabel(job, zimsStore.archives));
	const pct = $derived(job.total ? Math.min(100, Math.max(0, ((job.progress ?? 0) / job.total) * 100)) : null);

	const rateLabel = $derived.by(() => {
		if (job.rate == null) return null;
		return job.type === 'download_zim' ? `${formatBytes(job.rate)}/s` : `${job.rate.toLocaleString(undefined, { maximumFractionDigits: 1 })}/s`;
	});

	async function act(action: 'pause' | 'resume' | 'cancel') {
		busyAction = action;
		actionError = null;
		try {
			const res = await (action === 'pause' ? jobsApi.pause(job.id) : action === 'resume' ? jobsApi.resume(job.id) : jobsApi.cancel(job.id));
			// A 409 means someone else already changed the job's state — that's
			// success, not an error (treat 409 as success).
			// Either way the SSE stream (not this response) is what updates the UI.
			void res;
		} catch (err) {
			actionError = err instanceof Error ? err.message : `failed to ${action} job`;
		} finally {
			busyAction = null;
		}
	}
</script>

<div class="rounded-lg border border-border bg-surface p-4">
	<div class="flex flex-wrap items-start justify-between gap-3">
		<div class="min-w-0">
			<div class="font-medium text-ink">{label}</div>
			<div class="mt-1 flex flex-wrap items-center gap-2 text-xs text-faint">
				<JobStatusBadge status={job.status} />
				<span>job #{job.id}</span>
				{#if rateLabel}<span>· {rateLabel}</span>{/if}
				{#if job.eta_seconds != null}<span>· ETA {formatDuration(job.eta_seconds)}</span>{/if}
			</div>
		</div>
		<div class="flex shrink-0 gap-2">
			{#if job.status === 'running'}
				<button type="button" class="rounded-md border border-border px-2 py-1 text-xs font-medium text-ink-2 hover:bg-surface-muted disabled:opacity-40" disabled={busyAction !== null} onclick={() => act('pause')}>
					Pause
				</button>
			{:else if job.status === 'paused'}
				<button type="button" class="rounded-md border border-border px-2 py-1 text-xs font-medium text-ink-2 hover:bg-surface-muted disabled:opacity-40" disabled={busyAction !== null} onclick={() => act('resume')}>
					Resume
				</button>
			{/if}
			<button type="button" class="rounded-md border border-border px-2 py-1 text-xs font-medium text-danger hover:bg-danger-soft disabled:opacity-40" disabled={busyAction !== null} onclick={() => act('cancel')}>
				Cancel
			</button>
		</div>
	</div>

	{#if pct != null}
		<div class="mt-3">
			<div class="h-1.5 overflow-hidden rounded-full bg-surface-muted">
				<div class="h-full rounded-full bg-accent transition-all" style="width: {pct}%"></div>
			</div>
			<p class="mt-1 font-mono text-xs text-faint">
				{Math.round(pct)}% · {(job.progress ?? 0).toLocaleString()} / {(job.total ?? 0).toLocaleString()}
			</p>
		</div>
	{/if}

	{#if job.message}
		<p class="mt-2 truncate rounded-md bg-surface-sunken px-2 py-1 font-mono text-xs text-ink-2" title={job.message}>{job.message}</p>
	{/if}

	{#if actionError}<p class="mt-2 text-xs text-danger">{actionError}</p>{/if}
</div>
