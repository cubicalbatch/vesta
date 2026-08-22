<script lang="ts">
	// "On this machine" row.
	import type { Archive } from '$lib/types';
	import { depthLabel } from '$lib/index-depth';
	import { formatBytes, formatDuration } from '$lib/format';
	import { formatLanguageNames } from '$lib/languages';
	import FlavourPill from './FlavourPill.svelte';
	import { zimsApi } from '$lib/api/zims';
	import { zimsStore } from '$lib/stores/zims.svelte';
	import { settingsValuesStore } from '$lib/stores/settings.svelte';
	import { jobsStore } from '$lib/stores/jobs.svelte';
	import { lockBodyScroll } from '$lib/scroll-lock';
	import Trash2 from '@lucide/svelte/icons/trash-2';
	import Compass from '@lucide/svelte/icons/compass';

	let { archive }: { archive: Archive } = $props();

	let deleting = $state(false);
	let keepFile = $state(false);
	let busy = $state(false);
	// In-flight DELETE /api/zims/{id}. Separate from `busy` (which guards the
	// row's toggle/re-index actions) because the dialog needs to switch to its
	// "working…" state without the row's controls being the thing that's busy.
	let removing = $state(false);
	let error = $state<string | null>(null);

	$effect(() => (deleting ? lockBodyScroll() : undefined));

	// Matches "queued" as well as "running" so a job that's been submitted but
	// hasn't started executing yet (per-type concurrency defaults to 1 — see
	// src/vesta/jobs/runner.py's _concurrency_limit — so a click can sit queued
	// behind another archive's build) still counts as in-flight. Without this,
	// the trigger button reappears during that window and a second click queues
	// a duplicate job (Workstream A, Issue 1).
	const indexJob = $derived(
		jobsStore.list.find(
			(j) => j.type === 'index_zim' && j.target === String(archive.id) && (j.status === 'running' || j.status === 'queued')
		)
	);
	const hasPendingJob = $derived(indexJob != null);
	const pct = $derived(indexJob?.total ? Math.round(((indexJob.progress ?? 0) / indexJob.total) * 100) : null);
	const rateLabel = $derived.by(() => {
		if (!indexJob || indexJob.rate == null) return null;
		return `${indexJob.rate.toLocaleString(undefined, { maximumFractionDigits: 1 })}/s`;
	});
	// Queued-but-not-yet-running: indexJob exists but the archive's own
	// index_status hasn't flipped to "running" yet (that only happens once the
	// job actually starts executing — src/vesta/index/job.py's _run_build).
	const queued = $derived(indexJob?.status === 'queued' && archive.index_status !== 'running');

	// The server can report index_status "running" with no job behind it —
	// e.g. a backend restart killed/cancelled the index_zim job without the
	// zim's domain-level index_status being reset. Once the global job
	// stream's initial snapshot burst has landed (jobsStore.connected), an
	// absent job for a "running" archive is stale server state, not an
	// in-flight one. Without this, the row would show a permanently stuck 0%
	// progress bar with only a Delete action and no way to recover.
	const orphaned = $derived(archive.index_status === 'running' && !indexJob && jobsStore.connected);
	const retryDepth = $derived(archive.index_depth >= 1 && archive.index_depth <= 3 ? archive.index_depth : 3);

	async function toggleEnabled() {
		busy = true;
		try {
			await zimsApi.patch(archive.id, { enabled: !archive.enabled });
			await zimsStore.load();
		} catch (err) {
			error = err instanceof Error ? err.message : 'failed to update';
		} finally {
			busy = false;
		}
	}

	async function reindex(deeper: boolean, depthOverride?: number) {
		busy = true;
		error = null;
		try {
			const configured = Number(settingsValuesStore.values['index.default_depth']);
			const defaultDepth = configured >= 1 && configured <= 3 ? configured : 3;
			const depth = depthOverride ?? (deeper ? Math.min(3, archive.index_depth + 1) : Math.max(1, defaultDepth));
			await zimsApi.triggerIndex(archive.id, depth);
			await zimsStore.load();
		} catch (err) {
			error = err instanceof Error ? err.message : 'failed to start indexing';
		} finally {
			busy = false;
		}
	}

	async function confirmDelete() {
		removing = true;
		error = null;
		try {
			await zimsApi.remove(archive.id, keepFile);
			await zimsStore.load();
			deleting = false;
		} catch (err) {
			error = err instanceof Error ? err.message : 'failed to delete';
		} finally {
			removing = false;
		}
	}

	const archiveName = $derived(archive.corpus_label ?? archive.title ?? archive.name ?? `archive ${archive.id}`);
</script>

<div class="rounded-lg border border-border bg-surface p-4">
	<div class="flex flex-wrap items-start gap-3">
		<div class="min-w-0 flex-1">
			<div class="flex items-center gap-2">
				<span class="truncate font-semibold text-ink">{archiveName}</span>
				<FlavourPill flavour={archive.flavour} />
				{#if archive.language}
					<span class="max-w-[40%] shrink-0 truncate rounded-full bg-surface-muted px-2 py-0.5 text-xs text-muted" title={archive.language}>
						{formatLanguageNames(archive.language)}
					</span>
				{/if}
			</div>
			<div class="mt-1 text-xs text-faint">
				{#if archive.file_size}{formatBytes(archive.file_size)} · {/if}{archive.article_count.toLocaleString()} articles
			</div>
		</div>

		<label class="flex items-center gap-2 text-xs text-muted">
			<input type="checkbox" checked={archive.enabled} onchange={toggleEnabled} disabled={busy} />
			enabled
		</label>
	</div>

	<div class="mt-3 flex items-center gap-2 text-sm">
		<div class="flex gap-0.5" title="index depth {archive.index_depth}/3">
			{#each [1, 2, 3] as p (p)}
				<span class="size-1.5 rounded-full {p <= archive.index_depth ? 'bg-accent' : 'bg-border'}"></span>
			{/each}
		</div>
		<span class="text-ink-2">{depthLabel(archive.index_depth)}</span>
	</div>

	{#if orphaned}
		<p class="mt-2 text-xs text-warning">
			Indexing stopped without finishing (no active job found) — it may have been interrupted by a server restart.
		</p>
	{:else if queued}
		<p class="mt-2 text-xs text-muted">Queued — indexing will start once the current job finishes.</p>
	{:else if archive.index_status === 'running'}
		<div class="mt-2">
			<div class="h-1.5 overflow-hidden rounded-full bg-surface-muted">
				<div class="h-full rounded-full bg-accent" style="width: {pct ?? 0}%"></div>
			</div>
			<p class="mt-1 text-xs text-faint">
				Keyword search already works on the {pct ?? 0}% that's done. Meaning-aware search switches on when it finishes.
				{#if rateLabel}<span>· {rateLabel}</span>{/if}
				{#if indexJob?.eta_seconds != null}<span>· ETA {formatDuration(indexJob.eta_seconds)}</span>{/if}
			</p>
		</div>
	{:else if archive.index_status === 'stale'}
		<p class="mt-2 text-xs text-warning">Indexed with a different embedding model ({archive.embedding_model ?? 'unknown'}).</p>
	{:else if archive.index_status === 'error'}
		<p class="mt-2 text-xs text-danger">Indexing failed.</p>
	{/if}

	{#if error}<p class="mt-2 text-xs text-danger">{error}</p>{/if}

	<div class="mt-3 flex flex-wrap items-center gap-2 text-xs">
		<a
			href="/archive/{archive.id}"
			class="flex items-center gap-1 rounded-md border border-border px-2 py-1 font-medium text-ink-2 hover:bg-surface-muted"
		>
			<Compass class="size-3.5" /> Browse
		</a>
		{#if orphaned}
			<button type="button" class="rounded-md border border-border px-2 py-1 font-medium text-ink-2 hover:bg-surface-muted" disabled={busy} onclick={() => reindex(false, retryDepth)}>
				Retry indexing
			</button>
		{:else if !hasPendingJob && archive.index_status !== 'running'}
			{#if archive.index_depth < 3}
				<button type="button" class="rounded-md border border-border px-2 py-1 font-medium text-ink-2 hover:bg-surface-muted" disabled={busy} onclick={() => reindex(archive.index_depth > 0)}>
					{archive.index_depth === 0 ? 'Index' : 'Re-index deeper'}
				</button>
			{/if}
			{#if archive.index_status === 'stale' || archive.index_status === 'error'}
				<button type="button" class="rounded-md border border-border px-2 py-1 font-medium text-ink-2 hover:bg-surface-muted" disabled={busy} onclick={() => reindex(false)}>
					Re-index
				</button>
			{/if}
		{/if}
		{#if orphaned || (!hasPendingJob && archive.index_status !== 'running')}
			<label for="depth-picker-{archive.id}" class="text-faint">Index at</label>
			<select
				id="depth-picker-{archive.id}"
				class="rounded-md border border-border bg-surface px-1.5 py-1 text-xs text-ink-2 disabled:opacity-40"
				disabled={busy}
				value=""
				onchange={(e) => {
					const target = e.currentTarget as HTMLSelectElement;
					const depth = Number(target.value);
					target.value = '';
					if (depth) reindex(false, depth);
				}}
			>
				<option value="" disabled selected>choose depth…</option>
				{#each [1, 2, 3] as p (p)}
					<option value={p} disabled={p === archive.index_depth}>{depthLabel(p)}</option>
				{/each}
			</select>
		{/if}
		<button type="button" class="ml-auto flex items-center gap-1 rounded-md px-2 py-1 font-medium text-danger hover:bg-danger-soft" onclick={() => (deleting = true)}>
			<Trash2 class="size-3.5" /> Delete
		</button>
	</div>
</div>

<svelte:window onkeydown={(e) => e.key === 'Escape' && deleting && !removing && (deleting = false)} />

{#if deleting}
	<div class="fixed inset-0 z-[95] bg-black/40" role="presentation" onclick={() => !removing && (deleting = false)}></div>
	<div
		class="fixed left-1/2 top-1/2 z-[96] max-h-[90dvh] w-[min(420px,92vw)] -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-xl border border-border bg-surface p-5 shadow-pop"
		role="dialog"
		aria-modal="true"
		aria-label="Delete {archiveName}"
		aria-busy={removing}
	>
		{#if removing}
			<div class="flex items-start gap-3">
				<span class="mt-0.5 size-5 shrink-0 animate-spin rounded-full border-2 border-border-strong border-t-accent"></span>
				<div>
					<h3 class="mb-1 text-base font-semibold text-ink">Deleting {archiveName}…</h3>
					<p class="text-sm text-muted">
						Removing every article, chunk, and vector from the database{keepFile
							? '.'
							: ', then deleting the .zim file.'} This can take a minute for large archives — closing this
						dialog won't stop it.
					</p>
				</div>
			</div>
		{:else}
			<h3 class="mb-2 text-base font-semibold text-ink">Delete {archiveName}?</h3>
			<p class="mb-3 text-sm text-muted">
				This removes {archive.file_size ? formatBytes(archive.file_size) : 'the archive'} and every database reference — articles,
				aliases, chunks, vectors — unless you keep the file.
			</p>
			<label class="mb-3 flex items-center gap-2 text-sm text-ink-2">
				<input type="checkbox" bind:checked={keepFile} />
				Keep the .zim file on disk
			</label>
			{#if error}<p class="mb-3 text-sm text-danger">{error}</p>{/if}
			<div class="flex justify-end gap-2">
				<button type="button" class="rounded-md px-3 py-1.5 text-sm text-muted hover:bg-surface-muted" onclick={() => (deleting = false)}>Cancel</button>
				<button
					type="button"
					class="rounded-md bg-danger px-3 py-1.5 text-sm font-medium text-white hover:bg-danger-hover"
					onclick={confirmDelete}
				>
					Delete {archive.file_size ? formatBytes(archive.file_size) : ''}
				</button>
			</div>
		{/if}
	</div>
{/if}
