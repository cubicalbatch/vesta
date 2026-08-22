<script lang="ts">
	// Eval runs tab. Retrieval-only
	// golden-set metrics (recall@k, MRR, ndcg — see EvalRunMetrics in types.ts
	// for exactly what the backend returns; there is no citation-precision or
	// refusal-rate field here, those are answer-benchmark concepts).
	//
	// The five comparison pins (profile_hash, golden_hash, archive_checksum,
	// git_sha, machine_id) are shown on every run and checked before any
	// "vs previous" delta is rendered — two runs with different pins are not
	// comparable, and silently computing a delta anyway is the failure mode
	// this tab exists to prevent: "Two runs with different pins
	// are not comparable and the UI is where that gets noticed").
	import { evalApi } from '$lib/api/eval';
	import { retrievalApi } from '$lib/api/retrieval';
	import { formatDate } from '$lib/format';
	import type { EvalRunDetail, EvalSliceMetrics, ProfileItem } from '$lib/types';

	let runs = $state<EvalRunDetail[]>([]);
	let loading = $state(true);
	let loadError = $state<string | null>(null);

	let profiles = $state<ProfileItem[]>([]);
	let selectedProfile = $state('');
	let goldenSet = $state<'full' | 'fixture_subset'>('full');
	let notes = $state('');
	let starting = $state(false);
	let startError = $state<string | null>(null);
	let pollTimer: ReturnType<typeof setInterval> | null = null;
	let expanded = $state<number | null>(null);

	async function loadRuns() {
		try {
			runs = await evalApi.listRuns();
			loadError = null;
		} catch (err) {
			loadError = err instanceof Error ? err.message : 'failed to load eval runs';
		} finally {
			loading = false;
		}
	}

	$effect(() => {
		loadRuns();
		retrievalApi
			.listProfiles()
			.then((r) => (profiles = r.profiles))
			.catch(() => {});
		return () => {
			if (pollTimer) clearInterval(pollTimer);
		};
	});

	async function startRun() {
		starting = true;
		startError = null;
		try {
			const res = await evalApi.run({ profile: selectedProfile || null, golden_set: goldenSet, notes });
			await loadRuns();
			if (pollTimer) clearInterval(pollTimer);
			pollTimer = setInterval(async () => {
				const detail = await evalApi.getRun(res.id).catch(() => null);
				await loadRuns();
				if (detail && detail.status !== 'running' && pollTimer) {
					clearInterval(pollTimer);
					pollTimer = null;
				}
			}, 2000);
		} catch (err) {
			startError = err instanceof Error ? err.message : 'failed to start eval run';
		} finally {
			starting = false;
		}
	}

	function allSlice(run: EvalRunDetail): EvalSliceMetrics | undefined {
		return run.metrics?.metrics?.slices?.all;
	}

	function pinsMatch(a: EvalRunDetail, b: EvalRunDetail): boolean {
		return a.profile_hash === b.profile_hash && a.golden_hash === b.golden_hash && a.archive_checksum === b.archive_checksum;
	}

	// runs are newest-first (API: `ORDER BY id DESC`), so index+1 is "previous".
	function delta(run: EvalRunDetail, idx: number, metric: keyof EvalSliceMetrics): { value: number; comparable: boolean } | null {
		const prev = runs[idx + 1];
		if (!prev) return null;
		const a = allSlice(run);
		const b = allSlice(prev);
		if (!a || !b) return null;
		return { value: (a[metric] as number) - (b[metric] as number), comparable: pinsMatch(run, prev) };
	}

	function fmtPct(v: number | undefined): string {
		return v == null ? '—' : v.toFixed(2);
	}

	function fmtDelta(d: { value: number; comparable: boolean } | null): string {
		if (!d) return '—';
		const sign = d.value > 0 ? '▲' : d.value < 0 ? '▼' : '·';
		return `${sign} ${Math.abs(d.value).toFixed(3)}`;
	}

	const latest = $derived(runs[0]);
</script>

<div>
	<div class="mb-1 flex items-center justify-between">
		<h2 class="text-lg font-semibold text-ink">Eval runs</h2>
	</div>
	<p class="mb-4 text-sm text-muted">
		Golden-set retrieval metrics for a profile — recall@k, MRR and ndcg over the pinned archive. Not an answer-quality
		measure; see Benchmarks for that.
	</p>

	<form class="mb-6 flex flex-wrap items-end gap-3 rounded-lg border border-border bg-surface p-3" onsubmit={(e) => { e.preventDefault(); startRun(); }}>
		<label class="text-sm">
			<span class="mb-1 block text-xs text-faint">Profile</span>
			<select id="eval-profile" name="eval-profile" bind:value={selectedProfile} class="rounded-md border border-border bg-surface px-2 py-1.5 text-sm">
				<option value="">active profile</option>
				{#each profiles as p (p.name)}
					<option value={p.name}>{p.name}{p.builtin ? '' : ' (custom)'}</option>
				{/each}
			</select>
		</label>
		<label class="text-sm">
			<span class="mb-1 block text-xs text-faint">Golden set</span>
			<select id="eval-golden-set" name="eval-golden-set" bind:value={goldenSet} class="rounded-md border border-border bg-surface px-2 py-1.5 text-sm">
				<option value="full">full</option>
				<option value="fixture_subset">fixture_subset</option>
			</select>
		</label>
		<label class="min-w-0 flex-1 text-sm">
			<span class="mb-1 block text-xs text-faint">Notes</span>
			<input id="eval-notes" name="eval-notes" type="text" bind:value={notes} class="w-full rounded-md border border-border bg-surface px-2 py-1.5 text-sm outline-none focus:border-accent" placeholder="optional" />
		</label>
		<button type="submit" class="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-40" disabled={starting}>
			{starting ? 'Starting…' : 'Run eval'}
		</button>
	</form>
	{#if startError}<p class="mb-4 text-xs text-danger">{startError}</p>{/if}

	{#if latest}
		{@const a = allSlice(latest)}
		<div class="mb-6 grid grid-cols-2 gap-3 min-[640px]:grid-cols-4">
			<div class="rounded-lg border border-border bg-surface p-3">
				<div class="text-xs text-faint">Recall@5</div>
				<div class="text-xl font-semibold text-ink">{fmtPct(a?.['recall@5'])}</div>
			</div>
			<div class="rounded-lg border border-border bg-surface p-3">
				<div class="text-xs text-faint">MRR</div>
				<div class="text-xl font-semibold text-ink">{fmtPct(a?.mrr)}</div>
			</div>
			<div class="rounded-lg border border-border bg-surface p-3">
				<div class="text-xs text-faint">ndcg@10</div>
				<div class="text-xl font-semibold text-ink">{fmtPct(a?.['ndcg@10'])}</div>
			</div>
			<div class="rounded-lg border border-border bg-surface p-3">
				<div class="text-xs text-faint">Degraded</div>
				<div class="text-xl font-semibold {latest.metrics?.metrics?.degraded ? 'text-warning' : 'text-ink'}">
					{latest.metrics?.metrics?.degraded ? 'yes' : 'no'}
				</div>
			</div>
		</div>
	{/if}

	{#if loading}
		<p class="text-sm text-muted">Loading…</p>
	{:else if loadError}
		<p class="text-sm text-danger">{loadError}</p>
	{:else if runs.length === 0}
		<p class="text-sm text-muted">No eval runs yet.</p>
	{:else}
		<div class="overflow-x-auto rounded-lg border border-border">
			<table class="w-full min-w-[720px] text-left text-sm">
				<thead class="bg-surface-muted text-xs uppercase tracking-wide text-faint">
					<tr>
						<th class="px-3 py-2 font-medium">Run</th>
						<th class="px-3 py-2 font-medium">Profile</th>
						<th class="px-3 py-2 text-right font-medium">Recall@5</th>
						<th class="px-3 py-2 text-right font-medium">MRR</th>
						<th class="px-3 py-2 text-right font-medium">Δ vs previous</th>
						<th class="px-3 py-2 font-medium">Pins</th>
						<th class="px-3 py-2 font-medium"></th>
					</tr>
				</thead>
				<tbody class="divide-y divide-border">
					{#each runs as run, idx (run.id)}
						{@const a = allSlice(run)}
						{@const d = delta(run, idx, 'recall@5')}
						<tr>
							<td class="px-3 py-2 text-ink">#{run.id} <span class="text-faint">{formatDate(run.started_at)}</span></td>
							<td class="px-3 py-2 font-mono text-xs text-ink-2">{run.profile}</td>
							<td class="px-3 py-2 text-right font-mono text-xs">{fmtPct(a?.['recall@5'])}</td>
							<td class="px-3 py-2 text-right font-mono text-xs">{fmtPct(a?.mrr)}</td>
							<td class="px-3 py-2 text-right font-mono text-xs {d && !d.comparable ? 'text-warning' : d && d.value > 0 ? 'text-success' : d && d.value < 0 ? 'text-danger' : 'text-faint'}">
								{#if d && !d.comparable}
									not comparable
								{:else}
									{fmtDelta(d)}
								{/if}
							</td>
							<td class="px-3 py-2 font-mono text-[11px] text-faint" title="profile {run.profile_hash} · golden {run.golden_hash} · archive {run.archive_checksum} · git {run.git_sha} · machine {run.machine_id}">
								{run.profile_hash.slice(0, 8)}/{run.golden_hash.slice(0, 8)}
							</td>
							<td class="px-3 py-2 text-right">
								<button type="button" class="text-xs text-accent hover:underline" onclick={() => (expanded = expanded === run.id ? null : run.id)}>
									{expanded === run.id ? 'Hide' : 'Open'}
								</button>
							</td>
						</tr>
						{#if expanded === run.id}
							<tr>
								<td colspan="7" class="bg-surface-sunken px-3 py-3">
									<div class="grid grid-cols-1 gap-x-6 gap-y-1 font-mono text-xs text-ink-2 min-[640px]:grid-cols-2">
										<div>profile_hash <span class="text-faint">{run.profile_hash}</span></div>
										<div>golden_hash <span class="text-faint">{run.golden_hash}</span></div>
										<div>archive_checksum <span class="text-faint">{run.archive_checksum}</span></div>
										<div>git_sha <span class="text-faint">{run.git_sha}</span></div>
										<div>machine_id <span class="text-faint">{run.machine_id}</span></div>
										<div>query_count <span class="text-faint">{run.metrics?.metrics?.query_count ?? 0}</span></div>
									</div>
									{#if run.metrics?.metrics?.degraded_components?.length}
										<p class="mt-2 text-xs text-warning">degraded: {run.metrics.metrics.degraded_components.join(', ')}</p>
									{/if}
									{#if run.config?.notes}<p class="mt-2 text-xs text-faint">notes: {run.config.notes}</p>{/if}
								</td>
							</tr>
						{/if}
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</div>
