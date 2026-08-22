<script lang="ts">
	// Compare view for the Advanced → Benchmarks page. Takes 2+
	// selected runs, fetches the backend pairwise diff (GET /api/bench/compare)
	// plus each run's full results, and renders:
	//   - an aggregate delta table (one row per pair: strict accuracy + source
	//     recall deltas over the shared set),
	//   - the four buckets (fixed / broken / both correct / both wrong) with
	//     per-question rows for the active pair. `broken` is the regression
	//     catcher (trap 9).
	//
	// Buckets are computed client-side from the two runs' verdicts via
	// computeCompareBuckets (mirrors the backend), so every bucket row carries
	// the full per-question detail from the results feed. The backend pair's
	// deltas are used for the aggregate table.
	import { benchApi } from '$lib/api/bench';
	import { computeCompareBuckets } from '$lib/bench';
	import type { BenchComparePair, BenchResultRow, BenchRunSummary } from '$lib/types';

	let {
		runs,
		onClose
	}: {
		runs: BenchRunSummary[];
		onClose: () => void;
	} = $props();

	const runIds = $derived(runs.map((r) => r.id));
	const runById = $derived(new Map(runs.map((r) => [r.id, r])));

	let pairs = $state<BenchComparePair[]>([]);
	let resultsByRun = $state<Map<number, BenchResultRow[]>>(new Map());
	let loading = $state(true);
	let loadError = $state<string | null>(null);
	let activePairIdx = $state(0);

	$effect(() => {
		let cancelled = false;
		loading = true;
		loadError = null;
		activePairIdx = 0;
		(async () => {
			try {
				const cmp = await benchApi.compare(runIds);
				if (cancelled) return;
				pairs = cmp.pairs;
				const byRun = new Map<number, BenchResultRow[]>();
				for (const rid of runIds) byRun.set(rid, await benchApi.allResults(rid));
				if (cancelled) return;
				resultsByRun = byRun;
			} catch (err) {
				if (!cancelled) loadError = err instanceof Error ? err.message : 'failed to compare runs';
			} finally {
				if (!cancelled) loading = false;
			}
		})();
		return () => {
			cancelled = true;
		};
	});

	const activePair = $derived(pairs[activePairIdx] ?? null);

	const activeBuckets = $derived.by(() => {
		if (!activePair) return null;
		const rowsA = resultsByRun.get(activePair.run_a) ?? [];
		const rowsB = resultsByRun.get(activePair.run_b) ?? [];
		return computeCompareBuckets(rowsA, rowsB);
	});

	function runName(id: number): string {
		const r = runById.get(id);
		return r ? `#${r.id} ${r.system}` : `#${id}`;
	}

	function fmtDelta(v: number | undefined): string {
		if (v == null) return '—';
		const sign = v > 0 ? '+' : v < 0 ? '−' : '·';
		return `${sign} ${Math.abs(v).toFixed(3)}`;
	}

	function verdictClass(v: string): string {
		switch (v) {
			case 'correct':
				return 'text-success';
			case 'partial':
				return 'text-warning';
			case 'incorrect':
				return 'text-danger';
			case 'unjudged':
				return 'text-warning';
			default:
				return 'text-faint';
		}
	}

	interface BucketDef {
		key: 'fixed' | 'broken' | 'bothCorrect' | 'bothWrong';
		label: string;
		desc: string;
		accent: string;
	}

	const BUCKETS: BucketDef[] = [
		{ key: 'fixed', label: 'Fixed', desc: 'wrong in A, correct in B', accent: 'text-success' },
		{ key: 'broken', label: 'Broken (regression)', desc: 'correct in A, wrong in B', accent: 'text-danger' },
		{ key: 'bothCorrect', label: 'Both correct', desc: 'correct in both', accent: 'text-ink-2' },
		{ key: 'bothWrong', label: 'Both wrong', desc: 'wrong in both', accent: 'text-warning' }
	];
</script>

<div class="rounded-lg border border-border bg-surface">
	<div class="flex items-center justify-between border-b border-border p-4">
		<div>
			<h3 class="text-base font-semibold text-ink">Compare runs</h3>
			<p class="mt-0.5 text-xs text-muted">
				Comparing {runs.map((r) => runName(r.id)).join(' vs ')} — shared questions only, per backend deltas and client-side buckets.
			</p>
		</div>
		<button type="button" class="text-xs text-accent hover:underline" onclick={onClose}>Close compare</button>
	</div>

	{#if loading}
		<p class="p-4 text-sm text-muted">Comparing…</p>
	{:else if loadError}
		<p class="p-4 text-sm text-danger">{loadError}</p>
	{:else if pairs.length === 0}
		<p class="p-4 text-sm text-faint">No runs to compare.</p>
	{:else}
		<div class="border-b border-border p-4">
			<h4 class="mb-3 text-sm font-semibold text-ink">Aggregate deltas</h4>
			<div class="overflow-x-auto rounded-lg border border-border">
				<table class="w-full min-w-[480px] text-left text-sm">
					<thead class="bg-surface-muted text-xs uppercase tracking-wide text-faint">
						<tr>
							<th class="px-3 py-2 font-medium">Pair</th>
							<th class="px-3 py-2 text-right font-medium">Shared</th>
							<th class="px-3 py-2 text-right font-medium">Δ strict accuracy</th>
							<th class="px-3 py-2 text-right font-medium">Δ source recall</th>
						</tr>
					</thead>
					<tbody class="divide-y divide-border">
						{#each pairs as pair, idx (pair.run_a + '-' + pair.run_b)}
							<tr class="cursor-pointer {idx === activePairIdx ? 'bg-accent-soft' : 'hover:bg-surface-muted'}" onclick={() => (activePairIdx = idx)}>
								<td class="px-3 py-2 text-ink">
									{runName(pair.run_a)} → {runName(pair.run_b)}
									{#if idx === activePairIdx}<span class="ml-1 text-[10px] text-accent-soft-text">active</span>{/if}
								</td>
								<td class="px-3 py-2 text-right font-mono text-xs text-faint">{pair.shared_denominator}</td>
								<td class="px-3 py-2 text-right font-mono text-xs text-ink-2">{fmtDelta(pair.deltas.strict_accuracy)}</td>
								<td class="px-3 py-2 text-right font-mono text-xs text-ink-2">{fmtDelta(pair.deltas.source_recall)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</div>

		{#if activePair && activeBuckets}
			<div class="p-4">
				<h4 class="mb-1 text-sm font-semibold text-ink">
					{runName(activePair.run_a)} → {runName(activePair.run_b)}
				</h4>
				<p class="mb-3 text-xs text-faint">{activeBuckets.sharedDenominator} shared questions</p>
				<div class="grid grid-cols-1 gap-3 min-[640px]:grid-cols-2">
					{#each BUCKETS as b (b.key)}
						{@const rows = activeBuckets[b.key]}
						<div class="rounded-lg border border-border bg-surface-muted/40">
							<div class="flex items-baseline justify-between border-b border-border px-3 py-2">
								<span class="text-sm font-semibold {b.accent}">{b.label}</span>
								<span class="text-xs text-faint">{rows.length} · {b.desc}</span>
							</div>
							{#if rows.length === 0}
								<p class="px-3 py-3 text-xs text-faint">none</p>
							{:else}
								<ul class="max-h-64 divide-y divide-border overflow-y-auto">
									{#each rows as r (r.question_id)}
										<li class="flex items-start gap-2 px-3 py-2">
											<span class="mt-0.5 shrink-0 font-mono text-[10px] text-faint">{r.question_id}</span>
											<div class="min-w-0 flex-1">
												<p class="text-sm text-ink">{r.question_text}</p>
												<p class="mt-0.5 font-mono text-[11px] {verdictClass(r.verdict)}">
													{r.verdict}{r.source_hit_rank == null ? ' · source missed' : ` · source #${r.source_hit_rank}`}
												</p>
											</div>
										</li>
									{/each}
								</ul>
							{/if}
						</div>
					{/each}
				</div>
				{#if activeBuckets.onlyA.length > 0 || activeBuckets.onlyB.length > 0}
					<p class="mt-3 text-xs text-faint">
						Not in the shared set (different subsets): only in A —
						<span class="font-mono">{activeBuckets.onlyA.join(', ') || '—'}</span>; only in B —
						<span class="font-mono">{activeBuckets.onlyB.join(', ') || '—'}</span>. These don't count toward the deltas.
					</p>
				{/if}
			</div>
		{/if}
	{/if}
</div>
