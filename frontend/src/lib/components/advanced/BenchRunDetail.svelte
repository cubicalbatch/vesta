<script lang="ts">
	// Run detail panel for the Advanced → Benchmarks page.
	// Fetches the run's aggregates (GET /runs/{id}, which carries metrics_json)
	// plus every per-question row (GET /runs/{id}/results, paginated) and
	// renders:
	//   1. Untrusted / judge-shares-endpoint warnings
	//   2. Scorecard: ceiling / system / floor bar + headroom_realised, strict,
	//      weighted, source_recall@{1,5,10}, source_coverage, over-refusal,
	//      hallucination rate, p50/p95 latency.
	//   3. Capability breakdown table (sortable).
	//   4. Failure-attribution 2x2 — clicking a cell filters the per-question
	//      table (client-side, via attributionCellMatches).
	//   5. Per-question table, expandable to answer vs expected + retrieved
	//      paths + judge reason.
	//
	// The results feed deliberately excludes trace_json (trap 10), so there is
	// no trace to hand to TraceView here — the expandable row shows what the
	// public feed does carry.
	import { benchApi } from '$lib/api/bench';
	import { formatDate } from '$lib/format';
	import { ATTRIBUTION_CELLS, attributionCellMatches, latencyPercentiles } from '$lib/bench';
	import type { AttributionCell, BenchResultRow, BenchRunDetail } from '$lib/types';

	let { runId }: { runId: number } = $props();

	let detail = $state<BenchRunDetail | null>(null);
	let rows = $state<BenchResultRow[]>([]);
	let loading = $state(true);
	let loadError = $state<string | null>(null);

	let activeCell = $state<AttributionCell | null>(null);
	let capFilter = $state<string>('');
	let verdictFilter = $state<string>('');
	let sortKey = $state<keyof CapabilityRow>('strict_accuracy');
	let sortDir = $state<'asc' | 'desc'>('desc');
	let expandedRow = $state<string | null>(null);

	$effect(() => {
		let cancelled = false;
		loading = true;
		loadError = null;
		activeCell = null;
		capFilter = '';
		verdictFilter = '';
		expandedRow = null;
		benchApi
			.getRun(runId)
			.then((d) => {
				if (cancelled) return;
				detail = d;
			})
			.catch((err) => {
				if (!cancelled) loadError = err instanceof Error ? err.message : 'failed to load run detail';
			});
		benchApi
			.allResults(runId)
			.then((all) => {
				if (cancelled) return;
				rows = all;
			})
			.catch((err) => {
				if (!cancelled) loadError = err instanceof Error ? err.message : 'failed to load run results';
			})
			.finally(() => {
				if (!cancelled) loading = false;
			});
		return () => {
			cancelled = true;
		};
	});

	function fmtPct(v: number | null | undefined): string {
		return v == null ? '—' : `${(v * 100).toFixed(1)}%`;
	}

	function fmtCount(part: number | undefined, total: number | undefined): string {
		if (part == null || total == null) return '—';
		return `${part}/${total}`;
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

	function verdictLabel(v: string): string {
		if (v === 'pending') return 'pending';
		if (v === 'unjudged') return 'unjudged';
		return v;
	}

	// ── Capability breakdown ──────────────────────────────────────────────────
	interface CapabilityRow {
		capability: string;
		n: number;
		strict_accuracy: number;
		weighted_accuracy: number;
		source_recall_at_10: number;
		source_coverage: number;
	}

	const capabilityRows = $derived.by((): CapabilityRow[] => {
		if (!detail) return [];
		return Object.entries(detail.metrics.by_capability)
			.map(([capability, m]) => ({
				capability,
				n: m.n,
				strict_accuracy: m.strict_accuracy,
				weighted_accuracy: m.weighted_accuracy,
				source_recall_at_10: m.source_recall_at_10,
				source_coverage: m.source_coverage
			}))
			.sort((a, b) => {
				const av = a[sortKey];
				const bv = b[sortKey];
				const cmp = typeof av === 'number' && typeof bv === 'number' ? av - bv : String(av).localeCompare(String(bv));
				return sortDir === 'asc' ? cmp : -cmp;
			});
	});

	function toggleSort(key: keyof CapabilityRow) {
		if (sortKey === key) sortDir = sortDir === 'asc' ? 'desc' : 'asc';
		else {
			sortKey = key;
			sortDir = key === 'capability' ? 'asc' : 'desc';
		}
	}

	function sortArrow(key: keyof CapabilityRow): string {
		return sortKey === key ? (sortDir === 'asc' ? ' ▲' : ' ▼') : '';
	}

	// ── Attribution 2x2 ───────────────────────────────────────────────────────
	const attribution = $derived(detail?.metrics.attribution);

	function cellTitle(cell: AttributionCell): string {
		switch (cell) {
			case 'correct_source_found':
				return 'Correct + source found';
			case 'correct_source_missed':
				return 'Correct + source missed';
			case 'failed_source_found':
				return 'Failed + source found';
			case 'failed_source_missed':
				return 'Failed + source missed';
		}
	}

	// ── Per-question table ────────────────────────────────────────────────────
	const filteredRows = $derived.by((): BenchResultRow[] => {
		let out = rows;
		if (activeCell) out = out.filter((r) => attributionCellMatches(r, activeCell!));
		if (capFilter) out = out.filter((r) => r.capability === capFilter);
		if (verdictFilter) out = out.filter((r) => r.verdict === verdictFilter);
		return out;
	});

	const capabilities = $derived([...new Set(rows.map((r) => r.capability))].sort());
	const verdicts = $derived([...new Set(rows.map((r) => r.verdict))].sort());

	const latency = $derived(latencyPercentiles(rows));

	function clearFilter() {
		activeCell = null;
		capFilter = '';
		verdictFilter = '';
		expandedRow = null;
	}

	function sourceRankText(r: BenchResultRow): string {
		return r.source_hit_rank == null ? 'miss' : `#${r.source_hit_rank}`;
	}
</script>

<div class="rounded-lg border border-border bg-surface">
	{#if loading}
		<p class="p-4 text-sm text-muted">Loading run #{runId}…</p>
	{:else if loadError}
		<p class="p-4 text-sm text-danger">{loadError}</p>
	{:else if detail}
		<div class="border-b border-border p-4">
			<div class="flex flex-wrap items-center gap-2">
				<h3 class="text-base font-semibold text-ink">
					#{detail.id} <span class="text-faint">{detail.system}</span>
				</h3>
				{#if detail.label}<span class="text-sm text-faint">· {detail.label}</span>{/if}
				{#if !detail.trusted}
					<span
						class="rounded-full border border-warning bg-warning-soft px-2 py-0.5 text-[11px] font-semibold text-warning"
						>untrusted</span
					>
				{/if}
				<span class="ml-auto text-xs text-faint">{formatDate(detail.started_at)}</span>
			</div>
			<div class="mt-1 flex flex-wrap gap-x-4 gap-y-0.5 font-mono text-xs text-faint">
				<span>dataset {detail.dataset_name}</span>
				<span>profile {detail.profile_name || 'active'}</span>
				<span>answer {detail.answer_model}</span>
				<span>judge {detail.judge_model}</span>
				<span>scope {detail.scope || 'default'}</span>
			</div>
		</div>

		{#if !detail.trusted}
			<div class="mx-4 mt-4 rounded-md border border-warning bg-warning-soft p-3 text-xs text-warning">
				<strong>Untrusted run.</strong> The judge failed calibration
				{#if detail.calibration != null} (correlation {detail.calibration.toFixed(2)}, below threshold){/if}
				— the verdicts here are not reliable.
			</div>
		{/if}
		{#if detail.judge_shares_endpoint}
			<div class="mx-4 mt-4 rounded-md border border-border bg-surface-muted p-3 text-xs text-faint">
				The judge and the answer model share one endpoint, so judge concurrency was clamped to 1 — judging will be
				slower than the answer phase.
			</div>
		{/if}
		{#if detail.abort_reason}
			<div class="mx-4 mt-4 rounded-md border border-danger bg-danger-soft p-3 text-xs text-danger">
				Aborted: {detail.abort_reason}
			</div>
		{/if}

		<!-- Scorecard -->
		<div class="border-b border-border p-4">
			<h4 class="mb-3 text-sm font-semibold text-ink">Scorecard</h4>
			{#if detail}
				{@const ref = detail.metrics.reference}
				{@const ans = detail.metrics.answer}
				{@const src = detail.metrics.source}
				<div class="mb-3 space-y-2">
				{#if ref.total > 0}
					{@const pct = (part: number) => `${(part / ref.total) * 100}%`}
					<div class="flex items-center gap-3 text-xs">
						<span class="w-16 shrink-0 font-mono text-faint">ceiling</span>
						<div class="h-3 flex-1 overflow-hidden rounded-full bg-score-track">
							<div class="h-full bg-score-fill" style="width: {pct(ref.ceiling)}"></div>
						</div>
						<span class="w-28 shrink-0 font-mono text-ink-2">{fmtCount(ref.ceiling, ref.total)} · {pct(ref.ceiling)}</span>
					</div>
					<div class="flex items-center gap-3 text-xs">
						<span class="w-16 shrink-0 font-mono text-faint">system</span>
						<div class="h-3 flex-1 overflow-hidden rounded-full bg-score-track">
							<div class="h-full bg-accent" style="width: {pct(ref.system)}"></div>
						</div>
						<span class="w-28 shrink-0 font-mono text-ink-2">{fmtCount(ref.system, ref.total)} · {pct(ref.system)}</span>
					</div>
					<div class="flex items-center gap-3 text-xs">
						<span class="w-16 shrink-0 font-mono text-faint">floor</span>
						<div class="h-3 flex-1 overflow-hidden rounded-full bg-score-track">
							<div class="h-full bg-disabled" style="width: {pct(ref.floor)}"></div>
						</div>
						<span class="w-28 shrink-0 font-mono text-ink-2">{fmtCount(ref.floor, ref.total)} · {pct(ref.floor)}</span>
					</div>
				{:else}
					<p class="text-xs text-faint">No reference points recorded for this run.</p>
				{/if}
			</div>
			<div class="mb-3">
				<span class="text-xs text-faint">headroom realised </span>
				<span class="font-mono text-sm font-semibold text-ink">{fmtPct(ref.headroom_realised)}</span>
				{#if ref.suppressed_reason}<span class="ml-2 text-xs text-warning">({ref.suppressed_reason})</span>{/if}
			</div>
			<div class="grid grid-cols-2 gap-3 min-[640px]:grid-cols-4">
				<div class="rounded-md border border-border bg-surface-muted/60 p-2">
					<div class="text-xs text-faint">strict</div>
					<div class="font-mono text-lg font-semibold text-ink">{fmtPct(ans.strict_accuracy)}</div>
				</div>
				<div class="rounded-md border border-border bg-surface-muted/60 p-2">
					<div class="text-xs text-faint">weighted</div>
					<div class="font-mono text-lg font-semibold text-ink">{fmtPct(ans.weighted_accuracy)}</div>
				</div>
				<div class="rounded-md border border-border bg-surface-muted/60 p-2">
					<div class="text-xs text-faint">source recall@10</div>
					<div class="font-mono text-lg font-semibold text-ink">{fmtPct(src.recall_at_10)}</div>
				</div>
				<div class="rounded-md border border-border bg-surface-muted/60 p-2">
					<div class="text-xs text-faint">source coverage</div>
					<div class="font-mono text-lg font-semibold text-ink">{fmtPct(src.source_coverage)}</div>
				</div>
				<div class="rounded-md border border-border bg-surface-muted/60 p-2">
					<div class="text-xs text-faint">hallucination</div>
					<div class="font-mono text-lg font-semibold text-ink">{fmtPct(ans.hallucination_rate)}</div>
				</div>
				<div class="rounded-md border border-border bg-surface-muted/60 p-2">
					<div class="text-xs text-faint">over-refusal</div>
					<div class="font-mono text-lg font-semibold text-ink">{fmtPct(ans.over_refusal)}</div>
				</div>
				<div class="rounded-md border border-border bg-surface-muted/60 p-2">
					<div class="text-xs text-faint">latency p50</div>
					<div class="font-mono text-lg font-semibold text-ink">{latency.p50 == null ? '—' : `${latency.p50.toFixed(0)} ms`}</div>
				</div>
				<div class="rounded-md border border-border bg-surface-muted/60 p-2">
					<div class="text-xs text-faint">latency p95</div>
					<div class="font-mono text-lg font-semibold text-ink">{latency.p95 == null ? '—' : `${latency.p95.toFixed(0)} ms`}</div>
				</div>
			</div>
			<div class="mt-2 text-xs text-faint">
				source recall@1 {fmtPct(src.recall_at_1)} · recall@5 {fmtPct(src.recall_at_5)} · recall@20 {fmtPct(src.recall_at_20)} ·
				source MRR {fmtPct(src.source_mrr)} · retrieved precision {fmtPct(src.retrieved_precision)} · unjudged {ans.unjudged}
			</div>
			{/if}
		</div>

		<!-- Capability breakdown -->
		<div class="border-b border-border p-4">
			<h4 class="mb-3 text-sm font-semibold text-ink">By capability</h4>
			<div class="overflow-x-auto rounded-lg border border-border">
				<table class="w-full min-w-[560px] text-left text-sm">
					<thead class="bg-surface-muted text-xs uppercase tracking-wide text-faint">
						<tr>
							<th class="cursor-pointer px-3 py-2 font-medium" onclick={() => toggleSort('capability')}>Capability{sortArrow('capability')}</th>
							<th class="cursor-pointer px-3 py-2 text-right font-medium" onclick={() => toggleSort('n')}>N{sortArrow('n')}</th>
							<th class="cursor-pointer px-3 py-2 text-right font-medium" onclick={() => toggleSort('strict_accuracy')}>Strict{sortArrow('strict_accuracy')}</th>
							<th class="cursor-pointer px-3 py-2 text-right font-medium" onclick={() => toggleSort('weighted_accuracy')}>Weighted{sortArrow('weighted_accuracy')}</th>
							<th class="cursor-pointer px-3 py-2 text-right font-medium" onclick={() => toggleSort('source_recall_at_10')}>Recall@10{sortArrow('source_recall_at_10')}</th>
							<th class="cursor-pointer px-3 py-2 text-right font-medium" onclick={() => toggleSort('source_coverage')}>Coverage{sortArrow('source_coverage')}</th>
							<th class="px-3 py-2 text-right font-medium">Attribution</th>
						</tr>
					</thead>
					<tbody class="divide-y divide-border">
						{#each capabilityRows as c (c.capability)}
							<tr>
								<td class="px-3 py-2 text-ink">{c.capability}</td>
								<td class="px-3 py-2 text-right font-mono text-xs text-faint">{c.n}</td>
								<td class="px-3 py-2 text-right font-mono text-xs text-ink-2">{fmtPct(c.strict_accuracy)}</td>
								<td class="px-3 py-2 text-right font-mono text-xs text-ink-2">{fmtPct(c.weighted_accuracy)}</td>
								<td class="px-3 py-2 text-right font-mono text-xs text-ink-2">{fmtPct(c.source_recall_at_10)}</td>
								<td class="px-3 py-2 text-right font-mono text-xs text-ink-2">{fmtPct(c.source_coverage)}</td>
								<td class="px-3 py-2 text-right font-mono text-[11px] text-faint" title="correct/source-found · correct/missed · failed/found · failed/missed">
									{detail?.metrics.by_capability[c.capability].attribution.correct_source_found} /
									{detail?.metrics.by_capability[c.capability].attribution.correct_source_missed} /
									{detail?.metrics.by_capability[c.capability].attribution.failed_source_found} /
									{detail?.metrics.by_capability[c.capability].attribution.failed_source_missed}
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</div>

		<!-- Attribution 2x2 -->
		{#if attribution}
			<div class="border-b border-border p-4">
				<h4 class="mb-3 text-sm font-semibold text-ink">Failure attribution</h4>
				<div class="grid max-w-md grid-cols-2 gap-2">
					{#each ATTRIBUTION_CELLS as cell (cell)}
						<button
							type="button"
							class="rounded-md border p-2 text-left text-xs transition-colors {activeCell === cell
								? 'border-accent bg-accent-soft'
								: 'border-border bg-surface hover:border-border-strong'}"
							onclick={() => (activeCell = activeCell === cell ? null : cell)}
						>
							<div class="font-medium text-ink">{cellTitle(cell)}</div>
							<div class="mt-1 font-mono text-lg font-semibold text-ink-2">{attribution[cell]}</div>
							{#if activeCell === cell}<div class="mt-1 text-[10px] text-accent-soft-text">filtering…</div>{/if}
						</button>
					{/each}
				</div>
			</div>
		{/if}

		<!-- Per-question table -->
		<div class="p-4">
			<div class="mb-3 flex flex-wrap items-center gap-2">
				<h4 class="text-sm font-semibold text-ink">Questions <span class="font-normal text-faint">({filteredRows.length})</span></h4>
				<div class="ml-auto flex flex-wrap items-center gap-2">
					<select
						bind:value={capFilter}
						class="rounded-md border border-border bg-surface px-2 py-1 text-xs outline-none focus:border-accent"
					>
						<option value="">all capabilities</option>
						{#each capabilities as c (c)}
							<option value={c}>{c}</option>
						{/each}
					</select>
					<select
						bind:value={verdictFilter}
						class="rounded-md border border-border bg-surface px-2 py-1 text-xs outline-none focus:border-accent"
					>
						<option value="">all verdicts</option>
						{#each verdicts as v (v)}
							<option value={v}>{v}</option>
						{/each}
					</select>
					{#if activeCell || capFilter || verdictFilter}
						<button type="button" class="text-xs text-accent hover:underline" onclick={clearFilter}>clear filters</button>
					{/if}
				</div>
			</div>
			{#if filteredRows.length === 0}
				<p class="text-sm text-faint">No questions match this filter.</p>
			{:else}
				<div class="overflow-x-auto rounded-lg border border-border">
					<table class="w-full min-w-[760px] text-left text-sm">
						<thead class="bg-surface-muted text-xs uppercase tracking-wide text-faint">
							<tr>
								<th class="px-3 py-2 font-medium">Question</th>
								<th class="px-3 py-2 font-medium">Capability</th>
								<th class="px-3 py-2 font-medium">Verdict</th>
								<th class="px-3 py-2 text-right font-medium">Source</th>
								<th class="px-3 py-2 text-right font-medium">Coverage</th>
								<th class="px-3 py-2 text-right font-medium">Rounds</th>
								<th class="px-3 py-2 text-right font-medium">Latency</th>
								<th class="px-3 py-2 font-medium"></th>
							</tr>
						</thead>
						<tbody class="divide-y divide-border">
							{#each filteredRows as r (r.question_id)}
								{@const open = expandedRow === r.question_id}
								<tr class="align-top">
									<td class="max-w-[300px] px-3 py-2 text-ink">
										<span class="line-clamp-2">{r.question_text}</span>
										<span class="text-[11px] text-faint">{r.difficulty}</span>
									</td>
									<td class="px-3 py-2 font-mono text-xs text-faint">{r.capability}</td>
									<td class="px-3 py-2 font-mono text-xs font-medium {verdictClass(r.verdict)}">{verdictLabel(r.verdict)}</td>
									<td class="px-3 py-2 text-right font-mono text-xs text-faint">{sourceRankText(r)}</td>
									<td class="px-3 py-2 text-right font-mono text-xs text-faint">{fmtPct(r.source_coverage)}</td>
									<td class="px-3 py-2 text-right font-mono text-xs text-faint">{r.rounds}</td>
									<td class="px-3 py-2 text-right font-mono text-xs text-faint">{r.latency_ms > 0 ? `${r.latency_ms.toFixed(0)} ms` : '—'}</td>
									<td class="px-3 py-2 text-right">
										<button type="button" class="text-xs text-accent hover:underline" onclick={() => (expandedRow = open ? null : r.question_id)}>
											{open ? 'Hide' : 'Open'}
										</button>
									</td>
								</tr>
								{#if open}
									<tr>
										<td colspan="8" class="bg-surface-sunken px-3 py-3">
											<div class="grid grid-cols-1 gap-4 min-[640px]:grid-cols-2">
												<div>
													<div class="mb-1 text-xs font-semibold text-faint">Answer</div>
													<div class="text-sm text-ink-2">{r.answer_text || (r.abstained ? '(abstained)' : '—')}</div>
												</div>
												<div>
													<div class="mb-1 text-xs font-semibold text-faint">Expected</div>
													<div class="text-sm text-ink-2">{r.expected_answer || '—'}</div>
												</div>
											</div>
											{#if r.verdict_reason}
												<div class="mt-3">
													<div class="mb-1 text-xs font-semibold text-faint">Judge reason</div>
													<div class="text-sm text-ink-2">{r.verdict_reason}</div>
												</div>
											{/if}
											{#if r.retrieved_paths.length > 0}
												<div class="mt-3">
													<div class="mb-1 text-xs font-semibold text-faint">Retrieved paths</div>
													<ul class="space-y-1 font-mono text-xs text-ink-2">
														{#each r.retrieved_paths as path, i (i)}
															<li class="flex items-center gap-2">
																<span class="w-6 shrink-0 text-right text-faint">{i + 1}</span>
																<span class="{r.source_hit_rank != null && i + 1 === r.source_hit_rank ? 'font-semibold text-accent-soft-text' : ''}">{path}</span>
																{#if r.source_hit_rank != null && i + 1 === r.source_hit_rank}
																	<span class="rounded-full bg-accent-soft px-1.5 py-0.5 text-[10px] font-medium text-accent-soft-text">first gold hit</span>
																{/if}
															</li>
														{/each}
													</ul>
												</div>
											{/if}
											{#if r.error}
												<div class="mt-3 text-xs text-danger">error: {r.error}</div>
											{/if}
										</td>
									</tr>
								{/if}
							{/each}
						</tbody>
					</table>
				</div>
			{/if}
		</div>
	{/if}
</div>
