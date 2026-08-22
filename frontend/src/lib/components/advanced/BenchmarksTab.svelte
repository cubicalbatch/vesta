<script lang="ts">
	// Advanced → Benchmarks tab — the unified benchmark page.
	// Replaces the old single-run BenchmarksTab (which called the removed
	// /api/benchmark router). The only GUI for starting and viewing benchmarks.
	//
	// Sections:
	//   1. Run form — systems/profiles/models (multi-select), slice/capability/
	//      difficulty filters, scope/limit/repeats/judge/label. Shows the matrix
	//      size and an estimated wall time before pressing Run (trap 11).
	//   2. Run list grouped by run_group, with score chips + a compare checkbox.
	//      Expanding a run renders BenchRunDetail; 2+ checked enables Compare.
	//   3. Compare view (BenchCompareView) when Compare is clicked.
	import { benchApi } from '$lib/api/bench';
	import { retrievalApi } from '$lib/api/retrieval';
	import { formatDate } from '$lib/format';
	import { estimateWallTimeSeconds, formatSeconds, matrixSize } from '$lib/bench';
	import type { BenchRunSummary, ProfileItem } from '$lib/types';
	import BenchRunDetail from './BenchRunDetail.svelte';
	import BenchCompareView from './BenchCompareView.svelte';

	const SYSTEMS = [
		'retrieval_only',
		'sources_only',
		'agentic_pydantic',
		'oracle',
		'closed_book'
	];

	// Known answer-model ids served by the live gateway (context: the default
	// endpoint). Free text is always allowed; these just autocomplete.
	const KNOWN_MODELS = [
		'unsloth/qwen3.5-4b',
		'openai/gpt-oss-20b',
		'qwen3.5-9b',
		'qwen3.6-27b',
		'qwen3.6-35b',
		'gemma-4'
	];

	const SLICES = ['core', 'all'];

	// ── Run list state ────────────────────────────────────────────────────────
	let runs = $state<BenchRunSummary[]>([]);
	let loading = $state(true);
	let loadError = $state<string | null>(null);
	let expanded = $state<number | null>(null);
	let selectedForCompare = $state<number[]>([]);
	let showCompare = $state(false);
	let pollTimer: ReturnType<typeof setInterval> | null = null;

	// ── Run form state ────────────────────────────────────────────────────────
	let profiles = $state<ProfileItem[]>([]);
	let datasetCaps = $state<string[]>([]);
	let datasetTotal = $state(0);
	let sliceCounts = $state<Record<string, number>>({});

	let systems = $state(['agentic_pydantic']);
	let selectedProfiles = $state<string[]>([]);
	let modelsText = $state('');
	let slice = $state('core');
	let capabilities = $state<string[]>([]);
	let scope = $state('');
	let limit = $state('');
	let repeats = $state('1');
	let judgeModel = $state('');
	let label = $state('');
	let starting = $state(false);
	let startError = $state<string | null>(null);

	async function loadRuns() {
		try {
			runs = await benchApi.listRuns();
			loadError = null;
		} catch (err) {
			loadError = err instanceof Error ? err.message : 'failed to load benchmark runs';
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
		benchApi
			.dataset()
			.then((d) => {
				datasetCaps = Object.keys(d.by_capability).sort();
				datasetTotal = d.total;
				sliceCounts = d.by_slice;
			})
			.catch(() => {});
		return () => {
			if (pollTimer) clearInterval(pollTimer);
		};
	});

	function toggleSystem(s: string) {
		systems = systems.includes(s) ? systems.filter((x) => x !== s) : [...systems, s];
	}

	function toggleProfile(p: string) {
		selectedProfiles = selectedProfiles.includes(p)
			? selectedProfiles.filter((x) => x !== p)
			: [...selectedProfiles, p];
	}

	function toggleCapability(c: string) {
		capabilities = capabilities.includes(c) ? capabilities.filter((x) => x !== c) : [...capabilities, c];
	}

	const modelList = $derived(
		modelsText
			.split(/[\s,]+/)
			.map((m) => m.trim())
			.filter(Boolean)
	);

	const matrixN = $derived(matrixSize(systems.length, selectedProfiles.length || 1, modelList.length || 1));

	// Question-count estimate for the wall-time: prefer the slice count when a
	// specific slice is chosen, else the dataset total; an explicit limit caps it.
	const estimatedQuestions = $derived.by(() => {
		let n = slice && slice !== 'all' ? (sliceCounts[slice] ?? datasetTotal) : datasetTotal;
		if (capabilities.length > 0) n = Math.min(n, datasetTotal); // rough upper bound
		const lim = Number(limit);
		if (Number.isFinite(lim) && lim > 0) n = Math.min(n, Math.floor(lim));
		return Math.max(0, n);
	});

	const estWallSeconds = $derived(estimateWallTimeSeconds(matrixN, estimatedQuestions));

	// Runs grouped by run_group, preserving newest-first order.
	const groups = $derived.by(() => {
		const out: { run_group: string; runs: BenchRunSummary[] }[] = [];
		const seen = new Set<string>();
		for (const r of runs) {
			if (!seen.has(r.run_group)) {
				seen.add(r.run_group);
				out.push({ run_group: r.run_group, runs: [] });
			}
			out[out.length - 1].runs.push(r);
		}
		return out;
	});

	function toggleCompare(id: number) {
		selectedForCompare = selectedForCompare.includes(id)
			? selectedForCompare.filter((x) => x !== id)
			: [...selectedForCompare, id];
	}

	function openCompare() {
		showCompare = true;
	}

	function closeCompare() {
		showCompare = false;
		selectedForCompare = [];
	}

	async function startRun() {
		starting = true;
		startError = null;
		try {
			const profilesSent = selectedProfiles.length > 0 ? selectedProfiles : undefined;
			const modelsSent = modelList.length > 0 ? modelList : undefined;
			const res = await benchApi.run({
				systems: systems.length > 0 ? systems : undefined,
				profiles: profilesSent,
				models: modelsSent,
				slice: slice || undefined,
				capabilities: capabilities.length > 0 ? capabilities : undefined,
				limit: limit ? Number(limit) : undefined,
				scope: scope || undefined,
				judge_model: judgeModel || undefined,
				repeats: repeats ? Number(repeats) : undefined,
				label: label || undefined
			});
			await loadRuns();
			if (pollTimer) clearInterval(pollTimer);
			pollTimer = setInterval(async () => {
				await loadRuns();
				const done = runs.every((r) => r.status !== 'running');
				if (done && pollTimer) {
					clearInterval(pollTimer);
					pollTimer = null;
				}
			}, 2000);
		} catch (err) {
			startError = err instanceof Error ? err.message : 'failed to start benchmark run';
		} finally {
			starting = false;
		}
	}

	function fmtPct(v: number | null | undefined): string {
		return v == null ? '—' : `${(v * 100).toFixed(1)}%`;
	}

	function statusBadgeClass(status: string): string {
		switch (status) {
			case 'running':
				return 'text-accent';
			case 'aborted':
				return 'text-danger';
			default:
				return 'text-faint';
		}
	}

	function chips(r: BenchRunSummary) {
		return [
			{ label: 'headroom', value: fmtPct(r.headroom) },
			{ label: 'strict', value: fmtPct(r.strict_accuracy) },
			{ label: 'recall@10', value: fmtPct(r.source_recall_at_10) },
			{ label: 'halluc', value: fmtPct(r.hallucination_rate) }
		];
	}

	const selectedRuns = $derived(runs.filter((r) => selectedForCompare.includes(r.id)));
</script>

<div>
	<div class="mb-1 flex items-center justify-between">
		<h2 class="text-lg font-semibold text-ink">Benchmarks</h2>
	</div>
	<p class="mb-4 text-sm text-muted">
		The unified end-to-end answer benchmark — real questions through the real pipeline, judged by an LLM, with the
		oracle ceiling and closed-book floor for reference. This is what decides whether a retrieval, profile or prompt
		change actually helped.
	</p>

	<!-- Run form -->
	<form class="mb-6 rounded-lg border border-border bg-surface p-4" onsubmit={(e) => { e.preventDefault(); startRun(); }}>
		<div class="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
			<div>
				<div class="mb-1 text-xs font-medium text-faint">Systems</div>
				<div class="flex flex-wrap gap-1.5">
					{#each SYSTEMS as s (s)}
						<label class="flex cursor-pointer items-center gap-1 rounded-full border border-border px-2 py-1 text-xs {systems.includes(s) ? 'border-accent bg-accent-soft text-accent-soft-text' : 'text-muted hover:border-border-strong'}">
							<input type="checkbox" class="sr-only" checked={systems.includes(s)} onchange={() => toggleSystem(s)} />
							{s}
						</label>
					{/each}
				</div>
			</div>
			<div>
				<div class="mb-1 text-xs font-medium text-faint">Profiles</div>
				<div class="flex flex-wrap gap-1.5">
					{#each profiles as p (p.name)}
						<label class="flex cursor-pointer items-center gap-1 rounded-full border border-border px-2 py-1 text-xs {selectedProfiles.includes(p.name) ? 'border-accent bg-accent-soft text-accent-soft-text' : 'text-muted hover:border-border-strong'}">
							<input type="checkbox" class="sr-only" checked={selectedProfiles.includes(p.name)} onchange={() => toggleProfile(p.name)} />
							{p.name}
						</label>
					{/each}
					{#if profiles.length === 0}
						<span class="text-xs text-faint">(none loaded — uses active profile)</span>
					{/if}
				</div>
			</div>
			<div>
				<label class="block text-xs font-medium text-faint" for="bench-models">Models (blank = default)</label>
				<input
					id="bench-models"
					name="bench-models"
					type="text"
					bind:value={modelsText}
					list="bench-known-models"
					placeholder="unsloth/qwen3.5-4b, openai/gpt-oss-20b"
					class="mt-1 w-full rounded-md border border-border bg-surface px-2 py-1.5 text-sm outline-none focus:border-accent"
				/>
				<datalist id="bench-known-models">
					{#each KNOWN_MODELS as m (m)}
						<option value={m}></option>
					{/each}
				</datalist>
			</div>
			<div>
				<label class="block text-xs font-medium text-faint" for="bench-slice">Slice</label>
				<select id="bench-slice" name="bench-slice" bind:value={slice} class="mt-1 w-full rounded-md border border-border bg-surface px-2 py-1.5 text-sm outline-none focus:border-accent">
					{#each SLICES as s (s)}
						<option value={s}>{s}</option>
					{/each}
				</select>
			</div>
			<div>
				<div class="mb-1 text-xs font-medium text-faint">Capabilities</div>
				<div class="flex max-h-24 flex-wrap gap-1.5 overflow-y-auto">
					{#each datasetCaps as c (c)}
						<label class="flex cursor-pointer items-center gap-1 rounded-full border border-border px-2 py-0.5 text-xs {capabilities.includes(c) ? 'border-accent bg-accent-soft text-accent-soft-text' : 'text-muted hover:border-border-strong'}">
							<input type="checkbox" class="sr-only" checked={capabilities.includes(c)} onchange={() => toggleCapability(c)} />
							{c}
						</label>
					{/each}
					{#if datasetCaps.length === 0}
						<span class="text-xs text-faint">(loading…)</span>
					{/if}
				</div>
			</div>
			<div>
				<label class="block text-xs font-medium text-faint" for="bench-scope">Scope (ZIM, blank = default)</label>
				<input id="bench-scope" name="bench-scope" type="text" bind:value={scope} placeholder="optional" class="mt-1 w-full rounded-md border border-border bg-surface px-2 py-1.5 text-sm outline-none focus:border-accent" />
			</div>
			<div class="grid grid-cols-2 gap-3">
				<div>
					<label class="block text-xs font-medium text-faint" for="bench-limit">Limit</label>
					<input id="bench-limit" name="bench-limit" type="number" min="0" bind:value={limit} placeholder="all" class="mt-1 w-full rounded-md border border-border bg-surface px-2 py-1.5 text-sm outline-none focus:border-accent" />
				</div>
				<div>
					<label class="block text-xs font-medium text-faint" for="bench-repeats">Repeats</label>
					<input id="bench-repeats" name="bench-repeats" type="number" min="1" bind:value={repeats} class="mt-1 w-full rounded-md border border-border bg-surface px-2 py-1.5 text-sm outline-none focus:border-accent" />
				</div>
			</div>
			<div>
				<label class="block text-xs font-medium text-faint" for="bench-judge">Judge model</label>
				<input id="bench-judge" name="bench-judge" type="text" bind:value={judgeModel} placeholder="blank = default" class="mt-1 w-full rounded-md border border-border bg-surface px-2 py-1.5 text-sm outline-none focus:border-accent" />
			</div>
			<div>
				<label class="block text-xs font-medium text-faint" for="bench-label">Label</label>
				<input id="bench-label" name="bench-label" type="text" bind:value={label} placeholder="optional" class="mt-1 w-full rounded-md border border-border bg-surface px-2 py-1.5 text-sm outline-none focus:border-accent" />
			</div>
		</div>

		<div class="mt-4 flex flex-wrap items-center gap-3">
			{#if systems.length > 0}
				<div class="rounded-md bg-surface-muted px-3 py-2 text-xs text-faint">
					{matrixN} run{matrixN === 1 ? '' : 's'} over ~{estimatedQuestions} question{estimatedQuestions === 1 ? '' : 's'} ·
					estimated {formatSeconds(estWallSeconds)}
				</div>
			{:else}
				<div class="text-xs text-danger">Pick at least one system.</div>
			{/if}
			<button type="submit" class="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-40" disabled={starting || systems.length === 0}>
				{starting ? 'Starting…' : 'Run benchmark'}
			</button>
			{#if showCompare}
				<button type="button" class="rounded-md border border-border px-3 py-1.5 text-sm text-muted hover:bg-surface-muted" onclick={closeCompare}>Close compare</button>
			{/if}
		</div>
		{#if startError}<p class="mt-3 text-xs text-danger">{startError}</p>{/if}
	</form>

	<!-- Compare selection bar -->
	{#if !showCompare}
		<div class="mb-4 flex items-center gap-2">
			<button
				type="button"
				class="rounded-md border border-border px-3 py-1.5 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-40 {selectedForCompare.length >= 2 ? 'bg-accent text-white hover:bg-accent-hover' : 'text-muted'}"
				disabled={selectedForCompare.length < 2}
				onclick={openCompare}
			>
				Compare {selectedForCompare.length} run{selectedForCompare.length === 1 ? '' : 's'}
			</button>
			{#if selectedForCompare.length > 0}
				<button type="button" class="text-xs text-accent hover:underline" onclick={() => (selectedForCompare = [])}>clear</button>
			{/if}
			<span class="text-xs text-faint">check 2+ runs to compare</span>
		</div>
	{/if}

	{#if showCompare}
		{#if selectedRuns.length >= 2}
			<BenchCompareView runs={selectedRuns} onClose={closeCompare} />
		{:else}
			<p class="text-sm text-faint">Select at least two runs to compare.</p>
		{/if}
	{:else if loading}
		<p class="text-sm text-muted">Loading…</p>
	{:else if loadError}
		<p class="text-sm text-danger">{loadError}</p>
	{:else if runs.length === 0}
		<p class="text-sm text-muted">No benchmark runs yet.</p>
	{:else}
		<div class="space-y-6">
			{#each groups as group (group.run_group)}
				<div class="rounded-lg border border-border bg-surface">
					<div class="flex flex-wrap items-center gap-2 border-b border-border px-4 py-2">
						<span class="font-mono text-xs text-faint">{group.run_group.slice(0, 8)}</span>
						<span class="text-xs text-faint">{group.runs.length} run{group.runs.length === 1 ? '' : 's'}</span>
						<span class="ml-auto text-xs text-faint">{formatDate(group.runs[0].started_at)}</span>
					</div>
					<div class="overflow-x-auto">
						<table class="w-full min-w-[820px] text-left text-sm">
							<thead class="bg-surface-muted text-xs uppercase tracking-wide text-faint">
								<tr>
									<th class="w-10 px-3 py-2"></th>
									<th class="px-3 py-2 font-medium">Run</th>
									<th class="px-3 py-2 font-medium">System</th>
									<th class="px-3 py-2 text-right font-medium">Headroom</th>
									<th class="px-3 py-2 text-right font-medium">Strict</th>
									<th class="px-3 py-2 text-right font-medium">Recall@10</th>
									<th class="px-3 py-2 text-right font-medium">Halluc.</th>
									<th class="px-3 py-2 font-medium"></th>
								</tr>
							</thead>
							<tbody class="divide-y divide-border">
								{#each group.runs as r (r.id)}
									<tr class="align-top {expanded === r.id ? 'bg-surface-muted/50' : ''}">
										<td class="px-3 py-2">
											<input
												type="checkbox"
												class="accent-accent"
												checked={selectedForCompare.includes(r.id)}
												onchange={() => toggleCompare(r.id)}
											/>
										</td>
										<td class="px-3 py-2">
											<div class="flex items-center gap-2">
												<span class="text-ink">#{r.id}</span>
												{#if r.label}<span class="text-xs text-faint">{r.label}</span>{/if}
												{#if !r.trusted}
													<span class="rounded-full border border-warning bg-warning-soft px-1.5 py-0.5 text-[10px] font-semibold text-warning">untrusted</span>
												{/if}
											</div>
											<div class="mt-0.5 font-mono text-[11px] text-faint">
												{formatDate(r.started_at)} · {r.status}
											</div>
										</td>
										<td class="px-3 py-2">
											<div class="text-ink-2">{r.system}</div>
											<div class="font-mono text-[11px] text-faint">{r.answer_model}</div>
										</td>
										{#each chips(r) as chip (chip.label)}
											<td class="px-3 py-2 text-right font-mono text-xs text-ink-2" title={chip.label}>
												{chip.value}
											</td>
										{/each}
										<td class="px-3 py-2 text-right">
											<button type="button" class="text-xs text-accent hover:underline" onclick={() => (expanded = expanded === r.id ? null : r.id)}>
												{expanded === r.id ? 'Hide' : 'Open'}
											</button>
										</td>
									</tr>
									{#if expanded === r.id}
										<tr>
											<td colspan="9" class="bg-surface-sunken px-4 py-4">
												{#if r.status === 'running'}
													<p class="text-xs text-faint">Run in progress — refresh to see scores.</p>
												{/if}
												<BenchRunDetail runId={r.id} />
											</td>
										</tr>
									{/if}
								{/each}
							</tbody>
						</table>
					</div>
				</div>
			{/each}
		</div>
	{/if}
</div>