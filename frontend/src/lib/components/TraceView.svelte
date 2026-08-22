<script lang="ts">
	// Renders any trace shape the answer stream carries, including shapes
	// preserved in saved conversations and the recorded regression fixtures.
	//   - RetrievalTrace (sources_only today; single_shot and GET /api/search
	//     survive in saved/historical data): a versioned, stages-based trace
	//     (vesta/retrieval/trace.py).
	//   - AgentTrace (POST /api/chat pydantic-ai agent): a flat
	//     summary plus a per-step timing breakdown — pre_seed / agent_llm /
	//     search / read_article, each search step carrying nested retrieval
	//     stages (candidate_source, static_pass, cross_encoder, …) so the
	//     user can see where wall-clock time actually went.
	// A viewer that assumes one shape silently throws on the other and — because
	// this renders inside the streaming AskTurn — corrupts that subtree's
	// reactivity (the trace lands last, mid-flush). Discriminate instead.
	// Trace view.
	import type { AgentTrace, RetrievalTrace } from '$lib/types';
	import KeyValueTree from './KeyValueTree.svelte';

	let { trace }: { trace: RetrievalTrace | AgentTrace } = $props();

	function isAgent(t: RetrievalTrace | AgentTrace): t is AgentTrace {
		return 'system' in t;
	}

	function fmt(ms: number): string {
		if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
		return `${ms.toFixed(1)}ms`;
	}

	function pctOf(stepMs: number, totalMs: number): number {
		if (totalMs <= 0) return 0;
		return Math.min(100, (stepMs / totalMs) * 100);
	}
</script>

<div class="font-mono text-xs leading-normal text-ink-2">
	{#if isAgent(trace)}
		<!-- AgentTrace: flat summary + timing breakdown. -->
		<div class="mb-3 flex flex-wrap gap-x-4 gap-y-1 text-faint">
			<span>system <span class="text-ink">{trace.system}</span></span>
			<span>{fmt(trace.elapsed_ms)} total</span>
			<span>{trace.total_tokens.toLocaleString()} tokens ({trace.input_tokens.toLocaleString()} in / {trace.output_tokens.toLocaleString()} out)</span>
			{#if trace.search_calls > 0 || trace.read_calls > 0}
				<span>{trace.search_calls} search · {trace.read_calls} read</span>
			{/if}
			<span>{trace.card_count} source{trace.card_count === 1 ? '' : 's'}</span>
		</div>

		{#if trace.stages && trace.stages.length > 0}
			<div class="mb-2 font-semibold text-ink">Timing</div>
			<div class="flex flex-col gap-1.5">
				{#each trace.stages as step, i (i)}
					<div class="rounded-md border border-border bg-surface-sunken p-2">
						<div class="flex items-baseline gap-2">
							<span class="text-ink">{step.name}</span>
							<span class="text-faint">/ {step.component}</span>
							<span class="ml-auto text-faint">{fmt(step.duration_ms)}</span>
						</div>
						<div class="mt-1 h-1 w-full overflow-hidden rounded-full bg-surface-muted">
							<div
								class="h-full rounded-full bg-accent"
								style="width: {pctOf(step.duration_ms, trace.elapsed_ms)}%"
							></div>
						</div>
						{#if step.stages && step.stages.length > 0}
							<div class="mt-2 border-t border-border pt-1.5">
								{#each step.stages as rs, j (j)}
									<div class="flex items-baseline gap-2 py-0.5">
										<span class="text-faint">{rs.name}</span>
										<span class="text-faint/70">/ {rs.component}</span>
										<span class="ml-auto text-faint">{fmt(rs.duration_ms)}</span>
									</div>
								{/each}
							</div>
						{/if}
					</div>
				{/each}
			</div>
		{/if}

		<KeyValueTree value={trace} />
	{:else}
		<!-- RetrievalTrace: versioned, stages-based. -->
		<div class="mb-3 flex flex-wrap gap-x-4 gap-y-1 text-faint">
			<span>profile <span class="text-ink">{trace.profile}</span></span>
			{#if trace.profile_hash}<span>hash {trace.profile_hash.slice(0, 12)}</span>{/if}
			<span>{trace.stages.length} stage{trace.stages.length === 1 ? '' : 's'}</span>
		</div>

		{#if trace.degradations.length > 0}
			<div class="mb-4 rounded-md border border-warning/30 bg-warning-soft p-3">
				<div class="mb-1 font-semibold text-warning">Degradations</div>
				{#each trace.degradations as d, i (i)}
					<div class="mt-1 first:mt-0">
						<span class="text-warning">{d.component}</span>
						<span class="text-faint"> missing {d.missing} — {d.reason}</span>
					</div>
				{/each}
			</div>
		{/if}

		<div class="flex flex-col gap-2">
			{#each trace.stages as stage, i (i)}
				<details class="rounded-md border border-border bg-surface-sunken p-2">
					<summary class="cursor-pointer select-none">
						<span class="text-ink">{stage.name}</span>
						<span class="text-faint"> / {stage.component}</span>
						<span class="ml-2 text-faint">{stage.duration_ms.toFixed(1)}ms</span>
					</summary>
					<div class="mt-2 space-y-2 border-t border-border pt-2">
						<div><span class="text-faint">params</span> <KeyValueTree value={stage.params} /></div>
						<div><span class="text-faint">inputs</span> <KeyValueTree value={stage.inputs} /></div>
						<div><span class="text-faint">outputs</span> <KeyValueTree value={stage.outputs} /></div>
					</div>
				</details>
			{/each}
		</div>
	{/if}
</div>
