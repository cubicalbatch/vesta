<script lang="ts">
	// The two-step framing that runs across the whole page:
	//   1. Choose archives  ->  2. Set up AI (optional)
	// The download/index chain is now a live progress section, not a step.
	let { current }: { current: 1 | 2 } = $props();

	const steps = [
		{ n: 1 as const, title: 'Choose archives', desc: 'Pick what to install. Wikipedia, Stack Exchange, and more.' },
		{ n: 2 as const, title: 'Set up AI', desc: 'Optional: add an LLM for cited answers.' }
	];
</script>

<div class="grid grid-cols-1 gap-3 min-[721px]:grid-cols-2">
	{#each steps as s (s.n)}
		<div
			class="flex items-start gap-3 rounded-lg border p-3 {s.n === current
				? 'border-accent/40 bg-accent-soft'
				: 'border-border bg-surface'}"
		>
			<span
				class="inline-grid size-6 shrink-0 place-items-center rounded-full text-xs font-semibold {s.n === current
					? 'bg-accent text-white'
					: s.n < current
						? 'bg-success text-white'
						: 'bg-surface-muted text-muted'}"
			>
				{s.n < current ? '✓' : s.n}
			</span>
			<div>
				<div class="text-sm font-medium text-ink">{s.title}</div>
				<p class="text-xs text-muted">{s.desc}</p>
			</div>
		</div>
	{/each}
</div>
