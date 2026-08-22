<script lang="ts">
	// Steady green when idle, pulsing amber when any job is non-terminal, red
	// when the most recent terminal job errored. Links to the Settings → Jobs
	// tab.
	import { jobsStore } from '$lib/stores/jobs.svelte';

	const state = $derived(jobsStore.dotState);
	const label = $derived(
		state === 'active'
			? `${jobsStore.running.length} job${jobsStore.running.length === 1 ? '' : 's'} running`
			: state === 'error'
				? 'last job failed'
				: 'idle'
	);
</script>

<a
	href="/settings?tab=jobs"
	class="relative inline-grid size-9 place-items-center"
	title="Jobs: {label}"
	aria-label="Jobs: {label}"
>
	<span
		class="size-[9px] rounded-full {state === 'active'
			? 'animate-pulse bg-amber shadow-[0_0_0_3px_color-mix(in_srgb,var(--color-amber)_22%,transparent)]'
			: state === 'error'
				? 'bg-danger shadow-[0_0_0_3px_color-mix(in_srgb,var(--color-danger)_22%,transparent)]'
				: 'bg-success shadow-[0_0_0_3px_color-mix(in_srgb,var(--color-success)_18%,transparent)]'}"
	></span>
</a>
