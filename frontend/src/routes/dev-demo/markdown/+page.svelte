<script lang="ts">
	// Dev-only demo route. adapter-static ships every route's JS to production,
	// so the harness itself (and the recorded/synthetic SSE fixtures it
	// statically imports) lives in MarkdownDemo.svelte, pulled in through a
	// DEV-gated dynamic import: `import.meta.env.DEV` is statically false in
	// `npm run build`, so the branch — and the fixture chunks — are eliminated
	// from the production bundle. Not linked from the app nav.
	import type { Component } from 'svelte';

	let Demo = $state<Component | null>(null);

	if (import.meta.env.DEV) {
		import('./MarkdownDemo.svelte').then((m) => (Demo = m.default));
	}
</script>

{#if Demo}
	<Demo />
{/if}
