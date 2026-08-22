<script lang="ts">
	// Back-compat shim:
	// `/ask` is no longer a destination — Ask is now a *mode* of `/`. Preserve
	// every history entry and bookmark by redirecting to the
	// unified surface in AI mode. replaceState keeps browser history clean (no
	// junk entry between /ask and /). Costs ~10 lines; do not delete this phase.
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { goto } from '$app/navigation';

	onMount(() => {
		const params = new URLSearchParams();
		const q = page.url.searchParams.get('q');
		const scope = page.url.searchParams.get('scope');
		if (q) params.set('q', q);
		if (scope) params.set('scope', scope);
		params.set('ai', '1');
		goto(`/?${params}`, { replaceState: true });
	});
</script>

<svelte:head>
	<title>Search - Vesta</title>
</svelte:head>
