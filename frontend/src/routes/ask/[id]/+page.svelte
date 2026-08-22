<script lang="ts">
	// Back-compat shim:
	// `/ask/{id}` → `/?c={id}&ai=1`, restoring the whole thread from the API.
	// replaceState keeps browser history clean. The id is carried verbatim — a
	// non-numeric id (e.g. /ask/abc123) lands on / with c dropped to null and a
	// fresh AI hero, which keeps test_spa.py's fallback assertion meaningful.
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { goto } from '$app/navigation';

	onMount(() => {
		goto(`/?c=${page.params.id}&ai=1`, { replaceState: true });
	});
</script>

<svelte:head>
	<title>Search - Vesta</title>
</svelte:head>
