<script lang="ts">
	import { page } from '$app/state';
	import ArchiveBrowsePage from '$lib/components/archive/ArchiveBrowsePage.svelte';

	const zimId = $derived(Number(page.params.zimId));
</script>

<svelte:head>
	<title>Browse - Vesta</title>
</svelte:head>

{#key zimId}
	<ArchiveBrowsePage {zimId} />
{/key}
