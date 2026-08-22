<script lang="ts">
	// Search — the landing page, now the single surface for both modes
	// The "Use AI" toggle inside SearchPage
	// switches the same query between GET /api/search and POST /api/chat without
	// a navigation. All URL state and the zero-archive /welcome redirect live in
	// SearchPage; this route just hosts the title + the component.
	import SearchPage from '$lib/components/search/SearchPage.svelte';
</script>

<svelte:head>
	<title>Search - Vesta</title>
</svelte:head>

<SearchPage />
