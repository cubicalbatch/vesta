<script lang="ts">
	import '../app.css';
	import favicon from '$lib/assets/favicon.svg';
	import { page } from '$app/state';
	import AppShell from '$lib/components/shell/AppShell.svelte';
	import { healthStore } from '$lib/stores/health.svelte';
	import { zimsStore } from '$lib/stores/zims.svelte';
	import { settingsValuesStore } from '$lib/stores/settings.svelte';
	import { jobsStore } from '$lib/stores/jobs.svelte';

	let { children } = $props();

	// Welcome (zero-archive first run) has no nav — the menu appears once
	// there is something to search
	// "Page: Welcome").
	const showShell = $derived((page.url.pathname as string) !== '/welcome');

	$effect(() => {
		healthStore.load();
		zimsStore.load();
		settingsValuesStore.load();
		jobsStore.start(); // one app-lifetime connection; every page reads from it
		return () => jobsStore.stop();
	});
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

{#if showShell}
	<AppShell>
		{@render children()}
	</AppShell>
{:else}
	{@render children()}
{/if}
