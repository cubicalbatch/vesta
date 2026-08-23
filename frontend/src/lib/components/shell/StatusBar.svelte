<script lang="ts">
	// The permanent, honest answer to "what is this thing actually running?"
	// Sourced from
	// /health + /api/zims + /api/settings — never claims a capability the
	// system doesn't have.
	//
	// Counts articles, never passages/chunks: the backend only ever exposes
	// ArchiveOut.article_count (a passage/chunk count is an internal indexing
	// detail, not a stable user-facing number), so we render "articles" rather
	// than inventing a passage count.
	import { healthStore } from '$lib/stores/health.svelte';
	import { zimsStore } from '$lib/stores/zims.svelte';
	import { settingsValuesStore } from '$lib/stores/settings.svelte';
	import { modelStore } from '$lib/stores/model.svelte';
	import { formatCount } from '$lib/format';

	const ok = $derived(healthStore.data?.status === 'ok');
	const archiveCount = $derived(zimsStore.enabled.length);
	const articleCount = $derived(zimsStore.totalArticles);
	const model = $derived(
		modelStore.status?.display_name || modelStore.status?.model_file || null
	);
	const profile = $derived(settingsValuesStore.values['retrieval.active_profile'] || null);
</script>

<div
	class="mt-9 flex flex-wrap items-center gap-x-3 gap-y-2 border-t border-border pt-4 font-mono text-xs text-faint"
>
	<span class="inline-flex items-center gap-1 {ok ? 'text-success' : 'text-warning'}">
		<span class="size-1.5 rounded-full bg-current"></span>
		{ok ? 'offline' : (healthStore.data?.status ?? 'connecting…')}
	</span>
	{#if zimsStore.loaded}
		<span class="opacity-50">·</span>
		<span>{archiveCount} archive{archiveCount === 1 ? '' : 's'} · {formatCount(articleCount)} articles</span>
	{/if}
	{#if healthStore.loaded}
		<span class="opacity-50">·</span>
		<span>{healthStore.has('llm') && model ? model : 'no model configured'}</span>
	{/if}
	{#if profile}
		<span class="opacity-50">·</span>
		<span>profile {profile}</span>
	{/if}
</div>
