<script lang="ts">
	// Generic collapsible key-value renderer for trace stage params/inputs/outputs.
	// Deliberately doesn't special-case any key names — an unknown shape must
	// still render.
	import KeyValueTree from './KeyValueTree.svelte';

	let { value }: { value: unknown } = $props();

	function isPlainObject(v: unknown): v is Record<string, unknown> {
		return typeof v === 'object' && v !== null && !Array.isArray(v);
	}
</script>

{#if value === null || value === undefined}
	<span class="text-faint">null</span>
{:else if Array.isArray(value)}
	{#if value.length === 0}
		<span class="text-faint">[]</span>
	{:else}
		<ul class="ml-4 list-none">
			{#each value as item, i (i)}
				<li><span class="text-faint">{i}:</span> <KeyValueTree value={item} /></li>
			{/each}
		</ul>
	{/if}
{:else if isPlainObject(value)}
	{@const entries = Object.entries(value)}
	{#if entries.length === 0}
		<span class="text-faint">{'{}'}</span>
	{:else}
		<dl class="ml-2">
			{#each entries as [k, v] (k)}
				<dt class="mt-2 font-semibold text-faint first:mt-0">{k}</dt>
				<dd class="ml-0 mt-0.5"><KeyValueTree value={v} /></dd>
			{/each}
		</dl>
	{/if}
{:else if typeof value === 'number'}
	<span class="text-ink">{value}</span>
{:else}
	<span class="text-accent-soft-text">{String(value)}</span>
{/if}
