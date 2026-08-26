<script lang="ts">
	// One generated field-group (section -> optional subsections -> fields).
	// Dumb/presentational — the page has already filtered which sections and
	// items are visible for Basic vs. All settings.
	import type { SettingSection } from '$lib/settings-groups';
	import SettingField from './SettingField.svelte';

	let {
		section,
		draft,
		errors,
		onChange
	}: {
		section: SettingSection;
		draft: Record<string, string>;
		errors: Record<string, string | null | undefined>;
		onChange: (key: string, value: string) => void;
	} = $props();
</script>

<div class="mb-6 rounded-xl border border-border bg-surface p-5">
	<h2 class="text-lg font-semibold text-ink">{section.displayName}</h2>
	{#if section.name === 'Server'}
		<p class="mb-1 mt-2 rounded-md bg-warning-soft px-3 py-2 text-xs text-warning">
			Vesta is offline-first — these only matter if this machine is reachable by someone else. There is no
			authentication layer yet: anyone who can reach the port can use the API and this UI.
		</p>
	{/if}
	<div class="mt-2">
		{#each section.subsections as sub (sub.name ?? '__flat__')}
			{#if sub.name}
				<h3 class="mb-1 mt-4 text-xs font-semibold tracking-wide text-faint uppercase first:mt-0">{sub.name}</h3>
			{/if}
			{#each sub.items as item (item.key)}
				<SettingField {item} value={draft[item.key] ?? ''} error={errors[item.key] ?? null} {onChange} />
			{/each}
		{/each}
	</div>
</div>
