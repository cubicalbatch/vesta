<script lang="ts">
	// Composite control writing 4 keys at once
	// controls" → "How thorough"). Fast/Balanced/Thorough/Custom. There is no
	// separate "mode" field — `currentPresetId` is *derived* from the current
	// draft values, so editing any of the 4 underlying keys anywhere (here, or
	// as a raw field under "All settings" — same shared draft object) makes
	// this control fall out of every preset and read as Custom automatically.
	// That is what "round-trips" means here: no state to keep in sync by hand.
	import type { SettingSchemaItem } from '$lib/types';
	import SettingField from './SettingField.svelte';

	let {
		items,
		values,
		errors,
		onChange
	}: {
		/** The 2 schema items, in SHORTLIST/MAX_PER_ARTICLE order. */
		items: SettingSchemaItem[];
		/** Current draft values for exactly those 2 keys. */
		values: Record<string, string>;
		errors: Record<string, string | null | undefined>;
		onChange: (key: string, value: string) => void;
	} = $props();

	const SHORTLIST = 'retrieval.stage_b.shortlist';
	const MAX_PER_ARTICLE = 'retrieval.context.max_per_article';

	type PresetId = 'fast' | 'balanced' | 'thorough';
	const PRESETS: { id: PresetId; label: string; values: Record<string, string> }[] = [
		{
			id: 'fast',
			label: 'Fast',
			values: { [SHORTLIST]: '10', [MAX_PER_ARTICLE]: '2' }
		},
		{
			id: 'balanced',
			label: 'Balanced',
			values: { [SHORTLIST]: '20', [MAX_PER_ARTICLE]: '4' }
		},
		{
			id: 'thorough',
			label: 'Thorough',
			values: { [SHORTLIST]: '40', [MAX_PER_ARTICLE]: '6' }
		}
	];

	const currentPresetId = $derived.by((): PresetId | 'custom' => {
		for (const p of PRESETS) {
			if (
				values[SHORTLIST] === p.values[SHORTLIST] &&
				values[MAX_PER_ARTICLE] === p.values[MAX_PER_ARTICLE]
			) {
				return p.id;
			}
		}
		return 'custom';
	});

	// Lets "Custom" reveal the sub-controls even before any edit happens
	// (clicking Custom with Balanced's exact values still selected) — and
	// shows as selected itself in that case, rather than leaving the
	// still-technically-matching preset highlighted.
	let manuallyExpanded = $state(false);
	const showSubcontrols = $derived(manuallyExpanded || currentPresetId === 'custom');
	const displayedId = $derived(manuallyExpanded ? 'custom' : currentPresetId);

	function applyPreset(id: PresetId) {
		manuallyExpanded = false;
		const preset = PRESETS.find((p) => p.id === id)!;
		for (const [key, v] of Object.entries(preset.values)) onChange(key, v);
	}

	const resolvedLine = $derived.by(() => {
		const shortlist = values[SHORTLIST] ?? '?';
		const maxPerArticle = values[MAX_PER_ARTICLE] ?? '?';
		return `retrieve ${shortlist} → keep ${maxPerArticle} per article`;
	});
</script>

<div class="border-b border-border py-3.5 last:border-0">
	<div class="flex flex-col gap-3 md:flex-row md:items-start md:justify-between md:gap-6">
		<div>
			<div class="text-sm font-medium text-ink">How thorough</div>
			<p class="mt-0.5 max-w-md text-xs text-muted">
				Sets how many passages are retrieved and how many survive reranking into the
				model's context.
			</p>
			<p class="mt-1.5 font-mono text-xs text-faint">{resolvedLine}</p>
		</div>
		<!-- Four pills don't fit a 320px viewport; cap the group at the available
		     width and scroll it rather than pushing the page sideways. -->
		<div
			class="flex max-w-full shrink-0 items-center gap-1 self-start overflow-x-auto rounded-full border border-border bg-surface p-1"
		>
			{#each PRESETS as p (p.id)}
				<button
					type="button"
					class="shrink-0 whitespace-nowrap rounded-full px-3 py-1.5 text-xs font-medium {displayedId === p.id
						? 'bg-accent-soft text-accent-soft-text'
						: 'text-muted hover:bg-surface-muted'}"
					onclick={() => applyPreset(p.id)}
				>
					{p.label}
				</button>
			{/each}
			<button
				type="button"
				class="shrink-0 whitespace-nowrap rounded-full px-3 py-1.5 text-xs font-medium {displayedId === 'custom'
					? 'bg-accent-soft text-accent-soft-text'
					: 'text-muted hover:bg-surface-muted'}"
				onclick={() => (manuallyExpanded = true)}
			>
				Custom
			</button>
		</div>
	</div>

	{#if showSubcontrols}
		<div class="mt-3 rounded-lg border border-border bg-surface-muted/60 px-3">
			{#each items as item (item.key)}
				<SettingField {item} value={values[item.key] ?? ''} error={errors[item.key] ?? null} {onChange} />
			{/each}
		</div>
	{/if}
</div>
