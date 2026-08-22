<script lang="ts">
	// Composite control writing the context pair atomically
	// — the HowThoroughControl/THOROUGH_KEYS pattern applied to
	// answer.agent.context_profile + inference.local.context_size. There is
	// no separate "mode" field: the selected preset is *derived* from the
	// shared draft pair, so editing either key anywhere (here, the AI
	// section's context-window select, or the raw fields under "All
	// settings") makes this control read as Custom automatically.
	//
	// Presets always write BOTH keys locally (setting one without the other
	// a small window under a fat budget crashes the
	// turn into the no-tool fallback); on a remote source the window setting
	// doesn't apply, so presets write the profile only and the note says
	// why. All logic lives in lib/context-profile.ts (pure, unit-tested).
	import type { SettingSchemaItem } from '$lib/types';
	import { formatBytes } from '$lib/format';
	import { modelStore } from '$lib/stores/model.svelte';
	import { settingsValuesStore } from '$lib/stores/settings.svelte';
	import {
		CONTEXT_PRESETS,
		CONTEXT_PROFILE_KEY,
		CONTEXT_RAM_UNKNOWN_COPY,
		CONTEXT_REMOTE_COPY,
		CONTEXT_SIZE_KEY,
		CONTEXT_WINDOW_RESTART_COPY,
		autoPlanFor,
		contextPresetWrites,
		kvBytesPerToken,
		kvCacheBytes,
		kvDeltaBytes,
		matchContextPreset
	} from '$lib/context-profile';
	import SettingField from './SettingField.svelte';

	let {
		items,
		values,
		errors,
		onChange
	}: {
		/** The 2 schema items, PROFILE/SIZE order (CONTEXT_KEYS). */
		items: SettingSchemaItem[];
		/** Current draft values for exactly those 2 keys. */
		values: Record<string, string>;
		errors: Record<string, string | null | undefined>;
		onChange: (key: string, value: string) => void;
	} = $props();

	// The same local-vs-remote conditional the AI section uses: its whole
	// local-runtime block (incl. the context-window select) renders only
	// when inference.llm.source is 'local'.
	const isLocal = $derived(
		String(settingsValuesStore.values['inference.llm.source'] ?? 'local') !== 'remote'
	);

	const profileValue = $derived(values[CONTEXT_PROFILE_KEY] ?? '');
	const draftSize = $derived(Number(values[CONTEXT_SIZE_KEY] ?? ''));

	// RAM labels from the model chip's estimator inputs:
	// weights are constant per model, so only the KV term (ctx × kv/token)
	// moves with the preset. Null rate ⇒ graceful degrade, never a guess.
	const kvRate = $derived(kvBytesPerToken(modelStore.status));
	const liveSize = $derived(modelStore.status?.context_size ?? null);

	const currentPresetId = $derived(
		matchContextPreset(profileValue, values[CONTEXT_SIZE_KEY] ?? '', isLocal)
	);

	// Same Custom reveal semantics as HowThoroughControl.
	let manuallyExpanded = $state(false);
	const showSubcontrols = $derived(manuallyExpanded || currentPresetId === 'custom');
	const displayedId = $derived(manuallyExpanded ? 'custom' : currentPresetId);

	function applyPreset(id: (typeof CONTEXT_PRESETS)[number]['id']) {
		manuallyExpanded = false;
		const preset = CONTEXT_PRESETS.find((p) => p.id === id)!;
		for (const [key, v] of Object.entries(contextPresetWrites(preset, isLocal))) onChange(key, v);
	}

	const planLine = $derived.by(() => {
		const plan =
			profileValue === 'auto' ? `auto → ${isLocal ? autoPlanFor(draftSize) : 'full'}` : profileValue;
		const window = isLocal ? ` · window ${Number.isFinite(draftSize) ? draftSize : '?'} tokens` : '';
		return `profile ${plan}${window}`;
	});

	const ramLine = $derived.by(() => {
		const kv = isLocal ? kvCacheBytes(draftSize, kvRate) : null;
		const delta = isLocal ? kvDeltaBytes(draftSize, liveSize, kvRate) : null;
		const parts: string[] = [];
		if (kv != null) parts.push(`≈ ${formatBytes(kv)} of context memory`);
		if (delta != null) {
			parts.push(
				delta > 0
					? `saves ≈ ${formatBytes(delta)} vs the running window`
					: `costs ≈ ${formatBytes(-delta)} more than the running window`
			);
		}
		if (parts.length === 0) return null;
		return parts.join(' · ');
	});

	const noteLine = $derived.by(() => {
		if (!isLocal) return CONTEXT_REMOTE_COPY;
		if (kvRate == null) return `${CONTEXT_WINDOW_RESTART_COPY} ${CONTEXT_RAM_UNKNOWN_COPY}`;
		return CONTEXT_WINDOW_RESTART_COPY;
	});
</script>

<div class="border-b border-border py-3.5 last:border-0">
	<div class="flex flex-col gap-3 md:flex-row md:items-start md:justify-between md:gap-6">
		<div>
			<div class="text-sm font-medium text-ink">Answer speed &amp; memory</div>
			<p class="mt-0.5 max-w-md text-xs text-muted">
				How much the model can hold in mind while answering. Smaller windows answer faster and
				use far less memory, at some cost on the hardest questions.
			</p>
			<p class="mt-1.5 font-mono text-xs text-faint">
			{planLine}{#if ramLine}<span>&nbsp;·&nbsp;{ramLine}</span>{/if}
			</p>
		</div>
		<!-- Pills + suffixes don't fit a 320px viewport; cap the group at the
		     available width and scroll it rather than pushing the page sideways. -->
		<div
			class="flex max-w-full shrink-0 items-center gap-1 self-start overflow-x-auto rounded-full border border-border bg-surface p-1"
		>
			{#each CONTEXT_PRESETS as p (p.id)}
				{@const kvLabel = kvCacheBytes(p.sizeTokens, kvRate)}
				<button
					type="button"
					class="shrink-0 whitespace-nowrap rounded-full px-3 py-1.5 text-xs font-medium {displayedId === p.id
						? 'bg-accent-soft text-accent-soft-text'
						: 'text-muted hover:bg-surface-muted'}"
					onclick={() => applyPreset(p.id)}
				>
				{p.label}{#if isLocal && kvLabel != null}
					<span class="font-normal opacity-60">&nbsp;·&nbsp;{formatBytes(kvLabel)}</span>
				{/if}
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

	<p class="mt-1.5 text-xs text-faint">{noteLine}</p>

	{#if showSubcontrols}
		<div class="mt-3 rounded-lg border border-border bg-surface-muted/60 px-3">
			{#each items as item (item.key)}
				{#if isLocal || item.key !== CONTEXT_SIZE_KEY}
					<SettingField {item} value={values[item.key] ?? ''} error={errors[item.key] ?? null} {onChange} />
				{/if}
			{/each}
		</div>
	{/if}
</div>
