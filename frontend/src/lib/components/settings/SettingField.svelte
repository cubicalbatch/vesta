<script lang="ts">
	// Generic control-mapping
	// Settings" → "Control mapping"): bool -> toggle, bounded int/float ->
	// slider + readout, unbounded int/float -> number input, string+choices ->
	// select, string -> text (mono if the key ends path/url/dir/model).
	// `hot: false` appends the restart note to help text. Dumb/presentational —
	// the parent page owns the draft value and the dirty/error state.
	import type { SettingSchemaItem } from '$lib/types';
	import { humanizeKey, isMonoStringKey } from '$lib/settings-groups';
	import { settingLabel, settingHelp } from '$lib/settings-copy';

	let {
		item,
		value,
		error = null,
		onChange
	}: {
		item: SettingSchemaItem;
		value: string;
		error?: string | null;
		onChange: (key: string, value: string) => void;
	} = $props();

	const label = $derived(settingLabel(item.key, humanizeKey(item.key)));
	const help = $derived(settingHelp(item.key, item.help));
	const isBounded = $derived(item.type !== 'boolean' && item.type !== 'string' && item.min != null && item.max != null);
	const numericValue = $derived(Number(value));
	const boolChecked = $derived(['1', 'true', 'yes', 'on'].includes(value.trim().toLowerCase()));

	function stepFor(): number {
		if (item.type === 'integer') return 1;
		const span = (item.max ?? 1) - (item.min ?? 0);
		return span > 5 ? 0.1 : 0.01;
	}

	function formatReadout(): string {
		if (Number.isNaN(numericValue)) return value;
		return item.type === 'integer' ? String(Math.round(numericValue)) : numericValue.toFixed(2);
	}
</script>

<div
	class="grid grid-cols-1 gap-2 border-b border-border py-3.5 last:border-0 md:grid-cols-[minmax(0,1fr)_minmax(180px,300px)] md:items-center md:gap-6"
>
	<div>
		<div class="flex flex-wrap items-baseline gap-x-2">
			<span class="text-sm font-medium text-ink">{label}</span>
			<code class="text-[11px] text-faint">{item.key}</code>
		</div>
		<p class="mt-0.5 text-xs text-muted">
			{help}{#if !item.hot}<span class="text-warning"> — takes effect after a restart</span>{/if}
		</p>
		{#if error}<p class="mt-1 text-xs text-danger">{error}</p>{/if}
	</div>
	<div class="flex items-center">
		{#if item.type === 'boolean'}
			<input
				type="checkbox"
				name={item.key}
				checked={boolChecked}
				onchange={(e) => onChange(item.key, (e.target as HTMLInputElement).checked ? 'true' : 'false')}
				class="size-4 accent-accent"
				aria-label={label}
			/>
		{:else if isBounded}
			<div class="flex w-full items-center gap-3">
				<input
					type="range"
					name={item.key}
					min={item.min ?? 0}
					max={item.max ?? 1}
					step={stepFor()}
					value={Number.isNaN(numericValue) ? (item.min ?? 0) : numericValue}
					oninput={(e) => onChange(item.key, (e.target as HTMLInputElement).value)}
					class="h-1.5 flex-1 accent-accent"
					aria-label={label}
				/>
				<span class="w-16 shrink-0 text-right font-mono text-xs text-muted">{formatReadout()}</span>
			</div>
		{:else if item.choices && item.choices.length === 1}
			<span
				class="inline-flex items-center rounded-md border border-border bg-surface-muted px-2.5 py-1 text-xs font-medium text-ink {isMonoStringKey(item.key)
					? 'font-mono'
					: ''}"
			>
				{item.choices[0] === '' ? '(none)' : item.choices[0]}
			</span>
		{:else if item.type === 'string' && item.choices}
			<select
				name={item.key}
				{value}
				onchange={(e) => onChange(item.key, (e.target as HTMLSelectElement).value)}
				class="w-full rounded-md border border-border bg-surface px-2 py-1.5 text-sm"
				aria-label={label}
			>
				{#each item.choices as choice (choice)}
					<option value={choice}>{choice === '' ? '(none)' : choice}</option>
				{/each}
			</select>
		{:else if item.type === 'integer' || item.type === 'float'}
			<input
				type="number"
				name={item.key}
				min={item.min ?? undefined}
				max={item.max ?? undefined}
				step={item.type === 'integer' ? 1 : 'any'}
				{value}
				onchange={(e) => onChange(item.key, (e.target as HTMLInputElement).value)}
				class="w-full rounded-md border border-border bg-surface px-2 py-1.5 text-sm"
				aria-label={label}
			/>
		{:else}
			<input
				type="text"
				name={item.key}
				{value}
				onchange={(e) => onChange(item.key, (e.target as HTMLInputElement).value)}
				class="w-full rounded-md border border-border bg-surface px-2 py-1.5 text-sm {isMonoStringKey(item.key)
					? 'font-mono text-xs'
					: ''}"
				aria-label={label}
			/>
		{/if}
	</div>
</div>
