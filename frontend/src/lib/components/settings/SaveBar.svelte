<script lang="ts">
	// Sticky save bar — never lose the
	// button in a 110-field form. Always rendered; only the badge/actions
	// react to whether there's anything unsaved.
	let {
		unsavedCount,
		saving,
		message,
		error,
		onDiscard,
		onSave
	}: {
		unsavedCount: number;
		saving: boolean;
		message: string | null;
		error: string | null;
		onDiscard: () => void;
		onSave: () => void;
	} = $props();
</script>

<div
	class="sticky bottom-0 z-10 -mx-4 mt-6 flex flex-wrap items-center gap-3 border-t border-border bg-surface/95 px-4 py-3 backdrop-blur supports-[backdrop-filter]:bg-surface/80"
>
	{#if unsavedCount > 0}
		<span class="inline-flex items-center gap-1.5 rounded-full bg-warning-soft px-2.5 py-1 text-xs font-medium text-warning">
			<span class="size-1.5 rounded-full bg-warning"></span>
			{unsavedCount} unsaved
		</span>
	{:else}
		<span class="text-xs text-faint">All changes saved</span>
	{/if}

	{#if error}
		<span class="text-xs text-danger">{error}</span>
	{:else if message}
		<span class="text-xs text-muted">{message}</span>
	{/if}

	<span class="ml-auto flex items-center gap-2">
		<button
			type="button"
			class="rounded-md border border-border px-3 py-1.5 text-sm font-medium text-muted hover:bg-surface-muted disabled:cursor-not-allowed disabled:opacity-50"
			disabled={unsavedCount === 0 || saving}
			onclick={onDiscard}
		>
			Discard
		</button>
		<button
			type="button"
			class="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
			disabled={unsavedCount === 0 || saving}
			onclick={onSave}
		>
			{saving ? 'Saving…' : 'Save changes'}
		</button>
	</span>
</div>
