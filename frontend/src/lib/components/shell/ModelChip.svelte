<script lang="ts">
	// The LLM lifecycle chip : one honest dot + label in the
	// TopBar, popover for the details. Chrome, not a page — the plan caps it
	// around 140 lines on purpose. It also owns modelStore's polling lifetime:
	// the chip is mounted in the TopBar on every shell page, so start/stop in
	// its mount effect gives the store one app-lifetime, visibility-gated poll.
	import { Popover } from 'bits-ui';
	import { modelStore, chipView } from '$lib/stores/model.svelte';
	import { settingsValuesStore } from '$lib/stores/settings.svelte';

	$effect(() => {
		modelStore.start();
		return () => modelStore.stop();
	});

	const view = $derived(chipView(modelStore.status));
	const s = $derived(modelStore.status);
	const remoteHost = $derived.by(() => {
		if (s?.source !== 'remote') return null;
		try {
			return new URL(String(settingsValuesStore.values['inference.llm.endpoint_url'] || '')).host;
		} catch {
			return null;
		}
	});

	function fmtBytes(n: number): string {
		if (!n) return '—';
		if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} GB`;
		if (n >= 1024 ** 2) return `${Math.round(n / 1024 ** 2)} MB`;
		return `${Math.round(n / 1024)} KB`;
	}

	function fmtLastUsed(s: number | null): string {
		if (s == null) return 'never used';
		if (s < 90) return `last used ${Math.max(1, Math.round(s))} s ago`;
		if (s < 5400) return `last used ${Math.max(1, Math.round(s / 60))} min ago`;
		return `last used ${Math.round(s / 3600)} h ago`;
	}

	const dotClass = $derived(
		view.dot === 'amber'
			? 'animate-pulse bg-amber shadow-[0_0_0_3px_color-mix(in_srgb,var(--color-amber)_22%,transparent)]'
			: view.dot === 'red'
				? 'bg-danger'
				: view.dot === 'green'
					? 'bg-success'
					: 'border-2 border-faint bg-transparent'
	);
</script>

{#if view.popover}
	<Popover.Root>
		<Popover.Trigger
			class="inline-flex items-center gap-2 rounded-md px-2 py-1.5 text-xs font-medium text-muted transition-colors hover:bg-surface-muted hover:text-ink"
			title="AI model: {view.label}"
		>
			<span class="size-[9px] shrink-0 rounded-full {dotClass}"></span>
			<span class="max-w-44 truncate">{view.label}</span>
		</Popover.Trigger>
		<Popover.Content
			class="z-[101] w-72 rounded-xl border border-border bg-surface p-4 text-sm shadow-pop"
			align="end"
		>
			{#if s}
				<p class="mb-2 text-xs font-semibold uppercase tracking-wide text-faint">
					{s.source === 'remote' ? 'Remote model' : 'Local model'}{remoteHost ? ` · ${remoteHost}` : ''}
				</p>
				<dl class="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs">
					<dt class="text-faint">File</dt>
					<dd class="truncate" title={s.model_file ?? undefined}>{s.model_file ?? '—'}</dd>
					<dt class="text-faint">Context</dt>
					<dd>{s.context_size ? `${Math.round(s.context_size / 1000)}K tokens` : '—'}</dd>
					<dt class="text-faint">Thinking</dt>
					<dd>{s.thinking ? 'on' : 'off'}</dd>
					<dt class="text-faint">Memory</dt>
					<dd>{fmtBytes(s.estimated_ram_bytes)}</dd>
					<dt class="text-faint">Used</dt>
					<dd>{fmtLastUsed(s.seconds_since_last_use)}</dd>
				</dl>
				{#if s.error}
					<p class="mt-2 rounded-md bg-danger-soft px-2 py-1.5 text-xs text-danger">{s.error}</p>
				{/if}
			{:else}
				<p class="text-xs text-muted">{modelStore.error ?? 'Checking…'}</p>
			{/if}
			<div class="mt-3 flex items-center gap-2">
				{#if s && s.state !== 'loaded' && s.state !== 'loading'}
					<button
						type="button"
						class="rounded-md bg-accent px-2.5 py-1 text-xs font-medium text-white hover:bg-accent-hover disabled:opacity-40"
						disabled={modelStore.busy !== null}
						onclick={() => void modelStore.loadModel()}
					>
						{modelStore.busy === 'load' ? 'Loading…' : 'Load now'}
					</button>
				{:else if s && (s.state === 'loaded' || s.state === 'sleeping')}
					<button
						type="button"
						class="rounded-md border border-border px-2.5 py-1 text-xs font-medium text-muted hover:bg-surface-muted hover:text-ink disabled:opacity-40"
						disabled={modelStore.busy !== null}
						onclick={() => void modelStore.unloadModel()}
					>
						{modelStore.busy === 'unload' ? 'Unloading…' : 'Unload now'}
					</button>
				{/if}
				<a href="/settings?tab=settings#ai" class="ml-auto text-xs text-muted underline hover:text-ink">
					Settings → AI
				</a>
			</div>
		</Popover.Content>
	</Popover.Root>
{:else}
	<a
		href="/settings?tab=settings#ai"
		class="inline-flex items-center gap-2 rounded-md px-2 py-1.5 text-xs font-medium text-muted transition-colors hover:bg-surface-muted hover:text-ink"
		title="No AI model configured - open Settings → AI"
	>
		<span class="size-[9px] shrink-0 rounded-full {dotClass}"></span>
		<span>{view.label}</span>
	</a>
{/if}
