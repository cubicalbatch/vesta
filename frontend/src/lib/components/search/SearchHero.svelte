<script lang="ts">
	// The single hero: greeting, the query form, the "Use AI" toggle + its
	// load-bearing helper copy, and the scope chips. Presentational — every
	// action is a callback prop; SearchPage owns the state
	// Target behaviour.
	//
	// SearchPage only mounts this while there's no live conversation — once an
	// AI turn is streaming, this hero is unmounted and AskTurn's own per-turn
	// follow-up form takes over.
	import type { Archive } from '$lib/types';
	import SearchIcon from '@lucide/svelte/icons/search';

	let {
		mode,
		query = $bindable(),
		inputEl = $bindable(),
		greeting,
		archiveCount,
		articleCount,
		scope,
		archives,
		llmAvailable,
		healthLoaded,
		indexingPct,
		onSubmit,
		onToggleMode,
		onToggleScope
	}: {
		mode: 'sources' | 'ai';
		query: string;
		/** Bound up to SearchPage so it owns the single `/` focus listener
		 * (one page, one listener — the hero is the one query box). */
		inputEl: HTMLInputElement | null;
		greeting: string;
		archiveCount: number;
		articleCount: number;
		scope: Set<number>;
		archives: Archive[];
		llmAvailable: boolean;
		healthLoaded: boolean;
		indexingPct: (zimId: number) => number | null;
		onSubmit: (q: string) => void;
		onToggleMode: () => void;
		onToggleScope: (id: number | 'all') => void;
	} = $props();

	function submit(e: SubmitEvent) {
		e.preventDefault();
		onSubmit(query.trim());
	}

	// Helper line swaps with the mode — it's the load-bearing copy that the old
	// pill switch used to carry.
	const helperLine = $derived(
		mode === 'ai'
			? 'Reads the top sources and drafts a cited answer, with citations you can open. Slower - it runs the model.'
			: 'Ranks passages from your library. Never calls the model.'
	);
	// The fresh hero's input/button copy stays fixed to "Search" regardless of
	// mode — only the helper line beneath the toggle swaps with it.
	const placeholder = 'Search your library…';
	const buttonLabel = 'Search';
</script>

	<section class="relative isolate overflow-x-clip py-9 text-center max-[720px]:py-6">
		<div class="hearth-glow" aria-hidden="true"></div>
		<p class="mb-2 text-sm text-muted">
			{greeting} · {archiveCount} archive{archiveCount === 1 ? '' : 's'}, {articleCount.toLocaleString()}
			articles indexed.
		</p>
		<h1 class="mb-6 font-display text-3xl font-bold tracking-tight text-ink max-[720px]:text-2xl">What do you want to know?</h1>

		<form class="mx-auto flex max-w-xl items-center gap-2 rounded-xl border border-border bg-surface p-2 shadow-sm" onsubmit={submit}>
			<SearchIcon class="ml-2 size-4 shrink-0 text-faint" />
			<input
				bind:this={inputEl}
				type="text"
				placeholder={placeholder}
				aria-label={placeholder}
				class="min-w-0 flex-1 bg-transparent px-1 py-2 text-base outline-none"
				bind:value={query}
			/>
			<kbd class="hidden rounded border border-border px-1.5 py-0.5 font-mono text-xs text-faint min-[480px]:block">/</kbd>
			<button type="submit" class="flex items-center gap-1.5 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover">
				{buttonLabel}
			</button>
		</form>

		<div class="mt-5 flex flex-col items-center gap-2">
			<button
				type="button"
				role="switch"
				aria-checked={mode === 'ai'}
				aria-label="Use AI"
				// Optimistic until /health lands: a control that silently enables
				// itself half a second later is worse than one request the server
				// degrades on its own. Once health
				// has loaded and there's no LLM, disable it in place — never hide it.
				disabled={healthLoaded && !llmAvailable}
				onclick={onToggleMode}
				class="inline-flex items-center gap-2 text-sm {healthLoaded && !llmAvailable ? 'opacity-60' : ''}"
			>
				<!-- w-[2.25rem], not w-9: app.css remaps the spacing scale's "9" key to
				     3.5rem for section rhythm, so plain `w-9` here silently doubles the
				     track width and the knob's translate-x-4 lands mid-pill instead of
				     flush right. -->
				<span
					class="relative inline-flex h-5 w-[2.25rem] shrink-0 items-center rounded-full border border-border transition-colors {mode ===
					'ai'
						? 'bg-accent'
						: 'bg-surface-muted'}"
				>
					<span class="absolute left-0.5 size-4 rounded-full bg-white shadow transition-transform {mode === 'ai' ? 'translate-x-4' : ''}"></span>
				</span>
				<span class="font-medium text-muted">Use AI</span>
			</button>
			<p class="mx-auto max-w-lg text-xs text-faint">
				{#if healthLoaded && !llmAvailable}
					No model configured - set one up in <a href="/settings" class="text-accent underline">Settings → Models</a> to get drafted, cited answers.
				{:else}
					{helperLine}
				{/if}
			</p>
		</div>

		{#if archives.length > 0}
			<div class="mt-6 flex flex-wrap items-center justify-center gap-2 text-sm">
				<span class="text-xs text-faint">Search in</span>
				<button
					type="button"
					class="rounded-full border px-3 py-1 text-xs font-medium {scope.size === 0
						? 'border-accent/40 bg-accent-soft text-accent-soft-text'
						: 'border-border text-muted hover:bg-surface-muted'}"
					onclick={() => onToggleScope('all')}
				>
					All {archives.length}
				</button>
				{#each archives as archive (archive.id)}
					{@const pct = indexingPct(archive.id)}
					<button
						type="button"
						class="rounded-full border px-3 py-1 text-xs font-medium {scope.has(archive.id)
							? 'border-accent/40 bg-accent-soft text-accent-soft-text'
							: 'border-border text-muted hover:bg-surface-muted'}"
						onclick={() => onToggleScope(archive.id)}
					>
						{archive.corpus_label ?? archive.title ?? archive.name}
						{#if pct != null}<span class="ml-1 rounded-full bg-warning-soft px-1.5 py-0.5 text-warning">{pct}%</span>{/if}
					</button>
				{/each}
			</div>
		{/if}
	</section>
