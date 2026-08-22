<script lang="ts">
	// Searchable language picker for the catalog browse filters. Replaces the old
	// hardcoded 4-option <select>: the Kiwix feed ships hundreds of languages, so
	// a type-to-find combobox (match by name OR ISO 639-3 code) is the only sane
	// way to reach yours without scrolling. Controlled: the page owns `value`.
	import { languageName } from '$lib/languages';
	import { formatCount } from '$lib/format';
	import ChevronDown from '@lucide/svelte/icons/chevron-down';

	interface LanguageOption {
		code: string;
		count: number;
	}

	let {
		value,
		languages,
		onSelect
	}: {
		value: string;
		languages: LanguageOption[];
		onSelect: (code: string) => void;
	} = $props();

	let open = $state(false);
	let query = $state('');
	let activeIndex = $state(0);
	let inputEl: HTMLInputElement | null = null;
	let rootEl: HTMLDivElement | null = null;

	const displayValue = $derived(value ? languageName(value) : '');

	type Option = { code: string; name: string; count: number };
	const options = $derived<Option[]>(
		[
			{ code: '', name: 'Any language', count: 0 },
			...languages.map((l) => ({ code: l.code, name: languageName(l.code), count: l.count }))
		]
	);
	const filtered = $derived.by<Option[]>(() => {
		const q = query.trim().toLowerCase();
		if (!q) return options;
		return options.filter(
			(o) => o.name.toLowerCase().includes(q) || o.code.toLowerCase().includes(q)
		);
	});

	function openList() {
		open = true;
		query = '';
		activeIndex = 0;
	}

	function close() {
		open = false;
		query = '';
	}

	function choose(code: string) {
		onSelect(code);
		close();
		inputEl?.blur();
	}

	function onKeydown(e: KeyboardEvent) {
		if (!open) {
			if (e.key === 'ArrowDown' || e.key === 'Enter' || e.key === ' ') {
				openList();
				e.preventDefault();
			}
			return;
		}
		const max = filtered.length - 1;
		if (e.key === 'ArrowDown') {
			activeIndex = Math.min(activeIndex + 1, max);
			e.preventDefault();
		} else if (e.key === 'ArrowUp') {
			activeIndex = Math.max(activeIndex - 1, 0);
			e.preventDefault();
		} else if (e.key === 'Enter') {
			const o = filtered[activeIndex];
			if (o) choose(o.code);
			e.preventDefault();
		} else if (e.key === 'Escape') {
			close();
			e.preventDefault();
		}
	}

	function onWindowPointerDown(e: PointerEvent) {
		if (open && rootEl && !rootEl.contains(e.target as Node)) close();
	}

	// Keep the highlighted row visible as the list filters / the arrows move.
	$effect(() => {
		if (!open) return;
		activeIndex;
		rootEl?.querySelector(`[data-idx="${activeIndex}"]`)?.scrollIntoView({ block: 'nearest' });
	});
</script>
<!-- Outside-pointerdown guard. `<svelte:window>` must sit at the component's top
     level (never inside a block); the handler no-ops when the list is closed. -->
<svelte:window onpointerdown={onWindowPointerDown} />

<div class="relative" bind:this={rootEl}>
	<input
		type="text"
		bind:this={inputEl}
		value={open ? query : displayValue}
		placeholder="Any language"
		role="combobox"
		aria-expanded={open}
		aria-autocomplete="list"
		aria-controls="lang-listbox"
		autocomplete="off"
		spellcheck="false"
		onfocus={openList}
		oninput={(e) => {
			if (!open) openList();
			query = (e.currentTarget as HTMLInputElement).value;
			activeIndex = 0;
		}}
		onkeydown={onKeydown}
		class="w-[min(200px,50vw)] rounded-md border border-border bg-surface py-2 pl-2 pr-7 text-sm outline-none focus:border-accent"
	/>
	<ChevronDown
		size={14}
		class="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-faint"
	/>
	{#if open}
		<div
			class="absolute z-50 mt-1 max-h-72 w-full overflow-auto rounded-md border border-border bg-surface shadow-pop"
		>
			{#if filtered.length === 0}
				<div class="px-3 py-2 text-xs text-muted">No languages match “{query}”.</div>
			{:else}
				<ul id="lang-listbox" role="listbox">
					{#each filtered as opt, i (opt.code)}
						<li role="option" aria-selected={opt.code === value} data-idx={i}>
							<button
								type="button"
								class="flex w-full items-center justify-between gap-3 px-3 py-1.5 text-left text-sm {i ===
								activeIndex
									? 'bg-surface-muted'
									: ''} {opt.code === value ? 'font-medium text-accent' : 'text-ink-2'}"
								onpointerdown={(e) => e.preventDefault()}
								onclick={() => choose(opt.code)}
								onmouseenter={() => (activeIndex = i)}
							>
								<span class="truncate">{opt.name}</span>
								{#if opt.code}<span class="shrink-0 text-xs text-faint">{formatCount(opt.count)}</span>{/if}
							</button>
						</li>
					{/each}
				</ul>
			{/if}
		</div>
	{/if}
</div>
