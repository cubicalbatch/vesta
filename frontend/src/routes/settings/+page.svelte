<script lang="ts">
	// Settings (/settings) — three surfaces under one page, selected by `?tab=`:
	//   • Settings (default) — generated from GET /api/settings/schema
	//     110 settings,
	//     37 groups; a new backend setting needs zero frontend work and lands
	//     under "All settings" automatically.
	//   • Jobs — the global jobs stream (was the Advanced page's Jobs tab; the
	//     header job dot links here).
	//   • Advanced — eval/benchmark tooling, shown only when /health reports
	//     `advanced_menu: true` (env VESTA_ADVANCED_MENU). Off for end users.
	// The Settings tab's display pieces are dumb components under
	// components/settings/; Jobs/Advanced reuse components/advanced/*.
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import { settingsValuesStore } from '$lib/stores/settings.svelte';
	import { settingsApi } from '$lib/api/settings';
	import { ApiError } from '$lib/api/client';
	import { healthStore } from '$lib/stores/health.svelte';
	import { BASIC, THOROUGH_KEYS } from '$lib/settings-basic';
	import { CONTEXT_KEYS } from '$lib/context-profile';
	import { groupSettings } from '$lib/settings-groups';
	import { buildSaveMessage } from '$lib/settings-copy';
	import HowThoroughControl from '$lib/components/settings/HowThoroughControl.svelte';
	import type { SettingSchemaItem } from '$lib/types';
	import ContextBudgetControl from '$lib/components/settings/ContextBudgetControl.svelte';
	import SettingSectionView from '$lib/components/settings/SettingSectionView.svelte';
	import SaveBar from '$lib/components/settings/SaveBar.svelte';
	import JobsTab from '$lib/components/advanced/JobsTab.svelte';
	import EvalTab from '$lib/components/advanced/EvalTab.svelte';
	import BenchmarksTab from '$lib/components/advanced/BenchmarksTab.svelte';
	import AiSection from '$lib/components/settings/AiSection.svelte';

	function normalizeValues(values: Record<string, unknown>): Record<string, string> {
		const out: Record<string, string> = {};
		for (const [k, v] of Object.entries(values)) out[k] = String(v);
		return out;
	}

	let draft = $state<Record<string, string>>({});
	let errors = $state<Record<string, string | null>>({});
	let initialized = $state(false);
	let showAll = $state(false);
	let saving = $state(false);
	let saveError = $state<string | null>(null);
	let saveMessage = $state<string | null>(null);
	// Bumped on Discard and after a successful Save to force-remount
	// HowThoroughControl — it keeps one piece of UI-only local state (whether
	// "Custom" was manually revealed) that a value reset should also clear,
	// not just the draft values themselves.
	let formVersion = $state(0);

	$effect(() => {
		settingsValuesStore.load();
		settingsValuesStore.loadSchema();
	});

	$effect(() => {
		if (!initialized && settingsValuesStore.loaded && settingsValuesStore.schemaLoaded) {
			draft = normalizeValues(settingsValuesStore.values);
			initialized = true;
		}
	});

	const schemaByKey = $derived.by((): Record<string, SettingSchemaItem> => {
		const out: Record<string, SettingSchemaItem> = {};
		for (const s of settingsValuesStore.schema) out[s.key] = s;
		return out;
	});

	const originalValues = $derived(normalizeValues(settingsValuesStore.values));
	const dirtyKeys = $derived(Object.keys(draft).filter((k) => draft[k] !== (originalValues[k] ?? '')));

	const thoroughItems = $derived(
		THOROUGH_KEYS.map((k) => schemaByKey[k]).filter((i): i is SettingSchemaItem => Boolean(i))
	);
	const contextItems = $derived(
		CONTEXT_KEYS.map((k) => schemaByKey[k]).filter((i): i is SettingSchemaItem => Boolean(i))
	);
	const contextValues = $derived(Object.fromEntries(CONTEXT_KEYS.map((k) => [k, draft[k] ?? ''])));
	const thoroughValues = $derived(Object.fromEntries(THOROUGH_KEYS.map((k) => [k, draft[k] ?? ''])));

	function isBasicField(key: string): boolean {
		return BASIC.includes(key);
	}

	const visibleSections = $derived.by(() => {
		const all = groupSettings(settingsValuesStore.schema);
		if (showAll) return all;
		return all
			.map((sec) => ({
				...sec,
				subsections: sec.subsections
					.map((sub) => ({ ...sub, items: sub.items.filter((i) => isBasicField(i.key)) }))
					.filter((sub) => sub.items.length > 0)
			}))
			.filter((sec) => sec.subsections.length > 0);
	});

	const hiddenCount = $derived(
		settingsValuesStore.schema.length - BASIC.filter((k) => schemaByKey[k]).length
	);

	function handleChange(key: string, value: string) {
		draft = { ...draft, [key]: value };
		if (errors[key]) {
			const rest = { ...errors };
			delete rest[key];
			errors = rest;
		}
		saveMessage = null;
	}

	function handleDiscard() {
		draft = normalizeValues(settingsValuesStore.values);
		errors = {};
		saveError = null;
		saveMessage = null;
		formVersion += 1;
	}

	// The AI section writes settings on its own (immediate, per control); the
	// form draft below was seeded once at load, so resync the keys it wrote or
	// a later form save would silently revert them.
	function resyncInferenceKeys(keys: string[]) {
		const fresh = normalizeValues(settingsValuesStore.values);
		draft = { ...draft };
		for (const k of keys) draft[k] = fresh[k] ?? draft[k];
	}

	// `/settings?tab=settings#ai` (the ModelChip's "No AI" target): SvelteKit
	// scrolls hashes on navigation, but not when the hash arrives while the
	// tab content is still swapping — cover that with an explicit scroll.
	$effect(() => {
		if (page.url.hash !== '#ai' || activeTab !== 'settings') return;
		const el = document.getElementById('ai');
		if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
	});

	async function handleSave() {
		if (dirtyKeys.length === 0) return;
		saving = true;
		saveError = null;
		try {
			const payload: Record<string, string> = {};
			for (const k of dirtyKeys) payload[k] = draft[k];
			const res = await settingsApi.save(payload);
			errors = {};
			await settingsValuesStore.load();
			draft = normalizeValues(settingsValuesStore.values);
			formVersion += 1;
			saveMessage = buildSaveMessage(res.applied ?? dirtyKeys, schemaByKey);
		} catch (err) {
			if (err instanceof ApiError && err.status === 400) {
				const match = err.detail.match(/setting '([^']+)'/);
				if (match) {
					errors = { ...errors, [match[1]]: err.detail };
					saveMessage = null;
					saveError = `Not saved — fix the highlighted field before trying again. No changes were written.`;
				} else {
					saveError = err.detail;
				}
			} else {
				saveError = err instanceof Error ? err.message : 'Save failed.';
			}
		} finally {
			saving = false;
		}
	}

	// ── Tab orchestration ────────────────────────────────────────────────────
	type Tab = 'settings' | 'jobs' | 'advanced';
	type AdvancedView = 'eval' | 'benchmarks';

	// Advanced is exposed only when the backend opts in via VESTA_ADVANCED_MENU.
	const showAdvanced = $derived(healthStore.data?.advanced_menu === true);

	const TABS = $derived<{ id: Tab; label: string }[]>(
		[
			{ id: 'settings', label: 'Settings' },
			{ id: 'jobs', label: 'Jobs' },
			...(showAdvanced ? [{ id: 'advanced' as const, label: 'Advanced' }] : [])
		]
	);

	const activeTab = $derived.by((): Tab => {
		const raw = page.url.searchParams.get('tab');
		if (raw === 'jobs') return 'jobs';
		if (raw === 'advanced' && showAdvanced) return 'advanced';
		return 'settings';
	});

	const advancedView = $derived(
		page.url.searchParams.get('view') === 'eval' ? 'eval' : 'benchmarks'
	);

	function gotoTab(params: URLSearchParams) {
		const qs = params.toString();
		goto(`/settings${qs ? `?${qs}` : ''}`, { replaceState: true, noScroll: true, keepFocus: true });
	}

	function setTab(tab: Tab) {
		const params = new URLSearchParams(page.url.searchParams);
		if (tab === 'settings') params.delete('tab');
		else params.set('tab', tab);
		// `view` only applies inside Advanced; clear it when leaving.
		if (tab !== 'advanced') params.delete('view');
		gotoTab(params);
	}

	function setAdvancedView(view: AdvancedView) {
		const params = new URLSearchParams(page.url.searchParams);
		params.set('tab', 'advanced');
		if (view === 'benchmarks') params.delete('view');
		else params.set('view', view);
		gotoTab(params);
	}
</script>

<svelte:head>
	<title>Settings - Vesta</title>
</svelte:head>

<div class="mx-auto max-w-[var(--wide-max)] pb-4">
	<h1 class="mb-1 font-display text-2xl font-bold tracking-tight text-ink">Settings</h1>
	<p class="mb-5 text-muted">
		Tune how Vesta answers, watch background jobs, and — when enabled — run eval and benchmark tools.
	</p>

	<!-- Three labels don't fit a 320px viewport; scroll the strip rather than
	     the page (`whitespace-nowrap`/`shrink-0` keep labels on one line). -->
	<div class="mb-6 flex gap-1 overflow-x-auto border-b border-border" role="tablist">
		{#each TABS as t (t.id)}
			<button
				type="button"
				role="tab"
				aria-selected={activeTab === t.id}
				class="shrink-0 whitespace-nowrap border-b-2 px-3 py-2 text-sm font-medium {activeTab === t.id ? 'border-accent text-ink' : 'border-transparent text-muted hover:text-ink-2'}"
				onclick={() => setTab(t.id)}
			>
				{t.label}
			</button>
		{/each}
	</div>

	{#if activeTab === 'jobs'}
		<JobsTab />
	{:else if activeTab === 'advanced'}
		<div class="mb-5 inline-flex items-center gap-1 rounded-full border border-border bg-surface p-1" role="tablist" aria-label="Advanced tool">
			<button
				type="button"
				role="tab"
				aria-selected={advancedView === 'eval'}
				class="rounded-full px-3 py-1.5 text-sm font-medium {advancedView === 'eval' ? 'bg-accent-soft text-accent-soft-text' : 'text-muted hover:bg-surface-muted'}"
				onclick={() => setAdvancedView('eval')}
			>
				Eval runs
			</button>
			<button
				type="button"
				role="tab"
				aria-selected={advancedView === 'benchmarks'}
				class="rounded-full px-3 py-1.5 text-sm font-medium {advancedView === 'benchmarks' ? 'bg-accent-soft text-accent-soft-text' : 'text-muted hover:bg-surface-muted'}"
				onclick={() => setAdvancedView('benchmarks')}
			>
				Benchmarks
			</button>
		</div>
		{#if advancedView === 'eval'}
			<EvalTab />
		{:else}
			<BenchmarksTab />
		{/if}
	{:else}
	<!-- The AI section — above the generated form, which stays
	     as the every-inference-key safety net below. -->
	<AiSection onInferenceChange={resyncInferenceKeys} />
		{#if settingsValuesStore.schemaError}

			<p class="mb-4 rounded-md border border-danger bg-danger-soft px-3 py-2 text-sm text-danger">
				Couldn't load the settings schema: {settingsValuesStore.schemaError}
			</p>
		{/if}

		{#if !settingsValuesStore.schemaLoaded}
			<p class="text-sm text-faint">Loading settings…</p>
		{:else if settingsValuesStore.schema.length === 0}
			<p class="text-sm text-faint">No settings schema available.</p>
		{:else}
			<div class="mb-5 flex flex-wrap items-center gap-3">
				<div class="inline-flex items-center gap-1 rounded-full border border-border bg-surface p-1" role="tablist" aria-label="How many settings to show">
					<button
						type="button"
						role="tab"
						aria-selected={!showAll}
						class="rounded-full px-3 py-1.5 text-sm font-medium {!showAll
							? 'bg-accent-soft text-accent-soft-text'
							: 'text-muted hover:bg-surface-muted'}"
						onclick={() => (showAll = false)}
					>
						Basic
					</button>
					<button
						type="button"
						role="tab"
						aria-selected={showAll}
						class="rounded-full px-3 py-1.5 text-sm font-medium {showAll
							? 'bg-accent-soft text-accent-soft-text'
							: 'text-muted hover:bg-surface-muted'}"
						onclick={() => (showAll = true)}
					>
						All settings
					</button>
				</div>
				{#if !showAll && hiddenCount > 0}
					<span class="text-xs text-faint">{hiddenCount} advanced settings hidden</span>
				{/if}
			</div>

			<div class="mb-6 rounded-xl border border-border bg-surface p-5">
				<h2 class="text-lg font-semibold text-ink">How Vesta answers</h2>
				<p class="mt-0.5 text-xs text-muted">
					How careful Vesta is, and what it keeps in mind while answering. The controls below write
					to plain settings — every underlying key is also listed, and editable directly, under "All settings".
				</p>
				<div class="mt-2">
					{#if thoroughItems.length === THOROUGH_KEYS.length}
						{#key formVersion}
							<HowThoroughControl
								items={thoroughItems}
								values={thoroughValues}
								{errors}
								onChange={handleChange}
							/>
						{/key}
					{/if}
					{#if contextItems.length === CONTEXT_KEYS.length}
						{#key formVersion}
							<ContextBudgetControl
								items={contextItems}
								values={contextValues}
								{errors}
								onChange={handleChange}
							/>
						{/key}
					{/if}
				</div>
			</div>

			{#each visibleSections as section (section.name)}
				<SettingSectionView {section} {draft} {errors} onChange={handleChange} />
			{/each}

			<SaveBar
				unsavedCount={dirtyKeys.length}
				{saving}
				message={saveMessage}
				error={saveError}
				onDiscard={handleDiscard}
				onSave={handleSave}
			/>
		{/if}
	{/if}
</div>
