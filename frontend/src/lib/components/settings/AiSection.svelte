<script lang="ts">
	// Settings → AI — a first-class section at the top
	// of the Settings tab, ABOVE the generated schema form. Everything here is
	// convenience chrome over three surfaces: the D10 model management API
	// (`api/models.ts`), the live runtime status (`modelStore`),
	// and plain settings writes (`settingsApi`). The generated form
	// below still renders every `inference.*` key — the safety net stays.
	//
	// Writes are immediate (per control, no Save button): D7 made every
	// `inference.*` change effectively hot, so a draft buffer here would only
	// add a second copy of the same values to keep in sync. After each write
	// the section reloads `settingsValuesStore` and calls `onInferenceChange`
	// so the page's form draft (seeded once at page load) resyncs those keys.
	import { modelsApi, type InstalledModel, type ModelPreset } from '$lib/api/models';
	import { modelStore } from '$lib/stores/model.svelte';
	import { settingsValuesStore } from '$lib/stores/settings.svelte';
	import { jobsStore } from '$lib/stores/jobs.svelte';
	import { settingsApi } from '$lib/api/settings';
	import { systemApi } from '$lib/api/system';
	import { TERMINAL_JOB_STATUSES } from '$lib/types';
	import { formatBytes } from '$lib/format';
	import { autoPlanFor, CONTEXT_PROFILE_KEY, CONTEXT_SIZE_KEY } from '$lib/context-profile';
	import Check from '@lucide/svelte/icons/check';
	import Trash2 from '@lucide/svelte/icons/trash-2';

	let {
		/** Called with the settings keys this section just wrote, so the
		 *  page's generated-form draft can resync them. */
		onInferenceChange
	}: { onInferenceChange?: (keys: string[]) => void } = $props();

	// ── Settings snapshot (seeded once from the store the page loads) ─────────
	let initialized = $state(false);
	let source = $state<'local' | 'remote'>('local');
	let contextSize = $state(8192);
	let idleUnload = $state(900);
	let preload = $state(true);
	let thinking = $state(false);
	let threadsGen = $state(6);
	let threadsPrefill = $state(8);
	let remoteUrl = $state('');
	let remoteModel = $state('');
	let remoteKey = $state('');

	$effect(() => {
		if (initialized || !settingsValuesStore.loaded) return;
		initialized = true;
		const v = settingsValuesStore.values;
		source = String(v['inference.llm.source'] ?? 'local') === 'remote' ? 'remote' : 'local';
		contextSize = Number(v['inference.local.context_size'] ?? 8192) || 8192;
		idleUnload = Number(v['inference.local.idle_unload_seconds'] ?? 900) || 0;
		preload = String(v['inference.local.preload_on_ready'] ?? 'true') !== 'false';
		threadsGen = Number(v['inference.local.threads_gen'] ?? 6) || 6;
		threadsPrefill = Number(v['inference.local.threads_prefill'] ?? 8) || 8;
		remoteUrl = String(v['inference.llm.endpoint_url'] ?? '');
		remoteModel = String(v['inference.llm.model'] ?? '');
		remoteKey = String(v['inference.llm.api_key'] ?? '');
		// enable_thinking seeds from the setting, not the live status: the
		// status is only truthful for the *active* model.
		thinking = String(v['inference.llm.enable_thinking'] ?? 'false') === 'true';
	});

	// ── Model data: installed list, presets, live status, detected RAM ────────
	let installed = $state<InstalledModel[]>([]);
	let presets = $state<ModelPreset[]>([]);
	let listError = $state<string | null>(null);
	let activating = $state<string | null>(null);
	let deleting = $state<string | null>(null);
	let ramBytes = $state<number | null>(null);
	let cpuCount = $state<number | null>(null);

	async function refreshModels() {
		try {
			const res = await modelsApi.list();
			installed = res.installed;
			presets = res.presets;
			listError = null;
		} catch (err) {
			listError = err instanceof Error ? err.message : 'failed to load models';
		}
	}

	// modelStore.start() is idempotent (the TopBar chip starts it too); calling
	// it here covers direct visits to /settings without a prior chip render.
	$effect(() => {
		modelStore.start();
		modelStore.refresh().catch(() => {});
		refreshModels();
		systemApi
			.hardware()
			.then((h) => {
				ramBytes = h.ram_total_bytes;
				cpuCount = h.cpu_count;
			})
			.catch(() => {});
	});

	// A finished download changes the installed list and (via D8 preload) the
	// live status — refresh both when a download_model job lands.
	const seenModelJobs = new Set<number>();
	$effect(() => {
		let done = false;
		for (const j of jobsStore.list) {
			if (j.type === 'download_model' && TERMINAL_JOB_STATUSES.has(j.status)) {
				if (!seenModelJobs.has(j.id)) {
					seenModelJobs.add(j.id);
					done = true;
				}
			}
		}
		if (done) {
			refreshModels();
			modelStore.refresh().catch(() => {});
		}
	});

	// ── Writes: one settings PUT per control, then resync ─────────────────────
	let saving = $state(false);
	let saveError = $state<string | null>(null);
	let savedMessage = $state<string | null>(null);

	async function writeSettings(values: Record<string, string>) {
		saving = true;
		saveError = null;
		savedMessage = null;
		try {
			await settingsApi.save(values);
			await settingsValuesStore.load();
			onInferenceChange?.(Object.keys(values));
			await modelStore.refresh();
			// activate/delete rewrite inference.llm.model server-side; keep the
			// local seeds honest without a full re-init.
			remoteModel = String(settingsValuesStore.values['inference.llm.model'] ?? remoteModel);
			savedMessage = 'Saved — applies to your next question.';
		} catch (err) {
			saveError = err instanceof Error ? err.message : 'failed to save';
		} finally {
			saving = false;
		}
	}

	function setSource(next: 'local' | 'remote') {
		if (next === source) return;
		source = next;
		writeSettings({ 'inference.llm.source': next });
	}

	async function setContextSize(value: number) {
		contextSize = value;
		await writeSettings({
			[CONTEXT_SIZE_KEY]: String(value),
			[CONTEXT_PROFILE_KEY]: autoPlanFor(value)
		});
	}

	async function setIdleUnload(value: number) {
		idleUnload = value;
		await writeSettings({ 'inference.local.idle_unload_seconds': String(value) });
	}

	async function setPreload(value: boolean) {
		preload = value;
		await writeSettings({ 'inference.local.preload_on_ready': String(value) });
	}

	async function setThreadsGen(value: number) {
		threadsGen = value;
		await writeSettings({ 'inference.local.threads_gen': String(value) });
	}

	async function setThreadsPrefill(value: number) {
		threadsPrefill = value;
		await writeSettings({ 'inference.local.threads_prefill': String(value) });
	}

	async function setThinking(value: boolean) {
		thinking = value;
		await writeSettings({ 'inference.llm.enable_thinking': String(value) });
	}

	async function saveRemote() {
		await writeSettings({
			'inference.llm.source': 'remote',
			'inference.llm.endpoint_url': remoteUrl.trim(),
			'inference.llm.model': remoteModel.trim(),
			'inference.llm.api_key': remoteKey.trim()
		});
	}

	// ── Installed-model actions (D10) ─────────────────────────────────────────
	async function activate(filename: string) {
		activating = filename;
		saveError = null;
		try {
			await modelsApi.activate(filename);
			await settingsValuesStore.load();
			onInferenceChange?.(['inference.llm.model']);
			await modelStore.refresh();
			await refreshModels();
		} catch (err) {
			saveError = err instanceof Error ? err.message : 'failed to activate';
		} finally {
			activating = null;
		}
	}

	async function removeModel(filename: string) {
		if (!window.confirm(`Delete ${filename} from disk? This cannot be undone.`)) return;
		deleting = filename;
		saveError = null;
		try {
			await modelsApi.remove(filename);
			await settingsValuesStore.load();
			onInferenceChange?.(['inference.llm.model']);
			remoteModel = String(settingsValuesStore.values['inference.llm.model'] ?? '');
			await modelStore.refresh();
			await refreshModels();
		} catch (err) {
			saveError = err instanceof Error ? err.message : 'failed to delete';
		} finally {
			deleting = null;
		}
	}

	// ── Download block (reuses the wizard's preset-card shape) ────────────────
	let downloadingPreset = $state<string | null>(null);
	let downloadError = $state<string | null>(null);

	const activeDownloadJob = $derived(
		jobsStore.list.find(
			(j) => j.type === 'download_model' && !TERMINAL_JOB_STATUSES.has(j.status)
		)
	);

	function presetDownloading(p: ModelPreset): boolean {
		return activeDownloadJob?.target === p.filename;
	}

	async function downloadPreset(p: ModelPreset) {
		if (p.downloaded || presetDownloading(p)) return;
		downloadingPreset = p.id;
		downloadError = null;
		try {
			await modelsApi.download({ preset_id: p.id });
		} catch (err) {
			downloadError = err instanceof Error ? err.message : 'failed to start download';
		} finally {
			downloadingPreset = null;
		}
	}

	// ── Context-window RAM estimate (D12) ─────────────────────────────────────
	// Mirrors inference/models.py::estimate_ram_bytes = weights + ctx × kv/tok.
	// The KV rate is derived from the live status when possible (its
	// estimated_ram_bytes was computed with the real preset value), else the
	// backend's err-high default of 32 KiB/token.
	const DEFAULT_KV_BYTES_PER_TOKEN = 32 * 1024;
	const CONTEXT_OPTIONS = [4096, 8192, 16384, 32768, 65536, 131072];

	const status = $derived(modelStore.status);
	const activeInstalled = $derived(installed.find((m) => m.is_active) ?? null);
	const modelSizeBytes = $derived(status?.size_bytes || activeInstalled?.size_bytes || 0);
	const kvBytesPerToken = $derived.by(() => {
		const s = status;
		if (s && s.estimated_ram_bytes > 0 && s.size_bytes > 0 && s.context_size > 0) {
			return Math.max((s.estimated_ram_bytes - s.size_bytes) / s.context_size, 1);
		}
		return DEFAULT_KV_BYTES_PER_TOKEN;
	});
	const ramEstimateBytes = $derived(
		modelSizeBytes > 0 ? Math.round(modelSizeBytes + contextSize * kvBytesPerToken) : 0
	);
	const ramWarning = $derived(ramBytes !== null && ramEstimateBytes > 0.7 * ramBytes);
	// A stored context outside the 6 offered steps (set under "All settings")
	// stays selectable rather than silently snapping to the nearest option.
	const contextOptions = $derived(
		CONTEXT_OPTIONS.includes(contextSize) ? CONTEXT_OPTIONS : [...CONTEXT_OPTIONS, contextSize].sort((a, b) => a - b)
	);

	const IDLE_OPTIONS: { value: number; label: string }[] = [
		{ value: 300, label: 'After 5 minutes' },
		{ value: 900, label: 'After 15 minutes' },
		{ value: 1800, label: 'After 30 minutes' },
		{ value: 3600, label: 'After 1 hour' },
		{ value: 0, label: 'Never' }
	];
	const idleOptions = $derived(
		IDLE_OPTIONS.some((o) => o.value === idleUnload)
			? IDLE_OPTIONS
			: [...IDLE_OPTIONS, { value: idleUnload, label: `After ${Math.round(idleUnload / 60)} minutes` }].sort(
					(a, b) => (a.value || Infinity) - (b.value || Infinity)
				)
	);

	const THREAD_OPTIONS = [1, 2, 3, 4, 5, 6, 7, 8];
	const threadGenOptions = $derived(
		THREAD_OPTIONS.includes(threadsGen) ? THREAD_OPTIONS : [...THREAD_OPTIONS, threadsGen].sort((a, b) => a - b)
	);
	const threadPrefillOptions = $derived(
		THREAD_OPTIONS.includes(threadsPrefill)
			? THREAD_OPTIONS
			: [...THREAD_OPTIONS, threadsPrefill].sort((a, b) => a - b)
	);

	// ── Live status line (mirrors the TopBar chip) ────────────────────────────
	const statusLine = $derived.by(() => {
		const s = status;
		if (!s) return 'Checking…';
		const name = s.display_name ?? s.model_file ?? 'No model';
		switch (s.state) {
			case 'loaded':
				return `${name} is loaded and ready`;
			case 'loading':
				return `Loading ${name}…`;
			case 'error':
				return `AI unavailable — ${s.error ?? 'unknown error'}`;
			case 'unloaded':
			case 'sleeping':
			case 'stopped':
				return `${name} is asleep — loads again on your next question`;
			default:
				return 'No model installed';
		}
	});

	// ── Test connection (remote) ──────────────────────────────────────────────
	// Probes the user's own server directly from the browser (GET {base}/models
	// is the OpenAI-compatible discovery surface). This is a user-initiated
	// request to a user-configured host — not a runtime dependency of the app.
	let testing = $state(false);
	let testResult = $state<{ ok: boolean; message: string } | null>(null);

	async function testConnection() {
		const base = remoteUrl.trim().replace(/\/+$/, '');
		if (!base) {
			testResult = { ok: false, message: 'Enter a server address first.' };
			return;
		}
		testing = true;
		testResult = null;
		const ctrl = new AbortController();
		const timer = setTimeout(() => ctrl.abort(), 5000);
		try {
			const res = await fetch(`${base}/models`, { signal: ctrl.signal });
			if (!res.ok) throw new Error(`HTTP ${res.status}`);
			const body = (await res.json().catch(() => null)) as { data?: { id?: string }[] } | null;
			const ids = (body?.data ?? []).map((m) => m.id ?? '').filter(Boolean);
			const wanted = remoteModel.trim();
			testResult = {
				ok: true,
				message:
					`Reached ${base} — ${ids.length} model${ids.length === 1 ? '' : 's'}` +
					(wanted ? (ids.includes(wanted) ? `, including “${wanted}”` : `. “${wanted}” was not in the list`) : '')
			};
		} catch (err) {
			testResult = {
				ok: false,
				message: `Couldn't reach ${base}/models (${err instanceof Error ? err.message : 'failed'}). If the server is up, it may not allow browser requests (CORS) — Vesta itself can still use it.`
			};
		} finally {
			clearTimeout(timer);
			testing = false;
		}
	}

	function presetHost(p: ModelPreset): string {
		try {
			return new URL(p.url).host;
		} catch {
			return p.url;
		}
	}
</script>

<section id="ai" class="mb-6 scroll-mt-24 rounded-xl border border-border bg-surface p-5">
	<div class="mb-4 flex flex-wrap items-center justify-between gap-3">
		<div>
			<h2 class="font-display text-lg font-semibold text-ink">AI</h2>
			<p class="text-sm text-muted">Where answers come from, and what's loaded right now.</p>
		</div>
		<!-- Source: segmented Local/Remote -->
		<div class="inline-flex rounded-lg border border-border bg-surface p-0.5" role="group" aria-label="Where the model runs">
			<button
				type="button"
				aria-pressed={source === 'local'}
				class="rounded-md px-3 py-1.5 text-sm font-medium transition-colors {source === 'local'
					? 'bg-accent text-white'
					: 'text-muted hover:text-ink'}"
				onclick={() => setSource('local')}
			>
				On this machine
			</button>
			<button
				type="button"
				aria-pressed={source === 'remote'}
				class="rounded-md px-3 py-1.5 text-sm font-medium transition-colors {source === 'remote'
					? 'bg-accent text-white'
					: 'text-muted hover:text-ink'}"
				onclick={() => setSource('remote')}
			>
				Another server
			</button>
		</div>
	</div>

	{#if saveError}
		<p class="mb-3 rounded-md border border-danger bg-danger-soft px-3 py-2 text-sm text-danger">{saveError}</p>
	{/if}
	{#if savedMessage && !saving}
		<p class="mb-3 text-xs text-success">{savedMessage}</p>
	{/if}

	{#if source === 'local'}
		<!-- Installed GGUFs: active radio + delete (D10) -->
		{#if listError}
			<p class="mb-3 text-sm text-danger">{listError}</p>
		{/if}
		<div class="mb-4">
			<h3 class="mb-2 text-sm font-semibold text-ink-2">Installed models</h3>
			{#if installed.length === 0}
				<p class="text-sm text-faint">No models yet — download one below.</p>
			{:else}
				<ul class="flex flex-col gap-1.5">
					{#each installed as m (m.filename)}
						<li class="flex items-center gap-3 rounded-lg border border-border bg-surface px-3 py-2 {m.is_active ? 'border-accent/40 bg-accent-soft' : ''}">
							<input
								type="radio"
								id="active-{m.filename}"
								name="active-model"
								checked={m.is_active}
								disabled={activating !== null}
								onchange={() => activate(m.filename)}
								class="size-4 accent-[var(--color-accent)]"
							/>
							<label for="active-{m.filename}" class="min-w-0 flex-1 cursor-pointer">
								<span class="block truncate text-sm font-medium text-ink">
									{m.display_name}
									{#if m.is_active}<span class="ml-1 text-xs font-normal text-accent-soft-text">active</span>{/if}
								</span>
								<span class="block truncate text-xs text-faint">{m.filename} · {formatBytes(m.size_bytes)}</span>
							</label>
							<button
								type="button"
								class="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs text-muted hover:bg-danger-soft hover:text-danger disabled:opacity-40"
								disabled={deleting !== null || activating !== null}
								onclick={() => removeModel(m.filename)}
							>
								<Trash2 class="size-3.5" />
								{deleting === m.filename ? 'Deleting…' : 'Delete'}
							</button>
						</li>
					{/each}
				</ul>
			{/if}
		</div>

		<!-- Download a model: the wizard's preset cards -->
		{#if presets.length > 0}
			<div class="mb-4">
				<h3 class="mb-2 text-sm font-semibold text-ink-2">Download a model</h3>
				<div class="flex flex-col gap-2">
					{#each presets as p (p.id)}
						<div
							class="rounded-md border border-border p-3"
							title={`Downloads from ${p.url}`}
						>
							<div class="flex items-center justify-between gap-2">
								<span class="text-sm font-medium text-ink">{p.display_name}</span>
								{#if p.downloaded}
									<span class="inline-flex items-center gap-1 rounded-full bg-success-soft px-2 py-0.5 text-xs font-medium text-success">
										<Check class="size-3" />Downloaded
									</span>
								{:else if presetDownloading(p)}
									<span class="rounded-full bg-surface-muted px-2 py-0.5 text-xs font-medium text-muted">Downloading…</span>
								{:else}
									<button
										type="button"
										class="rounded-md bg-accent px-3 py-1 text-xs font-medium text-white hover:bg-accent-hover disabled:opacity-40"
										disabled={downloadingPreset !== null || activeDownloadJob != null}
										onclick={() => downloadPreset(p)}
									>
										{downloadingPreset === p.id ? 'Starting…' : 'Download'}
									</button>
								{/if}
							</div>
							<p class="mt-1 text-xs text-faint">{formatBytes(p.size_bytes)} · min {p.min_ram_gb} GB RAM</p>
							<p class="text-xs text-muted">{p.description}</p>
							<p class="truncate text-xs text-faint">↗ {presetHost(p)}</p>
							{#if presetDownloading(p) && activeDownloadJob}
								<div class="mt-2 h-1 overflow-hidden rounded-full bg-surface">
									<div
										class="h-full rounded-full bg-accent transition-all"
										style="width: {activeDownloadJob.total
											? Math.round(((activeDownloadJob.progress ?? 0) / activeDownloadJob.total) * 100)
											: 0}%"
									></div>
								</div>
								<p class="mt-1 text-xs text-faint">
									{activeDownloadJob.total
										? `${formatBytes(activeDownloadJob.progress ?? 0)} of ${formatBytes(activeDownloadJob.total)}`
										: activeDownloadJob.message ?? 'preparing…'}
								</p>
							{/if}
						</div>
					{/each}
				</div>
				{#if downloadError}<p class="mt-2 text-xs text-danger">{downloadError}</p>{/if}
			</div>
		{/if}

		<div class="grid grid-cols-1 gap-4 min-[721px]:grid-cols-2">
			<!-- Context window + live RAM estimate (D12) -->
			<div>
				<label for="ai-context" class="mb-1 block text-sm font-medium text-ink">Context window</label>
				<select
					id="ai-context"
					class="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
					onchange={(e) => setContextSize(Number(e.currentTarget.value))}
				>
					{#each contextOptions as o (o)}
						<option value={o} selected={o === contextSize}>
							{Math.round(o / 1024)}k tokens
						</option>
					{/each}
				</select>
				{#if ramEstimateBytes > 0}
					<p class="mt-1 text-xs {ramWarning ? 'text-danger' : 'text-faint'}">
						≈ {formatBytes(ramEstimateBytes)} of RAM{ramBytes !== null ? ` of ${formatBytes(ramBytes)} detected` : ''}
						{ramWarning ? '— that is more than 70% of this machine’s memory; answers may fail or the machine may swap.' : ''}
					</p>
				{:else}
					<p class="mt-1 text-xs text-faint">RAM estimate appears once a model is installed.</p>
				{/if}
			</div>

			<!-- Idle unload (D4) -->
			<div>
				<label for="ai-idle" class="mb-1 block text-sm font-medium text-ink">Free memory when idle</label>
				<select
					id="ai-idle"
					class="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
					onchange={(e) => setIdleUnload(Number(e.currentTarget.value))}
				>
					{#each idleOptions as o (o.value)}
						<option value={o.value} selected={o.value === idleUnload}>{o.label}</option>
					{/each}
				</select>
				<p class="mt-1 text-xs text-faint">The model reloads on your next question.</p>
			</div>
		</div>

		<!-- Thinking (D11) + warm-after-download (D8) -->
		<div class="mt-4 flex flex-col gap-3 border-t border-border pt-4">
			<label class="flex items-start gap-3 {status && !status.thinking_supported ? 'opacity-60' : ''}">
				<input
					type="checkbox"
					checked={thinking}
					disabled={saving || (status !== null && !status.thinking_supported)}
					onchange={(e) => setThinking(e.currentTarget.checked)}
					class="mt-0.5 size-4"
				/>
				<span>
					<span class="block text-sm font-medium text-ink">Thinking</span>
					{#if status !== null && !status.thinking_supported}
						<span class="block text-xs text-muted">
							{status.display_name ?? 'This model'} always reasons before answering — its chat template has
							no off switch. Answers will be slower; the reasoning itself is never shown.
						</span>
					{:else}
						<span class="block text-xs text-faint">
							Lets the model reason before answering. Off is faster and usually enough for lookups.
						</span>
					{/if}
				</span>
			</label>

			<label class="flex items-start gap-3">
				<input
					type="checkbox"
					checked={preload}
					disabled={saving}
					onchange={(e) => setPreload(e.currentTarget.checked)}
					class="mt-0.5 size-4"
				/>
				<span>
					<span class="block text-sm font-medium text-ink">Keep the model warm after download</span>
					<span class="block text-xs text-faint">
						Loads the model the moment its download finishes, so the first question is fast.
					</span>
				</span>
			</label>
		</div>

		<!-- Hardware & CPU threads -->
		<details class="mt-4 rounded-lg border border-border bg-surface-muted/30 p-3.5">
			<summary class="cursor-pointer select-none text-xs font-medium text-ink-2 hover:text-ink">
				Hardware &amp; CPU threads{cpuCount ? ` (${cpuCount} CPU cores detected)` : ''}
			</summary>
			<div class="mt-3 grid grid-cols-1 gap-4 min-[721px]:grid-cols-2">
				<div>
					<label for="ai-threads-gen" class="mb-1 block text-xs font-medium text-ink">Generation threads</label>
					<select
						id="ai-threads-gen"
						class="w-full rounded-md border border-border bg-surface px-3 py-1.5 text-sm outline-none focus:border-accent"
						disabled={saving}
						onchange={(e) => setThreadsGen(Number(e.currentTarget.value))}
					>
						{#each threadGenOptions as t (t)}
							<option value={t} selected={t === threadsGen}>{t} thread{t === 1 ? '' : 's'}</option>
						{/each}
					</select>
					<p class="mt-1 text-xs text-faint">Threads used while generating answers (-t). Default: 6.</p>
				</div>
				<div>
					<label for="ai-threads-prefill" class="mb-1 block text-xs font-medium text-ink">Prompt processing threads</label>
					<select
						id="ai-threads-prefill"
						class="w-full rounded-md border border-border bg-surface px-3 py-1.5 text-sm outline-none focus:border-accent"
						disabled={saving}
						onchange={(e) => setThreadsPrefill(Number(e.currentTarget.value))}
					>
						{#each threadPrefillOptions as t (t)}
							<option value={t} selected={t === threadsPrefill}>{t} thread{t === 1 ? '' : 's'}</option>
						{/each}
					</select>
					<p class="mt-1 text-xs text-faint">Threads used for prompt &amp; passage ingest (-tb). Default: 8.</p>
				</div>
			</div>
		</details>
	{:else}
		<!-- Remote: endpoint / model / API key + Test connection -->
		<div class="grid grid-cols-1 gap-3 min-[721px]:grid-cols-3">
			<label class="text-sm min-[721px]:col-span-2">
				<span class="mb-1 block text-xs text-faint">Server address</span>
				<input
					type="url"
					bind:value={remoteUrl}
					placeholder="http://localhost:1234/v1"
					class="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
				/>
			</label>
			<label class="text-sm">
				<span class="mb-1 block text-xs text-faint">Model</span>
				<input
					type="text"
					bind:value={remoteModel}
					placeholder="qwen3.5-4b"
					class="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
				/>
			</label>
			<label class="text-sm min-[721px]:col-span-3">
				<span class="mb-1 block text-xs text-faint">API key (optional)</span>
				<input
					type="password"
					bind:value={remoteKey}
					placeholder="sk-…"
					class="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
				/>
			</label>
		</div>
		<div class="mt-3 flex flex-wrap items-center gap-3">
			<button
				type="button"
				class="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-40"
				disabled={saving}
				onclick={saveRemote}
			>
				{saving ? 'Saving…' : 'Save'}
			</button>
			<button
				type="button"
				class="rounded-md border border-border px-4 py-2 text-sm font-medium text-ink hover:bg-surface-muted disabled:opacity-40"
				disabled={testing}
				onclick={testConnection}
			>
				{testing ? 'Testing…' : 'Test connection'}
			</button>
			{#if testResult}
				<span class="text-xs {testResult.ok ? 'text-success' : 'text-danger'}">{testResult.message}</span>
			{/if}
		</div>
	{/if}

	<!-- Live status line mirroring the chip, with Load/Unload (D10) -->
	<div class="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-border pt-4">
		<p class="flex items-center gap-2 text-sm text-muted">
			{#if status?.state === 'loaded'}
				<span class="size-2 rounded-full bg-success"></span>
			{:else if status?.state === 'loading'}
				<span class="size-2 animate-pulse rounded-full bg-warning"></span>
			{:else if status?.state === 'error'}
				<span class="size-2 rounded-full bg-danger"></span>
			{:else}
				<span class="size-2 rounded-full border border-border-strong"></span>
			{/if}
			{statusLine}
		</p>
		{#if source === 'local'}
			<div class="flex gap-2">
				<button
					type="button"
					class="rounded-md border border-border px-3 py-1.5 text-xs font-medium text-ink hover:bg-surface-muted disabled:opacity-40"
					disabled={modelStore.busy !== null || status?.state === 'loaded' || !status?.configured || !status?.installed}
					onclick={() => modelStore.loadModel()}
				>
					{modelStore.busy === 'load' ? 'Loading…' : 'Load now'}
				</button>
				<button
					type="button"
					class="rounded-md border border-border px-3 py-1.5 text-xs font-medium text-ink hover:bg-surface-muted disabled:opacity-40"
					disabled={modelStore.busy !== null || status?.state !== 'loaded'}
					onclick={() => modelStore.unloadModel()}
				>
					{modelStore.busy === 'unload' ? 'Unloading…' : 'Unload'}
				</button>
			</div>
		{/if}
	</div>
</section>
