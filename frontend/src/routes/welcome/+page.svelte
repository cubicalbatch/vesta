<script lang="ts">
	// Welcome — two-step first-run wizard
	// "Page: Welcome"). Reached by redirect from `/` when GET /api/zims comes
	// back empty/all-disabled (see routes/+page.svelte), and reachable directly.
	// +layout.svelte skips AppShell for this route, so the page renders its own
	// minimal top bar.
	//
	// Step 1 — Choose archives: two featured Wikipedia picks + a secondary
	// checkbox list. "Continue" batch-starts the acquisition chain (download →
	// register → index at depth 1) for every selected entry, then advances.
	// Step 2 — Set up AI: optional LLM — point at an existing endpoint, download
	// a GGUF (RAM-tuned recommendation), or skip.
	//
	// This is composition, not new logic: the acquisition chain, curated-to-
	// live-catalog-row matching, and the install-cost line are all Catalog's,
	// reused unmodified.
	import { goto } from '$app/navigation';
	import { catalogApi } from '$lib/api/catalog';
	import { zimsApi } from '$lib/api/zims';
	import { modelsApi, type ModelPreset } from '$lib/api/models';
	import { systemApi } from '$lib/api/system';
	import { setupApi } from '$lib/api/setup';
	import { settingsApi } from '$lib/api/settings';
	import { zimsStore } from '$lib/stores/zims.svelte';
	import { jobsStore } from '$lib/stores/jobs.svelte';
	import { healthStore } from '$lib/stores/health.svelte';
	import { acquisitionManager } from '$lib/stores/acquisition.svelte';
	import type { CatalogEntry, CuratedEntry } from '$lib/types';
	import { TERMINAL_JOB_STATUSES } from '$lib/types';
	import { catalogKey, selectFeatured, selectSecondary } from '$lib/welcome-starters';
	import { formatBytes } from '$lib/format';
	import { lockBodyScroll } from '$lib/scroll-lock';
	import WelcomeTopBar from '$lib/components/welcome/WelcomeTopBar.svelte';
	import WelcomeSteps from '$lib/components/welcome/WelcomeSteps.svelte';
	import FeaturedCard from '$lib/components/welcome/FeaturedCard.svelte';
	import ArchiveCheckbox from '$lib/components/welcome/ArchiveCheckbox.svelte';
	import AcquisitionProgress from '$lib/components/catalog/AcquisitionProgress.svelte';
	import DiskMeter from '$lib/components/catalog/DiskMeter.svelte';
	import Check from '@lucide/svelte/icons/check';
	import { DemoAsk } from '$lib/answer/demo.svelte';

	let step = $state<1 | 2>(1);

	// ── Catalog / curated (step 1 data) ────────────────────────────────────────
	let catalogEnabled = $state(true);
	let catalogAvailable = $state(true);
	let curated = $state<CuratedEntry[]>([]);
	let entries = $state<CatalogEntry[]>([]);
	// Bumped when a catalog refresh job completes so the entry list re-fetches;
	// seeded from the OPDS feed which is NOT cached on a fresh boot (without it,
	// curated cards match no live row and "Continue" silently downloads nothing).
	let catalogVersion = $state(0);
	let catalogRefreshTriggered = $state(false);

	// ── Selection (step 1) ─────────────────────────────────────────────────────
	// One set of curated-entry keys shared by the featured cards and the
	// secondary checkboxes. Reassigned (not mutated) so Svelte's proxy notices.
	let selectedKeys = $state<Set<string>>(new Set());

	let starting = $state(false);
	let startError = $state<string | null>(null);

	// ── Scan / add-from-URL (step 1 secondary options) ─────────────────────────
	let scanning = $state(false);
	let scanMessage = $state<string | null>(null);
	let addManualOpen = $state(false);
	let manualUrl = $state('');
	let manualName = $state('');
	let manualError = $state<string | null>(null);
	let manualBusy = $state(false);

	// ── LLM (step 2) ───────────────────────────────────────────────────────────
	let ramBytes = $state<number | null>(null);
	let presets = $state<ModelPreset[]>([]);
	let presetsError = $state(false);
	let llmChoice = $state<'remote' | 'download' | 'skip'>('download');
	let selectedPresetId = $state<string | null>(null);
	let customUrl = $state('');
	let customFilename = $state('');
	let modelJobId = $state<number | null>(null);
	let downloading = $state(false);
	let downloadError = $state<string | null>(null);
	let finishing = $state(false);

	let remoteUrl = $state('');
	let remoteModel = $state('');
	let remoteKey = $state('');
	let remoteSaving = $state(false);
	let remoteError = $state<string | null>(null);
	let remoteSaved = $state(false);

	// ── Last mile: after the download lands, watch the model
	// become usable. D8 preloads it, so status goes loading → loaded (or
	// error). Finish NEVER waits on this — the button below stays enabled
	// throughout; this block is the payoff, not a gate.
	type Readiness = 'idle' | 'waiting' | 'ready' | 'downloaded' | 'error';
	let readiness = $state<Readiness>('idle');
	let readyName = $state<string | null>(null);
	let readyError = $state<string | null>(null);
	let readinessTimer: ReturnType<typeof setInterval> | null = null;
	// Consecutive polls with the model neither loading nor loaded — two of
	// them means preload is off (or already asleep) and a Load button is more
	// truthful than "Getting ready…" forever.
	let idleReadinessPolls = 0;

	function stopReadinessPolling() {
		if (readinessTimer !== null) {
			clearInterval(readinessTimer);
			readinessTimer = null;
		}
	}

	async function pollReadinessOnce() {
		try {
			const s = await modelsApi.status();
			if (s.display_name) readyName = s.display_name;
			if (s.state === 'loaded') {
				readiness = 'ready';
				stopReadinessPolling();
			} else if (s.state === 'error') {
				readiness = 'error';
				readyError = s.error ?? 'the model failed to load';
				stopReadinessPolling();
			} else if (s.state === 'loading') {
				idleReadinessPolls = 0;
			} else {
				idleReadinessPolls += 1;
				if (idleReadinessPolls >= 2) {
					readiness = 'downloaded';
					stopReadinessPolling();
				}
			}
		} catch {
			// A dropped poll is transient — the interval retries; terminal
			// states above are what stop it.
		}
	}

	function beginReadiness() {
		readiness = 'waiting';
		readyError = null;
		idleReadinessPolls = 0;
		stopReadinessPolling();
		pollReadinessOnce();
		readinessTimer = setInterval(pollReadinessOnce, 2000);
	}

	// The status endpoint is read-only — polling it here cannot keep
	// the model alive or wake a sleeping one.
	async function loadNow() {
		readiness = 'waiting';
		try {
			await modelsApi.load(); // blocks server-side until loaded or error
		} catch {
			// the status poll below reports the outcome either way
		}
		await pollReadinessOnce();
	}

	// ── "Ask a test question" — a canned question through the real answer
	// stream, so the user sees their model write its very first tokens inline
	// before leaving the wizard. This is the moment the wizard is for.
	const testDemo = new DemoAsk();
	$effect(() => (addManualOpen ? lockBodyScroll() : undefined));

	// Curated ships in the repo and works with no network. A 503 here means
	// catalog.enabled=false (an admin killswitch); anything else means the live
	// feed is unreachable but the static starter list still renders.
	$effect(() => {
		catalogApi
			.curated()
			.then((res) => (curated = [...res.entries].sort((a, b) => a.rank - b.rank)))
			.catch((err) => {
				if (err?.status === 503) catalogEnabled = false;
				else catalogAvailable = false;
			});
		catalogApi
			.state()
			.then((s) => (catalogAvailable = s.available))
			.catch((err) => {
				if (err?.status === 503) catalogEnabled = false;
				else catalogAvailable = false;
			});
	});

	// Match curated against a small default browse list, never the full ~3000.
	// Re-runs whenever catalogVersion bumps (i.e. after a refresh completes).
	$effect(() => {
		if (!catalogEnabled) return;
		catalogVersion; // depend on refresh completions
		catalogApi
			.list({ limit: 60 })
			.then((res) => {
				entries = res.entries;
				catalogAvailable = res.available;
			})
			.catch(() => (catalogAvailable = false));
	});

	// Fresh-boot catalog is empty until something fetches the OPDS feed. Without
	// this, the curated cards render but match no live row, so "Continue" starts
	// no download — the "maybe it didn't download the catalog?" failure. Trigger
	// a refresh once on mount if the feed isn't cached.
	$effect(() => {
		if (catalogRefreshTriggered || !catalogEnabled) return;
		catalogApi
			.state()
			.then((s) => {
				if (!s.available) {
					catalogRefreshTriggered = true;
					catalogApi.refresh().catch(() => {});
				} else {
					catalogRefreshTriggered = true;
				}
			})
			.catch((err) => {
				catalogRefreshTriggered = true;
				if (err?.status === 503) catalogEnabled = false;
				else catalogAvailable = false;
			});
	});

	// When a catalog refresh job lands, bump catalogVersion so the entry list
	// above re-fetches against the now-populated feed.
	const seenRefreshJobs = new Set<number>();
	$effect(() => {
		for (const j of jobsStore.list) {
			if (
				j.type === 'refresh_catalog' &&
				TERMINAL_JOB_STATUSES.has(j.status) &&
				!seenRefreshJobs.has(j.id)
			) {
				seenRefreshJobs.add(j.id);
				catalogVersion++;
			}
		}
	});

	// Reconcile the acquisition chain whenever the job snapshot changes — what
	// makes the chain reload-safe if a download was kicked off from here and the
	// page is reloaded mid-chain.
	$effect(() => {
		jobsStore.jobs;
		acquisitionManager.reconcile();
	});

	// Step 2 data — fetched once on mount. Both degrade silently (the page still
	// renders; the download option just shows fewer/no presets).
	$effect(() => {
		systemApi
			.hardware()
			.then((h) => (ramBytes = h.ram_total_bytes))
			.catch(() => {});
		modelsApi
			.presets()
			.then((r) => (presets = r.presets))
			.catch(() => (presetsError = true));
	});

	function curatedToEntry(c: CuratedEntry): CatalogEntry | undefined {
		return entries.find((e) => catalogKey(e.name, e.flavour) === c.name);
	}

	const featured = $derived(selectFeatured(curated));
	const secondary = $derived(selectSecondary(curated));
	const hasSelection = $derived(selectedKeys.size > 0);

	// Default preset (Qwen 4B).
	const defaultPreset = $derived(presets[0] ?? null);

	// Seed the default selection once presets land.
	$effect(() => {
		if (selectedPresetId === null && presets.length > 0) selectedPresetId = presets[0].id;
	});

	// When a model download reaches a terminal state, refresh the presets so the
	// just-downloaded GGUF flips to its "Downloaded" check. `seenDoneModelJobs`
	// is a plain memo (not $state) so this effect depends only on the job list
	// and can't re-trigger itself by writing reactive state.
	const seenDoneModelJobs = new Set<number>();
	$effect(() => {
		let reload = false;
		let landed = false;
		for (const j of jobsStore.list) {
			if (
				j.type === 'download_model' &&
				TERMINAL_JOB_STATUSES.has(j.status) &&
				!seenDoneModelJobs.has(j.id)
			) {
				seenDoneModelJobs.add(j.id);
				reload = true;
				if (j.status === 'done') landed = true;
			}
		}
		if (reload) {
			modelsApi.presets().then((r) => (presets = r.presets)).catch(() => {});
		}
		// A successful download is the last mile's cue: watch the runtime load
		// the model (D8) until it's ready. Failed/cancelled downloads keep the
		// Download button path — no readiness line to show.
		if (landed) beginReadiness();
	});

	// Stop the readiness poll when the wizard goes away, and abort the demo
	// answer stream with it: abort() cancels the fetch; stop() makes consume()
	// drop any event it had already pulled, so nothing writes state after
	// unmount. (Same teardown idiom as AnswerSession's $effect.root cleanup.)
	$effect(() => {
		return () => {
			stopReadinessPolling();
			testDemo.dispose();
		};
	});

	// The in-flight model download. Derived from the global job stream so it
	// survives a reload (modelJobId is component state, lost on remount; the
	// jobs snapshot burst repopulates it), with a fallback to modelJobId for
	// the brief window between the download API returning and the stream
	// emitting the job.
	const activeModelJob = $derived(
		jobsStore.list.find(
			(j) => j.type === 'download_model' && !TERMINAL_JOB_STATUSES.has(j.status)
		)
	);
	const modelJob = $derived.by(() => {
		if (modelJobId !== null) {
			const byId = jobsStore.jobs.get(modelJobId);
			if (byId) return byId;
		}
		return activeModelJob;
	});
	const modelDownloading = $derived(
		modelJob != null && !TERMINAL_JOB_STATUSES.has(modelJob.status)
	);

	// The preset the user has highlighted (or default Qwen 4B preset) — drives the action button's state
	// (Download vs Already downloaded).
	const selectedPreset = $derived(
		selectedPresetId !== null
			? (presets.find((p) => p.id === selectedPresetId) ?? presets[0] ?? null)
			: (presets[0] ?? null)
	);

	// True when `p`'s GGUF is being fetched right now (a non-terminal
	// download_model job targeting its filename). Pairs with `p.downloaded`
	// (file on disk).
	function presetDownloading(p: ModelPreset): boolean {
		return jobsStore.list.some(
			(j) =>
				j.type === 'download_model' &&
				j.target === p.filename &&
				!TERMINAL_JOB_STATUSES.has(j.status)
		);
	}

	// The host of a preset's download URL, shown as a small source hint (the
	// full URL lives in the card's `title` tooltip). Wrapped so a malformed URL
	// never breaks the template.
	function presetHost(p: ModelPreset): string {
		try {
			return new URL(p.url).host;
		} catch {
			return p.url;
		}
	}

	function toggleKey(key: string) {
		const next = new Set(selectedKeys);
		if (next.has(key)) next.delete(key);
		else next.add(key);
		selectedKeys = next;
	}

	// Batch-start the acquisition chain for every selected entry that matches a
	// live catalog row, then advance. Unmatched selections (catalog feed down)
	// can't be downloaded — there's no entry_id/url to hand the backend — so
	// they're skipped silently; the scan/add-URL options below cover that case.
	async function continueToAi() {
		starting = true;
		startError = null;
		try {
			for (const key of selectedKeys) {
				const c = curated.find((x) => x.name === key);
				const e = c ? curatedToEntry(c) : undefined;
				if (e) await acquisitionManager.start(e.id, e.name, 1);
			}
			step = 2;
		} catch (err) {
			startError = err instanceof Error ? err.message : 'failed to start download';
		} finally {
			starting = false;
		}
	}

	async function scanForFiles() {
		scanning = true;
		try {
			const res = await zimsApi.scan();
			scanMessage = `${res.added.length} added, ${res.updated.length} updated`;
			await zimsStore.load();
		} catch (err) {
			scanMessage = err instanceof Error ? err.message : 'scan failed';
		} finally {
			scanning = false;
		}
	}

	async function submitManual(e: SubmitEvent) {
		e.preventDefault();
		manualBusy = true;
		manualError = null;
		try {
			await catalogApi.download({ url: manualUrl.trim(), name: manualName.trim() });
			addManualOpen = false;
			manualUrl = '';
			manualName = '';
		} catch (err) {
			manualError = err instanceof Error ? err.message : 'failed to start download';
		} finally {
			manualBusy = false;
		}
	}

	async function saveRemote(e: SubmitEvent) {
		e.preventDefault();
		remoteSaving = true;
		remoteError = null;
		remoteSaved = false;
		try {
			await settingsApi.save({
				'inference.llm.source': 'remote',
				'inference.llm.endpoint_url': remoteUrl.trim(),
				'inference.llm.model': remoteModel.trim(),
				'inference.llm.api_key': remoteKey.trim()
			});
			remoteSaved = true;
		} catch (err) {
			remoteError = err instanceof Error ? err.message : 'failed to save';
		} finally {
			remoteSaving = false;
		}
	}

	async function downloadModel() {
		// Never re-download a preset whose GGUF is already on disk — the UI hides
		// the button for it, but guard here too so a stale click is a no-op.
		if (!customUrl.trim() && selectedPreset?.downloaded) return;
		downloading = true;
		downloadError = null;
		try {
			const body = customUrl.trim()
				? { url: customUrl.trim(), filename: customFilename.trim() }
				: { preset_id: selectedPresetId ?? presets[0]?.id ?? 'qwen3.5-4b-q4_k_s' };
			const res = await modelsApi.download(body);
			modelJobId = res.job_id;
		} catch (err) {
			downloadError = err instanceof Error ? err.message : 'failed to start download';
		} finally {
			downloading = false;
		}
	}

	// Finish: mark the wizard complete so the app never bounces back to /welcome
	// on a zero-archive state, refresh /health so the flag is live before the
	// redirect is re-evaluated, then land on search. Any download/index chain
	// kicked off here keeps running in the background via the job runner — that
	// is the whole point of "you can finish, this continues in the background".
	async function completeAndFinish() {
		finishing = true;
		try {
			await setupApi.complete();
			await healthStore.load();
		} catch {
			// Non-fatal: still leave the wizard. Worst case a reload re-checks the
			// redirect, which won't loop because archive/job state has moved on.
		} finally {
			finishing = false;
		}
		goto('/');
	}
</script>

<svelte:head>
	<title>Welcome - Vesta</title>
</svelte:head>

<WelcomeTopBar />

<main class="mx-auto max-w-[var(--wide-max)] px-5 pb-16 pt-8 max-[720px]:px-4 max-[720px]:pt-6">
	<section class="relative isolate overflow-x-clip mb-8">
		<div class="hearth-glow" aria-hidden="true"></div>
		<p class="mb-1 text-xs font-semibold uppercase tracking-wide text-faint">First run</p>
		{#if step === 1}
			<h1 class="mb-2 font-display text-3xl font-bold tracking-tight text-ink max-[720px]:text-2xl">
				Let's get you set up.
			</h1>
			<p class="max-w-2xl text-sm text-muted">
				Vesta answers questions using files on your own disk. Nothing leaves this machine once the download
				finishes. Start with something small — you can add more whenever you like.
			</p>
		{:else}
			<h1 class="mb-2 font-display text-3xl font-bold tracking-tight text-ink max-[720px]:text-2xl">
				Add an AI (optional)
			</h1>
			<p class="max-w-2xl text-sm text-muted">
				Vesta works as a search engine out of the box. An LLM turns it into a smart appliance that reads the
				results and writes you a cited answer. This is optional — you can skip it and add one later.
			</p>
		{/if}
	</section>

	<WelcomeSteps current={step} />

	{#if step === 1}
		<!-- Step 1 — archive selection -->
		{#if acquisitionManager.pending.length > 0}
			<section class="mt-8">
				<div class="mb-3">
					<h2 class="text-lg font-semibold text-ink">What's happening now</h2>
					<p class="text-xs text-faint">downloads + indexing run in the background</p>
				</div>
				<div class="flex flex-col gap-3">
					{#each acquisitionManager.pending as p (p.name)}
						<AcquisitionProgress entry={p} />
					{/each}
				</div>
				<p class="mt-3 text-sm text-muted">
					Search already works on the part that's indexed. You can keep choosing archives below while it runs.
				</p>
			</section>
		{/if}

		{#if !catalogEnabled}
			<section class="mt-8">
				<p class="rounded-md bg-surface-muted p-3 text-sm text-muted">
					Catalog browsing is disabled on this install. Use "I already have ZIM files" or add one from a URL
					below.
				</p>
			</section>
		{:else}
			<section class="mt-8">
				<div class="mb-3 flex flex-wrap items-baseline justify-between gap-2">
					<h2 class="text-lg font-semibold text-ink">Good places to start</h2>
					{#if !catalogAvailable}
						<span class="text-xs text-warning">Catalog unavailable — showing the built-in starter list.</span>
					{/if}
				</div>

				{#if featured.length > 0}
					<div class="grid grid-cols-1 gap-3 min-[721px]:grid-cols-2">
						{#each featured as slot (slot.curated.name)}
							{@const entry = curatedToEntry(slot.curated)}
							<FeaturedCard
								tag={slot.tag}
								subtitle={slot.subtitle}
								curated={slot.curated}
								{entry}
								selected={selectedKeys.has(slot.curated.name)}
								onclick={() => toggleKey(slot.curated.name)}
							/>
						{/each}
					</div>
				{/if}

				<div class="mt-6 mb-3">
					<h3 class="text-sm font-semibold text-ink-2">Add more archives</h3>
					<p class="text-xs text-faint">Pick any combination — focused corpora that complement Wikipedia.</p>
				</div>

				{#if secondary.length > 0}
					<div class="grid grid-cols-1 gap-2 min-[721px]:grid-cols-2">
						{#each secondary as slot (slot.curated.name)}
							{@const entry = curatedToEntry(slot.curated)}
							<ArchiveCheckbox
								curated={slot.curated}
								{entry}
								checked={selectedKeys.has(slot.curated.name)}
								onchange={() => toggleKey(slot.curated.name)}
							/>
						{/each}
					</div>
				{/if}

				<DiskMeter />

				{#if startError}
					<p class="mt-4 text-sm text-danger">{startError}</p>
				{/if}

				<div class="mt-5 flex items-center justify-between gap-3">
					<a href="/catalog" class="text-sm text-accent underline">Browse the full catalog</a>
					<button
						type="button"
						class="rounded-md bg-accent px-5 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-40"
						disabled={!hasSelection || starting}
						onclick={continueToAi}
					>
						{starting ? 'Starting…' : 'Continue'}
					</button>
				</div>
			</section>
		{/if}

		<!-- Secondary options: scan a folder / add from URL -->
		<section class="mt-9 grid grid-cols-1 gap-3 min-[721px]:grid-cols-2">
			<div class="flex items-start gap-3 rounded-lg border border-border bg-surface p-4">
				<svg
					viewBox="0 0 24 24"
					fill="none"
					stroke="currentColor"
					stroke-width="1.8"
					stroke-linecap="round"
					stroke-linejoin="round"
					class="mt-0.5 size-5 shrink-0 text-muted"
					><path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.7-.9L9.9 4.2A2 2 0 0 0 8.2 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"
					/></svg
				>
				<div class="min-w-0 flex-1">
					<div class="text-sm font-medium text-ink">I already have ZIM files</div>
					<p class="mb-2 text-xs text-muted">Point Vesta at a folder and it will index what it finds. No re-download.</p>
					<div class="flex flex-wrap items-center gap-2">
						<button
							type="button"
							class="rounded-md border border-border px-3 py-1.5 text-xs font-medium text-ink-2 hover:bg-surface-muted disabled:opacity-40"
							disabled={scanning}
							onclick={scanForFiles}
						>
							{scanning ? 'Scanning…' : 'Scan for files'}
						</button>
						<button
							type="button"
							class="rounded-md border border-border px-3 py-1.5 text-xs font-medium text-ink-2 hover:bg-surface-muted"
							onclick={() => (addManualOpen = true)}
						>
							Add from a URL
						</button>
						{#if scanMessage}<span class="text-xs text-faint">{scanMessage}</span>{/if}
					</div>
				</div>
			</div>
		</section>
	{:else}
		<!-- Step 2 — LLM configuration -->
		{#if acquisitionManager.pending.length > 0}
			<section class="mt-8">
				<div class="mb-3">
					<h2 class="text-lg font-semibold text-ink">Still downloading your archives</h2>
					<p class="text-xs text-faint">keep going — search works on whatever's indexed so far</p>
				</div>
				<div class="flex flex-col gap-3">
					{#each acquisitionManager.pending as p (p.name)}
						<AcquisitionProgress entry={p} />
					{/each}
				</div>
			</section>
		{/if}

		<section class="mt-8">
			<div class="flex flex-col gap-3">
				<!-- Option: existing endpoint -->
				<button
					type="button"
					onclick={() => (llmChoice = 'remote')}
					class="flex items-start gap-3 rounded-lg border p-4 text-left transition-colors {llmChoice === 'remote'
						? 'border-accent/40 bg-accent-soft'
						: 'border-border bg-surface hover:bg-surface-muted'}"
				>
					<span
						class="mt-0.5 inline-grid size-5 shrink-0 place-items-center rounded-full border {llmChoice ===
						'remote'
							? 'border-accent bg-accent text-white'
							: 'border-border-strong'}"
					>
						{#if llmChoice === 'remote'}<Check class="size-3.5" />{/if}
					</span>
					<span class="block">
						<span class="block text-sm font-semibold text-ink">Use an existing endpoint</span>
						<span class="block text-xs text-muted">
							For Ollama, LMStudio, vLLM, or any OpenAI-compatible server you already run.
						</span>
					</span>
				</button>
				{#if llmChoice === 'remote'}
					<form class="ml-8 flex flex-col gap-3 rounded-lg border border-border bg-surface p-4" onsubmit={saveRemote}>
						<label class="text-sm">
							<span class="mb-1 block text-xs text-faint">Endpoint URL</span>
							<input
								type="url"
								required
								bind:value={remoteUrl}
								placeholder="http://localhost:1234/v1"
								class="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
							/>
						</label>
						<label class="text-sm">
							<span class="mb-1 block text-xs text-faint">Model</span>
							<input
								type="text"
								required
								bind:value={remoteModel}
								placeholder="qwen3.5-4b"
								class="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
							/>
						</label>
						<label class="text-sm">
							<span class="mb-1 block text-xs text-faint">API key (optional)</span>
							<input
								type="password"
								bind:value={remoteKey}
								placeholder="sk-…"
								class="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
							/>
						</label>
						{#if remoteError}<p class="text-xs text-danger">{remoteError}</p>{/if}
						{#if remoteSaved}<p class="text-xs text-success">Saved. You can finish, or pick another option.</p>{/if}
						<div class="flex justify-end">
							<button
								type="submit"
								class="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-40"
								disabled={remoteSaving}
							>
								{remoteSaving ? 'Saving…' : 'Save'}
							</button>
						</div>
					</form>
				{/if}

				<!-- Option: download a model -->
				<button
					type="button"
					onclick={() => (llmChoice = 'download')}
					class="flex items-start gap-3 rounded-lg border p-4 text-left transition-colors {llmChoice === 'download'
						? 'border-accent/40 bg-accent-soft'
						: 'border-border bg-surface hover:bg-surface-muted'}"
				>
					<span
						class="mt-0.5 inline-grid size-5 shrink-0 place-items-center rounded-full border {llmChoice ===
						'download'
							? 'border-accent bg-accent text-white'
							: 'border-border-strong'}"
					>
						{#if llmChoice === 'download'}<Check class="size-3.5" />{/if}
					</span>
				<span class="block">
					<span class="block text-sm font-semibold text-ink">Download local AI (Qwen 3.5 4B)</span>
					<span class="block text-xs text-muted">Vesta downloads and runs Qwen 3.5 4B locally on your machine.</span>
				</span>
				</button>
				{#if llmChoice === 'download'}
					<div class="ml-8 rounded-lg border border-border bg-surface p-4">
						{#if presetsError}
							<p class="text-sm text-muted">
								Couldn't load the model preset. You can still download a custom GGUF by URL below.
							</p>
						{:else if presets.length === 0}
							<p class="text-sm text-faint">Loading model…</p>
						{:else}
							{@const p = presets[0]}
							<div
								class="flex flex-col gap-1 rounded-md border border-border bg-surface-muted/30 p-3 text-left"
								title={`Downloads from ${p.url}`}
							>
								<div class="flex items-center justify-between gap-2">
									<span class="text-sm font-medium text-ink">{p.display_name}</span>
									<span class="flex items-center gap-2">
										{#if p.downloaded}
											<span
												class="inline-flex items-center gap-1 rounded-full bg-success-soft px-2 py-0.5 text-xs font-medium text-success"
												><Check class="size-3" />Downloaded</span
											>
										{:else if presetDownloading(p)}
											<span
												class="rounded-full bg-surface-muted px-2 py-0.5 text-xs font-medium text-muted"
												>Downloading…</span
											>
										{/if}
										{#if !p.downloaded}
											<span
												class="rounded-full bg-accent-soft px-2 py-0.5 text-xs font-medium text-accent-soft-text"
												>Recommended</span
											>
										{/if}
									</span>
								</div>
								<span class="block text-xs text-faint">
									{formatBytes(p.size_bytes)} · min {p.min_ram_gb} GB RAM
								</span>
								<span class="block text-xs text-muted">{p.description}</span>
								<span class="block truncate text-xs text-faint">
									↗ {presetHost(p)}
								</span>
							</div>
						{/if}

						<div class="mt-3 border-t border-border pt-3">
							<div class="mb-2 text-xs font-medium text-ink-2">Or download a custom GGUF</div>
							<div class="flex flex-col gap-2">
								<input
									type="url"
									bind:value={customUrl}
									placeholder="https://…/model.gguf"
									class="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
								/>
								<input
									type="text"
									bind:value={customFilename}
									placeholder="filename.gguf"
									class="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
								/>
							</div>
						</div>

						{#if downloadError}<p class="mt-3 text-xs text-danger">{downloadError}</p>{/if}

						{#if modelDownloading && modelJob}
							<div class="mt-3 rounded-md border border-border bg-surface-muted p-3">
								<div class="mb-1 flex items-center justify-between text-xs">
									<span class="font-medium text-ink">{modelJob.target || 'model'}</span>
									<span class="text-faint capitalize">{modelJob.status}</span>
								</div>
								<div class="h-1.5 overflow-hidden rounded-full bg-surface">
									<div
										class="h-full rounded-full bg-accent transition-all"
										style="width: {modelJob.total
											? Math.round(((modelJob.progress ?? 0) / modelJob.total) * 100)
											: 0}%"
									></div>
								</div>
								<div class="mt-1 text-xs text-faint">
									{modelJob.total
										? `${formatBytes(modelJob.progress ?? 0)} of ${formatBytes(modelJob.total)}`
										: modelJob.message ?? 'preparing…'}
								</div>
								<p class="mt-2 text-xs text-muted">
									You can finish the setup — this continues in the background.
								</p>
							</div>
						{/if}

					<!-- The last mile: the download landed, now watch
					     the model become usable. Finish stays enabled throughout. -->
					{#if readiness === 'waiting'}
						<div class="mt-3 rounded-md border border-border bg-surface-muted p-3">
							<p class="flex items-center gap-2 text-sm text-muted">
								<span class="size-2 animate-pulse rounded-full bg-warning"></span>
								Getting {readyName ?? selectedPreset?.display_name ?? 'the model'} ready…
							</p>
							<p class="mt-1 text-xs text-faint">
								The first load takes a moment — you can finish now and ask later.
							</p>
						</div>
					{:else if readiness === 'ready'}
						<div class="mt-3 rounded-md border border-success/30 bg-success-soft p-3">
							<p class="flex items-center gap-2 text-sm font-medium text-success">
								<Check class="size-4" />
								Ready — {readyName ?? 'the model'} is loaded
							</p>
							{#if testDemo.running || testDemo.stream.state.text || testDemo.stream.state.error}
								<div class="mt-2 rounded-md border border-border bg-surface p-3">
									{#if testDemo.stream.state.text}
										<p class="max-h-28 overflow-y-auto whitespace-pre-wrap text-sm text-ink">
											{testDemo.stream.state.text}
										</p>
									{:else if testDemo.stream.state.error}
										<p class="text-xs text-danger">{testDemo.stream.state.error.message}</p>
									{:else}
										<p class="text-xs text-faint">{testDemo.stream.state.detail || 'Reading your archives…'}</p>
									{/if}
								</div>
							{/if}
							<div class="mt-2 flex justify-end">
								<button
									type="button"
									class="rounded-md border border-border bg-surface px-4 py-1.5 text-sm font-medium text-ink hover:bg-surface-muted disabled:opacity-40"
									disabled={testDemo.running}
									onclick={() => testDemo.ask()}
								>
									{testDemo.running ? 'Asking…' : 'Ask a test question'}
								</button>
							</div>
						</div>
					{:else if readiness === 'downloaded'}
						<div class="mt-3 rounded-md border border-border bg-surface-muted p-3">
							<p class="text-sm text-muted">
								{readyName ?? 'The model'} is downloaded but not loaded yet.
							</p>
							<div class="mt-2 flex justify-end">
								<button
									type="button"
									class="rounded-md border border-border bg-surface px-4 py-1.5 text-sm font-medium text-ink hover:bg-surface-muted disabled:opacity-40"
									onclick={loadNow}
								>
									Load now
								</button>
							</div>
						</div>
					{:else if readiness === 'error'}
						<div class="mt-3 rounded-md border border-danger bg-danger-soft p-3">
							<p class="text-sm text-danger">{readyError}</p>
							<a
								href="/settings?tab=settings#ai"
								class="mt-1 inline-block text-xs font-medium text-danger underline"
							>
								Open Settings → AI to fix this
							</a>
						</div>
					{/if}

						{#if !modelDownloading}
							{#if !customUrl.trim() && selectedPreset?.downloaded}
								<div class="mt-3 flex items-center justify-end gap-1.5 text-xs font-medium text-success">
									<Check class="size-4" />
									Already downloaded
								</div>
							{:else}
								<div class="mt-3 flex justify-end">
									<button
										type="button"
										class="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-40"
										disabled={downloading || (!selectedPresetId && !customUrl.trim())}
										onclick={downloadModel}
									>
										{downloading ? 'Starting…' : 'Download'}
									</button>
								</div>
							{/if}
						{/if}
					</div>
				{/if}

				<!-- Option: skip -->
				<button
					type="button"
					onclick={() => (llmChoice = 'skip')}
					class="flex items-start gap-3 rounded-lg border p-4 text-left transition-colors {llmChoice === 'skip'
						? 'border-accent/40 bg-accent-soft'
						: 'border-border bg-surface hover:bg-surface-muted'}"
				>
					<span
						class="mt-0.5 inline-grid size-5 shrink-0 place-items-center rounded-full border {llmChoice === 'skip'
							? 'border-accent bg-accent text-white'
							: 'border-border-strong'}"
					>
						{#if llmChoice === 'skip'}<Check class="size-3.5" />{/if}
					</span>
				<span class="block">
					<span class="block text-sm font-semibold text-ink">Skip for now</span>
					<span class="block text-xs text-muted">Search works without an LLM. You can add one later from Settings.</span>
				</span>
				</button>
			</div>

			<div class="mt-6 flex items-center justify-between gap-3">
				<button
					type="button"
					class="text-sm text-muted hover:text-ink"
					onclick={() => (step = 1)}
				>
					← Back
				</button>
				<button
					type="button"
					class="rounded-md bg-accent px-5 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-40"
					disabled={finishing}
					onclick={completeAndFinish}
				>
					{finishing ? 'Finishing…' : llmChoice === 'skip' ? 'Finish without AI' : 'Finish'}
				</button>
			</div>
		</section>
	{/if}
</main>

<svelte:window onkeydown={(e) => e.key === 'Escape' && addManualOpen && (addManualOpen = false)} />

{#if addManualOpen}
	<div class="fixed inset-0 z-[95] bg-black/40" role="presentation" onclick={() => (addManualOpen = false)}></div>
	<div
		class="fixed left-1/2 top-1/2 z-[96] max-h-[90dvh] w-[min(420px,92vw)] -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-xl border border-border bg-surface p-5 shadow-pop"
		role="dialog"
		aria-modal="true"
		aria-label="Add an archive from a URL"
	>
		<h3 class="mb-3 text-base font-semibold text-ink">Add from URL</h3>
		<form class="flex flex-col gap-3" onsubmit={submitManual}>
			<label class="text-sm">
				<span class="mb-1 block text-xs text-faint">Download URL</span>
				<input
					type="url"
					required
					bind:value={manualUrl}
					class="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
				/>
			</label>
			<label class="text-sm">
				<span class="mb-1 block text-xs text-faint">Name</span>
				<input
					type="text"
					required
					bind:value={manualName}
					class="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
				/>
			</label>
			{#if manualError}<p class="text-xs text-danger">{manualError}</p>{/if}
			<div class="mt-1 flex justify-end gap-2">
				<button
					type="button"
					class="rounded-md px-3 py-1.5 text-sm text-muted hover:bg-surface-muted"
					onclick={() => (addManualOpen = false)}>Cancel</button
				>
				<button
					type="submit"
					class="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-40"
					disabled={manualBusy}
				>
					Download
				</button>
			</div>
		</form>
	</div>
{/if}
