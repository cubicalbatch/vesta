<script lang="ts">
	// Catalog: the critical path for getting a ZIM onto disk and indexed
	// Catalog page.
	import { zimsStore } from '$lib/stores/zims.svelte';
	import { jobsStore } from '$lib/stores/jobs.svelte';
	import { acquisitionManager } from '$lib/stores/acquisition.svelte';
	import { catalogApi } from '$lib/api/catalog';
	import { zimsApi } from '$lib/api/zims';
	import { TERMINAL_JOB_STATUSES, type CatalogEntry, type CatalogLanguage, type CuratedEntry } from '$lib/types';
	import { lockBodyScroll } from '$lib/scroll-lock';
	import CatalogRow from '$lib/components/catalog/CatalogRow.svelte';
	import CatalogCard from '$lib/components/catalog/CatalogCard.svelte';
	import AcquisitionProgress from '$lib/components/catalog/AcquisitionProgress.svelte';
	import DiskMeter from '$lib/components/catalog/DiskMeter.svelte';
	import CatalogRefreshProgress from '$lib/components/catalog/CatalogRefreshProgress.svelte';
	import LanguageSelect from '$lib/components/catalog/LanguageSelect.svelte';

	let catalogAvailable = $state(true);
	let catalogEnabled = $state(true);
	let catalogCount = $state<number | null>(null);
	let catalogFetchedAt = $state<string | null>(null);

	let curated = $state<CuratedEntry[]>([]);
	let entries = $state<CatalogEntry[]>([]);
	let entriesTotal = $state(0);
	let loadingEntries = $state(false);

	let q = $state('');
	let language = $state('');
	let recommendedOnly = $state(false);
	let sort = $state('default');
	let maxSizeBytes = $state<number | null>(null);
	let languages = $state<CatalogLanguage[]>([]);
	let debounceTimer: ReturnType<typeof setTimeout> | null = null;

	// Max-size filter buckets. Archive sizes span ~100 MB to 221 GB, so a few
	// preset chips are more discoverable than a raw byte input; `null` = Any.
	const SIZE_PRESETS: { label: string; bytes: number | null }[] = [
		{ label: 'Any', bytes: null },
		{ label: '≤2 GB', bytes: 2 * 1024 ** 3 },
		{ label: '≤10 GB', bytes: 10 * 1024 ** 3 },
		{ label: '≤50 GB', bytes: 50 * 1024 ** 3 }
	];

	let addManualOpen = $state(false);
	let manualUrl = $state('');
	let manualName = $state('');
	let manualError = $state<string | null>(null);
	let manualBusy = $state(false);

	// Catalog refresh state. `submitting` covers the gap between the refresh
	// POST resolving and the job's first SSE event landing; the global job
	// stream is otherwise the single source of truth for refresh state.
	let submitting = $state(false);
	let refreshError = $state<string | null>(null);
	let autoRefreshAttempted = $state(false);
	let processedRefreshId = $state<number | null>(null);
	// Refresh job ids this session has started or observed active. The global job
	// stream replays every job (terminal included) as a snapshot on connect, so
	// without this guard the settle effect would fire on a *historical* terminal
	// refresh job at mount — a spurious "refresh failed" banner on the exact
	// offline first-run path this feature exists for.
	let trackedRefreshIds = $state<Set<number>>(new Set());

	$effect(() => (addManualOpen ? lockBodyScroll() : undefined));

	async function loadBrowse() {
		if (!catalogEnabled) return;
		loadingEntries = true;
		try {
		const res = await catalogApi.list({ q, language, recommended: recommendedOnly, max_size: maxSizeBytes ?? undefined, sort });
			entries = res.entries;
			entriesTotal = res.total;
			catalogAvailable = res.available;
		} catch {
			catalogAvailable = false;
			entries = [];
		} finally {
			loadingEntries = false;
		}
	}

	function scheduleBrowse() {
		if (debounceTimer) clearTimeout(debounceTimer);
		debounceTimer = setTimeout(loadBrowse, 250);
	}

	$effect(() => {
		q;
		language;
		recommendedOnly;
		sort;
		maxSizeBytes;
		scheduleBrowse();
		// Leaving the page mid-debounce would otherwise fire loadBrowse() into
		// an unmounted component.
		return () => {
			if (debounceTimer) clearTimeout(debounceTimer);
		};
	});

	$effect(() => {
		catalogApi
			.curated()
			.then((res) => (curated = res.entries.sort((a, b) => a.rank - b.rank)))
			.catch(() => {
				catalogAvailable = false;
			});
		catalogApi
			.languages()
			.then((res) => (languages = res.languages))
			.catch(() => {});
		catalogApi
			.state()
			.then((s) => {
				catalogCount = s.count;
				catalogFetchedAt = s.fetched_at;
				catalogAvailable = s.available;
			})
			.catch((err) => {
				if (err?.status === 503) catalogEnabled = false;
				else catalogAvailable = false;
			});
	});

	// Reconcile the acquisition chain whenever the job snapshot changes —
	// this is what makes the chain reload-safe
	// "The chain must survive a reload").
	$effect(() => {
		jobsStore.jobs;
		acquisitionManager.reconcile();
	});

	// ── Catalog refresh ─────────────────────────────────────────────────
	// Refreshes are explicit (the header button) with one exception: the page
	// auto-downloads the feed once on an empty cache so a fresh install isn't
	// a blank page. The refresh_catalog job is tracked on the global job stream
	// (the single source of truth for job state); `submitting` only covers the
	// gap between the POST resolving and the job's first SSE event arriving.
	const activeRefreshJob = $derived(
		[...jobsStore.jobs.values()]
			.filter((j) => j.type === 'refresh_catalog' && !TERMINAL_JOB_STATUSES.has(j.status))
			.sort((a, b) => b.id - a.id)[0]
	);
	const latestRefreshJob = $derived(
		[...jobsStore.jobs.values()]
			.filter((j) => j.type === 'refresh_catalog')
			.sort((a, b) => b.id - a.id)[0]
	);
	const refreshing = $derived(Boolean(activeRefreshJob) || submitting);
	const firstCatalogDownload = $derived((catalogCount ?? 0) === 0);

	async function startRefresh() {
		refreshError = null;
		submitting = true;
		try {
			const res = await catalogApi.refresh();
			trackedRefreshIds = new Set([...trackedRefreshIds, res.job_id]);
		} catch (err) {
			submitting = false;
			refreshError = err instanceof Error ? err.message : 'failed to start catalog refresh';
		}
		// `submitting` stays true until the job is observed on the stream (next
		// effect) or the settle handler runs — whichever comes first.
	}

	async function onRefreshSettled(failed: boolean) {
		submitting = false;
		if (failed) {
			refreshError = 'Catalog refresh failed — check your connection and try again.';
			return;
		}
		refreshError = null;
		// The cache changed: reload everything derived from it.
		try {
			const s = await catalogApi.state();
			catalogCount = s.count;
			catalogFetchedAt = s.fetched_at;
			catalogAvailable = s.available;
		} catch {
			// keep whatever we have
		}
		await loadBrowse();
		catalogApi
			.languages()
			.then((res) => (languages = res.languages))
			.catch(() => {});
	}

	// Drop the submit latch as soon as the stream shows the job, so a
	// flash-finished job doesn't strand the UI in "refreshing".
	$effect(() => {
		const active = activeRefreshJob;
		if (active) {
			submitting = false;
			// Track jobs we observe active so the settle effect reloads on their
			// completion — covers a refresh started in a prior session that's still
			// running after a page reload.
			if (!trackedRefreshIds.has(active.id)) {
				trackedRefreshIds = new Set(trackedRefreshIds).add(active.id);
			}
		}
	});

	// Reload once when the latest refresh job settles (done/error), deduped by id.
	$effect(() => {
		const job = latestRefreshJob;
		if (!job || job.id === processedRefreshId) return;
		if (!TERMINAL_JOB_STATUSES.has(job.status)) return;
		// Only settle jobs we actually started or watched run this session — a
		// terminal job replayed by the connect snapshot is history, not a refresh
		// that just finished.
		if (!trackedRefreshIds.has(job.id)) return;
		processedRefreshId = job.id;
		void onRefreshSettled(job.status === 'error');
	});

	// Auto-download the catalog once on an empty cache (first run / cleared).
	// `autoRefreshAttempted` makes it fire at most once per page session; an
	// in-flight refresh (e.g. latched from a page reload) is adopted, not
	// duplicated.
	$effect(() => {
		catalogCount;
		catalogEnabled;
		Boolean(activeRefreshJob);
		submitting;
		if (autoRefreshAttempted || !catalogEnabled || catalogCount === null) return;
		if (activeRefreshJob || submitting) {
			autoRefreshAttempted = true;
			return;
		}
		if (catalogCount === 0) {
			autoRefreshAttempted = true;
			void startRefresh();
		}
	});

	function curatedToEntry(c: CuratedEntry): CatalogEntry | undefined {
		return entries.find((e) => e.name === c.name);
	}

	const resolvedCurated = $derived(
		curated.map(curatedToEntry).filter((e): e is CatalogEntry => e !== undefined)
	);
	const unresolvedCuratedCount = $derived(curated.length - resolvedCurated.length);

	const installedNames = $derived(new Set(zimsStore.archives.map((a) => a.name)));

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

	let scanning = $state(false);
	let scanMessage = $state<string | null>(null);
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
</script>

<svelte:head>
	<title>Catalog - Vesta</title>
</svelte:head>

<div class="mx-auto max-w-[var(--content-max)]">
	<div class="mb-6">
		<h1 class="mb-1 font-display text-2xl font-bold tracking-tight text-ink">Catalog</h1>
		<p class="mb-3 text-muted">Manage what's installed, and browse what else you could add.</p>
		<div class="flex flex-wrap items-center gap-2">
			<button type="button" class="rounded-md border border-border px-3 py-1.5 text-sm font-medium text-ink-2 hover:bg-surface-muted" onclick={() => (addManualOpen = true)}>
				Add from URL
			</button>
			<button type="button" class="rounded-md border border-border px-3 py-1.5 text-sm font-medium text-ink-2 hover:bg-surface-muted disabled:opacity-40" disabled={scanning} onclick={scanForFiles}>
				I already have ZIM files
			</button>
			{#if scanMessage}<span class="text-xs text-muted">{scanMessage}</span>{/if}
		</div>
		<DiskMeter />
	</div>

	{#if acquisitionManager.pending.length > 0}
		<div class="mb-6 flex flex-col gap-2">
			{#each acquisitionManager.pending as p (p.name)}
				<AcquisitionProgress entry={p} />
			{/each}
		</div>
	{/if}

	<section class="mb-8">
		<h2 class="mb-3 text-lg font-semibold text-ink">On this machine</h2>
		{#if zimsStore.archives.length === 0}
			<p class="text-sm text-muted">Nothing installed yet — browse below or add one from a URL.</p>
		{:else}
			<div class="flex flex-col gap-3">
				{#each zimsStore.archives as archive (archive.id)}
					<CatalogRow {archive} />
				{/each}
			</div>
		{/if}
	</section>

	{#if !catalogEnabled}
		<p class="rounded-md bg-surface-muted p-3 text-sm text-muted">
			Catalog browsing is disabled on this install (<code class="font-mono text-xs">catalog.enabled = false</code>). Installed archives above still
			work, and "Add from URL" is available.
		</p>
	{:else}
		<section>
			<div class="mb-1 flex items-center justify-between gap-3">
				<h2 class="text-lg font-semibold text-ink">Worth adding</h2>
				<button
					type="button"
					class="rounded-md border border-border px-3 py-1.5 text-sm font-medium text-ink-2 hover:bg-surface-muted disabled:opacity-40"
					disabled={refreshing}
					onclick={startRefresh}
				>
					{refreshing ? 'Refreshing…' : 'Refresh catalog'}
				</button>
			</div>
			{#if refreshing}
				<CatalogRefreshProgress firstDownload={firstCatalogDownload} message={activeRefreshJob?.message ?? null} />
			{:else if refreshError}
				<p class="mb-3 text-xs text-danger">{refreshError} <button type="button" class="underline" onclick={startRefresh}>Try again</button>.</p>
			{:else if !catalogAvailable && (catalogCount ?? 0) > 0}
				<p class="mb-3 text-xs text-warning">Catalog unavailable — showing {entries.length} cached entries.</p>
			{:else if (catalogCount ?? 0) > 0}
				<p class="mb-3 text-xs text-faint">{catalogCount} entries{catalogFetchedAt ? ` · refreshed ${new Date(catalogFetchedAt).toLocaleDateString()}` : ''}</p>
			{/if}

			<!-- Only picks that resolve to a live catalog row. Curated names and
			     OPDS names drift (flavour suffixes like `_nopic`/`_maxi` that the
			     feed's `name` doesn't carry), so rendering the unmatched ones gives
			     a wall of dead "unavailable" boxes above the browse grid. The count
			     below states honestly how many were dropped instead. -->
			{#if resolvedCurated.length > 0}
				<div class="mb-5">
					<div class="mb-2 flex flex-wrap items-baseline gap-x-2 text-xs font-semibold uppercase tracking-wide text-faint">
						Curated picks
						{#if unresolvedCuratedCount > 0}
							<span class="font-normal normal-case tracking-normal">
								· {unresolvedCuratedCount} not in this catalog
							</span>
						{/if}
					</div>
					<div class="grid grid-cols-1 gap-3 min-[721px]:grid-cols-2">
						{#each resolvedCurated as entry (entry.id)}
							<CatalogCard {entry} installed={installedNames.has(entry.name)} />
						{/each}
					</div>
				</div>
			{/if}

		{#if !(refreshing && firstCatalogDownload)}
		<div class="mb-4 flex flex-col gap-2">
			<div class="flex flex-wrap gap-2">
				<!-- basis-full below 480px: sharing one row with the language picker
				     and the recommended toggle squeezes the search box to ~58px on a
				     narrow phone, which is unusable. -->
				<input
					type="text"
					bind:value={q}
					placeholder="Search the catalog…"
					class="min-w-0 basis-full rounded-md border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent min-[480px]:basis-0 min-[480px]:flex-1"
				/>
				<LanguageSelect {languages} value={language} onSelect={(code: string) => (language = code)} />
				<button
					type="button"
					class="rounded-full border px-3 py-1.5 text-xs font-medium {recommendedOnly ? 'border-accent/40 bg-accent-soft text-accent-soft-text' : 'border-border text-muted'}"
					onclick={() => (recommendedOnly = !recommendedOnly)}
				>
					Recommended only
				</button>
			</div>
			<div class="flex flex-wrap items-center gap-2">
				<span class="text-xs text-faint">Size</span>
				{#each SIZE_PRESETS as preset (preset.label)}
					<button
						type="button"
						class="rounded-full border px-3 py-1 text-xs font-medium {maxSizeBytes === preset.bytes ? 'border-accent/40 bg-accent-soft text-accent-soft-text' : 'border-border text-muted'}"
						onclick={() => (maxSizeBytes = preset.bytes)}
					>
						{preset.label}
					</button>
				{/each}
				<select
					bind:value={sort}
					class="ml-auto rounded-md border border-border bg-surface px-2 py-1.5 text-sm"
				>
					<option value="default">Sort: Default</option>
					<option value="date_desc">Newest</option>
					<option value="date_asc">Oldest</option>
					<option value="size_desc">Largest</option>
					<option value="size_asc">Smallest</option>
				</select>
			</div>
		</div>

			{#if loadingEntries}
				<p class="text-sm text-muted">Loading…</p>
			{:else if entries.length === 0}
				<p class="text-sm text-muted">{firstCatalogDownload ? 'No catalog downloaded yet — click "Refresh catalog" to fetch the list.' : 'No catalog entries match.'}</p>
			{:else}
				<!-- No `content-visibility: auto` here. On the *container* it applies
				     `contain: size layout paint` to the whole grid, so the list renders
				     at height 0 until it scrolls into view — the browse results are
				     invisible on load and the page's scroll height jumps by ~12000px
				     the moment you reach the bottom. `limit` is 60 rows; there is
				     nothing here worth virtualising. -->
				<div class="grid grid-cols-1 gap-3 min-[721px]:grid-cols-2">
					{#each entries as entry (entry.id)}
						<CatalogCard {entry} installed={installedNames.has(entry.name)} />
					{/each}
				</div>
				{#if entriesTotal > entries.length}
					<p class="mt-3 text-xs text-faint">Showing {entries.length} of {entriesTotal} — narrow your search to see more.</p>
				{/if}
			{/if}
		{/if}
		</section>
	{/if}
</div>

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
				<input type="url" required bind:value={manualUrl} class="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent" />
			</label>
			<label class="text-sm">
				<span class="mb-1 block text-xs text-faint">Name</span>
				<input type="text" required bind:value={manualName} class="w-full rounded-md border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent" />
			</label>
			{#if manualError}<p class="text-xs text-danger">{manualError}</p>{/if}
			<div class="mt-1 flex justify-end gap-2">
				<button type="button" class="rounded-md px-3 py-1.5 text-sm text-muted hover:bg-surface-muted" onclick={() => (addManualOpen = false)}>Cancel</button>
				<button type="submit" class="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-40" disabled={manualBusy}>
					Download
				</button>
			</div>
		</form>
	</div>
{/if}
