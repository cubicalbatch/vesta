<script lang="ts">
	// Archive browse page — a per-archive "look inside" surface reached from the
	// Catalog row's Browse button (route: /archive/[zimId]). Three things live
	// here that the landing-page Search doesn't need: a random-article action
	// (GET /api/zims/{id}/random, a direct libzim read), a "discover" card grid
	// of a few random articles for inspiration (GET /api/zims/{id}/samples), and
	// search pre-scoped to exactly this archive. All work at any index_depth,
	// including 0 — keyword search is Xapian full-text via libzim, independent of
	// the depth-based vector index (see CatalogRow.svelte's progress copy).
	import { search as runSearch, type SearchResponse } from '$lib/api/search';
	import { zimsApi } from '$lib/api/zims';
	import { zimsStore } from '$lib/stores/zims.svelte';
	import { readerStore } from '$lib/stores/reader.svelte';
	import { depthLabel } from '$lib/index-depth';
	import { formatMediaDuration } from '$lib/format';
	import type { ArticleOut, DocumentOut } from '$lib/types';
	import { documentDisplayTitle } from '$lib/document-title';
	import SourceCard from '$lib/components/SourceCard.svelte';
	import SearchIcon from '@lucide/svelte/icons/search';
	import Shuffle from '@lucide/svelte/icons/shuffle';
	import ArrowLeft from '@lucide/svelte/icons/arrow-left';

	let { zimId }: { zimId: number } = $props();

	const archive = $derived(zimsStore.archives.find((a) => a.id === zimId));
	const archiveLabel = $derived(archive?.corpus_label ?? archive?.title ?? archive?.name ?? `archive ${zimId}`);

	let queryInput = $state('');
	let result = $state<SearchResponse | null>(null);
	let searching = $state(false);
	let searchError = $state<string | null>(null);
	let lastQuery = $state('');

	let randomLoading = $state(false);
	let randomError = $state<string | null>(null);

	// "Discover" card grid — a few random articles shown on the landing state
	// (before any search) as inspiration. Loaded once the archive resolves and
	// re-rolled on demand via the Shuffle button. Same direct libzim read as
	// the Random article action, so it works at any index_depth.
	let samples = $state<ArticleOut[]>([]);
	let samplesLoading = $state(false);

	// A nautiluszim document-library archive (kind "documents", 0013) doesn't
	// have meaningful text-article samples — its content is PDFs. When the
	// archive is documents-kind we load the manifest catalog instead and render
	// it as a titled library, replacing the discover grid entirely.
	let documents = $state<DocumentOut[]>([]);
	let documentsLoading = $state(false);

	const isDocuments = $derived(archive?.kind === 'documents');

	async function loadDocuments() {
		documentsLoading = true;
		try {
			documents = (await zimsApi.documents(zimId)).documents;
		} catch {
			documents = [];
		} finally {
			documentsLoading = false;
		}
	}

	function openDocument(doc: DocumentOut) {
		readerStore.open({
			zimId,
			path: doc.doc_path,
			title: documentDisplayTitle(doc)
		});
	}

	async function doSearch() {
		const q = queryInput.trim();
		if (!q) return;
		searching = true;
		searchError = null;
		try {
			result = await runSearch(q, String(zimId));
			lastQuery = q;
		} catch (err) {
			searchError = err instanceof Error ? err.message : 'search failed';
			result = null;
		} finally {
			searching = false;
		}
	}

	function submit(e: SubmitEvent) {
		e.preventDefault();
		doSearch();
	}

	function openCard(card: SearchResponse['cards'][number], i: number) {
		readerStore.open({
			zimId: card.zim_id,
			path: card.path,
			title: card.title,
			media: card.media,
			cards: result?.cards,
			cardIndex: i
		});
	}

	async function openRandomArticle() {
		// On a documents archive GET /api/zims/{id}/random returns one of the
		// viewer shell's stub HTML entries (e.g. "home"), which would open junk.
		// Repoint it at a random document from the already-loaded manifest so it
		// never hits the stub route . No backend change needed.
		if (isDocuments) {
			if (documents.length === 0) {
				randomError = 'This library has no documents to open.';
				return;
			}
			randomError = null;
			const doc = documents[Math.floor(Math.random() * documents.length)];
			readerStore.open({ zimId, path: doc.doc_path, title: documentDisplayTitle(doc) });
			return;
		}
		randomLoading = true;
		randomError = null;
		try {
			const article = await zimsApi.randomArticle(zimId);
			readerStore.open({ zimId, path: article.path, title: article.title, media: article.media });
		} catch (err) {
			randomError = err instanceof Error ? err.message : 'failed to load a random article';
		} finally {
			randomLoading = false;
		}
	}

	function snippet(text: string, max = 160): string {
		const flat = text.replace(/\s+/g, ' ').trim();
		return flat.length > max ? `${flat.slice(0, max).trimEnd()}…` : flat;
	}

	function openSample(article: ArticleOut) {
		readerStore.open({
			zimId: article.zim_id,
			path: article.path,
			title: article.title,
			media: article.media
		});
	}

	async function loadSamples() {
		samplesLoading = true;
		try {
			samples = await zimsApi.samples(zimId);
		} catch {
			samples = [];
		} finally {
			samplesLoading = false;
		}
	}

	// The route is keyed by zimId ({#key zimId} in +page.svelte), so this fires
	// once per archive once it resolves — no manual teardown needed.
	$effect(() => {
		if (!archive) return;
		if (archive.kind === 'documents') void loadDocuments();
		else void loadSamples();
	});
</script>

<svelte:head>
	<title>{archiveLabel} - Vesta</title>
</svelte:head>

<div class="mx-auto max-w-[var(--content-max)]">
	<a href="/catalog" class="mb-4 inline-flex items-center gap-1.5 text-sm text-muted hover:text-ink">
		<ArrowLeft class="size-4" /> Back to Catalog
	</a>

	{#if !zimsStore.loaded}
		<div class="py-10 text-center text-sm text-muted">Loading…</div>
	{:else if !archive}
		<div class="rounded-lg border border-border bg-surface p-6 text-center">
			<p class="text-sm text-ink">This archive isn't installed (or was removed).</p>
			<a href="/catalog" class="mt-1 inline-block text-sm text-accent underline">Go to Catalog</a>
		</div>
	{:else}
		<header class="mb-6">
			<div class="flex flex-wrap items-center gap-2">
				<h1 class="font-display text-2xl font-bold tracking-tight text-ink">{archiveLabel}</h1>
				{#if archive.language}<span class="rounded-full bg-surface-muted px-2 py-0.5 text-xs text-muted">{archive.language}</span>{/if}
			</div>
			<p class="mt-1 text-sm text-muted">
				{#if isDocuments}
					{documents.length} document{documents.length === 1 ? '' : 's'} ·
					{depthLabel(archive.index_depth)}
				{:else}
					{archive.article_count.toLocaleString()} articles ·
					{depthLabel(archive.index_depth)}
				{/if}
			</p>
		</header>

		<section class="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center">
			<form class="flex flex-1 items-center gap-2 rounded-xl border border-border bg-surface p-2 shadow-sm" onsubmit={submit}>
				<SearchIcon class="ml-2 size-4 shrink-0 text-faint" />
				<input
					bind:value={queryInput}
					type="text"
					placeholder="Search within {archiveLabel}…"
					aria-label="Search within {archiveLabel}"
					class="min-w-0 flex-1 bg-transparent px-1 py-2 text-base outline-none"
				/>
				<button type="submit" class="flex items-center gap-1.5 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-hover">
					Search
				</button>
			</form>
			<button
				type="button"
				class="flex items-center justify-center gap-1.5 rounded-lg border border-border bg-surface px-4 py-2.5 text-sm font-medium text-ink-2 hover:bg-surface-muted disabled:opacity-50"
				disabled={randomLoading}
				onclick={openRandomArticle}
			>
				<Shuffle class="size-4" /> {randomLoading ? 'Picking…' : isDocuments ? 'Random document' : 'Random article'}
			</button>
		</section>

		{#if randomError}
			<p class="mb-4 text-sm text-danger">{randomError}</p>
		{/if}

		{#if !searching && !result}
			{#if isDocuments}
				<section class="mb-2">
					<div class="mb-3 flex items-center justify-between gap-2">
						<h2 class="text-lg font-semibold tracking-tight text-ink">
							Library
							{#if documents.length > 0}
								<span class="ml-1 text-sm font-normal text-muted">({documents.length})</span>
							{/if}
						</h2>
					</div>
					{#if documentsLoading && documents.length === 0}
						<div class="py-8 text-center text-sm text-muted">Loading library…</div>
					{:else if documents.length > 0}
						<div class="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
							{#each documents as doc, i (i)}
								<button
									type="button"
									onclick={() => openDocument(doc)}
									class="flex flex-col rounded-lg border border-border bg-surface p-0 text-left shadow-xs transition-colors hover:border-border-strong hover:shadow-sm"
								>
									<span class="flex flex-1 flex-col gap-1 p-4">
										<span class="line-clamp-2 font-semibold tracking-tight text-ink">{documentDisplayTitle(doc)}</span>
										{#if doc.author}
											<span class="text-xs text-faint">{doc.author}</span>
										{/if}
										{#if doc.description}
											<span class="mt-1 line-clamp-3 text-sm leading-snug text-muted">{doc.description}</span>
										{/if}
									</span>
								</button>
							{/each}
						</div>
					{:else}
						<div class="rounded-lg border border-border bg-surface p-6 text-center">
							<p class="text-sm text-muted">This library has no documents to show.</p>
						</div>
					{/if}
				</section>
			{:else}
				<section class="mb-2">
					<div class="mb-3 flex items-center justify-between gap-2">
						<h2 class="text-lg font-semibold tracking-tight text-ink">Discover</h2>
						<button
							type="button"
							class="inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface px-3 py-1.5 text-xs font-medium text-muted hover:bg-surface-muted hover:text-ink-2 disabled:opacity-50"
							disabled={samplesLoading}
							onclick={loadSamples}
							title="Shuffle the suggestions"
						>
							<Shuffle class="size-3.5" />{samplesLoading ? '…' : 'Shuffle'}
						</button>
					</div>
					{#if samplesLoading && samples.length === 0}
						<div class="py-8 text-center text-sm text-muted">Finding a few things to read…</div>
					{:else if samples.length > 0}
						<div class="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
							{#each samples as article, i (i)}
								<button
									type="button"
									onclick={() => openSample(article)}
									class="flex flex-col rounded-lg border border-border bg-surface p-0 text-left shadow-xs transition-colors hover:border-border-strong hover:shadow-sm"
								>
									{#if article.media?.poster_path}
										<img
											src={`/api/zim/${article.zim_id}/${article.media.poster_path}`}
											alt=""
											class="h-32 w-full rounded-t-lg object-cover"
											loading="lazy"
										/>
									{/if}
									<span class="flex flex-1 flex-col gap-1 p-4">
										<span class="line-clamp-2 font-semibold tracking-tight text-ink">{article.title}</span>
										{#if article.media?.video_path}
											<span class="text-xs text-faint">{article.media.duration ? formatMediaDuration(article.media.duration) : 'video'}</span>
										{/if}
										{#if article.text}
											<span class="mt-1 line-clamp-3 text-sm leading-snug text-muted">{snippet(article.text)}</span>
										{/if}
									</span>
								</button>
							{/each}
						</div>
					{/if}
				</section>
			{/if}
		{/if}

		{#if searching}
			<div class="py-10 text-center text-sm text-muted">Searching…</div>
		{:else if searchError}
			<div class="rounded-lg border border-danger/30 bg-danger-soft p-4 text-sm text-danger">{searchError}</div>
		{:else if result}
			<section>
				<h2 class="mb-3 text-lg font-semibold tracking-tight text-ink">
					{result.cards.length} passage{result.cards.length === 1 ? '' : 's'}
				</h2>
				{#if result.cards.length === 0}
					<div class="rounded-lg border border-border bg-surface p-6 text-center">
						<p class="text-sm text-ink">Nothing in {archiveLabel} matched "{lastQuery}".</p>
					</div>
				{:else}
					<div class="flex flex-col gap-3">
						{#each result.cards as card, i (i)}
							<SourceCard {card} rank={i + 1} queryForHighlight={lastQuery} onOpen={() => openCard(card, i)} />
						{/each}
					</div>
				{/if}
			</section>
		{/if}
	{/if}
</div>
