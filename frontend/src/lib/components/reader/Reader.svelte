<script lang="ts">
	// Sandboxed same-origin iframe over the path-preserving ZIM passthrough
	// zero HTML rewriting, and `sandbox` without `allow-scripts`
	// kills the entire XSS/style-collision class
	// Reader overlay.
	// Kept lean on purpose (open question Q4): source verification, not a
	// general-purpose Wikipedia browser.
	import { readerStore } from '$lib/stores/reader.svelte';
	import { zimsStore } from '$lib/stores/zims.svelte';
	import { depthLabel } from '$lib/index-depth';
	import { formatMediaDuration } from '$lib/format';
	import { api } from '$lib/api/client';
	import { lockBodyScroll } from '$lib/scroll-lock';
	import { goto } from '$app/navigation';
	import type { ArticleOut } from '$lib/types';
	import { isPdfTarget, loadPdf, PdfError, type PdfDocumentHandle } from '$lib/pdf';
	import X from '@lucide/svelte/icons/x';
	import ChevronLeft from '@lucide/svelte/icons/chevron-left';
	import ChevronRight from '@lucide/svelte/icons/chevron-right';

	const target = $derived(readerStore.target);
	const archive = $derived(target ? zimsStore.archives.find((a) => a.id === target.zimId) : undefined);
	const archiveLabel = $derived(archive?.corpus_label ?? archive?.title ?? archive?.name ?? 'archive');
	// A nautiluszim document-library archive (kind "documents", 0013) opens PDFs.
	// We render them IN the app with Mozilla pdf.js (pdfjs-dist) into a canvas
	// (see $lib/pdf.ts) instead of relying on a browser PDF plugin inside the
	// sandboxed <iframe> — every browser runs JS + paints a canvas, so the PDF
	// always renders. Security invariant preserved: we fetch the ZIM's
	// *bytes* from the same-origin passthrough route and render client-side; we
	// never run scripts from the ZIM and never relax the iframe sandbox/CSP.
	// The "Open PDF in new tab" affordance stays (print / very large / failure).
	const isPdf = $derived(isPdfTarget(target, archive?.kind));

	let article = $state<ArticleOut | null>(null);
	let articleError = $state<string | null>(null);
	let iframe = $state<HTMLIFrameElement | null>(null);
	// The zim-relative path actually showing in the iframe right now — starts
	// at `target.path` but tracks in-iframe navigation (see the poll effect
	// below), independent of `passthroughUrl` (which must stay pinned to the
	// original target so re-assigning the iframe `src` never fights the
	// user's own navigation inside it).
	let currentPath = $state<string | null>(null);

	// --- pdf.js rendering state (documents-kind targets) ---
	let pdfHandle = $state<PdfDocumentHandle | null>(null);
	let pdfStatus = $state<'idle' | 'loading' | 'ready' | 'error'>('idle');
	let pdfError = $state<string | null>(null);
	let pdfPage = $state(1);
	let pdfNumPages = $state(0);
	let pdfCanvas = $state<HTMLCanvasElement | null>(null);
	// The scrollable container holding the canvas — measured for a crisp,
	// container-filling render (the canvas's own clientWidth is its default
	// intrinsic width until laid out, which would yield a tiny render).
	let pdfStage = $state<HTMLDivElement | null>(null);

	// A new reader target (a fresh open, or prev/next across the answer's
	// cards) resets which page we think is showing.
	$effect(() => {
		currentPath = target?.path ?? null;
	});

	// The drawer covers the viewport on a phone; without this the page keeps
	// scrolling behind it and closing the reader lands you somewhere else.
	$effect(() => (target ? lockBodyScroll() : undefined));

	// Fetch outline + title for whichever path is currently showing — the
	// originally-opened one, or wherever in-iframe navigation has taken the
	// user.
	$effect(() => {
		const t = target;
		const p = currentPath;
		article = null;
		articleError = null;
		if (!t || !p) return;
		api
			.get<ArticleOut>(`/api/article/${t.zimId}/${encodeURIComponent(p)}`)
			.then((a) => (article = a))
			.catch((err) => (articleError = err instanceof Error ? err.message : 'failed to load article'));
	});

	// Load the PDF bytes (same-origin passthrough route) with pdf.js and give
	// the component a render handle. Resets whenever the target (or the
	// archive kind) changes. Reads `pdfHandle` only inside the async
	// callbacks / cleanup closure — never in the effect body — so this can't
	// become a self-triggering effect loop (frontend gotcha #2).
	$effect(() => {
		const t = target;
		const isPdf = isPdfTarget(t, archive?.kind);
		// pdfStatus/pdfPage/pdfNumPages/pdfError are not read in this body, so
		// assigning them here doesn't re-trigger the effect.
		pdfStatus = isPdf && t ? 'loading' : 'idle';
		pdfPage = 1;
		pdfNumPages = 0;
		pdfError = null;
		if (!isPdf || !t) return;
		let cancelled = false;
		const url = `/api/zim/${t.zimId}/${encodeURIComponent(t.path)}`;
		loadPdf(url)
			.then((h) => {
				if (cancelled) {
					h.destroy();
					return;
				}
				pdfHandle = h;
				pdfNumPages = h.numPages;
				pdfStatus = 'ready';
			})
			.catch((err) => {
				if (cancelled) return;
				pdfStatus = 'error';
				pdfError = err instanceof PdfError ? err.message : 'Failed to render this PDF.';
			});
		return () => {
			cancelled = true;
			pdfHandle?.destroy();
			pdfHandle = null;
		};
	});

	// (Re)render the current page into the canvas whenever the handle, the
	// canvas node, or the requested page changes.
	$effect(() => {
		const h = pdfHandle;
		const c = pdfCanvas;
		const p = pdfPage;
		if (!h || !c || pdfStatus !== 'ready') return;
		// Fill the drawer: the stage's content width (p-4 = 16px padding each
		// side). Falling back to undefined lets render() pick a sane default.
		const widthPx = pdfStage ? Math.max(1, pdfStage.clientWidth - 32) : undefined;
		h.render(p, c, widthPx).catch((err) => {
			pdfStatus = 'error';
			pdfError = err instanceof PdfError ? err.message : 'Failed to render this page.';
		});
	});

	// Because the ZIM passthrough preserves paths with zero HTML rewriting
	// (Component: Reader), internal links navigate the iframe directly and
	// Vesta only finds out by polling — there is no other signal available
	// without `allow-scripts`. Keeps the drawer breadcrumb, title and "Open
	// full article" pointed at the current entry ("Observe
	// in-iframe navigation ... to keep the drawer breadcrumb ... pointed at
	// the current entry"). Reads `currentPath` only inside the timer
	// callback, never in the effect body itself, so this can't become a
	// self-triggering effect loop (see frontend gotcha #2).
	$effect(() => {
		const t = target;
		if (!t) return;
		const prefix = `/api/zim/${t.zimId}/`;
		const id = setInterval(() => {
			let pathname: string;
			try {
				const win = iframe?.contentWindow;
				if (!win) return;
				pathname = win.location.pathname;
			} catch {
				return; // transient cross-origin state during navigation; try again next tick
			}
			if (!pathname.startsWith(prefix)) return;
			const next = decodeURIComponent(pathname.slice(prefix.length));
			if (next && next !== currentPath) currentPath = next;
		}, 400);
		return () => clearInterval(id);
	});

	const passthroughUrl = $derived(target ? `/api/zim/${target.zimId}/${target.path}` : '');
	// Media-ZIM entries (0008) carry their asset paths on the target/api payload:
	// build the native <video> src/poster from them (served via the path-
	// preserving reader route). No JS player, no sandbox relaxation needed.
	const mediaVideoUrl = $derived(
		target?.media?.video_path ? `/api/zim/${target.zimId}/${target.media.video_path}` : null
	);
	const mediaPosterUrl = $derived(
		target?.media?.poster_path ? `/api/zim/${target.zimId}/${target.media.poster_path}` : null
	);
	// Where "Open full article" points — the current in-iframe location if
	// navigation has moved past the original target, else the same URL.
	const currentPassthroughUrl = $derived(
		target && currentPath ? `/api/zim/${target.zimId}/${currentPath}` : passthroughUrl
	);
	// For a documents-kind entry the `/api/article` top-level `title` is the
	// raw libzim/filename title — the human title lives on `document.title`
	// (0013), so prefer it when present. Regular articles have `document: null`
	// and fall through to `article.title` as before.
	const displayTitle = $derived(
		article?.document?.title ??
			article?.title ??
			(currentPath === target?.path ? target?.title : undefined) ??
			currentPath ??
			target?.path ??
			''
	);

	function scrollToHeading(headingText: string) {
		try {
			const doc = iframe?.contentDocument;
			if (!doc) return;
			const heading = [...doc.querySelectorAll('h1,h2,h3,h4,h5,h6')].find(
				(h) => h.textContent?.trim() === headingText.trim()
			);
			heading?.scrollIntoView({ behavior: 'smooth', block: 'start' });
		} catch {
			// Cross-document access can throw before the iframe finishes loading; ignore.
		}
	}

	function askAboutThisArticle() {
		if (!target) return;
		const q = encodeURIComponent(`Tell me about ${displayTitle || target.title || target.path}`);
		readerStore.close();
		// Seed AI mode on the unified surface, scoped to this archive
		// Reader navigation.
		goto(`/?q=${q}&scope=${target.zimId}&ai=1`);
	}
</script>
{#if target}
	<div class="fixed inset-0 z-[90] bg-black/40" role="presentation" onclick={() => readerStore.close()}></div>
	<div
		class="fixed inset-y-0 right-0 z-[91] flex w-[60vw] flex-col bg-surface shadow-pop max-[720px]:w-screen"
		role="dialog"
		aria-modal="true"
		aria-label="Article reader"
	>
		<header class="flex items-center gap-2 border-b border-border px-4 py-3">
			<div class="min-w-0 flex-1">
				<div class="truncate text-xs text-faint">{archiveLabel}</div>
				<div class="truncate text-base font-semibold text-ink">{displayTitle}</div>
			</div>
			{#if target.cards && target.cards.length > 1 && target.cardIndex != null}
				<div class="flex shrink-0 items-center gap-1 text-xs text-muted">
					<button
						type="button"
						class="inline-grid size-7 place-items-center rounded-md hover:bg-surface-muted"
						onclick={() => readerStore.prev()}
						title="Previous source"><ChevronLeft class="size-4" /></button
					>
					{target.cardIndex + 1} / {target.cards.length}
					<button
						type="button"
						class="inline-grid size-7 place-items-center rounded-md hover:bg-surface-muted"
						onclick={() => readerStore.next()}
						title="Next source"><ChevronRight class="size-4" /></button
					>
				</div>
			{/if}
			<button
				type="button"
				class="inline-grid size-8 shrink-0 place-items-center rounded-md text-muted hover:bg-surface-muted hover:text-ink"
				onclick={() => readerStore.close()}
				aria-label="Close reader"><X class="size-[18px]" /></button
			>
		</header>

		<div class="border-b border-border bg-surface-muted px-4 py-2 font-mono text-xs text-faint">
			{archiveLabel}
			{#if archive}· indexed at depth {archive.index_depth} ({depthLabel(archive.index_depth)}){/if}
			{#if target.passageSpan}· passage span [{target.passageSpan[0]}, {target.passageSpan[1]}]{/if}
			{#if target.citedAs != null}· cited as [{target.citedAs}]{/if}
		</div>

		<div class="flex min-h-0 flex-1">
			{#if article && article.sections.length > 0}
				<nav
					class="hidden w-52 shrink-0 overflow-y-auto border-r border-border p-3 text-xs min-[1180px]:block"
					aria-label="Article outline"
				>
					{#each article.sections as section, i (i)}
						<button
							type="button"
							class="block w-full truncate rounded px-2 py-1 text-left text-muted hover:bg-surface-muted hover:text-ink"
							style="padding-left: {0.5 + (section.level - 1) * 0.75}rem"
							onclick={() => scrollToHeading(section.heading_path.at(-1) ?? '')}
						>
							{section.heading_path.at(-1)}
						</button>
					{/each}
				</nav>
			{/if}
			{#if mediaVideoUrl}
				<div class="flex min-h-0 flex-1 flex-col bg-black">
					<div class="flex items-center justify-center">
						<!-- svelte-ignore a11y_media_has_caption — media-ZIM captions are
						     optional vtt sidecars not guaranteed in the manifest (0008) -->
						<video
							controls
							autoplay
							class="max-h-full w-full object-contain"
							src={mediaVideoUrl}
							poster={mediaPosterUrl ?? undefined}
							title={displayTitle}
						></video>
					</div>
					<div class="px-4 py-3 text-sm text-white/70">
						<p class="font-medium text-white">{displayTitle}</p>
						{#if target?.media?.duration}<p class="text-xs">{formatMediaDuration(target.media.duration)}</p>{/if}
						<p class="mt-2 text-xs text-white/50">Native offline playback — this archive's entries are videos.</p>
					</div>
				</div>
			{:else}
				<div class="flex min-h-0 flex-1 flex-col">
					{#if isPdf}
						<!-- PDFs render in-app via pdf.js into a canvas ($lib/pdf.ts) —
						     works in every browser, unlike a plugin-hosted iframe. The
						     new-tab open stays a guaranteed top-level fallback (print /
						     very large docs / a corrupt file). Brief loading + graceful
						     error states; sandbox/CSP untouched (render bytes
						     client-side, never run ZIM scripts). -->
						<div class="flex items-center gap-3 border-b border-border bg-surface-muted px-4 py-2 text-sm">
							<span class="flex-1 truncate text-faint">PDF document</span>
							<a
								href={passthroughUrl}
								target="_blank"
								rel="noreferrer"
								class="shrink-0 rounded-md bg-accent px-3 py-1 font-medium text-white hover:bg-accent-hover"
								title="Open the PDF at full size in a new tab"
							>Open PDF in new tab</a>
						</div>
						<div class="flex min-h-0 flex-1 flex-col bg-neutral-100">
							{#if pdfStatus === 'loading'}
								<div class="flex min-h-0 flex-1 items-center justify-center p-6 text-sm text-muted">
									Loading PDF…
								</div>
							{:else if pdfStatus === 'error'}
								<div class="flex min-h-0 flex-1 flex-col items-center justify-center gap-3 p-6 text-center text-sm text-muted">
									<p>{pdfError}</p>
									<a
										href={passthroughUrl}
										target="_blank"
										rel="noreferrer"
										class="rounded-md bg-accent px-3 py-1.5 font-medium text-white hover:bg-accent-hover"
									>Open PDF in new tab</a>
								</div>
							{:else if pdfStatus === 'ready'}
								<div class="flex min-h-0 flex-1 flex-col">
									<div bind:this={pdfStage} class="min-h-0 flex-1 overflow-auto p-4">
										<canvas bind:this={pdfCanvas} class="mx-auto block bg-white shadow"></canvas>
									</div>
									<div class="flex items-center justify-center gap-3 border-t border-border bg-surface px-4 py-2 text-sm">
										<button
											type="button"
											class="rounded-md p-1 text-muted hover:text-ink disabled:opacity-40"
											onclick={() => (pdfPage = Math.max(1, pdfPage - 1))}
											disabled={pdfPage <= 1}
											aria-label="Previous page"
										><ChevronLeft class="size-4" /></button>
										<span class="tabular-nums text-muted">Page {pdfPage} / {pdfNumPages}</span>
										<button
											type="button"
											class="rounded-md p-1 text-muted hover:text-ink disabled:opacity-40"
											onclick={() => (pdfPage = Math.min(pdfNumPages, pdfPage + 1))}
											disabled={pdfPage >= pdfNumPages}
											aria-label="Next page"
										><ChevronRight class="size-4" /></button>
									</div>
								</div>
							{/if}
						</div>
					{:else}
						<iframe
							bind:this={iframe}
							src={passthroughUrl}
							title={target.title ?? target.path}
							sandbox="allow-same-origin"
							class="min-h-0 flex-1 border-0"
						></iframe>
					{/if}
				</div>
			{/if}
		</div>

		<footer class="flex items-center gap-3 border-t border-border px-4 py-3 text-sm">
			<button type="button" class="rounded-md bg-accent px-3 py-1.5 font-medium text-white hover:bg-accent-hover" onclick={askAboutThisArticle}>
				Ask about this article
			</button>
			<a href={currentPassthroughUrl} target="_blank" rel="noreferrer" class="text-muted hover:text-ink">Open full article</a>
		</footer>
	</div>
{/if}
