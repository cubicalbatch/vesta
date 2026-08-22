// Client-side PDF rendering via Mozilla PDF.js (`pdfjs-dist`), wrapped behind a
// small module so the Svelte component stays clean and the wiring/decisions are
// unit-testable without a real PDF document.
//
// Security invariant: we always render
// the ZIM's *bytes* fetched client-side from the same-origin path-preserving
// route (`/api/zim/{zim_id}/{path}`, `application/pdf`). We never execute any
// script from the ZIM file and never relax the reader iframe's sandbox/CSP —
// this is strictly safer than relying on a browser PDF plugin, and needs no
// server change. The worker (the only other code that runs) is bundled locally
// by Vite (`?url` asset) — never fetched from a CDN (`npm run check:offline`).
import * as pdfjs from 'pdfjs-dist';
import type { PDFDocumentProxy } from 'pdfjs-dist';
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import type { Archive } from '$lib/types';
import type { ReaderTarget } from '$lib/stores/reader.svelte';

export class PdfError extends Error {}

let workerConfigured = false;

/** Point PDF.js at the Vite-bundled copy of its worker (idempotent). */
export function configurePdfWorker(): string {
	if (!workerConfigured) {
		pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;
		workerConfigured = true;
	}
	return workerUrl;
}

/**
 * Is this reader target a PDF (render with pdf.js instead of the iframe)?
 * A nautiluszim document-library archive (kind `"documents"`, 0013) has every
 * indexable entry as a PDF, so the archive kind is the reliable signal — it
 * covers BOTH touchpoints (archive "Library" card click and PDF search-source
 * card click), since a PDF search card's `zim_id` belongs to a documents archive.
 */
export function isPdfTarget(
	target: ReaderTarget | null,
	archiveKind: Archive['kind'] | undefined
): boolean {
	return !!target && archiveKind === 'documents';
}

export interface PdfDocumentHandle {
	numPages: number;
	/**
	 * Render page `pageNum` (1-based) into `canvas`, sized for the container's
	 * CSS width × devicePixelRatio (a crisp, non-tiny render). Uses the canvas's
	 * own client width, falling back to `widthPx` when layout isn't ready.
	 */
	render(pageNum: number, canvas: HTMLCanvasElement, widthPx?: number): Promise<void>;
	destroy(): void;
}

// Cap the backing-store scale so an absurd devicePixelRatio can't allocate a
// multi-thousand-pixel canvas; visually the page just never exceeds this.
const MAX_SCALE = 4;

/**
 * Load a PDF from a same-origin URL (the ZIM passthrough route) and return a
 * render handle. Throws `PdfError` for fetch/document failures so the caller
 * can show a graceful message + the guaranteed open-in-new-tab fallback.
 */
export async function loadPdf(url: string): Promise<PdfDocumentHandle> {
	configurePdfWorker();
	let res: Response;
	try {
		res = await fetch(url);
	} catch (err) {
		throw new PdfError(
			`Could not fetch the document (${err instanceof Error ? err.message : 'network error'}).`
		);
	}
	if (!res.ok) {
		throw new PdfError(`The document could not be fetched (HTTP ${res.status}).`);
	}
	let data: ArrayBuffer;
	try {
		data = await res.arrayBuffer();
	} catch {
		throw new PdfError('Could not read the document bytes.');
	}
	let doc: PDFDocumentProxy;
	let task: pdfjs.PDFDocumentLoadingTask;
	try {
		task = pdfjs.getDocument({ data });
		doc = await task.promise;
	} catch {
		throw new PdfError('This file is not a readable PDF.');
	}
	return new PdfDocumentHandleImpl(task, doc);
}

class PdfDocumentHandleImpl implements PdfDocumentHandle {
	constructor(
		private readonly task: pdfjs.PDFDocumentLoadingTask,
		private readonly doc: PDFDocumentProxy
	) {}

	get numPages(): number {
		return this.doc.numPages;
	}

	async render(pageNum: number, canvas: HTMLCanvasElement, widthPx?: number): Promise<void> {
		const page = await this.doc.getPage(pageNum);
		const dpr = (typeof window !== 'undefined' ? window.devicePixelRatio : 1) || 1;
		const layoutWidth =
			widthPx ?? canvas.clientWidth ?? canvas.parentElement?.clientWidth ?? undefined;
		const cssWidth = layoutWidth || 300;
		const base = page.getViewport({ scale: 1 });
		const scale = Math.min((cssWidth * dpr) / base.width, MAX_SCALE);
		const viewport = page.getViewport({ scale });
		canvas.width = Math.max(1, Math.floor(viewport.width));
		canvas.height = Math.max(1, Math.floor(viewport.height));
		canvas.style.width = `${Math.max(1, Math.floor(viewport.width / dpr))}px`;
		canvas.style.height = `${Math.max(1, Math.floor(viewport.height / dpr))}px`;
		// pdf.js v6 uses the `canvas` param (it obtains the 2D context itself);
		// a plain-text page with no unembedding will reject if the context is gone.
		await page.render({ canvas, viewport }).promise;
	}

	destroy(): void {
		this.task.destroy();
	}
}
