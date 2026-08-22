import { describe, it, expect, vi, afterEach } from 'vitest';
import type { ReaderTarget } from './stores/reader.svelte';

// pdf.js (imported transitively by ./pdf) touches `DOMMatrix` at module
// top-level; Vitest's default node environment has no DOM. Shim the bare
// minimum (a no-op transform matrix) *before* the module graph loads so the
// wrapper can be imported, then use a dynamic import. The canvas-render path
// (which would actually need a real matrix/canvas) is not exercised here —
// these tests cover classification, worker bundling, and the error path only.
globalThis.DOMMatrix ??= class DOMMatrixStub {
	constructor(_?: unknown) {}
	preMultiplySelf() {
		return this;
	}
	multiplySelf() {
		return this;
	}
	invertSelf() {
		return this;
	}
	translate() {
		return this;
	}
	scale() {
		return this;
	}
} as unknown as typeof globalThis.DOMMatrix;

const { isPdfTarget, loadPdf, PdfError, configurePdfWorker } = await import('./pdf');

const pdfTarget: ReaderTarget = {
	zimId: 19,
	path: 'files/Water (1).pdf',
	title: 'Distillation For Home Water Treatment'
};

describe('isPdfTarget — target classification', () => {
	it('renders via pdf.js for a documents-kind archive (archive card + search result)', () => {
		expect(isPdfTarget(pdfTarget, 'documents')).toBe(true);
	});

	it('keeps the iframe for non-documents kinds and missing kind', () => {
		expect(isPdfTarget(pdfTarget, 'articles')).toBe(false);
		expect(isPdfTarget(pdfTarget, 'media')).toBe(false);
		expect(isPdfTarget(pdfTarget, undefined)).toBe(false);
	});

	it('never renders pdf.js with no target', () => {
		expect(isPdfTarget(null, 'documents')).toBe(false);
	});

	it('covers the search-result touchpoint: the PDF card belongs to a documents archive', () => {
		const searchCard: ReaderTarget = {
			zimId: 19,
			path: 'files/Water (3).pdf',
			title: 'Giardia: Drinking Water Factsheet'
		};
		expect(isPdfTarget(searchCard, 'documents')).toBe(true);
	});
});

describe('worker bundling', () => {
	it('configures the worker to a local Vite asset, never a CDN', () => {
		const src = configurePdfWorker();
		expect(src).toBeTruthy();
		// The bundled worker asset path names the worker file…
		expect(src).toContain('pdf.worker');
		// …and must not reference any host the offline gate forbids.
		expect(src).not.toMatch(/https?:\/\/(fonts\.|cdn\.|unpkg|jsdelivr|api\.iconify|esm\.sh)/);
	});
});

describe('loadPdf error path', () => {
	afterEach(() => vi.restoreAllMocks());

	it('throws a PdfError when the fetch itself fails', async () => {
		vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('network down')));
		await expect(loadPdf('/api/zim/19/x.pdf')).rejects.toBeInstanceOf(PdfError);
	});

	it('throws a PdfError on a non-200 response', async () => {
		vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 404 }));
		await expect(loadPdf('/api/zim/19/x.pdf')).rejects.toBeInstanceOf(PdfError);
	});

	it('throws a PdfError for bytes that are not a readable PDF', async () => {
		vi.stubGlobal(
			'fetch',
			vi.fn().mockResolvedValue({
				ok: true,
				status: 200,
				arrayBuffer: async () => new TextEncoder().encode('definitely not a pdf').buffer
			})
		);
		await expect(loadPdf('/api/zim/19/x.pdf')).rejects.toBeInstanceOf(PdfError);
	});
});
