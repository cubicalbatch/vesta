import { api } from './client';
import type { ArticleOut, DocumentOut, ScanResult } from '../types';

export const zimsApi = {
	scan: () => api.post<ScanResult>('/api/zims/scan'),
	patch: (id: number, body: { enabled?: boolean; corpus_label?: string }) =>
		api.patch<{ id: number; ok: boolean }>(`/api/zims/${id}`, body),
	remove: (id: number, keepFile: boolean) =>
		api.delete<{ id: number; ok: boolean; file_removed: boolean }>(`/api/zims/${id}?keep_file=${keepFile}`),
	triggerIndex: (id: number, depth: number) =>
		api.post<{ zim_id: number; depth: number; job_id: number }>(`/api/zims/${id}/index`, { depth }),
	// A random article, same ArticleOut shape as /api/article/{zim}/{path} —
	// backs the archive-browse page's "Random article" action (works at any
	// index_depth, including 0; it's a direct libzim read).
	randomArticle: (id: number) => api.get<ArticleOut>(`/api/zims/${id}/random`),
	// A deduplicated set of random articles, for the archive-browse page's
	// "discover" card grid. Same ArticleOut shape, server-side deduped by path
	// and attempt-capped so small archives return fewer than `count`.
	samples: (id: number, count = 6) =>
		api.get<ArticleOut[]>(`/api/zims/${id}/samples?count=${count}`),
	// The document catalog for a nautiluszim document-library ZIM (kind
	// "documents", 0013) — browsable PDF entries with the manifest title/
	// author/description + a path-preserving reader URL. Empty for non-docs
	// archives; the browse page checks `archive.kind` before calling it.
	documents: (id: number) => api.get<{ documents: DocumentOut[] }>(`/api/zims/${id}/documents`)
};
