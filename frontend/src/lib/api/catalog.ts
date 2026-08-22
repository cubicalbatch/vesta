import { api } from './client';
import type {
	CatalogLanguage,
	CatalogList,
	CatalogState,
	CuratedEntry
} from '../types';

export interface CatalogFilter {
	q?: string;
	language?: string;
	recommended?: boolean;
	max_size?: number; // bytes; excludes larger archives
	sort?: string; // default|relevance|size_asc|size_desc|date_asc|date_desc
	limit?: number;
	offset?: number;
}

export const catalogApi = {
	list: (f: CatalogFilter = {}) => {
		const params = new URLSearchParams();
		if (f.q) params.set('q', f.q);
		if (f.language) params.set('language', f.language);
		if (f.recommended) params.set('recommended', 'true');
		if (f.max_size != null) params.set('max_size', String(f.max_size));
		if (f.sort) params.set('sort', f.sort);
		params.set('limit', String(f.limit ?? 60));
		params.set('offset', String(f.offset ?? 0));
		return api.get<CatalogList>(`/api/catalog?${params}`);
	},
	curated: () => api.get<{ entries: CuratedEntry[] }>('/api/catalog/curated'),
	state: () => api.get<CatalogState>('/api/catalog/state'),
	languages: () => api.get<{ languages: CatalogLanguage[] }>('/api/catalog/languages'),
	refresh: () => api.post<{ job_id: number; job_type: string }>('/api/catalog/refresh'),
	download: (body: { entry_id?: string; url?: string; name?: string }) =>
		api.post<{ job_id: number; job_type: string; target: string | null }>('/api/zims/download', body)
};
