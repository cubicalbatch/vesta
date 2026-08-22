// Advanced → Benchmarks (src/vesta/api/bench.py) — the unified end-to-end
// answer benchmark. Replaces the removed /api/benchmark router.
// The results feed deliberately never carries trace_json (trap 10), so the
// per-question rows here have no trace to render.
import { api } from './client';
import type {
	BenchCompareResponse,
	BenchDatasetInfo,
	BenchResultsPage,
	BenchResultRow,
	BenchRunDetail,
	BenchRunResponse,
	BenchRunSummary
} from '../types';

export type AttributionFilter = 'correct_source_found' | 'correct_source_missed' | 'failed_source_found' | 'failed_source_missed';

export const benchApi = {
	run: (body: {
		systems?: string[];
		profiles?: string[];
		models?: string[];
		dataset?: string | null;
		slice?: string | null;
		capabilities?: string[];
		limit?: number | null;
		scope?: string | null;
		judge_model?: string | null;
		repeats?: number | null;
		label?: string | null;
	}) => api.post<BenchRunResponse>('/api/bench/run', body),
	listRuns: () => api.get<BenchRunSummary[]>('/api/bench/runs'),
	getRun: (id: number) => api.get<BenchRunDetail>(`/api/bench/runs/${id}`),
	results: (id: number, params?: { verdict?: string; capability?: string; attribution?: AttributionFilter; offset?: number; limit?: number }) => {
		const qs = new URLSearchParams();
		if (params?.verdict) qs.set('verdict', params.verdict);
		if (params?.capability) qs.set('capability', params.capability);
		if (params?.attribution) qs.set('attribution', params.attribution);
		if (params?.offset != null) qs.set('offset', String(params.offset));
		if (params?.limit != null) qs.set('limit', String(params.limit));
		const q = qs.toString();
		return api.get<BenchResultsPage>(`/api/bench/runs/${id}/results${q ? `?${q}` : ''}`);
	},
	compare: (runs: number[]) => api.get<BenchCompareResponse>(`/api/bench/compare?runs=${runs.join(',')}`),
	dataset: () => api.get<BenchDatasetInfo>('/api/bench/dataset'),
	remove: (id: number) => api.delete<{ deleted: number }>(`/api/bench/runs/${id}`),
	/** Fetch every per-question row for a run (paginates the results feed). */
	allResults: async (id: number): Promise<BenchResultRow[]> => {
		const items: BenchResultRow[] = [];
		const pageSize = 500;
		let offset = 0;
		for (;;) {
			const page = await api.get<BenchResultsPage>(
				`/api/bench/runs/${id}/results?offset=${offset}&limit=${pageSize}`
			);
			items.push(...page.items);
			if (offset + page.items.length >= page.total) break;
			offset += page.items.length;
		}
		return items;
	}
};
