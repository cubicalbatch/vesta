// Advanced → Eval (src/vesta/api/eval.py). Retrieval-only golden-set runs —
// see EvalRunMetrics in types.ts for exactly what fields exist (recall@k,
// mrr, ndcg, latency, degraded — no citation-precision/refusal-rate; those
// are answer-benchmark concepts, see api/bench.ts).
import { api } from './client';
import type { EvalRunDetail, EvalRunResponse } from '../types';

export const evalApi = {
	run: (body: { profile?: string | null; golden_set?: string; notes?: string }) =>
		api.post<EvalRunResponse>('/api/eval/run', body),
	listRuns: (limit = 50) => api.get<EvalRunDetail[]>(`/api/eval/runs?limit=${limit}`),
	getRun: (id: number) => api.get<EvalRunDetail>(`/api/eval/runs/${id}`)
};
