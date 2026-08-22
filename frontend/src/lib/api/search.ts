import { api } from './client';
import type { SourceCard, RetrievalTrace } from '../types';

export interface Confidence {
	top_score: number | null;
	score_dropoff: number | null;
	density: number;
	agreement: number;
}

export interface SearchResponse {
	cards: SourceCard[];
	trace: RetrievalTrace;
	confidence: Confidence;
	profile: string | null;
	profile_hash: string | null;
}

export function search(q: string, scope?: string, profile?: string): Promise<SearchResponse> {
	const params = new URLSearchParams({ q });
	if (scope) params.set('scope', scope);
	if (profile) params.set('profile', profile);
	return api.get<SearchResponse>(`/api/search?${params}`);
}
