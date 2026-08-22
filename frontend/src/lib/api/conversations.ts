import { api } from './client';
import type { ConversationDetail, ConversationSummary } from '../types';

export const conversationsApi = {
	list: (limit = 50) => api.get<ConversationSummary[]>(`/api/conversations?limit=${limit}`),
	get: (id: number, limit = 200) =>
		api.get<ConversationDetail>(`/api/conversations/${id}?limit=${limit}`),
	remove: (id: number) => api.delete<{ deleted: boolean }>(`/api/conversations/${id}`)
};
