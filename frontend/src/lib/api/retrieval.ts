// Retrieval profiles list — used by the eval + benchmark forms to populate
// their profile dropdowns. The profile *editor* UI was removed; only the
// read-only listing remains (src/vesta/api/retrieval.py).
import { api } from './client';
import type { ProfileItem } from '../types';

export const retrievalApi = {
	listProfiles: () => api.get<{ profiles: ProfileItem[] }>('/api/retrieval/profiles')
};
