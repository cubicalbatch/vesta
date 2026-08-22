import { api } from './client';

// First-run wizard bookkeeping: marks setup finished so the SPA never forces the
// user back to /welcome on a zero-archive state. See api/setup.py.
export const setupApi = {
	complete: () => api.post<{ ok: boolean }>('/api/setup/complete', {})
};
