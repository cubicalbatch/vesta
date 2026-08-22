// Job control (src/vesta/api/jobs.py). Listing/streaming is jobsStore's job
// (the global SSE connection).
// records") — this file is only the mutating calls Advanced's Jobs tab needs.
import { api, ApiError } from './client';

export interface JobControlResult {
	id: number;
	status: string;
	/** true when the server 409'd — someone else already changed the job's
	 * state. Treated as success ("treat 409 as someone
	 * else already did it, not an error"), never surfaced as a failure. */
	alreadyChanged: boolean;
}

async function control(id: number, action: 'pause' | 'resume' | 'cancel'): Promise<JobControlResult> {
	try {
		const res = await api.post<{ id: number; status: string }>(`/api/jobs/${id}/${action}`);
		return { ...res, alreadyChanged: false };
	} catch (err) {
		if (err instanceof ApiError && err.status === 409) {
			return { id, status: '', alreadyChanged: true };
		}
		throw err;
	}
}

export const jobsApi = {
	pause: (id: number) => control(id, 'pause'),
	resume: (id: number) => control(id, 'resume'),
	cancel: (id: number) => control(id, 'cancel')
};
