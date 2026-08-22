// The one global job stream, opened once for the app's lifetime
// Job records: "Open the global
// stream once, in a root-level store, and let every page read from it. Do not
// open a stream per component."). Reconnects with backoff on close; the
// `snapshot` burst on reconnect makes that idempotent, which is why no job
// state may live anywhere else in the browser.
import { fetchEventStream } from '../sse';
import { TERMINAL_JOB_STATUSES, type JobRecord } from '../types';

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 15000;

export class JobsStore {
	jobs = $state<Map<number, JobRecord>>(new Map());
	connected = $state(false);

	private controller: AbortController | null = null;
	private reconnectDelay = RECONNECT_BASE_MS;
	private stopped = false;

	get list(): JobRecord[] {
		return [...this.jobs.values()].sort((a, b) => b.id - a.id);
	}

	get running(): JobRecord[] {
		// "Running now" should lead with what's actually executing, then the
		// queue in FIFO order, then paused. `list` sorts by descending id, which
		// would put the most-recently-queued job on top and bury the active one
		// (it was submitted first, so it has the lowest id). Sort non-terminal
		// jobs by status rank, ties broken by ascending id (next-to-run first).
		const rank: Record<string, number> = { running: 0, queued: 1, paused: 2 };
		return [...this.jobs.values()]
			.filter((j) => !TERMINAL_JOB_STATUSES.has(j.status))
			.sort((a, b) => {
				const ra = rank[a.status] ?? 3;
				const rb = rank[b.status] ?? 3;
				if (ra !== rb) return ra - rb;
				return a.id - b.id;
			});
	}

	/** Steady / pulsing-amber / red, per the header job dot's three states. */
	get dotState(): 'idle' | 'active' | 'error' {
		if (this.running.length > 0) return 'active';
		const mostRecent = this.list[0];
		if (mostRecent?.status === 'error') return 'error';
		return 'idle';
	}

	start() {
		if (this.controller) return; // already running
		this.stopped = false;
		this.connect();
	}

	stop() {
		this.stopped = true;
		this.controller?.abort();
		this.controller = null;
	}

	private async connect() {
		this.controller = new AbortController();
		try {
			for await (const event of fetchEventStream('/api/jobs/stream', {
				signal: this.controller.signal
			})) {
				this.connected = true;
				this.reconnectDelay = RECONNECT_BASE_MS;
				this.applyWire(event.event, event.data);
			}
		} catch {
			// fall through to reconnect below
		}
		this.connected = false;
		if (this.stopped) return;
		const delay = this.reconnectDelay;
		this.reconnectDelay = Math.min(this.reconnectDelay * 2, RECONNECT_MAX_MS);
		setTimeout(() => {
			if (!this.stopped) this.connect();
		}, delay);
	}

	private applyWire(eventName: string, rawData: string) {
		let data: unknown;
		try {
			data = rawData ? JSON.parse(rawData) : null;
		} catch {
			return;
		}
		if (eventName === 'snapshot' || eventName === 'status') {
			const job = data as JobRecord;
			const next = new Map(this.jobs);
			next.set(job.id, job);
			this.jobs = next;
		} else if (eventName === 'progress') {
			const delta = data as Pick<JobRecord, 'id' | 'progress' | 'total' | 'message' | 'status'>;
			const existing = this.jobs.get(delta.id);
			if (!existing) return; // a progress delta with no prior snapshot is dropped, not guessed at
			const next = new Map(this.jobs);
			next.set(delta.id, { ...existing, ...delta });
			this.jobs = next;
		}
	}
}

export const jobsStore = new JobsStore();
