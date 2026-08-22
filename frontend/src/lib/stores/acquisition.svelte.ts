// Owns the download -> index acquisition chain
// The acquisition chain:
//
//   POST /api/zims/download {entry_id}      -> {job_id}
//     watch job on the global stream until status == "done"
//   GET  /api/zims                          -> find the new archive by name
//   POST /api/zims/{new_id}/index {depth}   -> {job_id}
//     watch job until done
//   archive is answerable
//
// `POST /api/zims/download` does NOT index — the UI owns the whole chain.
// Persisted to localStorage so it survives a reload at either job boundary;
// reconciliation replays purely from `/api/zims` + `/api/jobs` (the global
// job store's `snapshot` burst), never from in-memory state alone.
import { catalogApi } from '../api/catalog';
import { zimsApi } from '../api/zims';
import { zimsStore } from './zims.svelte';
import { jobsStore } from './jobs.svelte';
import { settingsValuesStore } from './settings.svelte';
import { recordDownloadRate } from '../download-rate';
import { TERMINAL_JOB_STATUSES } from '../types';

const STORAGE_KEY = 'vesta:pending-acquisitions';

export type AcquisitionStage = 'downloading' | 'registering' | 'indexing' | 'ready' | 'failed';

export interface PendingAcquisition {
	name: string;
	depth: number;
	downloadJobId: number;
	indexJobId?: number;
	zimId?: number;
	registrationFailed?: boolean;
}

function loadPending(): PendingAcquisition[] {
	try {
		const raw = localStorage.getItem(STORAGE_KEY);
		return raw ? (JSON.parse(raw) as PendingAcquisition[]) : [];
	} catch {
		return [];
	}
}

function savePending(list: PendingAcquisition[]) {
	try {
		localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
	} catch {
		// best-effort persistence only
	}
}

class AcquisitionManager {
	pending = $state<PendingAcquisition[]>(loadPending());
	private reconciling = new Set<string>();

	private update(name: string, patch: Partial<PendingAcquisition>) {
		const next = this.pending.map((p) => (p.name === name ? { ...p, ...patch } : p));
		this.pending = next;
		savePending(next);
	}

	private remove(name: string) {
		const next = this.pending.filter((p) => p.name !== name);
		this.pending = next;
		savePending(next);
	}

	stageFor(name: string): AcquisitionStage | null {
		const p = this.pending.find((x) => x.name === name);
		if (!p) return null;
		if (p.registrationFailed) return 'failed';
		if (!p.indexJobId) return p.zimId ? 'registering' : 'downloading';
		const job = jobsStore.jobs.get(p.indexJobId);
		if (job?.status === 'done') return 'ready';
		if (job?.status === 'error') return 'failed';
		return 'indexing';
	}

	async start(entryId: string, name: string, depth?: number) {
		// index.default_depth is read
		// ("do not hardcode 3"), but the setting's valid range for *this app*
		// includes 0 ("no semantic index yet" — hardware-tuned
		// first-run wizard hasn't run). POST /api/zims/{id}/index only accepts
		// 1..3, so a literal 0 here isn't "skip indexing", it's an out-of-range
		// value for this call — clamp to a real depth so the chain still makes
		// the archive answerable.
		const configured = Number(settingsValuesStore.values['index.default_depth']);
		const resolvedDepth = depth ?? (configured >= 1 && configured <= 3 ? configured : 3);
		const res = await catalogApi.download({ entry_id: entryId });
		const record: PendingAcquisition = { name: res.target ?? name, depth: resolvedDepth, downloadJobId: res.job_id };
		this.pending = [...this.pending.filter((p) => p.name !== record.name), record];
		savePending(this.pending);
	}

	/** Called reactively (from a page-level $effect) with the latest job map — advances every pending entry one step if ready. */
	async reconcile() {
		for (const p of this.pending) {
			if (this.reconciling.has(p.name)) continue;
			const stage = this.stageFor(p.name);

			if (stage === 'downloading') {
				const job = jobsStore.jobs.get(p.downloadJobId);
				if (job?.status === 'done') {
					if (job.rate) recordDownloadRate(job.rate);
					this.reconciling.add(p.name);
					try {
						await zimsStore.load();
						const archive = zimsStore.archives.find((a) => a.name === p.name);
						if (!archive) {
							this.update(p.name, { registrationFailed: true });
						} else {
							// Defensive clamp for entries persisted before this fix, or
							// any other source of an out-of-range depth (see start()).
							const depth = p.depth >= 1 && p.depth <= 3 ? p.depth : 3;
							const idx = await zimsApi.triggerIndex(archive.id, depth);
							this.update(p.name, { zimId: archive.id, indexJobId: idx.job_id });
						}
					} catch {
						// A failed index-trigger must not throw out of a fire-and-forget
						// reconcile() call (would surface only as an unhandled rejection);
						// surface it in the UI instead, same as a registration failure.
						this.update(p.name, { registrationFailed: true });
					} finally {
						this.reconciling.delete(p.name);
					}
				} else if (job?.status === 'error') {
					this.update(p.name, { registrationFailed: true });
				}
			} else if (stage === 'indexing') {
				const job = p.indexJobId ? jobsStore.jobs.get(p.indexJobId) : undefined;
				if (job && TERMINAL_JOB_STATUSES.has(job.status)) {
					await zimsStore.load();
					this.remove(p.name);
				}
			} else if (stage === 'ready') {
				// The index job can reach "done" between two reconcile() ticks (a
				// small/fast archive indexes in well under a second), so this stage
				// is reached directly without ever passing through the 'indexing'
				// terminal-job branch above — zimsStore must be refreshed here too,
				// or "On this machine" is left showing the pre-index snapshot
				// (e.g. "not indexed") forever, even though the archive is done.
				await zimsStore.load();
				this.remove(p.name);
			} else if (stage === 'failed') {
				// 'failed' is a sticky terminal state — reconcile() never touched
				// it before, so a registration that later succeeded (the user ran
				// "Scan for files" to register the .zim, or the file just landed
				// between ticks) left a dead "registration failed" banner at the
				// top of the Catalog page forever. Once the archive exists in
				// /api/zims the chain has done its job; drop the banner and let
				// the user drive indexing (and its depth) from the archive's row
				// — the phase plan calls this failure mode "recoverable by hand".
				if (this.reconciling.has(p.name)) continue;
				this.reconciling.add(p.name);
				try {
					await zimsStore.load();
					if (zimsStore.archives.some((a) => a.name === p.name)) {
						this.remove(p.name);
					}
				} finally {
					this.reconciling.delete(p.name);
				}
			}
		}
	}

	dismiss(name: string) {
		this.remove(name);
	}
}

export const acquisitionManager = new AcquisitionManager();
