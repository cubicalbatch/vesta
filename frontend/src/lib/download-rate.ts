// Rolling median download rate, derived from completed download_zim jobs'
// `rate` field, persisted in localStorage. With no history, callers must show
// size only — never invent a bandwidth figure
// The install cost line.
const KEY = 'vesta:download-rate-samples';
const MAX_SAMPLES = 10;

function load(): number[] {
	try {
		const raw = localStorage.getItem(KEY);
		return raw ? (JSON.parse(raw) as number[]) : [];
	} catch {
		return [];
	}
}

export function recordDownloadRate(bytesPerSecond: number) {
	if (!(bytesPerSecond > 0)) return;
	const samples = [...load(), bytesPerSecond].slice(-MAX_SAMPLES);
	try {
		localStorage.setItem(KEY, JSON.stringify(samples));
	} catch {
		// localStorage unavailable (private mode, quota) — the rate just won't persist.
	}
}

export function medianDownloadRate(): number | null {
	const samples = load();
	if (samples.length === 0) return null;
	const sorted = [...samples].sort((a, b) => a - b);
	const mid = Math.floor(sorted.length / 2);
	return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}
