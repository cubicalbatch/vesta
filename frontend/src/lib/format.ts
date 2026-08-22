export function formatBytes(bytes: number): string {
	if (bytes < 1024) return `${bytes} B`;
	const units = ['KB', 'MB', 'GB', 'TB'];
	let v = bytes / 1024;
	let i = 0;
	while (v >= 1024 && i < units.length - 1) {
		v /= 1024;
		i += 1;
	}
	return `${v.toFixed(v >= 10 ? 0 : 1)} ${units[i]}`;
}

export function formatDuration(seconds: number): string {
	if (seconds < 60) return `${Math.round(seconds)}s`;
	const minutes = seconds / 60;
	if (minutes < 60) return `${Math.round(minutes)} min`;
	const hours = minutes / 60;
	return `${hours.toFixed(1)} h`;
}

/** Renders an ISO datetime, or `—` for null/empty/unparseable input — never
 * the literal "Invalid Date" a bare `new Date(x).toLocaleString()` produces.
 * Needed because at least one backend row (eval run started_at — a bug in
 * `SqliteEvalStore.update_run`, src/vesta/api/eval.py: the UPDATE statement
 * never touches the started_at column) can genuinely ship an empty string. */
export function formatDate(iso: string | null | undefined): string {
	if (!iso) return '—';
	const d = new Date(iso);
	return Number.isNaN(d.getTime()) ? '—' : d.toLocaleString();
}

/** Catalog `zim_date` is dc:issued — a full ISO datetime ("2026-07-06T00:00:00Z")
 * or a bare year/month ("2026-06"). Render a compact "Mon YYYY" using UTC (these
 * are UTC timestamps; a local-tz render would shift a midnight UTC date back a
 * day/month). Returns "—" for null/empty/unparseable input — never the literal
 * "Invalid Date". A bare four-digit year (which `new Date` rejects) is passed
 * through unchanged. Deterministic month names (no locale dependency). */
const ZIM_MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
export function formatZimDate(iso: string | null | undefined): string {
	if (!iso) return '—';
	if (/^\d{4}$/.test(iso)) return iso; // bare year — no month to render
	const d = new Date(iso);
	if (Number.isNaN(d.getTime())) return '—';
	return `${ZIM_MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
}

export function formatCount(n: number): string {
	if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
	if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
	return String(n);
}
