// Mirrors src/vesta/index/estimate.py's constants and pre-commit formula
// exactly, because the calibrated estimator endpoint
// (POST /api/zims/{id}/index/estimate) only works AFTER an archive is
// registered — for a not-yet-downloaded catalog entry we need the same
// honest-range math client-side
// "The install cost line": "use the catalog's article_count with the same
// formula the estimator uses... label it a range, never a point estimate").
export const VECTORS_PER_ARTICLE: Record<number, number> = { 1: 1.5, 2: 2.2, 3: 3.0 };
export const BYTES_PER_VECTOR = 1600;
export const PRIOR_RATE_LOW = 2.0;
export const PRIOR_RATE_HIGH = 16.0;

export interface SecondsRange {
	low: number;
	high: number;
}

/** Pre-download index time range — wide and honestly uncalibrated (estimate.py's `initial_estimate`). */
export function preDownloadIndexTimeRange(articleCount: number, depth: number): SecondsRange {
	return {
		low: articleCount / PRIOR_RATE_HIGH,
		high: articleCount / PRIOR_RATE_LOW
	};
}

/** size_bytes + article_count * VECTORS_PER_ARTICLE[depth] * BYTES_PER_VECTOR — "roughly 2.4x the ZIM at depth 3". */
export function diskBytesForDepth(sizeBytes: number, articleCount: number, depth: number): number {
	const vpa = VECTORS_PER_ARTICLE[depth] ?? VECTORS_PER_ARTICLE[1];
	return sizeBytes + Math.round(articleCount * vpa * BYTES_PER_VECTOR);
}
