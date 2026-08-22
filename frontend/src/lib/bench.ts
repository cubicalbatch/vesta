// Pure, testable helpers for the Advanced → Benchmarks page .
// Everything here is a pure function over the API shapes in types.ts — no
// fetch, no Svelte. Split out of the .svelte components so the compare
// bucketing, attribution filter, latency percentiles and wall-time estimate
// are unit-testable (mirrors lib/answer/reducer.ts being tested separately
// from the components that consume it).
import type { AttributionCell, BenchResultRow } from './types';

// ── Failure-attribution 2x2 ──────────────────────────────────────────────────
// Mirrors api/bench.py `_ATTRIBUTION_FILTERS`: a row lands in a cell by verdict
// (correct vs failed) crossed with whether the first required source was found
// (source_hit_rank != null). `unjudged`/`pending` rows match no cell — the
// backend excludes them the same way (the attribution counts come from scored
// questions only).

export const ATTRIBUTION_CELLS: AttributionCell[] = [
	'correct_source_found',
	'correct_source_missed',
	'failed_source_found',
	'failed_source_missed'
];

function isFailed(verdict: string): boolean {
	return verdict === 'incorrect' || verdict === 'partial';
}

/** Does this per-question row belong to the given attribution cell? */
export function attributionCellMatches(row: BenchResultRow, cell: AttributionCell): boolean {
	const correct = row.verdict === 'correct';
	const found = row.source_hit_rank != null;
	switch (cell) {
		case 'correct_source_found':
			return correct && found;
		case 'correct_source_missed':
			return correct && !found;
		case 'failed_source_found':
			return isFailed(row.verdict) && found;
		case 'failed_source_missed':
			return isFailed(row.verdict) && !found;
	}
}

// ── Compare bucketing ────────────────────────────────────────────────────────
// Mirrors eval/bench_runner.py `compare_runs`: fixed/broken/both over the
// shared question set. `broken` is the regression catcher (correct in A, not in
// B). only_a/only_b are the questions present in just one run (different
// subsets — the shared set is the real denominator).

export interface CompareBuckets {
	sharedDenominator: number;
	fixed: BenchResultRow[];
	broken: BenchResultRow[];
	bothCorrect: BenchResultRow[];
	bothWrong: BenchResultRow[];
	onlyA: string[];
	onlyB: string[];
}

export function computeCompareBuckets(
	rowsA: readonly BenchResultRow[],
	rowsB: readonly BenchResultRow[]
): CompareBuckets {
	const a = new Map(rowsA.map((r) => [r.question_id, r]));
	const b = new Map(rowsB.map((r) => [r.question_id, r]));
	const idsA = new Set(a.keys());
	const idsB = new Set(b.keys());
	const shared = [...idsA].filter((id) => idsB.has(id)).sort();
	const onlyA = [...idsA].filter((id) => !idsB.has(id)).sort();
	const onlyB = [...idsB].filter((id) => !idsA.has(id)).sort();

	const fixed: BenchResultRow[] = [];
	const broken: BenchResultRow[] = [];
	const bothCorrect: BenchResultRow[] = [];
	const bothWrong: BenchResultRow[] = [];
	for (const id of shared) {
		const isCorrectA = a.get(id)!.verdict === 'correct';
		const isCorrectB = b.get(id)!.verdict === 'correct';
		const rowB = b.get(id)!;
		if (isCorrectA && isCorrectB) bothCorrect.push(rowB);
		else if (!isCorrectA && !isCorrectB) bothWrong.push(rowB);
		else if (!isCorrectA && isCorrectB) fixed.push(rowB);
		else broken.push(rowB);
	}

	return { sharedDenominator: shared.length, fixed, broken, bothCorrect, bothWrong, onlyA, onlyB };
}

// ── Latency percentiles ─────────────────────────────────────────────────────
// The scorecard needs p50/p95 latency, but metrics_json has no latency
// aggregate — compute it from the per-question rows (latency_ms).

function percentile(sorted: readonly number[], p: number): number {
	if (sorted.length === 0) return 0;
	const idx = ((sorted.length - 1) * p) / 100;
	const lo = Math.floor(idx);
	const hi = Math.ceil(idx);
	if (lo === hi) return sorted[lo];
	return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

export function latencyPercentiles(
	rows: readonly BenchResultRow[],
	p1 = 50,
	p2 = 95
): { p50: number | null; p95: number | null } {
	const latencies = rows
		.map((r) => r.latency_ms)
		.filter((ms) => Number.isFinite(ms) && ms > 0)
		.sort((x, y) => x - y);
	if (latencies.length === 0) return { p50: null, p95: null };
	return { p50: percentile(latencies, p1), p95: percentile(latencies, p2) };
}

// ── Matrix size + wall-time estimate ─────────────────────────────────────────
// The run form shows the matrix size (systems x profiles x models — each cell
// is one run) and a rough wall-time estimate before pressing Run (trap 11).

export function matrixSize(systems: number, profiles: number, models: number): number {
	return systems * profiles * models;
}

/** Heuristic seconds per question (answer + judge LLM calls). Order-of-magnitude only. */
export const ESTIMATED_SECONDS_PER_QUESTION = 6;

export function estimateWallTimeSeconds(matrixSize: number, questionCount: number): number {
	return matrixSize * questionCount * ESTIMATED_SECONDS_PER_QUESTION;
}

/** Render a seconds count as a human duration ("~4 min 12 s"). */
export function formatSeconds(sec: number): string {
	const total = Math.max(0, Math.round(sec));
	if (total < 60) return `${total}s`;
	const m = Math.floor(total / 60);
	const s = total % 60;
	if (m < 60) return s ? `~${m} min ${s} s` : `~${m} min`;
	const h = Math.floor(m / 60);
	const rem = m % 60;
	return rem ? `~${h} h ${rem} min` : `~${h} h`;
}
