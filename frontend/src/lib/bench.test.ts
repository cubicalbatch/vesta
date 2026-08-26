import { describe, expect, it } from 'vitest';
import type { BenchResultRow } from './types';
import {
	ATTRIBUTION_CELLS,
	attributionCellMatches,
	computeCompareBuckets,
	estimateWallTimeSeconds,
	formatSeconds,
	latencyPercentiles,
	matrixSize
} from './bench';

function row(question_id: string, verdict: string, source_hit_rank: number | null, latency_ms = 0): BenchResultRow {
	return {
		run_id: 1,
		question_id,
		capability: 'lookup',
		difficulty: 'easy',
		question_text: `q ${question_id}`,
		expected_answer: 'expected',
		answer_text: 'answer',
		abstained: false,
		verdict,
		verdict_reason: '',
		source_hit_rank,
		source_coverage: 0,
		sub_fact_coverage: null,
		retrieved_paths: [],
		rounds: 0,
		latency_ms,
		error: null
	};
}

describe('attributionCellMatches (mirrors api/bench.py _ATTRIBUTION_FILTERS)', () => {
	it('covers exactly the four cells', () => {
		expect(ATTRIBUTION_CELLS).toEqual([
			'correct_source_found',
			'correct_source_missed',
			'failed_source_found',
			'failed_source_missed'
		]);
	});

	it('correct + source found → correct_source_found only', () => {
		const r = row('a', 'correct', 1);
		expect(attributionCellMatches(r, 'correct_source_found')).toBe(true);
		expect(attributionCellMatches(r, 'correct_source_missed')).toBe(false);
		expect(attributionCellMatches(r, 'failed_source_found')).toBe(false);
		expect(attributionCellMatches(r, 'failed_source_missed')).toBe(false);
	});

	it('correct + source missed → correct_source_missed only', () => {
		const r = row('b', 'correct', null);
		expect(attributionCellMatches(r, 'correct_source_missed')).toBe(true);
		expect(attributionCellMatches(r, 'correct_source_found')).toBe(false);
	});

	it('partial/incorrect + source found → failed_source_found only', () => {
		expect(attributionCellMatches(row('c', 'incorrect', 2), 'failed_source_found')).toBe(true);
		expect(attributionCellMatches(row('d', 'partial', 1), 'failed_source_found')).toBe(true);
		expect(attributionCellMatches(row('c', 'incorrect', 2), 'failed_source_missed')).toBe(false);
		expect(attributionCellMatches(row('c', 'incorrect', 2), 'correct_source_found')).toBe(false);
	});

	it('partial/incorrect + source missed → failed_source_missed only', () => {
		expect(attributionCellMatches(row('e', 'incorrect', null), 'failed_source_missed')).toBe(true);
		expect(attributionCellMatches(row('f', 'partial', null), 'failed_source_missed')).toBe(true);
	});

	it('unjudged/pending rows match no cell (backend excludes them too)', () => {
		for (const v of ['unjudged', 'pending']) {
			const r = row('g', v, 1);
			for (const cell of ATTRIBUTION_CELLS) expect(attributionCellMatches(r, cell)).toBe(false);
		}
	});
});

describe('computeCompareBuckets (mirrors eval/bench_runner.py compare_runs)', () => {
	const a = [
		row('both_correct', 'correct', 1),
		row('both_wrong', 'incorrect', null),
		row('fixed', 'incorrect', null),
		row('broken', 'correct', 1)
	];
	const b = [
		row('both_correct', 'correct', 1),
		row('both_wrong', 'incorrect', null),
		row('fixed', 'correct', 1),
		row('broken', 'incorrect', null)
	];

	it('buckets the shared set: fixed / broken / both correct / both wrong', () => {
		const buckets = computeCompareBuckets(a, b);
		expect(buckets.sharedDenominator).toBe(4);
		expect(buckets.fixed.map((r) => r.question_id)).toEqual(['fixed']);
		expect(buckets.broken.map((r) => r.question_id)).toEqual(['broken']);
		expect(buckets.bothCorrect.map((r) => r.question_id)).toEqual(['both_correct']);
		expect(buckets.bothWrong.map((r) => r.question_id)).toEqual(['both_wrong']);
	});

	it('broken is the regression catcher: correct in A, not correct in B', () => {
		expect(a.find((r) => r.question_id === 'broken')?.verdict).toBe('correct');
		expect(b.find((r) => r.question_id === 'broken')?.verdict).toBe('incorrect');
	});

	it('only_a / only_b separate the questions that are not in the shared set', () => {
		const onlyA = [row('only_a', 'correct', 1)];
		const onlyB = [row('only_b', 'incorrect', null)];
		const buckets = computeCompareBuckets([...a, ...onlyA], [...b, ...onlyB]);
		expect(buckets.onlyA).toEqual(['only_a']);
		expect(buckets.onlyB).toEqual(['only_b']);
		expect(buckets.sharedDenominator).toBe(4);
	});

	it('empty intersection → no buckets, zero shared denominator', () => {
		const buckets = computeCompareBuckets([row('x', 'correct', 1)], [row('y', 'correct', 1)]);
		expect(buckets.sharedDenominator).toBe(0);
		expect(buckets.fixed).toEqual([]);
		expect(buckets.broken).toEqual([]);
		expect(buckets.bothCorrect).toEqual([]);
		expect(buckets.bothWrong).toEqual([]);
		expect(buckets.onlyA).toEqual(['x']);
		expect(buckets.onlyB).toEqual(['y']);
	});

	it('pending/unjudged on either side → unjudged bucket, never fixed/broken', () => {
		for (const v of ['unjudged', 'pending']) {
			// would-be fix: A unjudged, B correct
			let buckets = computeCompareBuckets([row('q', v, 1)], [row('q', 'correct', 1)]);
			expect(buckets.unjudged.map((r) => r.question_id)).toEqual(['q']);
			expect(buckets.fixed).toEqual([]);
			expect(buckets.bothCorrect).toEqual([]);
			// would-be regression: A correct, B unjudged
			buckets = computeCompareBuckets([row('q', 'correct', 1)], [row('q', v, 1)]);
			expect(buckets.unjudged.map((r) => r.question_id)).toEqual(['q']);
			expect(buckets.broken).toEqual([]);
			expect(buckets.bothWrong).toEqual([]);
		}
	});

	it('both sides unjudged → unjudged bucket', () => {
		const buckets = computeCompareBuckets(
			[row('q', 'unjudged', null)],
			[row('q', 'pending', 1)]
		);
		expect(buckets.unjudged.map((r) => r.question_id)).toEqual(['q']);
		expect(buckets.fixed).toEqual([]);
		expect(buckets.broken).toEqual([]);
		expect(buckets.bothCorrect).toEqual([]);
		expect(buckets.bothWrong).toEqual([]);
	});

	it('unjudged stays in the shared denominator but out of scored buckets', () => {
		const a = [row('ok', 'correct', 1), row('uj', 'correct', 1)];
		const b = [row('ok', 'incorrect', null), row('uj', 'unjudged', 1)];
		const buckets = computeCompareBuckets(a, b);
		expect(buckets.sharedDenominator).toBe(2);
		expect(buckets.broken.map((r) => r.question_id)).toEqual(['ok']);
		expect(buckets.unjudged.map((r) => r.question_id)).toEqual(['uj']);
	});
});

describe('latencyPercentiles', () => {
	it('returns null when there are no finite positive latencies', () => {
		expect(latencyPercentiles([row('a', 'correct', 1)])).toEqual({ p50: null, p95: null });
		expect(latencyPercentiles([row('a', 'correct', 1, 0), row('b', 'correct', 1, -5)])).toEqual({
			p50: null,
			p95: null
		});
	});

	it('computes p50/p95 over the positive latencies, ignoring non-positive', () => {
		const rows = [row('a', 'correct', 1, 100), row('b', 'correct', 1, 200), row('c', 'correct', 1, 300), row('d', 'correct', 1, 0)];
		const { p50, p95 } = latencyPercentiles(rows);
		expect(p50).toBe(200);
		expect(p95).toBe(290);
	});
});

describe('matrix size + wall-time estimate', () => {
	it('matrix size is systems x profiles x models', () => {
		expect(matrixSize(2, 1, 1)).toBe(2);
		expect(matrixSize(3, 2, 2)).toBe(12);
		expect(matrixSize(0, 1, 1)).toBe(0);
	});

	it('wall time scales with matrix size and question count', () => {
		expect(estimateWallTimeSeconds(2, 100)).toBe(2 * 100 * 6);
		expect(estimateWallTimeSeconds(1, 50)).toBe(300);
	});

	it('formatSeconds renders human durations', () => {
		expect(formatSeconds(45)).toBe('45s');
		expect(formatSeconds(60)).toBe('~1 min');
		expect(formatSeconds(252)).toBe('~4 min 12 s');
		expect(formatSeconds(3600)).toBe('~1 h');
		expect(formatSeconds(4500)).toBe('~1 h 15 min');
	});
});
