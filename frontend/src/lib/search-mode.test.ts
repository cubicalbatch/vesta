import { describe, expect, it } from 'vitest';
import {
	parseSearchMode,
	searchModeEqual,
	serializeSearchMode,
	type SearchModeState
} from './search-mode';

describe('parseSearchMode', () => {
	it('an empty URL is sources mode with no query, scope, or conversation', () => {
		const s = parseSearchMode('');
		expect(s.query).toBe('');
		expect(s.ai).toBe(false);
		expect(s.conversationId).toBeNull();
		expect(s.scope.size).toBe(0);
	});

	it('handles a bare "?" prefix the same as no prefix', () => {
		expect(parseSearchMode('?q=foo')).toEqual(parseSearchMode('q=foo'));
	});

	it('reads `q` verbatim (URL-decoded by URLSearchParams)', () => {
		expect(parseSearchMode('q=hello%20world').query).toBe('hello world');
	});

	it('parses scope csv into a Set<number>, dropping non-numeric junk', () => {
		const s = parseSearchMode('scope=1,2,abc,3,,4');
		expect([...s.scope].sort((a, b) => a - b)).toEqual([1, 2, 3, 4]);
	});

	it('treats an empty/absent scope as the empty set (all archives)', () => {
		expect(parseSearchMode('q=x').scope.size).toBe(0);
		expect(parseSearchMode('scope=').scope.size).toBe(0);
	});

	it('ai=1 is on; absent, ai=0, and any other value are off', () => {
		expect(parseSearchMode('ai=1').ai).toBe(true);
		expect(parseSearchMode('ai=0').ai).toBe(false);
		expect(parseSearchMode('').ai).toBe(false);
		expect(parseSearchMode('ai=true').ai).toBe(false);
		expect(parseSearchMode('ai=yes').ai).toBe(false);
	});

	it('c is numeric or null (garbage never becomes NaN)', () => {
		expect(parseSearchMode('c=17').conversationId).toBe(17);
		expect(parseSearchMode('c=abc123').conversationId).toBeNull();
		expect(parseSearchMode('c=').conversationId).toBeNull();
		expect(parseSearchMode('').conversationId).toBeNull();
	});
});

describe('serializeSearchMode', () => {
	it('round-trips a sources-mode state', () => {
		const s: SearchModeState = { query: 'cats', scope: new Set([1, 2]), ai: false, conversationId: null };
		expect(serializeSearchMode(s)).toBe('q=cats&scope=1%2C2');
	});

	it('round-trips an AI-mode state with no conversation yet', () => {
		const s: SearchModeState = { query: 'cats', scope: new Set([3]), ai: true, conversationId: null };
		expect(serializeSearchMode(s)).toBe('q=cats&scope=3&ai=1');
	});

	it('drops `q` once a conversation id exists (the conversation supersedes it)', () => {
		const s: SearchModeState = { query: 'cats', scope: new Set([3]), ai: true, conversationId: 17 };
		expect(serializeSearchMode(s)).toBe('scope=3&ai=1&c=17');
	});

	it('omits an empty scope entirely', () => {
		const s: SearchModeState = { query: 'cats', scope: new Set(), ai: false, conversationId: null };
		expect(serializeSearchMode(s)).toBe('q=cats');
	});

	it('omits an empty query rather than emitting q=', () => {
		const s: SearchModeState = { query: '', scope: new Set(), ai: false, conversationId: null };
		expect(serializeSearchMode(s)).toBe('');
	});
});

describe('round-trip stability', () => {
	// Note: a state with BOTH a query AND a conversation id is intentionally
	// NOT round-trippable — serializeSearchMode drops `q` once `c` exists (the
	// conversation supersedes the query; see the dedicated test below). So it
	// is excluded from this set.
	const cases: SearchModeState[] = [
		{ query: '', scope: new Set(), ai: false, conversationId: null },
		{ query: 'foo', scope: new Set([1, 2]), ai: false, conversationId: null },
		{ query: 'foo', scope: new Set([1, 2]), ai: true, conversationId: null },
		{ query: '', scope: new Set([1, 2]), ai: true, conversationId: 99 }
	];

	for (const original of cases) {
		it(`round-trips ${JSON.stringify({ ...original, scope: [...original.scope] })}`, () => {
			const reparsed = parseSearchMode(serializeSearchMode(original));
			expect(searchModeEqual(original, reparsed)).toBe(true);
		});
	}

	it('the "c drops q" rule: a query is lost across a serialize/parse round-trip once a conversation id exists', () => {
		// This is the documented, deliberate non-injective case — asserted here
		// so a future "fix" that re-adds q breaks loudly.
		const withQuery: SearchModeState = {
			query: 'foo',
			scope: new Set([1]),
			ai: true,
			conversationId: 99
		};
		const reparsed = parseSearchMode(serializeSearchMode(withQuery));
		expect(reparsed.query).toBe('');
		expect(reparsed.conversationId).toBe(99);
	});

	it('round-trip is stable regardless of scope insertion order', () => {
		const a: SearchModeState = { query: 'x', scope: new Set([3, 1, 2]), ai: false, conversationId: null };
		const reparsed = parseSearchMode(serializeSearchMode(a));
		expect(searchModeEqual(a, reparsed)).toBe(true);
	});
});

describe('searchModeEqual', () => {
	it('treats scope as order-independent', () => {
		const a: SearchModeState = { query: '', scope: new Set([1, 2, 3]), ai: false, conversationId: null };
		const b: SearchModeState = { query: '', scope: new Set([3, 2, 1]), ai: false, conversationId: null };
		expect(searchModeEqual(a, b)).toBe(true);
	});

	it('distinguishes different queries, modes, ids, and scope sizes', () => {
		const base: SearchModeState = { query: 'a', scope: new Set([1]), ai: false, conversationId: null };
		expect(searchModeEqual(base, { ...base, query: 'b' })).toBe(false);
		expect(searchModeEqual(base, { ...base, ai: true })).toBe(false);
		expect(searchModeEqual(base, { ...base, conversationId: 1 })).toBe(false);
		expect(searchModeEqual(base, { ...base, scope: new Set([1, 2]) })).toBe(false);
	});
});
