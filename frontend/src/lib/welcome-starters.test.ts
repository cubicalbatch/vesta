import { describe, expect, it } from 'vitest';
import { parseSizeNoteBytes, selectFeatured, selectSecondary, catalogKey } from './welcome-starters';
import type { CuratedEntry } from './types';

function curated(overrides: Partial<CuratedEntry>): CuratedEntry {
	return {
		name: 'x',
		rank: 1,
		size_note: '~1 GB',
		description: '',
		article_count: 0,
		warning: null,
		...overrides
	};
}

describe('parseSizeNoteBytes', () => {
	it('parses GB/MB/KB notes', () => {
		expect(parseSizeNoteBytes('~2.24 GB')).toBeCloseTo(2.24 * 1024 ** 3, 0);
		expect(parseSizeNoteBytes('~0.02 GB')).toBeCloseTo(0.02 * 1024 ** 3, 0);
		expect(parseSizeNoteBytes('~812 MB')).toBeCloseTo(812 * 1024 ** 2, 0);
	});

	it('sorts unparseable notes to the end (treated as unknown/large)', () => {
		expect(parseSizeNoteBytes('who knows')).toBe(Number.POSITIVE_INFINITY);
	});
});

describe('catalogKey', () => {
	it('joins name + flavour with underscore', () => {
		expect(catalogKey('wikipedia_en_top', 'nopic')).toBe('wikipedia_en_top_nopic');
	});

	it('returns bare name when flavour is empty', () => {
		expect(catalogKey('nhs.uk_en_medicines', '')).toBe('nhs.uk_en_medicines');
	});
});

describe('selectFeatured', () => {
	const realish: CuratedEntry[] = [
		curated({ name: 'wikipedia_en_100', rank: 1, size_note: '~0.18 GB' }),
		curated({ name: 'wikipedia_en_top_nopic', rank: 2, size_note: '~2.24 GB' }),
		curated({ name: 'wikivoyage_en_all_nopic', rank: 10, size_note: '~0.07 GB' }),
		curated({ name: 'gardening.stackexchange.com_en_all', rank: 16, size_note: '~0.88 GB' }),
		curated({ name: 'mdwiki_en_all_maxi', rank: 20, size_note: '~2.30 GB' })
	];

	it('picks exactly 2 featured slots from ranks 1-2', () => {
		const slots = selectFeatured(realish);
		expect(slots).toHaveLength(2);
		expect(slots[0].tag).toBe('Fastest start');
		expect(slots[0].curated.name).toBe('wikipedia_en_100');
		expect(slots[1].tag).toBe('Most useful');
		expect(slots[1].curated.name).toBe('wikipedia_en_top_nopic');
	});

	it('featured slots carry subtitles', () => {
		const slots = selectFeatured(realish);
		expect(slots[0].subtitle).toContain('Get started quick');
		expect(slots[1].subtitle).toContain('great knowledge base');
	});

	it('never picks ranks above 2 as featured', () => {
		const slots = selectFeatured(realish);
		expect(slots.every((s) => s.curated.rank <= 2)).toBe(true);
	});

	it('degrades gracefully with fewer candidates', () => {
		const slots = selectFeatured([curated({ name: 'only_one', rank: 1 })]);
		expect(slots).toHaveLength(1);
	});

	it('returns empty for no matching ranks', () => {
		const slots = selectFeatured([curated({ name: 'high_rank', rank: 20 })]);
		expect(slots).toHaveLength(0);
	});
});

describe('selectSecondary', () => {
	const realish: CuratedEntry[] = [
		curated({ name: 'wikipedia_en_100_nopic', rank: 1 }),
		curated({ name: 'wikipedia_en_top_nopic', rank: 2 }),
		curated({ name: 'wikivoyage_en_all_nopic', rank: 10 }),
		curated({ name: 'history.stackexchange.com_en_all', rank: 11 }),
		curated({ name: 'appropedia_en_all_maxi', rank: 12 }),
		curated({ name: 'nhs.uk_en_medicines', rank: 13 }),
		curated({ name: 'restarters_en_all_maxi', rank: 14 }),
		curated({ name: 'gardening.stackexchange.com_en_all', rank: 16 }),
		curated({ name: 'mdwiki_en_all_maxi', rank: 20 })
	];

	it('picks exactly 6 entries from ranks 10-16', () => {
		const slots = selectSecondary(realish);
		expect(slots).toHaveLength(6);
	});

	it('orders by rank ascending', () => {
		const slots = selectSecondary(realish);
		const ranks = slots.map((s) => s.curated.rank);
		expect(ranks).toEqual([10, 11, 12, 13, 14, 16]);
	});

	it('never includes featured or catalog-only ranks', () => {
		const slots = selectSecondary(realish);
		expect(slots.every((s) => s.curated.rank >= 10 && s.curated.rank <= 16)).toBe(true);
	});
});
