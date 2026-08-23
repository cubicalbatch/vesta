// Picks the featured cards and secondary checkbox list shown on /welcome from
// GET /api/catalog/curated.
//
// The curated list has three rank bands (see catalog/curated.py):
//   ranks 1-2   — the two featured cards ("Fastest start" + "Most useful")
//   ranks 10-16 — the multi-select secondary checkbox list
//   ranks >= 20 — additional entries for the Catalog page, not on /welcome
//
// CuratedEntry.size_note ("~2.24 GB") is available for every curated entry
// regardless of whether it currently matches a live catalog row. Display always
// uses the matched CatalogEntry.size_bytes (via CatalogCard)
// once one exists; this module never feeds a number to the UI directly.
import type { CuratedEntry } from './types';

const UNIT_BYTES: Record<string, number> = {
	KB: 1024,
	MB: 1024 ** 2,
	GB: 1024 ** 3,
	TB: 1024 ** 4
};

/** "~2.24 GB" -> bytes. Unparseable input sorts last (treated as "unknown, assume large"). */
export function parseSizeNoteBytes(note: string): number {
	const m = /([\d.]+)\s*(KB|MB|GB|TB)/i.exec(note);
	if (!m) return Number.POSITIVE_INFINITY;
	const value = parseFloat(m[1]);
	const unit = UNIT_BYTES[m[2].toUpperCase()] ?? 1;
	return value * unit;
}

export interface FeaturedSlot {
	tag: string;
	subtitle: string;
	curated: CuratedEntry;
}

export interface SecondarySlot {
	curated: CuratedEntry;
}

/**
 * The ZIM filename stem for a catalog entry's `name`/`flavour` pair — the join
 * key between a CatalogEntry and a CuratedEntry. Mirrors `catalog_key()` in
 * src/vesta/catalog/curated.py.
 */
export function catalogKey(name: string, flavour: string): string {
	return flavour ? `${name}_${flavour}` : name;
}

/** Rank ceiling for the two featured cards. */
const FEATURED_RANK_CEILING = 2;
/** Rank range for the secondary checkbox list. */
const SECONDARY_RANK_FLOOR = 10;
const SECONDARY_RANK_CEILING = 16;

/**
 * The two featured cards on /welcome: rank 1 = "Fastest start", rank 2 =
 * "Most useful". Exactly two — a third is never offered (choice paralysis on
 * the first screen is the worst UX failure for a self-hosted tool).
 *
 * Returns fewer slots if the pool has fewer entries — callers must render
 * however many come back, not assume exactly 2.
 */
export function selectFeatured(
	curated: CuratedEntry[]
): FeaturedSlot[] {
	const ranked = curated
		.filter((c) => c.rank <= FEATURED_RANK_CEILING)
		.sort((a, b) => a.rank - b.rank);

	const tags: Record<number, { tag: string; subtitle: string }> = {
		1: {
			tag: 'Fastest start',
			subtitle: 'Get started quick — you\u2019ll probably want more'
		},
		2: {
			tag: 'Most useful',
			subtitle: 'A great knowledge base for a useful Vesta'
		}
	};

	return ranked.map((c) => ({
		curated: c,
		...(tags[c.rank] ?? { tag: '', subtitle: '' })
	}));
}

/**
 * The secondary checkbox-list picks: ranks 10-16. The user selects any
 * combination. Ordered by rank.
 */
export function selectSecondary(
	curated: CuratedEntry[]
): SecondarySlot[] {
	return curated
		.filter((c) => c.rank >= SECONDARY_RANK_FLOOR && c.rank <= SECONDARY_RANK_CEILING)
		.sort((a, b) => a.rank - b.rank)
		.map((c) => ({ curated: c }));
}

/**
 * Convenience: all /welcome picks in one call (featured + secondary).
 */
export function selectWelcomePicks(curated: CuratedEntry[]) {
	return {
		featured: selectFeatured(curated),
		secondary: selectSecondary(curated)
	};
}
