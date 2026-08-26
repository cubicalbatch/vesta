// Helpers for formatting curated entry names and flavours when catalog data
// is unavailable during welcome flow.

const KNOWN: Record<string, string> = {
	wikipedia: 'Wikipedia',
	wikivoyage: 'Wikivoyage',
	mdwiki: 'MDWiki',
	appropedia: 'Appropedia'
};

const FLAVOURS = new Set(['nopic', 'maxi', 'mini', 'all', 'medicines']);

/** Extract flavour suffix from a curated entry key. */
export function flavourFromKey(key: string): string {
	const parts = key.split('_');
	const last = parts[parts.length - 1];
	return FLAVOURS.has(last) ? last : '';
}

/** Humanize a curated entry key into a readable title string. */
export function humanizeName(key: string): string {
	const parts = key.split('_').filter(Boolean);
	if (parts.length > 1 && FLAVOURS.has(parts[parts.length - 1])) {
		parts.pop();
	}
	return parts
		.filter((p) => !/^[a-z]{2}$/.test(p)) // drop 2-letter language codes
		.map((p) => KNOWN[p] ?? (p.charAt(0).toUpperCase() + p.slice(1)))
		.join(' ');
}
