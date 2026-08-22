// Client-side snippet highlighting over `snippet` — the backend never returns
// markup. Returns segments for the caller to render as text nodes / <mark>
// elements; never build this via innerHTML
// Text highlighting in search results.
export interface HighlightSegment {
	text: string;
	mark: boolean;
}

const STOPWORDS = new Set([
	'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'and', 'or', 'is',
	'are', 'was', 'were', 'what', 'who', 'when', 'where', 'why', 'how'
]);

export function contentWords(query: string): string[] {
	return query
		.toLowerCase()
		.split(/[^a-z0-9]+/)
		.filter((w) => w.length > 1 && !STOPWORDS.has(w));
}

export function highlightSegments(text: string, query: string): HighlightSegment[] {
	const words = contentWords(query);
	if (words.length === 0) return [{ text, mark: false }];

	const pattern = new RegExp(`(${words.map(escapeRegExp).join('|')})`, 'gi');
	const parts = text.split(pattern);
	return parts
		.filter((p) => p.length > 0)
		.map((part) => ({ text: part, mark: words.includes(part.toLowerCase()) }));
}

function escapeRegExp(s: string): string {
	return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
