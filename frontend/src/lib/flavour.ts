// Kiwix ZIM "flavour" — the variant of an archive within the same title line
// (e.g. Wikipedia ships maxi / nopic / mini). Meanings confirmed from the ZIM
// `Tags` metadata:
//   maxi  = _pictures:yes;_videos:no;_details:yes — full article + images
//   nopic = _pictures:no;_videos:no;_details:yes  — full text, no images (best for RAG)
//   mini  = lead section only
// The pill shows the canonical code; this map supplies the plain-language
// tooltip. Unknown codes return null (no tooltip — the raw code still shows).

const FLAVOUR_DESCRIPTIONS: Readonly<Record<string, string>> = {
	maxi: 'full article with images',
	nopic: 'full text, no images',
	mini: 'lead section only'
};

/** One-line meaning for a known Kiwix flavour code, or `null` when unknown
 * (the pill still renders the raw code; only the hover tooltip is omitted). */
export function flavourDescription(code: string): string | null {
	return FLAVOUR_DESCRIPTIONS[code] ?? null;
}
