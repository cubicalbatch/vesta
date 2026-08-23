// Scheme allow-list for rendered markdown links (AUDIT_0822 M9).
//
// `marked` ships no sanitizer and raw HTML tokens are escaped by the render
// components, but link hrefs pass straight through to `<a href=...>`. LLM
// output or restored conversation text can therefore carry
// `[click](javascript:...)` / `data:text/html,...`. Every link renderer must
// gate on this helper before emitting an anchor; a refused href renders as its
// label text instead.
//
// The list is deliberately tiny: http(s), mailto, and scheme-less relative
// URLs. Anything else — javascript:, data:, vbscript:, file:, unknown schemes,
// unparseable junk — is refused. Browsers strip tab/LF/CR anywhere in a URL
// and leading/trailing C0 controls, so an obfuscated `java\tscript:` must be
// judged on its cleaned form; we clean exactly like a browser would, then let
// the URL parser normalize case and classify.
const ALLOWED_SCHEMES = new Set(['http:', 'https:', 'mailto:']);

const SCHEME = /^([a-zA-Z][a-zA-Z0-9+.-]*):/;

export function isSafeLinkHref(rawHref: string): boolean {
	if (typeof rawHref !== 'string') return false;
	// Browsers remove tab/LF/CR wherever they appear in a URL and strip
	// C0-controls/space from both ends, so mirror that first or
	// "\u0020javascript:" / "java\tscript:" looks harmless here yet executes.
	const cleaned = rawHref
		.replace(/[\t\n\r]/g, '')
		.replace(/^[\u0000-\u0020]+|[\u0000-\u0020]+$/g, '');
	// Any other control char in a URL is either stripped at the edges or makes
	// the whole thing invalid; refuse instead of guessing what a browser does.
	if (/[\u0000-\u001f\u007f]/.test(cleaned)) return false;
	const match = SCHEME.exec(cleaned);
	if (!match) return true; // no scheme: path, root-relative, fragment, query
	try {
		return ALLOWED_SCHEMES.has(new URL(cleaned).protocol);
	} catch {
		return false; // scheme-shaped but unparseable → refuse
	}
}
