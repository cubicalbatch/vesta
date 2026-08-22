// Configured `marked` instance shared by the whole app. `lexer()` (not
// `parse()`) is the point of this module: it returns a block-token array so
// MarkdownTokens.svelte can render *components*, never `{@html}` — that's what
// makes the injection problem structural rather than something a sanitizer
// has to catch on every chunk (research/frontend-stack.md "Sanitization").
import { Marked, type Tokens, type TokenizerExtension } from 'marked';

export interface CitationToken {
	type: 'citation';
	raw: string;
	ids: number[];
}

// Directly modelled on Open WebUI's citation-extension.ts (research/frontend-stack.md).
// Turns `[1]`, `[1,2]`, `[1][3]` into a first-class inline token instead of
// letting `marked` treat it as plain text, so CitationChip can bind reactively
// to the sources store instead of the server re-emitting chip HTML every tick.
const citationExtension: TokenizerExtension = {
	name: 'citation',
	level: 'inline',
	start(src: string) {
		return src.search(/\[\d/);
	},
	tokenizer(src: string) {
		if (/^\[\^/.test(src)) return undefined; // don't eat footnote syntax
		const m = /^(\[\d+(?:,\s*\d+)*\])+/.exec(src);
		if (!m) return undefined;
		const ids = [...m[0].matchAll(/\d+/g)].map((x) => Number(x[0]));
		return { type: 'citation', raw: m[0], ids } satisfies CitationToken;
	}
};

export const md = new Marked({ gfm: true, breaks: false }).use({
	extensions: [citationExtension]
});

export type { Tokens };
