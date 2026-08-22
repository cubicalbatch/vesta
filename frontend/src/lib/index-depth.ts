// Index depth in words, not numbers
// "Page: Catalog"): depth 0 -> "not indexed", 1 -> "keyword only", 2 ->
// "keyword + meaning", 3 -> "keyword + meaning + rerank". Keep the 0-3 pip as
// a secondary cue with the number in its `title`, never as the primary label.
const LABELS = ['not indexed', 'keyword only', 'keyword + meaning', 'keyword + meaning + rerank'];

export function depthLabel(depth: number): string {
	return LABELS[depth] ?? `depth ${depth}`;
}
