<script lang="ts">
	// buffer -> remend(heals unterminated syntax) -> marked.lexer -> keyed tokens.
	// rAF-throttled while streaming; synchronous on `done` so the final render
	// is exact (research/frontend-stack.md "The streaming-markdown deep dive").
	import remend from 'remend';
	import { md, type Tokens } from './marked';
	import MarkdownTokens from './MarkdownTokens.svelte';

	let { text = '', done = false }: { text: string; done: boolean } = $props();

	let tokens = $state<Tokens.Generic[]>([]);
	let frame: number | null = null;
	let lastParsed = '';

	function parse() {
		// marked's lexer already treats EOF as closing a ``` fence, so fences
		// need no help from remend. remend closes unterminated **, `, ~~, [](, $$.
		const healed = done ? text : remend(text);
		if (healed === lastParsed) return;
		lastParsed = healed;
		tokens = md.lexer(healed);
	}

	$effect(() => {
		text;
		done;
		if (done) {
			if (frame !== null) cancelAnimationFrame(frame);
			frame = null;
			parse();
			return;
		}
		if (frame !== null) return; // coalesce: at most one parse per animation frame
		frame = requestAnimationFrame(() => {
			frame = null;
			parse();
		});
	});

	$effect(() => {
		return () => {
			if (frame !== null) cancelAnimationFrame(frame);
		};
	});
</script>

<div class="prose prose-neutral dark:prose-invert max-w-none">
	<MarkdownTokens {tokens} {done} />
</div>
