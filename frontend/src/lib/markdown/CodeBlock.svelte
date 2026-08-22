<script lang="ts">
	// Plain <pre><code> while the fence is still open; Shiki only once closed.
	// Highlighting a growing code block on every rAF tick is the single most
	// expensive thing a streaming markdown renderer can do
	// (research/frontend-stack.md "Syntax highlighting").
	let { code, lang, closed }: { code: string; lang?: string; closed: boolean } = $props();

	let html = $state<string | null>(null);

	const SUPPORTED = new Set(['python', 'ts', 'js', 'bash', 'json', 'sql', 'yaml', 'diff']);

	$effect(() => {
		if (!closed) {
			html = null;
			return;
		}
		const language = lang && SUPPORTED.has(lang) ? lang : 'text';
		let cancelled = false;
		import('shiki').then(async ({ codeToHtml }) => {
			const rendered = await codeToHtml(code, {
				lang: language,
				themes: { light: 'github-light', dark: 'github-dark' }
			});
			if (!cancelled) html = rendered;
		});
		return () => {
			cancelled = true;
		};
	});
</script>

{#if html}
	<div class="not-prose overflow-x-auto rounded-lg border border-border text-sm [&_pre]:p-4">
		{@html html}
	</div>
{:else}
	<pre class="not-prose overflow-x-auto rounded-lg border border-border bg-surface-muted p-4 text-sm"><code
		>{code}</code
		></pre>
{/if}
