<script lang="ts">
	import type { Tokens } from './marked';
	import CodeBlock from './CodeBlock.svelte';
	import MarkdownInline from './MarkdownInline.svelte';
	import MarkdownTokens from './MarkdownTokens.svelte';

	let { tokens, done }: { tokens: Tokens.Generic[]; done: boolean } = $props();
</script>

<!--
  KEY BY INDEX, NEVER BY token.raw.
  During append-only streaming, tokens[0..n-2] are byte-identical every tick
  and only tokens[n-1] mutates. Index keys mean Svelte patches exactly one
  subtree and leaves the rest of the DOM — and therefore the user's text
  selection — untouched. Keying by `raw` destroys+recreates the tail token on
  every single chunk.
-->
{#each tokens as token, i (i)}
	{#if token.type === 'paragraph'}
		<p><MarkdownInline tokens={token.tokens ?? []} /></p>
	{:else if token.type === 'heading'}
		<svelte:element this={`h${Math.min(token.depth ?? 1, 6)}`}>
			<MarkdownInline tokens={token.tokens ?? []} />
		</svelte:element>
	{:else if token.type === 'code'}
		<!-- closed gates Shiki: highlight only once this fence can no longer change. -->
		<CodeBlock code={token.text} lang={token.lang} closed={done || i < tokens.length - 1} />
	{:else if token.type === 'list'}
		<svelte:element this={token.ordered ? 'ol' : 'ul'}>
			{#each token.items as item, j (j)}
				<li><MarkdownTokens tokens={item.tokens} {done} /></li>
			{/each}
		</svelte:element>
	{:else if token.type === 'blockquote'}
		<blockquote><MarkdownTokens tokens={token.tokens ?? []} {done} /></blockquote>
	{:else if token.type === 'table'}
		<div class="overflow-x-auto">
			<table>
				<thead>
					<tr>
						{#each token.header as cell, c (c)}
							<th><MarkdownInline tokens={cell.tokens ?? []} /></th>
						{/each}
					</tr>
				</thead>
				<tbody>
					{#each token.rows as row, r (r)}
						<tr>
							{#each row as cell, c (c)}
								<td><MarkdownInline tokens={cell.tokens ?? []} /></td>
							{/each}
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{:else if token.type === 'hr'}
		<hr />
	{:else if token.type === 'html'}
		<!-- LLM answers have no legitimate need for raw HTML — escape, don't render. -->
		<span>{token.raw}</span>
	{:else if token.type === 'space'}
		<!-- nothing -->
	{:else if token.type === 'text'}
		<p><MarkdownInline tokens={'tokens' in token && token.tokens ? token.tokens : [token]} /></p>
	{/if}
{/each}
