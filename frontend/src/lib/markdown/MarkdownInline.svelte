<script lang="ts">
	import type { Tokens } from './marked';
	import type { CitationToken } from './marked';
	import CitationChip from './CitationChip.svelte';
	import MarkdownInline from './MarkdownInline.svelte';

	let { tokens }: { tokens: (Tokens.Generic | CitationToken)[] } = $props();
</script>

<!-- Keyed by index — see MarkdownTokens.svelte for why. -->
{#each tokens as token, i (i)}
	{#if token.type === 'text'}
		{#if 'tokens' in token && token.tokens}
			<MarkdownInline tokens={token.tokens} />
		{:else}
			{token.raw}
		{/if}
	{:else if token.type === 'citation'}
		<CitationChip ids={(token as CitationToken).ids} />
	{:else if token.type === 'strong'}
		<strong><MarkdownInline tokens={token.tokens ?? []} /></strong>
	{:else if token.type === 'em'}
		<em><MarkdownInline tokens={token.tokens ?? []} /></em>
	{:else if token.type === 'del'}
		<del><MarkdownInline tokens={token.tokens ?? []} /></del>
	{:else if token.type === 'codespan'}
		<code>{token.text}</code>
	{:else if token.type === 'link'}
		<a href={token.href} title={token.title ?? undefined} target="_blank" rel="noreferrer">
			<MarkdownInline tokens={token.tokens ?? []} />
		</a>
	{:else if token.type === 'br'}
		<br />
	{:else if token.type === 'html'}
		<!-- LLM answers have no legitimate need for raw HTML — escape, don't render. -->
		<span>{token.raw}</span>
	{:else if token.type === 'escape'}
		{token.text}
	{:else}
		{token.raw ?? ''}
	{/if}
{/each}
