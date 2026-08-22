<script lang="ts">
	// Renders one AnswerSession: prior turns collapsed, the live turn expanded,
	// and the loading-history state. Presentational — it owns no state beyond
	// reading the session. SearchHero unmounts once a conversation is live
	// (shared follow-up box didn't hold up
	// in practice), so AskTurn's own follow-up form is the only way to
	// continue a thread.
	import type { AnswerSession } from '$lib/answer/session.svelte';
	import AskTurn from '$lib/components/ask/AskTurn.svelte';

	let {
		session,
		onRegenerate,
		onFollowUp
	}: {
		session: AnswerSession;
		onRegenerate: () => void;
		onFollowUp: (q: string) => void;
	} = $props();
</script>

{#if session.loadingHistory}
	<div class="mx-auto max-w-[var(--wide-max)] text-sm text-muted">Loading conversation…</div>
{:else}
	<div class="flex flex-col gap-10">
		{#each session.priorTurns as turn, i (i)}
			<AskTurn query={turn.query} answerState={turn.state} collapsedByDefault />
		{/each}
		{#if session.liveTurn}
			<AskTurn
				query={session.liveTurn.query}
				answerState={session.liveTurn.stream.state}
				live
				{onRegenerate}
				{onFollowUp}
			/>
		{/if}
	</div>
{/if}
