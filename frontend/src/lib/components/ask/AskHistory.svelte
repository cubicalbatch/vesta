<script lang="ts">
	// GET /api/conversations behind a History button
	// "Page: Ask" → "History"). Each row: derived title, updated_at. Delete per row.
	import { conversationsApi } from '$lib/api/conversations';
	import type { ConversationSummary } from '$lib/types';
	import { goto } from '$app/navigation';
	import { lockBodyScroll } from '$lib/scroll-lock';
	import Trash2 from '@lucide/svelte/icons/trash-2';

	let {
		open = $bindable(false),
		onDeleted
	}: { open: boolean; onDeleted?: (id: number) => void } = $props();

	let items = $state<ConversationSummary[]>([]);
	let loading = $state(false);
	let error = $state<string | null>(null);

	$effect(() => (open ? lockBodyScroll() : undefined));

	$effect(() => {
		if (!open) return;
		loading = true;
		error = null;
		conversationsApi
			.list(50)
			.then((res) => (items = res))
			.catch((err) => (error = err instanceof Error ? err.message : 'failed to load history'))
			.finally(() => (loading = false));
	});

	async function remove(id: number, e: MouseEvent) {
		e.stopPropagation();
		try {
			await conversationsApi.remove(id);
			items = items.filter((c) => c.id !== id);
			// The orchestrator must drop its live session if THIS conversation
			// is the one on screen — otherwise every follow-up 404s on the
			// deleted id.
			onDeleted?.(id);
		} catch {
			// leave the row — the user can retry
		}
	}

	function openConversation(id: number) {
		open = false;
		// Ask is now a mode of `/`; restore the thread on the unified surface
		// (History overlay).
		goto(`/?c=${id}&ai=1`);
	}
</script>

{#if open}
	<div class="fixed inset-0 z-[95] bg-black/30" role="presentation" onclick={() => (open = false)}></div>
	<div class="fixed right-4 top-[calc(var(--topbar-h)+0.5rem)] z-[96] max-h-[70vh] w-[min(380px,92vw)] overflow-y-auto rounded-xl border border-border bg-surface shadow-pop">
		<div class="border-b border-border px-4 py-2 text-xs font-semibold uppercase tracking-wide text-faint">Past questions</div>
		{#if loading}
			<div class="p-4 text-sm text-muted">Loading…</div>
		{:else if error}
			<div class="p-4 text-sm text-danger">{error}</div>
		{:else if items.length === 0}
			<div class="p-4 text-sm text-muted">No conversations yet.</div>
		{:else}
			{#each items as conv (conv.id)}
				<div
					role="button"
					tabindex="0"
					class="flex w-full items-center gap-2 border-b border-border px-4 py-3 text-left last:border-0 hover:bg-surface-muted"
					onclick={() => openConversation(conv.id)}
					onkeydown={(e) => e.key === 'Enter' && openConversation(conv.id)}
				>
					<div class="min-w-0 flex-1">
						<div class="truncate text-sm text-ink">{conv.title ?? `Conversation ${conv.id}`}</div>
						{#if conv.updated_at}<div class="text-xs text-faint">{new Date(conv.updated_at).toLocaleString()}</div>{/if}
					</div>
					<button
						type="button"
						class="inline-grid size-7 shrink-0 place-items-center rounded-md text-faint hover:bg-danger-soft hover:text-danger"
						onclick={(e) => remove(conv.id, e)}
						title="Delete"
					>
						<Trash2 class="size-3.5" />
					</button>
				</div>
			{/each}
		{/if}
	</div>
{/if}
