<script lang="ts">
	import { Dialog, Command } from 'bits-ui';
	import { goto } from '$app/navigation';
	import { NAV_ITEMS } from '$lib/nav';
	import { themeStore } from '$lib/stores/theme.svelte';
	import { zimsStore } from '$lib/stores/zims.svelte';

	let { open = $bindable(false) }: { open: boolean } = $props();

	function navigate(href: string) {
		open = false;
		goto(href);
	}
</script>

<Dialog.Root bind:open>
	<Dialog.Portal>
		<Dialog.Overlay class="fixed inset-0 z-[100] bg-black/40" />
		<Dialog.Content
			class="fixed left-1/2 top-[18vh] z-[101] w-[min(560px,92vw)] -translate-x-1/2 overflow-hidden rounded-xl border border-border bg-surface shadow-pop"
		>
			<Command.Root class="flex flex-col">
				<Command.Input
					placeholder="Search archives, jump to a page…"
					class="w-full border-b border-border bg-transparent px-4 py-3 text-sm outline-none placeholder:text-faint"
				/>
				<Command.List class="max-h-[50vh] overflow-y-auto p-2">
					<Command.Empty class="px-3 py-6 text-center text-sm text-muted">No matches.</Command.Empty>

					<Command.Group>
						<Command.GroupHeading class="px-2 py-1.5 text-xs font-medium text-faint">Go to</Command.GroupHeading>
						<Command.GroupItems>
						{#each NAV_ITEMS as item (item.href)}
							{@const Icon = item.icon}
							<Command.Item
								class="flex cursor-pointer items-center gap-2 rounded-md px-2 py-2 text-sm text-ink data-[selected]:bg-accent-soft data-[selected]:text-accent-soft-text"
								onSelect={() => navigate(item.href)}
							>
								<Icon class="size-4" />
								{item.label}
							</Command.Item>
						{/each}
							<Command.Item
								class="flex cursor-pointer items-center gap-2 rounded-md px-2 py-2 text-sm text-ink data-[selected]:bg-accent-soft data-[selected]:text-accent-soft-text"
								onSelect={() => {
									themeStore.toggle();
									open = false;
								}}
							>
								Toggle light / dark
							</Command.Item>
						</Command.GroupItems>
					</Command.Group>

					{#if zimsStore.enabled.length > 0}
						<Command.Group>
							<Command.GroupHeading class="px-2 py-1.5 text-xs font-medium text-faint">Archives</Command.GroupHeading>
							<Command.GroupItems>
								{#each zimsStore.enabled as archive (archive.id)}
								<Command.Item
									class="flex cursor-pointer items-center justify-between rounded-md px-2 py-2 text-sm text-ink data-[selected]:bg-accent-soft data-[selected]:text-accent-soft-text"
									onSelect={() => navigate(`/archive/${archive.id}`)}
								>
									<span>{archive.corpus_label ?? archive.title ?? archive.name}</span>
								</Command.Item>
								{/each}
							</Command.GroupItems>
						</Command.Group>
					{/if}
				</Command.List>
			</Command.Root>
		</Dialog.Content>
	</Dialog.Portal>
</Dialog.Root>
