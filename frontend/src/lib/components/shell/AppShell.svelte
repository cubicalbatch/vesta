<script lang="ts">
	import TopBar from './TopBar.svelte';
	import TabBar from './TabBar.svelte';
	import StatusBar from './StatusBar.svelte';
	import CommandPalette from './CommandPalette.svelte';
	import Reader from '../reader/Reader.svelte';
	import { readerStore } from '$lib/stores/reader.svelte';

	let { children } = $props();

	let paletteOpen = $state(false);

	function onKeydown(e: KeyboardEvent) {
		const target = e.target as HTMLElement | null;
		const inField =
			!!target &&
			(target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable);

		if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
			e.preventDefault();
			paletteOpen = !paletteOpen;
			return;
		}
		if (e.key === '/' && !inField && !paletteOpen) {
			e.preventDefault();
			window.dispatchEvent(new CustomEvent('vesta:focus-search'));
			return;
		}
		if (e.key === 'Escape') {
			if (readerStore.isOpen) readerStore.close();
		}
	}
</script>

<svelte:window onkeydown={onKeydown} />

<TopBar onOpenPalette={() => (paletteOpen = true)} />

<main
	class="px-5 pb-10 pt-7 max-[720px]:px-4 max-[720px]:pt-5 max-[720px]:pb-[calc(var(--tabbar-h)+env(safe-area-inset-bottom,0px)+1.25rem)]"
>
	{@render children()}
	<div class="mx-auto max-w-[var(--wide-max)]">
		<StatusBar />
	</div>
</main>

<TabBar />

<CommandPalette bind:open={paletteOpen} />
<Reader />
