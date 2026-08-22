<script lang="ts">
	// Welcome's own minimal top bar — wordmark, "Skip for now", theme toggle.
	// No nav links, no job dot, no command palette: "the menu appears once
	// there is something to search"
	// "Page: Welcome"). The full AppShell/TopBar is deliberately not reused
	// here since it renders the five-item nav and job dot this page must not
	// show; +layout.svelte already skips AppShell entirely for this route.
	import Sun from '@lucide/svelte/icons/sun';
	import Moon from '@lucide/svelte/icons/moon';
	import { themeStore } from '$lib/stores/theme.svelte';
</script>

<header class="sticky top-0 z-50 flex h-[var(--topbar-h)] items-center gap-4 border-b border-border bg-surface/88 px-5 backdrop-blur-md">
	<a href="/welcome" class="inline-flex items-center gap-2 text-lg font-bold tracking-tight text-ink">
		<span class="vesta-logo inline-grid size-[26px] place-items-center">
		<!-- V Ember mark — keep byte-identical to TopBar.svelte and favicon.svg. -->
		<svg viewBox="0 0 24 24" fill="none" class="size-[26px]" aria-hidden="true">
			<g transform="translate(12 12.3) scale(0.9) translate(-12 -11.125)">
				<path d="M4 5 L9.5 15.5 Q12 19 14.5 15.5 L20 5" stroke="currentColor" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" />
				<path d="M8 12.8 C6.5 11.2 6.9 5.5 7.6 5.5 C8.4 5.5 9.2 8.6 10.2 8.6 C11.2 8.6 12.1 2.5 12.8 2.5 C13.5 2.5 13.8 8 14.6 8 C15.4 8 15.1 5 15.8 5 C16.6 5 17.3 11 16 12.8 A4 4 0 0 1 8 12.8 Z" fill="currentColor" transform="translate(12 4.6) scale(0.62) translate(-12 -2.5)" />
			</g>
		</svg>
		</span>
		Vesta
	</a>

	<div class="flex-1"></div>

	<div class="flex items-center gap-2">
		<a href="/" class="rounded-md border border-border px-3 py-1.5 text-sm font-medium text-ink-2 hover:bg-surface-muted">Skip for now</a>
		<button
			type="button"
			class="inline-grid size-9 place-items-center rounded-md text-muted transition-colors hover:bg-surface-muted hover:text-ink"
			onclick={() => themeStore.toggle()}
			title="Toggle light / dark"
		>
			{#if themeStore.current === 'dark'}
				<Moon class="size-[18px]" />
			{:else}
				<Sun class="size-[18px]" />
			{/if}
		</button>
	</div>
</header>
