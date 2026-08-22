<script lang="ts">
	import { page } from '$app/state';
	import { NAV_ITEMS, isActiveRoute } from '$lib/nav';
	const cols = `repeat(${NAV_ITEMS.length}, minmax(0, 1fr))`;
</script>

<nav aria-label="Mobile primary" class="fixed inset-x-0 bottom-0 z-50 hidden max-[720px]:block">
	<!-- The safe-area inset belongs on the element that carries the background,
	     not the transparent wrapper — otherwise page content scrolls visibly
	     through the strip under an iPhone's home indicator. -->
	<div
		class="grid border-t border-border bg-surface/92 backdrop-blur-md"
		style="grid-template-columns: {cols}; padding-bottom: env(safe-area-inset-bottom, 0px);"
	>
		{#each NAV_ITEMS as item (item.href)}
			{@const active = isActiveRoute(page.url.pathname, item.href)}
			{@const Icon = item.icon}
			<a
				href={item.href}
				class="flex min-w-0 flex-col items-center gap-0.5 px-1 py-2 text-xs font-medium max-[360px]:text-[10px] {active
					? 'text-accent'
					: 'text-faint'}"
				aria-current={active ? 'page' : undefined}
			>
				<Icon class="size-[22px] shrink-0" />
				<span class="max-w-full truncate">{item.label}</span>
			</a>
		{/each}
	</div>
</nav>
