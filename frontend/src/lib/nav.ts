// Route map, shared by TopBar,
// TabBar and CommandPalette so the nav links never drift apart.
// collapsed Search and Ask into one surface (`/`); "Ask" survives as a *mode*
// of `/` (the `Use AI` toggle), not a destination, so it is no longer a nav item.
// Three nav surfaces: Search (`/`), Catalog (`/catalog`), Settings (`/settings`).
// Jobs + Advanced (eval/benchmark) live as tabs *inside* Settings, gated by
// VESTA_ADVANCED_MENU (surfaced via /health). The AI answer lives behind the
// "Use AI" toggle on Search, not a separate tab.
import Search from '@lucide/svelte/icons/search';
import Library from '@lucide/svelte/icons/library';
import Settings from '@lucide/svelte/icons/settings';
import type { Component } from 'svelte';

export interface NavItem {
	label: string;
	href: string;
	icon: Component;
}

export const NAV_ITEMS: NavItem[] = [
	{ label: 'Search', href: '/', icon: Search },
	{ label: 'Catalog', href: '/catalog', icon: Library },
	{ label: 'Settings', href: '/settings', icon: Settings }
];

export function isActiveRoute(pathname: string, href: string): boolean {
	if (href === '/') return pathname === '/';
	return pathname === href || pathname.startsWith(href + '/');
}
