// Light-first, warm terracotta (Clay Hearth) accent, full dark theme
// Theme. Persists to localStorage;
// honours prefers-color-scheme only when the user has never chosen explicitly.
// `appearance.theme` isn't a backend setting today — this is deliberately
// client-only state.
const STORAGE_KEY = 'vesta:theme';

type Theme = 'light' | 'dark';

function systemPrefersDark(): boolean {
	return typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches;
}

function readStored(): Theme | null {
	if (typeof localStorage === 'undefined') return null;
	const value = localStorage.getItem(STORAGE_KEY);
	return value === 'light' || value === 'dark' ? value : null;
}

class ThemeStore {
	current = $state<Theme>(readStored() ?? (systemPrefersDark() ? 'dark' : 'light'));

	constructor() {
		$effect.root(() => {
			$effect(() => {
				if (typeof document === 'undefined') return;
				document.documentElement.classList.toggle('dark', this.current === 'dark');
			});
		});
	}

	toggle() {
		const theme = this.current === 'dark' ? 'light' : 'dark';
		this.current = theme;
		localStorage.setItem(STORAGE_KEY, theme);
	}
}

export const themeStore = new ThemeStore();
