// `vesta:use-ai` persistence — default OFF, mirroring lib/stores/theme.svelte.ts
// Toggle persistence. The mode itself is
// component state owned by SearchPage and seeded from the URL (`ai` param
// always wins over the stored value); these are just the read/write helpers.
//
// Storage is written ONLY on a real user toggle, never when seeding from a URL
// — a link someone sent must land in the mode it was written for, and must not
// silently rewrite the recipient's stored preference.
const STORAGE_KEY = 'vesta:use-ai';

export function readUseAi(): boolean {
	if (typeof localStorage === 'undefined') return false;
	return localStorage.getItem(STORAGE_KEY) === '1';
}

export function writeUseAi(on: boolean): void {
	if (typeof localStorage === 'undefined') return;
	localStorage.setItem(STORAGE_KEY, on ? '1' : '0');
}
