// Guarded sources-search state for a page-level search surface (AUDIT_0824
// F3). SearchPage and ArchiveBrowsePage both ran the identical unguarded
// fetch-then-assign: submit query A, edit, submit query B — when A's slower
// response landed after B's it clobbered `result`/`lastQuery` with stale
// answers for a query the user had already moved past.
//
// One monotonic sequence token per run: only the newest started run (and
// nothing invalidated since) may assign state. `clear()` invalidates without
// starting anything, so a response racing home after "New question" or an
// external re-seed lands on a blank surface instead of repopulating it.
import { search as runSearch, type SearchResponse } from '$lib/api/search';

export class PageSearch {
	result = $state<SearchResponse | null>(null);
	loading = $state(false);
	error = $state<string | null>(null);
	lastQuery = $state('');

	private seq = 0;

	async run(q: string, scope?: string): Promise<void> {
		const token = ++this.seq;
		this.loading = true;
		this.error = null;
		try {
			const res = await runSearch(q, scope);
			// A newer run started while we were in flight: drop everything,
			// including the `finally` — the newer run owns `loading` now.
			if (token !== this.seq) return;
			this.result = res;
			this.lastQuery = q;
		} catch (err) {
			if (token !== this.seq) return;
			this.error = err instanceof Error ? err.message : 'search failed';
			this.result = null;
		} finally {
			if (token === this.seq) this.loading = false;
		}
	}

	/** Blank the surface AND invalidate any in-flight run without starting a
	 *  new one — its late response must not repaint what we just cleared. */
	clear(): void {
		this.seq++;
		this.result = null;
		this.lastQuery = '';
		this.error = null;
		this.loading = false;
	}
}
