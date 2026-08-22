// Pure URL ⇄ state for the unified search surface
// URL state — one route, four params.
// No Svelte imports — unit-tested in isolation alongside reducer.test.ts.
//
// The route is `/?q=<query>&scope=<csv>&ai=1&c=<conversationId>`:
//   - `ai` present ⇔ AI mode; absent ⇔ sources mode.
//   - `c` is the conversation id (supersedes `q` once a turn has streamed).
//   - `scope` is a comma-separated list of zim ids → Set<number>.
//
// The one non-obvious rule: **AI mode with a `c`
// drops `q`** — once the conversation id is known, the conversation *is* the
// state and `q` is no longer serialized. The hero input keeps showing what the
// user typed (component state, not $derived from the URL),
// but the URL no longer carries it.

export interface SearchModeState {
	query: string;
	/** Scoped zim ids; empty ⇔ "all". */
	scope: Set<number>;
	/** AI mode on (Use AI toggle). */
	ai: boolean;
	/** Active conversation id, or null in sources mode / before the first turn. */
	conversationId: number | null;
}

/** Parse a `q`/`scope`/`ai`/`c` query string into a SearchModeState. */
export function parseSearchMode(search: string): SearchModeState {
	const params = new URLSearchParams(search.startsWith('?') ? search.slice(1) : search);
	const conversationId = parseConversationId(params.get('c'));
	return {
		query: params.get('q') ?? '',
		scope: parseScope(params.get('scope')),
		ai: parseAi(params.get('ai')),
		conversationId
	};
}

/** Serialize a SearchModeState back to a query string (without the leading `?`). */
export function serializeSearchMode(state: SearchModeState): string {
	const params = new URLSearchParams();
	// Once a conversation exists, it supersedes the query in the URL — a reload
	// restores the whole thread from the API, and `q` would only mislead.
	if (state.conversationId == null && state.query) params.set('q', state.query);
	if (state.scope.size > 0) params.set('scope', [...state.scope].join(','));
	if (state.ai) params.set('ai', '1');
	if (state.conversationId != null) params.set('c', String(state.conversationId));
	return params.toString();
}

/** True when `a` and `b` describe the same URL state (order-independent scope). */
export function searchModeEqual(a: SearchModeState, b: SearchModeState): boolean {
	if (a.query !== b.query) return false;
	if (a.ai !== b.ai) return false;
	if (a.conversationId !== b.conversationId) return false;
	if (a.scope.size !== b.scope.size) return false;
	for (const id of a.scope) if (!b.scope.has(id)) return false;
	return true;
}

function parseScope(raw: string | null): Set<number> {
	if (!raw) return new Set();
	const out = new Set<number>();
	for (const part of raw.split(',')) {
		// `Number('')` is 0 (finite), so empty segments (trailing/double commas)
		// must be rejected explicitly — otherwise a stray comma silently injects
		// zim id 0. Non-numeric junk falls out as NaN here too.
		if (part === '') continue;
		const n = Number(part);
		if (Number.isFinite(n)) out.add(n);
	}
	return out;
}

function parseAi(raw: string | null): boolean {
	// `ai=1` ⇔ on. Absent, `ai=0`, or anything else ⇔ off. Only the literal
	// "1" turns it on so a stray value can't silently flip modes.
	return raw === '1';
}

function parseConversationId(raw: string | null): number | null {
	if (raw == null || raw === '') return null;
	const n = Number(raw);
	// Non-numeric garbage (`/ask/abc123` falls back through here) ⇒ null, not
	// NaN and not a bogus id — keeps test_spa.py's fallback assertion honest.
	return Number.isFinite(n) ? n : null;
}
