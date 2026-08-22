// BASIC-allowlist test ("The
// BASIC allowlist contains no key absent from the live schema (tested).").
// Imports the *actual* BASIC array the Settings page renders from — not a
// hardcoded re-typed copy — and checks it against a live
// GET /api/settings/schema, so drift between this file and the real backend
// schema is what gets caught, not drift between two hand-maintained lists.
//
// Requires a running dev backend.
// `npx vitest run` must stay green
// with no backend running too, so this probes reachability first and skips
// (not fails) the live assertion when nothing answers — CI-without-a-backend
// is a normal context for this suite, not an error state.
import { describe, expect, it } from 'vitest';
import { BASIC, CONTEXT_KEYS, THOROUGH_KEYS } from './settings-basic';

const BACKEND = process.env.VESTA_TEST_BACKEND ?? 'http://127.0.0.1:5586';

async function backendReachable(): Promise<boolean> {
	try {
		const ctrl = new AbortController();
		const timer = setTimeout(() => ctrl.abort(), 2000);
		const res = await fetch(`${BACKEND}/health`, { signal: ctrl.signal });
		clearTimeout(timer);
		return res.ok;
	} catch {
		return false;
	}
}

const reachable = await backendReachable();

if (!reachable) {
	// eslint-disable-next-line no-console
	console.warn(
		`[settings-basic.test] backend not reachable at ${BACKEND} — skipping the live BASIC-allowlist check ` +
			`(set VESTA_TEST_BACKEND to point elsewhere, or start ./start.sh)`
	);
}

describe.skipIf(!reachable)('BASIC allowlist vs. live GET /api/settings/schema', () => {
	it('every BASIC key exists in the live schema', async () => {
		const res = await fetch(`${BACKEND}/api/settings/schema`);
		expect(res.ok).toBe(true);
		const data = (await res.json()) as { settings: { key: string }[] };
		const liveKeys = new Set(data.settings.map((s) => s.key));

		const missing = BASIC.filter((k) => !liveKeys.has(k));
		expect(missing).toEqual([]);
	});

	it('the "How thorough" composite\'s backing keys also exist live', async () => {
		const res = await fetch(`${BACKEND}/api/settings/schema`);
		const data = (await res.json()) as { settings: { key: string }[] };
		const liveKeys = new Set(data.settings.map((s) => s.key));

		const missing = THOROUGH_KEYS.filter((k) => !liveKeys.has(k));
		expect(missing).toEqual([]);
	});

	it('the "Answer speed & memory" composite\'s backing keys also exist live', async () => {
		const res = await fetch(`${BACKEND}/api/settings/schema`);
		const data = (await res.json()) as { settings: { key: string }[] };
		const liveKeys = new Set(data.settings.map((s) => s.key));

		const missing = CONTEXT_KEYS.filter((k) => !liveKeys.has(k));
		expect(missing).toEqual([]);
	});
});
