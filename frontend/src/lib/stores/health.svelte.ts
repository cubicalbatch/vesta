// GET /health, fetched once at app start. Drives capability gating everywhere
// Capabilities: shape the UI, don't
// hide it. An empty capability set is a healthy state — /health returns 200
// with nothing configured, and every route must still be navigable.
import { api } from '../api/client';

export type Capability =
	| 'zim_fulltext'
	| 'vectors'
	| 'static_encoder'
	| 'cross_encoder'
	| 'llm'
	| 'tool_calling'
	| 'query_reformulation';

export interface HealthResponse {
	status: string;
	capabilities: Partial<Record<Capability, boolean>>;
	capabilities_available: Capability[];
	// VESTA_ADVANCED_MENU gate for the Settings → Advanced tab. Off unless the
	// backend sets it truthy; the UI hides that tab when absent/false.
	advanced_menu?: boolean;
	// First-run wizard bookkeeping. False on a fresh install (so `/` redirects
	// to /welcome); true after POST /api/setup/complete, which stops the
	// zero-archive redirect from ever forcing the user back.
	setup_completed?: boolean;
	[key: string]: unknown;
}

class HealthStore {
	data = $state<HealthResponse | null>(null);
	loaded = $state(false);
	error = $state<string | null>(null);

	has(capability: Capability): boolean {
		return this.data?.capabilities?.[capability] === true;
	}

	async load() {
		try {
			this.data = await api.get<HealthResponse>('/health');
			this.error = null;
		} catch (err) {
			this.error = err instanceof Error ? err.message : 'failed to load /health';
		} finally {
			this.loaded = true;
		}
	}
}

export const healthStore = new HealthStore();
