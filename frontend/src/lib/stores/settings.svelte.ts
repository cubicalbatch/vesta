// GET /api/settings — current values. NOTE: despite PUT requiring every
// value as a string (Settings values are strings on the wire),
// GET returns each value as its *native* JSON type (bool/number/string) —
// verified live against the real backend. `values` is typed as
// Record<string,string> for convenience at existing call sites (they all
// interpolate into template strings or wrap in Number(...), both of which
// coerce fine), but don't assume `typeof values[k] === 'string'` in new code
// — String(values[k]) it first. The Settings page (Task 10) adds schema
// loading on top of this store for the full schema-driven form.
import { settingsApi } from '../api/settings';
import type { SettingSchemaItem } from '../types';

class SettingsValuesStore {
	values = $state<Record<string, string>>({});
	loaded = $state(false);
	schema = $state<SettingSchemaItem[]>([]);
	schemaLoaded = $state(false);
	schemaError = $state<string | null>(null);

	async load() {
		try {
			const res = await settingsApi.values();
			this.values = res.values as Record<string, string>;
		} catch {
			// Status bar degrades to omitting the model/profile fields; not fatal.
		} finally {
			this.loaded = true;
		}
	}

	/** 110 settings across 37 groups — only the Settings page needs this, so
	 * it's loaded lazily rather than at app boot alongside `load()`. */
	async loadSchema() {
		if (this.schemaLoaded) return;
		try {
			const res = await settingsApi.schema();
			this.schema = res.settings;
			this.schemaError = null;
		} catch (err) {
			this.schemaError = err instanceof Error ? err.message : 'failed to load settings schema';
		} finally {
			this.schemaLoaded = true;
		}
	}
}

export const settingsValuesStore = new SettingsValuesStore();
