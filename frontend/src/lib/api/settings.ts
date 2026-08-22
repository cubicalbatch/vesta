import { api } from './client';
import type { SettingSchemaItem } from '../types';

export const settingsApi = {
	schema: () => api.get<{ settings: SettingSchemaItem[] }>('/api/settings/schema'),
	values: () => api.get<{ values: Record<string, unknown> }>('/api/settings'),
	// Body is {key: value-as-string} both directions — PUT /api/settings
	// coerces + validates server-side and 400s (with a `setting '<key>': ...`
	// message) on the first bad key, so the whole batch fails together.
	save: (values: Record<string, string>) =>
		api.put<{ values: Record<string, unknown>; applied: string[] }>('/api/settings', { values })
};
