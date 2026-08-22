import { api } from './client';

export interface ModelPreset {
	id: string;
	display_name: string;
	url: string;
	filename: string;
	size_bytes: number;
	min_ram_gb: number;
	description: string;
	// True when the GGUF already exists under data/models/ — the wizard shows a
	// "Downloaded" check instead of a Download button. Mirrors ModelPresetOut.
	downloaded: boolean;
}

export interface ModelDownloadResponse {
	job_id: number;
	job_type: string;
	target: string | null;
	model_filename: string;
}

// LlmStatus on the wire (src/vesta/api/models.py LlmStatusOut, faithful field
// for field). `state` is the lifecycle the whole UI keys off:
// absent|stopped|unloaded|sleeping|loading|loaded|error.
export interface LlmStatus {
	source: string;
	configured: boolean;
	installed: boolean;
	state:
		| 'absent'
		| 'stopped'
		| 'unloaded'
		| 'sleeping'
		| 'loading'
		| 'loaded'
		| 'error'
		| (string & {});
	model_file: string | null;
	display_name: string | null;
	model_id: string | null;
	size_bytes: number;
	context_size: number;
	thinking: boolean;
	thinking_supported: boolean;
	idle_unload_seconds: number;
	seconds_since_last_use: number | null;
	estimated_ram_bytes: number;
	error: string | null;
}

export interface InstalledModel {
	filename: string;
	size_bytes: number;
	display_name: string;
	is_active: boolean;
	preset_id: string | null;
	thinking_supported: boolean;
}

export interface ModelsResponse {
	installed: InstalledModel[];
	presets: ModelPreset[];
	status: LlmStatus;
}

export const modelsApi = {
	presets: () => api.get<{ presets: ModelPreset[] }>('/api/models/presets'),
	download: (body: { preset_id?: string; url?: string; filename?: string }) =>
		api.post<ModelDownloadResponse>('/api/models/download', body),
	// The D10 surface — Settings → AI and the ModelChip both ride these.
	list: () => api.get<ModelsResponse>('/api/models'),
	status: () => api.get<LlmStatus>('/api/models/status'),
	activate: (filename: string) =>
		api.post<LlmStatus>('/api/models/activate', { filename }),
	load: () => api.post<LlmStatus>('/api/models/load'),
	unload: () => api.post<LlmStatus>('/api/models/unload'),
	remove: (filename: string) => api.delete<{ deleted: string }>(`/api/models/${encodeURIComponent(filename)}`)
};
