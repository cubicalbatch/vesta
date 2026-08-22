import { api } from './client';

export interface StorageInfo {
	data_dir: string;
	total_bytes: number;
	free_bytes: number;
	used_by_zims_bytes: number;
}

export interface HardwareInfo {
	ram_total_bytes: number;
	cpu_count: number;
}

// GET /api/system/* — storage for the disk meter, hardware for model
// recommendation. If rejected/absent, callers degrade gracefully — never a
// hard dependency.
export const systemApi = {
	storage: () => api.get<StorageInfo>('/api/system/storage'),
	hardware: () => api.get<HardwareInfo>('/api/system/hardware')
};
