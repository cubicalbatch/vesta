// GET /api/zims — installed archives. Backs the status bar, Search's scope
// chips, and Catalog's "On this machine" list. Loaded once at app start and
// refetched by whoever mutates it (download/index/delete flows).
import { api } from '../api/client';
import type { Archive } from '../types';

class ZimsStore {
	archives = $state<Archive[]>([]);
	loaded = $state(false);
	error = $state<string | null>(null);

	get enabled(): Archive[] {
		return this.archives.filter((a) => a.enabled);
	}

	get totalArticles(): number {
		return this.enabled.reduce((sum, a) => sum + a.article_count, 0);
	}

	async load() {
		try {
			const res = await api.get<{ archives: Archive[] }>('/api/zims');
			this.archives = res.archives;
			this.error = null;
		} catch (err) {
			this.error = err instanceof Error ? err.message : 'failed to load /api/zims';
		} finally {
			this.loaded = true;
		}
	}
}

export const zimsStore = new ZimsStore();
