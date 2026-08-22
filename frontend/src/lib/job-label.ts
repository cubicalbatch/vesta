// Resolve a JobRecord's type+target to a human name, e.g. `index_zim` +
// `target: "3"` -> "Indexing — Project Gutenberg"
// (job labels). Target semantics differ per job type — verified
// against the actual submit() call sites, not guessed:
//   index_zim       target = str(zim_id)         (src/vesta/api/zims.py)
//   download_zim     target = archive name string (src/vesta/api/library.py)
//   refresh_catalog target = "catalog"           (src/vesta/api/library.py)
import type { Archive, JobRecord } from './types';

export function jobLabel(job: JobRecord, archives: Archive[]): string {
	switch (job.type) {
		case 'index_zim': {
			const zimId = Number(job.target);
			const archive = archives.find((a) => a.id === zimId);
			const name = archive ? (archive.corpus_label ?? archive.title ?? archive.name ?? `archive ${zimId}`) : `archive ${job.target}`;
			return `Indexing — ${name}`;
		}
		case 'download_zim':
			return `Downloading — ${job.target || 'archive'}`;
		case 'refresh_catalog':
			return 'Refreshing catalog';
		case 'noop':
			return 'No-op job';
		default:
			// A registered job type this client doesn't know about yet must not
			// render blank — fall through to the raw type.
			return `${job.type}${job.target ? ` — ${job.target}` : ''}`;
	}
}
