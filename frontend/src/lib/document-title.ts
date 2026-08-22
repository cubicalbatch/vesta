// Display-title resolution for a document-library entry (0013). The manifest
// title is optional (null-able on the wire); when absent we fall back to the
// file's basename so a card still reads sensibly instead of showing a bare
// "files/Water (1).pdf" path. Shared by the archive-browse document cards and
// the reader open call so both agree on the fallback.
import type { DocumentOut } from './types';

export function documentDisplayTitle(doc: DocumentOut): string {
	if (doc.title) return doc.title;
	const base = doc.doc_path.split('/').pop();
	return base ? base : doc.doc_path;
}
