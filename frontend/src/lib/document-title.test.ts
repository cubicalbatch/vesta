import { describe, expect, it } from 'vitest';
import { documentDisplayTitle } from './document-title';
import type { DocumentOut } from './types';

function doc(overrides: Partial<DocumentOut>): DocumentOut {
	return {
		doc_path: 'files/Water (1).pdf',
		title: null,
		description: null,
		author: null,
		doc_mime: 'application/pdf',
		url: '/api/zim/19/files/Water (1).pdf',
		...overrides
	};
}

describe('documentDisplayTitle', () => {
	it('prefers the manifest title when present', () => {
		expect(documentDisplayTitle(doc({ title: 'Distillation For Home Water Treatment' }))).toBe(
			'Distillation For Home Water Treatment'
		);
	});

	it('falls back to the file basename when the manifest title is null', () => {
		expect(documentDisplayTitle(doc({ title: null }))).toBe('Water (1).pdf');
	});

	it('falls back to the file basename for an empty-string title', () => {
		expect(documentDisplayTitle(doc({ title: '' }))).toBe('Water (1).pdf');
	});

	it('falls back to the full path when the path has no basename', () => {
		expect(documentDisplayTitle(doc({ doc_path: 'files/', title: null }))).toBe('files/');
	});
});
