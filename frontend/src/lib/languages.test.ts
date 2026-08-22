import { describe, expect, it } from 'vitest';
import { formatLanguageNames, languageName } from './languages';

describe('languageName', () => {
	it('maps a known ISO 639-3 code to its English name', () => {
		expect(languageName('eng')).toBe('English');
		expect(languageName('fra')).toBe('French');
		expect(languageName('zho')).toBe('Chinese');
	});

	it('falls back to the raw code for an unmapped language', () => {
		expect(languageName('xxx')).toBe('xxx');
	});
});

describe('formatLanguageNames', () => {
	it('maps a short list to names', () => {
		expect(formatLanguageNames('eng')).toBe('English');
		expect(formatLanguageNames('eng,fra')).toBe('English, French');
	});

	it('caps a long list and counts the remainder', () => {
		expect(formatLanguageNames('eng,fra,deu,spa,ita')).toBe('English, French, German +2');
	});

	it('leaves unmapped codes as their raw code among the names', () => {
		expect(formatLanguageNames('eng,xxx')).toBe('English, xxx');
	});

	it('returns an empty string for an empty list', () => {
		expect(formatLanguageNames('')).toBe('');
		expect(formatLanguageNames(' , ')).toBe('');
	});
});
