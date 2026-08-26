import { describe, expect, it } from 'vitest';
import { humanizeName, flavourFromKey } from './curated-helpers';

describe('flavourFromKey', () => {
	it('extracts known flavours from end of key', () => {
		expect(flavourFromKey('wikipedia_en_top_nopic')).toBe('nopic');
		expect(flavourFromKey('mdwiki_en_all_maxi')).toBe('maxi');
		expect(flavourFromKey('wikivoyage_en_all_mini')).toBe('mini');
		expect(flavourFromKey('gardening.stackexchange.com_en_all')).toBe('all');
		expect(flavourFromKey('nhs.uk_en_medicines')).toBe('medicines');
	});

	it('returns empty string when key has no recognised flavour', () => {
		expect(flavourFromKey('wikipedia_en_100')).toBe('');
		expect(flavourFromKey('unknown_key_foo')).toBe('');
		expect(flavourFromKey('')).toBe('');
	});
});

describe('humanizeName', () => {
	it('formats known archive names and strips flavours and lang codes', () => {
		expect(humanizeName('wikipedia_en_100')).toBe('Wikipedia 100');
		expect(humanizeName('wikipedia_en_top_nopic')).toBe('Wikipedia Top');
		expect(humanizeName('wikivoyage_en_all_nopic')).toBe('Wikivoyage All');
		expect(humanizeName('mdwiki_en_all_maxi')).toBe('MDWiki All');
		expect(humanizeName('appropedia_en_all_maxi')).toBe('Appropedia All');
	});

	it('capitalizes unknown segments', () => {
		expect(humanizeName('gardening.stackexchange.com_en_all')).toBe('Gardening.stackexchange.com');
	});

	it('handles medicines flavour trimming', () => {
		expect(humanizeName('nhs.uk_en_medicines')).toBe('Nhs.uk');
	});
});
