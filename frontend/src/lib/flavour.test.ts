import { describe, expect, it } from 'vitest';
import { flavourDescription } from './flavour';

describe('flavourDescription', () => {
	it('describes the three known Kiwix flavours', () => {
		expect(flavourDescription('maxi')).toBe('full article with images');
		expect(flavourDescription('nopic')).toBe('full text, no images');
		expect(flavourDescription('mini')).toBe('lead section only');
	});

	it('returns null for an unknown flavour', () => {
		expect(flavourDescription('zzz')).toBeNull();
	});
});
