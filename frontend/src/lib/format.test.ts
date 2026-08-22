import { describe, expect, it } from 'vitest';
import { formatZimDate } from './format';

describe('formatZimDate', () => {
	it('renders a full ISO datetime as Mon YYYY in UTC', () => {
		// Midnight UTC must not shift back a month under a negative local tz.
		expect(formatZimDate('2026-07-06T00:00:00Z')).toBe('Jul 2026');
		expect(formatZimDate('2026-01-15T23:59:59Z')).toBe('Jan 2026');
	});

	it('renders a bare year-month', () => {
		expect(formatZimDate('2026-06')).toBe('Jun 2026');
	});

	it('passes a bare four-digit year through', () => {
		expect(formatZimDate('2026')).toBe('2026');
	});

	it('returns an em dash for null / empty / unparseable', () => {
		expect(formatZimDate(null)).toBe('—');
		expect(formatZimDate('')).toBe('—');
		expect(formatZimDate('not a date')).toBe('—');
	});
});
