import { describe, expect, it } from 'vitest';
import { md } from './marked';
import { isSafeLinkHref } from './link';

describe('isSafeLinkHref', () => {
	it('allows http(s) and mailto', () => {
		expect(isSafeLinkHref('http://example.com/a?b=c#d')).toBe(true);
		expect(isSafeLinkHref('https://example.com')).toBe(true);
		expect(isSafeLinkHref('mailto:user@example.com')).toBe(true);
	});

	it('allows relative URLs: paths, root-relative, fragments, queries', () => {
		expect(isSafeLinkHref('articles/foo.html')).toBe(true);
		expect(isSafeLinkHref('/catalog?page=2')).toBe(true);
		expect(isSafeLinkHref('#section-3')).toBe(true);
		expect(isSafeLinkHref('?q=test')).toBe(true);
		expect(isSafeLinkHref('../up/one')).toBe(true);
	});

	it('blocks hostile schemes', () => {
		for (const href of [
			'javascript:alert(1)',
			'data:text/html,<script>alert(1)</script>',
			'vbscript:msgbox(1)',
			'file:///etc/passwd',
			'blob:https://example.com/x',
			'tel:+1234567890',
			'unknown-scheme:x'
		]) {
			expect(isSafeLinkHref(href), href).toBe(false);
		}
	});

	it('blocks scheme casing games', () => {
		expect(isSafeLinkHref('JaVaScRiPt:alert(1)')).toBe(false);
		expect(isSafeLinkHref('JAVASCRIPT:alert(1)')).toBe(false);
		expect(isSafeLinkHref('DATA:text/html,x')).toBe(false);
	});

	it('blocks whitespace/control-char obfuscation', () => {
		expect(isSafeLinkHref('java\tscript:alert(1)')).toBe(false);
		expect(isSafeLinkHref('java\nscript:alert(1)')).toBe(false);
		expect(isSafeLinkHref('java\rscript:alert(1)')).toBe(false);
		expect(isSafeLinkHref('\x00javascript:alert(1)')).toBe(false);
		expect(isSafeLinkHref('java\x00script:alert(1)')).toBe(false);
		expect(isSafeLinkHref('javascript\x01:alert(1)')).toBe(false);
		expect(isSafeLinkHref('\x7fjavascript:alert(1)')).toBe(false);
	});

	it('strips surrounding whitespace before judging the scheme', () => {
		expect(isSafeLinkHref('  javascript:alert(1)  ')).toBe(false);
		expect(isSafeLinkHref('\n\t https://example.com \t\n')).toBe(true);
	});

	it('still allows http(s)/mailto written with odd casing or padding', () => {
		expect(isSafeLinkHref('HTTPS://EXAMPLE.COM/A')).toBe(true);
		expect(isSafeLinkHref('MailTo:user@example.com')).toBe(true);
		expect(isSafeLinkHref('  /relative/path ')).toBe(true);
	});

	it('refuses scheme-shaped values the URL parser rejects', () => {
		expect(isSafeLinkHref('http://example.com:99999999999999')).toBe(false);
	});
});

describe('isSafeLinkHref against real marked link tokens', () => {
	function hrefOf(markdownSrc: string): string | undefined {
		const [block] = md.lexer(markdownSrc);
		const paragraph = block.type === 'paragraph' ? block : undefined;
		const inline = paragraph?.tokens?.find((t) => t.type === 'link') as
			| { type: 'link'; href?: string }
			| undefined;
		return inline?.href;
	}

	it('passes safe links through untouched', () => {
		const href = hrefOf('[docs](https://example.com/guide?a=1)');
		expect(href).toBeTypeOf('string');
		expect(isSafeLinkHref(href ?? '')).toBe(true);
		const rel = hrefOf('[rel](/catalog#models)');
		expect(rel).toBe('/catalog#models');
		expect(isSafeLinkHref(rel ?? '')).toBe(true);
	});

	it('rejects hostile schemes as they appear in link tokens', () => {
		for (const src of [
			'[click](javascript:fetch("/api/settings"))',
			'[click](JaVaScRiPt:alert(1))',
			'[click](data:text/html;base64,PHNjcmlwdD4=)',
			'[click](vbscript:msgbox)'
		]) {
			const href = hrefOf(src);
			expect(href, src).toBeTypeOf('string');
			expect(isSafeLinkHref(href ?? ''), `${src} -> ${href}`).toBe(false);
		}
	});
});
