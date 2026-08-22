// Background-scroll lock for the overlays this app hand-rolls (the Reader
// drawer, the Ask history popover, the add-from-URL and delete-confirm
// dialogs). Without it the page keeps scrolling under the overlay — on a phone
// a swipe over the drawer's chrome moves the page behind it, and closing the
// overlay leaves you somewhere you never navigated to.
//
// bits-ui's Dialog (CommandPalette) brings its own lock; this is only for the
// plain `{#if open}` overlays. Refcounted so two of them open at once (Reader
// over Ask history, say) don't unlock each other on the first close.
//
// `overflow: hidden` on <html> rather than `position: fixed` on <body>: it
// preserves scroll position for free, and every overlay here is a full-height
// fixed element, so the iOS rubber-band case the fixed-body trick exists to
// solve doesn't arise.

let depth = 0;
let restoreOverflow = '';
let restorePadding = '';

/**
 * Locks page scrolling and returns the unlock function — call it from a Svelte
 * `$effect` and return the result, so unmount and `open = false` both unlock:
 *
 * ```svelte
 * $effect(() => (open ? lockBodyScroll() : undefined));
 * ```
 */
export function lockBodyScroll(): () => void {
	if (typeof document === 'undefined') return () => {};

	const html = document.documentElement;
	if (depth === 0) {
		// Removing the scrollbar reflows everything a few px wider; pad the body
		// by exactly its width so the sticky top bar doesn't visibly shift.
		const scrollbarWidth = window.innerWidth - html.clientWidth;
		restoreOverflow = html.style.overflow;
		restorePadding = document.body.style.paddingRight;
		html.style.overflow = 'hidden';
		if (scrollbarWidth > 0) document.body.style.paddingRight = `${scrollbarWidth}px`;
	}
	depth += 1;

	let released = false;
	return () => {
		if (released) return; // an effect re-run must not double-decrement
		released = true;
		depth -= 1;
		if (depth === 0) {
			html.style.overflow = restoreOverflow;
			document.body.style.paddingRight = restorePadding;
		}
	};
}
