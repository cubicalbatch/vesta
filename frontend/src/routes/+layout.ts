// SPA mode: no server-rendering, no prerendering. adapter-static's `fallback:
// index.html` plus this combination is what makes a hard refresh on a client
// route (e.g. /catalog) work with zero Node at runtime.
export const ssr = false;
export const prerender = false;
