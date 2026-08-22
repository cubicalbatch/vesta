import tailwindcss from '@tailwindcss/vite';
import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

// In prod, FastAPI serves the built SPA and the API from the same origin
// (api/spa.py). In dev, proxy the same paths to a locally running backend so
// `fetch('/api/...')` needs no environment-specific base URL anywhere in the
// app. Override the target with VESTA_API_PROXY_TARGET if the backend isn't
// on the default port — which is 5586, the port ./start.sh binds (`make dev`
// passes it explicitly; a bare `npm run dev` relies on this default matching).
const API_PROXY_TARGET = process.env.VESTA_API_PROXY_TARGET ?? 'http://127.0.0.1:5586';

export default defineConfig({
	plugins: [
		tailwindcss(),
		sveltekit({
			compilerOptions: {
				// Force runes mode for the project, except for libraries. Can be removed in svelte 6.
				runes: ({ filename }) =>
					filename.split(/[/\\]/).includes('node_modules') ? undefined : true
			},
			// Static build served by FastAPI (src/vesta/static/app) — no Node at
			// runtime, no SSR server (phased_plan/11-production-frontend.md "Stack").
			// `pages`/`assets` point `npm run build` directly at the Python
			// package's static dir so there is no separate copy step — the
			// backend's `api/spa.py` serves straight from what this writes.
			adapter: adapter({
				pages: '../src/vesta/static/app',
				assets: '../src/vesta/static/app',
				fallback: 'index.html',
				strict: false
			})
		})
	],
	server: {
		// Vite's dev server rejects unrecognized Host headers by default; this
		// machine is also reached via a custom domain, not just localhost.
		allowedHosts: ['loki.onoz.cc'],
		proxy: {
			'/api': { target: API_PROXY_TARGET, changeOrigin: true },
			'/health': { target: API_PROXY_TARGET, changeOrigin: true }
		}
	}
});
