// Vite `?url` imports (e.g. the bundled PDF.js worker asset) are typed by
// `vite/client`, which this project does not reference — declare the shape the
// compiler needs so `npm run check` stays clean.
declare module '*?url' {
	const src: string;
	export default src;
}
