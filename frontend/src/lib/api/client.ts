// Thin typed JSON fetch wrapper shared by every page. Not a cache/dedup layer
// (research/frontend-stack.md rules out TanStack Query here: one user hitting
// localhost has no contention problem to solve) — just a place to put the one
// error-shape decision consistently: FastAPI's 4xx/5xx bodies carry
// `{"detail": "..."}` and every page shows that message inline, never a toast
// that scrolls away.

export class ApiError extends Error {
	status: number;
	detail: string;

	constructor(status: number, detail: string) {
		super(detail);
		this.status = status;
		this.detail = detail;
	}
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
	let response: Response;
	try {
		response = await fetch(path, {
			...init,
			headers: {
				...(init?.body ? { 'Content-Type': 'application/json' } : {}),
				...init?.headers
			}
		});
	} catch (err) {
		throw new ApiError(0, err instanceof Error ? err.message : 'network request failed');
	}

	if (!response.ok) {
		let detail = `HTTP ${response.status}`;
		try {
			const body = await response.json();
			if (typeof body?.detail === 'string') detail = body.detail;
		} catch {
			// non-JSON error body — keep the generic HTTP status message
		}
		throw new ApiError(response.status, detail);
	}

	if (response.status === 204) return undefined as T;
	return (await response.json()) as T;
}

export const api = {
	get: <T>(path: string) => request<T>(path),
	post: <T>(path: string, body?: unknown) =>
		request<T>(path, { method: 'POST', body: body !== undefined ? JSON.stringify(body) : undefined }),
	put: <T>(path: string, body?: unknown) =>
		request<T>(path, { method: 'PUT', body: body !== undefined ? JSON.stringify(body) : undefined }),
	patch: <T>(path: string, body?: unknown) =>
		request<T>(path, { method: 'PATCH', body: body !== undefined ? JSON.stringify(body) : undefined }),
	delete: <T>(path: string) => request<T>(path, { method: 'DELETE' })
};
