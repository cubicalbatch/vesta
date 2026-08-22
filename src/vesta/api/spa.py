"""Production SPA static file server (11-production-frontend.md "How it is served").

`npm run build` (adapter-static, `frontend/vite.config.ts`) writes the built
SvelteKit app to `src/vesta/static/app/`. This module serves it:

- `GET /` and any unmatched `GET /{path:path}` return `index.html` so
  client-side routing keeps working on a hard refresh of a route like
  `/catalog` or `/ask/abc123` — the browser always gets the app shell, which
  then does its own routing.
- `GET /_app/{path:path}` serves the build's asset directory (SvelteKit's
  default `kit.appDir`) with a year-long immutable cache. Everything under
  `_app/immutable/` is content-hashed — the hash is in the filename, so a
  changed file is a new URL, never a stale hit under that cache header.
  (The phase doc sketches this route as `/assets/{path:path}`; SvelteKit's
  actual default build layout puts hashed assets under `_app/`, not
  `assets/`, and nothing renames that — see the frontend handoff doc for the
  full note.)
- `index.html` itself always gets `Cache-Control: no-store`, since it
  references the current build's hashed asset filenames and must never be
  served stale after a rebuild.

Registered LAST in `api/__init__.py`: a catch-all `GET /{path:path}` must
never get a chance to shadow `/api/*`, `/health` or `/dev`. FastAPI/Starlette
matches routes in registration order, so this router being added after every
other router is what guarantees a real route always wins first — see
`tests/test_spa.py` for the assertion.

Path-traversal guard mirrors an earlier ``_guard``
resolve-and-compare shape: resolve the requested path against the static root
and refuse anything that resolves outside it.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(tags=["spa"])

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static" / "app"
_INDEX = "index.html"
_NO_STORE_HEADERS = {"Cache-Control": "no-store"}
_IMMUTABLE_HEADERS = {"Cache-Control": "public, max-age=31536000, immutable"}


def _index_response() -> FileResponse:
    return FileResponse(_STATIC_DIR / _INDEX, headers=_NO_STORE_HEADERS)


def _resolve_static_file(relative_path: str) -> Path | None:
    """Resolve ``relative_path`` against the static root; ``None`` if it
    escapes the root or isn't an existing file. Same resolve-and-compare shape
    as the guard noted above."""
    static_root = _STATIC_DIR.resolve()
    candidate = (static_root / relative_path).resolve()
    if static_root not in candidate.parents or not candidate.is_file():
        return None
    return candidate


@router.get("/", include_in_schema=False)
async def spa_index() -> FileResponse:
    return _index_response()


@router.get("/_app/{path:path}", include_in_schema=False)
async def spa_build_asset(path: str) -> FileResponse:
    """The build's content-hashed asset directory. Must be registered before
    the generic catch-all below so a `/_app/...` request doesn't fall
    through to the `index.html` fallback."""
    candidate = _resolve_static_file(f"_app/{path}")
    if candidate is None:
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(candidate, headers=_IMMUTABLE_HEADERS)


@router.get("/{path:path}", include_in_schema=False)
async def spa_catch_all(path: str) -> FileResponse:
    """Any other top-level static file the build emitted (e.g. `robots.txt`)
    is served as-is; every client-side route — anything with no matching
    static file — falls back to `index.html` so a hard refresh renders the
    app shell, which then does its own client-side routing."""
    if not path or path == _INDEX:
        return _index_response()
    candidate = _resolve_static_file(path)
    if candidate is None:
        return _index_response()
    return FileResponse(candidate)


__all__ = ["router"]
