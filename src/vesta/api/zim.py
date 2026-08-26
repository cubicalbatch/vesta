"""``GET /api/zim/{zim_id}/{path:path}`` — path-preserving ZIM passthrough.

Serves articles AND their assets **unmodified** by mirroring the ZIM's internal
path structure. Because the route preserves paths, ZIM-relative links and asset
references resolve with **zero HTML rewriting** — flat Wikipedia uses
bare-relative links (133/133 tested); nested archives use ``../../`` matching
path depth exactly. This is how kiwix-serve works, and it
removes both the link-rewriting work and the XSS/style-collision risk class.

Two load-bearing rules:

* **Serve ``item.mimetype`` from libzim, never infer from the extension.**
  Wikipedia assets named ``.jpg``/``.png`` are frequently actually WebP
  (``RIFF…WEBP``); libzim reports the true type. Correct MIME is what makes
  images render without ``allow-scripts``.
* **Soft (``<meta refresh>``) and hard redirects become a real 302.** A sandboxed
  iframe without ``allow-scripts`` blocks meta refresh; the 302 also keeps the
  browser's base URL correct for relative asset resolution.

No trailing-slash normalisation (research: a spurious 301 shifts the relative
base and breaks every asset). Caching headers reflect that ZIM content is
immutable; Range is honoured for media.
"""

from __future__ import annotations

import urllib.parse

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse

from vesta.api.state import AppState, app_state
from vesta.api.zims import _archive_or_404
from vesta.zim.reader import EntryNotFound
from vesta.zim.types import EntryPath

router = APIRouter(tags=["reader"])

#: ZIM content is immutable once written — cache forever.
_CACHE_HEADERS = {
    "Cache-Control": "public, max-age=31536000, immutable",
    "X-Content-Type-Options": "nosniff",
    # The iframe sandbox (no allow-scripts) is what enforces the security
    # property; CSP here is belt-and-braces for direct fetches.
    "Content-Security-Policy": (
        "default-src 'none'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
        "font-src 'self'; media-src 'self'; frame-ancestors 'self'"
    ),
}


def _url_for(zim_id: int, path: EntryPath) -> str:
    """Build a passthrough URL, percent-encoding path segments but keeping ``/``.

    A ``#fragment`` suffix (a section anchor — modern mwoffliner soft
    redirects target ``./Article#Section``) must stay a URL fragment, not be
    quoted into the path: ``quote('A/Foo#Bar')`` yields ``A/Foo%23Bar``,
    which the browser requests as a literal path and 404s. The fragment is
    percent-encoded separately (Location headers must stay ASCII).
    """
    base, _, fragment = path.partition("#")
    url = f"/api/zim/{zim_id}/{urllib.parse.quote(base, safe='/')}"
    if fragment:
        return f"{url}#{urllib.parse.quote(fragment, safe='/')}"
    return url


def _range_headers(mimetype: str, size: int) -> dict[str, str]:
    """Advertise Range support for media so video/audio ZIMs seek."""
    if mimetype.startswith(("audio/", "video/", "application/octet-stream")):
        return {"Accept-Ranges": "bytes"}
    return {}


@router.get("/api/zim/{zim_id}/{path:path}")
async def serve_zim_entry(
    zim_id: int,
    path: str,
    request: Request,
    state: AppState = Depends(app_state),
) -> Response:
    archive = _archive_or_404(state, zim_id)

    # Bare route (``/api/zim/{id}/``) → the main page, served at a slash-
    # terminated URL so ZIM-relative assets resolve. We resolve the main
    # entry's real path then serve ITS bytes here, keeping the slash base.
    if not path.strip("/"):
        resolved = await archive.main_path()
        raw = await archive.read(resolved)
        return _serve(raw.content, raw.mimetype, request, zim_id=zim_id)

    decoded = urllib.parse.unquote(path)
    try:
        raw = await archive.read(decoded)
    except EntryNotFound as exc:
        raise HTTPException(status_code=404, detail=f"not found: {decoded}") from exc

    if raw.is_redirect and raw.redirect_target:
        # Hard redirect: 302 so the browser's base URL tracks the target.
        return RedirectResponse(url=_url_for(zim_id, raw.redirect_target), status_code=302)
    if raw.soft_redirect_target:
        # Soft redirect: sandboxed iframes block meta refresh → 302 here.
        return RedirectResponse(url=_url_for(zim_id, raw.soft_redirect_target), status_code=302)

    return _serve(raw.content, raw.mimetype, request, zim_id=zim_id)


class _RangeUnsatisfiable(Exception):
    """Raised when an HTTP Range header is syntactically valid but unsatisfiable."""


def _serve(content: bytes, mimetype: str, request: Request, *, zim_id: int) -> Response:
    headers = {**_CACHE_HEADERS, **_range_headers(mimetype, len(content))}
    # Basic single-range support for media (research: only needed for A/V ZIMs).
    range_hdr = request.headers.get("range")
    if range_hdr and headers.get("Accept-Ranges"):
        try:
            parsed = _parse_range(range_hdr, len(content))
        except _RangeUnsatisfiable:
            headers["Content-Range"] = f"bytes */{len(content)}"
            return Response(
                content=b"",
                media_type=mimetype,
                status_code=416,
                headers=headers,
            )
        if parsed is not None:
            start, end = parsed
            headers["Content-Range"] = f"bytes {start}-{end}/{len(content)}"
            headers["Content-Length"] = str(end - start + 1)
            return Response(
                content=content[start : end + 1],
                media_type=mimetype,
                status_code=206,
                headers=headers,
            )
    return Response(content=content, media_type=mimetype, headers=headers)


def _parse_range(range_hdr: str, size: int) -> tuple[int, int] | None:
    """Parse a ``bytes=start-end`` header (single range only).

    Returns ``(start, end)`` inclusive byte indices when satisfiable,
    ``None`` when the header is syntactically invalid (ignored per RFC 9110 §14.2),
    or raises ``_RangeUnsatisfiable`` when the range cannot be satisfied (HTTP 416).
    """
    if not range_hdr.startswith("bytes="):
        return None
    spec = range_hdr[len("bytes=") :].split(",", maxsplit=1)[0].strip()
    if "-" not in spec:
        return None
    start_s, _, end_s = (p.strip() for p in spec.partition("-"))
    if (
        (not start_s and not end_s)
        or (start_s and not start_s.isdigit())
        or (end_s and not end_s.isdigit())
    ):
        return None

    if not start_s:
        # Suffix range: bytes=-N
        n = int(end_s)
        if n <= 0 or size == 0:
            raise _RangeUnsatisfiable()
        return max(size - n, 0), size - 1

    start = int(start_s)
    if size == 0 or start >= size:
        raise _RangeUnsatisfiable()

    if not end_s:
        # Open-ended range: bytes=N-
        return start, size - 1

    end = int(end_s)
    if start > end:
        raise _RangeUnsatisfiable()
    return start, min(end, size - 1)


__all__ = ["router"]
