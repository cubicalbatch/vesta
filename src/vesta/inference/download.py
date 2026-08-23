"""The ``download_model`` job — resumable HTTP download of a GGUF to ``data/models/``.

Mirrors :mod:`vesta.catalog.download`'s pattern (stream + checkpoint + progress)
but simpler: no metalink, no mirrors, no SHA verification — a direct URL fetch
to a single file. GGUF URLs point at HuggingFace ``resolve/main/<file>``; HF
honours HTTP range requests so resume works identically to the ZIM path.

The job writes to ``<models_dir>/<filename>.part`` and atomically renames on
completion. Untrusted filenames are validated to bare ``*.gguf`` basenames by
:func:`safe_gguf_basename` before anything touches disk — ``POST /api/jobs``
forwards arbitrary params, so the endpoint's guard cannot be the only one. The
post-download "configure settings" step is owned by the API endpoint
(``POST /api/models/download``), not the job — the job's sole responsibility
is the file on disk.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from vesta import config
from vesta.jobs.types import RESUME_CHECKPOINT_KEY, JobHandle, register_job_type

_log = logging.getLogger(__name__)

#: Download chunk size (1 MiB) — matches the ZIM download path.
_CHUNK = 1024 * 1024

#: HTTP timeout per request (seconds).
_HTTP_TIMEOUT_S = 120.0


class DownloadModelError(RuntimeError):
    """Raised when a model download cannot complete."""


def safe_gguf_basename(filename: str, *, append_suffix: bool = False) -> str:
    """Validate that ``filename`` is a bare ``*.gguf`` name that cannot escape
    the models dir, returning it unchanged.

    The single guard behind every path that turns an untrusted filename into
    ``models_dir / <name>`` (pathlib lets absolute paths win and ``..`` climb
    out). With ``append_suffix=True`` a bare name gets ``.gguf`` appended
    first, so custom downloads stay friendly; otherwise the name must already
    end in ``.gguf``. Rejects path separators (URL-decoded ``%2F`` included),
    ``..``, absolute paths, and empty stems — the models dir also holds the
    ONNX encoder trees. Raises :class:`ValueError`; callers translate it into
    their own error type.
    """
    name = f"{filename}.gguf" if append_suffix and not filename.endswith(".gguf") else filename
    if (
        not name.endswith(".gguf")
        or not name.removesuffix(".gguf")
        or "/" in name
        or "\\" in name
        or ".." in name
        or Path(name).name != name
    ):
        raise ValueError(f"unsafe GGUF filename: {filename!r}")
    return name


class DownloadModelJob:
    """Registered as job type ``download_model``.

    Params: ``url`` (the direct GGUF URL), ``filename`` (the basename to write
    under ``data/models/`` — validated by :func:`safe_gguf_basename`; anything
    else fails the job).
    """

    name = "download_model"

    async def run(self, job: JobHandle, params: Mapping[str, Any]) -> None:
        await _run_download(job, params)


async def _run_download(job: JobHandle, params: Mapping[str, Any]) -> None:
    from vesta.inference import get_models_dir

    url = str(params["url"])
    # Last-line defense: nothing touches disk until the filename is a bare
    # *.gguf basename under the models dir.
    try:
        filename = safe_gguf_basename(str(params["filename"]), append_suffix=True)
    except ValueError as exc:
        raise DownloadModelError(str(exc)) from exc

    models_dir = get_models_dir()
    if models_dir is None:
        raise RuntimeError("download_model: models dir not bound (run inside the app lifespan)")
    models_dir.mkdir(parents=True, exist_ok=True)

    final_path = models_dir / filename
    part_path = models_dir / f"{filename}.part"

    # Resume from checkpoint.
    resume = params.get(RESUME_CHECKPOINT_KEY)
    bytes_done = 0
    if isinstance(resume, Mapping):
        bytes_done = max(0, int(resume.get("bytes_done", 0)))
        if (resume.get("url") or "") != url:
            bytes_done = 0

    if bytes_done == 0 and part_path.exists():
        part_path.unlink(missing_ok=True)
    elif bytes_done > 0 and part_path.exists() and part_path.stat().st_size < bytes_done:
        part_path.unlink(missing_ok=True)
        bytes_done = 0

    total = await _content_length(url)
    await job.progress(bytes_done, total or 0, _msg(bytes_done, total or 0, "starting"))

    bytes_done = await _download_with_resume(
        job=job,
        url=url,
        part_path=part_path,
        bytes_done=bytes_done,
        total=total,
    )

    part_path.replace(final_path)
    await job.progress(bytes_done or final_path.stat().st_size, total or 0, "downloaded")
    await job.progress(total or final_path.stat().st_size, total or 0, "done")
    # A finished download must refresh the runtime — and a running router
    # does NOT discover files dropped into --models-dir, so the callback also
    # restarts the child. A callback failure must not fail the
    # job: the GGUF is on disk and complete either way.
    from vesta.inference import notify_model_ready

    try:
        await notify_model_ready(final_path)
    except Exception as exc:
        _log.warning("download_model.ready_callback_failed", extra={"error": repr(exc)})


async def _content_length(url: str) -> int:
    """Best-effort Content-Length via HEAD; 0 if unavailable."""
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.head(url)
            if resp.status_code >= 400:
                return 0
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl else 0
    except Exception:
        return 0


async def _download_with_resume(
    *,
    job: JobHandle,
    url: str,
    part_path: Path,
    bytes_done: int,
    total: int,
) -> int:
    """Stream ``url`` to ``part_path`` (append from ``bytes_done``)."""
    # Read the shared bandwidth-limit setting by key to avoid importing
    # catalog/ (boundary: inference may import only config + jobs).
    bandwidth_desc = config.all_settings().get("catalog.download.bandwidth_limit_kbps")
    limit_kbps = int(config.get(bandwidth_desc)) if bandwidth_desc else 0
    headers: dict[str, str] = {}
    if bytes_done > 0:
        headers["Range"] = f"bytes={bytes_done}-"

    t0 = asyncio.get_event_loop().time()
    async with (
        httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S, follow_redirects=True) as client,
        client.stream("GET", url, headers=headers) as resp,
    ):
        if resp.status_code not in (200, 206):
            raise DownloadModelError(f"HTTP {resp.status_code} fetching {url}")

        served_from_zero = resp.status_code == 200
        if served_from_zero and bytes_done > 0:
            bytes_done = 0

        content_length = resp.headers.get("Content-Length")
        if content_length and content_length.isdigit():
            served_now = int(content_length)
            if not total:
                total = served_now + (bytes_done if not served_from_zero else 0)
            elif served_from_zero and served_now != total:
                _log.warning(
                    "download.size_mismatch",
                    extra={"expected": total, "served": served_now},
                )

        mode = "ab" if (bytes_done > 0 and not served_from_zero) else "wb"
        if mode == "wb":
            bytes_done = 0
        written = bytes_done

        with part_path.open(mode) as f:
            async for chunk in resp.aiter_bytes(_CHUNK):
                if job.cancelled():
                    raise asyncio.CancelledError
                f.write(chunk)
                written += len(chunk)
                await _checkpoint(job, written, total, url)
                await _maybe_throttle(written, bytes_done, limit_kbps, t0)
            f.flush()

    return written


async def _checkpoint(job: JobHandle, written: int, total: int, url: str) -> None:
    await job.progress(written, total, _msg(written, total, "downloading"))
    await job.checkpoint({"bytes_done": written, "size": total, "url": url})


def _msg(written: int, total: int, action: str) -> str:
    if total > 0:
        pct = written * 100 // total
        return f"{action}: {pct}% ({written // (1024 * 1024)} MB / {total // (1024 * 1024)} MB)"
    return f"{action}: {written // (1024 * 1024)} MB"


async def _maybe_throttle(written: int, start: int, limit_kbps: int, t0: float) -> None:
    """Best-effort per-download throttle (mirrors catalog/download.py)."""
    if limit_kbps <= 0:
        return
    elapsed = asyncio.get_event_loop().time() - t0
    served = written - start
    allowed = limit_kbps * 1024 * elapsed
    if served > allowed:
        ahead = (served - allowed) / (limit_kbps * 1024)
        await asyncio.sleep(min(ahead, 0.5))


# Register at import.
register_job_type(DownloadModelJob())


__all__ = [
    "DownloadModelError",
    "DownloadModelJob",
    "safe_gguf_basename",
]
