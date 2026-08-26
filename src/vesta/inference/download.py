"""The ``download_model`` job — resumable, checksummed HTTP download of a GGUF
to ``data/models/``.

Mirrors :mod:`vesta.catalog.download`'s pattern (stream + checkpoint + progress)
with ``Range: bytes=N-`` resume, written to ``*.part`` and atomically renamed
after whole-file SHA-256 verification (if provided). Direct URL fetch to a
single file (e.g. HuggingFace ``resolve/main/<file>``).

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
import hashlib
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from vesta import config
from vesta.config.netguard import (
    EgressBlocked,
    assert_public_http_url,
    guarded_request,
    guarded_stream,
    safe_client,
)
from vesta.jobs.types import (
    RESUME_CHECKPOINT_KEY,
    JobHandle,
    maybe_throttle,
    register_job_type,
)

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
    else fails the job), optional ``sha256`` and ``size``.
    """

    name = "download_model"

    async def run(self, job: JobHandle, params: Mapping[str, Any]) -> None:
        await _run_download(job, params)


def _initial_resume_offset(
    params: Mapping[str, Any],
    url: str,
    sha256: str | None,
    size: int,
    part_path: Path,
) -> int:
    """Read checkpoint offset or start from 0 if checkpoint/part is invalid."""
    resume = params.get(RESUME_CHECKPOINT_KEY)
    bytes_done = 0
    if isinstance(resume, Mapping):
        bytes_done = max(0, int(resume.get("bytes_done", 0)))
        if (
            (resume.get("url") or "") != url
            or (resume.get("sha256") or "") != (sha256 or "")
            or (size and int(resume.get("size", 0) or 0) != size)
        ):
            bytes_done = 0

    if bytes_done == 0 and part_path.exists():
        part_path.unlink(missing_ok=True)
        return 0
    if bytes_done > 0 and (not part_path.exists() or part_path.stat().st_size < bytes_done):
        part_path.unlink(missing_ok=True)
        return 0
    return _trim_part_to(part_path, bytes_done)


async def _run_download(job: JobHandle, params: Mapping[str, Any]) -> None:
    from vesta.inference import get_models_dir, notify_model_ready

    url = str(params["url"])
    # Last-line defense: nothing touches disk until the filename is a bare
    # *.gguf basename under the models dir.
    try:
        filename = safe_gguf_basename(str(params["filename"]), append_suffix=True)
    except ValueError as exc:
        raise DownloadModelError(str(exc)) from exc

    # AUDIT_0824 A1: ``url`` is request-controlled (raw URL on
    # POST /api/models/download or POST /api/jobs) — it must be public-internet
    # http(s) before anything is fetched. Owner-configured inference endpoints
    # are unaffected: they never pass through here.
    try:
        assert_public_http_url(url)
    except EgressBlocked as exc:
        raise DownloadModelError(str(exc)) from exc

    models_dir = get_models_dir()
    if models_dir is None:
        raise RuntimeError("download_model: models dir not bound (run inside the app lifespan)")
    models_dir.mkdir(parents=True, exist_ok=True)

    sha256 = str(params.get("sha256") or "").lower() or None
    fallback_size = int(params.get("size") or 0)
    final_path = models_dir / filename
    part_path = models_dir / f"{filename}.part"

    bytes_done = _initial_resume_offset(params, url, sha256, fallback_size, part_path)
    total = await _content_length(url) or fallback_size
    if total > 0 and bytes_done > total:
        part_path.unlink(missing_ok=True)
        bytes_done = 0

    await job.progress(bytes_done, total or 0, _msg(bytes_done, total or 0, "starting"))
    bytes_done = await _download_with_resume(
        job=job,
        url=url,
        part_path=part_path,
        bytes_done=bytes_done,
        total=total,
        sha256=sha256,
    )
    await _verify_checksum(job, part_path, sha256, bytes_done, total)

    part_path.replace(final_path)
    await job.progress(bytes_done or final_path.stat().st_size, total or 0, "downloaded")
    await job.progress(total or final_path.stat().st_size, total or 0, "done")

    try:
        await notify_model_ready(final_path)
    except Exception as exc:
        _log.warning("download_model.ready_callback_failed", extra={"error": repr(exc)})


async def _verify_checksum(
    job: JobHandle, part_path: Path, sha256: str | None, bytes_done: int, total: int
) -> None:
    """Verify SHA-256 before rename if hash provided and checksumming enabled."""
    if not sha256:
        return
    verify_desc = config.all_settings().get("catalog.download.verify_checksums")
    if verify_desc and not bool(config.get(verify_desc)):
        return
    await job.progress(bytes_done, total or 0, "verifying checksum")
    actual = await _sha256_of(part_path)
    if actual != sha256:
        part_path.unlink(missing_ok=True)
        raise DownloadModelError(
            f"checksum mismatch: expected {sha256}, got {actual}; download discarded"
        )


async def _content_length(url: str) -> int:
    """Best-effort Content-Length via HEAD; 0 if unavailable."""
    try:
        async with safe_client(timeout=30.0) as client:
            resp = await guarded_request(client, "HEAD", url)
            if resp.status_code >= 400:
                return 0
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl else 0
    except Exception:
        return 0


def _validate_response_headers(
    resp: httpx.Response, start: int, total: int, url: str
) -> tuple[int, bool, int]:
    """Validate status code, Range/Content-Range headers, and Content-Length."""
    if resp.status_code not in (200, 206):
        raise DownloadModelError(f"HTTP {resp.status_code} fetching {url}")

    served_from_zero = resp.status_code == 200
    if served_from_zero and start > 0:
        _log.warning("download_model.range_ignored", extra={"url": url})
        start = 0

    if resp.status_code == 206:
        content_range = resp.headers.get("Content-Range", "")
        if content_range.startswith("bytes "):
            spec = content_range[len("bytes ") :].strip()
            range_spec, _, total_spec = spec.partition("/")
            range_start_str, _, _ = range_spec.partition("-")
            if range_start_str.isdigit():
                range_start = int(range_start_str)
                if range_start == 0 and start > 0:
                    _log.warning("download_model.range_ignored", extra={"url": url})
                    served_from_zero = True
                    start = 0
                elif range_start != start:
                    raise DownloadModelError(
                        f"server Content-Range start {range_start} != expected {start}"
                    )
            if total_spec.isdigit() and not total:
                total = int(total_spec)

    content_length = resp.headers.get("Content-Length")
    if content_length and content_length.isdigit():
        served_now = int(content_length)
        if not total:
            total = served_now + (start if not served_from_zero else 0)
        elif served_from_zero and served_now != total:
            raise DownloadModelError(f"server Content-Length {served_now} != expected {total}")

    return start, served_from_zero, total


async def _download_with_resume(
    *,
    job: JobHandle,
    url: str,
    part_path: Path,
    bytes_done: int,
    total: int,
    sha256: str | None = None,
) -> int:
    """Stream ``url`` to ``part_path`` (append from ``bytes_done``)."""
    bandwidth_desc = config.all_settings().get("catalog.download.bandwidth_limit_kbps")
    limit_kbps = int(config.get(bandwidth_desc)) if bandwidth_desc else 0

    start = _trim_part_to(part_path, bytes_done)
    headers = {"Range": f"bytes={start}-"} if start > 0 else {}
    t0 = asyncio.get_event_loop().time()

    async with (
        safe_client(timeout=_HTTP_TIMEOUT_S) as client,
        guarded_stream(client, "GET", url, headers=headers) as resp,
    ):
        start, _served_from_zero, total = _validate_response_headers(resp, start, total, url)
        written = start

        with part_path.open("wb" if start == 0 else "r+b") as f:
            if start > 0:
                f.seek(start)
            async for chunk in resp.aiter_bytes(_CHUNK):
                if job.cancelled():
                    raise asyncio.CancelledError
                f.write(chunk)
                written += len(chunk)
                await _checkpoint(job, written, total, url, sha256)
                await maybe_throttle(written, start, limit_kbps, t0)
            f.flush()
            os.fsync(f.fileno())

    if total and written < total:
        raise DownloadModelError(f"server served {written}/{total} bytes then closed")
    return written


def _trim_part_to(part_path: Path, size: int) -> int:
    """Re-sync ``part_path`` with the ``size``-byte checkpoint before a resume
    (audit AUDIT_0824 I1), returning the offset to download from.

    Same guarantee as :func:`vesta.catalog.download._trim_part_to`: a crash
    between a chunk write and its checkpoint leaves bytes past ``size``; they
    are trimmed so the resumed range append lands exactly at the checkpoint.
    A .part missing or *shorter* than ``size`` has lost prefix bytes that
    cannot be reconstructed, so it is removed and the download restarts from
    zero."""
    try:
        actual = part_path.stat().st_size
    except FileNotFoundError:
        return 0  # nothing on disk; the wb path creates the file
    if actual < size:
        _log.warning(
            "download_model.part_shrank",
            extra={"path": str(part_path), "had": actual, "want": size},
        )
        part_path.unlink(missing_ok=True)
        return 0
    if actual > size:
        _log.warning(
            "download_model.part_trimmed",
            extra={"path": str(part_path), "from": actual, "to": size},
        )
        try:
            with part_path.open("r+b") as fh:
                fh.truncate(size)
                os.fsync(fh.fileno())
        except OSError as exc:
            raise DownloadModelError(
                f"cannot resume cleanly: {part_path} has {actual} bytes but the "
                f"checkpoint says {size}; truncation failed: {exc}"
            ) from exc
    return size


async def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()

    def _compute() -> None:
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)

    await asyncio.to_thread(_compute)
    return digest.hexdigest()


async def _checkpoint(
    job: JobHandle, written: int, total: int, url: str, sha256: str | None = None
) -> None:
    await job.progress(written, total, _msg(written, total, "downloading"))
    await job.checkpoint({"bytes_done": written, "size": total, "url": url, "sha256": sha256 or ""})


def _msg(written: int, total: int, action: str) -> str:
    if total > 0:
        pct = written * 100 // total
        return f"{action}: {pct}% ({written // (1024 * 1024)} MB / {total // (1024 * 1024)} MB)"
    return f"{action}: {written // (1024 * 1024)} MB"


# Register at import.
register_job_type(DownloadModelJob())


__all__ = [
    "DownloadModelError",
    "DownloadModelJob",
    "safe_gguf_basename",
]
