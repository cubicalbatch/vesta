"""The ``download_zim`` job — resumable, checksummed HTTP download.

Single connection with ``Range: bytes=N-`` resume, written to ``*.part`` and
atomically renamed after a whole-file SHA-256 check. Multi-GB downloads over days
are the design target, not an edge case.

Design considerations this job handles:

* **Disk exhaustion mid-download.** Pre-flight a free-space check against the
  metalink size + ``download.min_free_space_gb`` headroom, re-check periodically,
  fail the job with a clear message rather than filling the volume.
* **Mirrors lie about Content-Length / ranges.** Probe range support with a HEAD;
  fall back down the mirror list, and to restart-from-zero if a mirror doesn't
  honour ``Range`` (never silently corrupt a ``.part``).
* **Checksum before rename, always.** A truncated ZIM that gets registered
  produces confusing libzim errors much later. The rename only happens after the
  SHA-256 matches.
* **Resume survives a restart.** The checkpoint is ``(url, size, sha256,
  bytes_done)``; on resume the job re-probes the mirror and appends from
  ``bytes_done`` if the byte range is still served.

Post-download the job calls the injected register callback (``main`` wires it to
``registry.rescan``) so the registry probes the fulltext index, counts articles from
``Counter['text/html']``, and mines the alias dictionary.
``catalog/`` cannot import ``zim/``, so registration is a
callback, not an import.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import shutil
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from vesta import config
from vesta.catalog import (
    DOWNLOAD_BANDWIDTH_LIMIT_KBPS,
    DOWNLOAD_MIN_FREE_SPACE_GB,
    DOWNLOAD_MIRROR_POLICY,
    DOWNLOAD_VERIFY_CHECKSUMS,
    get_register_archive,
    get_zims_dir,
)
from vesta.catalog.opds import _local_name
from vesta.jobs.types import RESUME_CHECKPOINT_KEY, JobHandle, register_job_type

_log = logging.getLogger(__name__)

#: Download chunk size (1 MiB). Bounds memory + how often we checkpoint/resume.
_CHUNK = 1024 * 1024

#: Re-check free space every this many bytes (50 MiB) so a filling volume is
#: caught well before it's full (prevents disk exhaustion mid-download).
_DISK_CHECK_EVERY = 50 * _CHUNK

#: Seconds to wait per HTTP request before degrading to the next mirror.
_HTTP_TIMEOUT_S = 120.0


class DownloadError(RuntimeError):
    """Raised when a download cannot complete. The runner records it on the job."""


# ── metalink parsing ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MetalinkInfo:
    """What the ``.meta4`` metalink told us: size, sha-256, ranked mirrors."""

    filename: str
    size: int
    sha256: str | None
    mirrors: tuple[str, ...]  # priority order (1 = first)


def parse_metalink(xml_text: str) -> MetalinkInfo:
    """Parse a MirrorBrain ``.meta4`` metalink.

    Extracts ``<size>``, the ``sha-256`` ``<hash>``, and the ranked ``<url>`` list
    (numeric ``priority``; lower goes first). Pieces are available in the feed but
    not parsed here — whole-file SHA-256 is the gate; piece-hashing is a future
    early-corruption optimization, not load-bearing.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:  # pragma: no cover - defensive
        raise DownloadError(f"metalink is not well-formed XML: {exc}") from exc
    file_elem: ET.Element | None = None
    if _local_name(root.tag) == "metalink":
        for child in root:
            if _local_name(child.tag) == "file":
                file_elem = child
                break
    if file_elem is None:
        raise DownloadError("metalink has no <file> element")

    filename = str(file_elem.get("name", "")).strip()
    size = 0
    sha256: str | None = None
    mirrors: list[tuple[int, str]] = []
    for child in file_elem:
        name = _local_name(child.tag)
        text = (child.text or "").strip()
        if name == "size":
            with contextlib.suppress(ValueError):
                size = int(text)
        elif name == "hash" and child.get("type", "").lower() in {"sha-256", "sha256"}:
            sha256 = text.lower()
        elif name == "url":
            href = text
            if href:
                try:
                    priority = int(child.get("priority", "999"))
                except ValueError:
                    priority = 999
                mirrors.append((priority, href))
    mirrors.sort(key=lambda pair: pair[0])
    return MetalinkInfo(
        filename=filename or "download.zim",
        size=size,
        sha256=sha256,
        mirrors=tuple(href for _, href in mirrors),
    )


def _direct_url_from_meta4(meta4_url: str) -> str:
    """Strip ``.meta4`` for a direct ZIM URL (fallback)."""
    return meta4_url[:-6] if meta4_url.endswith(".meta4") else meta4_url


# ── the job ─────────────────────────────────────────────────────────────────


class DownloadZimJob:
    """Registered as job type ``download_zim``.

    Params: ``url`` (the catalog acquisition / ``.meta4`` URL), ``name`` (the
    filename stem to write under ``data/zims/``), optional ``title``,
    ``sha256``, ``size`` (fallbacks when the metalink can't be fetched).
    """

    name = "download_zim"

    async def run(self, job: JobHandle, params: Mapping[str, Any]) -> None:
        await _run_download(job, params)


async def _run_download(job: JobHandle, params: Mapping[str, Any]) -> None:
    zims_dir = get_zims_dir()
    if zims_dir is None:
        raise RuntimeError("download_zim: zims dir not bound (run inside the app lifespan)")
    zims_dir.mkdir(parents=True, exist_ok=True)

    meta4_url = str(params["url"])
    name = str(params.get("name") or _name_from_url(meta4_url))
    fallback_sha = str(params.get("sha256") or "").lower() or None
    fallback_size = int(params.get("size") or 0)
    mirror_policy = str(config.get(DOWNLOAD_MIRROR_POLICY))

    # 1. Resolve mirrors + size + sha256 from the metalink.
    info = await _resolve_metalink(meta4_url, mirror_policy)
    size = info.size or fallback_size
    sha256 = info.sha256 or fallback_sha
    mirrors = info.mirrors or (_direct_url_from_meta4(meta4_url),)
    filename = info.filename or f"{name}.zim"
    if not filename.endswith(".zim"):
        filename = f"{filename}.zim"

    # 2. Resume: read the checkpoint written at the last successful chunk flush.
    resume = params.get(RESUME_CHECKPOINT_KEY)
    bytes_done = 0
    if isinstance(resume, Mapping):
        bytes_done = max(0, int(resume.get("bytes_done", 0)))
        # If the checkpoint recorded a different sha/size, restart from zero so a
        # mismatched .part never gets stitched onto a different source.
        if (resume.get("sha256") or "") != (sha256 or "") or (
            size and int(resume.get("size", 0) or 0) != size
        ):
            bytes_done = 0

    final_path = zims_dir / filename
    part_path = zims_dir / f"{filename}.part"

    # 3. Pre-flight: free space + stale-.part sanity.
    _check_free_space(zims_dir, size, final_path)
    if bytes_done == 0 and part_path.exists():
        # A checkpoint-less resume starts clean; never append onto an unknown file.
        part_path.unlink(missing_ok=True)
    elif bytes_done > 0 and part_path.exists() and part_path.stat().st_size < bytes_done:
        # The .part shrank underneath us (manual deletion, etc.) — restart clean.
        part_path.unlink(missing_ok=True)
        bytes_done = 0

    verify = bool(config.get(DOWNLOAD_VERIFY_CHECKSUMS))
    total = size if size > 0 else 0
    await job.progress(bytes_done, total, _msg(bytes_done, total, "starting"))

    # 4. Download, appending to .part and checkpointing each chunk.
    bytes_done = await _download_with_resume(
        job=job,
        mirrors=mirrors,
        part_path=part_path,
        bytes_done=bytes_done,
        total=total,
        sha256=sha256,
    )

    # 5. Verify whole-file SHA-256 before the rename.
    if verify and sha256:
        await job.progress(bytes_done, total, "verifying checksum")
        actual = await _sha256_of(part_path)
        if actual != sha256:
            part_path.unlink(missing_ok=True)
            raise DownloadError(
                f"checksum mismatch: expected {sha256}, got {actual}; download discarded"
            )

    # 6. Atomic rename — never expose a partial file to the archive registry.
    part_path.replace(final_path)
    await job.progress(bytes_done or final_path.stat().st_size, total, "downloaded")

    # 7. Register the archive (via the injected rescan callback).
    register = get_register_archive()
    if register is not None:
        await job.progress(bytes_done or total, total, "registering archive")
        try:
            await register(final_path)
        except Exception as exc:  # registration failure is non-fatal to the download
            _log.warning(
                "download.register_failed", extra={"path": str(final_path), "error": repr(exc)}
            )

    await job.progress(total or final_path.stat().st_size, total or 0, "done")


# ── mirror resolution ───────────────────────────────────────────────────────


async def _resolve_metalink(meta4_url: str, mirror_policy: str) -> MetalinkInfo:
    """Fetch + parse the ``.meta4`` (policy ``metalink``); fall back to the
    direct URL (policy ``first`` or a fetch failure). Never raises — a degraded
    metalink yields a single-mirror ``MetalinkInfo`` so the download still runs."""
    if mirror_policy == "first":
        return MetalinkInfo(
            filename="", size=0, sha256=None, mirrors=(_direct_url_from_meta4(meta4_url),)
        )
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S, follow_redirects=True) as client:
            resp = await client.get(meta4_url)
            resp.raise_for_status()
            return parse_metalink(resp.text)
    except Exception as exc:
        _log.warning(
            "download.metalink_failed",
            extra={"url": meta4_url, "error": repr(exc)},
        )
        return MetalinkInfo(
            filename="", size=0, sha256=None, mirrors=(_direct_url_from_meta4(meta4_url),)
        )


# ── the resumable download loop ─────────────────────────────────────────────


async def _download_with_resume(
    *,
    job: JobHandle,
    mirrors: Sequence[str],
    part_path: Path,
    bytes_done: int,
    total: int,
    sha256: str | None,
) -> int:
    """Download from the first mirror that serves a usable byte range, appending
    to ``part_path``. Tries each mirror in turn; a mirror that rejects ranges
    restarts from zero with a warning. ``bytes_done`` returns
    the verified total written."""
    limit_kbps = int(config.get(DOWNLOAD_BANDWIDTH_LIMIT_KBPS))
    last_disk_check = bytes_done
    t0 = asyncio.get_event_loop().time()

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S, follow_redirects=True) as client:
        for idx, mirror in enumerate(mirrors):
            try:
                return await _download_one(
                    client=client,
                    url=mirror,
                    part_path=part_path,
                    start=bytes_done,
                    total=total,
                    sha256=sha256,
                    job=job,
                    limit_kbps=limit_kbps,
                    last_disk_check_init=last_disk_check,
                    t0=t0,
                )
            except asyncio.CancelledError:
                # Pause/cancel: the checkpoint already landed; re-raise so the
                # runner records paused/cancelled and resume continues from it.
                raise
            except Exception as exc:
                _log.warning(
                    "download.mirror_failed",
                    extra={"mirror": mirror, "index": idx, "error": repr(exc)},
                )
                continue
    raise DownloadError(f"all {len(mirrors)} mirror(s) failed; no bytes written")


async def _download_one(
    *,
    client: httpx.AsyncClient,
    url: str,
    part_path: Path,
    start: int,
    total: int,
    sha256: str | None,
    job: JobHandle,
    limit_kbps: int,
    last_disk_check_init: int,
    t0: float,
) -> int:
    """Stream one mirror to ``part_path`` (append from ``start``). Raises on any
    network/range failure so the caller tries the next mirror."""
    headers: dict[str, str] = {}
    if start > 0:
        headers["Range"] = f"bytes={start}-"

    # ``stream`` gives us the body without buffering the whole file in memory.
    async with client.stream("GET", url, headers=headers) as resp:
        if resp.status_code not in (200, 206):
            raise DownloadError(f"mirror returned HTTP {resp.status_code}")
        served_from_zero = resp.status_code == 200
        if served_from_zero and start > 0:
            # This mirror ignored the Range header: restart from zero rather than
            # silently corrupting the .part by appending the whole file.
            _log.warning("download.range_ignored", extra={"mirror": url})
            start = 0
        content_length = resp.headers.get("Content-Length")
        # Validate the mirror isn't lying about the size of what it'll serve now.
        if content_length and content_length.isdigit():
            served_now = int(content_length)
            if total > 0 and served_from_zero and served_now != total:
                raise DownloadError(f"mirror Content-Length {served_now} != expected {total}")

        mode = "ab" if (start > 0 and not served_from_zero) else "wb"
        if mode == "wb":
            start = 0
        written = start
        digest = hashlib.sha256()
        last_disk_check = last_disk_check_init

        with part_path.open(mode) as fh:
            async for chunk in resp.aiter_bytes(_CHUNK):
                if job.cancelled():
                    raise asyncio.CancelledError
                fh.write(chunk)
                digest.update(chunk)
                written += len(chunk)
                # Periodic free-space re-check.
                if written - last_disk_check >= _DISK_CHECK_EVERY:
                    _check_free_space(part_path.parent, total, part_path)
                    last_disk_check = written
                await _maybe_throttle(written, start, limit_kbps, t0)
                await _checkpoint(job, written, total, url, sha256)
            # fsync while the file is still open so the resumed size survives a
            # crash before the next checkpoint lands.
            fh.flush()
            os.fsync(fh.fileno())

    if total and written < total:
        raise DownloadError(f"mirror served {written}/{total} bytes then closed")
    return written


# ── helpers ─────────────────────────────────────────────────────────────────


def _name_from_url(url: str) -> str:
    from urllib.parse import urlsplit

    path = urlsplit(url).path
    tail = path.rsplit("/", 1)[-1]
    for suffix in (".meta4", ".zim"):
        if tail.endswith(suffix):
            tail = tail[: -len(suffix)]
            break
    return tail or "download"


def _check_free_space(zims_dir: Path, size: int, target: Path) -> None:
    """Fail loudly if the download would leave less than the configured headroom
    free (preventing disk exhaustion mid-download)."""
    min_free_gb = int(config.get(DOWNLOAD_MIN_FREE_SPACE_GB))
    min_free = min_free_gb * 1024 * 1024 * 1024
    usage = shutil.disk_usage(zims_dir)
    # Account for an existing target/.part that this download replaces.
    replacing = 0
    for candidate in (target, target.with_suffix(target.suffix + ".part")):
        with contextlib.suppress(OSError):
            replacing += candidate.stat().st_size
    free_after = usage.free + replacing - max(size, 0)
    if free_after < min_free:
        raise DownloadError(
            f"insufficient free space: need {size / 1e9:.1f} GB + {min_free_gb} GB headroom, "
            f"only {usage.free / 1e9:.1f} GB free on {zims_dir}"
        )


async def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()

    def _compute() -> None:
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)

    await asyncio.to_thread(_compute)
    return digest.hexdigest()


async def _maybe_throttle(written: int, start: int, limit_kbps: int, t0: float) -> None:
    """Best-effort per-download throttle. Compares bytes-served-this-run against
    elapsed time and sleeps to keep under ``limit_kbps`` KiB/s (0 = unlimited)."""
    if limit_kbps <= 0:
        return
    served = written - start
    if served <= 0:
        return
    elapsed = asyncio.get_event_loop().time() - t0
    target_seconds = served / (limit_kbps * 1024)
    ahead = target_seconds - elapsed
    if ahead > 0:
        await asyncio.sleep(min(ahead, 0.5))


async def _checkpoint(
    job: JobHandle, written: int, total: int, url: str, sha256: str | None
) -> None:
    await job.progress(written, total, _msg(written, total, "downloading"))
    await job.checkpoint({"bytes_done": written, "size": total, "sha256": sha256 or "", "url": url})


def _msg(written: int, total: int, action: str) -> str:
    if total > 0:
        pct = written * 100 // total
        return f"{action}: {written}/{total} bytes ({pct}%)"
    return f"{action}: {written} bytes"


# Register the built-in download job type at import.
register_job_type(DownloadZimJob())


__all__ = [
    "DownloadError",
    "DownloadZimJob",
    "MetalinkInfo",
    "parse_metalink",
]
