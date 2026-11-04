"""Validate HTTP byte ranges shared by the model and archive download jobs."""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class DownloadBounds:
    start: int
    total: int  # zero when the full size remains unknown
    end: int | None  # exclusive end of this response, if known


def response_bounds(resp: httpx.Response, start: int, total: int) -> DownloadBounds:
    """Validate before writing any body bytes; raise ValueError on ambiguity.

    A server ignoring Range and sending the whole object restarts at zero.
    A partial response must identify its exact offset and length, even when
    no checksum or prior size is available. Unknown-size full responses are
    still supported; unknown-size partial responses cannot prove completion.
    """
    if resp.status_code not in (200, 206):
        raise ValueError(f"HTTP {resp.status_code} fetching download")
    if resp.headers.get("Content-Encoding", "identity").lower() != "identity":
        raise ValueError("encoded download response cannot be resumed by byte offset")
    length = _content_length(resp)
    if resp.status_code == 200:
        if length is not None and total > 0 and length != total:
            raise ValueError(f"server Content-Length {length} != expected {total}")
        return DownloadBounds(
            0, total or length or 0, length if length is not None else total or None
        )

    match = re.fullmatch(
        r"bytes ([0-9]+)-([0-9]+)/([0-9]+|\*)", resp.headers.get("Content-Range", "")
    )
    if match is None:
        raise ValueError("server returned missing or malformed Content-Range")
    first, last = int(match[1]), int(match[2])
    if first not in (0, start):
        raise ValueError(f"server Content-Range start {first} != expected {start}")
    if match[3] != "*":
        reported_total = int(match[3])
        if total > 0 and reported_total != total:
            raise ValueError(f"server Content-Range total {reported_total} != expected {total}")
        total = reported_total
    if total <= 0 or last < first or last >= total:
        raise ValueError("server Content-Range does not identify a valid complete size")
    if length is not None and length != last - first + 1:
        raise ValueError("server Content-Length disagrees with Content-Range")
    return DownloadBounds(first, total, last + 1)


def _content_length(resp: httpx.Response) -> int | None:
    value = resp.headers.get("Content-Length")
    if value is None:
        return None
    if re.fullmatch(r"[0-9]+", value) is None:
        raise ValueError("server returned malformed Content-Length")
    return int(value)
