"""HTTP download integrity without checksums or a live network."""

from __future__ import annotations

import ipaddress
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest

from vesta import config
from vesta.catalog import download as zim_download
from vesta.config import netguard
from vesta.inference import download as model_download


class _Job:
    async def progress(self, done: int, total: int, message: str) -> None:
        pass

    async def checkpoint(self, blob: Mapping[str, Any]) -> None:
        pass

    def cancelled(self) -> bool:
        return False


@pytest.fixture(autouse=True)
def _configure(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setattr(
        netguard, "_resolve_host", lambda _host: [ipaddress.ip_address("93.184.216.34")]
    )
    config.configure()
    yield
    config.reset_for_test()


def _mock_http(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    def client(**kwargs: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(netguard, "safe_client", client)
    monkeypatch.setattr(model_download, "safe_client", client)


async def _download(kind: str, part: Path, total: int = 8) -> int:
    if kind == "zim":
        return await zim_download._download_with_resume(
            job=_Job(),
            mirrors=["https://example.com/file"],
            part_path=part,
            bytes_done=4,
            total=total,
            sha256=None,
        )
    return await model_download._download_with_resume(
        job=_Job(),
        url="https://example.com/file",
        part_path=part,
        bytes_done=4,
        total=total,
    )


@pytest.mark.parametrize("kind", ["zim", "model"])
@pytest.mark.parametrize(
    "content_range",
    [None, "garbage", "bytes 2-5/8", "bytes 4-7/9", "bytes 4-3/8", "bytes 4-8/8", "bytes 4-6/8"],
)
async def test_invalid_partial_response_never_appends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str, content_range: str | None
) -> None:
    headers = {} if content_range is None else {"Content-Range": content_range}
    _mock_http(monkeypatch, lambda req: httpx.Response(206, headers=headers, content=b"efgh"))
    part = tmp_path / "file.part"
    part.write_bytes(b"abcd")
    with pytest.raises((zim_download.DownloadError, model_download.DownloadModelError)):
        await _download(kind, part)
    assert part.read_bytes() == b"abcd"


class _Body(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.reads = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            self.reads += 1
            yield chunk


@pytest.mark.parametrize("kind", ["zim", "model"])
async def test_oversized_body_is_not_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str
) -> None:
    _mock_http(
        monkeypatch,
        lambda req: httpx.Response(
            206, headers={"Content-Range": "bytes 4-7/8"}, stream=_Body([b"efghEXTRA"])
        ),
    )
    part = tmp_path / "file.part"
    part.write_bytes(b"abcd")
    with pytest.raises((zim_download.DownloadError, model_download.DownloadModelError)):
        await _download(kind, part)
    assert part.read_bytes() == b"abcd"


@pytest.mark.parametrize("kind", ["zim", "model"])
async def test_valid_range_is_byte_exact_without_checksum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str
) -> None:
    _mock_http(
        monkeypatch,
        lambda req: httpx.Response(206, headers={"Content-Range": "bytes 4-7/8"}, content=b"efgh"),
    )
    part = tmp_path / "file.part"
    part.write_bytes(b"abcd")
    assert await _download(kind, part) == 8
    assert part.read_bytes() == b"abcdefgh"


@pytest.mark.parametrize("kind", ["zim", "model"])
async def test_valid_range_supplies_missing_total(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str
) -> None:
    _mock_http(
        monkeypatch,
        lambda req: httpx.Response(206, headers={"Content-Range": "bytes 4-7/8"}, content=b"efgh"),
    )
    part = tmp_path / "file.part"
    part.write_bytes(b"abcd")
    assert await _download(kind, part, total=0) == 8
    assert part.read_bytes() == b"abcdefgh"


@pytest.mark.parametrize("kind", ["zim", "model"])
async def test_range_ignored_restarts_cleanly_with_identity_encoding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str
) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["Accept-Encoding"] == "identity"
        return httpx.Response(200, content=b"abcdefgh")

    _mock_http(monkeypatch, handler)
    part = tmp_path / "file.part"
    part.write_bytes(b"abcd")
    assert await _download(kind, part) == 8
    assert part.read_bytes() == b"abcdefgh"


async def test_short_chunked_mirror_falls_back_from_written_offset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    starts: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        starts.append(req.headers["Range"])
        if req.url.host == "short.example":
            return httpx.Response(
                206, headers={"Content-Range": "bytes 4-7/8"}, stream=_Body([b"ef"])
            )
        return httpx.Response(206, headers={"Content-Range": "bytes 6-7/8"}, content=b"gh")

    _mock_http(monkeypatch, handler)
    part = tmp_path / "file.part"
    part.write_bytes(b"abcd")
    assert (
        await zim_download._download_with_resume(
            job=_Job(),
            mirrors=["https://short.example/file", "https://good.example/file"],
            part_path=part,
            bytes_done=4,
            total=8,
            sha256=None,
        )
        == 8
    )
    assert starts == ["bytes=4-", "bytes=6-"]
    assert part.read_bytes() == b"abcdefgh"


async def test_range_ignoring_mirror_failure_retains_new_smaller_offset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    starts: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        starts.append(req.headers["Range"])
        if req.url.host == "short.example":
            return httpx.Response(200, stream=_Body([b"ab"]))
        return httpx.Response(206, headers={"Content-Range": "bytes 2-7/8"}, content=b"cdefgh")

    _mock_http(monkeypatch, handler)
    part = tmp_path / "file.part"
    part.write_bytes(b"abcd")
    assert (
        await zim_download._download_with_resume(
            job=_Job(),
            mirrors=["https://short.example/file", "https://good.example/file"],
            part_path=part,
            bytes_done=4,
            total=8,
            sha256=None,
        )
        == 8
    )
    assert starts == ["bytes=4-", "bytes=2-"]
    assert part.read_bytes() == b"abcdefgh"


async def test_direct_zim_url_skips_buffered_metadata_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        return httpx.Response(200, content=b"ZIM payload")

    _mock_http(monkeypatch, handler)
    url = "https://example.com/huge.zim?download=1"
    info = await zim_download._resolve_metalink(url, "metalink")
    assert info.mirrors == (url,)
    assert requests == []


async def test_metadata_body_is_bounded_without_content_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _Body([b"x" * 65536] * 64)
    _mock_http(monkeypatch, lambda req: httpx.Response(200, stream=body))
    info = await zim_download._resolve_metalink("https://example.com/huge.zim.meta4", "metalink")
    assert info.mirrors == ("https://example.com/huge.zim",)
    assert body.reads <= 17  # metadata is capped at 1 MiB, plus the rejecting chunk


async def test_oversized_metadata_content_length_rejected_before_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _Body([b"x"])
    _mock_http(
        monkeypatch,
        lambda req: httpx.Response(200, headers={"Content-Length": "5000000"}, stream=body),
    )
    info = await zim_download._resolve_metalink("https://example.com/a.zim.meta4", "metalink")
    assert info.mirrors == ("https://example.com/a.zim",)
    assert body.reads == 0


def test_meta4_fallback_preserves_query_while_stripping_only_path_suffix() -> None:
    assert zim_download._direct_url_from_meta4("https://example.com/a.zim.meta4?token=abc") == (
        "https://example.com/a.zim?token=abc"
    )
