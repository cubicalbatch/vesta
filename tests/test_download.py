"""The ``download_zim`` job — resumable, checksummed HTTP download over a local
test server.

No large files, no real network: a tiny in-process threaded HTTP server serves a
small file with range support. The tests exercise the load-bearing paths:

* full download + whole-file SHA-256 verification + atomic rename + register.
* resume after a simulated interruption (the job restarts from its checkpoint).
* disk-space guard fails loudly on a constrained volume.
* metalink parsing picks mirrors by priority + the sha-256.

The "kill the container mid-download, restart, watch it resume and finish" DoD
item is simulated here by running the job twice with a checkpoint handoff (the
same mechanism the runner uses across a real restart).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from vesta import config
from vesta.catalog import bind_runtime
from vesta.catalog.download import (
    _CHUNK,
    DownloadError,
    DownloadZimJob,
    parse_metalink,
)
from vesta.jobs.handle import JobHandleImpl
from vesta.jobs.types import RESUME_CHECKPOINT_KEY

# ── a tiny range-capable HTTP server (no external deps) ─────────────────────


class _RangeHandler(BaseHTTPRequestHandler):
    serve_bytes: bytes = b""  # set per server instance via subclass attribute
    serve_meta4: str | None = None  # when set, *.meta4 paths get this XML
    # When set, only this many bytes of any data response are sent before the
    # connection is cut mid-body (a mirror dying partway through).
    drop_after: int | None = None
    ignore_range: bool = False
    bad_content_range: str | None = None
    seen_starts: ClassVar[list[int]] = []  # range-start of every data request served

    def log_message(self, format: str, *args: object) -> None:  # silence
        pass

    def do_GET(self) -> None:
        if self.serve_meta4 is not None and self.path.endswith(".meta4"):
            body = self.serve_meta4.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/xml")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        data = self.serve_bytes
        total = len(data)
        range_header = self.headers.get("Range")
        start = 0
        end = total - 1
        ranged = False
        if range_header and range_header.startswith("bytes=") and not self.ignore_range:
            ranged = True
            spec = range_header[len("bytes=") :].split("-")
            start = int(spec[0]) if spec[0] else 0
            end = int(spec[1]) if len(spec) > 1 and spec[1] else total - 1
        self.seen_starts.append(start)
        chunk = data[start : end + 1]
        self.send_response(206 if ranged else 200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(chunk)))
        if ranged:
            cr = self.bad_content_range or f"bytes {start}-{end}/{total}"
            self.send_header("Content-Range", cr)
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        if self.drop_after is not None:
            # Send only the promised prefix, then let the handler return so the
            # socket closes with Content-Length unmet — a mid-body failure.
            self.wfile.write(chunk[: self.drop_after])
            return
        self.wfile.write(chunk)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _Server:
    """A threaded HTTP server serving fixed bytes (and optionally metalink
    XML on ``*.meta4`` paths, with ``{base_url}`` placeholders filled in)
    with range support."""

    def __init__(
        self,
        data: bytes,
        meta4: str | None = None,
        *,
        drop_after: int | None = None,
        ignore_range: bool = False,
        bad_content_range: str | None = None,
    ) -> None:
        self.data = data

        class Handler(_RangeHandler):
            pass

        Handler.serve_bytes = data
        Handler.seen_starts = []  # fresh list per server (class attrs would be shared)
        if meta4 is not None:
            Handler.serve_meta4 = meta4
        if drop_after is not None:
            Handler.drop_after = drop_after
        Handler.ignore_range = ignore_range
        Handler.bad_content_range = bad_content_range
        self._handler_cls: type[_RangeHandler] = Handler
        self._srv = ThreadingHTTPServer(("127.0.0.1", _free_port()), Handler)
        if meta4 is not None and "{base_url}" in meta4:
            Handler.serve_meta4 = meta4.format(base_url=self.base_url)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)

    @property
    def seen_starts(self) -> list[int]:
        """Range starts of the data requests this server has served."""
        return self._handler_cls.seen_starts

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._srv.server_address[1]}"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()
        self._thread.join(timeout=2)


@pytest.fixture
def zims_dir(tmp_path: Path) -> Path:
    d = tmp_path / "zims"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(autouse=True)
def _reset_config() -> object:
    """Ensure the settings resolver doesn't leak across download tests."""
    yield None
    config.reset_for_test()


@pytest.fixture(autouse=True)
def _allow_test_egress(monkeypatch: pytest.MonkeyPatch) -> None:
    """The egress guard (AUDIT_0824 A1) rejects loopback targets, but these
    tests download from an in-process 127.0.0.1 server. Stub the guard's DNS
    resolution so every host looks globally routable; the rejection classes
    themselves are covered in ``test_netguard.py``."""
    import ipaddress

    from vesta.config import netguard

    monkeypatch.setattr(
        netguard,
        "_resolve_host",
        lambda _host: [ipaddress.ip_address("93.184.216.34")],
    )


# A minimal fake JobRunner so JobHandleImpl can publish progress/checkpoints
# without a real DB. Captures the last checkpoint for resume handoff.
class _FakeRunner:
    def __init__(self) -> None:
        self.checkpoint: str | None = None
        self.progress: list[tuple[int, int]] = []

    async def _publish_progress(self, job_id: int, done: int, total: int, message: str) -> None:
        self.progress.append((done, total))

    async def _write_progress(
        self, job_id: int, done: int, total: int, message: str, *, final: bool
    ) -> None:
        pass

    async def _write_checkpoint(self, job_id: int, blob: object) -> None:
        self.checkpoint = json.dumps(dict(blob))  # type: ignore[arg-type]

    def _is_cancelling(self, job_id: int) -> bool:
        return False


async def _run_job(
    zims_dir: Path,
    params: dict[str, object],
    *,
    resume_checkpoint: str | None = None,
    register: object = None,
) -> _FakeRunner:
    if register is None:
        register = _noop_register
    bind_runtime(db=None, zims_dir=str(zims_dir), register_archive=register)
    # The job reads download.* settings through the resolver; configure it from
    # the current environment so per-test env overrides (mirror_policy etc.)
    # take effect for this run.
    config.configure()
    runner = _FakeRunner()
    if resume_checkpoint is not None:
        params = {**params, RESUME_CHECKPOINT_KEY: json.loads(resume_checkpoint)}
    handle = JobHandleImpl(runner, job_id=1)  # type: ignore[arg-type]
    job = DownloadZimJob()
    await job.run(handle, params)
    return runner


async def _noop_register(path: object) -> None:
    pass


# ── metalink parsing ────────────────────────────────────────────────────────


def test_parse_metalink_picks_mirrors_by_priority_and_sha256() -> None:
    meta4 = f"""<?xml version="1.0"?>
<metalink xmlns="urn:ietf:params:xml:ns:metalink">
  <file name="wikipedia_en_top_nopic_2026-06.zim">
    <size>{len(b"x" * 100)}</size>
    <hash type="sha-256">{"a" * 64}</hash>
    <hash type="md5">{"b" * 32}</hash>
    <url location="us" priority="3">https://c.example/zim.zim</url>
    <url location="nl" priority="1">https://a.example/zim.zim</url>
    <url location="fr" priority="2">https://b.example/zim.zim</url>
  </file>
</metalink>"""
    info = parse_metalink(meta4)
    assert info.filename == "wikipedia_en_top_nopic_2026-06.zim"
    assert info.size == 100
    assert info.sha256 == "a" * 64  # sha-256 only; md5 ignored
    assert info.mirrors == (
        "https://a.example/zim.zim",
        "https://b.example/zim.zim",
        "https://c.example/zim.zim",
    )  # priority 1, 2, 3


def test_parse_metalink_tolerates_missing_hash_and_size() -> None:
    info = parse_metalink(
        '<?xml version="1.0"?><metalink xmlns="urn:ietf:params:xml:ns:metalink">'
        '<file name="x.zim"><url priority="1">https://x/x.zim</url></file></metalink>'
    )
    assert info.size == 0
    assert info.sha256 is None
    assert info.mirrors == ("https://x/x.zim",)


# ── full download + verify + rename + register ─────────────────────────────


async def test_full_download_verifies_and_renames_and_registers(
    zims_dir: Path,
) -> None:
    payload = os.urandom(2048)
    sha = hashlib.sha256(payload).hexdigest()
    registered: list[Path] = []

    async def register(path: object) -> None:
        registered.append(Path(path))  # type: ignore[arg-type]

    server = _Server(payload)
    server.start()
    try:
        # Use mirror_policy=first to skip metalink fetch and download the URL
        # directly; sha256/size come from the params fallback.
        os.environ["download.mirror_policy"] = "first"
        os.environ["download.min_free_space_gb"] = "0"
        await _run_job(
            zims_dir,
            {
                "url": f"{server.base_url}/file.zim",
                "name": "test_archive",
                "sha256": sha,
                "size": len(payload),
            },
            register=register,
        )
    finally:
        os.environ.pop("download.mirror_policy", None)
        os.environ.pop("download.min_free_space_gb", None)
        server.stop()

    final = zims_dir / "test_archive.zim"
    assert final.exists()
    assert final.read_bytes() == payload
    # The .part was atomically renamed away.
    assert not (zims_dir / "test_archive.zim.part").exists()
    # The register callback was handed the final path.
    assert registered == [final]


async def test_checksum_mismatch_discards_download(zims_dir: Path) -> None:
    payload = os.urandom(1024)
    wrong_sha = "0" * 64
    server = _Server(payload)
    server.start()
    try:
        os.environ["download.mirror_policy"] = "first"
        os.environ["download.min_free_space_gb"] = "0"
        with pytest.raises(DownloadError, match="checksum mismatch"):
            await _run_job(
                zims_dir,
                {
                    "url": f"{server.base_url}/file.zim",
                    "name": "bad_archive",
                    "sha256": wrong_sha,
                    "size": len(payload),
                },
            )
    finally:
        os.environ.pop("download.mirror_policy", None)
        os.environ.pop("download.min_free_space_gb", None)
        server.stop()
    # A corrupt download is discarded: neither the final file nor the .part remain.
    assert not (zims_dir / "bad_archive.zim").exists()
    assert not (zims_dir / "bad_archive.zim.part").exists()


# ── resume after a simulated interruption (DoD item) ────────────────────────


async def test_resume_continues_from_checkpoint(zims_dir: Path) -> None:
    payload = os.urandom(4096)
    sha = hashlib.sha256(payload).hexdigest()
    server = _Server(payload)
    server.start()
    try:
        os.environ["download.mirror_policy"] = "first"
        os.environ["download.min_free_space_gb"] = "0"

        # Simulate a prior interrupted run: write a .part with the first half,
        # and seed the job's params with the resume checkpoint as the runner would.
        part = zims_dir / "resumable.zim.part"
        half = len(payload) // 2
        part.write_bytes(payload[:half])
        checkpoint = json.dumps(
            {"bytes_done": half, "size": len(payload), "sha256": sha, "url": ""}
        )
        await _run_job(
            zims_dir,
            {
                "url": f"{server.base_url}/file.zim",
                "name": "resumable",
                "sha256": sha,
                "size": len(payload),
            },
            resume_checkpoint=checkpoint,
        )
    finally:
        os.environ.pop("download.mirror_policy", None)
        os.environ.pop("download.min_free_space_gb", None)
        server.stop()

    final = zims_dir / "resumable.zim"
    assert final.read_bytes() == payload  # whole file reassembled correctly


# ── disk-space guard ────────────────────────────────────────────────────────


async def test_disk_space_guard_fails_loudly(
    zims_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A download that would leave less than the configured headroom fails with a
    clear message rather than filling the volume."""
    # Force the guard to think free space is tiny regardless of the real volume.
    import vesta.catalog.download as dl

    class _Tiny:
        free = 0
        total = 0
        used = 0

    monkeypatch.setattr(dl.shutil, "disk_usage", lambda p: _Tiny())
    monkeypatch.setenv("download.min_free_space_gb", "1")
    payload = os.urandom(64)
    server = _Server(payload)
    server.start()
    try:
        os.environ["download.mirror_policy"] = "first"
        with pytest.raises(DownloadError, match="insufficient free space"):
            await _run_job(
                zims_dir,
                {
                    "url": f"{server.base_url}/f.zim",
                    "name": "toobig",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size": len(payload),
                },
            )
    finally:
        os.environ.pop("download.mirror_policy", None)
        server.stop()


# ── the download_model job sink (vesta.inference.download, audit M1) ────────


async def test_model_job_rejects_hostile_filenames_before_touching_disk(
    tmp_path: Path,
) -> None:
    """POST /api/jobs forwards arbitrary params to any registered type, so the
    job itself must refuse anything that is not a bare ``*.gguf`` basename —
    pathlib would let absolute paths win and ``..`` climb out."""
    from vesta.inference import bind_models_dir
    from vesta.inference.download import DownloadModelError, DownloadModelJob

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    canary = tmp_path / "canary.txt"
    canary.write_text("safe")
    bind_models_dir(models_dir)
    try:
        for bad in ("/etc/passwd.gguf", "../../escaped", "sub/dir/x", "..evil"):
            handle = JobHandleImpl(_FakeRunner(), job_id=1)  # type: ignore[arg-type]
            with pytest.raises(DownloadModelError, match="unsafe GGUF filename"):
                await DownloadModelJob().run(
                    handle, {"url": "https://example.com/x.gguf", "filename": bad}
                )
    finally:
        bind_models_dir(None)
    assert canary.read_text() == "safe"
    assert list(models_dir.iterdir()) == []


async def test_model_job_writes_bare_basename_to_final_and_part_paths(
    tmp_path: Path,
) -> None:
    """A bare custom name is normalized once and used for both the ``.part``
    and the final file, inside the bound models dir only."""
    from vesta.inference import bind_models_dir, bind_on_model_ready
    from vesta.inference.download import DownloadModelJob

    payload = os.urandom(1024)
    server = _Server(payload)
    server.start()
    models_dir = tmp_path / "models"
    ready: list[Path] = []

    async def on_ready(path: object) -> None:
        ready.append(Path(path))  # type: ignore[arg-type]

    bind_models_dir(models_dir)
    bind_on_model_ready(on_ready)
    config.configure()  # the job resolves catalog.download.bandwidth_limit_kbps
    try:
        handle = JobHandleImpl(_FakeRunner(), job_id=1)  # type: ignore[arg-type]
        await DownloadModelJob().run(
            handle, {"url": f"{server.base_url}/m.gguf", "filename": "my.model"}
        )
    finally:
        bind_on_model_ready(None)
        bind_models_dir(None)
        config.reset_for_test()
        server.stop()

    final = models_dir / "my.model.gguf"
    assert final.is_file()
    assert final.read_bytes() == payload
    # The .part was atomically renamed away.
    assert not (models_dir / "my.model.gguf.part").exists()
    assert ready == [final]


async def test_model_resume_trims_stale_part_tail_beyond_checkpoint(
    tmp_path: Path,
) -> None:
    """A crash between a chunk write and its checkpoint leaves bytes past the
    checkpoint in the ``.part``; the resumed run must trim back to the
    checkpoint before appending (mirroring the ZIM downloader) so the final
    GGUF is byte-exact instead of poisoned by the stale tail (AUDIT_0824 I1)."""
    from vesta.inference import bind_models_dir, bind_on_model_ready
    from vesta.inference.download import DownloadModelJob

    payload = os.urandom(4096)
    server = _Server(payload)
    server.start()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    ready: list[Path] = []

    async def on_ready(path: object) -> None:
        ready.append(Path(path))  # type: ignore[arg-type]

    half = len(payload) // 2
    part = models_dir / "trim.gguf.part"
    part.write_bytes(payload[:half] + b"stale-tail-bytes-from-a-crashed-run")
    checkpoint = json.dumps(
        {"bytes_done": half, "size": len(payload), "url": f"{server.base_url}/trim.gguf"}
    )

    bind_models_dir(models_dir)
    bind_on_model_ready(on_ready)
    config.configure()  # the job resolves catalog.download.bandwidth_limit_kbps
    try:
        handle = JobHandleImpl(_FakeRunner(), job_id=1)  # type: ignore[arg-type]
        await DownloadModelJob().run(
            handle,
            {
                "url": f"{server.base_url}/trim.gguf",
                "filename": "trim",
                RESUME_CHECKPOINT_KEY: json.loads(checkpoint),
            },
        )
    finally:
        bind_on_model_ready(None)
        bind_models_dir(None)
        config.reset_for_test()
        server.stop()

    final = models_dir / "trim.gguf"
    assert final.is_file()
    assert final.read_bytes() == payload
    assert not part.exists()  # renamed away atomically
    assert ready == [final]


async def test_model_job_with_matching_sha256_verifies_and_completes(
    tmp_path: Path,
) -> None:
    """When sha256 is provided and matches the downloaded file, verify passes
    and the file is renamed."""
    from vesta.inference import bind_models_dir, bind_on_model_ready
    from vesta.inference.download import DownloadModelJob

    payload = os.urandom(2048)
    sha = hashlib.sha256(payload).hexdigest()
    server = _Server(payload)
    server.start()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    ready: list[Path] = []

    async def on_ready(path: object) -> None:
        ready.append(Path(path))  # type: ignore[arg-type]

    bind_models_dir(models_dir)
    bind_on_model_ready(on_ready)
    config.configure()
    try:
        handle = JobHandleImpl(_FakeRunner(), job_id=1)  # type: ignore[arg-type]
        await DownloadModelJob().run(
            handle,
            {
                "url": f"{server.base_url}/model.gguf",
                "filename": "verified",
                "sha256": sha,
                "size": len(payload),
            },
        )
    finally:
        bind_on_model_ready(None)
        bind_models_dir(None)
        config.reset_for_test()
        server.stop()

    final = models_dir / "verified.gguf"
    assert final.is_file()
    assert final.read_bytes() == payload
    assert not (models_dir / "verified.gguf.part").exists()
    assert ready == [final]


async def test_model_job_with_mismatched_sha256_discards_and_raises(
    tmp_path: Path,
) -> None:
    """When sha256 is provided and mismatches the downloaded file, download is discarded
    and DownloadModelError is raised."""
    from vesta.inference import bind_models_dir
    from vesta.inference.download import DownloadModelError, DownloadModelJob

    payload = os.urandom(2048)
    server = _Server(payload)
    server.start()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    bind_models_dir(models_dir)
    config.configure()
    try:
        handle = JobHandleImpl(_FakeRunner(), job_id=1)  # type: ignore[arg-type]
        with pytest.raises(DownloadModelError, match="checksum mismatch"):
            await DownloadModelJob().run(
                handle,
                {
                    "url": f"{server.base_url}/model.gguf",
                    "filename": "bad_sha",
                    "sha256": "0" * 64,
                    "size": len(payload),
                },
            )
    finally:
        bind_models_dir(None)
        config.reset_for_test()
        server.stop()

    assert not (models_dir / "bad_sha.gguf").exists()
    assert not (models_dir / "bad_sha.gguf.part").exists()


async def test_model_resume_when_server_ignores_range_restarts_from_zero(
    tmp_path: Path,
) -> None:
    """When resuming with bytes_done > 0, if the server ignores Range (returns 200),
    the downloader must restart from byte 0 and overwrite .part cleanly."""
    from vesta.inference import bind_models_dir, bind_on_model_ready
    from vesta.inference.download import DownloadModelJob

    payload = os.urandom(4096)
    server = _Server(payload, ignore_range=True)
    server.start()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    ready: list[Path] = []

    async def on_ready(path: object) -> None:
        ready.append(Path(path))  # type: ignore[arg-type]

    half = len(payload) // 2
    part = models_dir / "norange.gguf.part"
    part.write_bytes(payload[:half])
    checkpoint = json.dumps(
        {"bytes_done": half, "size": len(payload), "url": f"{server.base_url}/norange.gguf"}
    )

    bind_models_dir(models_dir)
    bind_on_model_ready(on_ready)
    config.configure()
    try:
        handle = JobHandleImpl(_FakeRunner(), job_id=1)  # type: ignore[arg-type]
        await DownloadModelJob().run(
            handle,
            {
                "url": f"{server.base_url}/norange.gguf",
                "filename": "norange",
                RESUME_CHECKPOINT_KEY: json.loads(checkpoint),
            },
        )
    finally:
        bind_on_model_ready(None)
        bind_models_dir(None)
        config.reset_for_test()
        server.stop()

    final = models_dir / "norange.gguf"
    assert final.is_file()
    assert final.read_bytes() == payload
    assert not part.exists()
    assert ready == [final]


async def test_model_resume_when_server_returns_mismatched_content_range(
    tmp_path: Path,
) -> None:
    """If server returns 206 with Content-Range start not matching requested start,
    fail with DownloadModelError."""
    from vesta.inference import bind_models_dir
    from vesta.inference.download import DownloadModelError, DownloadModelJob

    payload = os.urandom(4096)
    server_mismatch = _Server(payload, bad_content_range="bytes 50-4095/4096")
    server_mismatch.start()
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    half = len(payload) // 2
    part = models_dir / "bad_range.gguf.part"
    part.write_bytes(payload[:half])
    checkpoint = json.dumps(
        {
            "bytes_done": half,
            "size": len(payload),
            "url": f"{server_mismatch.base_url}/bad_range.gguf",
        }
    )

    bind_models_dir(models_dir)
    config.configure()
    try:
        handle = JobHandleImpl(_FakeRunner(), job_id=1)  # type: ignore[arg-type]
        with pytest.raises(DownloadModelError, match="server Content-Range start 50 != expected"):
            await DownloadModelJob().run(
                handle,
                {
                    "url": f"{server_mismatch.base_url}/bad_range.gguf",
                    "filename": "bad_range",
                    RESUME_CHECKPOINT_KEY: json.loads(checkpoint),
                },
            )
    finally:
        bind_models_dir(None)
        config.reset_for_test()
        server_mismatch.stop()


async def test_model_resume_when_part_shrank_restarts_from_zero(
    tmp_path: Path,
) -> None:
    """If the .part file on disk shrank below checkpoint bytes_done, remove it
    and restart clean from zero."""
    from vesta.inference import bind_models_dir, bind_on_model_ready
    from vesta.inference.download import DownloadModelJob

    payload = os.urandom(4096)
    server = _Server(payload)
    server.start()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    ready: list[Path] = []

    async def on_ready(path: object) -> None:
        ready.append(Path(path))  # type: ignore[arg-type]

    part = models_dir / "shrank.gguf.part"
    part.write_bytes(payload[:100])  # only 100 bytes on disk
    checkpoint = json.dumps(
        {"bytes_done": 2048, "size": len(payload), "url": f"{server.base_url}/shrank.gguf"}
    )

    bind_models_dir(models_dir)
    bind_on_model_ready(on_ready)
    config.configure()
    try:
        handle = JobHandleImpl(_FakeRunner(), job_id=1)  # type: ignore[arg-type]
        await DownloadModelJob().run(
            handle,
            {
                "url": f"{server.base_url}/shrank.gguf",
                "filename": "shrank",
                RESUME_CHECKPOINT_KEY: json.loads(checkpoint),
            },
        )
    finally:
        bind_on_model_ready(None)
        bind_models_dir(None)
        config.reset_for_test()
        server.stop()

    final = models_dir / "shrank.gguf"
    assert final.is_file()
    assert final.read_bytes() == payload
    assert not part.exists()
    assert ready == [final]


async def test_model_resume_when_checkpoint_mismatches_restarts_clean(
    tmp_path: Path,
) -> None:
    """If checkpoint recorded a different url, sha256, or size, restart from zero."""
    from vesta.inference import bind_models_dir, bind_on_model_ready
    from vesta.inference.download import DownloadModelJob

    payload = os.urandom(2048)
    sha = hashlib.sha256(payload).hexdigest()
    server = _Server(payload)
    server.start()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    ready: list[Path] = []

    async def on_ready(path: object) -> None:
        ready.append(Path(path))  # type: ignore[arg-type]

    part = models_dir / "mismatch.gguf.part"
    part.write_bytes(b"garbage-from-old-source")
    # Checkpoint has a different sha256
    checkpoint = json.dumps(
        {
            "bytes_done": 10,
            "size": len(payload),
            "sha256": "different_sha",
            "url": f"{server.base_url}/mismatch.gguf",
        }
    )

    bind_models_dir(models_dir)
    bind_on_model_ready(on_ready)
    config.configure()
    try:
        handle = JobHandleImpl(_FakeRunner(), job_id=1)  # type: ignore[arg-type]
        await DownloadModelJob().run(
            handle,
            {
                "url": f"{server.base_url}/mismatch.gguf",
                "filename": "mismatch",
                "sha256": sha,
                "size": len(payload),
                RESUME_CHECKPOINT_KEY: json.loads(checkpoint),
            },
        )
    finally:
        bind_on_model_ready(None)
        bind_models_dir(None)
        config.reset_for_test()
        server.stop()

    final = models_dir / "mismatch.gguf"
    assert final.is_file()
    assert final.read_bytes() == payload
    assert not part.exists()
    assert ready == [final]


# ── the download_zim sink (vesta.catalog.download, audit M2) ────────────────


async def test_job_rejects_hostile_user_names_before_touching_disk(
    zims_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /api/jobs forwards arbitrary params to any registered type, so the
    job itself must refuse anything that is not a bare ``*.zim`` basename —
    pathlib would let absolute paths win and ``..`` climb out."""
    canary = tmp_path / "canary.txt"
    canary.write_text("safe")
    monkeypatch.setenv("download.mirror_policy", "first")  # no metalink fetch
    for bad in ("/etc/passwd.zim", "../../escaped", "sub/dir/x", "..evil"):
        with pytest.raises(DownloadError, match="unsafe ZIM filename"):
            await _run_job(
                zims_dir,
                {"url": "https://example.com/x.zim.meta4", "name": bad},
            )
    assert canary.read_text() == "safe"
    assert list(zims_dir.iterdir()) == []


async def test_hostile_metalink_name_fails_job_and_writes_nothing(
    zims_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A metalink ``<file name>`` is REMOTE-controlled data: a traversal name
    there must fail the job cleanly with the offending value named — never
    silently rewritten, never written outside the zims dir."""
    payload = os.urandom(512)
    meta4 = (
        '<?xml version="1.0"?><metalink xmlns="urn:ietf:params:xml:ns:metalink">'
        '<file name="../../evil.zim">'
        '<url priority="1">{base_url}/payload</url></file></metalink>'
    )
    server = _Server(payload, meta4=meta4)
    server.start()
    monkeypatch.setenv("download.min_free_space_gb", "0")
    try:
        with pytest.raises(DownloadError, match="unsafe ZIM filename") as excinfo:
            await _run_job(
                zims_dir,
                {"url": f"{server.base_url}/x.zim.meta4", "name": "benign"},
            )
    finally:
        server.stop()
    # The error names the offending value.
    assert "../../evil.zim" in str(excinfo.value)
    # Nothing landed in the zims dir or escaped it.
    assert list(zims_dir.iterdir()) == []
    assert not list(tmp_path.rglob("evil*"))
    assert not list(tmp_path.rglob("*.part"))


async def test_metalink_name_downloads_to_expected_path(
    zims_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The benign path is unchanged: a well-formed suffix-less metalink name
    lands as ``<name>.zim`` (appended exactly once) inside the zims dir."""
    payload = os.urandom(1024)
    sha = hashlib.sha256(payload).hexdigest()
    registered: list[Path] = []

    async def register(path: object) -> None:
        registered.append(Path(path))  # type: ignore[arg-type]

    meta4 = (
        '<?xml version="1.0"?><metalink xmlns="urn:ietf:params:xml:ns:metalink">'
        '<file name="wikimed_en_all_maxi_2026-08">'
        f"<size>{len(payload)}</size>"
        f'<hash type="sha-256">{sha}</hash>'
        '<url priority="1">{base_url}/payload</url></file></metalink>'
    )
    server = _Server(payload, meta4=meta4)
    server.start()
    monkeypatch.setenv("download.min_free_space_gb", "0")
    try:
        await _run_job(
            zims_dir,
            {"url": f"{server.base_url}/x.zim.meta4", "name": "ignored_stem"},
            register=register,
        )
    finally:
        server.stop()

    final = zims_dir / "wikimed_en_all_maxi_2026-08.zim"  # suffix appended once
    assert final.read_bytes() == payload
    assert not (zims_dir / "wikimed_en_all_maxi_2026-08.zim.part").exists()
    assert registered == [final]


# ── mirror failover keeps the checkpoint authoritative (audit M5) ───────────


async def test_mirror_failover_after_mid_body_drop_is_byte_exact(
    zims_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resuming at ``bytes_done`` > 0, mirror #1 serves part of the requested
    range then drops the connection mid-body; mirror #2 must take over from
    the last checkpoint instead of appending its range after the dead
    attempt's leftover tail. The final file is byte-exact against the true
    content and passes the whole-file SHA-256 gate."""
    payload = os.urandom(3 * _CHUNK + 123)
    sha = hashlib.sha256(payload).hexdigest()
    half = _CHUNK + 777  # non-aligned resume offset inside the first chunk
    part = zims_dir / "failover.zim.part"
    part.write_bytes(payload[:half])
    checkpoint = json.dumps({"bytes_done": half, "size": len(payload), "sha256": sha, "url": ""})
    # Flaky mirror sends 1.5 chunks of the range then dies; httpx's chunker
    # flushes exactly one full _CHUNK (written + checkpointed) before the
    # truncation surfaces, so mirror #2 resumes at half + _CHUNK.
    flaky = _Server(payload, drop_after=_CHUNK + _CHUNK // 2)
    good = _Server(payload)
    meta4 = (
        '<?xml version="1.0"?><metalink xmlns="urn:ietf:params:xml:ns:metalink">'
        '<file name="failover.zim">'
        f"<size>{len(payload)}</size>"
        f'<hash type="sha-256">{sha}</hash>'
        f'<url priority="1">{flaky.base_url}/payload</url>'
        f'<url priority="2">{good.base_url}/payload</url>'
        "</file></metalink>"
    )
    good._handler_cls.serve_meta4 = meta4  # harness wiring: both mirror URLs are literal
    flaky.start()
    good.start()
    registered: list[Path] = []

    async def register(path: object) -> None:
        registered.append(Path(path))  # type: ignore[arg-type]

    monkeypatch.setenv("download.min_free_space_gb", "0")
    try:
        await _run_job(
            zims_dir,
            {"url": f"{good.base_url}/x.zim.meta4", "name": "ignored_stem"},
            register=register,
            resume_checkpoint=checkpoint,
        )
    finally:
        flaky.stop()
        good.stop()

    final = zims_dir / "failover.zim"
    assert final.read_bytes() == payload
    assert not (zims_dir / "failover.zim.part").exists()
    assert registered == [final]
    # Mirror #1 was asked for the checkpoint range and died mid-body; mirror
    # #2 was asked for exactly the bytes past what mirror #1 checkpointed.
    assert flaky.seen_starts == [half]
    assert good.seen_starts == [half + _CHUNK]


def _seed_stale_tail(
    zims_dir: Path, payload: bytes, sha: str | None, *, name: str = "stale"
) -> str:
    """Craft the post-failure state audit M5 describes: a ``<name>.zim.part``
    whose size runs past its checkpoint (a dead mirror's un-checkpointed
    tail), plus the resume checkpoint JSON the runner would hand back."""
    half = len(payload) // 2
    part = zims_dir / f"{name}.zim.part"
    part.write_bytes(payload[:half] + b"stale-tail-bytes-from-a-dead-mirror")
    return json.dumps({"bytes_done": half, "size": len(payload), "sha256": sha or "", "url": ""})


async def test_stale_part_tail_beyond_checkpoint_is_trimmed(
    zims_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A .part longer than its checkpoint must be truncated back to the
    checkpoint before resuming — appending after the stale tail would stitch
    the range data into the wrong offset. With checksums on, the pre-fix
    behavior fails the job here instead of producing a correct file."""
    payload = os.urandom(4096)
    sha = hashlib.sha256(payload).hexdigest()
    checkpoint = _seed_stale_tail(zims_dir, payload, sha)
    server = _Server(payload)
    server.start()
    monkeypatch.setenv("download.mirror_policy", "first")
    monkeypatch.setenv("download.min_free_space_gb", "0")
    try:
        await _run_job(
            zims_dir,
            {
                "url": f"{server.base_url}/file.zim",
                "name": "stale",
                "sha256": sha,
                "size": len(payload),
            },
            resume_checkpoint=checkpoint,
        )
    finally:
        server.stop()

    final = zims_dir / "stale.zim"
    assert final.read_bytes() == payload
    assert not (zims_dir / "stale.zim.part").exists()


async def test_stale_part_tail_without_checksums_is_trimmed(
    zims_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sha256=None path where silent corruption shipped pre-fix: no hash in
    the metalink and verification off means nothing downstream catches a
    mis-stitched resume — the trim itself is what guarantees byte-exactness."""
    payload = os.urandom(4096)
    meta4 = (
        '<?xml version="1.0"?><metalink xmlns="urn:ietf:params:xml:ns:metalink">'
        '<file name="stale_nohash.zim">'
        f"<size>{len(payload)}</size>"
        '<url priority="1">{base_url}/payload</url></file></metalink>'
    )
    server = _Server(payload, meta4=meta4)
    server.start()
    checkpoint = _seed_stale_tail(zims_dir, payload, None, name="stale_nohash")
    registered: list[Path] = []

    async def register(path: object) -> None:
        registered.append(Path(path))  # type: ignore[arg-type]

    monkeypatch.setenv("download.verify_checksums", "false")
    monkeypatch.setenv("download.min_free_space_gb", "0")
    try:
        await _run_job(
            zims_dir,
            {"url": f"{server.base_url}/x.zim.meta4", "name": "ignored_stem"},
            register=register,
            resume_checkpoint=checkpoint,
        )
    finally:
        server.stop()

    final = zims_dir / "stale_nohash.zim"
    assert final.read_bytes() == payload
    assert registered == [final]


class _PausingRunner(_FakeRunner):
    """Flips the cancel flag as soon as progress reaches ``cancel_at_done``,
    so the job raises CancelledError mid-download like a user pause."""

    def __init__(self, cancel_at_done: int) -> None:
        super().__init__()
        self.cancel_at_done = cancel_at_done

    def _is_cancelling(self, job_id: int) -> bool:
        done = self.progress[-1][0] if self.progress else 0
        return done >= self.cancel_at_done


async def test_pause_during_mirror_leaves_part_at_checkpoint_and_resumes(
    zims_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling during mirror #1 leaves a .part sized exactly at the last
    checkpoint, and a resumed run finishes byte-exact from that offset."""
    payload = os.urandom(3 * _CHUNK + 5)
    sha = hashlib.sha256(payload).hexdigest()
    server = _Server(payload)
    server.start()
    monkeypatch.setenv("download.mirror_policy", "first")
    monkeypatch.setenv("download.min_free_space_gb", "0")
    params: dict[str, object] = {
        "url": f"{server.base_url}/file.zim",
        "name": "pausable",
        "sha256": sha,
        "size": len(payload),
    }
    try:
        bind_runtime(db=None, zims_dir=str(zims_dir), register_archive=_noop_register)
        config.configure()
        runner = _PausingRunner(cancel_at_done=_CHUNK)
        handle = JobHandleImpl(runner, job_id=1)  # type: ignore[arg-type]
        task = asyncio.create_task(DownloadZimJob().run(handle, params))
        with pytest.raises(asyncio.CancelledError):
            await task

        checkpoint = json.loads(runner.checkpoint or "{}")
        part = zims_dir / "pausable.zim.part"
        assert part.exists()
        assert checkpoint["bytes_done"] == _CHUNK  # one chunk flushed before pause
        assert part.stat().st_size == checkpoint["bytes_done"]

        await _run_job(zims_dir, params, resume_checkpoint=runner.checkpoint)
    finally:
        server.stop()

    final = zims_dir / "pausable.zim"
    assert final.read_bytes() == payload
