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

import hashlib
import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from vesta import config
from vesta.catalog import bind_runtime
from vesta.catalog.download import (
    DownloadError,
    DownloadZimJob,
    parse_metalink,
)
from vesta.jobs.handle import JobHandleImpl
from vesta.jobs.types import RESUME_CHECKPOINT_KEY

# ── a tiny range-capable HTTP server (no external deps) ─────────────────────


class _RangeHandler(BaseHTTPRequestHandler):
    serve_bytes: bytes = b""  # set per server instance via subclass attribute

    def log_message(self, format: str, *args: object) -> None:  # silence
        pass

    def do_GET(self) -> None:
        data = self.serve_bytes
        total = len(data)
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            spec = range_header[len("bytes=") :].split("-")
            start = int(spec[0]) if spec[0] else 0
            end = int(spec[1]) if len(spec) > 1 and spec[1] else total - 1
            chunk = data[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(chunk)))
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(chunk)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(total))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(data)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _Server:
    """A threaded HTTP server serving fixed bytes with range support."""

    def __init__(self, data: bytes) -> None:
        self.data = data

        class Handler(_RangeHandler):
            pass

        Handler.serve_bytes = data
        self._srv = ThreadingHTTPServer(("127.0.0.1", _free_port()), Handler)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)

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
