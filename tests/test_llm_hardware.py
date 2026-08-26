"""Hardware detection plumbing for the token-economy gate.

Covers the supervisor's ``hw_class`` (``--list-devices`` parsing, timeout and
error degrade, one-shot caching, GPU banner override), the drained child
output, and the ``hardware`` field threaded through ``LlmTarget`` / ``LlmStatus``.
No real binary is ever spawned: the subprocess seam is monkeypatched throughout.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from vesta import config
from vesta.config.settings import SettingsSnapshot
from vesta.inference.local import LlamaServerSupervisor, _parse_list_devices
from vesta.inference.runtime import LlmRuntime, LlmStatus, LlmTarget


def make_sup(tmp_path: Path) -> LlamaServerSupervisor:
    return LlamaServerSupervisor(
        binary_path=str(tmp_path / "llama-server"),
        models_dir=tmp_path / "models",
        config_dir=tmp_path / "config",
    )


class FakeProbeProc:
    """The surface ``_probe_devices`` consumes from a spawned process."""

    def __init__(self, stdout: bytes, delay: float = 0.0) -> None:
        self._stdout = stdout
        self._delay = delay
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        await asyncio.sleep(self._delay)
        return self._stdout, b""

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        return 0


def patch_probe(
    monkeypatch: pytest.MonkeyPatch,
    proc: FakeProbeProc | None,
) -> list[list[str]]:
    """Patch ``asyncio.create_subprocess_exec`` inside vesta.inference.local.

    Returns the list of argv lists the supervisor tried to spawn. ``None``
    makes every spawn raise (binary unusable)."""

    calls: list[list[str]] = []

    async def fake_exec(*argv: str, **_kwargs: Any) -> FakeProbeProc:
        calls.append(list(argv))
        if proc is None:
            raise OSError("spawn failed")
        return proc

    monkeypatch.setattr("vesta.inference.local.asyncio.create_subprocess_exec", fake_exec)
    return calls


# ── --list-devices parsing ──────────────────────────────────────────────────


def test_parse_none_means_cpu() -> None:
    text = "Available devices:\n  (none)\n"
    assert _parse_list_devices(text) == "cpu"


def test_parse_device_lines_mean_gpu() -> None:
    text = "Available devices:\n  Vulkan0: Intel(R) Iris Xe Graphics\n"
    assert _parse_list_devices(text) == "gpu"


def test_parse_missing_header_or_empty_list_means_cpu() -> None:
    assert _parse_list_devices("some unrelated output\n") == "cpu"
    assert _parse_list_devices("Available devices:\n\n") == "cpu"


# ── hw_class over a monkeypatched subprocess ────────────────────────────────


async def test_hw_class_none_devices_is_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sup = make_sup(tmp_path)
    patch_probe(monkeypatch, FakeProbeProc(b"Available devices:\n  (none)\n"))
    assert await sup.hw_class() == "cpu"


async def test_hw_class_device_is_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sup = make_sup(tmp_path)
    patch_probe(monkeypatch, FakeProbeProc(b"Available devices:\n  Vulkan0: some gpu\n"))
    assert await sup.hw_class() == "gpu"


async def test_hw_class_spawn_error_degrades_to_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sup = make_sup(tmp_path)
    patch_probe(monkeypatch, None)
    assert await sup.hw_class() == "cpu"


async def test_hw_class_timeout_kills_and_degrades_to_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sup = make_sup(tmp_path)
    proc = FakeProbeProc(b"Available devices:\n  Vulkan0: gpu\n", delay=10.0)
    patch_probe(monkeypatch, proc)
    monkeypatch.setattr("vesta.inference.local._LIST_DEVICES_TIMEOUT_S", 0.05)
    assert await sup.hw_class() == "cpu"
    assert proc.killed


async def test_hw_class_probes_once_then_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sup = make_sup(tmp_path)
    calls = patch_probe(monkeypatch, FakeProbeProc(b"Available devices:\n  (none)\n"))
    assert await sup.hw_class() == "cpu"
    assert await sup.hw_class() == "cpu"
    assert len(calls) == 1
    assert calls[0] == [sup._binary_path, "--list-devices"]
    assert sup.hardware == "cpu"


# ── startup banner cross-check ──────────────────────────────────────────────


class FakeStreamReader:
    def __init__(self, lines: list[str]) -> None:
        self._lines = [line.encode() + b"\n" for line in lines]

    async def readline(self) -> bytes:
        return self._lines.pop(0) if self._lines else b""


class FakeDrainProc:
    def __init__(self, out: list[str], err: list[str]) -> None:
        self.stdout: FakeStreamReader | None = FakeStreamReader(out) if out else None
        self.stderr: FakeStreamReader | None = FakeStreamReader(err) if err else None


async def test_drain_records_gpu_banner_and_truncates_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sup = make_sup(tmp_path)
    long_line = "x" * 900
    proc = FakeDrainProc(
        out=["starting up", f"offloaded 33/35 layers to GPU | {long_line}"],
        err=["a warning"],
    )
    with caplog.at_level(logging.DEBUG, logger="vesta.inference.local"):
        await sup._drain_output(proc)  # type: ignore[arg-type]

    # Banner with n>0 ⇒ hardware recorded as gpu (cached, wins over probe).
    assert sup._hw_banner == "gpu"
    assert await sup.hw_class() == "gpu"
    logged = [
        getattr(r, "line", "") for r in caplog.records if r.getMessage() == "llama_server.output"
    ]
    assert len(logged) == 3
    assert all(len(line) <= 500 for line in logged)


async def test_banner_overrides_cached_cpu_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sup = make_sup(tmp_path)
    patch_probe(monkeypatch, FakeProbeProc(b"Available devices:\n  (none)\n"))
    assert await sup.hw_class() == "cpu"  # cached…
    proc = FakeDrainProc(out=["offloaded 16/33 layers to GPU"], err=[])
    await sup._drain_output(proc)  # type: ignore[arg-type]
    assert await sup.hw_class() == "gpu"  # …but the banner outranks it
    assert sup.hardware == "gpu"


async def test_zero_offload_banner_is_not_gpu(tmp_path: Path) -> None:
    sup = make_sup(tmp_path)
    sup._hw = "cpu"
    proc = FakeDrainProc(out=["offloaded 0/33 layers to GPU"], err=[])
    await sup._drain_output(proc)  # type: ignore[arg-type]
    assert sup._hw_banner is None
    assert await sup.hw_class() == "cpu"


async def test_stop_cancels_drain_task(tmp_path: Path) -> None:
    sup = make_sup(tmp_path)
    drain = asyncio.get_running_loop().create_future()

    async def hang() -> None:
        await drain

    sup._drain_task = asyncio.create_task(hang())
    await sup.stop()
    await asyncio.sleep(0)  # let the cancellation land
    assert sup._drain_task.cancelled()
    drain.cancel()


# ── LlmTarget / LlmRuntime threading ────────────────────────────────────────


def test_llm_target_hardware_defaults_to_none() -> None:
    target = LlmTarget(
        source="local",
        base_url="http://x/v1",
        api_key="local",
        model_id="m",
        enable_thinking=None,
    )
    assert target.hardware is None


class HwStubSupervisor:
    """The supervisor surface LlmRuntime consumes for these tests."""

    def __init__(self, hardware: str | None) -> None:
        self.base_url = "http://127.0.0.1:9999/v1"
        self.router_url = "http://127.0.0.1:9999"
        self.hardware = hardware

    async def hw_class(self) -> str:
        return self.hardware or "cpu"

    def is_running(self) -> bool:
        return False


def make_snapshot(**overrides: object) -> SettingsSnapshot:
    config.configure()
    values = dict(config.snapshot().values)
    values.update(overrides)
    return SettingsSnapshot(values=values)


LOCAL = {"inference.llm.source": "local", "inference.llm.model": "stub.gguf"}
REMOTE = {
    "inference.llm.source": "remote",
    "inference.llm.model": "remote-model",
    "inference.llm.endpoint_url": "http://remote.example:1234/v1",
    "inference.llm.api_key": "sk-x",
}


async def test_runtime_target_local_carries_hardware() -> None:
    runtime = LlmRuntime(
        supervisor=HwStubSupervisor("cpu"),  # type: ignore[arg-type]
        snapshot=make_snapshot(**LOCAL),
    )
    assert runtime.target().source == "local"
    assert runtime.target().hardware == "cpu"


async def test_runtime_target_remote_hardware_none() -> None:
    runtime = LlmRuntime(
        supervisor=HwStubSupervisor("cpu"),  # type: ignore[arg-type]
        snapshot=make_snapshot(**REMOTE),
    )
    assert runtime.target().source == "remote"
    assert runtime.target().hardware is None


async def test_runtime_status_local_reports_hardware() -> None:
    runtime = LlmRuntime(
        supervisor=HwStubSupervisor("gpu"),  # type: ignore[arg-type]
        snapshot=make_snapshot(**LOCAL),
    )
    status = await runtime.status()
    assert isinstance(status, LlmStatus)
    assert status.source == "local"
    assert status.hardware == "gpu"


async def test_runtime_status_remote_hardware_none() -> None:
    runtime = LlmRuntime(
        supervisor=HwStubSupervisor("gpu"),  # type: ignore[arg-type]
        snapshot=make_snapshot(**REMOTE),
    )
    status = await runtime.status()
    assert status.source == "remote"
    assert status.hardware is None


async def test_runtime_status_survives_hw_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenSup(HwStubSupervisor):
        async def hw_class(self) -> str:
            raise RuntimeError("probe exploded")

    runtime = LlmRuntime(
        supervisor=BrokenSup(None),  # type: ignore[arg-type]
        snapshot=make_snapshot(**LOCAL),
    )
    status = await runtime.status()  # must not raise
    assert status.hardware is None


# ── test isolation ──────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _reset_config() -> Iterator[None]:
    yield
    config.reset_for_test()
