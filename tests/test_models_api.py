"""The model management API and its seams.

Covers the full surface: the installed scan ignores ``<org>/<repo>/``
encoder directories and ``.part`` files; DELETE rejects traversal/absolute
paths/separators and asserts direct parenthood; deleting the active model
clears the setting and unloads; activate on a missing file is a 404;
fresh-install status is ``state="absent"`` with ``configured=false`` and zero
network; child-restart settings rebind the runtime via
``build_runtime_from_settings``; the D8 warm-on-download callback preloads
when ``inference.local.preload_on_ready`` is set.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from fixtures.llm_runtime import FakeLlmRuntime
from vesta.inference import get_runtime
from vesta.inference.runtime import LlmRuntimeError, LlmStatus

QWEN = "Qwen3.5-4B-Q4_K_S.gguf"
LFM = "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"


def _make_status(**over: object) -> LlmStatus:
    fields: dict[str, object] = {
        "source": "local",
        "configured": True,
        "installed": True,
        "state": "loaded",
        "model_file": "stub.gguf",
        "display_name": "Stub",
        "model_id": "stub",
        "size_bytes": 10,
        "context_size": 8192,
        "thinking": False,
        "thinking_supported": True,
        "idle_unload_seconds": 900,
        "seconds_since_last_use": None,
        "estimated_ram_bytes": 0,
        "error": None,
    }
    fields.update(over)
    return LlmStatus(**fields)  # type: ignore[arg-type]


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> FakeLlmRuntime:
    """Bind a stub runtime at the seam the models API and rebuilds resolve."""
    fake = FakeLlmRuntime()
    monkeypatch.setattr("vesta.inference.get_runtime", lambda: fake)
    return fake


# ── GET /api/models — the installed scan (D10) ──────────────────────────────


@pytest.mark.asyncio
async def test_list_models_scan_ignores_encoders_and_part_files(
    app_client: httpx.AsyncClient, tmp_path: Path
) -> None:
    models = tmp_path / "models"
    encoder = models / "unsloth" / "Qwen3.5-4B-GGUF"
    encoder.mkdir(parents=True)
    (encoder / "encoder.gguf").write_bytes(b"x" * 4)  # ONNX tree — not ours
    (models / "partial.gguf.part").write_bytes(b"x")  # cancelled download
    (models / QWEN).write_bytes(b"x" * 16)
    (models / LFM).write_bytes(b"x" * 8)
    (models / "my-model.gguf").write_bytes(b"x" * 4)

    resp = await app_client.get("/api/models")
    assert resp.status_code == 200
    body = resp.json()
    names = [e["filename"] for e in body["installed"]]
    assert names == sorted([LFM, "my-model.gguf", QWEN])
    by_name = {e["filename"]: e for e in body["installed"]}

    qwen = by_name[QWEN]
    assert qwen["preset_id"] == "qwen3.5-4b-q4_k_s"
    assert qwen["display_name"] == "Qwen3.5 4B (Q4_K_S)"
    assert qwen["thinking_supported"] is True
    assert qwen["size_bytes"] == 16
    assert qwen["is_active"] is False

    lfm = by_name[LFM]
    assert lfm["preset_id"] is None
    assert lfm["thinking_supported"] is False  # thinking="never" from filename heuristic

    mine = by_name["my-model.gguf"]
    assert mine["preset_id"] is None
    assert mine["display_name"] == "my model"  # prettified stem
    assert mine["thinking_supported"] is True

    # The wizard wire shape rides along unchanged.
    assert any(p["filename"] == QWEN for p in body["presets"])
    assert body["status"]["source"] == "local"


@pytest.mark.asyncio
async def test_list_models_marks_active_entry(
    app_client: httpx.AsyncClient, tmp_path: Path, fake_runtime: FakeLlmRuntime
) -> None:
    models = tmp_path / "models"
    models.mkdir(parents=True)
    (models / QWEN).write_bytes(b"x" * 16)
    (models / "other.gguf").write_bytes(b"x" * 4)
    resp = await app_client.post("/api/models/activate", json={"filename": QWEN})
    assert resp.status_code == 200

    listed = await app_client.get("/api/models")
    by_name = {e["filename"]: e for e in listed.json()["installed"]}
    assert by_name[QWEN]["is_active"] is True
    assert by_name["other.gguf"]["is_active"] is False


# ── GET /api/models/status (D9) ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_status_fresh_install_is_absent_and_offline(
    app_client: httpx.AsyncClient,
) -> None:
    """Fresh install: absent, unconfigured, and provably no network — the
    runtime never even lazily created its router client."""
    resp = await app_client.get("/api/models/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "absent"
    assert body["configured"] is False
    assert body["installed"] is False
    assert body["source"] == "local"
    runtime = get_runtime()
    assert runtime is not None
    assert runtime._http is None  # proof no HTTP client was made


# ── POST /api/models/activate ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_activate_missing_file_is_404(app_client: httpx.AsyncClient, tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    resp = await app_client.post("/api/models/activate", json={"filename": "nope.gguf"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_activate_sets_model_and_rebuilds(
    app_client: httpx.AsyncClient, tmp_path: Path, fake_runtime: FakeLlmRuntime
) -> None:
    models = tmp_path / "models"
    models.mkdir(parents=True)
    (models / QWEN).write_bytes(b"x" * 16)
    resp = await app_client.post("/api/models/activate", json={"filename": QWEN})
    assert resp.status_code == 200

    got = await app_client.get("/api/settings")
    assert got.json()["values"]["inference.llm.model"] == QWEN
    assert len(fake_runtime.rebuild_snapshots) == 1
    from vesta.inference import INFERENCE_LLM_MODEL

    assert str(fake_runtime.rebuild_snapshots[0].get(INFERENCE_LLM_MODEL)) == QWEN


@pytest.mark.asyncio
async def test_activate_rejects_traversal_names(
    app_client: httpx.AsyncClient,
) -> None:
    for bad in ("../evil.gguf", "a/b.gguf", "/abs.gguf", "bare", ".gguf"):
        resp = await app_client.post("/api/models/activate", json={"filename": bad})
        assert resp.status_code == 400, bad


# ── POST /api/models/load and /unload ────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_returns_loaded_status(
    app_client: httpx.AsyncClient, fake_runtime: FakeLlmRuntime
) -> None:
    fake_runtime.status_value = _make_status(state="loaded")
    resp = await app_client.post("/api/models/load")
    assert resp.status_code == 200
    assert resp.json()["state"] == "loaded"
    assert fake_runtime.load_calls == 1


@pytest.mark.asyncio
async def test_load_error_returns_error_status(
    app_client: httpx.AsyncClient, fake_runtime: FakeLlmRuntime
) -> None:
    fake_runtime.error = LlmRuntimeError("no model configured")
    resp = await app_client.post("/api/models/load")
    assert resp.status_code == 200  # blocks until loaded OR errors (D10)
    body = resp.json()
    assert body["state"] == "error"
    assert "no model configured" in body["error"]


@pytest.mark.asyncio
async def test_unload_calls_runtime(
    app_client: httpx.AsyncClient, fake_runtime: FakeLlmRuntime
) -> None:
    resp = await app_client.post("/api/models/unload")
    assert resp.status_code == 200
    assert resp.json()["state"] == "unloaded"
    assert fake_runtime.unload_calls == 1


# ── DELETE /api/models/{filename} ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_rejects_traversal_absolute_and_separators(
    app_client: httpx.AsyncClient, tmp_path: Path
) -> None:
    """No filename may escape the models dir — a routed 404 is as safe as a
    400; the canary file proves nothing outside was touched."""
    models = tmp_path / "models"
    models.mkdir()
    canary = tmp_path / "canary.txt"
    canary.write_text("safe")
    attempts = [
        "..%2Fevil.gguf",  # ../evil.gguf
        "..%2F..%2Fcanary.txt.gguf",
        "a%2Fb.gguf",  # a/b.gguf — separator in the name
        "%2Fetc%2Fpasswd.gguf",  # absolute /etc/passwd.gguf
        "..evil.gguf",  # contains ..
        "a%5Cb.gguf",  # backslash separator
        "plain",  # not a .gguf
        "..",  # the models dir itself
    ]
    for attempt in attempts:
        resp = await app_client.delete(f"/api/models/{attempt}")
        # 405 = the router refused the malformed path before our handler; a
        # routed 404 (decoded separators split the path) and our 400 are the
        # other safe rejections.
        assert resp.status_code in {400, 404, 405}, (attempt, resp.status_code)
    assert canary.read_text() == "safe"
    assert list(models.iterdir()) == []


@pytest.mark.asyncio
async def test_delete_rejects_symlink_escape(app_client: httpx.AsyncClient, tmp_path: Path) -> None:
    """A top-level symlink named like a GGUF must not resolve outside — the
    parenthood assert (resolve + compare) is the guard."""
    models = tmp_path / "models"
    models.mkdir()
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"x" * 4)
    (models / "link.gguf").symlink_to(outside)

    resp = await app_client.delete("/api/models/link.gguf")
    assert resp.status_code == 400
    assert outside.is_file()


@pytest.mark.asyncio
async def test_delete_missing_file_is_404(app_client: httpx.AsyncClient, tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    resp = await app_client.delete("/api/models/ghost.gguf")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_active_model_clears_setting_and_unloads(
    app_client: httpx.AsyncClient, tmp_path: Path, fake_runtime: FakeLlmRuntime
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "active.gguf").write_bytes(b"x" * 4)
    activate = await app_client.post("/api/models/activate", json={"filename": "active.gguf"})
    assert activate.status_code == 200
    assert fake_runtime.rebuild_snapshots  # activate rebuilt

    resp = await app_client.delete("/api/models/active.gguf")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == "active.gguf"
    assert not (models / "active.gguf").exists()

    got = await app_client.get("/api/settings")
    assert got.json()["values"]["inference.llm.model"] == ""
    assert fake_runtime.unload_calls == 1
    # activate + delete-clear both rebuilt
    assert len(fake_runtime.rebuild_snapshots) == 2


@pytest.mark.asyncio
async def test_delete_inactive_model_leaves_setting_alone(
    app_client: httpx.AsyncClient, tmp_path: Path, fake_runtime: FakeLlmRuntime
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "active.gguf").write_bytes(b"x" * 4)
    (models / "spare.gguf").write_bytes(b"x" * 4)
    await app_client.post("/api/models/activate", json={"filename": "active.gguf"})
    rebuilds_before = len(fake_runtime.rebuild_snapshots)

    resp = await app_client.delete("/api/models/spare.gguf")
    assert resp.status_code == 200
    got = await app_client.get("/api/settings")
    assert got.json()["values"]["inference.llm.model"] == "active.gguf"
    assert fake_runtime.unload_calls == 0
    assert len(fake_runtime.rebuild_snapshots) == rebuilds_before
    assert (models / "active.gguf").is_file()


# ── D7: child-restart settings rebind the runtime ────────────────────────────


@pytest.mark.asyncio
async def test_context_size_change_rebinds_runtime(
    app_client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child-restart setting triggers a fresh runtime built from settings
    (supervisor restart — deliberately not a rebuild() in place)."""
    from vesta.inference import build_runtime_from_settings as real_build

    builds: list[object] = []

    def _recording_build(*args: object, **kwargs: object) -> object:
        builds.append(kwargs)
        return real_build(*args, **kwargs)  # type: ignore[arg-type,return-value]

    monkeypatch.setattr("vesta.inference.build_runtime_from_settings", _recording_build)

    before = get_runtime()
    assert before is not None
    resp = await app_client.put(
        "/api/settings", json={"values": {"inference.local.context_size": "4096"}}
    )
    assert resp.status_code == 200

    assert len(builds) == 1
    after = get_runtime()
    assert after is not None and after is not before
    from vesta.inference import INFERENCE_LOCAL_CONTEXT_SIZE

    assert int(after.snapshot.get(INFERENCE_LOCAL_CONTEXT_SIZE)) == 4096
    # The rebind carried the running watchdog over.
    assert after.watchdog_running is True


@pytest.mark.asyncio
async def test_non_restart_inference_change_does_not_rebind(
    app_client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _recording_build(*args: object, **kwargs: object) -> object:
        raise AssertionError("non-restart change must not rebind the runtime")

    monkeypatch.setattr("vesta.inference.build_runtime_from_settings", _recording_build)

    before = get_runtime()
    assert before is not None
    resp = await app_client.put(
        "/api/settings", json={"values": {"inference.llm.enable_thinking": "true"}}
    )
    assert resp.status_code == 200
    assert get_runtime() is before


@pytest.mark.asyncio
async def test_idle_unload_presence_flip_rebinds(
    app_client: httpx.AsyncClient,
) -> None:
    """900 → 0 removes ``--sleep-idle-seconds`` from the command line — the
    flag's presence changed, so the child restarts (D7 row 4)."""
    before = get_runtime()
    resp = await app_client.put(
        "/api/settings", json={"values": {"inference.local.idle_unload_seconds": "0"}}
    )
    assert resp.status_code == 200
    after = get_runtime()
    assert after is not None and before is not None and after is not before


# ── D8: warm on download ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_model_ready_rebuilds_and_preloads(
    app_client: httpx.AsyncClient, fake_runtime: FakeLlmRuntime
) -> None:
    from vesta.inference import notify_model_ready

    await notify_model_ready(Path("/data/models/new.gguf"))
    assert len(fake_runtime.rebuild_snapshots) == 1
    assert fake_runtime.ready_calls == 1  # preload_on_ready defaults to true


@pytest.mark.asyncio
async def test_model_ready_skips_preload_when_disabled(
    app_client: httpx.AsyncClient, fake_runtime: FakeLlmRuntime
) -> None:
    resp = await app_client.put(
        "/api/settings", json={"values": {"inference.local.preload_on_ready": "false"}}
    )
    assert resp.status_code == 200
    from vesta.inference import notify_model_ready

    await notify_model_ready(Path("/data/models/new.gguf"))
    assert len(fake_runtime.rebuild_snapshots) == 2  # settings write + callback
    assert fake_runtime.ready_calls == 0  # …but no eager load


def test_production_imports_register_download_model_job() -> None:
    """B2 E2E regression: ``POST /api/models/download`` 500s with "unknown job
    type 'download_model'" unless ``vesta.main`` (the production import graph)
    pulls in ``vesta.inference.download`` for its side-effect registration."""
    import vesta.main  # noqa: F401
    from vesta.jobs.types import JOB_TYPES

    assert "download_model" in JOB_TYPES, sorted(JOB_TYPES)
