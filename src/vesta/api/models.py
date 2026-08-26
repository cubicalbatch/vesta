"""The model management API: presets, download, and the D10 surface.

The model presets are shipped data (no network needed to list them). The
download endpoint submits a ``download_model`` job and immediately writes the
inference settings so ``source=local`` and ``model=<filename>`` are persisted
before the job even finishes. The management endpoints — installed scan,
status, activate/load/unload, delete — drive the bound ``LlmRuntime``; the
wire shapes of ``GET /api/models/presets`` and ``POST /api/models/download``
are frozen (the shipped wizard depends on them).
"""

import contextlib
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from vesta import config
from vesta.api.settings import persist_settings_and_reload
from vesta.api.state import AppState, app_state
from vesta.inference.models import (
    display_name_for,
    model_presets,
    preset_by_filename,
    thinking_for_filename,
)

router = APIRouter(tags=["models"])


class ModelPresetOut(BaseModel):
    id: str
    display_name: str
    url: str
    filename: str
    size_bytes: int
    min_ram_gb: float
    description: str
    # Whether the GGUF already exists under data/models/ — the wizard shows a
    # "Downloaded" check instead of a Download button when true. Computed per
    # request (not cached) so a download that just finished is reflected on the
    # next presets fetch.
    downloaded: bool


class PresetsResponse(BaseModel):
    presets: list[ModelPresetOut]


class ModelDownloadRequest(BaseModel):
    """Submit a GGUF download. Either a known preset id or a raw URL+filename."""

    preset_id: str | None = None
    url: str | None = None
    filename: str | None = None


class ModelDownloadResponse(BaseModel):
    job_id: int
    job_type: str
    target: str | None = None
    model_filename: str


class LlmStatusOut(BaseModel):
    """``LlmStatus`` on the wire — the UI polls this."""

    source: str
    configured: bool
    installed: bool
    state: str
    model_file: str | None
    display_name: str | None
    model_id: str | None
    size_bytes: int
    context_size: int
    thinking: bool
    thinking_supported: bool
    idle_unload_seconds: int
    seconds_since_last_use: float | None
    estimated_ram_bytes: int
    error: str | None


class InstalledModel(BaseModel):
    """One top-level GGUF under the models dir."""

    filename: str
    size_bytes: int
    display_name: str
    is_active: bool
    preset_id: str | None
    thinking_supported: bool


class ModelsResponse(BaseModel):
    installed: list[InstalledModel]
    presets: list[ModelPresetOut]
    status: LlmStatusOut


class ActivateRequest(BaseModel):
    filename: str


def _safe_gguf_name(filename: str, *, append_suffix: bool = False) -> str:
    """Validate a bare ``*.gguf`` filename that cannot escape the models dir.

    Thin HTTP translation of the shared sink guard
    (:func:`vesta.inference.download.safe_gguf_basename`) — one predicate for
    activate/delete/download so they cannot drift apart.
    """
    from vesta.inference.download import safe_gguf_basename

    try:
        return safe_gguf_basename(filename, append_suffix=append_suffix)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid model filename") from exc


def _active_model_name() -> str:
    """The configured model's basename (``inference.llm.model`` stores one)."""
    from vesta.inference import INFERENCE_LLM_MODEL

    return Path(str(config.get(INFERENCE_LLM_MODEL))).name


def _scan_installed(models_dir: Path, active_name: str) -> list[InstalledModel]:
    """Top-level ``*.gguf`` files only.

    The ONNX encoders live in ``<org>/<repo>/`` subdirectories and a cancelled
    download leaves ``<name>.gguf.part`` — neither is an installed chat model.
    """
    entries: list[InstalledModel] = []
    for path in sorted(models_dir.glob("*.gguf")):
        if not path.is_file():
            continue
        preset = preset_by_filename(path.name)
        entries.append(
            InstalledModel(
                filename=path.name,
                size_bytes=path.stat().st_size,
                display_name=display_name_for(path.name),
                is_active=path.name == active_name,
                preset_id=preset.id if preset is not None else None,
                thinking_supported=thinking_for_filename(path.name) == "toggle",
            )
        )
    return entries


async def _current_status() -> dict[str, object]:
    """The bound runtime's ``LlmStatus`` as a JSON-ready dict (503 unbound)."""
    from vesta.inference import get_runtime

    runtime = get_runtime()
    if runtime is None:
        raise HTTPException(status_code=503, detail="LLM runtime not ready")
    return asdict(await runtime.status())


@router.get("/api/models/presets", response_model=PresetsResponse)
async def list_presets() -> dict[str, object]:
    """The shipped GGUF preset list — works with no network.

    Each preset carries a ``downloaded`` flag computed from the models dir on
    disk, so the wizard can show "Downloaded" instead of a Download button for
    a GGUF that's already present (re-download is a no-op the user shouldn't
    be offered).
    """
    from vesta.inference import get_models_dir

    models_dir = get_models_dir()
    return {
        "presets": [
            ModelPresetOut(
                id=p.id,
                display_name=p.display_name,
                url=p.url,
                filename=p.filename,
                size_bytes=p.size_bytes,
                min_ram_gb=p.min_ram_gb,
                description=p.description,
                downloaded=models_dir is not None and (models_dir / p.filename).is_file(),
            ).model_dump()
            for p in model_presets()
        ]
    }


@router.post("/api/models/download", response_model=ModelDownloadResponse)
async def download_model(
    body: ModelDownloadRequest, state: AppState = Depends(app_state)
) -> dict[str, object]:
    """Enqueue a GGUF download and configure the local inference source.

    Accepts either a ``preset_id`` (resolves the URL + filename from the shipped
    list) or a raw ``url`` + ``filename`` for a custom GGUF. Immediately writes
    ``inference.llm.source=local`` and ``inference.llm.model=<filename>`` so the
    settings are persisted before the download finishes.

    The actual llama-server serving is handled separately (the supervisor reads
    these settings on the next gateway build).
    """
    if state.runner is None:
        raise HTTPException(status_code=503, detail="job runner not ready")

    url = body.url
    filename = body.filename

    if body.preset_id:
        from vesta.inference.models import preset_by_id

        preset = preset_by_id(body.preset_id)
        if preset is None:
            raise HTTPException(status_code=404, detail=f"unknown preset {body.preset_id!r}")
        url = url or preset.url
        filename = filename or preset.filename

    if not url:
        raise HTTPException(status_code=400, detail="url or preset_id required")
    if not filename:
        raise HTTPException(status_code=400, detail="filename required (or supply preset_id)")

    # Validate BEFORE anything persists or is submitted — a traversal name
    # here used to land in inference.llm.model and the job's write path.
    filename = _safe_gguf_name(filename, append_suffix=True)

    # Write inference settings immediately — the job just downloads the file.
    # Reload the config resolver so the new values are visible to GET /api/settings
    # and the capability probe (the shared post-write reload).
    await persist_settings_and_reload(
        state.db, [("inference.llm.source", "local"), ("inference.llm.model", filename)]
    )
    # The writes above changed inference.* keys — rebuild the LLM
    # runtime so the new model file is picked up (drops the cached router id;
    # restarts a running child). Non-fatal: the download proceeds regardless,
    # and the D8 post-download callback rebuilds again once the file lands.
    from vesta.inference import rebuild_runtime

    await rebuild_runtime()

    job_id = await state.runner.submit(
        "download_model",
        target=filename,
        params={"url": url, "filename": filename},
    )
    return ModelDownloadResponse(
        job_id=job_id,
        job_type="download_model",
        target=filename,
        model_filename=filename,
    ).model_dump()


@router.get("/api/models", response_model=ModelsResponse)
async def list_models() -> dict[str, object]:
    """Installed GGUFs + presets + runtime status — the models page's one call."""
    from vesta.inference import get_models_dir

    models_dir = get_models_dir()
    installed = _scan_installed(models_dir, _active_model_name()) if models_dir is not None else []
    presets = [
        ModelPresetOut(
            id=p.id,
            display_name=p.display_name,
            url=p.url,
            filename=p.filename,
            size_bytes=p.size_bytes,
            min_ram_gb=p.min_ram_gb,
            description=p.description,
            downloaded=models_dir is not None and (models_dir / p.filename).is_file(),
        ).model_dump()
        for p in model_presets()
    ]
    return {
        "installed": [e.model_dump() for e in installed],
        "presets": presets,
        "status": await _current_status(),
    }


@router.get("/api/models/status", response_model=LlmStatusOut)
async def model_status() -> dict[str, object]:
    """``LlmStatus`` as JSON. Read-only — never stamps ``last_used``
    and never wakes a sleeping model (the status is resolved from local state
    and the exempt router surface only)."""
    return await _current_status()


@router.post("/api/models/activate", response_model=LlmStatusOut)
async def activate_model(
    body: ActivateRequest, state: AppState = Depends(app_state)
) -> dict[str, object]:
    """Select the active GGUF: set ``inference.llm.model`` and rebuild (D10).

    The rebuild drops the runtime's cached router id and restarts a running
    child so the router discovers the file; the model loads lazily on the
    next question (or an explicit ``POST /api/models/load``).
    """
    from vesta.inference import get_models_dir

    filename = _safe_gguf_name(body.filename)
    models_dir = get_models_dir()
    if models_dir is None:
        raise HTTPException(status_code=503, detail="models directory not ready")
    if not (models_dir / filename).is_file():
        raise HTTPException(status_code=404, detail=f"model file not found: {filename}")

    await persist_settings_and_reload(state.db, [("inference.llm.model", filename)])
    from vesta.inference import rebuild_runtime

    await rebuild_runtime()
    return await _current_status()


@router.post("/api/models/load", response_model=LlmStatusOut)
async def load_model() -> dict[str, object]:
    """Explicitly load the active model — blocks until loaded or errors (D10)."""
    from vesta.inference import get_runtime
    from vesta.inference.local import BinaryMissing, LlamaServerError
    from vesta.inference.runtime import LlmRuntimeError

    runtime = get_runtime()
    if runtime is None:
        raise HTTPException(status_code=503, detail="LLM runtime not ready")
    try:
        await runtime.load()
    except (LlmRuntimeError, BinaryMissing, LlamaServerError) as exc:
        body = await _current_status()
        body["state"] = "error"
        body["error"] = str(exc)
        return body
    return await _current_status()


@router.post("/api/models/unload", response_model=LlmStatusOut)
async def unload_model() -> dict[str, object]:
    """Explicitly unload the model — frees the weights all the way (D10, F7)."""
    from vesta.inference import get_runtime

    runtime = get_runtime()
    if runtime is None:
        raise HTTPException(status_code=503, detail="LLM runtime not ready")
    await runtime.unload()
    return await _current_status()


@router.delete("/api/models/{filename}")
async def delete_model(filename: str, state: AppState = Depends(app_state)) -> dict[str, object]:
    """Remove a GGUF from disk (D10).

    Deleting the active model clears ``inference.llm.model`` and unloads
    first — the router may hold the file open, but the setting must not point
    at a deleted file.
    """
    from vesta.inference import get_models_dir, get_runtime, rebuild_runtime

    name = _safe_gguf_name(filename)
    models_dir = get_models_dir()
    if models_dir is None:
        raise HTTPException(status_code=503, detail="models directory not ready")
    # Path-traversal guard: resolve and assert direct parenthood — the
    # models dir also holds the baked encoder symlinks.
    path = (models_dir / name).resolve()
    if path.parent != models_dir.resolve():
        raise HTTPException(status_code=400, detail="invalid model filename")
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"model file not found: {name}")

    active = _active_model_name() == name
    runtime = get_runtime()
    if active and runtime is not None:
        with contextlib.suppress(Exception):
            await runtime.unload()
    path.unlink()

    if active:
        await persist_settings_and_reload(state.db, [("inference.llm.model", "")])
        await rebuild_runtime()
    return {"deleted": name}


__all__ = ["router"]
