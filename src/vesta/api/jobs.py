"""Job endpoints: list, create, inspect, control, and SSE progress streams.

SSE is hand-formatted so we add no
extra dependency. A reconnecting client always gets a snapshot first, then live
``progress``/``status`` events; a terminal status closes the per-job stream.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from vesta.api.state import AppState, app_state
from vesta.jobs.types import job_types

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class CreateJobRequest(BaseModel):
    type: str
    target: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


@router.get("")
async def list_jobs(state: AppState = Depends(app_state)) -> dict[str, object]:
    jobs = await state.runner.list_jobs()
    return {"jobs": [j.to_dict() for j in jobs]}


@router.get("/stream")
async def stream_all_jobs(
    state: AppState = Depends(app_state),
) -> StreamingResponse:
    """Global job-status SSE stream (for the header indicator)."""

    async def gen() -> AsyncIterator[bytes]:
        async for event in state.runner.stream_all():
            yield _sse(event)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("")
async def create_job(
    req: CreateJobRequest, state: AppState = Depends(app_state)
) -> dict[str, object]:
    if req.type not in job_types():
        raise HTTPException(status_code=400, detail=f"unknown job type {req.type!r}")
    _validate_job_params(req.type, req.params)
    job_id = await state.runner.submit(req.type, req.target, req.params)
    return {"id": job_id}


def _reject_unknown_keys(jtype: str, params: dict[str, Any], known: set[str]) -> None:
    extra = set(params) - known
    if extra:
        raise HTTPException(status_code=400, detail=f"unexpected {jtype} params: {sorted(extra)}")


def _require_non_empty_str(jtype: str, key: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise HTTPException(
            status_code=400, detail=f"{jtype} requires {key} to be a non-empty string"
        )


def _optional_str(jtype: str, key: str, value: Any) -> None:
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail=f"{jtype} requires {key} to be a string")


def _int_value(value: Any) -> int | None:
    """The value as an int, or ``None`` if it isn't one (bools are not ints)."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _validate_download_model_params(params: dict[str, Any]) -> None:
    """Per-type param gate for ``download_model`` submissions.

    The runner accepts arbitrary params for any registered type; without this,
    ``POST /api/jobs`` would bypass the download endpoint's filename hygiene
    entirely. Only the two known keys are accepted and the filename must pass
    the same guard the endpoint uses — the job still re-validates at the sink.
    """
    from vesta.inference.download import safe_gguf_basename

    _reject_unknown_keys("download_model", params, {"url", "filename"})
    _require_non_empty_str("download_model", "url", params.get("url"))
    filename = params.get("filename")
    if not isinstance(filename, str) or not filename:
        raise HTTPException(status_code=400, detail="download_model requires a filename")
    try:
        safe_gguf_basename(filename, append_suffix=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid model filename") from exc


def _validate_download_zim_params(params: dict[str, Any]) -> None:
    """Known keys only; ``url`` required, optional strings typed, ``size >= 0``,
    and an explicit ``name`` must pass the same guard ``POST /api/zims/download``
    applies — the job still re-validates at the sink."""
    from vesta.catalog.download import safe_zim_basename

    _reject_unknown_keys("download_zim", params, {"url", "name", "title", "sha256", "size"})
    _require_non_empty_str("download_zim", "url", params.get("url"))
    for key in ("name", "title", "sha256"):
        if key in params:
            _optional_str("download_zim", key, params[key])
    size = _int_value(params.get("size", 0))
    if size is None or size < 0:
        raise HTTPException(status_code=400, detail="download_zim size must be an int >= 0")
    name = params.get("name")
    if name:
        try:
            safe_zim_basename(name, append_suffix=True)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid zim filename") from exc


def _validate_refresh_catalog_params(params: dict[str, Any]) -> None:
    """Only ``url`` is consumed (an OPDS feed override); anything else is refused
    rather than passed through verbatim into a server-side fetch."""
    _reject_unknown_keys("refresh_catalog", params, {"url"})
    if "url" in params:
        _require_non_empty_str("refresh_catalog", "url", params["url"])


def _validate_index_zim_params(params: dict[str, Any]) -> None:
    _reject_unknown_keys("index_zim", params, {"zim_id", "depth"})
    if _int_value(params.get("zim_id")) is None:
        raise HTTPException(status_code=400, detail="index_zim requires an integer zim_id")
    depth = _int_value(params.get("depth", 1))
    if depth is None or not 1 <= depth <= 3:
        raise HTTPException(status_code=400, detail="index_zim depth must be an int in 1..3")


def _validate_noop_params(params: dict[str, Any]) -> None:
    _reject_unknown_keys("noop", params, {"total", "delay"})
    total = _int_value(params.get("total", 10))
    if total is None or total <= 0:
        raise HTTPException(status_code=400, detail="noop total must be an int > 0")
    delay = params.get("delay", 0.05)
    if isinstance(delay, bool) or not isinstance(delay, (int, float)) or delay < 0:
        raise HTTPException(status_code=400, detail="noop delay must be a number >= 0")


# Per-type param gates for POST /api/jobs. A registered type without an entry
# fails closed: params are refused until its schema is declared here.
_PARAM_VALIDATORS: dict[str, Callable[[dict[str, Any]], None]] = {
    "download_model": _validate_download_model_params,
    "download_zim": _validate_download_zim_params,
    "refresh_catalog": _validate_refresh_catalog_params,
    "index_zim": _validate_index_zim_params,
    "noop": _validate_noop_params,
}


def _validate_job_params(jtype: str, params: dict[str, Any]) -> None:
    validator = _PARAM_VALIDATORS.get(jtype)
    if validator is None:
        raise HTTPException(status_code=400, detail=f"job type {jtype!r} accepts no params")
    validator(params)


@router.get("/{job_id}")
async def get_job(job_id: int, state: AppState = Depends(app_state)) -> dict[str, object]:
    record = await state.runner.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job not found")
    return record.to_dict()


@router.get("/{job_id}/stream")
async def stream_job(job_id: int, state: AppState = Depends(app_state)) -> StreamingResponse:
    record = await state.runner.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="job not found")

    async def gen() -> AsyncIterator[bytes]:
        async for event in state.runner.stream(job_id):
            yield _sse(event)

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/{job_id}/pause")
async def pause_job(job_id: int, state: AppState = Depends(app_state)) -> dict[str, object]:
    ok = await state.runner.pause(job_id)
    if not ok:
        raise HTTPException(status_code=409, detail="job is not running")
    return {"id": job_id, "status": "paused"}


@router.post("/{job_id}/resume")
async def resume_job(job_id: int, state: AppState = Depends(app_state)) -> dict[str, object]:
    ok = await state.runner.resume(job_id)
    if not ok:
        raise HTTPException(status_code=409, detail="job is not paused")
    return {"id": job_id, "status": "queued"}


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: int, state: AppState = Depends(app_state)) -> dict[str, object]:
    ok = await state.runner.cancel(job_id)
    if not ok:
        raise HTTPException(status_code=409, detail="job already terminal")
    return {"id": job_id, "status": "cancelled"}


def _sse(event: dict[str, Any]) -> bytes:
    name = str(event.get("event", "message"))
    data = json.dumps(event.get("data"), default=str, separators=(",", ":"))
    return f"event: {name}\ndata: {data}\n\n".encode()
