"""Job endpoints: list, create, inspect, control, and SSE progress streams.

SSE is hand-formatted so we add no
extra dependency. A reconnecting client always gets a snapshot first, then live
``progress``/``status`` events; a terminal status closes the per-job stream.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
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
    if req.type == "download_model":
        _validate_download_model_params(req.params)
    job_id = await state.runner.submit(req.type, req.target, req.params)
    return {"id": job_id}


def _validate_download_model_params(params: dict[str, Any]) -> None:
    """Minimal per-type param gate for ``download_model`` submissions.

    The runner accepts arbitrary params for any registered type; without this,
    ``POST /api/jobs`` would bypass the download endpoint's filename hygiene
    entirely. Only the two known keys are accepted and the filename must pass
    the same guard the endpoint uses — the job still re-validates at the sink.
    """
    from vesta.inference.download import safe_gguf_basename

    extra = set(params) - {"url", "filename"}
    if extra:
        raise HTTPException(
            status_code=400, detail=f"unexpected download_model params: {sorted(extra)}"
        )
    url = params.get("url")
    if not isinstance(url, str) or not url:
        raise HTTPException(status_code=400, detail="download_model requires a non-empty url")
    filename = params.get("filename")
    if not isinstance(filename, str) or not filename:
        raise HTTPException(status_code=400, detail="download_model requires a filename")
    try:
        safe_gguf_basename(filename, append_suffix=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid model filename") from exc


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
