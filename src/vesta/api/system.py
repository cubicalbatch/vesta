"""``GET /api/system/storage``.

Exposes the same free-space number the download job already computes in
``catalog/download.py::_check_free_space`` before committing to a download —
without this, the only way to discover insufficient disk is to start a
multi-gigabyte download and have it fail. Turns a late failure into an early
one; the reason stands independent of the UI (any client, including a script,
benefits from a pre-check).
"""

import os
import shutil

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from vesta.api.state import AppState, app_state
from vesta.catalog import get_zims_dir

router = APIRouter(tags=["system"])


class StorageOut(BaseModel):
    data_dir: str
    total_bytes: int
    free_bytes: int
    used_by_zims_bytes: int


@router.get("/api/system/storage", response_model=StorageOut)
async def system_storage(state: AppState = Depends(app_state)) -> dict[str, object]:
    zims_dir = get_zims_dir()
    if zims_dir is None:
        raise HTTPException(status_code=503, detail="archive registry not ready")
    zims_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(zims_dir)

    async with (
        state.db.read() as conn,
        conn.execute("SELECT COALESCE(SUM(file_size), 0) AS total FROM zims") as cur,
    ):
        row = await cur.fetchone()
    used_by_zims = int(row["total"]) if row is not None else 0

    return StorageOut(
        data_dir=str(zims_dir.parent),
        total_bytes=usage.total,
        free_bytes=usage.free,
        used_by_zims_bytes=used_by_zims,
    ).model_dump()


class HardwareOut(BaseModel):
    ram_total_bytes: int
    cpu_count: int


@router.get("/api/system/hardware", response_model=HardwareOut)
async def system_hardware() -> dict[str, object]:
    """Detect total RAM and CPU count for first-run model recommendation.

    Uses ``/proc/meminfo`` (Linux-only — Vesta is a Linux appliance) so no
    ``psutil`` dependency is needed. Returns 0 for RAM if ``/proc/meminfo`` is
    unreadable (e.g. non-Linux dev host); the wizard treats 0 as "unknown" and
    falls back to the smaller model.
    """
    return HardwareOut(
        ram_total_bytes=_read_meminfo_total(),
        cpu_count=os.cpu_count() or 1,
    ).model_dump()


def _read_meminfo_total() -> int:
    """Total RAM in bytes from ``/proc/meminfo``."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    # "MemTotal:       16384000 kB"
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


__all__ = ["router"]
