"""``POST /api/setup/complete`` — mark the first-run wizard finished.

Once the user completes the welcome wizard (whether they picked archives, an
LLM, or skipped everything), the app must never force them back to ``/welcome``
on a zero-archive state. This endpoint writes a ``setup.completed`` flag to the
settings table (the system of record) so the decision survives reloads and is
visible to ``GET /health``'s ``setup_completed`` field — which is what the SPA's
zero-archive redirect keys off.

The flag is intentionally *not* a declared ``config/settings`` descriptor: it is
wizard bookkeeping, not a tunable, and declaring it would pollute the
``GET /api/settings/schema`` form. It is written/read only through the
``settings`` table directly.
"""

from __future__ import annotations

import datetime as _dt

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from vesta.api.state import AppState, app_state
from vesta.db.settings_store import upsert_setting

router = APIRouter(tags=["setup"])


class SetupCompleteOut(BaseModel):
    ok: bool


@router.post("/api/setup/complete", response_model=SetupCompleteOut)
async def complete_setup(state: AppState = Depends(app_state)) -> dict[str, object]:
    """Persist ``setup.completed=true``. Idempotent — safe to call repeatedly."""
    now = _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()
    async with state.db.write() as conn:
        await upsert_setting(conn, "setup.completed", "true", now)
    return {"ok": True}


__all__ = ["router"]
