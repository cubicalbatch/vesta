"""``GET /health`` — liveness + component + capability breakdown (01-foundations).

Must return 200 with nothing configured: no ZIMs, no models, no
index — the appliance is still alive and search would still work. Every layer is
optional above the one below it, so an empty capability set is a *healthy* state.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from vesta.api.state import AppState, app_state
from vesta.config.capabilities import Capability, compute_capabilities
from vesta.config.feature_flags import advanced_menu_enabled

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(state: AppState = Depends(app_state)) -> dict[str, object]:
    db_ok = await _check_db(state)
    caps = compute_capabilities()
    return {
        "status": "ok" if db_ok else "degraded",
        "components": {"database": "ok" if db_ok else "error"},
        # Every capability reported with its *current* availability so the UI
        # shows what the pipeline can actually do right now. All
        # false — search works, nothing else is configured, and that is healthy.
        "capabilities": {c.value: c in caps for c in Capability},
        "capabilities_available": sorted(c.value for c in caps),
        # Admin/dev affordance gate: the Settings → Advanced tab (eval/
        # benchmarks) is hidden unless VESTA_ADVANCED_MENU is set truthy.
        # Off by default so end users never see the benchmark tooling.
        "advanced_menu": advanced_menu_enabled(),
        # First-run wizard bookkeeping: once the user finishes /welcome (picked
        # archives, an LLM, or skipped), POST /api/setup/complete flips this so
        # the SPA's zero-archive redirect never forces them back. False until
        # then — a genuinely fresh install still routes `/` → `/welcome`.
        "setup_completed": await _setup_completed(state),
    }


async def _setup_completed(state: AppState) -> bool:
    """Read the ``setup.completed`` flag from the settings table.

    ``False`` if the row is absent or not literally ``"true"`` (defensive: the
    value is only ever written by ``POST /api/setup/complete``, but a manually
    edited DB shouldn't crash ``/health``). Any read error degrades to
    ``False`` so a healthy DB-less state still reports a usable payload.
    """
    try:
        async with (
            state.db.read() as conn,
            conn.execute("SELECT value FROM settings WHERE key = 'setup.completed'") as cur,
        ):
            row = await cur.fetchone()
    except Exception:
        return False
    return row is not None and row["value"] == "true"


async def _check_db(state: AppState) -> bool:
    try:
        async with state.db.read() as conn, conn.execute("SELECT 1") as cur:
            await cur.fetchone()
        return True
    except Exception:
        return False
