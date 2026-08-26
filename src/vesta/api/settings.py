"""``GET/PUT /api/settings`` and ``GET /api/settings/schema`` (01-foundations).

The schema endpoint is what makes the settings form free: every declared
setting ships its own type, bounds, group and help, so a form can be rendered
with zero per-knob wiring. ``PUT`` writes through to the ``settings`` table and
refreshes the resolver in place, so a hot change is visible to the very next
request without a restart — the table is authoritative over env.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from vesta import config
from vesta.api.state import AppState, app_state
from vesta.config.settings import SettingSchema
from vesta.db.connection import Database
from vesta.db.settings_store import load_settings, upsert_setting

router = APIRouter(prefix="/api/settings", tags=["settings"])


def utc_now_iso() -> str:
    """The second-resolution UTC timestamp every settings row is stamped with."""
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()


async def persist_settings_and_reload(db: Database, pairs: Sequence[tuple[str, str]]) -> None:
    """The settings write-through ritual, shared by every endpoint that writes
    the ``settings`` table directly (PUT /api/settings, model download/activate/
    delete, profile persistence): upsert the pairs under one timestamp, then
    reload the WHOLE table and reseed the resolver so the next request sees
    authoritative values — forgetting the reload is how a write silently
    doesn't happen.
    """
    now = utc_now_iso()
    async with db.write() as conn:
        for key, value in pairs:
            await upsert_setting(conn, key, value, now)
    async with db.read() as conn:
        fresh = await load_settings(conn)
    config.set_db_values(fresh)


class SettingSchemaOut(BaseModel):
    key: str
    type: str
    default: object
    group: str
    help: str
    min: float | None = None
    max: float | None = None
    choices: list[object] | None = None
    hot: bool
    secret: bool = False


class SettingsValues(BaseModel):
    """A ``{key: value-string}`` patch. Values arrive as strings on the wire."""

    values: dict[str, str] = Field(default_factory=dict)


def _schema_dict(s: SettingSchema) -> SettingSchemaOut:
    return SettingSchemaOut(
        key=s.key,
        type=s.type,
        default=s.default,
        group=s.group,
        help=s.help,
        min=s.min,
        max=s.max,
        choices=list(s.choices) if s.choices is not None else None,
        hot=s.hot,
        secret=s.secret,
    )


@router.get("/schema")
async def get_schema() -> dict[str, object]:
    """The full declared tuning surface — drives the settings UI form."""
    items = [_schema_dict(s).model_dump() for s in config.schema()]
    return {"settings": items}


@router.get("")
async def get_settings() -> dict[str, object]:
    """Current resolved values for every setting (snapshot pinned for the call).

    Secret settings (API keys, the auth password) come back as
    ``config.SECRET_MASK`` when configured, never their stored value — this
    response lands in browser tabs, devtools, and copy-paste.
    """
    snap = config.snapshot()
    return {"values": config.redact_values(snap.values)}


@router.put("")
async def put_settings(
    patch: SettingsValues,
    state: AppState = Depends(app_state),
) -> dict[str, object]:
    """Write one or more settings. Hot settings take effect on the next request.

    Each value is validated + coerced against its declaration; an out-of-bounds
    or unknown key is a 400 so the UI gets immediate feedback.
    """
    registry = config.all_settings()
    coerced: dict[str, object] = {}
    for key, raw in patch.values.items():
        descriptor = registry.get(key)
        if descriptor is None:
            raise HTTPException(status_code=400, detail=f"unknown setting {key!r}")
        if descriptor.secret and raw.strip() in ("", config.SECRET_MASK):
            # A secret arrives blank or masked when a client round-trips the
            # GET values untouched — that means "leave the stored value as
            # is", never "wipe it". Only a fresh non-masked string replaces.
            continue
        try:
            value = config.validate_and_coerce(descriptor, raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        coerced[key] = value
    await persist_settings_and_reload(
        state.db, [(key, _to_storage(value)) for key, value in coerced.items()]
    )
    # An inference.* change rebuilds the LLM runtime in-process
    # (source/model/endpoint/idle changes apply on the next question, no
    # restart). Keys baked into the llama-server command line (context size,
    # threads, binary path…) trigger a full runtime rebind — a child restart
    # — inside ``rebuild_runtime``. Non-fatal by contract — a bad endpoint
    # must always stay correctable from the UI, and ``rebuild_runtime`` logs
    # instead of raising.
    if any(key.startswith("inference.") for key in coerced):
        from vesta.inference import rebuild_runtime

        await rebuild_runtime(changed=set(coerced))
    snap = config.snapshot()
    return {
        "values": config.redact_values(snap.values),
        "applied": sorted(coerced),
    }


def _to_storage(value: object) -> str:
    """Round-trip a typed value back to the TEXT the table stores."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
