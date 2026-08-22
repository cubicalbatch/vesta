"""Retrieval introspection API — profiles and components.

Profile CRUD ("profile editor"):

* ``GET /api/retrieval/profiles`` — list built-in *and* user-saved profiles; each
  entry carries a ``builtin`` flag so the console can mark read-only rows.
* ``GET /api/retrieval/profiles/{name}`` — full detail: the parsed ``profile``
  dict (for the generated form), the canonical ``yaml`` text (for the textarea),
  the content ``hash``, and the ``builtin`` flag.
* ``POST /api/retrieval/profiles`` — save a user profile from either ``yaml``
  text (the escape hatch) or a ``profile`` dict (the generated-form fast path).
  Validates against the registry, rejects built-in names, persists to the
  ``retrieval.profiles`` settings blob, and refreshes the resolver so the saved
  profile is usable by the very next search — no restart.
* ``DELETE /api/retrieval/profiles/{name}`` — remove a user profile. Built-ins
  are read-only (400); unknown names 404.

User profiles live in the ``settings`` table under ``retrieval.profiles`` as a
JSON blob ``{name: yaml_text}`` — the table is authoritative
over env. Built-ins ship as read-only YAML in ``retrieval/profiles/*.yaml``.

``GET /api/retrieval/components`` — registry introspection: per kind, list
registered impls with their param schemas. This is what lets a generic
profile editor be built with NO hand-written UI per component.
"""

from __future__ import annotations

import datetime as _dt
import json
from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from vesta import config as app_config
from vesta.api.state import AppState, app_state
from vesta.db.settings_store import load_settings, upsert_setting
from vesta.retrieval import RETRIEVAL_PROFILES
from vesta.retrieval.profiles import (
    BUILTIN_PROFILES,
    RetrievalProfile,
    load_user_profiles,
    parse_profile_text,
    profile_to_dict,
    profile_to_yaml,
)
from vesta.retrieval.registry import component_schemas

router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


# ── DTOs ────────────────────────────────────────────────────────────────────


class ProfileItem(BaseModel):
    name: str
    description: str
    hash: str
    builtin: bool


class ProfilesResponse(BaseModel):
    profiles: list[ProfileItem]


class ProfileDetailResponse(BaseModel):
    """One profile in full — drives both the YAML textarea and the generated form."""

    name: str
    description: str
    hash: str
    builtin: bool
    profile: dict[str, Any]
    yaml: str


class SaveProfileRequest(BaseModel):
    """Accept either form. Exactly one of ``yaml`` / ``profile`` is required.

    ``yaml`` is the escape hatch (paste/edited YAML text); ``profile`` is the
    generated-form fast path (a plain dict the server re-serializes). Allowing
    both is what lets the console's two editors share one save button.
    """

    yaml: str | None = None
    profile: dict[str, Any] | None = None


class DeleteResponse(BaseModel):
    deleted: str


class ComponentsResponse(BaseModel):
    components: dict[str, Any]


# ── User-profile blob plumbing ──────────────────────────────────────────────
# The blob is a setting value (TEXT holding JSON ``{name: yaml_text}``); the
# composition root owns the read/modify/persist round-trip so the resolver is
# refreshed in place, identical to ``PUT /api/settings``.


def _load_blob() -> dict[str, str]:
    """The raw user-profile blob ``{name: yaml_text}`` from the active resolver.

    Degrades to empty on any resolver/parse error rather than raising — a
    misconfigured blob must not take down profile listing.
    """
    try:
        raw = str(app_config.get(RETRIEVAL_PROFILES))
    except RuntimeError:
        return {}
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}


async def _persist_blob(state: AppState, blob: dict[str, str]) -> None:
    """Write the blob to the ``settings`` table and refresh the resolver.

    Mirrors ``PUT /api/settings``: write through, then reload the whole table so
    the resolver reflects the authoritative state on the next request.
    """
    now = _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()
    value = json.dumps(blob, ensure_ascii=False, sort_keys=True)
    async with state.db.write() as conn:
        await upsert_setting(conn, RETRIEVAL_PROFILES.key, value, now)
    async with state.db.read() as conn:
        fresh = await load_settings(conn)
    app_config.set_db_values(fresh)


def _user_profiles() -> dict[str, RetrievalProfile]:
    """Parsed user profiles from the blob (invalid entries already skipped)."""
    return load_user_profiles(str(app_config.get(RETRIEVAL_PROFILES)))


def _detail(profile: RetrievalProfile, *, builtin: bool) -> ProfileDetailResponse:
    return ProfileDetailResponse(
        name=profile.name,
        description=profile.description,
        hash=profile.hash,
        builtin=builtin,
        profile=profile_to_dict(profile),
        yaml=profile_to_yaml(profile),
    )


# ── Routes ──────────────────────────────────────────────────────────────────


@router.get("/profiles", response_model=ProfilesResponse)
async def get_profiles() -> ProfilesResponse:
    """List built-in and user-saved profiles, each flagged ``builtin``."""
    items: list[ProfileItem] = [
        ProfileItem(name=p.name, description=p.description, hash=p.hash, builtin=True)
        for p in BUILTIN_PROFILES.values()
    ]
    for name, p in _user_profiles().items():
        # Built-in names are rejected at save time; skip any shadow defensively.
        if name in BUILTIN_PROFILES:
            continue
        items.append(
            ProfileItem(name=p.name, description=p.description, hash=p.hash, builtin=False)
        )
    return ProfilesResponse(profiles=items)


@router.get("/profiles/{name}", response_model=ProfileDetailResponse)
async def get_profile_detail(name: str) -> ProfileDetailResponse:
    """Full detail for one profile: parsed dict + canonical YAML + hash.

    Built-ins first (read-only), then user-saved. 404 if the name is neither.
    """
    builtin = BUILTIN_PROFILES.get(name)
    if builtin is not None:
        return _detail(builtin, builtin=True)
    user = _user_profiles().get(name)
    if user is None:
        raise HTTPException(status_code=404, detail=f"profile {name!r} not found")
    return _detail(user, builtin=False)


@router.post("/profiles", response_model=ProfileDetailResponse)
async def save_profile(
    req: SaveProfileRequest,
    state: AppState = Depends(app_state),
) -> ProfileDetailResponse:
    """Save a user profile from ``yaml`` text or a ``profile`` dict.

    The profile is validated against the live registry (every component ``impl``
    must resolve), then persisted to the ``retrieval.profiles`` blob. A saved
    profile is immediately usable by ``GET /api/search?profile=`` — no restart.
    """
    if req.yaml is None and req.profile is None:
        raise HTTPException(status_code=400, detail="must supply 'yaml' or 'profile'")

    # Normalize to canonical YAML text so hashing and storage are consistent
    # regardless of which editor produced the input.
    if req.yaml is not None:
        yaml_text = req.yaml
    else:
        yaml_text = yaml.dump(
            req.profile, sort_keys=True, default_flow_style=False, allow_unicode=True
        )

    try:
        parsed = parse_profile_text(yaml_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Built-ins are read-only: a user save must not shadow or replace one
    # (00-conventions: built-in profiles ship read-only).
    if parsed.name in BUILTIN_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"profile name {parsed.name!r} is a built-in and cannot be overwritten",
        )

    blob = _load_blob()
    blob[parsed.name] = yaml_text
    await _persist_blob(state, blob)
    return _detail(parsed, builtin=False)


@router.delete("/profiles/{name}", response_model=DeleteResponse)
async def delete_profile(
    name: str,
    state: AppState = Depends(app_state),
) -> DeleteResponse:
    """Delete a user-saved profile. Built-ins are read-only (400)."""
    if name in BUILTIN_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"profile {name!r} is a built-in and cannot be deleted",
        )
    blob = _load_blob()
    if name not in blob:
        raise HTTPException(status_code=404, detail=f"profile {name!r} not found")
    del blob[name]
    await _persist_blob(state, blob)
    return DeleteResponse(deleted=name)


@router.get("/components", response_model=ComponentsResponse)
async def get_components() -> ComponentsResponse:
    """Registry introspection: for each kind, list registered impls with param
    JSON schemas. This feeds a generic profile editor."""
    return ComponentsResponse(components=component_schemas())
