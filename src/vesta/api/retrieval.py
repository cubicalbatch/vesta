"""Retrieval introspection API — profiles.

* ``GET /api/retrieval/profiles`` — list built-in *and* user-saved profiles; each
  entry carries a ``builtin`` flag so the console can mark read-only rows.

User profiles live in the ``settings`` table under ``retrieval.profiles`` as a
JSON blob ``{name: yaml_text}`` — the table is authoritative
over env. Built-ins ship as read-only YAML in ``retrieval/profiles/*.yaml``.
The profile-editor write surface (detail/save/delete/components) was removed
(AUDIT_0822 DEAD): no client ever called it — the SPA consumes only this
listing.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from vesta import config as app_config
from vesta.retrieval import RETRIEVAL_PROFILES
from vesta.retrieval.profiles import (
    BUILTIN_PROFILES,
    RetrievalProfile,
    load_user_profiles,
)

router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


# ── DTOs ────────────────────────────────────────────────────────────────────


class ProfileItem(BaseModel):
    name: str
    description: str
    hash: str
    builtin: bool


class ProfilesResponse(BaseModel):
    profiles: list[ProfileItem]


def _user_profiles() -> dict[str, RetrievalProfile]:
    """Parsed user profiles from the blob (invalid entries already skipped)."""
    return load_user_profiles(str(app_config.get(RETRIEVAL_PROFILES)))


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
