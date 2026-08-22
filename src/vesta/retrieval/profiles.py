"""Profile loading, validation, content hashing.

Built-in profiles ship as YAML files in ``retrieval/profiles/*.yaml`` and are
read-only. User profiles live in the ``settings`` table as a blob (the UI
edits them; cloning built-ins is supported). A profile is versioned by
content hash — recorded in every trace so "which config produced this number" is
answerable six weeks later.

Rules:

* Missing required component in a profile = validation error at load time, not a
  silent default.
* ``?profile=`` query param overrides the active profile (gated by
  ``api.allow_profile_override``).
* Profile hash is SHA-256 of the canonical YAML (sorted keys, no comments).
* Built-in profiles are read-only; the API may clone them with overrides.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from vesta.retrieval.registry import resolve

_PROFILES_DIR = Path(__file__).resolve().parent / "profiles"

#: Component kinds present in a profile and their required registry kind.
_KIND_MAP: dict[str, str] = {
    "preparers": "query_preparer",
    "sources": "candidate_source",
    "fusion": "fuser",
    "passages": "passage_builder",
    "scorers": "passage_scorer",
    "assembler": "context_assembler",
}

#: Component kinds that store a list of impls (vs a single impl).
_LIST_KINDS: frozenset[str] = frozenset({"preparers", "sources", "scorers"})

#: Component kinds that store a single impl.
_SINGLE_KINDS: frozenset[str] = frozenset({"fusion", "passages", "assembler"})


@dataclass(frozen=True)
class ProfileComponent:
    """One component slot in a profile — impl + params."""

    impl: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalProfile:
    """A validated, hashed retrieval profile (03 spec)."""

    name: str
    description: str
    hash: str  # SHA-256 of normalized YAML text
    preparers: tuple[ProfileComponent, ...]
    sources: tuple[ProfileComponent, ...]
    fusion: ProfileComponent
    passages: ProfileComponent
    scorers: tuple[ProfileComponent, ...]
    assembler: ProfileComponent


def _normalize_yaml(text: str) -> str:
    """Round-trip a YAML string through canonical form for hashing."""
    data = yaml.safe_load(text)
    result = yaml.dump(data, sort_keys=True, default_flow_style=False, allow_unicode=True)
    return str(result)


def _hash_yaml(text: str) -> str:
    """SHA-256 hex digest of a YAML profile string."""
    return hashlib.sha256(_normalize_yaml(text).encode("utf-8")).hexdigest()


def _validate_components(profile_name: str, data: dict[str, Any]) -> None:
    """Raise ``ValueError`` if any component ref does not resolve in the registry."""

    def _check(kind_tags: dict[str, str], key: str, entry: Any) -> None:
        if not isinstance(entry, dict):
            raise ValueError(
                f"profile {profile_name!r}: {key} entry must be a dict, got {type(entry).__name__}"
            )
        impl = entry.get("impl")
        if not impl:
            raise ValueError(f"profile {profile_name!r}: {key}.impl is required")
        for tag in kind_tags:
            resolved = resolve(tag, impl)
            if resolved is not None:
                return
        raise ValueError(
            f"profile {profile_name!r}: {key}.impl={impl!r} not found in registry "
            f"(looked under {sorted(kind_tags)})"
        )

    for key in _LIST_KINDS & set(data):
        entries = data[key]
        if not isinstance(entries, list):
            entries = []
        for entry in entries:
            _check({_KIND_MAP[key]: key}, key, entry)

    for key in _SINGLE_KINDS & set(data):
        _check({_KIND_MAP[key]: key}, key, data[key])

    # Required single-valued stages must all be present.
    for key in _SINGLE_KINDS:
        if key not in data:
            raise ValueError(f"profile {profile_name!r}: missing required key {key!r}")

    # A profile with zero sources has nothing to search with. Reject at load
    # rather than failing at runtime ("missing required component =
    # validation error at profile load, not a silent default"). The pipeline
    # separately guards the "all sources capability-dropped" case.
    sources = data.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"profile {profile_name!r}: at least one candidate source is required")


def _parse_profile(name: str, desc: str, yaml_text: str) -> RetrievalProfile:
    """Parse a YAML profile doc into a validated ``RetrievalProfile``."""
    data = yaml.safe_load(yaml_text)
    if not isinstance(data, dict):
        raise ValueError(f"profile {name!r}: YAML must map to a dict")

    def _to_component(raw: dict[str, Any]) -> ProfileComponent:
        return ProfileComponent(impl=str(raw["impl"]), params=dict(raw.get("params") or {}))

    def _to_components(raw: list[dict[str, Any]] | None) -> tuple[ProfileComponent, ...]:
        if raw is None:
            return ()
        return tuple(_to_component(r) for r in raw)

    return RetrievalProfile(
        name=name,
        description=desc,
        hash=_hash_yaml(yaml_text),
        preparers=_to_components(data.get("preparers")),
        sources=_to_components(data.get("sources")),
        fusion=_to_component(data["fusion"]),
        passages=_to_component(data["passages"]),
        scorers=_to_components(data.get("scorers")),
        assembler=_to_component(data["assembler"]),
    )


def parse_profile_text(yaml_text: str, *, default_name: str | None = None) -> RetrievalProfile:
    """Parse + validate a standalone YAML profile document (no file backing).

    Used for both user-saved profiles (the ``retrieval.profiles``
    settings blob) and file-backed built-ins. ``default_name`` is used only
    when the document omits ``name`` (built-in files fall back to their
    filename); a user-saved profile must name itself explicitly.
    """
    data = yaml.safe_load(yaml_text)
    if not isinstance(data, dict):
        raise ValueError("profile YAML must map to a dict")
    name = data.get("name") or default_name
    if not name:
        raise ValueError("profile YAML must include a 'name' field")
    desc = data.get("description") or ""
    _validate_components(name, data)
    return _parse_profile(name, desc, yaml_text)


def _load_and_validate(path: Path) -> RetrievalProfile | None:
    """Parse, validate, and hash one profile YAML file."""
    text = path.read_text(encoding="utf-8")
    if not isinstance(yaml.safe_load(text), dict):
        return None
    return parse_profile_text(text, default_name=path.stem)


#: Built-in profiles, loaded once at import and read-only thereafter.
BUILTIN_PROFILES: dict[str, RetrievalProfile] = {}


def _load_builtins() -> dict[str, RetrievalProfile]:
    """Load every ``*.yaml`` from the profiles directory (called at import)."""
    loaded: dict[str, RetrievalProfile] = {}
    if not _PROFILES_DIR.is_dir():
        return loaded
    for path in sorted(_PROFILES_DIR.glob("*.yaml")):
        try:
            profile = _load_and_validate(path)
        except ValueError:
            # A bad built-in profile aborts startup — it is a deploy error.
            raise
        if profile is not None:
            loaded[profile.name] = profile
    return loaded


def load_profile(name: str) -> RetrievalProfile | None:
    """Look up a profile by name in the built-in set."""
    return BUILTIN_PROFILES.get(name)


def list_profiles() -> list[dict[str, str]]:
    """All built-in profiles with name, description, and content hash."""
    return [
        {"name": p.name, "description": p.description, "hash": p.hash}
        for p in BUILTIN_PROFILES.values()
    ]


def clone_profile(name: str, overrides: dict[str, Any]) -> RetrievalProfile | None:
    """Clone a built-in profile, applying param overrides.

    Returns ``None`` if the named profile doesn't exist. Overrides are applied at
    the component level: ``{"fusion": {"params": {"k": 30}}}`` replaces ``k`` in
    the fusion component's params.
    """
    original = BUILTIN_PROFILES.get(name)
    if original is None:
        return None

    def _apply_overrides(
        components: tuple[ProfileComponent, ...], key: str
    ) -> tuple[ProfileComponent, ...]:
        override = overrides.get(key, {})
        new_params = dict(override.get("params", {}))
        return tuple(
            ProfileComponent(impl=c.impl, params={**c.params, **new_params}) for c in components
        )

    def _apply_single(component: ProfileComponent, key: str) -> ProfileComponent:
        override = overrides.get(key, {})
        new_params = dict(override.get("params", {}))
        return ProfileComponent(impl=component.impl, params={**component.params, **new_params})

    # Build a new profile with overrides applied and a fresh hash.
    new = RetrievalProfile(
        name=original.name,
        description=f"{original.description} (user override)",
        hash="",  # recomputed below
        preparers=_apply_overrides(original.preparers, "preparers"),
        sources=_apply_overrides(original.sources, "sources"),
        fusion=_apply_single(original.fusion, "fusion"),
        passages=_apply_single(original.passages, "passages"),
        scorers=_apply_overrides(original.scorers, "scorers"),
        assembler=_apply_single(original.assembler, "assembler"),
    )
    # Recompute hash from the canonical YAML representation.
    yaml_text = profile_to_yaml(new)
    new = RetrievalProfile(
        name=new.name,
        description=new.description,
        hash=_hash_yaml(yaml_text),
        preparers=new.preparers,
        sources=new.sources,
        fusion=new.fusion,
        passages=new.passages,
        scorers=new.scorers,
        assembler=new.assembler,
    )
    return new


def _component_dict(pc: ProfileComponent) -> dict[str, Any]:
    d: dict[str, Any] = {"impl": pc.impl}
    if pc.params:
        d["params"] = pc.params
    return d


def profile_to_dict(profile: RetrievalProfile) -> dict[str, Any]:
    """The plain-dict shape a profile round-trips through (same shape the YAML
    files and ``retrieval.profiles`` blob use). This is what the dev console's
    generated form edits directly — it is sent back as ``profile`` on save and
    turned into YAML server-side, so the form never needs a client-side YAML
    parser."""
    data: dict[str, Any] = {
        "name": profile.name,
        "description": profile.description,
        "assembler": _component_dict(profile.assembler),
        "fusion": _component_dict(profile.fusion),
        "passages": _component_dict(profile.passages),
    }
    if profile.preparers:
        data["preparers"] = [_component_dict(p) for p in profile.preparers]
    if profile.sources:
        data["sources"] = [_component_dict(s) for s in profile.sources]
    if profile.scorers:
        data["scorers"] = [_component_dict(s) for s in profile.scorers]
    return data


def profile_to_yaml(profile: RetrievalProfile) -> str:
    """Serialize a profile back to YAML (used to hash clone overrides and to
    populate the dev console's YAML editor)."""
    result = yaml.dump(
        profile_to_dict(profile), sort_keys=True, default_flow_style=False, allow_unicode=True
    )
    return str(result)


# Load built-ins at import time.
BUILTIN_PROFILES.update(_load_builtins())


def load_user_profiles(blob: str) -> dict[str, RetrievalProfile]:
    """Parse the ``retrieval.profiles`` settings blob: ``{name: yaml_text}``.

    An entry that fails to parse or validate is skipped rather than raised —
    one bad user-saved profile must not take down every search (degrade-
    don't-fail). The API layer that writes this blob validates on
    save, so a skip here means the registry lost a component between save and
    load, not a routine occurrence.
    """
    try:
        data = json.loads(blob) if blob else {}
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, RetrievalProfile] = {}
    for name, yaml_text in data.items():
        if not isinstance(yaml_text, str):
            continue
        try:
            out[str(name)] = parse_profile_text(yaml_text, default_name=str(name))
        except ValueError:
            continue
    return out


def resolve_profile(
    name: str, user_profiles: dict[str, RetrievalProfile] | None = None
) -> RetrievalProfile | None:
    """Look up a profile by name, user-saved profiles taking priority over
    built-ins of the same name (a saved profile must be
    usable by name without a restart)."""
    if user_profiles is not None and name in user_profiles:
        return user_profiles[name]
    return BUILTIN_PROFILES.get(name)
