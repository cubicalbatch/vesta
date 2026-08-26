"""Unit tests for retrieval profile resolution.

Covers ``resolve_profile_from_settings`` across built-in, user-saved, active,
strict, and fallback paths (AUDIT_0824 L6).
"""

from __future__ import annotations

import json
from typing import Any

import yaml

from vesta.config.settings import SettingsSnapshot, all_settings
from vesta.retrieval.profiles import (
    BUILTIN_PROFILES,
    load_profile,
    profile_to_dict,
    resolve_profile_from_settings,
)


def _make_snapshot(**overrides: Any) -> SettingsSnapshot:
    values: dict[str, object] = {s.key: s.default for s in all_settings().values()}
    values.update(overrides)
    return SettingsSnapshot(values=values)


def _custom_profile_yaml(name: str = "custom_prof", desc: str = "Custom Profile") -> str:
    lexical = load_profile("lexical")
    assert lexical is not None
    d = profile_to_dict(lexical)
    d["name"] = name
    d["description"] = desc
    return str(yaml.dump(d))


def test_resolve_profile_from_settings_builtins() -> None:
    """Built-in profiles (lexical, standard, hybrid) resolve by name."""
    for name in ("lexical", "standard", "hybrid"):
        p = resolve_profile_from_settings(name)
        assert p is not None
        assert p.name == name
        assert p.hash == BUILTIN_PROFILES[name].hash


def test_resolve_profile_from_settings_active_default() -> None:
    """When name is omitted, active profile (default: hybrid) is resolved."""
    p_none = resolve_profile_from_settings(None)
    assert p_none is not None
    assert p_none.name == "hybrid"

    p_empty = resolve_profile_from_settings("")
    assert p_empty is not None
    assert p_empty.name == "hybrid"


def test_resolve_profile_from_settings_active_custom() -> None:
    """Active profile setting determines resolution when name is omitted."""
    sn = _make_snapshot(**{"retrieval.active_profile": "standard"})
    p = resolve_profile_from_settings(None, snapshot=sn)
    assert p is not None
    assert p.name == "standard"


def test_resolve_profile_from_settings_unknown_fallback() -> None:
    """Unknown profile name with fallback_to_default=True resolves lexical."""
    p = resolve_profile_from_settings("non_existent_profile", fallback_to_default=True)
    assert p is not None
    assert p.name == "lexical"


def test_resolve_profile_from_settings_unknown_strict() -> None:
    """Unknown profile name with fallback_to_default=False returns None."""
    p = resolve_profile_from_settings("non_existent_profile", fallback_to_default=False)
    assert p is None


def test_resolve_profile_from_settings_none_strict() -> None:
    """Omitted name with fallback_to_default=False returns None."""
    assert resolve_profile_from_settings(None, fallback_to_default=False) is None
    assert resolve_profile_from_settings("", fallback_to_default=False) is None


def test_resolve_profile_from_settings_user_profile() -> None:
    """User-saved profile in settings resolves by name and as active profile."""
    custom_yaml = _custom_profile_yaml("custom_prof", "My custom search")
    user_blob = json.dumps({"custom_prof": custom_yaml})

    sn = _make_snapshot(
        **{
            "retrieval.profiles": user_blob,
            "retrieval.active_profile": "custom_prof",
        }
    )

    # 1. Resolve explicit user profile name
    p = resolve_profile_from_settings("custom_prof", snapshot=sn)
    assert p is not None
    assert p.name == "custom_prof"
    assert p.description == "My custom search"

    # 2. Resolve omitted name with active profile pointing to custom user profile
    p_active = resolve_profile_from_settings(None, snapshot=sn)
    assert p_active is not None
    assert p_active.name == "custom_prof"
    assert p_active.description == "My custom search"


def test_resolve_profile_from_settings_user_shadows_builtin() -> None:
    """A user profile with the same name as a built-in takes priority."""
    custom_yaml = _custom_profile_yaml("hybrid", "User override of hybrid")
    user_blob = json.dumps({"hybrid": custom_yaml})

    sn = _make_snapshot(**{"retrieval.profiles": user_blob})
    p = resolve_profile_from_settings("hybrid", snapshot=sn)
    assert p is not None
    assert p.name == "hybrid"
    assert p.description == "User override of hybrid"
    # Content hash should differ from built-in hybrid because description changed
    assert p.hash != BUILTIN_PROFILES["hybrid"].hash


def test_resolve_profile_from_settings_corrupt_user_profile_ignored() -> None:
    """Corrupt user profiles JSON/YAML are skipped gracefully without crashing."""
    corrupt_blob = "{not valid json"
    sn = _make_snapshot(**{"retrieval.profiles": corrupt_blob})
    p = resolve_profile_from_settings("lexical", snapshot=sn)
    assert p is not None
    assert p.name == "lexical"

    invalid_yaml_blob = json.dumps({"broken": "not: [valid: yaml"})
    sn2 = _make_snapshot(**{"retrieval.profiles": invalid_yaml_blob})
    assert resolve_profile_from_settings("broken", snapshot=sn2, fallback_to_default=False) is None


def test_resolve_profile_from_settings_custom_default_name() -> None:
    """Custom default_name can be specified for fallback."""
    p = resolve_profile_from_settings(
        "unknown_profile",
        fallback_to_default=True,
        default_name="standard",
    )
    assert p is not None
    assert p.name == "standard"


def test_resolve_profile_from_settings_broken_active_fallback() -> None:
    """If active profile setting names an unknown profile, fallback to lexical."""
    sn = _make_snapshot(**{"retrieval.active_profile": "ghost_profile"})
    p = resolve_profile_from_settings(None, snapshot=sn, fallback_to_default=True)
    assert p is not None
    assert p.name == "lexical"
