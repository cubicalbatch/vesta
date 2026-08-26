"""The active settings resolver and the layered ``get``/``snapshot`` accessors.

This module is split from :mod:`vesta.config.settings` (which owns the
declarations) because the resolver carries runtime state — the DB-backed layer
that is handed in by the composition root. Keeping them apart means importing
the descriptors does not also drag in a live resolver.

This is the *only* module in the codebase that reads the process environment
(enforced by ``tests/test_no_getenv.py``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import TypeVar, cast

from vesta.config.settings import (
    _REGISTRY,
    Setting,
    SettingsSnapshot,
    resolve_value,
)

T = TypeVar("T")


class _Resolver:
    """Holds the current layers and resolves settings against them."""

    def __init__(self, *, db_values: Mapping[str, str] | None, env: Mapping[str, str]) -> None:
        self._db_values = db_values
        self._env = env

    def get(self, s: Setting[T]) -> T:
        return resolve_value(s, db_values=self._db_values, env=self._env)

    def snapshot(self) -> SettingsSnapshot:
        values: dict[str, object] = {}
        for key, s in _REGISTRY.items():
            values[key] = resolve_value(s, db_values=self._db_values, env=self._env)
        return SettingsSnapshot(values=MappingProxyType(values))


_RESOLVER: _Resolver | None = None


def configure(
    *,
    db_values: Mapping[str, str] | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    """Install the active resolver. Called by ``main`` at startup and again once
    the settings table has been loaded (to add the DB layer).

    ``env`` defaults to the real process environment, which is the *only* place
    this module reads the environment (see ``tests/test_no_getenv.py``).
    """
    global _RESOLVER
    _RESOLVER = _Resolver(
        db_values=db_values,
        env=env if env is not None else MappingProxyType(dict(os.environ)),
    )


def set_db_values(db_values: Mapping[str, str]) -> None:
    """Add the settings-table layer to the active resolver (after migrations).

    If no resolver is configured yet, installs one with the current environment.
    Called by the composition root once the ``settings`` table is open and read.
    """
    global _RESOLVER
    if _RESOLVER is None:
        _RESOLVER = _Resolver(db_values=db_values, env=MappingProxyType(dict(os.environ)))
        return
    _RESOLVER = _Resolver(db_values=db_values, env=_RESOLVER._env)


def reset_for_test() -> None:
    """Test hook: drop the active resolver (forces re-configure)."""
    global _RESOLVER
    _RESOLVER = None


def get_resolver() -> _Resolver:
    """The active resolver, raising if none is configured."""
    if _RESOLVER is None:
        raise RuntimeError(
            "settings not configured — call vesta.config.resolution.configure() first"
        )
    return _RESOLVER


def get(s: Setting[T]) -> T:
    """The resolved value of ``s`` through the active resolver."""
    return get_resolver().get(s)


def get_or_default(s: Setting[T]) -> T:
    """The resolved value of ``s`` where a resolver exists, else its default.

    For consumers that must work in the bootstrap window before
    :func:`configure` runs (module-level defaults, unit tests): instead of
    raising, the declared default is returned.
    """
    if _RESOLVER is None:
        return s.default
    return _RESOLVER.get(s)


def snapshot() -> SettingsSnapshot:
    """An immutable snapshot of every setting's current resolved value."""
    return get_resolver().snapshot()


def validate_and_coerce(s: Setting[T], raw: str) -> T:
    """Coerce a user-provided string to the setting's type and check bounds.

    Used by ``PUT /api/settings``: the wire form is always a string, and the
    table stores TEXT, so we round-trip through coercion + validation.
    """
    from vesta.config.settings import _coerce, _validate

    value: object = _coerce(s, raw)
    _validate(s, value)
    return cast("T", value)
