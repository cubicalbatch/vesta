"""Typed, declared-once, layered settings.

Three things hold this module's shape together:

* **Declaration carries its own UI metadata.** ``setting()`` is called at import
  time in the module that owns the knob; the ``group``/``help``/bounds travel with
  it so ``GET /api/settings/schema`` fully describes the tuning surface. No
  frontend change is needed when a knob is added.
* **The ``settings`` table is authoritative over env**: the UI can change
  anything without a restart; env supplies first-boot defaults only.
* **No ``os.getenv`` outside ``config/``** — enforced by a test. The only
  place the environment is read is the resolver here.

Resolution order (later wins):
    built-in default → env var → ``settings`` table →
    active retrieval profile → per-request override (dev/eval only).

``config`` depends on nothing internal: the DB-backed layer
is handed in as a plain ``Mapping`` by the composition root (``main``), which
keeps the dependency arrow one-way.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Generic, TypeVar, cast

T = TypeVar("T")

# The primitive types a setting may hold. ``type`` in the signature is the
# builtin; we keep this narrow so coercion round-trips losslessly through the
# ``settings`` table (which stores TEXT) and back.
_SUPPORTED_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "float",
    bool: "boolean",
}


@dataclass(frozen=True)
class Setting(Generic[T]):
    """A declared setting: identity + metadata + default. Resolved elsewhere."""

    key: str
    type: type
    default: T
    group: str
    help: str
    min: float | None = None
    max: float | None = None
    choices: tuple[object, ...] | None = None
    hot: bool = True
    #: Credential-like value (API key, password): GET redacts it, PUT treats
    #: blank/masked as "unchanged", persisted run snapshots drop it entirely.
    secret: bool = False


@dataclass(frozen=True)
class SettingSchema:
    """The public, JSON-friendly description of one declared setting."""

    key: str
    type: str
    default: object
    group: str
    help: str
    min: float | None = None
    max: float | None = None
    choices: tuple[object, ...] | None = None
    hot: bool = True
    secret: bool = False


# The registry stores Setting[Any] (invariance makes Setting[int] !<: Setting[object]);
# each entry's concrete type is preserved by the descriptor the owner module holds.
_REGISTRY: dict[str, Setting[Any]] = {}


def setting(
    key: str,
    type: type[T],
    default: T,
    *,
    group: str,
    help: str,
    min: float | None = None,
    max: float | None = None,
    choices: tuple[object, ...] | None = None,
    hot: bool = True,
    secret: bool = False,
) -> Setting[T]:
    """Declare a setting and register it. Called at module import time.

    The descriptor is plain data; resolution happens through :func:`get`, which
    consults the active resolver. Generic over ``T`` so ``setting("x", int, 5)``
    yields ``Setting[int]`` and ``get(it)`` returns ``int``.
    """
    if type not in _SUPPORTED_TYPES:
        raise TypeError(
            f"setting {key!r}: type must be one of {sorted(t.__name__ for t in _SUPPORTED_TYPES)}"
        )
    if min is not None and max is not None and min > max:
        raise ValueError(f"setting {key!r}: min ({min}) > max ({max})")
    descriptor: Setting[T] = Setting(
        key=key,
        type=type,
        default=default,
        group=group,
        help=help,
        min=min,
        max=max,
        choices=choices,
        hot=hot,
        secret=secret,
    )
    existing = _REGISTRY.get(key)
    if existing is not None and existing != descriptor:
        raise RuntimeError(f"setting {key!r} declared twice with different values")
    _REGISTRY[key] = descriptor  # stored as Setting[Any]; concrete type kept by caller
    return descriptor


def schema() -> list[SettingSchema]:
    """Every declared setting's public schema, sorted by key (stable order)."""
    out: list[SettingSchema] = []
    for s in sorted(_REGISTRY.values(), key=lambda d: d.key):
        out.append(
            SettingSchema(
                key=s.key,
                type=_SUPPORTED_TYPES[s.type],
                default=s.default,
                group=s.group,
                help=s.help,
                min=s.min,
                max=s.max,
                choices=s.choices,
                hot=s.hot,
                secret=s.secret,
            )
        )
    return out


def all_settings() -> Mapping[str, Setting[Any]]:
    """Read-only view of the registry (used by the resolver)."""
    return MappingProxyType(_REGISTRY)


#: What GET /api/settings returns in place of a configured secret's value.
#: Chosen to be obviously not-a-key; PUT accepts it back as "unchanged".
SECRET_MASK = "********"


def is_secret(key: str) -> bool:
    """Whether the named declared setting carries credential-like material."""
    d = _REGISTRY.get(key)
    return d is not None and d.secret


def redact_values(values: Mapping[str, object]) -> dict[str, object]:
    """Copy of ``values`` safe to serve: every non-empty secret becomes
    :data:`SECRET_MASK`. Non-secrets and unset secrets pass through as-is."""
    return {
        key: SECRET_MASK if is_secret(key) and str(value) else value
        for key, value in values.items()
    }


def strip_secret_values(values: Mapping[str, object]) -> dict[str, object]:
    """Copy of ``values`` with secret keys removed — for persisted pins such
    as run ``config_json`` snapshots that outlive the process and get served
    back by run-detail endpoints."""
    return {key: value for key, value in values.items() if not is_secret(key)}


# ── Resolution ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SettingsSnapshot:
    """An immutable pin of every resolved setting value, captured at a point in
    time. A request holds one of these so a mid-request settings change can't
    produce a half-old / half-new pipeline; A/B comparisons depend on it. The
    trace can also record exactly what config produced a result.
    """

    values: Mapping[str, object]

    def get(self, s: Setting[T]) -> T:
        return cast("T", self.values[s.key])


def _coerce(s: Setting[Any], raw: str) -> object:
    t = s.type
    try:
        if t is bool:
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        if t is int:
            return int(raw)
        if t is float:
            return float(raw)
        return raw  # str
    except ValueError as exc:
        raise ValueError(f"setting {s.key!r}: cannot coerce {raw!r} to {t.__name__}") from exc


def _validate(s: Setting[Any], value: object) -> None:
    if s.choices is not None and value not in s.choices:
        raise ValueError(f"setting {s.key!r}: {value!r} not in choices {s.choices}")
    if isinstance(value, bool | int | float) and not isinstance(value, str):
        if s.min is not None and value < s.min:
            raise ValueError(f"setting {s.key!r}: {value} < min {s.min}")
        if s.max is not None and value > s.max:
            raise ValueError(f"setting {s.key!r}: {value} > max {s.max}")


def resolve_value(
    s: Setting[T],
    *,
    db_values: Mapping[str, str] | None,
    env: Mapping[str, str],
) -> T:
    """Resolve a single setting through default → env → settings-table.

    ``db_values`` is ``None`` during the bootstrap window before the DB exists;
    in that window only default + env are consulted (the settings table can't
    override anything because it isn't open yet).
    """
    value: object = s.default
    env_raw = env.get(s.key)
    if env_raw is not None:
        value = _coerce(s, env_raw)
    if db_values is not None and s.key in db_values:
        value = _coerce(s, db_values[s.key])
    _validate(s, value)
    return cast("T", value)
