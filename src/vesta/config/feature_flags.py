"""Env-only feature flags — not settings.

Unlike the settings registry, these are not user-editable and never appear in
``GET /api/settings/schema``: they gate admin/dev affordances that should not be
part of the tuning surface. Read here because ``config/`` is the one place the
environment may be read (enforced by ``tests/test_no_getenv.py``).
"""

from __future__ import annotations

import os

# strtobool-ish, case-insensitive. Empty / unset / "0" / "false" ⇒ off.
_TRUE = {"1", "true", "yes", "on", "t", "y"}


def advanced_menu_enabled() -> bool:
    """Whether the Settings → Advanced tab (eval/benchmarks) is exposed.

    Off by default; set ``VESTA_ADVANCED_MENU=True`` (any truthy form) to show it.
    Surfaced to the SPA via ``GET /health``.
    """
    return os.environ.get("VESTA_ADVANCED_MENU", "").strip().lower() in _TRUE
