"""Retrieval package root.

This package must remain dependency-free of ``answer``/``api``/``index`` —
enforced by ``tests/test_boundaries.py``. It depends on ``zim``, ``config``, and
``encoders``.

Imports here register components via ``impls/``, ``scorers/`` and
``assemblers/`` so that ``resolve(kind, name)`` finds them. Built-in profiles
are loaded at the same time from the YAML files in ``profiles/``.
"""

from __future__ import annotations

from vesta.config.settings import setting

# Re-export the trace types.
from vesta.retrieval.trace import TRACE_VERSION, DegradationRecord, StageCtx, Trace

# ── Retrieval settings ──────────────────────────────────────────────────────

RETRIEVAL_ACTIVE_PROFILE = setting(
    "retrieval.active_profile",
    str,
    "hybrid",
    group="Retrieval / General",
    help="Default retrieval profile. ?profile= query param overrides this "
    "(gated by api.allow_profile_override). Built-in: hybrid, standard, "
    "lexical. ``hybrid`` is the default: ``standard``'s lexical "
    "Stage A + two-pass Stage B scoring (static shortlist + cross-encoder "
    "rerank) + lead_boost assembly, plus the dense ``vector_knn`` source. "
    "With ``VECTORS`` unmet (depth-0 box) ``vector_knn`` is capability-dropped "
    "and ``hybrid`` is byte-equivalent to ``standard``. ``standard`` is the "
    "pure lexical+Stage-B profile; ``lexical`` is the depth-0 baseline and "
    "the fallback for unknown profile names. All degrade gracefully.",
    hot=True,
)
RETRIEVAL_PROFILES = setting(
    "retrieval.profiles",
    str,
    "{}",
    group="Retrieval / General",
    help="User-defined retrieval profiles as JSON blob. Cloned from built-ins "
    "with param overrides (the UI edits this).",
    hot=True,
)
RETRIEVAL_MAX_ARCHIVES_CONCURRENT = setting(
    "retrieval.max_archives_concurrent",
    int,
    8,
    group="Retrieval / Concurrency",
    help="Semaphore bound for concurrent archive searches. "
    "Prevents libzim pool exhaustion on multi-archive corpora.",
    min=1,
    max=64,
    hot=True,
)

API_ALLOW_PROFILE_OVERRIDE = setting(
    "api.allow_profile_override",
    bool,
    True,
    group="API",
    help="Allow ?profile= query param to override the active retrieval profile "
    "(dev console and eval harness use this; disable for production).",
    hot=True,
)

# Imports deferred to avoid circular imports — settings must be registered
# before impls/scorers/assemblers and profiles are loaded (which reference the
# registry). Order matters: profiles.py validates every component reference
# against the registry at import time, so it must load LAST. Using
# ``import_module`` (not plain ``import`` statements) is deliberate — isort/
# ruff's I001 alphabetizes contiguous import blocks, which would silently
# reorder ``profiles`` before ``scorers``/``assemblers`` and break this.
from importlib import import_module as _import_module  # noqa: E402

for _module_name in (
    "vesta.retrieval.assemblers",
    "vesta.retrieval.impls",
    "vesta.retrieval.scorers",
    "vesta.retrieval.profiles",
):
    _import_module(_module_name)

__all__ = [
    "API_ALLOW_PROFILE_OVERRIDE",
    "RETRIEVAL_ACTIVE_PROFILE",
    "RETRIEVAL_MAX_ARCHIVES_CONCURRENT",
    "TRACE_VERSION",
    "DegradationRecord",
    "StageCtx",
    "Trace",
]
