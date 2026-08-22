"""ZIM data layer — everything Vesta knows how to do with a ``.zim`` file, with
**no retrieval policy in it**.

This package answers "what is in this archive, how do I search it, how do I get
text out of it, how do I serve it to a browser". It does *not* decide what to
search, how to rank, or what to send an LLM — that belongs to retrieval and
answer pipelines. That separation is the point: ``zim/`` is the widest surface
of raw-material complexity in the project, and if retrieval strategy leaks into
it, both become unmaintainable.

Capabilities: on import this package registers a probe that turns on
``ZIM_FULLTEXT`` when at least one *enabled* archive has a probed full-text index.
The composition root binds the live registry via :func:`bind_registry`; the
probe re-evaluates on every capability computation, so a rescan that finds
(or loses) an index is reflected immediately.

``zim/`` depends only on ``config`` and ``db`` — at most two internal packages,
enforced by ``tests/test_boundaries.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vesta.config.capabilities import Capability, CapabilitySet, register_probe

if TYPE_CHECKING:
    from vesta.zim.registry import ArchiveRegistry

#: The live registry, bound by the composition root (``main`` lifespan). Held as
#: a module-level reference the capability probe reads — a configured singleton,
#: like the settings resolver.
_REGISTRY: ArchiveRegistry | None = None


def _capability_probe() -> CapabilitySet:
    """``ZIM_FULLTEXT`` is on iff ≥1 enabled archive has a probed index."""
    if _REGISTRY is not None and _REGISTRY.has_any_fulltext():
        return frozenset({Capability.ZIM_FULLTEXT})
    return frozenset()


# Registered exactly once at import; re-evaluated whenever capabilities are
# computed (after a rescan, a model load, …).
register_probe(_capability_probe)


def bind_registry(registry: ArchiveRegistry | None) -> None:
    """Attach (or detach, with ``None``) the live registry for the capability probe."""
    global _REGISTRY
    _REGISTRY = registry


__all__ = ["bind_registry"]
