"""Capability set + computation hook.

A capability is something the running system can actually do *right now* — not a
config flag. It is computed at startup and re-evaluated when the inputs change.
Real probes register their availability (a ZIM with a probed full-text index turns
on ``ZIM_FULLTEXT``, a loaded ONNX encoder turns on ``STATIC_ENCODER``, etc.).

The core rule (*degrade, don't fail*): when a profile names a component
whose ``requires`` are unmet, the pipeline drops it, records the drop in the
trace, and continues. That decision is made against this set — never by raising.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import TypeAlias


class Capability(StrEnum):
    """Everything an upper layer may rely on being present. StrEnum so the
    value flows straight into the health/trace JSON without conversion."""

    ZIM_FULLTEXT = "zim_fulltext"  # at least one archive with a probed index
    VECTORS = "vectors"  # at least one archive at index depth ≥ 1
    STATIC_ENCODER = "static_encoder"  # Stage B1 model present
    CROSS_ENCODER = "cross_encoder"  # Stage B2 model present
    LLM = "llm"  # a chat model is configured and reachable


CapabilitySet: TypeAlias = frozenset[Capability]

# A probe reports the capabilities it can vouch for given current runtime state.
# Subsystems register probes as they initialize. If none are registered,
# the set is empty — which is a valid degraded execution state.
Probe: TypeAlias = Callable[[], CapabilitySet]

_PROBES: list[Probe] = []


def register_probe(probe: Probe) -> None:
    """Register a capability probe."""
    _PROBES.append(probe)


def clear_probes() -> None:
    """Test hook: forget every registered probe."""
    _PROBES.clear()


def compute_capabilities() -> CapabilitySet:
    """Union of every probe's reported capabilities.

    Order-independent and pure; callers may re-evaluate whenever the inputs that
    a probe reads have changed (a ZIM lands, a model loads, …).
    """
    result: set[Capability] = set()
    for probe in _PROBES:
        result.update(probe())
    return frozenset(result)
