"""The retrieval trace — a first-class, always-on output.

Tracing is *not* debug logging. Every pipeline stage writes into the ``Trace``
passed down the call chain, and ``to_dict()`` produces stable, versioned JSON
consumed by the dev console, the eval harness, the production trace panel,
and ``messages.trace_json``.

This module lives under ``retrieval/`` because jobs and the API share the
same trace structure (01-foundations).

Per-stage record: component id + resolved params, inputs (counts, query
forms), outputs (ids, ranks, scores), timing, and any degradation decision with
its reason. If a component's decision cannot be reconstructed from the trace,
the component is not finished.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

from vesta.config.capabilities import Capability

#: Bumped only on a breaking change to the JSON shape. Consumers pin to a major
#: version; the dev console/eval harness must not break silently when it moves.
TRACE_VERSION = 1


@dataclass(frozen=True)
class DegradationRecord:
    """Why a component was dropped from the pipeline (degrade-don't-fail)."""

    component: str
    missing: str  # Capability value (StrEnum serializes to its string) or error category
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"component": self.component, "missing": self.missing, "reason": self.reason}


@dataclass
class StageCtx(AbstractContextManager["StageCtx"]):
    """Context manager for one pipeline stage. Records timing on exit.

    Usage::

        with trace.stage("stage_b", "static_embedder", {"limit": 20}) as st:
            ...  # do work
            st.add_inputs({"passage_count": 200})
            st.add_outputs({"topk": 20, "scores": [...]})

    Timing uses ``perf_counter``; it is not wall-clock-serializable across
    machines, which is why the trace also records the resolved params — the
    *cause* of the timing, not just the number.
    """

    name: str
    component: str
    params: Mapping[str, Any]
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    _started: float = 0.0
    _duration_ms: float | None = None
    _trace: Trace | None = None

    def __enter__(self) -> StageCtx:
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self._duration_ms = (time.perf_counter() - self._started) * 1000.0
        if self._trace is not None:
            self._trace._record(self)

    def add_inputs(self, values: Mapping[str, Any]) -> None:
        self.inputs.update(values)

    def add_outputs(self, values: Mapping[str, Any]) -> None:
        self.outputs.update(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "component": self.component,
            "params": dict(self.params),
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
            "duration_ms": self._duration_ms,
        }


class Trace:
    """The live, append-only trace for one pipeline execution."""

    def __init__(self) -> None:
        self._stages: list[StageCtx] = []
        self._degradations: list[DegradationRecord] = []
        self._profile_hash: str | None = None
        self._profile_name: str | None = None

    def set_profile(self, name: str, hash: str) -> None:
        """Record which profile produced this trace (03 spec: content hash in every trace)."""
        self._profile_name = name
        self._profile_hash = hash

    def stage(self, name: str, component: str, params: Mapping[str, Any] | None = None) -> StageCtx:
        """Begin a stage. Returns a context manager that records itself on exit."""
        ctx = StageCtx(
            name=name,
            component=component,
            params=params or {},
            _trace=self,
        )
        return ctx

    def _record(self, stage: StageCtx) -> None:
        self._stages.append(stage)

    def degraded(self, component: str, missing: Capability | str, reason: str) -> None:
        """Record that a component was dropped because a capability was missing or a runtime error occurred.

        Degrade-don't-fail: the pipeline does not raise when a profile names a component
        whose ``requires`` are unmet or encounters a runtime failure — it records the drop here and continues.
        """
        self._degradations.append(
            DegradationRecord(component=component, missing=str(missing), reason=reason)
        )

    def to_dict(self) -> dict[str, Any]:
        """Stable, versioned JSON. Key order is fixed on purpose; downstream
        tooling diffs two traces field-by-field."""
        d: dict[str, Any] = {
            "version": TRACE_VERSION,
            "stages": [stage.to_dict() for stage in self._stages],
            "degradations": [d.to_dict() for d in self._degradations],
        }
        if self._profile_name is not None:
            d["profile"] = self._profile_name
        if self._profile_hash is not None:
            d["profile_hash"] = self._profile_hash
        return d
