"""Trace: stable versioned JSON, per-stage timing, degradation records."""

from __future__ import annotations

import json

from vesta.config.capabilities import Capability
from vesta.retrieval.trace import TRACE_VERSION, Trace


def test_trace_schema_and_json_serializability() -> None:
    """Trace produces fixed-schema, versioned dict and serializes cleanly to JSON."""
    tr = Trace()
    with tr.stage("s", "c", {"x": 1}) as st:
        st.add_outputs({"r": [1, 2, 3]})
    tr.degraded("llm", Capability.LLM, "none")

    out = tr.to_dict()
    assert out["version"] == TRACE_VERSION == 1
    # The keys are fixed on purpose — downstream tooling diffs field-by-field.
    assert set(out) == {"version", "stages", "degradations"}

    blob = json.dumps(out)
    loaded = json.loads(blob)
    assert loaded["version"] == 1
    assert len(loaded["stages"]) == 1
    assert len(loaded["degradations"]) == 1


def test_stage_records_component_params_inputs_outputs_and_timing() -> None:
    """Stage context manager records params, inputs, outputs, and duration on exit."""
    tr = Trace()
    with tr.stage("stage_a", "xapian_fts", {"limit": 40, "fallback": True}) as st:
        st.add_inputs({"query": "einstein", "archives": 2})
        st.add_outputs({"paths": ["A/Einstein"], "count": 1})
    out = tr.to_dict()
    assert len(out["stages"]) == 1
    stage = out["stages"][0]
    assert stage["name"] == "stage_a"
    assert stage["component"] == "xapian_fts"
    assert stage["params"] == {"limit": 40, "fallback": True}
    assert stage["inputs"] == {"query": "einstein", "archives": 2}
    assert stage["outputs"] == {"paths": ["A/Einstein"], "count": 1}
    assert stage["duration_ms"] is not None
    assert stage["duration_ms"] >= 0


def test_degradation_recorded_with_capability_and_reason() -> None:
    tr = Trace()
    tr.degraded("cross_encoder", Capability.CROSS_ENCODER, "model not present")
    out = tr.to_dict()
    assert out["degradations"] == [
        {"component": "cross_encoder", "missing": "cross_encoder", "reason": "model not present"}
    ]
