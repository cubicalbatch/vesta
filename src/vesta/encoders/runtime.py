"""ONNX Runtime session construction — in-process, int8, torch-free.

One function, one place session options are set: ``intra_op_num_threads`` and
spinning-disabled are both set here from settings, never hardcoded per model.

``session.intra_op.allow_spinning=0`` prevents ONNX Runtime's default spin-wait
from keeping CPU cores hot between inferences, reducing power consumption and cache
pressure on a local appliance.
"""

from __future__ import annotations

from pathlib import Path

import onnxruntime as ort


def build_session(
    model_path: Path,
    *,
    intra_op_threads: int,
    spinning: bool,
    cpu_mem_arena: bool,
) -> ort.InferenceSession:
    """Load an ONNX model with Vesta's standard CPU session configuration.

    ``providers=["CPUExecutionProvider"]`` always — no CUDA/DirectML/etc; this
    is a CPU-only appliance. Raises if the file is missing or the graph
    fails to load; callers (the encoder manager) catch this and treat it as
    "capability unavailable", never a hard failure (degrade-don't-fail).

    ``enable_cpu_mem_arena`` is set from ``cpu_mem_arena`` (defaulting off via
    the ``encoders.cpu_mem_arena`` setting). ONNX's arena caches peak tensor
    allocations for the session lifetime and fragments under varying input
    shapes (a depth-1 index run's leads vary 30→512 tokens after padding),
    driving RSS up monotonically. Disabling it makes ONNX malloc/free each
    tensor so RSS tracks the live working set.
    """
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, intra_op_threads)
    options.add_session_config_entry("session.intra_op.allow_spinning", "1" if spinning else "0")
    options.enable_cpu_mem_arena = bool(cpu_mem_arena)
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


__all__ = ["build_session"]
