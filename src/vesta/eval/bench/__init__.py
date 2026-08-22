"""Hardware benchmark harness.

One reporting format, one consumer: a committed ``bench_results/<machine>-<date>.md``
annotated ``confirms``/``replaces`` against projected numbers.

What runs for real here: FP32 GEMM ceiling + memory bandwidth
(``hardware``), extraction threads-vs-processes (``extraction``), and per-stage
latency percentiles from retrieval traces (``latency``). The encoder rows
(``encoder``) benchmark ONNX runtime throughput and latency.
"""

from __future__ import annotations

from . import encoder, extraction, hardware, latency

__all__ = ["encoder", "extraction", "hardware", "latency"]
