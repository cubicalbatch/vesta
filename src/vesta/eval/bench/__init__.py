"""Hardware benchmark harness.

One reporting format, one consumer: a committed ``bench_results/<machine>-<date>.md``
annotated ``confirms``/``replaces`` against projected numbers.

What runs for real here: FP32 GEMM ceiling + memory bandwidth
(``hardware``), extraction threads-vs-processes (``extraction``), and ONNX
runtime throughput/latency rows (``encoder``).
"""

from __future__ import annotations

from . import encoder, extraction, hardware

__all__ = ["encoder", "extraction", "hardware"]
