"""Hardware benchmarks — FP32 GEMM ceiling + memory bandwidth.

Encoder-independent measurements to anchor hardware performance:

* **FP32 GEMM ceiling** — the GFLOPS a single-threaded matmul sustains. The
  encoder's cost is dominated by GEMM, so this is the ceiling against which
  ONNX int8 speedup is measured. ``numpy`` here, never
  ``torch`` (torch-free image is mandatory and slower for this anyway).
* **Memory bandwidth** — GB/s sustained over a large array scan. Memory-
  bandwidth-bound stages (decode, bit-quantized vector scan) are capped by this,
  not by FLOPS.

Each measurement is annotated ``confirms`` or ``replaces`` against baseline
projections so the committed bench file is a verdict, not raw data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HardwareResult:
    """One hardware measurement, annotated vs the baseline projection."""

    name: str
    value: float
    unit: str
    projection: float  # baseline projection value
    projection_source: str  # projection source citation
    verdict: str  # "confirms" | "replaces"
    notes: str = ""

    def to_row(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": round(self.value, 2),
            "unit": self.unit,
            "projection": self.projection,
            "projection_source": self.projection_source,
            "verdict": self.verdict,
            "notes": self.notes,
        }


def measure_gemm_ceiling(*, n: int = 1024, warmup: int = 1, repeats: int = 5) -> HardwareResult:
    """FP32 GEMM GFLOPS — the encoder cost ceiling.

    A single ``nxn`` matmul is ``2n³`` FLOPs; GFLOPS = ``(2n³ · repeats) /
    elapsed_seconds / 1e9``. ``numpy`` calls BLAS, so this reflects the real
    single-process ceiling (multi-threaded BLAS) the encoder would see — the
    anchor for the int8 speedup measurement.
    """
    a = np.random.default_rng(0).random((n, n), dtype=np.float32)
    b = np.random.default_rng(1).random((n, n), dtype=np.float32)
    for _ in range(warmup):
        _ = a @ b
    start = time.perf_counter()
    for _ in range(repeats):
        _ = a @ b
    elapsed = max(time.perf_counter() - start, 1e-9)
    flops = 2 * (n**3) * repeats
    gflops = flops / elapsed / 1e9
    return HardwareResult(
        name="FP32 GEMM ceiling",
        value=gflops,
        unit="GFLOPS",
        projection=30.0,
        projection_source="Estimated baseline (throttled 4-core laptop anchor)",
        verdict="replaces",
        notes=(
            "Real single-process FP32 matmul throughput (numpy/BLAS). This is the "
            "anchor for the ONNX int8 speedup measurement. "
            f"n={n}, repeats={repeats}."
        ),
    )


def measure_memory_bandwidth(*, size_mb: int = 512, repeats: int = 5) -> HardwareResult:
    """Sustained memory bandwidth in GB/s over a large array reduction.

    A memory-bound pass (sum a ``size_mb`` float32 array) measures the GB/s the
    CPU+RAM sustain, capping decode throughput and the bit-quantized vector scan.
    Reported GB/s = ``(size_mb · repeats) / elapsed / 1024`` (binary).
    """
    n = (size_mb * 1024 * 1024) // 4
    arr = np.ones(n, dtype=np.float32)
    acc = 0.0
    for _ in range(repeats):
        acc += float(arr.sum())
    start = time.perf_counter()
    for _ in range(repeats):
        acc += float(arr.sum())
    elapsed = max(time.perf_counter() - start, 1e-9)
    bytes_moved = size_mb * 1024 * 1024 * repeats
    gbps = bytes_moved / elapsed / (1024**3)
    return HardwareResult(
        name="Memory bandwidth",
        value=gbps,
        unit="GB/s",
        projection=64.0,
        projection_source="Estimated baseline (single-CCD Zen 5 ceiling ~64 GB/s)",
        verdict="replaces",
        notes=(
            f"Sustained read bandwidth over a {size_mb:.0f} MB array. Caps decode "
            f"tok/s and the bit-quantized vector scan. [acc={acc:.0f}]"
        ),
    )


def measure_cpu_info() -> dict[str, object]:
    """Static CPU facts recorded alongside every bench file."""
    import os
    import platform

    return {
        "machine_id": f"{platform.node()}-{os.cpu_count()}cpu",
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
    }


__all__ = [
    "HardwareResult",
    "measure_cpu_info",
    "measure_gemm_ceiling",
    "measure_memory_bandwidth",
]
