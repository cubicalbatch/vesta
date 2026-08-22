"""Encoder benchmark rows — real measurements now that the ONNX runtime exists.

``eval/`` may import only ``retrieval`` and ``config`` (the ≤2
cap), so this module never imports ``vesta.encoders`` — the composition root
(``cli.py``'s ``_cmd_bench``) constructs the real ``Encoder``/``CrossEncoder``
instances and hands them in as ``session`` (duck-typed: anything with an async
``embed(texts, kind=...)`` or ``score(query, passages)`` method works, so this
module has no import-time dependency on the encoder package). ``session=None``
still returns a ``DeferredRow`` — the signature is unchanged ("the swap is one
body per function").

Three rows:

* **ONNX int8 vs fp32 speedup** — needs *two* sessions (``session`` = int8,
  ``fp32_session`` = fp32) over the same batch. Both optional; the row stays
  deferred unless both are supplied.
* **Embedder throughput (chunks/s)** — real encode
  throughput over synthetic ~400-token passages.
* **Reranker latency @256 tokens** — 20 query/passage pairs through the real
  cross-encoder, reporting mean wall-clock per call.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class DeferredRow:
    """A bench row whose measurement is deferred (no session supplied)."""

    name: str
    projected: float
    unit: str
    projection_source: str
    notes: str

    def to_row(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": None,
            "projected": self.projected,
            "unit": self.unit,
            "projection_source": self.projection_source,
            "verdict": "deferred — encoder runtime absent",
            "notes": self.notes,
        }


@dataclass(frozen=True)
class MeasuredRow:
    """A real measurement, replacing a projected estimate."""

    name: str
    value: float
    unit: str
    projected: float
    projection_source: str
    verdict: str
    notes: str

    def to_row(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": round(self.value, 3),
            "projected": self.projected,
            "unit": self.unit,
            "projection_source": self.projection_source,
            "verdict": self.verdict,
            "notes": self.notes,
        }


@runtime_checkable
class _EmbedLike(Protocol):
    async def embed(self, texts: Any, *, kind: str) -> Any: ...


@runtime_checkable
class _ScoreLike(Protocol):
    async def score(self, query: str, passages: Any) -> Any: ...


#: Synthetic ~400-token passage text (no ZIM needed) — throughput measurement
#: only cares about token count, not content.
_SAMPLE_PASSAGE = ("The history of computing spans many decades of innovation. " * 30).strip()
_SAMPLE_QUERY = "What is the history of computing and how did it develop over time?"


def onnx_int8_speedup(
    session: object | None = None,
    *,
    fp32_session: object | None = None,
    batch_size: int = 16,
    repeats: int = 3,
) -> DeferredRow | MeasuredRow:
    """ONNX int8 speedup vs fp32. Needs both an int8 ``session`` (the
    production encoder) and an ``fp32_session`` over the identical graph.

    int8 vs fp32 speedup reflects the
    difference between an affordable encoder and an unaffordable one.
    """
    if not isinstance(session, _EmbedLike) or not isinstance(fp32_session, _EmbedLike):
        return DeferredRow(
            name="ONNX int8 vs fp32 speedup",
            projected=2.0,
            unit="x",
            projection_source="Estimated baseline (assumed 2x; literature 1.4-4x)",
            notes=(
                "int8 keeps math in integers and hits AVX-512-VNNI for a real 2-4x. "
                "To measure: time int8 vs fp32 encode of a fixed chunk batch "
                "through the ONNX session."
            ),
        )
    texts = [_SAMPLE_PASSAGE] * batch_size

    async def _time(enc: _EmbedLike) -> float:
        await enc.embed(texts, kind="passage")  # warmup
        start = time.perf_counter()
        for _ in range(repeats):
            await enc.embed(texts, kind="passage")
        return max(time.perf_counter() - start, 1e-9)

    int8_elapsed = asyncio.run(_time(session))
    fp32_elapsed = asyncio.run(_time(fp32_session))
    speedup = fp32_elapsed / int8_elapsed
    return MeasuredRow(
        name="ONNX int8 vs fp32 speedup",
        value=speedup,
        unit="x",
        projected=2.0,
        projection_source="Estimated baseline (assumed 2x; literature 1.4-4x)",
        verdict="confirms" if 1.4 <= speedup <= 4.0 else "replaces",
        notes=(
            f"batch={batch_size}, repeats={repeats}. int8={int8_elapsed / repeats * 1000:.1f}ms/batch "
            f"fp32={fp32_elapsed / repeats * 1000:.1f}ms/batch."
        ),
    )


def embedder_throughput(
    session: object | None = None, *, batch_size: int = 32, repeats: int = 3
) -> DeferredRow | MeasuredRow:
    """Embedder throughput in chunks/s — real encode of synthetic ~400-token
    passages through the configured ``encoders.embed.model`` session.
    """
    if not isinstance(session, _EmbedLike):
        return DeferredRow(
            name="Embedder throughput",
            projected=115.7,
            unit="chunks/s",
            projection_source="Estimated baseline (embed/Balanced tier ~12h for 5M chunks int8, "
            "not the Fast/static tier's 3000-6000 chunks/s)",
            notes=(
                "Cost baseline for encoders.embed.model specifically "
                "(Balanced tier — the Fast/static tier's 3000-6000 chunks/s is a "
                "different model and belongs to static_pass, not this row). To "
                "measure: encode N 400-token chunks, report chunks/s, with intra_op=2 "
                "+ spinning disabled."
            ),
        )
    texts = [_SAMPLE_PASSAGE] * batch_size

    async def _run() -> float:
        await session.embed(texts, kind="passage")  # warmup
        start = time.perf_counter()
        for _ in range(repeats):
            await session.embed(texts, kind="passage")
        return max(time.perf_counter() - start, 1e-9)

    elapsed = asyncio.run(_run())
    chunks_per_sec = (batch_size * repeats) / elapsed
    return MeasuredRow(
        name="Embedder throughput",
        value=chunks_per_sec,
        unit="chunks/s",
        projected=115.7,
        projection_source="Estimated baseline (embed/Balanced tier ~12h for 5M chunks int8, "
        "not the Fast/static tier's 3000-6000 chunks/s)",
        verdict="replaces",
        notes=(
            f"batch={batch_size}, repeats={repeats}, ~400-token synthetic passages, "
            "real ONNX session (intra_op/spinning settings applied)."
        ),
    )


def reranker_latency(
    session: object | None = None, *, num_passages: int = 20
) -> DeferredRow | MeasuredRow:
    """Cross-encoder reranker latency @256 tokens — real score() call over 20
    query/passage pairs through the configured ``encoders.rerank.model`` session.

    Budget: MiniLM-L6 int8 over 20 passages ~= 90-180 ms. Anything
    278 M+ costs 4-7 s on CPU and is unaffordable.
    """
    if not isinstance(session, _ScoreLike):
        return DeferredRow(
            name="Reranker latency @256 tokens",
            projected=135.0,
            unit="ms",
            projection_source="Estimated baseline (MiniLM-L6 int8, 20 passages, ~90-180 ms)",
            notes=(
                "To measure: score 20 256-token query/passage pairs through the "
                "cross-encoder session; report mean latency. Also check "
                "if it helps nDCG on our golden set."
            ),
        )
    passages = [_SAMPLE_PASSAGE] * num_passages

    async def _run() -> float:
        await session.score(_SAMPLE_QUERY, passages)  # warmup
        start = time.perf_counter()
        await session.score(_SAMPLE_QUERY, passages)
        return max(time.perf_counter() - start, 1e-9)

    elapsed_ms = asyncio.run(_run()) * 1000.0
    return MeasuredRow(
        name="Reranker latency @256 tokens",
        value=elapsed_ms,
        unit="ms",
        projected=135.0,
        projection_source="Estimated baseline (MiniLM-L6 int8, 20 passages, ~90-180 ms)",
        verdict="confirms" if 50.0 <= elapsed_ms <= 400.0 else "replaces",
        notes=f"num_passages={num_passages}, truncated to 256 tokens (retrieval.rerank.truncate_tokens).",
    )


def deferred_rows() -> list[DeferredRow]:
    """All encoder rows with no session — for callers that haven't wired the
    ONNX runtime in yet (kept for backward-compatible callers/tests)."""
    rows = [onnx_int8_speedup(), embedder_throughput(), reranker_latency()]
    return [r for r in rows if isinstance(r, DeferredRow)]


__all__ = [
    "DeferredRow",
    "MeasuredRow",
    "deferred_rows",
    "embedder_throughput",
    "onnx_int8_speedup",
    "reranker_latency",
]
