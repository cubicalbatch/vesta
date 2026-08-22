"""Encoder manager — lazy-loaded, cached, bounded-pool access to all three
model roles.

Two things this class is responsible for that no caller should reimplement:

* **Cheap readiness checks.** ``static_ready()``/``embed_ready()``/
  ``rerank_ready()`` are a filesystem stat, nothing more — capability probes
  call these on every request (search.py recomputes capabilities per request),
  so they must never touch the network or load a session.
* **Lazy, once-per-process session construction.** The first real call to
  ``get_static``/``get_embed``/``get_rerank`` loads the ONNX session + tokenizer
  off the event loop and caches the result (success *or* failure — a missing or
  corrupt model degrades the role's capability rather than retrying every
  request, adhering to degrade-don't-fail).

A single semaphore is shared across all three roles — they
are all CPU-bound ONNX inference and should share one CPU budget rather than
each getting an independent pool that lets total concurrency multiply.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import cast

from vesta.encoders.contracts import CrossEncoder, Encoder
from vesta.encoders.onnx_cross_encoder import OnnxCrossEncoder
from vesta.encoders.onnx_encoder import OnnxBiEncoder, load_tokenizer
from vesta.encoders.registry import ModelSpec, Role, resolve_spec
from vesta.encoders.runtime import build_session

_log = logging.getLogger(__name__)


class EncoderManager:
    """Owns the lazy singletons for the ``static``/``embed``/``rerank`` roles."""

    def __init__(
        self,
        *,
        model_dir: Path,
        static_model: str,
        embed_model: str,
        rerank_model: str,
        intra_op_threads: int,
        spinning: bool,
        cpu_mem_arena: bool,
        pool_size: int,
        rerank_truncate_tokens: int,
        index_intra_op_threads: int | None = None,
    ) -> None:
        self._model_dir = model_dir
        self._repos: dict[Role, str] = {
            "static": static_model,
            "embed": embed_model,
            "rerank": rerank_model,
        }
        self._intra_op_threads = intra_op_threads
        #: Thread count for the bulk-indexing session only (``get_embed_for``).
        #: Falls back to the query-side count when unset so existing callers and
        #: tests keep today's behaviour.
        self._index_intra_op_threads = (
            intra_op_threads if index_intra_op_threads is None else index_intra_op_threads
        )
        self._spinning = spinning
        self._cpu_mem_arena = cpu_mem_arena
        self._rerank_truncate_tokens = rerank_truncate_tokens
        self._semaphore = asyncio.Semaphore(max(1, pool_size))
        self._locks: dict[Role, asyncio.Lock] = {
            "static": asyncio.Lock(),
            "embed": asyncio.Lock(),
            "rerank": asyncio.Lock(),
        }
        self._cache: dict[Role, object | None] = {}
        #: Per-repo ``embed``-role cache for :meth:`get_embed_for` (the
        #: indexer honors ``index.embedder``, which may differ from
        #: ``encoders.embed.model``). Keyed by repo id; same load-once semantics.
        self._repo_lock = asyncio.Lock()
        self._repo_cache: dict[str, Encoder | None] = {}

    # ── cheap readiness (capability probes) ──────────────────────────────

    def static_ready(self) -> bool:
        return self._files_present("static")

    def embed_ready(self) -> bool:
        return self._files_present("embed")

    def rerank_ready(self) -> bool:
        return self._files_present("rerank")

    def _spec_for(self, role: Role) -> ModelSpec | None:
        return resolve_spec(self._repos[role], role)

    def _local_files(self, spec: ModelSpec) -> list[Path]:
        base = self._model_dir / spec.repo_id
        files = [base / spec.onnx_file, base / "tokenizer.json"]
        if spec.onnx_data_file:
            files.append(base / spec.onnx_data_file)
        return files

    def _files_present(self, role: Role) -> bool:
        spec = self._spec_for(role)
        if spec is None:
            return False
        return all(p.is_file() for p in self._local_files(spec))

    # ── lazy load + cache ─────────────────────────────────────────────────

    async def get_static(self) -> Encoder | None:
        return cast("Encoder | None", await self._get("static"))

    async def get_embed(self) -> Encoder | None:
        return cast("Encoder | None", await self._get("embed"))

    async def get_rerank(self) -> CrossEncoder | None:
        return cast("CrossEncoder | None", await self._get("rerank"))

    async def get_embed_for(self, repo_id: str) -> Encoder | None:
        """An ``embed``-role encoder for an arbitrary registered repo id.

        The indexer honors ``index.embedder`` — a separate setting from
        ``encoders.embed.model`` (the static embedder is not the index
        embedder; they are separate registry roles with separate settings).
        Unknown repo id or missing files degrade to ``None``, exactly like a
        role miss.

        This ALWAYS builds a dedicated session, even when ``repo_id`` matches
        ``encoders.embed.model``, because the two sessions are configured
        differently: the query role runs at ``encoders.intra_op_threads`` (tuned
        for search latency on a box that is also serving requests) while the
        indexer runs at ``encoders.index_intra_op_threads`` (tuned for bulk
        throughput). The cost is one extra copy of the weights — ~52 MB for the
        default quantized Granite export.
        """
        if repo_id in self._repo_cache:
            return self._repo_cache[repo_id]
        async with self._repo_lock:
            if repo_id in self._repo_cache:  # a concurrent waiter already loaded it
                return self._repo_cache[repo_id]
            instance = await asyncio.to_thread(self._load_embed_repo_sync, repo_id)
            self._repo_cache[repo_id] = instance
            return instance

    async def _get(self, role: Role) -> object | None:
        if role in self._cache:
            return self._cache[role]
        async with self._locks[role]:
            if role in self._cache:  # a concurrent waiter already loaded it
                return self._cache[role]
            instance = await asyncio.to_thread(self._load_sync, role)
            self._cache[role] = instance
            return instance

    def _load_sync(self, role: Role) -> object | None:
        spec = self._spec_for(role)
        if spec is None or not self._files_present(role):
            return None
        base = self._model_dir / spec.repo_id
        try:
            session = build_session(
                base / spec.onnx_file,
                intra_op_threads=self._intra_op_threads,
                spinning=self._spinning,
                cpu_mem_arena=self._cpu_mem_arena,
            )
            tokenizer = load_tokenizer(base / "tokenizer.json")
        except Exception as exc:  # a corrupt/incompatible model degrades, not crashes
            _log.warning(
                "encoders.load_failed",
                extra={"role": role, "repo": spec.repo_id, "error": repr(exc)},
            )
            return None
        if spec.kind == "cross_encoder_pair":
            return OnnxCrossEncoder(
                spec,
                session,
                tokenizer,
                max_tokens=self._rerank_truncate_tokens,
                semaphore=self._semaphore,
            )
        return OnnxBiEncoder(spec, session, tokenizer, semaphore=self._semaphore)

    def _load_embed_repo_sync(self, repo_id: str) -> Encoder | None:
        """Load one ``embed``-role bi-encoder by repo id (see :meth:`get_embed_for`).

        Uses ``index_intra_op_threads`` — this is the bulk-indexing session.
        """
        spec = resolve_spec(repo_id, "embed")
        if spec is None or not all(p.is_file() for p in self._local_files(spec)):
            return None
        base = self._model_dir / spec.repo_id
        try:
            session = build_session(
                base / spec.onnx_file,
                intra_op_threads=self._index_intra_op_threads,
                spinning=self._spinning,
                cpu_mem_arena=self._cpu_mem_arena,
            )
            tokenizer = load_tokenizer(base / "tokenizer.json")
        except Exception as exc:  # a corrupt/incompatible model degrades, not crashes
            _log.warning(
                "encoders.load_failed",
                extra={"role": "embed", "repo": spec.repo_id, "error": repr(exc)},
            )
            return None
        return OnnxBiEncoder(spec, session, tokenizer, semaphore=self._semaphore)


__all__ = ["EncoderManager"]
