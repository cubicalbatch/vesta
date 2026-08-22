"""ONNX bi-encoder — the ``embed`` and ``static`` model roles.

Both roles satisfy :class:`~vesta.encoders.contracts.Encoder`; they differ only
in the ONNX graph's input contract (``registry.RuntimeKind``):

* ``bi_encoder_padded`` (Granite, GTE-ModernBERT): a padded batch of
  ``input_ids``/``attention_mask``. The graph's ``sentence_embedding`` output is
  already pooled — no manual pooling needed. Falls back to mean-pooling
  ``last_hidden_state`` over the attention mask for a model whose export lacks
  that named output (the spec's ``pooling`` field is for this case).
* ``bi_encoder_ragged`` (potion/model2vec): a model2vec distillation exports as
  an ``EmbeddingBag``-shaped graph — flat ``input_ids`` + per-item ``offsets``,
  no attention mask, no padding. This is table lookup and mean pooling: the
  ONNX graph performs the identical computation model2vec's own numpy runtime
  would, so no second runtime is needed to get the static tier's ~0 FLOP cost.

L2 normalization (when ``spec.normalize``) always runs in this class, even if
the graph already emits unit vectors — idempotent, and it means a future model
whose export does *not* pre-normalize behaves identically (normalization
is registry-owned, never assumed from the graph).
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from tokenizers import Tokenizer

from vesta.encoders.registry import ModelSpec


def _mean_pool(
    last_hidden: NDArray[np.float32], attention_mask: NDArray[np.int64]
) -> NDArray[np.float32]:
    """Mean-pool token embeddings over real (non-pad) tokens only."""
    mask = attention_mask.astype(np.float32)[:, :, None]
    summed = (last_hidden * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return (summed / counts).astype(np.float32)


def _l2_normalize(vecs: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (vecs / norms).astype(np.float32)


#: Bound on ``batch_size * seq_len ** 2`` for one ``bi_encoder_padded`` forward
#: pass. Granite-R2 and GTE-ModernBERT are ModernBERT-family models with
#: alternating global/local attention (a Tile/Where node materializes a
#: ``batch * heads * seq_len * seq_len`` mask); an uncapped batch — the indexer
#: flattens every chunk from a whole ``index.batch_size`` articles into one
#: ``embed()`` call — can push that tensor past ONNX Runtime's 4 GiB
#: single-allocation ceiling (confirmed: 128 texts @ 512 tokens on Granite-R2's
#: 12 heads needs a 1.5 GiB tensor; depth-2 indexing routinely submits 500+
#: texts in one call).
#:
#: 8M caps that mask at ~400 MB (32 items @ seq_len 512), a 4x memory cut over
#: the previous 32M for no measured throughput loss: CPU GEMM efficiency
#: saturates by batch ~16-32 (with 32 as the default batch size), so the larger
#: bound bought allocation risk rather than speed. Short chunks still batch
#: aggressively — the bound is on ``count * width**2``, and :func:`_length_buckets`
#: pads each bucket to its own width, so a bucket of 64-token chunks admits
#: far more items than one of 512-token chunks.
_MAX_PADDED_ELEMENTS = 8 * 1024 * 1024

#: Hard cap on items per forward pass, independent of sequence length.
#:
#: This does double duty. It bounds the allocation for SHORT chunks, which
#: :data:`_MAX_PADDED_ELEMENTS` barely constrains (a bucket of 64-token chunks
#: would otherwise admit ~2000 items). More importantly it bounds how far a
#: bucket's width can drift from its shortest member: buckets are filled from
#: length-sorted input, so a smaller cap means each pass spans a narrower band
#: of lengths and wastes less padding. 32 keeps passes in the efficient GEMM
#: range (16-64) while the padding band stays tight — and it lines up with
#: :data:`_MAX_PADDED_ELEMENTS` exactly at the worst case (32 * 512**2 == 8M),
#: so one bound takes over from the other rather than the two fighting.
_MAX_BUCKET_ITEMS = 32


def _length_buckets(order: Sequence[int], lengths: Sequence[int]) -> Iterator[list[int]]:
    """Group ``order`` (indices sorted by ascending token length) into buckets.

    Each bucket is padded to its **own** longest member rather than to the
    longest member of the whole call, which is what makes the sort worth doing:
    the indexer submits one ``embed()`` call carrying every chunk from
    ``index.batch_size`` articles, where a title+lead chunk (~90 tokens) sits
    beside a full H2 section that hits the 512-token truncation ceiling. Padding
    all of them to 512 computes — and discards — several times the necessary
    FLOPs, since a padded position costs exactly as much GEMM as a real one.

    A bucket grows while it fits BOTH bounds (:data:`_MAX_PADDED_ELEMENTS`,
    :data:`_MAX_BUCKET_ITEMS`), and always yields at least one item so a single
    over-budget chunk still runs alone rather than looping forever.
    """
    start = 0
    total = len(order)
    while start < total:
        end = start + 1
        while end < total:
            width = max(lengths[order[end]], 1)
            count = end - start + 1
            if count > _MAX_BUCKET_ITEMS or count * width * width > _MAX_PADDED_ELEMENTS:
                break
            end += 1
        yield list(order[start:end])
        start = end


def _resolve_pad_id(tokenizer: Any) -> int:
    """The tokenizer's pad token id, best-effort (0 when undiscoverable).

    Padded positions always carry ``attention_mask == 0``, and both output paths
    (the graph's own ``sentence_embedding`` and the :func:`_mean_pool` fallback)
    honour the mask — so the id chosen here cannot change an embedding. Reading
    the real one anyway keeps a hand-padded batch byte-identical to what the
    tokenizer's own ``enable_padding()`` would have produced, which is what lets
    the bucketing change be asserted as behaviour-preserving.
    """
    with suppress(Exception):
        tokenizer.enable_padding()
        padding = getattr(tokenizer, "padding", None)
        tokenizer.no_padding()
        if isinstance(padding, dict):
            pad_id: Any = padding.get("pad_id")
            if pad_id is not None:
                return int(pad_id)
    token_to_id = getattr(tokenizer, "token_to_id", None)
    if callable(token_to_id):
        for token in ("[PAD]", "<pad>"):
            with suppress(Exception):
                pad_id = token_to_id(token)
                if pad_id is not None:
                    return int(pad_id)
    return 0


class OnnxBiEncoder:
    """An :class:`~vesta.encoders.contracts.Encoder` backed by one ONNX graph."""

    def __init__(
        self,
        spec: ModelSpec,
        session: ort.InferenceSession,
        tokenizer: Tokenizer,
        *,
        semaphore: asyncio.Semaphore,
    ) -> None:
        self._spec = spec
        self._session = session
        self._tokenizer = tokenizer
        self._semaphore = semaphore
        self._output_names = {o.name for o in session.get_outputs()}
        #: ``enable_padding``/``enable_truncation`` mutate shared tokenizer state,
        #: and the manager's semaphore is shared across all three roles (size
        #: ``encoders.pool_size``) rather than per-encoder — so two concurrent
        #: ``embed()`` calls on THIS instance can both be inside ``_embed_sync``.
        #: Serialize tokenization: the forward pass itself is stateless and stays
        #: outside the lock, so concurrent callers still overlap on inference.
        self._tokenize_lock = threading.Lock()
        self._pad_id = _resolve_pad_id(tokenizer)

        self.id = spec.repo_id
        self.dim = spec.dim
        self.max_tokens = spec.max_tokens
        self.query_prefix = spec.query_prefix
        self.passage_prefix = spec.passage_prefix
        self.normalize = spec.normalize
        self.pooling = spec.pooling

    async def embed(
        self, texts: Sequence[str], *, kind: Literal["query", "passage"]
    ) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prefix = self.query_prefix if kind == "query" else self.passage_prefix
        prefixed = [f"{prefix}{t}" for t in texts] if prefix else list(texts)
        async with self._semaphore:
            return await asyncio.to_thread(self._embed_sync, prefixed)

    def _embed_sync(self, texts: list[str]) -> NDArray[np.float32]:
        if self._spec.kind == "bi_encoder_ragged":
            vecs = self._embed_ragged(texts)
        else:
            vecs = self._embed_padded(texts)
        return _l2_normalize(vecs) if self.normalize else vecs

    def _embed_padded(self, texts: list[str]) -> NDArray[np.float32]:
        """Length-bucketed padded inference.

        Tokenize unpadded to learn true lengths, sort by length, run each bucket
        padded only to its own width, then scatter the rows back to their input
        positions. The scatter is the load-bearing step: ``_materialize_batch``
        pairs ``vecs[i]`` with ``pending_rows[i]`` positionally, so a permutation
        bug here would attach every vector to the wrong chunk id and fail
        silently at query time rather than raising.
        """
        ids_per_text = self._tokenize_unpadded(texts)
        lengths = [len(ids) for ids in ids_per_text]
        order = sorted(range(len(texts)), key=lengths.__getitem__)
        out: NDArray[np.float32] | None = None
        for bucket in _length_buckets(order, lengths):
            width = max(*(lengths[j] for j in bucket), 1)
            input_ids = np.full((len(bucket), width), self._pad_id, dtype=np.int64)
            attention_mask = np.zeros((len(bucket), width), dtype=np.int64)
            for row, j in enumerate(bucket):
                ids = ids_per_text[j]
                input_ids[row, : len(ids)] = ids
                attention_mask[row, : len(ids)] = 1
            vecs = self._run_padded(input_ids, attention_mask)
            if out is None:
                out = np.empty((len(texts), vecs.shape[1]), dtype=np.float32)
            # Scatter by original index — restores input order across buckets.
            out[bucket] = vecs
        return out if out is not None else np.zeros((0, self.dim), dtype=np.float32)

    def _tokenize_unpadded(self, texts: list[str]) -> list[list[int]]:
        """Token ids per text, truncated but NOT padded (see :pyattr:`_tokenize_lock`)."""
        with self._tokenize_lock:
            self._tokenizer.no_padding()
            self._tokenizer.enable_truncation(max_length=self.max_tokens)
            encoded = self._tokenizer.encode_batch(texts)
            return [list(e.ids) for e in encoded]

    def _run_padded(
        self, input_ids: NDArray[np.int64], attention_mask: NDArray[np.int64]
    ) -> NDArray[np.float32]:
        outputs = self._session.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        output_map = dict(zip((o.name for o in self._session.get_outputs()), outputs, strict=True))
        if "sentence_embedding" in output_map:
            return np.asarray(output_map["sentence_embedding"], dtype=np.float32)
        last_hidden = np.asarray(output_map["last_hidden_state"], dtype=np.float32)
        return _mean_pool(last_hidden, attention_mask)

    def _embed_ragged(self, texts: list[str]) -> NDArray[np.float32]:
        with self._tokenize_lock:  # shared tokenizer state; see _tokenize_lock
            self._tokenizer.no_padding()
            self._tokenizer.no_truncation()
            encoded = self._tokenizer.encode_batch(texts, add_special_tokens=False)
        lengths = [max(len(e.ids), 1) for e in encoded]  # avoid a zero-length bag
        offsets = np.array([0, *np.cumsum(lengths)[:-1]], dtype=np.int64)
        flat_ids = np.array([tid for e in encoded for tid in (e.ids or [0])], dtype=np.int64)
        (embeddings,) = self._session.run(None, {"input_ids": flat_ids, "offsets": offsets})
        return np.asarray(embeddings, dtype=np.float32)


def load_tokenizer(path: Path) -> Tokenizer:
    """Load a ``tokenizer.json`` file."""
    return Tokenizer.from_file(str(path))


__all__ = ["OnnxBiEncoder", "load_tokenizer"]
