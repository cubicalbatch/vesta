"""ONNX cross-encoder — the ``rerank`` model role, Stage B2.

A sequence-pair classification forward pass: ``(query, passage)`` tokenized as
one packed sequence (``[CLS] query [SEP] passage [SEP]``, via
``tokenizer.encode_batch`` on tuples) through a BERT-family classifier head
producing one logit per pair. The raw logit is unbounded (e.g.
+7.5 for a relevant pair, -11.1 for an irrelevant one) — a sigmoid maps it to
``(0, 1)`` for a score that is directly comparable across passages and
provides a trustworthy *cross-archive* score in the system. Sigmoid
is monotonic, so it changes nothing about the ranking, only the scale.

``max_tokens`` is fixed at construction (from ``retrieval.rerank.truncate_tokens``,
default 256) — the caller never passes truncation length per call.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from vesta.encoders.registry import ModelSpec


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class OnnxCrossEncoder:
    """A :class:`~vesta.encoders.contracts.CrossEncoder` backed by one ONNX graph."""

    def __init__(
        self,
        spec: ModelSpec,
        session: ort.InferenceSession,
        tokenizer: Tokenizer,
        *,
        max_tokens: int,
        semaphore: asyncio.Semaphore,
    ) -> None:
        self._session = session
        self._tokenizer = tokenizer
        self._semaphore = semaphore
        self.id = spec.repo_id
        self.max_tokens = max_tokens

    async def score(self, query: str, passages: Sequence[str]) -> Sequence[float]:
        if not passages:
            return []
        async with self._semaphore:
            return await asyncio.to_thread(self._score_sync, query, list(passages))

    def _score_sync(self, query: str, passages: list[str]) -> list[float]:
        self._tokenizer.enable_padding()
        self._tokenizer.enable_truncation(max_length=self.max_tokens)
        encoded = self._tokenizer.encode_batch([(query, p) for p in passages])
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encoded], dtype=np.int64)
        (logits,) = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        scores = _sigmoid(np.asarray(logits, dtype=np.float32).reshape(-1))
        return [float(s) for s in scores]


__all__ = ["OnnxCrossEncoder"]
