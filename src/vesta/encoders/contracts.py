"""Encoder contracts — the two Protocols every model role implements.

Three model roles share these two Protocols:

* ``static`` (Stage B1 shortlister) and ``embed`` (the indexing bi-encoder)
  both satisfy :class:`Encoder` — the *shape* of the work (text in, vector out)
  is identical even though the runtimes differ (a ragged EmbeddingBag-style ONNX
  graph for the static model vs a padded-batch transformer for ``embed``).
* ``rerank`` satisfies :class:`CrossEncoder` — a joint query/passage forward
  pass, not a vector.

**The registry owns prefixes, pooling and normalization — never caller code**:
``query_prefix``/``passage_prefix``/``normalize``/``pooling`` are
attributes of the encoder instance itself, populated from
``encoders/registry.py`` at construction time. A caller that wants a query
embedding never composes a prefix by hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray


class Encoder(Protocol):
    """A bi-encoder (or static embedder): text in, one vector out per text.

    ``kind`` selects which of ``query_prefix``/``passage_prefix`` is applied —
    the caller never composes the prefix itself. Returns an ``(n, dim)`` float32
    array, L2-normalized iff ``normalize`` is true.
    """

    id: str
    dim: int
    max_tokens: int
    query_prefix: str
    passage_prefix: str
    normalize: bool
    pooling: Literal["cls", "mean"]

    async def embed(
        self, texts: Sequence[str], *, kind: Literal["query", "passage"]
    ) -> NDArray[np.float32]: ...


class CrossEncoder(Protocol):
    """A joint query/passage scorer: one float per passage, higher = more relevant.

    ``max_tokens`` is the instance's configured truncation length (not
    necessarily the model's architectural ceiling — Vesta truncates
    Stage B2 inputs to 256 tokens for bounded latency).
    """

    id: str
    max_tokens: int

    async def score(self, query: str, passages: Sequence[str]) -> Sequence[float]: ...


__all__ = ["CrossEncoder", "Encoder"]
