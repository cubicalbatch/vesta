"""Vector store contracts — Protocol + frozen domain types.

A narrow interface with one implementation (``vectors/`` depends only on
``db``). Callers program to ``VectorStore``, never to concrete implementations.

Why these shapes:

* ``VectorRow.id`` is ``chunks.id`` (the vec0 primary key), NOT ``articles.id``.
  Depths 2/3 have more vectors than articles, so ``articles.id``-as-vector-id only
  works at depth 1. The uniform ``chunks`` table (migration 0004) keeps one
  id-space across all depths; at depth 1 chunk↔article is 1:1.
* ``VectorHit`` carries ``article_id`` + ``char_start``/``char_end`` so the dense
  retriever can recover the entry path (join ``articles.entry_path``) and the
  text span from the ZIM — no article text is stored in the vector table.
* ``IndexMeta`` records the embedder shape an archive's index was built with, so a
  query against a mismatched index is refused, not silently served (a mismatched
  embedder returns plausible-looking garbage).

``vectors/`` must not import ``retrieval``/``answer``/``api``/``index`` (the
dense source receives the store via DI, never by import).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True)
class VectorRow:
    """One vector to upsert.

    ``id`` is ``chunks.id`` and the vec0 primary key. ``article_id`` is the FK to
    ``articles.id`` so the retriever can join to ``entry_path``. ``char_start``/
    ``char_end`` recover the span from the ZIM at query time (no text stored).
    ``ordinal``/``depth`` are deliberately not here — they are the indexer's
    concern; the store's upsert reconciles the chunk row on the ``id`` PK and
    leaves ``ordinal``/``depth`` untouched.
    """

    id: int
    zim_id: int
    article_id: int
    embedding: NDArray[np.float32]
    char_start: int | None
    char_end: int | None


@dataclass(frozen=True)
class VectorHit:
    """One search result — enough to build a ``Candidate``.

    ``score`` is the vec0 distance negated (``-distance``), so **higher is more
    similar**, matching ``Candidate.score``'s "higher is better" convention
    (retrieval/contracts.py). For L2-normalized embeddings (Granite, GTE) vec0's
    default L2 distance is monotonically related to cosine, so ranking is
    cosine-faithful; consumers may rescale to a comparable score if desired.

    ``path`` is the article entry path (the store joins ``chunks``→``articles`` so
    the dense source in ``retrieval/`` never touches the DB — ``retrieval`` must
    not depend on ``db``). ``article_id`` + ``char_start``/``char_end`` let
    callers re-extract the span from the ZIM if desired.
    """

    zim_id: int
    article_id: int
    chunk_id: int
    path: str
    score: float
    char_start: int | None
    char_end: int | None


@dataclass(frozen=True)
class IndexMeta:
    """The embedder shape an archive's index was built with. Persisted per-zim
    (table ``index_meta``) so a mismatched query is refused, not silently served."""

    embedder_id: str
    dim: int
    query_prefix: str
    passage_prefix: str
    pooling: str  # "cls" | "mean" (encoders/contracts.py Encoder.pooling)
    normalize: bool


@dataclass(frozen=True)
class StoreStats:
    """Snapshot of vector-store occupancy for the admin/eval surface."""

    total_rows: int
    per_dim: dict[int, int]  # dim → row count in vectors_d{dim}
    per_zim: dict[int, int]  # zim_id → row count (summed across dims via chunks)
    disk_bytes: int
    vec0_available: bool  # False ⇒ semantic index disabled on this build


class VectorStore(Protocol):
    """The narrow vector-store interface. One implementation ships
    (``SqliteVecStore``); alternative backends can be implemented as drop-ins
    behind this Protocol.

    The store resolves the correct ``vectors_d{N}`` table by dimension internally
    — callers never name the table (one vec0 table per distinct embedding
    dimension, resolved through the interface, never named in caller code).
    """

    async def upsert(self, rows: Sequence[VectorRow]) -> None:
        """Persist vectors + their chunk metadata. Idempotent on ``VectorRow.id``
        (the chunk ``id`` PK); re-upserting the same id updates the embedding and
        char spans without disturbing the indexer's ``ordinal``/``depth``."""
        ...

    async def search(
        self, q: NDArray[np.float32], *, zim_ids: Sequence[int], k: int
    ) -> list[VectorHit]:
        """k-NN over the table matching ``q``'s dimensionality, pre-filtered to
        ``zim_ids`` via the vec0 ``zim_id`` PARTITION KEY (a genuine pre-filter,
        not a post-filter)."""
        ...

    async def delete_by_zim(self, zim_id: int) -> None:
        """Remove every vector + chunk row for an archive. vec0 tables are NOT
        FK-cascaded, so this is an explicit per-dim ``DELETE ... WHERE zim_id = ?``
        plus the chunks delete."""
        ...

    async def delete_by_article(self, zim_id: int, article_id: int) -> None:
        """Remove every vector + chunk row for one article (clean re-processing of
        an article interrupted mid-write so no orphaned vectors survive a worker
        crash). Deletes that article's chunk ids from every ``vectors_d{N}`` table,
        then the chunk rows themselves."""
        ...

    async def stats(self) -> StoreStats: ...

    async def record_index_meta(self, zim_id: int, meta: IndexMeta) -> None:
        """Record the embedder an archive was indexed with. The indexer calls
        this once per index build."""
        ...

    async def describe(self, zim_id: int) -> IndexMeta | None:
        """Read an archive's embedder metadata, or ``None`` if it has no index.
        The dense source compares this against the live query embedder and
        refuses to search a mismatched index."""
        ...


__all__ = [
    "IndexMeta",
    "StoreStats",
    "VectorHit",
    "VectorRow",
    "VectorStore",
]
