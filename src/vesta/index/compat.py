"""Index-settings enforcement.

The index records the embedder it was built with (``index_meta`` table). A query
whose embedder differs from the index's embedder is **refused, not silently
served**: a mismatched embedder returns plausible-looking garbage — the worst
possible failure mode for a grounded-answer product.

Two enforcement points:

* **Index time** (here): when the configured embedder changes vs an archive's
  recorded ``IndexMeta``, that archive's ``zims.index_status`` is marked
  ``stale`` and the re-index cost is surfaced via the estimator ("Changing
  the embedder marks the archive's index stale with the re-index cost shown
  before confirming").
* **Query time** (dense retrieval): compares the live query
  embedder against ``store.describe(zim_id)`` and drops a mismatched archive
  rather than searching it. That logic lives in the retriever; this module
  exposes the pure comparison it shares.

Pure comparison + a thin DB write helper; unit-testable with fakes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vesta.vectors.contracts import IndexMeta

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vesta.db.connection import Database
    from vesta.vectors.contracts import VectorStore


@dataclass(frozen=True)
class EmbedderFingerprint:
    """The subset of an encoder's identity that must match an index.

    ``embedder_id`` is the HF repo id; ``dim``/``prefixes``/``pooling``/
    ``normalize`` are the runtime shape the registry owns (``encoders/registry``).
    Two embedders with the same repo id but different settings is impossible by
    construction (the registry is the single source), so the repo id alone is
    usually enough — the rest is belt-and-braces against a hand-edited row.
    """

    embedder_id: str
    dim: int
    query_prefix: str
    passage_prefix: str
    pooling: str
    normalize: bool


def fingerprint_from_meta(meta: IndexMeta) -> EmbedderFingerprint:
    return EmbedderFingerprint(
        embedder_id=meta.embedder_id,
        dim=meta.dim,
        query_prefix=meta.query_prefix,
        passage_prefix=meta.passage_prefix,
        pooling=meta.pooling,
        normalize=meta.normalize,
    )


def fingerprint_from_spec(spec: Any) -> EmbedderFingerprint:
    """Fingerprint from an encoder-registry ``ModelSpec`` (duck-typed).

    ``index/`` must not import ``encoders/`` (module boundaries), so the spec
    arrives as ``Any``; the attribute names are the ``ModelSpec`` contract
    (``encoders/registry.py``). The composition root (``main``) resolves the
    spec for the configured ``index.embedder`` / ``encoders.embed.model`` and
    hands the fingerprint in — pure registry data, no session load.
    """
    return EmbedderFingerprint(
        embedder_id=str(spec.repo_id),
        dim=int(spec.dim),
        query_prefix=str(spec.query_prefix),
        passage_prefix=str(spec.passage_prefix),
        pooling=str(spec.pooling),
        normalize=bool(spec.normalize),
    )


def is_compatible(recorded: IndexMeta | None, configured: EmbedderFingerprint) -> bool:
    """True iff an archive indexed with ``recorded`` is queryable with
    ``configured``. ``None`` means "not indexed" → incompatible (don't search)."""
    if recorded is None:
        return False
    rec = fingerprint_from_meta(recorded)
    return (
        rec.embedder_id == configured.embedder_id
        and rec.dim == configured.dim
        and rec.query_prefix == configured.query_prefix
        and rec.passage_prefix == configured.passage_prefix
        and rec.pooling == configured.pooling
        and rec.normalize == configured.normalize
    )


async def mark_stale(db: Database, zim_id: int, *, reason: str = "embedder changed") -> None:
    """Flag an archive's index out-of-sync with the configured embedder.

    Sets ``zims.index_status='stale'`` so the UI shows the re-index cost before
    confirming; the vectors stay on disk (deleting them would lose the work
    silently). A stale archive is **not** served: the dense source's fingerprint
    check refuses it at query time, and the startup capability seed counts only
    ``complete``/``running`` archives toward ``Capability.VECTORS`` — so a box
    whose only indexes are stale behaves as depth-0 (degrade-don't-fail) until
    re-indexed."""
    async with db.write() as conn:
        await conn.execute("UPDATE zims SET index_status='stale' WHERE id=?", (zim_id,))


async def reconcile_stale(
    db: Database,
    store: VectorStore,
    configured: Sequence[EmbedderFingerprint],
) -> list[int]:
    """Mark every indexed archive whose recorded embedder differs from the
    configured one(s) ``stale``; clear ``stale`` back to ``complete`` for
    archives that match again ("Changing the embedder marks the archive's
    index stale with the re-index cost shown before confirming").

    ``configured`` carries the fingerprints of every setting that must agree
    with the index for it to be usable: the build embedder (``index.embedder``)
    AND the query embedder (``encoders.embed.model``). A recorded meta that
    mismatches EITHER marks the archive stale — the index cannot be served
    (query mismatch, refused by the dense source) or a re-index would silently
    change vector spaces (build mismatch). Symmetric un-marking keeps the
    change-it-back flow honest without a re-index.

    Called once at startup by the composition root (both embedder settings are
    ``hot=False``, so a change always arrives via restart). Returns the ids
    newly marked stale, for the startup log line.
    """
    async with db.read() as conn, conn.execute("SELECT zim_id FROM index_meta") as cur:
        rows = await cur.fetchall()
    newly_stale: list[int] = []
    for row in rows:
        zim_id = int(row["zim_id"])
        meta = await store.describe(zim_id)
        compatible = bool(configured) and all(is_compatible(meta, fp) for fp in configured)
        if not compatible:
            await mark_stale(db, zim_id)
            newly_stale.append(zim_id)
        else:
            async with db.write() as conn:
                await conn.execute(
                    "UPDATE zims SET index_status='complete' WHERE id=? AND index_status='stale'",
                    (zim_id,),
                )
    return newly_stale


__all__ = [
    "EmbedderFingerprint",
    "fingerprint_from_meta",
    "fingerprint_from_spec",
    "is_compatible",
    "mark_stale",
    "reconcile_stale",
]
