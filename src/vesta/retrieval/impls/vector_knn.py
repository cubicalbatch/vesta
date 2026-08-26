"""Stage A3 — dense (vector kNN) candidate source.

A registered ``CandidateSource`` like any other: ``requires = {Capability.VECTORS}``,
so on a depth-0 box (no semantic index) the profile drops it and records the drop
— no branching anywhere in the pipeline. It is the first
source that returns a ``Candidate`` **with a score** (lexical sources can't).

One logically global index, physically sharded by ``zim_id``: the store's
partition-key pre-filter restricts the scan
to the scoped archive(s), and the dense side contributes a genuinely
cross-archive-comparable score, unlike lexical.

**Index-settings enforcement**: before
searching, each scoped archive's recorded embedder fingerprint
(``store.describe(zim_id)``) is compared against the live query embedder. A
mismatched archive is **dropped with a trace entry**, never searched — a
mismatched embedder returns plausible-looking garbage, the worst failure mode for
a grounded-answer product. ``retrieval/`` cannot import ``index/`` (forbidden),
so the comparison is inline (a few field checks against the duck-typed
``IndexMeta`` the store returns); the canonical helper lives in ``index/compat``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

from vesta.config.capabilities import Capability
from vesta.retrieval.contracts import Candidate, PreparedQuery, Scope
from vesta.retrieval.registry import register

if TYPE_CHECKING:
    from vesta.encoders.manager import EncoderManager
    from vesta.retrieval.trace import Trace
    from vesta.vectors.contracts import VectorStore
    from vesta.zim.registry import ArchiveRegistry


@register("candidate_source", "vector_knn")
class VectorKnn:
    """Dense kNN candidate source over the semantic index."""

    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.VECTORS})

    class Params(BaseModel):
        #: Neighbours fetched per query before per-article dedup.
        k: int = 40
        #: A/B toggle: False drops this source without removing it from the profile.
        enabled: bool = True

    def __init__(
        self,
        params: Params | None = None,
        archives: ArchiveRegistry | None = None,
        vectors: VectorStore | None = None,
        encoders: EncoderManager | None = None,
    ) -> None:
        self._params = params or self.Params()
        self._archives = archives
        self._vectors = vectors
        self._encoders = encoders

    async def find(self, q: PreparedQuery, scope: Scope, tr: Trace) -> list[Candidate]:
        if not self._params.enabled or self._vectors is None or self._encoders is None:
            return []
        if self._archives is None:
            return []

        from vesta.retrieval.impls._scope import archives_for_scope

        archives = await archives_for_scope(self._archives, scope)
        if not archives:
            return []

        encoder = await self._encoders.get_embed()
        if encoder is None:
            # No embed model → can't embed the query. Treat as "no dense candidates"
            # (the capability probe should have dropped us, but be defensive).
            return []

        # Index-settings enforcement: keep only archives whose recorded embedder
        # matches the live query embedder.
        compatible_zim_ids: list[int] = []
        for archive in archives:
            meta = await self._vectors.describe(archive.id)
            if meta is None:
                continue  # not indexed — skip silently
            if not _matches(meta, encoder):
                tr.degraded(
                    component="candidate_source/vector_knn",
                    missing=Capability.VECTORS,
                    reason=(
                        f"archive {archive.id} indexed with {meta.embedder_id} "
                        f"(dim {meta.dim}); query embedder is {encoder.id} "
                        f"(dim {encoder.dim}) — mismatched index refused"
                    ),
                )
                continue
            compatible_zim_ids.append(archive.id)
        if not compatible_zim_ids:
            return []

        qvec = await encoder.embed([q.text or q.raw], kind="query")
        hits = await self._vectors.search(qvec[0], zim_ids=compatible_zim_ids, k=self._params.k)

        # Dedup by (zim_id, path): a depth-2/3 index can return several chunks of
        # the same article; Stage A nominates articles, not passages, so
        # keep the strongest hit per article. Hits arrive score-interleaved
        # across archives, so ranks count per zim_id — the Candidate contract
        # scopes ``rank`` to its (zim_id, source) group only.
        seen: set[tuple[int, str]] = set()
        next_rank: dict[int, int] = {}
        out: list[Candidate] = []
        for hit in hits:
            key = (hit.zim_id, hit.path)
            if key in seen:
                continue
            seen.add(key)
            rank = next_rank.get(hit.zim_id, 0)
            next_rank[hit.zim_id] = rank + 1
            out.append(
                Candidate(
                    zim_id=hit.zim_id,
                    path=hit.path,
                    source="vector_knn",
                    rank=rank,
                    score=float(hit.score),
                )
            )
        return out


def _matches(meta: Any, encoder: Any) -> bool:
    """Inline embedder-fingerprint comparison.

    ``retrieval/`` cannot import ``index/compat`` (forbidden), so the check is
    inlined; it mirrors :func:`vesta.index.compat.is_compatible` field-for-field.
    A mismatch on any of repo id / dim / prefixes / pooling / normalization means
    the index and the query embedder produce incompatible vector spaces."""
    return (
        str(meta.embedder_id) == str(encoder.id)
        and int(meta.dim) == int(encoder.dim)
        and str(meta.query_prefix) == str(encoder.query_prefix)
        and str(meta.passage_prefix) == str(encoder.passage_prefix)
        and str(meta.pooling) == str(encoder.pooling)
        and bool(meta.normalize) == bool(encoder.normalize)
    )


__all__ = ["VectorKnn"]
