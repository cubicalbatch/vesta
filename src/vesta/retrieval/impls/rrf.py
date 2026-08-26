"""RRF (Reciprocal Rank Fusion) — merge candidates from multiple sources.

Registered as ``fuser`` ``rrf``. Implements RRF with k=20
(equal weights, k tuned from the literature default of 60 down to 20 for our
short lists). Receives groups keyed by ``FusionKey(zim_id, source)``.

Two modes, controlled by profile params:

``across_archives: union`` (the default, ``lexical`` profile):
    RRF within each archive (merge all sources for that archive), then flat
    union across archives. Ranks from different archives are NOT comparable
    (``Albert_Einstein`` and ``questions/tagged/find_page=56`` both
    rank 1). This mode respects that reality.

``across_archives: rrf`` (global RRF across all archives):
    RRF across ALL candidates globally — the known-bad control that
    predicts cross-archive ranking failures.

Speed: O(N) per group, trivial.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

from vesta.config.capabilities import Capability
from vesta.retrieval.contracts import Candidate, FusionKey
from vesta.retrieval.registry import register

if TYPE_CHECKING:
    from vesta.retrieval.trace import Trace


@dataclass(frozen=True)
class _CandidateWithScore:
    """Internal: a candidate with its RRF score attached for sorting."""

    candidate: Candidate
    rrf_score: float


def _rrf_fuse(candidates: list[Candidate], k: int = 20) -> list[Candidate]:
    """RRF score = Σ 1/(k + rank) across all sources within one group.

    Ties are common (same candidate returned by multiple sources); the RRF score
    naturally sums their contributions. Returns candidates sorted by RRF score
    descending, re-ranked from 0.

    Deterministic under delivery order: contributions are accumulated in a
    canonically sorted pass (float addition is not associative), duplicate
    ``(zim_id, path)`` pairs keep their smallest ``(rank, source)``
    representative, and exact score ties are broken by ``(zim_id, path)`` —
    so identical inputs fuse identically no matter which concurrent source
    finished first.

    Candidates are keyed by ``(zim_id, path)``, never by bare path: two
    archives may legitimately expose the same path, and fusing them into one
    entry would attribute both nominations to a single archive and silently
    drop the other's downstream extraction.
    """
    by_key: dict[tuple[int, str], _CandidateWithScore] = {}
    for c in sorted(candidates, key=lambda c: (c.zim_id, c.path, c.rank, c.source)):
        key = (c.zim_id, c.path)
        if key not in by_key:
            by_key[key] = _CandidateWithScore(candidate=c, rrf_score=1.0 / (k + c.rank))
        else:
            existing = by_key[key]
            by_key[key] = _CandidateWithScore(
                candidate=existing.candidate,
                rrf_score=existing.rrf_score + 1.0 / (k + c.rank),
            )

    sorted_ = sorted(
        by_key.values(), key=lambda x: (-x.rrf_score, x.candidate.zim_id, x.candidate.path)
    )

    result: list[Candidate] = []
    for rank, item in enumerate(sorted_):
        result.append(
            Candidate(
                zim_id=item.candidate.zim_id,
                path=item.candidate.path,
                source="rrf",
                rank=rank,
                score=item.rrf_score,
            )
        )
    return result


@register("fuser", "rrf")
class RRF:
    """Reciprocal Rank Fusion fuser (k=20, equal weights)."""

    requires: ClassVar[frozenset[Capability]] = frozenset()

    class Params(BaseModel):
        k: int = 20
        across_archives: str = "union"  # "union" or "rrf"

    def __init__(self, params: Params | None = None) -> None:
        self._params = params or self.Params()

    def fuse(self, groups: Mapping[FusionKey, Sequence[Candidate]], tr: Trace) -> list[Candidate]:
        """Fuse candidates per the profile's strategy.

        ``across_archives: union`` (default): RRF within each archive, then flat
        union. ``across_archives: rrf``: merge everything globally — the
        known-bad case.
        """
        k = self._params.k
        across = self._params.across_archives

        if across == "rrf":
            # Global RRF: flatten all groups into one list and fuse.
            all_cands: list[Candidate] = []
            for _key, candidates in groups.items():
                for c in candidates:
                    all_cands.append(
                        Candidate(
                            zim_id=c.zim_id,
                            path=c.path,
                            source=c.source,
                            rank=c.rank,
                            score=c.score,
                        )
                    )
            return _rrf_fuse(all_cands, k=k)

        # Per-archive RRF + cross-archive union. Within-archive scores are NOT
        # comparable across archives, so the union interleaves archives
        # by within-archive rank (round-robin) instead of concatenating by zim_id
        # — otherwise the lowest-numbered archive would dominate ``cands[:N]``
        # downstream and bias multi-archive retrieval.
        by_zim: dict[int, list[Candidate]] = {}
        for key, candidates in groups.items():
            by_zim.setdefault(key.zim_id, []).extend(list(candidates))

        fused_per_archive: list[list[Candidate]] = []
        for zim_id in sorted(by_zim):
            fused_per_archive.append(_rrf_fuse(by_zim[zim_id], k=k))

        result: list[Candidate] = []
        overall_rank = 0
        # Round-robin: take rank 0 from every archive, then rank 1, …
        max_len = max((len(lst) for lst in fused_per_archive), default=0)
        for i in range(max_len):
            for lst in fused_per_archive:
                if i < len(lst):
                    c = lst[i]
                    result.append(
                        Candidate(
                            zim_id=c.zim_id,
                            path=c.path,
                            source="rrf",
                            rank=overall_rank,
                            score=c.score,
                        )
                    )
                    overall_rank += 1

        return result
