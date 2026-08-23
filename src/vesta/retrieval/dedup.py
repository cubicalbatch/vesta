"""Near-exact passage dedup.

Two distinct duplicate sources this covers with **one** mechanism:

* **Cross-archive duplicates** — Wikipedia and a mirror of it will both match
  the same query and produce near-identical passage text under different
  ``zim_id``s. The check below never looks at ``zim_id``, so it catches this
  automatically.
* **Soft-redirect duplicates** —
  ``LocalArchive.extract`` already returns empty text for a soft-redirect
  entry (``zim/registry.py``), so a soft-redirect *article* never reaches
  Stage B as passages at all. What remains at this layer is near-duplicate
  *content* between two differently-titled candidates that both resolved to
  (near) the same underlying text — the same Jaccard check catches it without
  a second, redirect-specific code path.

Word-*bigram* Jaccard (not unigram bag-of-words): two passages sharing many
individual words but a different word order score low here, which is the
point — dedup should catch copy-paste/near-copy text, not "same topic".
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import pairwise

from vesta.retrieval.contracts import ScoredPassage

#: Above this Jaccard similarity, two passages are treated as duplicates.
#: Empirically: near-identical Wikipedia-mirror passages score >0.9; two
#: independent passages about the same topic rarely clear 0.5.
DEFAULT_THRESHOLD = 0.85

#: Passages shorter than this many bigrams are never flagged as duplicates —
#: short text produces unstable Jaccard scores (a two-word passage matches
#: almost anything).
_MIN_BIGRAMS = 4


def _bigrams(text: str) -> frozenset[tuple[str, str]]:
    words = text.lower().split()
    return frozenset(pairwise(words))


def _any_near_duplicate(
    cand_bg: frozenset[tuple[str, str]],
    selected_bigrams: Iterable[frozenset[tuple[str, str]]],
    *,
    threshold: float,
) -> bool:
    """The Jaccard sweep itself, over already-computed bigram sets — both the
    one-shot :func:`is_near_duplicate` and the incremental
    :class:`NearDuplicateGate` share it so their verdicts cannot drift."""
    if len(cand_bg) < _MIN_BIGRAMS:
        return False
    for other_bg in selected_bigrams:
        if not other_bg:
            continue
        union = cand_bg | other_bg
        if not union:
            continue
        jaccard = len(cand_bg & other_bg) / len(union)
        if jaccard > threshold:
            return True
    return False


def is_near_duplicate(
    candidate: ScoredPassage,
    selected: Sequence[ScoredPassage],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> bool:
    """True if ``candidate`` is a near-exact duplicate of any passage already
    in ``selected`` (word-bigram Jaccard, archive-blind by design)."""
    cand_bg = _bigrams(candidate.passage.text)
    return _any_near_duplicate(
        cand_bg, (_bigrams(other.passage.text) for other in selected), threshold=threshold
    )


class NearDuplicateGate:
    """Incremental near-duplicate check for a selection loop that only ever
    grows: each accepted passage's bigram set is computed once and reused for
    every subsequent candidate, instead of re-tokenizing the whole selection
    per candidate (O(selected²) splits → O(selected)).

    Semantically identical to calling :func:`is_near_duplicate` against the
    list of everything accepted so far — call sites append/accept in lockstep.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self._threshold = threshold
        self._selected_bigrams: list[frozenset[tuple[str, str]]] = []

    def is_near_duplicate(self, candidate: ScoredPassage) -> bool:
        """True if ``candidate`` duplicates anything previously :meth:`accept`ed."""
        return _any_near_duplicate(
            _bigrams(candidate.passage.text), self._selected_bigrams, threshold=self._threshold
        )

    def accept(self, sp: ScoredPassage) -> None:
        """Record ``sp`` as part of the selection future candidates are
        checked against. Call exactly when the caller appends it to its own
        ``selected`` list."""
        self._selected_bigrams.append(_bigrams(sp.passage.text))


__all__ = ["DEFAULT_THRESHOLD", "NearDuplicateGate", "is_near_duplicate"]
