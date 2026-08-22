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

from collections.abc import Sequence
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


def is_near_duplicate(
    candidate: ScoredPassage,
    selected: Sequence[ScoredPassage],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> bool:
    """True if ``candidate`` is a near-exact duplicate of any passage already
    in ``selected`` (word-bigram Jaccard, archive-blind by design)."""
    cand_bg = _bigrams(candidate.passage.text)
    if len(cand_bg) < _MIN_BIGRAMS:
        return False
    for other in selected:
        other_bg = _bigrams(other.passage.text)
        if not other_bg:
            continue
        union = cand_bg | other_bg
        if not union:
            continue
        jaccard = len(cand_bg & other_bg) / len(union)
        if jaccard > threshold:
            return True
    return False


__all__ = ["DEFAULT_THRESHOLD", "is_near_duplicate"]
