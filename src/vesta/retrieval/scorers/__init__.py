"""Stage B passage scorers.

Two entries in the ``PassageScorer`` chain ("scores from N+1 replace scores
from N") implement Stage B's two-pass design: ``static_pass`` (B1, shortlist
~200→20) then ``cross_encoder`` (B2, rerank the shortlist with the only
cross-corpus-comparable score in the system). Importing these modules here
registers them.
"""

from __future__ import annotations

from vesta.retrieval.scorers import cross_encoder, static_pass

__all__ = ["cross_encoder", "static_pass"]
