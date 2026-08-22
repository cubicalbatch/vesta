"""Abstention messaging — the shared no-match abstention marker.

Emitted by ``GET /api/answer`` when retrieval finds no candidates, and
recognized by the benchmark as the abstention marker.
"""

from __future__ import annotations

#: Harness-side abstention string (Appendix A.4 notes — no model involved).
#: Emitted as a token when retrieval returns no candidates; the benchmark's
#: reduction matches it to flag an abstention.
ABSTENTION_NO_MATCH = (
    "No passage in your archives closely matches this query. The closest results "
    "are below — you may want to try different wording, or a broader term."
)


__all__ = [
    "ABSTENTION_NO_MATCH",
]
