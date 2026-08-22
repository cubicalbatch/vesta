"""Retrieval contracts — frozen domain types and Protocols.

Every swappable pipeline stage is a Protocol; every implementation is registered
against one of these contracts. The seam decisions encode key design invariants:

* ``Candidate.score`` is ``Optional`` — lexical sources produce no score, so
  nothing downstream may assume one exists.
* ``Fuser`` receives groups keyed by ``FusionKey(zim_id, source)`` — making
  the archive dimension explicit allows fusing within each archive then unioning
  across archives.
* ``PassageScorer`` is a chain, not a single component — Stage B's two-pass
  design (static → cross-encoder) is represented as entries in a list.

``retrieval/`` must not import ``answer/``, ``api/``, or ``index/`` (enforced by
``tests/test_boundaries.py``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

from vesta.config.capabilities import Capability
from vesta.zim.types import EntryPath, Passage

if TYPE_CHECKING:
    from vesta.retrieval.trace import Trace


# ── Domain types ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Scope:
    """Which archives a query runs over.

    ``None`` means "every enabled archive". ``corpus_labels`` scopes by the
    user-facing label (e.g. ``{"wikipedia", "stackexchange"}``).
    """

    zim_ids: frozenset[int] | None = None
    corpus_labels: frozenset[str] | None = None


@dataclass(frozen=True)
class PreparedQuery:
    """The query after every preparer has processed it.

    ``terms`` are normalized for lexical use; ``text`` is the running-text form
    for dense/scoring use; ``aliases`` are terms appended from the ZIM redirect
    table; ``is_keyword_query`` distinguishes a bare-term search from a
    natural-language question; ``rung`` records which query-preprocessing rung
    was chosen.

    ``history`` is the prior conversation as ``(role, content)`` string pairs.
    It is deliberately a plain tuple-of-pairs, NOT ``ChatMessage``:
    ``retrieval/`` must not import ``inference/`` (and the boundary AST walker
    counts ``if TYPE_CHECKING`` imports too), so the cross-package type is
    avoided entirely. The composition root adapts ``ChatMessage`` ↔ this shape.
    """

    raw: str
    terms: tuple[str, ...]
    text: str
    aliases: tuple[str, ...]
    is_keyword_query: bool
    rung: str
    history: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Candidate:
    """One article nominated by a CandidateSource.

    ``score`` is ``None`` for lexical sources — python-libzim exposes neither
    ``getScore`` nor ``getSnippet``. ``rank`` is WITHIN its ``(zim_id,
    source)`` group only and is not comparable across archives or sources.
    """

    zim_id: int
    path: EntryPath
    source: str  # component id that nominated it
    rank: int  # within archive+source only
    score: float | None  # None for lexical


@dataclass(frozen=True)
class FusionKey:
    """A ``(zim_id, source)`` group key for the Fuser.

    Making the archive dimension explicit is what lets the Fuser express "RRF
    within each archive, flat union across archives".
    """

    zim_id: int
    source: str  # component id


@dataclass(frozen=True)
class ScoredPassage:
    """A Passage with a score assigned by a PassageScorer.

    ``source_info`` records which component produced the current score so the
    trace can distinguish static-embedder scores from cross-encoder scores.
    """

    passage: Passage
    score: float
    source_info: str  # component id of the scorer


@dataclass(frozen=True)
class SourceCard:
    """What appears in the UI alongside search results.

    ``snippet`` is a short extract (first ~200 chars of the article lead or
    cross-encoder alignment snippet).
    """

    zim_id: int
    path: EntryPath
    title: str
    snippet: str
    breadcrumb: str
    score: float | None
    source: str  # which CandidateSource nominated it


@dataclass(frozen=True)
class Budget:
    """Context-assembly budget — a latency budget disguised as tokens.

    On CPU, ~4 s of prefill per 1000 tokens; tighter local budgets keep latency
    under control without changing the retrieval pipeline.
    """

    token_total: int
    max_per_article: int


@dataclass(frozen=True)
class ConfidenceSignals:
    """Computed here, consumed by the abstention gates and the agent loop.

    Nothing branches on these — they are recorded for calibration against
    eval datasets.
    """

    top_score: float | None  # max passage score, or None if no scorer ran
    score_dropoff: float | None  # ratio of score at rank 1 vs rank N (or None if <2 passages)
    density: float  # fraction of passages from the most-represented article (0..1)
    agreement: float  # Jaccard overlap between FTS-derived and suggest-derived candidate sets


@dataclass(frozen=True)
class RetrievalResult:
    """The pipeline's output — what reaches the LLM and the UI.

    ``passages`` are the evidence the answer prompt receives; ``cards`` drive the
    source-card panel in the UI; ``trace`` is the versioned, always-on audit log;
    ``confidence`` feeds the abstention gates.
    """

    passages: tuple[ScoredPassage, ...]
    cards: tuple[SourceCard, ...]
    trace: Trace
    confidence: ConfidenceSignals


class QueryRewriter(Protocol):
    """Resolve a follow-up question against conversation history into a
    standalone search query.

    Defined HERE (``retrieval/contracts.py``) rather than in ``inference/`` so
    the conversational-rewrite preparer can depend on it without ``retrieval/``
    importing ``inference/`` (retrieval must not know an LLM exists). The
    composition root constructs the concrete implementation (which wraps the
    gateway) and injects it via ``Deps.rewriter``.

    ``history`` is ``(role, content)`` string pairs — the same shape
    :attr:`PreparedQuery.history` carries — so no ``inference`` type crosses the
    boundary. The rewriter returns the rewritten query text (empty/failure ⇒ the
    caller falls back to the raw query; never make things worse).
    """

    async def rewrite(self, query: str, history: Sequence[tuple[str, str]]) -> str: ...


# ── Pipeline stage contracts ────────────────────────────────────────────────


class QueryPreparer(Protocol):
    """Stage: transform the query before candidate generation.

    Receives a ``PreparedQuery`` and returns a modified one. Ordered list — later
    preparers see the output of earlier ones. Examples: normalize, alias_expand,
    conversational_rewrite.
    """

    requires: ClassVar[frozenset[Capability]]

    async def prepare(self, q: PreparedQuery, tr: Trace) -> PreparedQuery: ...


class CandidateSource(Protocol):
    """Stage A: generate candidate articles from one signal.

    Called concurrently across all sources for a profile (bounded by a semaphore
    sized from settings). Returns ``list[Candidate]`` with ``source`` set to the
    component id — the Fuser groups them by ``FusionKey``.
    """

    requires: ClassVar[frozenset[Capability]]

    async def find(self, q: PreparedQuery, scope: Scope, tr: Trace) -> Sequence[Candidate]: ...


class Fuser(Protocol):
    """Stage A merge: combine candidates from multiple sources into one list.

    Receives groups keyed by ``FusionKey(zim_id, source)`` (making the
    archive dimension explicit). The default profile fuses within each archive
    then unions; ``across_archives: rrf`` fuses globally.
    """

    requires: ClassVar[frozenset[Capability]]

    def fuse(
        self, groups: Mapping[FusionKey, Sequence[Candidate]], tr: Trace
    ) -> Sequence[Candidate]: ...


class PassageBuilder(Protocol):
    """Stage B: split candidate articles into passages (chunks).

    Reads the full text of each candidate from the ZIM, extracts, and splits into
    ~400-token sentence-aligned passages with breadcrumb prefixes.
    """

    requires: ClassVar[frozenset[Capability]]

    async def build(
        self, cands: Sequence[Candidate], q: PreparedQuery, tr: Trace
    ) -> Sequence[Passage]: ...


class PassageScorer(Protocol):
    """Stage B scoring: assign or refine scores on passages (chainable).

    The chain pattern: scores from ``N+1`` replace scores from ``N``. The
    scorers are entries in the profile's ``scorers`` list, so the chain
    reorders or extends with zero pipeline code changes.
    """

    requires: ClassVar[frozenset[Capability]]

    async def score(
        self, passages: Sequence[ScoredPassage], q: PreparedQuery, tr: Trace
    ) -> Sequence[ScoredPassage]: ...


class ContextAssembler(Protocol):
    """Stage B output: select top passages, deduplicate, build the final result.

    The assembler decides *which* passages and how many — an independent axis
    from how they are scored. Token budgets, per-article caps, dedup policy, and
    lead-boosting all live here so they can be tuned without touching scoring.

    ``q`` is the prepared query: query-term-centred
    snippet generation and relevance-aware strategies (``lead_boost``,
    ``diverse``) need the query text, which is not reconstructable from
    ``Sequence[ScoredPassage]`` alone.
    """

    requires: ClassVar[frozenset[Capability]]

    def assemble(
        self, scored: Sequence[ScoredPassage], budget: Budget, q: PreparedQuery, tr: Trace
    ) -> RetrievalResult: ...


__all__ = [
    "Budget",
    "Candidate",
    "CandidateSource",
    "ConfidenceSignals",
    "ContextAssembler",
    "Fuser",
    "FusionKey",
    "Passage",
    "PassageBuilder",
    "PassageScorer",
    "PreparedQuery",
    "QueryPreparer",
    "QueryRewriter",
    "RetrievalResult",
    "Scope",
    "ScoredPassage",
    "SourceCard",
]
