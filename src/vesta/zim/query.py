"""Query preprocessing — a first-class component, not a utility.

Why this exists: ``libzim``'s ``src/search.cpp`` calls
``parse_query(q, FLAG_CJK_NGRAM)`` which *replaces* Xapian's ``FLAG_DEFAULT``,
and sets ``default_op = AND``. Consequences:

* phrase (``"..."``), boolean (``AND``/``OR``/``NOT``), ``+``/``-`` and wildcard
  (``*``) syntax are all **silently inert** — parsed as ordinary terms;
* **every term must match**, so a natural-language question like
  ``"how do i mount a usb drive"`` returns **0 results**.

The fix is a fallback ladder tried in order until a rung is non-empty:
**all-terms → stopword-stripped → OR-of-terms → title search**. Each rung
records itself in the trace. Without this the product's primary input — a
question — returns nothing.

This module owns the *implementation*; retrieval owns the candidate-source
component registration. It is self-contained, settings-driven, and importable
without any retrieval-pipeline code.

Multi-archive fan-out is retrieval policy: the preparer drives the ladder
against **one** archive's raw primitives (``Archive.search`` /
``Archive.suggest``) and returns that archive's unmerged paths.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from vesta.zim.types import EntryPath


class StageSink(Protocol):
    """The slice of the trace a single ladder rung writes to.

    This is zim's *consumer view* of the trace surface defined in
    ``vesta.retrieval.trace``: it deliberately mirrors ``StageCtx`` so the real
    Trace satisfies it structurally, without ``zim`` importing ``retrieval``
    (which would point the dependency arrow backwards — retrieval depends on
    zim, not the other way — and violate boundary rules).
    """

    def add_inputs(self, values: Mapping[str, Any]) -> None: ...

    def add_outputs(self, values: Mapping[str, Any]) -> None: ...


@runtime_checkable
class Tracer(Protocol):
    """Structural trace surface the preparer records each rung into.

    ``vesta.retrieval.trace.Trace`` satisfies this by construction; callers
    pass the real Trace here. Keeping the dependency as a Protocol (not an
    import) is what keeps ``zim`` free of retrieval-policy code.
    """

    def stage(
        self, name: str, component: str, params: Mapping[str, Any] | None = None
    ) -> AbstractContextManager[StageSink]: ...


class _NullStage:
    """No-op stage sink used when no tracer is supplied (default)."""

    def add_inputs(self, values: Mapping[str, Any]) -> None:
        pass

    def add_outputs(self, values: Mapping[str, Any]) -> None:
        pass


class _NullTracer:
    """No-op tracer: records nothing. The default when ``trace`` is ``None``."""

    def stage(
        self, name: str, component: str, params: Mapping[str, Any] | None = None
    ) -> AbstractContextManager[StageSink]:
        return nullcontext(_NullStage())


_NULL_TRACER = _NullTracer()

#: A per-archive fulltext primitive: AND the given terms, return up to ``limit``
#: paths. (libzim's default operator is AND.) Returns ``[]`` if the
#: archive has no fulltext index or the terms match nothing.
SearchFn = Callable[[tuple[str, ...], int], Awaitable[list[EntryPath]]]

#: A per-archive title/suggestion primitive (the universal ``X/title/xapian``
#: index — present even in archives with no fulltext index).
SuggestFn = Callable[[str, int], Awaitable[list[EntryPath]]]


@dataclass(frozen=True)
class QueryRung:
    """One step of the preprocessing ladder.

    ``term_sets`` holds the AND-queries whose path-sets are unioned for this
    rung (one set for the all-terms/stopword-stripped rungs; one per term for
    the OR-of-terms rung; empty for the title rung, which uses ``prefix``).
    """

    name: str  # "all_terms" | "stopword_stripped" | "or_of_terms" | "title"
    signal: str  # "fulltext" | "title"
    term_sets: tuple[tuple[str, ...], ...]
    prefix: str | None  # suggest prefix, for the title rung


# Interrogatives + common English stopwords. These turn a 0-hit NL question
# into correct hits ("how do i mount a usb drive" → "mount usb drive" →
# 6 correct results). Kept short: a longer list risks stripping meaningful
# terms from short keyword queries. Public: shared with the retrieval-level
# normalize preparer and xapian_fts's ladder fallback so there is exactly
# one list, not three copies drifting apart.
DEFAULT_STOPWORDS = (
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "from",
    "into",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "doing",
    "have",
    "has",
    "had",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "this",
    "that",
    "these",
    "those",
    "my",
    "your",
    "me",
    "him",
    "her",
    "us",
    "them",
    "how",
    "what",
    "why",
    "who",
    "whom",
    "when",
    "where",
    "which",
    "whose",
    "explain",
    "tell",
    "about",
    "there",
    "here",
)

# Words that are inert under libzim's parser — stripped pre-query so a user's
# quotes/booleans don't become literal AND-ed terms that force a 0-result.
_INERT_TOKENS = frozenset({"and", "or", "not", "near", "xor", "a", "an", "the", "+", "-", "*", '"'})

# Splits on anything that is not a unicode letter/digit/underscore. Keeps CJK
# runs intact (libzim does its own CJK n-gramming).
_TOKEN_RE = re.compile(r"[^\W]+", re.UNICODE)


def normalize_terms(raw: str) -> list[str]:
    """Lowercase, strip punctuation/quotes, split into unicode word tokens.

    Quotes and boolean operators are inert under libzim so they are dropped
    here rather than sent as literal terms.
    """
    tokens: list[str] = []
    for tok in _TOKEN_RE.findall(raw.lower()):
        if tok in _INERT_TOKENS:
            continue
        tokens.append(tok)
    return tokens


class QueryPreparer:
    """Settings-driven preprocessing ladder in front of every libzim query.

    The preparer is a plain, testable class with no dependency on the
    retrieval pipeline. ``execute`` drives the ladder against one archive's
    primitives and records every rung in the trace.
    """

    def __init__(
        self,
        *,
        stopwords: frozenset[str] | None = None,
        stopword_stripping: bool = True,
        ladder_enabled: bool = True,
    ) -> None:
        self._stopwords = stopwords if stopwords is not None else frozenset(DEFAULT_STOPWORDS)
        self._stopword_stripping = stopword_stripping
        self._ladder_enabled = ladder_enabled

    @classmethod
    def from_settings(
        cls,
        *,
        stopword_stripping: bool,
        stopword_list: str,
        ladder_enabled: bool,
    ) -> QueryPreparer:
        """Build from the raw ``query.*`` setting values (comma-joined list)."""
        words = frozenset(w.strip().lower() for w in stopword_list.split(",") if w.strip())
        return cls(
            stopwords=words,
            stopword_stripping=stopword_stripping,
            ladder_enabled=ladder_enabled,
        )

    def _strip_stopwords(self, terms: list[str]) -> list[str]:
        if not self._stopword_stripping:
            return terms
        return [t for t in terms if t not in self._stopwords]

    def ladder(self, raw: str) -> list[QueryRung]:
        """The ordered rungs for ``raw`` (pure — no I/O, no archive).

        The rungs are tried in order; the first non-empty one wins. Without the
        ladder a single all-terms AND would dead-end at zero results for any
        question-shaped query.
        """
        terms = normalize_terms(raw)
        rungs: list[QueryRung] = []
        if terms:
            # Rung 1: every normalized term, AND-ed (libzim's default op).
            rungs.append(
                QueryRung(
                    name="all_terms",
                    signal="fulltext",
                    term_sets=(tuple(terms),),
                    prefix=None,
                )
            )
            stripped = self._strip_stopwords(terms)
            # Rung 2: stopword-stripped terms (only if it actually removed any).
            if stripped and stripped != terms:
                rungs.append(
                    QueryRung(
                        name="stopword_stripped",
                        signal="fulltext",
                        term_sets=(tuple(stripped),),
                        prefix=None,
                    )
                )
            # Rung 3: OR-of-terms — one single-term query per term, unioned. We
            # cannot ask libzim for OR (the operator is inert); issuing one
            # query per term and unioning is the equivalent that works.
            if stripped:
                rungs.append(
                    QueryRung(
                        name="or_of_terms",
                        signal="fulltext",
                        term_sets=tuple((t,) for t in stripped),
                        prefix=None,
                    )
                )
                # Rung 4: title/suggestion index — present in every archive
                # tested, the universal fallback.
                rungs.append(
                    QueryRung(
                        name="title",
                        signal="title",
                        term_sets=(),
                        prefix=" ".join(stripped),
                    )
                )
        if not rungs:
            # Degenerate input (no tokens): fall straight to a title probe on
            # the raw string so the caller still gets *something* traceable.
            rungs.append(QueryRung(name="title", signal="title", term_sets=(), prefix=raw.strip()))
        return rungs

    async def execute(
        self,
        raw: str,
        search: SearchFn,
        suggest: SuggestFn,
        *,
        limit: int,
        trace: Tracer | None = None,
    ) -> list[EntryPath]:
        """Drive the ladder against one archive's primitives.

        Tries each rung in order; the first non-empty rung's unioned paths are
        returned (merging rungs is retrieval policy). Each attempted rung is
        recorded in ``trace`` with its query form, signal and hit count, so the
        dev console / eval harness can show which rung produced a result.
        ``trace`` defaults to a no-op tracer; pass the real retrieval Trace to
        capture the ladder into the shipped trace.
        """
        rungs = self.ladder(raw)
        # When the ladder is disabled we still try the raw all-terms form once,
        # preserving the preprocessing boundary but not the fallback.
        if not self._ladder_enabled:
            rungs = rungs[:1]
        tracer = trace if trace is not None else _NULL_TRACER
        with tracer.stage(
            "query_preprocessing",
            "zim.query_preparer",
            {
                "raw": raw,
                "ladder_enabled": self._ladder_enabled,
                "stopword_stripping": self._stopword_stripping,
            },
        ) as stage:
            stage.add_inputs({"rungs": [r.name for r in rungs]})
            chosen: list[EntryPath] = []
            chosen_rung: str | None = None
            for rung in rungs:
                hits = await self._run(rung, search, suggest, limit)
                stage.add_outputs(
                    {f"rung.{rung.name}.hits": len(hits), f"rung.{rung.name}.signal": rung.signal}
                )
                if hits:
                    chosen = hits[:limit]
                    chosen_rung = rung.name
                    break
            stage.add_outputs({"chosen_rung": chosen_rung, "result_count": len(chosen)})
        return chosen

    async def _run(
        self,
        rung: QueryRung,
        search: SearchFn,
        suggest: SuggestFn,
        limit: int,
    ) -> list[EntryPath]:
        if rung.signal == "title" and rung.prefix is not None:
            return await suggest(rung.prefix, limit)
        seen: list[EntryPath] = []
        seen_set: set[EntryPath] = set()
        for term_set in rung.term_sets:
            if not term_set:
                continue
            for path in await search(term_set, limit):
                if path not in seen_set:
                    seen_set.add(path)
                    seen.append(path)
            if len(seen) >= limit:
                break
        return seen


__all__ = [
    "DEFAULT_STOPWORDS",
    "QueryPreparer",
    "QueryRung",
    "SearchFn",
    "StageSink",
    "SuggestFn",
    "Tracer",
    "normalize_terms",
]
