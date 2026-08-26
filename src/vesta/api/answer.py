"""SSE answer protocol — ``GET /api/answer``.

Frozen protocol property: source cards go out immediately (~0.7 s),
then truthful status during the prefill gap, then tokens. The server is
responsible for there always being a next truthful thing to show.

Event ordering (docs/sse-protocol.md):
    event: sources       data: {cards: [...]}            # first
    event: status        data: {phase, detail}
    event: token         data: {text}
    event: answer_reset  data: {reason}                  # zero or more, mid-stream
    event: citations     data: {spans: [...], answer_text}
    event: trace         data: {...}                     # last before done
    event: error         data: {code, message, recoverable}
    event: done          data: {}                        # terminal

Killing ``llama-server`` mid-answer produces a clean ``error`` event + auto-
restart. ``sources_only`` works with no LLM and is auto-selected when
``Capability.LLM`` is absent.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import string
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from vesta import config as app_config
from vesta.answer import ANSWER_STRATEGY, resolve_strategy_name, select_strategy
from vesta.answer.abstention import ABSTENTION_NO_MATCH
from vesta.answer.contracts import (
    AnswerContext,
    AnswerDeps,
    AnswerResetEvent,
    CitationsEvent,
    DoneEvent,
    ErrorEvent,
    SourcesEvent,
    StatusEvent,
    TokenEvent,
    TraceEvent,
)
from vesta.api.state import AppState, app_state
from vesta.config.capabilities import compute_capabilities
from vesta.retrieval import RETRIEVAL_MAX_ARCHIVES_CONCURRENT, RETRIEVAL_PROFILES
from vesta.retrieval.contracts import RetrievalResult, ScoredPassage, SourceCard
from vesta.retrieval.contracts import Scope as RetScope
from vesta.retrieval.pipeline import Deps, NoCandidatesError, run_pipeline
from vesta.retrieval.profiles import RetrievalProfile, resolve_profile
from vesta.vectors import get_store as get_vector_store

if TYPE_CHECKING:
    from vesta.zim.registry import ArchiveRegistry

router = APIRouter(prefix="/api", tags=["answer"])

_log = logging.getLogger(__name__)


@router.get("/answer")
async def answer(
    request: Request,
    q: str = Query(..., description="Question or search term"),
    scope: str | None = Query(None, description="Comma-separated zim_ids"),
    profile: str | None = Query(None, description="Retrieval profile override"),
    strategy: str | None = Query(None, description="Answer strategy override"),
) -> StreamingResponse:
    """Stream a grounded, cited answer as SSE events."""
    state: AppState = app_state(request)

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for event in iter_answer_events(state, q, scope, profile, strategy):
                yield _serialize_event(event)
        except Exception as exc:
            yield _serialize_event(ErrorEvent(code="fatal", message=str(exc), recoverable=False))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


async def iter_answer_events(
    state: AppState,
    query: str,
    scope: str | None,
    profile_override: str | None,
    strategy_override: str | None,
    *,
    history: tuple[tuple[str, str], ...] = (),
) -> AsyncIterator[object]:
    """Run retrieval → answer strategy, yielding AnswerEvent objects.

    Public alias of the in-process answer pipeline. The benchmark
    driver iterates this same event stream ``GET /api/answer`` serializes, so
    the harness exercises the real pipeline minus the frozen SSE wire layer
    (faithful to ``/api/answer`` — see ``docs/sse-protocol.md``; the serializer
    is contract-tested separately).

    ``history`` (09) is the prior conversation as ``(role, content)`` pairs,
    threaded into both the retrieval pipeline (for conversational rewrite) and
    the answer context. Defaults empty (single-turn ``/api/answer``).
    """
    # ── Step 1: retrieval ────────────────────────────────────────────────
    capabilities = compute_capabilities()
    try:
        sn = app_config.snapshot()
    except RuntimeError:
        sn = None

    retrieval_profile = _resolve_profile(profile_override)
    if retrieval_profile is None:
        yield ErrorEvent(
            code="no_profile", message="no retrieval profile could be resolved", recoverable=False
        )
        yield DoneEvent()
        return
    ret_scope = _parse_scope(scope, state.registry)

    # 09: wire the gateway-backed conversational rewriter into the retrieval
    # Deps so the conversational_rewrite preparer can resolve turn-≥2 follow-ups.
    # Built here (the composition root) because it bridges inference (gateway)
    # with retrieval (the QueryRewriter Protocol) — neither package may import
    # the other. NullGateway / no-LLM → no rewriter (the
    # preparer degrades to a no-op).
    rewriter = _build_rewriter(state, sn)

    deps = Deps(
        archives=state.registry,
        settings=sn,
        capabilities=capabilities,
        semaphore=asyncio.Semaphore(_concurrency_bound(sn)),
        encoders=state.encoders,
        # Dense source DI.
        vectors=get_vector_store(),
        # Conversational rewrite DI.
        rewriter=rewriter,
    )

    try:
        result = await run_pipeline(
            profile=retrieval_profile,
            query=query,
            scope=ret_scope,
            deps=deps,
            history=history,
        )
    except NoCandidatesError as exc:
        # Hard requirement, surfaced the same way /api/search handles it:
        # "nothing matched" is a valid, explainable outcome, not a stream error.
        # Emit empty sources + the harness abstention string; the
        # exception carries the trace built up to the point of failure.
        yield SourcesEvent(cards=())
        yield StatusEvent(phase="abstaining", detail="no candidates")
        yield TokenEvent(text=ABSTENTION_NO_MATCH)
        yield TraceEvent(trace=exc.trace.to_dict())
        yield DoneEvent()
        return
    except Exception as exc:
        yield ErrorEvent(code="retrieval_failed", message=str(exc), recoverable=False)
        yield DoneEvent()
        return

    # The trace is a first-class output shared across the whole request
    # (docs/sse-protocol.md): answer stages are appended to
    # the retrieval trace, not recorded in a separate one.
    tr = result.trace

    # ── Step 2: select + run answer strategy ──────────────────────────────
    configured_strategy = strategy_override or str(app_config.get(ANSWER_STRATEGY))
    resolved_name = resolve_strategy_name(configured_strategy, capabilities, settings=sn)
    strategy_cls = select_strategy(resolved_name)

    # Check capabilities for the strategy's requires.
    from vesta.config.capabilities import Capability

    requires: frozenset[Capability] = getattr(strategy_cls, "requires", frozenset())
    unmet = requires - capabilities
    if unmet:
        # Fall back to sources_only.
        strategy_cls = select_strategy("sources_only")

    archive_labels_map = await _archive_labels(state)

    # 09: build the tool runtime for the agentic loop. The callables adapt the
    # archive registry + retrieval pipeline to the tool protocol's primitive
    # signatures — ``answer/`` never imports ``zim/``.
    tools = _build_tool_runtime(
        state, sn, ret_scope, retrieval_profile, query, archive_labels=archive_labels_map
    )

    answer_deps = AnswerDeps(
        gateway=state.gateway,
        settings=sn,
        capabilities=capabilities,
        tools=tools,
        archive_labels=archive_labels_map,
    )

    is_search_term = not _looks_like_question(query)
    # Convert (role, content) history pairs → ChatMessage for the answer context.
    from vesta.inference.gateway import ChatMessage

    history_messages = tuple(ChatMessage(role=r, content=c) for r, c in history) if history else ()
    ctx = AnswerContext(
        query=query,
        retrieval=result,
        is_search_term=is_search_term,
        history=history_messages,
    )

    # Construct the strategy. Pass deps if the constructor accepts it.
    try:
        instance = strategy_cls(deps=answer_deps)
    except TypeError:
        instance = strategy_cls()

    # Yield events from the strategy.
    async for event in instance.answer(ctx, answer_deps, tr):
        yield event


def _serialize_event(event: object) -> str:
    """Serialize an AnswerEvent to an SSE wire string."""
    if isinstance(event, SourcesEvent):
        cards = [_card_to_dict(c) for c in event.cards]
        return _sse("sources", {"cards": cards, "merge": event.merge})
    if isinstance(event, StatusEvent):
        return _sse("status", {"phase": event.phase, "detail": event.detail})
    if isinstance(event, TokenEvent):
        return _sse("token", {"text": event.text})
    if isinstance(event, AnswerResetEvent):
        return _sse("answer_reset", {"reason": event.reason})
    if isinstance(event, CitationsEvent):
        spans = [
            {
                "answer_span": [s.answer_start, s.answer_end],
                "card_id": s.source_index,
                "passage_span": (
                    [s.passage_start, s.passage_end]
                    if s.passage_start is not None and s.passage_end is not None
                    else None
                ),
                "score": s.score,
            }
            for s in event.spans
        ]
        return _sse("citations", {"spans": spans, "answer_text": event.answer_text})
    if isinstance(event, TraceEvent):
        return _sse("trace", event.trace)
    if isinstance(event, ErrorEvent):
        return _sse(
            "error",
            {"code": event.code, "message": event.message, "recoverable": event.recoverable},
        )
    if isinstance(event, DoneEvent):
        return _sse("done", {})
    return _sse(
        "error",
        {
            "code": "unknown_event",
            "message": f"unknown event type {type(event)}",
            "recoverable": True,
        },
    )


def _sse(event: str, data: dict[str, Any]) -> str:
    """Format one SSE event (event name + JSON data line)."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _card_to_dict(c: Any) -> dict[str, Any]:
    """Map a retrieval SourceCard to the public card DTO.

    The single serializer for that shape — the SSE ``sources`` payload here and
    the persisted ``messages.sources_json`` in ``api/chat.py`` must never drift
    apart.
    """
    return {
        "zim_id": c.zim_id,
        "path": c.path,
        "title": c.title,
        "snippet": c.snippet,
        "breadcrumb": c.breadcrumb,
        "score": c.score,
        "source": c.source,
    }


def _looks_like_question(text: str) -> bool:
    """True when text looks like a question rather than a keyword query."""
    if text.endswith("?"):
        return True
    first_word = text.split(maxsplit=1)[0].lower() if text else ""
    return first_word in {"what", "who", "how", "when", "where", "why", "which"}


async def _archive_labels(state: AppState) -> dict[int, str]:
    """``zim_id`` -> human-readable archive name, for prompt-visible source
    labels (replacing the opaque ``archive-{zim_id}`` form).

    ``answer/`` cannot import ``zim/`` (dependency cap), so the API layer reads
    the ``zims`` table directly and injects the mapping via ``AnswerDeps``.
    Prefers the registry's ``title``; falls back to ``corpus_label`` when the
    title is empty. Any failure degrades to an empty dict (every label falls
    back to ``archive-{zim_id}``), never a 500.
    """
    try:
        async with (
            state.db.read() as conn,
            conn.execute("SELECT id, title, corpus_label FROM zims WHERE enabled = 1") as cur,
        ):
            rows = await cur.fetchall()
    except Exception:
        return {}
    labels: dict[int, str] = {}
    for zid, title, corpus_label in rows:
        label = (title or "").strip() or (corpus_label or "").strip()
        if label:
            labels[int(zid)] = label
    return labels


def _concurrency_bound(sn: Any) -> int:
    """The retrieval fan-out semaphore size from settings (mirrors ``api/search``).

    Without this the pipeline silently falls back to a hardcoded bound of 4
    instead of the configured ``retrieval.max_archives_concurrent``.
    """
    if sn is not None:
        try:
            return max(1, int(sn.get(RETRIEVAL_MAX_ARCHIVES_CONCURRENT)))
        except Exception:
            pass
    return int(RETRIEVAL_MAX_ARCHIVES_CONCURRENT.default)


def _resolve_stopwords(sn: Any) -> frozenset[str]:
    """The query pipeline's own stopword list, for the ``read_article`` tool's
    focused-view budgeting (iter 17). Resolved here in the composition root
    because ``AnswerDeps.settings`` isn't in scope from inside the tool
    closure."""
    from vesta.config import QUERY_STOPWORDS_LIST

    raw = str(QUERY_STOPWORDS_LIST.default)
    if sn is not None:
        with contextlib.suppress(Exception):
            raw = str(sn.get(QUERY_STOPWORDS_LIST))
    return frozenset(w.strip().lower() for w in raw.split(",") if w.strip())


def _parse_scope(scope: str | None, registry: ArchiveRegistry | None = None) -> RetScope:
    """Parse the comma-separated ``--scope``/``?scope=`` parameter into a Scope.

    Each token is either a bare integer ``zim_id`` (``"3"``) or an archive name
    or filename (``"wikipedia_en_top"``, ``"wikipedia_en_top_nopic_2026-06.zim"``
    — the form ``benchmarks/README.md`` documents), resolved against
    ``registry`` via :meth:`ArchiveRegistry.resolve_scope_token`. ``registry``
    is optional so existing test fakes / no-registry call sites keep working;
    with it omitted, only integer ids resolve.

    **An unresolvable token degrades the whole scope to "matches nothing"
    (``zim_ids=frozenset()``), never to "matches everything" (``None``).** The
    previous implementation swallowed a bad token via a bare
    ``suppress(ValueError)`` around the whole parse, which silently produced an
    *unscoped* ``RetScope()`` — every enabled archive got searched, exactly the
    opposite of what a caller asking for a specific archive wants, and with no
    signal that anything had gone wrong. A scope matching no archive instead
    trips the retrieval pipeline's existing ``NoCandidatesError`` "hard
    requirement" path — degrade, don't silently produce a
    different result — a real, visible failure — and a warning is logged
    immediately so the cause doesn't wait on that.
    """
    if not scope:
        return RetScope()
    tokens = [s.strip() for s in scope.split(",") if s.strip()]
    if not tokens:
        return RetScope()
    zim_ids: set[int] = set()
    for token in tokens:
        try:
            zim_ids.add(int(token))
            continue
        except ValueError:
            pass
        resolved = registry.resolve_scope_token(token) if registry is not None else None
        if resolved is None:
            _log.warning(
                "answer.scope_token_unresolved token=%r scope=%r — scoping to NO archives "
                "(not all of them) so the failure is visible",
                token,
                scope,
            )
            return RetScope(zim_ids=frozenset())
        zim_ids.add(resolved)
    return RetScope(zim_ids=frozenset(zim_ids))


#: Below this ``confidence.top_score``, a search's results are considered too
#: weak to be useful — almost certainly noise from xapian's OR-of-terms
#: fallback matching common words. Set just under the 0.25 abstention floor
#: so genuinely-found evidence (even at depth-0-typical scores) is untouched.
_SEARCH_SHORTEN_THRESHOLD = 0.20

#: Above this score, the search result is confident enough that surfacing
#: individual-term candidates is unnecessary (and would add context noise).
#: Below this, the result MIGHT be a wrong-but-decent match — the prefix
#: shortening often finds a related article scoring 0.20-0.30 that is NOT the
#: target. Term candidates surface the actual target's title for the model to
#: read_article. Set well above the shortening threshold to cover this gap.
_TERM_SURFACE_THRESHOLD = 0.35

#: Minimum word length to try as a standalone term search. Shorter words are
#: almost always common English words whose single-term search floods results
#: with irrelevant popular articles. 4 chars filters "the/of/how/what" while
#: keeping distinctive entity terms ("habenula", "Boyle", "chloride").
_TERM_MIN_WORD = 4

#: Cap on individual-term searches per tool call. Each is a local retrieval
#: pipeline call (~1-2s, no LLM). Only fires when prefix shortening already
_TERM_MAX_SEARCHES = 5

#: Cap on articles appended to a failed Round-0 by the conditional
#: reformulation. Union-append, never replace: the reformulated
#: runs' new cards join after the base result's, so recall can only gain.
#: 10 covers the measured deepest recovery (gold at re-search rank 7).
_REFORM_MAX_APPEND = 10


async def _maybe_shorten_search(
    result: RetrievalResult,
    *,
    query: str,
    profile: RetrievalProfile,
    scope: RetScope,
    deps: Deps,
) -> RetrievalResult:
    """Retry a long, low-confidence tool-driven search with shorter prefixes.

    Trace-confirmed (iter 6 traces): the model's continuation/recovery search
    queries are often 5-8 word fact-focused rephrases ("Robert Boyle copper
    chloride archaic name") whose distinctive terms drown under common-word
    matches in xapian's OR-of-terms fallback ladder — returning completely
    irrelevant popular articles (Maya_civilization, Human, Lady_Gaga). The
    entity name alone ("Robert Boyle") surfaces the target at rank 1 via
    ``title_suggest``. This is a general, harness-side fix for any long NL
    query over any corpus — never question-specific.

    Only triggers when (a) the query is 4+ words AND (b) the result's
    ``top_score`` is below :data:`_SEARCH_SHORTEN_THRESHOLD`. Retries at most
    twice (first 3 words, then first 2 words), keeping whichever result has
    the higher ``top_score``. Round 0 / golden-eval retrieval never pass
    through here — only the agentic loop's ``search`` tool callable
    (``_build_tool_runtime``'s ``_search`` with ``shorten=True``, the default).
    The Round-0 pre-seed uses ``search_exact`` (``shorten=False``) precisely
    to keep this invariant true: a raw multi-clause question's entity often
    sits mid-sentence, and shortening to a leading prefix degrades to
    interrogative stopwords ("In what year...") that would replace a correct
    result with junk.
    """
    words = query.split()
    if len(words) < 4:
        return result
    best_score = result.confidence.top_score if result.confidence else 0.0
    if (best_score or 0.0) >= _SEARCH_SHORTEN_THRESHOLD:
        return result
    best = result
    for n in (3, 2):
        if len(words) <= n:
            continue
        short_q = " ".join(words[:n])
        try:
            short_result = await run_pipeline(
                profile=profile, query=short_q, scope=scope, deps=deps
            )
        except Exception:
            break
        short_score = short_result.confidence.top_score if short_result.confidence else 0.0
        if (short_score or 0.0) > (best_score or 0.0):
            best, best_score = short_result, short_score
        if (best_score or 0.0) >= _SEARCH_SHORTEN_THRESHOLD:
            break
    return best


async def _maybe_reformulate_round0(
    result: RetrievalResult,
    *,
    query: str,
    profile: RetrievalProfile,
    scope: RetScope,
    deps: Deps,
    reformulator: Any,
    sn: Any,
) -> tuple[RetrievalResult, tuple[SourceCard, ...]]:
    """Conditionally reformulate a *visibly failed* Round 0.

    Fires only when ``answer.reformulate.enabled`` is on, a reformulator is
    wired (gateway + model), and the Round-0 result's ``top_score`` is below
    ``answer.reformulate.trigger_score`` — a healthy Round 0 pays nothing
    (the gateway is never touched above the trigger). One
    ``chat_once`` names the article the fact would live in; each returned
    query re-runs the same pipeline (no parallel retrieval path).

    **Union-append, never replace** (measured 2026-08-16, artifact
    ``20260816-phase19-4-trigger-probe.json``): the plan's original
    keep-the-higher-top_score rule lost 3 correct-but-weakly-scored results to
    win 1 — cross-encoder scores are incomparable *across queries*, so a
    single-entity re-search routinely scores higher against the wrong article
    than the original scored against the right one ("Habenular commissure"
    0.13 displacing gold-at-rank-1 at 0.011). Instead the reformulated runs'
    NEW cards (and their passages) are appended after the base result's, in
    re-search order, deduplicated by ``(zim_id, path)``, capped at
    :data:`_REFORM_MAX_APPEND`. Base ordering, confidence, and trace are
    untouched — recall can only gain, the never-worse contract holds by
    construction, and the caller surfaces the appended articles to the model
    as ``read_article`` candidates (the iter-9/10 text shape). On gateway
    exception or empty/stagnant output the original result returns with no
    appends.
    """
    if reformulator is None or sn is None:
        return result, ()
    from vesta.answer import (
        ANSWER_REFORMULATE_ENABLED,
        ANSWER_REFORMULATE_MAX_QUERIES,
        ANSWER_REFORMULATE_TRIGGER_SCORE,
    )

    with contextlib.suppress(Exception):
        if not bool(sn.get(ANSWER_REFORMULATE_ENABLED)):
            return result, ()
    top_score = result.confidence.top_score if result.confidence else 0.0
    with contextlib.suppress(Exception):
        if (top_score or 0.0) >= float(sn.get(ANSWER_REFORMULATE_TRIGGER_SCORE)):
            return result, ()
    limit = 1
    with contextlib.suppress(Exception):
        limit = max(1, int(sn.get(ANSWER_REFORMULATE_MAX_QUERIES)))

    started = time.monotonic()
    try:
        queries = await reformulator.reformulate(query, limit=limit)
    except Exception as exc:
        _log.warning("answer.reformulate.failed — keeping original result (%s)", exc)
        return result, ()
    llm_ms = (time.monotonic() - started) * 1000.0
    if not queries:
        _log.info("answer.reformulate.stagnant — empty or duplicate output for %r", query)
        return result, ()

    seen = {(c.zim_id, c.path) for c in result.cards}
    appended_cards: list[SourceCard] = []
    appended_passages: list[ScoredPassage] = []
    for q in queries:
        try:
            r = await run_pipeline(profile=profile, query=q, scope=scope, deps=deps)
        except Exception:
            continue
        new_keys: set[tuple[int, Any]] = set()
        for c in r.cards:
            if len(appended_cards) >= _REFORM_MAX_APPEND:
                break
            key = (c.zim_id, c.path)
            if key in seen:
                continue
            seen.add(key)
            new_keys.add(key)
            appended_cards.append(c)
        if new_keys:
            appended_passages.extend(
                sp for sp in r.passages if (sp.passage.zim_id, sp.passage.path) in new_keys
            )
    if not appended_cards:
        _log.info(
            "answer.reformulate.stagnant — no new articles (queries=%r llm_ms=%.0f)",
            queries,
            llm_ms,
        )
        return result, ()
    merged = replace(
        result,
        cards=tuple(result.cards) + tuple(appended_cards),
        passages=tuple(result.passages) + tuple(appended_passages),
    )
    _log.info(
        "answer.reformulate.appended n_cards=%d/%d queries=%r llm_ms=%.0f",
        len(appended_cards),
        _REFORM_MAX_APPEND,
        queries,
        llm_ms,
    )
    return merged, tuple(appended_cards)


@dataclass(frozen=True)
class _TermCandidate:
    """One article found by searching an individual term from the query."""

    title: str
    zim_id: int
    path: str


def _distinctive_terms(query: str) -> list[str]:
    """Extract individual distinctive content words from a long NL query.

    Returns words >= :data:`_TERM_MIN_WORD` chars, excluding stopwords, in
    query order, deduplicated. These are tried as single-word searches when
    prefix shortening fails — a rare entity term (e.g. "habenula" at word 7)
    matches strongly as a single-term AND query where the full NL question
    AND-matched to 0 hits. General for any long NL query over any corpus.
    """
    from vesta.zim.query import DEFAULT_STOPWORDS

    stopwords = frozenset(DEFAULT_STOPWORDS)
    terms: list[str] = []
    seen: set[str] = set()
    for w in query.split():
        cleaned = w.strip(string.punctuation)
        low = cleaned.lower()
        if len(cleaned) >= _TERM_MIN_WORD and low not in stopwords and low not in seen:
            seen.add(low)
            terms.append(cleaned)
    return terms[:_TERM_MAX_SEARCHES]


async def _surface_term_candidates(
    query: str,
    existing_paths: set[str],
    *,
    profile: RetrievalProfile,
    scope: RetScope,
    deps: Deps,
) -> list[_TermCandidate]:
    """Search individual distinctive terms; return NEW article titles to surface.

    When prefix shortening fails (entity buried mid-query, or entity is a
    paraphrase), individual rare terms can still find the target article via
    xapian's single-term AND-match or ``title_suggest``. Instead of merging
    these as cards (which mixes incomparable cross-encoder scores and tanks
    recall@10 — learned iter 9), the titles are surfaced as TEXT for the model
    to ``read_article``. The model reads the question and judges which title
    is relevant — a task it's good at (the 50/50 full-context ceiling proves
    it), unlike ``top_score`` which can't distinguish a relevant rare-entity
    match from an irrelevant common-word match.

    Only returns articles whose ``path`` is NOT already in ``existing_paths``
    (no duplicates with what the model already has). Tool-driven searches
    only (continuation/recovery); Round 0 never passes through here.
    """
    terms = _distinctive_terms(query)
    candidates: list[_TermCandidate] = []
    seen: set[str] = set()

    for term in terms:
        try:
            result = await run_pipeline(profile=profile, query=term, scope=scope, deps=deps)
        except Exception:
            continue
        if not result.cards:
            continue
        top = result.cards[0]
        if top.path in existing_paths or top.path in seen:
            continue
        seen.add(top.path)
        candidates.append(_TermCandidate(title=top.title, zim_id=top.zim_id, path=top.path))

    return candidates


def _format_term_candidates(
    candidates: list[_TermCandidate],
    *,
    header: str = (
        "Other candidate articles found by searching individual terms\n"
        "(use read_article to open any that look relevant):"
    ),
) -> str:
    """Format surfaced-article candidates as a readable block for the model.

    ``header`` lets the Round-0 reformulation path label its
    appended articles as second-search finds without a second formatter.
    """
    if not candidates:
        return ""
    lines = ["", header]
    for c in candidates:
        lines.append(f'- "{c.title}" — read_article(zim_id={c.zim_id}, path="{c.path}")')
    return "\n".join(lines)


def _resolve_profile(profile_override: str | None) -> Any:
    """Resolve the retrieval profile (user-saved first, then built-in)."""
    from vesta.retrieval import RETRIEVAL_ACTIVE_PROFILE
    from vesta.retrieval.profiles import load_profile, load_user_profiles

    name = profile_override
    if not name:
        try:
            name = str(app_config.get(RETRIEVAL_ACTIVE_PROFILE))
        except Exception:
            name = "lexical"
    if not name:
        name = "lexical"

    try:
        users = load_user_profiles(str(app_config.get(RETRIEVAL_PROFILES)))
    except Exception:
        users = {}
    resolved = resolve_profile(name, users)
    if resolved is not None:
        return resolved
    return load_profile("lexical")


def _build_rewriter(state: AppState, sn: Any) -> Any:
    """Construct the gateway-backed conversational rewriter.

    Returns ``None`` when there is no usable gateway or no model configured —
    the conversational_rewrite preparer degrades to a no-op in that case.
    """
    if state.gateway is None:
        return None
    from vesta.inference import INFERENCE_LLM_MODEL
    from vesta.inference.gateway import NullGateway

    if isinstance(state.gateway, NullGateway):
        return None
    model = "unsloth/qwen3.5-4b"
    if sn is not None:
        with contextlib.suppress(Exception):
            model = str(sn.get(INFERENCE_LLM_MODEL))
    from vesta.answer.rewriter import GatewayQueryRewriter

    return GatewayQueryRewriter(state.gateway, model=model)


def _build_reformulator(state: AppState, sn: Any) -> Any:
    """Construct the gateway-backed Round-0 reformulator.

    Returns ``None`` when there is no usable gateway or model —
    ``_maybe_reformulate_round0`` then degrades to a no-op, so a box
    with no LLM keeps today's behaviour exactly. Mirrors
    ``_build_rewriter``'s posture, including the snapshot fallbacks for
    in-process callers that never ran the lifespan.
    """
    if state.gateway is None:
        return None
    from vesta.inference import INFERENCE_LLM_MODEL
    from vesta.inference.gateway import NullGateway

    if isinstance(state.gateway, NullGateway):
        return None
    model = "unsloth/qwen3.5-4b"
    if sn is not None:
        with contextlib.suppress(Exception):
            model = str(sn.get(INFERENCE_LLM_MODEL))
    variant, max_tokens = "exemplified", 64
    if sn is not None:
        from vesta.answer import (
            ANSWER_REFORMULATE_MAX_TOKENS,
            ANSWER_REFORMULATE_PROMPT_VARIANT,
        )

        with contextlib.suppress(Exception):
            variant = str(sn.get(ANSWER_REFORMULATE_PROMPT_VARIANT))
        with contextlib.suppress(Exception):
            max_tokens = int(sn.get(ANSWER_REFORMULATE_MAX_TOKENS))
    from vesta.answer.reformulate import (
        MINIMAL_REFORMULATE_SYSTEM_PROMPT,
        REFORMULATE_SYSTEM_PROMPT,
        GatewayReformulator,
    )

    prompt = (
        MINIMAL_REFORMULATE_SYSTEM_PROMPT if variant == "minimal" else REFORMULATE_SYSTEM_PROMPT
    )
    return GatewayReformulator(state.gateway, model=model, max_tokens=max_tokens, prompt=prompt)


def _build_tool_runtime(
    state: AppState,
    sn: Any,
    scope: RetScope,
    profile: Any,
    question: str,
    *,
    archive_labels: dict[int, str],
) -> Any:
    """Construct the :class:`~vesta.answer.tools.ToolRuntime` for the pydantic-ai agent.

    The callables adapt the archive registry + retrieval pipeline to the tool
    protocol's primitive signatures. ``answer/`` never imports ``zim/`` —
    this composition-root function does, and hands the callables in as plain
    ``async (int, str) -> str`` lambdas. Returns ``None`` when the archive
    registry is unavailable (degrade-don't-fail).

    ``search_exact`` is ``search`` with the shortening/term-surfacing recovery
    ladder disabled (see ``_search``'s ``shorten`` parameter) — for callers
    that need a raw, un-recovered retrieval call on a full natural-language
    question (the agent's Round-0 pre-seed), not a tool-driven query.

    ``question`` is the live request's query, closed over by ``_read_article``
    below (iter 17) so the model's own ``read_article`` tool call can apply the
    same focused-view treatment as the other read sites. Since AUDIT_0824 N11,
    :class:`~vesta.answer.tools.ReadArticleFn` also takes a keyword-only
    ``must_include`` snippet — agent_chat passes each card's retrieval snippet
    so the stage-1 focused window guarantees it survives elision.
    """
    if state.registry is None:
        return None
    from vesta.answer.tools import SearchToolResult, ToolRuntime, format_search_result

    registry = state.registry
    capabilities = compute_capabilities()
    reformulator = _build_reformulator(state, sn)

    async def _search(
        query: str, scope_str: str, *, shorten: bool = True, round0: bool = False
    ) -> SearchToolResult:
        if not query.strip():
            return SearchToolResult(text="Search query was empty.")
        tool_scope = _parse_scope(scope_str, registry) if scope_str else scope
        tool_deps = Deps(
            archives=registry,
            settings=sn,
            capabilities=capabilities,
            semaphore=asyncio.Semaphore(_concurrency_bound(sn)),
            encoders=state.encoders,
            vectors=get_vector_store(),
        )
        try:
            result = await run_pipeline(
                profile=profile,
                query=query,
                scope=tool_scope,
                deps=tool_deps,
            )

            # ``shorten=False`` (the ``search_exact`` callable, used for the
            # agentic loop's Round-0 pre-seed — see ``ToolRuntime.search_exact``)
            # bypasses the two tool-driven recovery mechanisms below: they were
            # designed and measured for tool-driven continuation/recovery
            # queries (short, fact-shaped rephrases the model writes itself),
            # never for a raw multi-clause natural-language question. A
            # question's entity often sits mid-sentence, so prefix shortening
            # degrades to leading interrogative stopwords ("In what year...")
            # that score HIGH and *replace* the correct result with junk.
            #
            # Round 0 instead gets its OWN conditional rung
            # (``round0=True``): only after a *visibly failed* Round 0
            # (top_score below ``answer.reformulate.trigger_score``) does it
            # make one LLM call naming the article the fact would live in,
            # re-search, and append the new articles it found to the Round-0
            # result. A healthy Round 0 touches no LLM (S4).
            if shorten:
                # Harness-side query shortening (iter 7): a long tool-driven
                # search (4+ words) returning low-confidence results is almost
                # always matching generic common words via xapian's OR-of-terms
                # fallback ladder (trace-confirmed iter 6: "Robert Boyle copper
                # chloride archaic name" → Maya_civilization, Human, etc.).
                # Retrying with progressively shorter prefixes surfaces the
                # entity name the title_suggest stage finds at rank 1 — a
                # general fix for any long NL query over any corpus, not
                # question-specific. Only affects tool-driven searches
                # (continuation/recovery), never Round 0 or the golden eval.
                result = await _maybe_shorten_search(
                    result, query=query, profile=profile, scope=tool_scope, deps=tool_deps
                )

            appended_cards: tuple[SourceCard, ...] = ()
            if round0:
                result, appended_cards = await _maybe_reformulate_round0(
                    result,
                    query=query,
                    profile=profile,
                    scope=tool_scope,
                    deps=tool_deps,
                    reformulator=reformulator,
                    sn=sn,
                )

            text = format_search_result(result, archive_labels=archive_labels)
            candidates_text = ""
            if appended_cards:
                # Model visibility for the appended articles: the pre-seed
                # text renders only the top passages, so union-appended cards
                # would be invisible to the model without this block. Same
                # shape as the tool-path term candidates (iter 10) — titles +
                # read_article calls, never score-merged passages.
                block = _format_term_candidates(
                    [
                        _TermCandidate(title=c.title, zim_id=c.zim_id, path=str(c.path))
                        for c in appended_cards
                    ],
                    header=(
                        "Articles found by a second search after the first one "
                        "came back weak (use read_article to open any that "
                        "look relevant):"
                    ),
                )
                text += block
                candidates_text += block

            if shorten:
                # Surface individual-term candidates when the result is still
                # weak (iter 10): prefix shortening handles entities at the
                # front, but entities buried mid-query (e.g. "habenula" at word
                # 7) or that are paraphrases are missed by every prefix.
                # Individual distinctive terms CAN find the target via xapian's
                # single-term AND-match, but merging their cards mixes
                # incomparable scores (learned iter 9). Instead, surface their
                # TITLES as text the model can read_article — the model judges
                # relevance from the title + question, which it's good at
                # (50/50 full-context ceiling), unlike top_score. Only fires
                # for tool-driven searches (continuation/recovery) on long
                # queries whose shortened result is still below the confidence
                # threshold.
                result_score = result.confidence.top_score if result.confidence else 0.0
                if len(query.split()) >= 4 and (result_score or 0.0) < _TERM_SURFACE_THRESHOLD:
                    existing_paths = {c.path for c in result.cards}
                    term_candidates = await _surface_term_candidates(
                        query,
                        existing_paths,
                        profile=profile,
                        scope=tool_scope,
                        deps=tool_deps,
                    )
                    block = _format_term_candidates(term_candidates)
                    text += block
                    candidates_text += block

            # Return the structured result too (not just the formatted
            # string) so the agentic loop can merge tool-round evidence into
            # source cards + citations instead of discarding everything but the
            # prompt-ready text.
            return SearchToolResult(
                text=text,
                passages=result.passages,
                cards=result.cards,
                confidence=result.confidence,
                trace=result.trace.to_dict(),
                candidates_text=candidates_text,
            )
        except Exception as exc:
            return SearchToolResult(text=f"[search failed: {exc}]")

    async def _search_exact(query: str, scope_str: str) -> SearchToolResult:
        """Non-shortening variant of :func:`_search` — the Round-0 pre-seed's
        callable (:attr:`~vesta.answer.tools.ToolRuntime.search_exact`).

        Same ``(query, scope_str) -> SearchToolResult`` shape as ``search`` (the
        :data:`~vesta.answer.tools.SearchFn` contract), so it slots into
        ``ToolRuntime`` without widening that Protocol. Skips the tool-driven
        recovery ladder (``shorten=False``) but carries the conditional
        Round-0 reformulation rung (``round0=True``) — see
        ``_search``'s parameters and ``_maybe_reformulate_round0``.
        """
        return await _search(query, scope_str, shorten=False, round0=True)

    async def _read_article(zim_id: int, path: str, *, must_include: str = "") -> str:
        try:
            arc = registry.get(zim_id)
            article = await arc.extract(path)
            text = article.text
            if not text:
                return f"[article '{path}' has no extractable text]"
            # Focused view (iter 17): the model's own read_article call used to
            # return the article verbatim, unbounded — a very long article
            # could blow well past any reasonable prefill budget. Bound it at
            # the same absolute cap the harness-driven escalation read uses
            # (``_MAX_FULL_ARTICLE_CHARS``), but pick WHICH part focuses on
            # the live question rather than truncating by position — see
            # ``answer/focus.py``'s module docstring for the measured
            # rationale (the cross-encoder can't discriminate intra-article;
            # IDF-weighted question-term overlap can). Short articles (the
            # common case) are returned unchanged.
            from vesta.answer.focus import focused_view
            from vesta.answer.tools import _MAX_FULL_ARTICLE_CHARS

            # AUDIT_0824 N11: force the card's retrieval-scored snippet into
            # this FIRST-stage window via ``must_include_spans`` — otherwise,
            # for articles longer than the 32k cap, the elision can drop the
            # passage retrieval scored highest and the harness's stage-2
            # ``find()`` (on the already-elided excerpt, agent_chat's
            # ``_capped_read``) can never recover it. The span is located on
            # the FULL text here, before any elision; same probe shape as the
            # stage-2 re-derivation so both stages agree.
            probe = must_include.strip()[:200]
            idx = text.find(probe) if probe else -1
            must_spans: tuple[tuple[int, int], ...] = ((idx, idx + len(probe)),) if idx >= 0 else ()
            view = focused_view(
                text,
                question,
                _MAX_FULL_ARTICLE_CHARS,
                breadcrumb=article.title,
                stopwords=_resolve_stopwords(sn),
                must_include_spans=must_spans,
            )
            return view.excerpt
        except Exception as exc:
            return f"[read_article failed: {exc}]"

    return ToolRuntime(
        search=_search,
        read_article=_read_article,
        search_exact=_search_exact,
    )


__all__ = ["router"]
