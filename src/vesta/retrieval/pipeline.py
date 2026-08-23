"""Pipeline engine — pure orchestration, no domain knowledge.

~150 lines that resolve components from the registry, check capabilities, drop
unmet ones, call them in order, time each stage, and record the trace. Has NO
``if strategy == ...`` anywhere. Adding a new CandidateSource is a registration +
profile edit — zero pipeline code changes.

Pipeline stages:

    QueryPreparer* → CandidateSource* → Fuser → PassageBuilder →
    PassageScorer* → ContextAssembler

* ``*`` = ordered list of components.
* ``CandidateSource*`` fan-out: call EVERY source concurrently (bounded by a
  semaphore), collect all Candidates.
* ``PassageScorer*`` is a chain: scores from step N+1 replace scores from step N.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from vesta.config.capabilities import Capability, CapabilitySet
from vesta.retrieval.contracts import (
    Budget,
    Candidate,
    FusionKey,
    PreparedQuery,
    QueryRewriter,
    RetrievalResult,
    Scope,
    ScoredPassage,
)
from vesta.retrieval.registry import resolve as resolve_component
from vesta.zim.query import CORE_QUESTION_WORDS, looks_like_question

if TYPE_CHECKING:
    from vesta.config.settings import SettingsSnapshot
    from vesta.encoders.manager import EncoderManager
    from vesta.retrieval.profiles import ProfileComponent, RetrievalProfile
    from vesta.retrieval.trace import Trace
    from vesta.vectors.contracts import VectorStore
    from vesta.zim.registry import ArchiveRegistry
    from vesta.zim.types import Passage

#: ``retrieval.context.max_per_article``/``retrieval.context.budget_tokens``
#: are *mirrored* defaults for the dev console's
#: generated profile-editor form, not pipeline-enforced ceilings — the
#: assembler's own ``Params.max_per_article``/``Params.budget_tokens``
#: (profile-level) is the real control (matching ``ordering``/``dedup``, which
#: are entirely profile-owned already). The pipeline supplies non-binding
#: ceilings here so the assembler's ``min(budget, params)`` always defers to
#: the profile value — otherwise a profile requesting a bigger budget than
#: whatever the global setting happens to hold (default 2400) would be
#: silently clamped back down, defeating e.g. ``wide_context``'s 10 000-token
#: budget out of the box.
_MAX_PER_ARTICLE_UNCAPPED = 1_000
_TOKEN_BUDGET_UNCAPPED = 1_000_000


@dataclass
class Deps:
    """Dependency-injection container passed to every pipeline component.

    Components receive deps through their constructor — never by importing
    singletons. The pipeline owns constructing components
    and wiring the deps container. Note ``retrieval/`` must not depend on ``db``:
    alias lookup and label resolution go through the
    archive registry, which is the ZIM-layer owner of those tables.
    """

    archives: ArchiveRegistry | None = None
    settings: SettingsSnapshot | None = None
    capabilities: CapabilitySet = frozenset()
    semaphore: asyncio.Semaphore | None = None
    #: Stage B model access. ``None`` outside a
    #: composition root (e.g. a unit test), in which case scorer construction
    #: falls back to a no-``encoders`` constructor the same way preparers already
    #: fall back on a missing ``archives`` (see the ``TypeError`` catch below).
    encoders: EncoderManager | None = None
    #: Dense vector store. ``None`` outside a composition
    #: root or on a vec0-less build; candidate-source construction falls back to
    #: an ``archives``-only / params-only constructor via the ``TypeError``
    #: ladder in ``_run_source`` (the sanctioned contract gap). ``retrieval/``
    #: receives the
    #: store via DI only; it never imports the ``vectors`` package at runtime.
    vectors: VectorStore | None = None
    #: Conversational query rewriter. ``None`` outside a multi-turn composition
    #: root. The
    #: ``conversational_rewrite`` preparer receives it via the ``TypeError``
    #: ladder in Step 1 (the sanctioned contract gap). ``retrieval/`` defines the
    #: :class:`~vesta.retrieval.contracts.QueryRewriter` Protocol itself and
    #: never imports ``inference/``; the composition root builds the concrete
    #: gateway-backed implementation and injects it here.
    rewriter: QueryRewriter | None = None


class NoCandidatesError(RuntimeError):
    """Every candidate source was dropped or returned nothing (hard requirement).

    Carries the trace built so far — degradations, per-source candidate counts,
    everything up to the point of failure — so a caller can still explain *why*
    there was nothing to search with, instead of losing the trace to a bare
    exception (the trace is a first-class output, including
    on this path).
    """

    def __init__(self, trace: Trace) -> None:
        self.trace = trace
        super().__init__(
            "All candidate sources were dropped or produced no candidates. "
            "Nothing to search — pipeline cannot continue."
        )


def _check_capabilities(
    cls: Any, kind: str, name: str, capabilities: CapabilitySet, tr: Trace
) -> bool:
    """Return True if ``cls``'s ``requires`` are met by the live capability set.

    When unmet, records a degradation entry in ``tr`` so the eval harness can
    detect and reject comparisons where components were dropped.
    """
    requires: frozenset[Capability] = getattr(cls, "requires", frozenset())
    if not requires:
        return True
    unmet = requires - capabilities
    if unmet:
        tr.degraded(
            component=f"{kind}/{name}",
            missing=next(iter(unmet)),
            reason=f"capability {' '.join(unmet)} not available",
        )
        return False
    return True


def _resolve_params(cls: Any, params: dict[str, Any]) -> Any:
    """Instantiate the component's ``Params`` model from a raw dict.

    Returns ``None`` if the class has no ``Params`` attribute or the params dict
    is empty and no defaults are needed.
    """
    params_cls = getattr(cls, "Params", None)
    if params_cls is None:
        return None
    return params_cls(**params)


async def run_pipeline(  # noqa: PLR0912,PLR0915
    profile: RetrievalProfile,
    query: str,
    scope: Scope,
    deps: Deps,
    *,
    history: tuple[tuple[str, str], ...] = (),
) -> RetrievalResult:
    """Execute the retrieval pipeline for a profile and query.

    Returns a ``RetrievalResult`` with scored passages, source cards, trace, and
    confidence signals. The trace records every stage, component, params, timing,
    and degradation decision.

    Raises :class:`NoCandidatesError` if every candidate source is dropped — a
    hard requirement: with nothing to search, there is nothing to degrade to.

    ``history`` (09) is the prior conversation as ``(role, content)`` pairs,
    threaded into ``PreparedQuery.history`` so the conversational-rewrite preparer
    can resolve pronouns/context on turn ≥ 2. Defaults empty (turn 1 / single-shot).
    """
    from vesta.retrieval.trace import Trace

    tr = Trace()
    tr.set_profile(profile.name, profile.hash)
    capabilities = deps.capabilities

    # ── Step 0: bootstrap PreparedQuery ───────────────────────────────────
    pq = PreparedQuery(
        raw=query,
        terms=(),
        text=query.strip(),
        aliases=(),
        is_keyword_query=not _looks_like_question(query),
        rung="initial",
        history=history,
    )

    # ── Step 1: QueryPreparers (ordered chain) ────────────────────────────
    for pc in profile.preparers:
        cls = resolve_component("query_preparer", pc.impl)
        if cls is None:
            continue
        if not _check_capabilities(cls, "preparer", pc.impl, capabilities, tr):
            continue
        params = _resolve_params(cls, pc.params)
        try:
            instance = cls(params=params, archives=deps.archives, rewriter=deps.rewriter)
        except TypeError:
            # Preparers fall back rung by rung as they drop deps they don't take:
            # ``rewriter`` (only ``conversational_rewrite`` accepts it)
            # → ``archives`` (``alias_expand``) → params-only
            # (``normalize``). Same ladder pattern ``_run_source`` uses for the
            # ``vectors``/``encoders`` deps.
            try:
                instance = cls(params=params, archives=deps.archives)
            except TypeError:
                # Preparers that take no archives dep fall back to params-only.
                try:
                    instance = cls(params=params)
                except Exception:
                    continue
            except Exception:
                continue
        except Exception:
            continue
        with tr.stage("preparer", pc.impl, pc.params) as st:
            st.add_inputs({"terms": list(pq.terms), "text": pq.text})
            try:
                pq = await instance.prepare(pq, tr)
            except Exception as exc:
                st.add_outputs({"error": str(exc)})
            st.add_outputs({"terms": list(pq.terms), "text": pq.text})

    # ── Step 2: CandidateSources (concurrent fan-out) ─────────────────────
    all_candidates: dict[FusionKey, list[Candidate]] = {}
    sem = deps.semaphore or asyncio.Semaphore(4)

    async def _run_source(pc: ProfileComponent) -> None:
        cls = resolve_component("candidate_source", pc.impl)
        if cls is None:
            return
        if not _check_capabilities(cls, "candidate_source", pc.impl, capabilities, tr):
            return
        params = _resolve_params(cls, pc.params)
        try:
            instance = cls(
                params=params,
                archives=deps.archives,
                vectors=deps.vectors,
                encoders=deps.encoders,
            )
        except TypeError:
            # Sources without a vectors/encoders dep (every lexical source today;
            # ``vector_knn`` is the first to accept them) fall
            # back to archives-only, then to params-only — the same ladder scorers
            # use for the ``encoders`` dep.
            try:
                instance = cls(params=params, archives=deps.archives)
            except TypeError:
                try:
                    instance = cls(params=params) if params is not None else cls()
                except Exception:
                    return
            except Exception:
                return
        except Exception:
            return
        async with sem:
            with tr.stage("candidate_source", pc.impl, pc.params) as st:
                st.add_inputs({"terms": list(pq.terms)})
                try:
                    cands = await instance.find(pq, scope, tr)
                except Exception as exc:
                    st.add_outputs({"error": str(exc)})
                    cands = []
                st.add_outputs({"candidate_count": len(cands)})
        for c in cands:
            k = FusionKey(zim_id=c.zim_id, source=c.source)
            all_candidates.setdefault(k, []).append(c)

    await asyncio.gather(*(_run_source(pc) for pc in profile.sources))

    # Agreement signal: Jaccard overlap between the candidate sets
    # produced by distinct sources. Computed here — only the pipeline sees the
    # pre-fusion per-source sets — and folded into ConfidenceSignals after the
    # assembler returns. Recorded, not acted on.
    agreement = _source_agreement(all_candidates)

    # Hard requirement: at least one source must produce candidates or the
    # pipeline fails loudly (degrade, not silently produce empty). Note a
    # profile with zero sources is rejected at load time (profiles.py); this
    # guards the "all sources capability-dropped or empty" case.
    if not all_candidates:
        raise NoCandidatesError(tr)

    # ── Step 3: Fuser ─────────────────────────────────────────────────────
    fuser_cls = resolve_component("fuser", profile.fusion.impl)
    fused: list[Candidate] = []
    if fuser_cls is not None and _check_capabilities(
        fuser_cls, "fuser", profile.fusion.impl, capabilities, tr
    ):
        params = _resolve_params(fuser_cls, profile.fusion.params)
        try:
            fuser = fuser_cls(params=params) if params is not None else fuser_cls()
        except Exception:
            fuser = None
        if fuser is not None:
            with tr.stage("fuser", profile.fusion.impl, profile.fusion.params) as st:
                st.add_inputs(
                    {
                        "group_count": len(all_candidates),
                        "total_candidates": sum(len(v) for v in all_candidates.values()),
                    }
                )
                try:
                    fused = list(fuser.fuse(all_candidates, tr))
                except Exception as exc:
                    st.add_outputs({"error": str(exc)})
                    fused = list(_flat_union(all_candidates))
                st.add_outputs({"fused_count": len(fused)})
    if not fused:
        fused = list(_flat_union(all_candidates))

    # ── Step 4: PassageBuilder ────────────────────────────────────────────
    passages_cls = resolve_component("passage_builder", profile.passages.impl)
    passages: list[Passage] = []
    if passages_cls is not None and _check_capabilities(
        passages_cls, "passage_builder", profile.passages.impl, capabilities, tr
    ):
        params = _resolve_params(passages_cls, profile.passages.params)
        try:
            builder = passages_cls(params=params, archives=deps.archives)
        except Exception:
            builder = None
        if builder is not None:
            with tr.stage("passage_builder", profile.passages.impl, profile.passages.params) as st:
                st.add_inputs({"candidate_count": len(fused)})
                try:
                    passages = await builder.build(fused, pq, tr)
                except Exception as exc:
                    st.add_outputs({"error": str(exc)})
                st.add_outputs({"passage_count": len(passages)})

    # ── Step 5: PassageScorers (chain) ────────────────────────────────────
    scored: list[ScoredPassage] = [
        ScoredPassage(passage=p, score=0.0, source_info="initial") for p in passages
    ]

    for pc in profile.scorers:
        cls = resolve_component("passage_scorer", pc.impl)
        if cls is None:
            continue
        if not _check_capabilities(cls, "passage_scorer", pc.impl, capabilities, tr):
            continue
        params = _resolve_params(cls, pc.params)
        try:
            scorer = cls(params=params, encoders=deps.encoders)
        except TypeError:
            # Scorers with no encoders dep (e.g. lexical_overlap) fall back to
            # params-only construction — the same dance preparers already do
            # for the ``archives`` dep.
            try:
                scorer = cls(params=params) if params is not None else cls()
            except Exception:
                continue
        except Exception:
            continue
        with tr.stage("passage_scorer", pc.impl, pc.params) as st:
            st.add_inputs({"passage_count": len(scored)})
            try:
                scored = list(await scorer.score(scored, pq, tr))
            except Exception as exc:
                st.add_outputs({"error": str(exc)})
            st.add_outputs({"passage_count": len(scored)})

    # ── Step 6: ContextAssembler ──────────────────────────────────────────
    assembler_cls = resolve_component("context_assembler", profile.assembler.impl)
    if assembler_cls is None:
        raise RuntimeError(f"context_assembler {profile.assembler.impl!r} not found in registry")

    if not _check_capabilities(
        assembler_cls, "context_assembler", profile.assembler.impl, capabilities, tr
    ):
        raise RuntimeError(f"context_assembler {profile.assembler.impl!r} has unmet capabilities")

    params = _resolve_params(assembler_cls, profile.assembler.params)
    try:
        assembler = assembler_cls(params=params, archives=deps.archives)
    except Exception:
        assembler = assembler_cls()

    with tr.stage("context_assembler", profile.assembler.impl, profile.assembler.params) as st:
        st.add_inputs({"passage_count": len(scored)})
        budget = Budget(
            token_total=_TOKEN_BUDGET_UNCAPPED,
            max_per_article=_MAX_PER_ARTICLE_UNCAPPED,
        )
        result_maybe: Any = assembler.assemble(scored, budget, pq, tr)
        if not isinstance(result_maybe, RetrievalResult):  # safety cast
            raise RuntimeError("ContextAssembler did not return a RetrievalResult")
        # Fold the upstream agreement signal (computed pre-fusion in Step 2)
        # into the confidence object; the assembler cannot see the per-source
        # candidate sets, so it leaves ``agreement`` at its default.
        result = RetrievalResult(
            passages=result_maybe.passages,
            cards=result_maybe.cards,
            trace=result_maybe.trace,
            confidence=replace(result_maybe.confidence, agreement=agreement),
        )
        st.add_outputs({"selected_count": len(result.passages), "card_count": len(result.cards)})

    return result


def _looks_like_question(text: str) -> bool:
    """True when ``text`` looks like a question rather than a keyword query.

    Uses the CORE interrogative set (see :mod:`vesta.zim.query`): widening it
    would change which queries retrieve as keyword lookups — a measured
    retrieval change, not a cleanup.
    """
    return looks_like_question(text, CORE_QUESTION_WORDS)


def _flat_union(groups: dict[FusionKey, list[Candidate]]) -> list[Candidate]:
    """Flat union of all candidates, preserving within-source rank ordering."""
    result: list[Candidate] = []
    overall = 0
    for key in sorted(groups, key=lambda k: (k.zim_id, k.source)):
        for c in groups[key]:
            result.append(
                Candidate(
                    zim_id=c.zim_id,
                    path=c.path,
                    source=c.source,
                    rank=overall,
                    score=c.score,
                )
            )
            overall += 1
    return result


def _source_agreement(all_candidates: dict[FusionKey, list[Candidate]]) -> float:
    """Mean pairwise Jaccard overlap between distinct candidate sources.

    Generalizes "FTS-vs-suggest agreement" to N sources: the candidate path-sets
    of every pair of distinct sources are compared and the mean Jaccard is
    returned in ``[0, 1]``. With fewer than two sources the signal is undefined
    and returns ``0.0``. This naturally extends to dense sources.
    """
    by_source: dict[str, set[str]] = {}
    for key, cands in all_candidates.items():
        by_source.setdefault(key.source, set()).update(c.path for c in cands)
    sources = list(by_source.values())
    if len(sources) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            union = sources[i] | sources[j]
            if not union:
                continue
            total += len(sources[i] & sources[j]) / len(union)
            pairs += 1
    return total / pairs if pairs else 0.0
