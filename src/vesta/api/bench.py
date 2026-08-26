"""Unified benchmark composition root — systems, judge, store.

``api/`` is the composition root: it wires the five
:class:`~vesta.eval.bench_runner.SystemUnderTest` implementations, the
:class:`GatewayJudgeLLM` (moved from ``api/benchmark.py`` and extended), and the
:class:`SqliteBenchStore` (concrete :class:`~vesta.eval.bench_runner.BenchStore`
over ``bench_runs`` / ``bench_question_results`` / ``bench_judge_cache``) into
the boundary-clean :mod:`vesta.eval.bench_runner`.

This module exports the building blocks the CLI and the ``/api/bench/*`` routes
compose.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any

import aiosqlite
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from vesta import config as app_config
from vesta.answer.abstention import ABSTENTION_NO_MATCH
from vesta.answer.contracts import (
    AnswerResetEvent,
    CitationsEvent,
    DoneEvent,
    ErrorEvent,
    SourcesEvent,
    StatusEvent,
    TokenEvent,
    TraceEvent,
)
from vesta.api.answer import iter_answer_events
from vesta.api.state import AppState, app_state
from vesta.eval.answer_metrics import JudgeLLM
from vesta.eval.bench_dataset import (
    BENCH_DATASET,
    BenchQuestion,
    apply_flag_filters,
    load_bench_dataset,
    subset_hash,
)
from vesta.eval.bench_runner import (
    BENCH_JUDGE_CONCURRENCY,
    BENCH_MAX_CONCURRENT,
    BENCH_REPEATS,
    BENCH_SYSTEMS,
    BenchQuestionResult,
    BenchRunRecord,
    BenchStore,
    IncomparableRuns,
    QuestionOutput,
    compare_runs,
    resolve_judge_concurrency,
    resolve_matrix_axes,
    run_benchmark,
)
from vesta.eval.bench_scoring import (
    BENCH_JUDGE_MAX_TOKENS,
    BENCH_JUDGE_TEMPERATURE,
    JudgeOutcome,
    Verdict,
    _parse_judge_json,
    metric_lookup,
)
from vesta.eval.golden import (
    EVAL_JUDGE_API_KEY,
    EVAL_JUDGE_ENDPOINT_URL,
    EVAL_JUDGE_MODEL,
)
from vesta.eval.runner import now_iso
from vesta.inference import (
    INFERENCE_LLM_API_KEY,
    INFERENCE_LLM_ENDPOINT_URL,
    INFERENCE_LLM_MODEL,
)
from vesta.retrieval.profiles import resolve_profile_from_settings


async def _find_archive_by_name(state: AppState, zim: str) -> Any:
    """Find an open archive by ZIM filename or name.

    The single copy of the lookup both bench system classes share (the
    end-to-end oracle and the answer-only replay) — byte-identical twins here
    would silently diverge on which archive a bench question resolves to.
    """
    db = state.db
    async with db.read() as conn:
        cur = await conn.execute(
            "SELECT id FROM zims WHERE filename=? OR name=? LIMIT 1", (zim, zim)
        )
        row = await cur.fetchone()
    if row is None:
        return None
    registry = state.registry
    if registry is None:
        return None
    with suppress(KeyError):
        return registry.get(int(row[0]))
    return None


# ── GatewayJudgeLLM (moved from api/benchmark.py, extended) ─────────────────


class GatewayJudgeLLM:
    """Wire the inference gateway into the ``JudgeLLM`` Protocol.

    ``enable_thinking=False`` is mandatory for judging (the 4B/20B are reasoning
    models that otherwise burn the token budget on hidden CoT — confirmed
    live: gpt-oss-20b with max_tokens=16 returned ``"P"``). Temperature
    0 makes judging deterministic so a stored run re-judged by a stronger model
    is byte-identical at any judge_concurrency.
    """

    def __init__(
        self,
        gateway: Any,
        model: str,
        *,
        temperature: float = float(BENCH_JUDGE_TEMPERATURE.default),
        max_tokens: int = int(BENCH_JUDGE_MAX_TOKENS.default),
    ) -> None:
        self._gateway = gateway
        self._model = model
        #: Public: the judge cache key folds these into its digest so verdicts
        #: minted at one operating point / endpoint are never served for another.
        self.temperature = temperature
        self.max_tokens = max_tokens
        base = getattr(gateway, "_inner", gateway)  # unwrap UsageRecorder if wrapped
        self.endpoint = str(getattr(base, "base_url", ""))

    async def judge(self, prompt: str) -> str:
        from vesta.inference.gateway import ChatMessage

        # Cline (and reasoning sinks) only answer via streaming — a non-streaming
        # chat completion against the cline route returns an empty body. Stream
        # and assemble the content text: chat_stream's .text carries only the
        # content deltas, never the hidden thinking/reasoning tokens.
        parts: list[str] = []
        async for delta in self._gateway.chat_stream(
            [ChatMessage(role="user", content=prompt)],
            model=self._model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            enable_thinking=False,
        ):
            if delta.text:
                parts.append(delta.text)
        return "".join(parts)


def build_judge_gateway(state: AppState) -> Any:
    """The gateway the benchmark's judge talks to (reused pattern).

    When ``eval.judge.endpoint_url`` is set, construct a dedicated
    OpenAI-compatible gateway for the judge (the caller owns it and must
    ``aclose()`` it after the run). Otherwise reuse ``state.gateway``.
    """
    endpoint = str(app_config.get(EVAL_JUDGE_ENDPOINT_URL)).strip()
    if endpoint:
        from vesta.inference.gateway import OpenAIGateway

        return OpenAIGateway(
            base_url=endpoint,
            api_key=str(app_config.get(EVAL_JUDGE_API_KEY)),
        )
    return state.gateway


def make_judge_llm(
    state: AppState,
    judge_model: str,
) -> tuple[JudgeLLM | None, Any | None]:
    """Construct the benchmark's LLM-judge from settings.

    Returns ``(judge, owned_gateway)``. ``owned_gateway`` is a dedicated judge
    gateway this call created (reusing ``state.gateway`` yields ``None``); the
    caller must close it in a ``finally`` once the run is done.
    """
    if not judge_model:
        return None, None
    from vesta.inference.gateway import NullGateway

    judge_gateway = build_judge_gateway(state)
    if judge_gateway is None or isinstance(judge_gateway, NullGateway):
        return None, None
    owned = judge_gateway if judge_gateway is not state.gateway else None
    judge: JudgeLLM = GatewayJudgeLLM(
        judge_gateway,
        judge_model,
        temperature=float(app_config.get(BENCH_JUDGE_TEMPERATURE)),
        max_tokens=int(app_config.get(BENCH_JUDGE_MAX_TOKENS)),
    )
    return judge, owned


# ── In-process answer driver (reduced from api/benchmark.py's InProcessAnswerDriver) ──


async def _drive_answer_events(  # noqa: PLR0912
    state: AppState,
    question: str,
    *,
    profile: str | None,
    scope: str | None,
    strategy: str | None,
) -> QuestionOutput:
    """Iterate ``iter_answer_events`` and reduce to a :class:`QuestionOutput`.

    Faithful to ``GET /api/answer`` minus the SSE serializer:
    ``SourcesEvent.cards[].path`` → ``retrieved_paths``; ``CitationsEvent.
    answer_text`` → final answer; abstention token detection.

    The gateway is wrapped in a :class:`UsageRecorder` for the duration of the
    call so every LLM round (generation, reformulation, rewrite, tool decisions)
    is attributed to this question's token count. ``state.gateway`` is restored
    in ``finally`` — cells run at pipeline concurrency 1 and whole groups are
    serialized by ``_RUN_GATE``, so the swap never nests.
    """
    from vesta.inference.gateway import UsageRecorder

    inner_gw = state.gateway
    recorder = UsageRecorder(inner_gw) if inner_gw is not None else None
    if recorder is not None:
        state.gateway = recorder
    answer_parts: list[str] = []
    final_answer: str | None = None
    cards: list[Any] = []
    abstained = False
    error: str | None = None
    trace: dict[str, object] = {}
    try:
        async for ev in iter_answer_events(state, question, scope, profile, strategy):
            if isinstance(ev, TokenEvent):
                answer_parts.append(ev.text)
            elif isinstance(ev, AnswerResetEvent):
                answer_parts = []
            elif isinstance(ev, CitationsEvent):
                if ev.answer_text is not None:
                    final_answer = ev.answer_text
            elif isinstance(ev, SourcesEvent):
                cards.extend(ev.cards)
            elif isinstance(ev, StatusEvent) and ev.phase == "abstaining":
                abstained = True
            elif isinstance(ev, TraceEvent):
                trace = ev.trace
            elif isinstance(ev, ErrorEvent):
                error = f"{ev.code}: {ev.message}"
            elif isinstance(ev, DoneEvent):
                pass
    finally:
        if recorder is not None:
            state.gateway = inner_gw
    answer_text = final_answer if final_answer is not None else "".join(answer_parts)
    if answer_text.strip() == ABSTENTION_NO_MATCH.strip():
        abstained = True
    retrieved = tuple(c.path for c in cards)
    return QuestionOutput(
        answer_text=answer_text,
        retrieved_paths=retrieved,
        abstained=abstained,
        error=error,
        trace=trace,
        resolved_strategy=strategy or "",
        # rounds is 0 for every system that reaches here (sources_only /
        # retrieval_only); the recorded column keeps its historical value
        # for past runs only.
        input_tokens=recorder.input_tokens if recorder else 0,
        output_tokens=recorder.output_tokens if recorder else 0,
    )


# ── The five registered SystemUnderTest implementations ─────────────────────


class _BaseSystem:
    """Shared metadata pins for all benchmark systems."""

    name: str = ""

    def __init__(
        self,
        *,
        answer_model: str = "",
        profile_name: str = "",
        profile_hash: str = "",
    ) -> None:
        self.answer_model = answer_model
        self.profile_name = profile_name
        self.profile_hash = profile_hash


class RetrievalOnlySystem(_BaseSystem):
    """Runs the retrieval pipeline only — returns cards, EMPTY answer.

    Zero LLM calls (mode 1): the runner bypasses the judge entirely and
    the answer metrics are reported as skipped. With ``collect=True`` the
    assembled passages are also kept per question so the CLI can write a
    context snapshot for answer-only replay (mode 2).
    """

    name = "retrieval_only"
    generates_answers = False

    def __init__(
        self,
        state: AppState,
        *,
        profile: str | None = None,
        scope: str | None = None,
        collect: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._state = state
        self._profile = profile
        self._scope = scope
        self._collect = collect
        self._context: dict[str, dict[str, object]] = {}

    def context_snapshot(self) -> dict[str, dict[str, object]]:
        """Per-question frozen retrieval outcome: ``{qid: {paths, passages}}``."""
        return dict(self._context)

    async def run_one(self, q: BenchQuestion) -> QuestionOutput:
        from vesta.api.answer import (
            _build_rewriter,
            _concurrency_bound,
            _parse_scope,
            _resolve_profile,
        )
        from vesta.config.capabilities import compute_capabilities
        from vesta.retrieval.pipeline import Deps, NoCandidatesError, run_pipeline
        from vesta.vectors import get_store as get_vector_store

        try:
            sn = app_config.snapshot()
        except RuntimeError:
            sn = None
        capabilities = compute_capabilities()
        retrieval_profile = _resolve_profile(self._profile, snapshot=sn)
        ret_scope = _parse_scope(self._scope, self._state.registry)
        rewriter = _build_rewriter(self._state, sn)
        deps = Deps(
            archives=self._state.registry,
            settings=sn,
            capabilities=capabilities,
            semaphore=asyncio.Semaphore(_concurrency_bound(sn)),
            encoders=self._state.encoders,
            vectors=get_vector_store(),
            rewriter=rewriter,
        )
        try:
            result = await run_pipeline(
                profile=retrieval_profile,
                query=q.question,
                scope=ret_scope,
                deps=deps,
            )
        except NoCandidatesError as exc:
            if self._collect:
                self._context[q.id] = {"paths": [], "passages": []}
            return QuestionOutput(
                answer_text="",
                retrieved_paths=(),
                abstained=True,
                error=None,
                trace=exc.trace.to_dict(),
                resolved_strategy="retrieval_only",
            )
        paths = tuple(c.path for c in result.cards)
        if self._collect:
            titles = {c.path: c.title for c in result.cards}
            self._context[q.id] = {
                "paths": list(paths),
                "passages": [
                    {
                        "path": sp.passage.path,
                        "title": titles.get(sp.passage.path, sp.passage.breadcrumb),
                        "breadcrumb": sp.passage.breadcrumb,
                        "score": sp.score,
                        "text": sp.passage.text,
                    }
                    for sp in result.passages
                ],
            }
        return QuestionOutput(
            answer_text="",
            retrieved_paths=paths,
            abstained=False,
            error=None,
            trace=result.trace.to_dict(),
            resolved_strategy="retrieval_only",
        )


class SourcesOnlySystem(_BaseSystem):
    """Drives the live ``sources_only`` answer strategy via iter_answer_events."""

    name = "sources_only"

    def __init__(
        self,
        state: AppState,
        *,
        profile: str | None = None,
        scope: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._state = state
        self._profile = profile
        self._scope = scope

    async def run_one(self, q: BenchQuestion) -> QuestionOutput:
        return await _drive_answer_events(
            self._state,
            q.question,
            profile=self._profile,
            scope=self._scope,
            strategy="sources_only",
        )


class AgenticPydanticSystem(_BaseSystem):
    """Drives ``api/agent_chat.run_one_turn`` — the pydantic-ai agent.

    ``TurnResult.cards[].path`` → retrieved_paths, ``.answer`` → answer_text,
    ``.tool_calls`` → tool_calls. NOW persisted and comparable.
    """

    name = "agentic_pydantic"

    def __init__(
        self,
        state: AppState,
        *,
        model_id: str = "",
        endpoint: str = "",
        api_key: str = "",
        enable_thinking: bool | None = None,
        profile: str | None = None,
        scope: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._state = state
        self._model_id = model_id
        self._endpoint = endpoint
        self._api_key = api_key
        self._enable_thinking = enable_thinking
        self._profile = profile
        self._scope = scope

    async def run_one(self, q: BenchQuestion) -> QuestionOutput:
        from vesta.api.agent_chat import looks_abstained, run_one_turn

        try:
            sn = app_config.snapshot()
        except RuntimeError:
            sn = None
        result = await run_one_turn(
            self._state,
            sn,
            q.question,
            model_id=self._model_id,
            endpoint=self._endpoint,
            api_key=self._api_key,
            enable_thinking=self._enable_thinking,
            profile_override=self._profile,
            scope=self._scope,
        )
        answer = result.answer
        abstained = looks_abstained(answer)
        paths = tuple(c.path for c in result.cards)
        return QuestionOutput(
            answer_text=answer,
            retrieved_paths=paths,
            abstained=abstained,
            error=None,
            trace={
                **result.trace,
                "system": "agentic_pydantic",
                "elapsed_ms": result.elapsed_ms,
                "total_tokens": result.total_tokens,
            },
            resolved_strategy="agentic_pydantic",
            tool_calls=len(result.tool_calls),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )


_MAX_CONTEXT_CHARS = (
    90_000  # ~28k tokens at worst-case density (~3.2 chars/token), under the 32k window
)


async def build_oracle_context(
    state: AppState,
    q: BenchQuestion,
    find_archive: Any,
) -> tuple[str, tuple[str, ...]]:
    """Budgeted gold-article context + the required gold paths it was built from.

    Shared by ``oracle`` (the ceiling) and ``answer_only --oracle-context``.
    A missing/unreadable article degrades (drops its text) — an imperfect
    oracle is harder, not broken.
    """
    context_parts: list[str] = []
    gold_paths: list[str] = []
    for src in q.sources:
        if not src.required:
            continue
        gold_paths.append(src.article_path)
        archive = await find_archive(src.zim)
        if archive is None:
            continue
        try:
            article = await archive.extract(src.article_path)
        except Exception:
            continue
        context_parts.append(f"=== {src.article_title} ===\n{article.text}")
    return _budget_context(context_parts), tuple(gold_paths)


# ── Context snapshots (mode 2: freeze retrieval, replay generation) ─────────


CONTEXT_SNAPSHOT_FORMAT = "vesta-bench-context-snapshot/1"


def write_context_snapshot(
    path: str | Path,
    *,
    questions: Mapping[str, Mapping[str, object]],
    dataset: Any,
    subset_hash_val: str,
    system: str,
    profile: str,
    level: int | None,
) -> Path:
    """Write a retrieval-context snapshot for answer-only replay.

    ``questions`` is ``{qid: {paths, passages}}`` as collected by
    :class:`RetrievalOnlySystem`. The dataset pin (name + full hash +
    subset hash) travels with the file so a replay can prove it is feeding
    the same questions the retrieval run measured.
    """
    import datetime as _dt

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CONTEXT_SNAPSHOT_FORMAT,
        "generated": _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat(),
        "system": system,
        "profile": profile,
        "level": level,
        "dataset": {
            "name": str(getattr(dataset, "name", "")),
            "hash": str(getattr(dataset, "hash", "")),
            "subset_hash": subset_hash_val,
        },
        "questions": dict(questions),
    }
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


def load_context_snapshot(path: str | Path) -> dict[str, object]:
    """Load + shape-check a context snapshot written by :func:`write_context_snapshot`."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read context snapshot {p}: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != CONTEXT_SNAPSHOT_FORMAT:
        raise ValueError(f"{p}: not a context snapshot (expected format {CONTEXT_SNAPSHOT_FORMAT})")
    qs = data.get("questions")
    if not isinstance(qs, dict) or not qs:
        raise ValueError(f"{p}: snapshot carries no questions")
    return data


def snapshot_missing_ids(snapshot: Mapping[str, object], qids: Sequence[str]) -> tuple[str, ...]:
    """Question ids a replay selected that the snapshot does not cover."""
    qs = snapshot.get("questions")
    covered = set(qs) if isinstance(qs, Mapping) else set()
    return tuple(qid for qid in qids if qid not in covered)


def _budget_context(parts: Sequence[str]) -> str:
    """Join article contexts, truncating tail-first so the total fits the window.

    A single article (always <= the budget here) passes through whole. When
    several long articles overflow the window, each is capped at an equal share
    — keeping the lead and infobox, where the gold facts live — so the combined
    prompt fits.
    """
    if not parts:
        return "[no source articles available]"
    total = sum(len(p) for p in parts)
    if total <= _MAX_CONTEXT_CHARS:
        return "\n\n".join(parts)
    share = _MAX_CONTEXT_CHARS // len(parts)
    return "\n\n".join(p[:share] for p in parts)


class OracleSystem(_BaseSystem):
    """Gold article text as context, ONE generation (the ceiling).

    Reads the archive via the registry, extracts each required source's article
    text, and budgets the combined context to fit the model window. Single
    articles pass through whole; multi-source questions that overflow are
    truncated tail-first (the lead + infobox hold the gold facts), so the
    ceiling measures answerability rather than context-fit.
    """

    name = "oracle"

    def __init__(
        self,
        state: AppState,
        *,
        model_id: str = "",
        endpoint: str = "",
        api_key: str = "",
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._state = state
        self._model_id = model_id
        self._endpoint = endpoint
        self._api_key = api_key
        self._max_tokens = max_tokens

    async def _find_archive(self, zim: str) -> Any:
        """Find an open archive by ZIM filename or name."""
        return await _find_archive_by_name(self._state, zim)

    async def _oracle_context(self, q: BenchQuestion) -> tuple[str, tuple[str, ...]]:
        """Budgeted gold-article context + the gold paths it was built from."""
        return await build_oracle_context(self._state, q, self._find_archive)

    async def run_one(self, q: BenchQuestion) -> QuestionOutput:
        from vesta.inference.gateway import ChatMessage, NullGateway

        gw = self._state.gateway
        if gw is None or isinstance(gw, NullGateway) or not self._model_id:
            return QuestionOutput(
                answer_text="",
                retrieved_paths=(),
                abstained=True,
                error="no LLM gateway configured",
                trace={"system": "oracle"},
                resolved_strategy="oracle",
            )
        # Read each required source's full article text, then budget the
        # combined context to the model window (shared with answer_only's
        # oracle mode). Single articles pass through whole; multi-source
        # questions that overflow are truncated tail-first.
        context, gold_paths = await self._oracle_context(q)
        prompt = (
            f"Use the following source articles to answer the question.\n\n"
            f"SOURCE ARTICLES:\n{context}\n\n"
            f"QUESTION: {q.question}\n\nAnswer concisely."
        )
        try:
            res = await gw.chat_once(
                [ChatMessage(role="user", content=prompt)],
                model=self._model_id,
                temperature=0.0,
                max_tokens=self._max_tokens,
                enable_thinking=False,
            )
            answer_text = str(res.text)
        except Exception as exc:
            return QuestionOutput(
                answer_text="",
                retrieved_paths=tuple(gold_paths),
                abstained=False,
                error=f"oracle generation failed: {exc!r}",
                trace={"system": "oracle"},
                resolved_strategy="oracle",
            )
        return QuestionOutput(
            answer_text=answer_text,
            retrieved_paths=tuple(gold_paths),
            abstained=False,
            error=None,
            trace={"system": "oracle"},
            resolved_strategy="oracle",
            input_tokens=res.input_tokens,
            output_tokens=res.output_tokens,
        )


class ClosedBookSystem(_BaseSystem):
    """Question only, no context, ONE generation (the floor)."""

    name = "closed_book"

    def __init__(
        self,
        state: AppState,
        *,
        model_id: str = "",
        endpoint: str = "",
        api_key: str = "",
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._state = state
        self._model_id = model_id
        self._endpoint = endpoint
        self._api_key = api_key
        self._max_tokens = max_tokens

    async def run_one(self, q: BenchQuestion) -> QuestionOutput:
        from vesta.inference.gateway import ChatMessage, NullGateway

        gw = self._state.gateway
        if gw is None or isinstance(gw, NullGateway) or not self._model_id:
            return QuestionOutput(
                answer_text="",
                retrieved_paths=(),
                abstained=True,
                error="no LLM gateway configured",
                trace={"system": "closed_book"},
                resolved_strategy="closed_book",
            )
        prompt = f"QUESTION: {q.question}\n\nAnswer concisely."
        try:
            res = await gw.chat_once(
                [ChatMessage(role="user", content=prompt)],
                model=self._model_id,
                temperature=0.0,
                max_tokens=self._max_tokens,
                enable_thinking=False,
            )
            answer_text = str(res.text)
        except Exception as exc:
            return QuestionOutput(
                answer_text="",
                retrieved_paths=(),
                abstained=False,
                error=f"closed_book generation failed: {exc!r}",
                trace={"system": "closed_book"},
                resolved_strategy="closed_book",
            )
        return QuestionOutput(
            answer_text=answer_text,
            retrieved_paths=(),
            abstained=False,
            error=None,
            trace={"system": "closed_book"},
            resolved_strategy="closed_book",
            input_tokens=res.input_tokens,
            output_tokens=res.output_tokens,
        )


class AnswerOnlySystem(_BaseSystem):
    """Generation on FROZEN context — the mode-2 grounding benchmark.

    Two context sources, chosen at construction:

    - ``context_path``: a snapshot written by a ``retrieval_only`` run
      (``--from-context``). Every replayed model/prompt sees the *identical*
      passages, so score deltas are attributable to synthesis, not search.
      ``retrieved_paths`` replay the snapshot's frozen card order — source
      metrics stay meaningful (``source_found`` = the snapshot contained gold).
    - ``oracle_context=True``: gold article text (``--oracle-context``),
      built by the shared :func:`build_oracle_context`.

    Unlike ``oracle`` (the raw ceiling), the prompt carries an explicit
    abstention clause — adversarial/out-of-corpus replay must be able to
    refuse, or ``abstention_correctness`` / ``hallucination_rate`` measure
    nothing.
    """

    name = "answer_only"

    def __init__(
        self,
        state: AppState,
        *,
        model_id: str = "",
        endpoint: str = "",
        api_key: str = "",
        max_tokens: int = 1024,
        context_path: str | None = None,
        oracle_context: bool = False,
        context_passages: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if bool(context_path) == bool(oracle_context):
            raise ValueError(
                "answer_only needs exactly one context source: "
                "context_path (snapshot) or oracle_context"
            )
        if context_passages is not None:
            if oracle_context:
                raise ValueError("context_passages applies to snapshot replay only")
            if context_passages < 1:
                raise ValueError("context_passages must be >= 1")
        self._state = state
        self._model_id = model_id
        self._endpoint = endpoint
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._mode = "snapshot" if context_path else "oracle"
        self._context_path = context_path
        # Pre-seed sensitivity knob: replay only the top N
        # snapshot passages (rank order — what a thinner agent pre-seed
        # means). ``None`` = every passage (today's behaviour). Exposed as a
        # public pin so the runner records it in ``config_json``.
        self.context_passages = context_passages
        self._snapshot: dict[str, object] = {}
        if context_path is not None:
            # Load eagerly — a malformed/missing snapshot must fail the run
            # before any LLM call, not midway through the matrix.
            self._snapshot = load_context_snapshot(context_path)

    async def _find_archive(self, zim: str) -> Any:
        """Find an open archive by ZIM filename or name."""
        return await _find_archive_by_name(self._state, zim)

    def _snapshot_entry(self, qid: str) -> Mapping[str, object]:
        qs = self._snapshot.get("questions")
        if not isinstance(qs, Mapping):
            return {}
        entry = qs.get(qid)
        return entry if isinstance(entry, Mapping) else {}

    async def run_one(self, q: BenchQuestion) -> QuestionOutput:
        from vesta.inference.gateway import ChatMessage, NullGateway

        trace: dict[str, object] = {"system": "answer_only", "context": self._mode}
        gw = self._state.gateway
        if gw is None or isinstance(gw, NullGateway) or not self._model_id:
            return QuestionOutput(
                answer_text="",
                retrieved_paths=(),
                abstained=True,
                error="no LLM gateway configured",
                trace=trace,
                resolved_strategy="answer_only",
            )
        if self._mode == "snapshot":
            entry = self._snapshot_entry(q.id)
            paths_raw = entry.get("paths")
            paths = tuple(str(p) for p in paths_raw) if isinstance(paths_raw, list) else ()
            if not paths:
                trace["snapshot_missing"] = q.id
            passages_raw = entry.get("passages")
            # `--context-passages N` keeps only the first N
            # passages in rank order (the snapshot stores `result.passages`
            # order — the same list the agent pre-seed slices). `paths`
            # deliberately stays the FULL frozen card list: retrieval is the
            # frozen axis of this replay, so source metrics must not move
            # with N — only what the model sees does.
            if self.context_passages is not None:
                trace["context_passages"] = self.context_passages
            parts: list[str] = []
            if isinstance(passages_raw, list):
                for p in passages_raw[: self.context_passages]:
                    if not isinstance(p, Mapping) or not p.get("text"):
                        continue
                    header = str(p.get("title") or p.get("breadcrumb") or p.get("path") or "")
                    parts.append(f"=== {header} ===\n{p['text']}")
            trace["passages_used"] = len(parts)
            context = _budget_context(parts)
        else:
            context, paths = await build_oracle_context(self._state, q, self._find_archive)
            trace["context"] = "oracle"
        prompt = (
            f"Use the following source passages to answer the question.\n\n"
            f"SOURCE PASSAGES:\n{context}\n\n"
            f"QUESTION: {q.question}\n\n"
            f"Answer concisely using the source passages. If the passages do "
            f"not contain the answer, reply with exactly this sentence and "
            f"nothing else: {ABSTENTION_NO_MATCH}"
        )
        try:
            res = await gw.chat_once(
                [ChatMessage(role="user", content=prompt)],
                model=self._model_id,
                temperature=0.0,
                max_tokens=self._max_tokens,
                enable_thinking=False,
            )
            answer_text = str(res.text)
        except Exception as exc:
            return QuestionOutput(
                answer_text="",
                retrieved_paths=paths,
                abstained=False,
                error=f"answer_only generation failed: {exc!r}",
                trace=trace,
                resolved_strategy="answer_only",
            )
        abstained = ABSTENTION_NO_MATCH.split(".")[0] in answer_text
        return QuestionOutput(
            answer_text=answer_text,
            retrieved_paths=paths,
            abstained=abstained,
            error=None,
            trace=trace,
            resolved_strategy="answer_only",
            input_tokens=res.input_tokens,
            output_tokens=res.output_tokens,
        )


# ── System registry ────────────────────────────────────────────────────────

#: Names → constructor. The CLI/API look up by string and bind profile/model/scope.
SYSTEM_CLASSES: dict[str, type] = {
    "retrieval_only": RetrievalOnlySystem,
    "sources_only": SourcesOnlySystem,
    "agentic_pydantic": AgenticPydanticSystem,
    "oracle": OracleSystem,
    "closed_book": ClosedBookSystem,
    "answer_only": AnswerOnlySystem,
}


def make_system(
    name: str,
    state: AppState,
    *,
    profile: str | None = None,
    scope: str | None = None,
    model_id: str = "",
    endpoint: str = "",
    api_key: str = "",
    enable_thinking: bool | None = None,
    profile_hash: str = "",
    context_path: str | None = None,
    oracle_context: bool = False,
    collect_context: bool = False,
    context_passages: int | None = None,
) -> Any:
    """Construct a registered SystemUnderTest by name, binding its pins."""
    cls = SYSTEM_CLASSES.get(name)
    if cls is None:
        raise ValueError(f"unknown benchmark system: {name!r}")
    common: dict[str, Any] = {
        "answer_model": model_id,
        "profile_name": profile or "",
        "profile_hash": profile_hash,
    }
    if name == "retrieval_only":
        return cls(state, profile=profile, scope=scope, collect=collect_context, **common)
    if name == "sources_only":
        return cls(state, profile=profile, scope=scope, **common)
    if name == "agentic_pydantic":
        return cls(
            state,
            model_id=model_id,
            endpoint=endpoint,
            api_key=api_key,
            enable_thinking=enable_thinking,
            profile=profile,
            scope=scope,
            **common,
        )
    if name == "answer_only":
        return cls(
            state,
            model_id=model_id,
            endpoint=endpoint,
            api_key=api_key,
            context_path=context_path,
            oracle_context=oracle_context,
            context_passages=context_passages,
            **common,
        )
    # oracle, closed_book
    return cls(state, model_id=model_id, endpoint=endpoint, api_key=api_key, **common)


# ── SqliteBenchStore (concrete BenchStore over the new tables) ──────────────


def _outcome_to_payload(outcome: JudgeOutcome) -> str:
    """Serialize a JudgeOutcome to the cache payload JSON."""
    return json.dumps(
        {
            "verdict": outcome.verdict.value,
            "reason": outcome.reason,
            "sub_facts_present": list(outcome.sub_facts_present),
            "abstained": outcome.abstained,
            "judge_model": outcome.judge_model,
        }
    )


def _payload_to_outcome(payload: str) -> JudgeOutcome | None:
    """Deserialize a cache payload JSON back to a JudgeOutcome."""
    parsed = _parse_judge_json(payload, "")
    return parsed


class SqliteBenchStore:
    """Concrete :class:`BenchStore` over ``bench_runs`` / ``bench_question_results``
    / ``bench_judge_cache`` (migration 0009).

    Lives in ``api/`` (not ``eval/``) because it imports aiosqlite — keeping the
    DB dep out of the eval package preserves the ≤2 dependency cap.
    """

    def __init__(self, db: Any) -> None:
        self._db = db

    # ── runs ──────────────────────────────────────────────────────────────

    async def insert_run(self, record: BenchRunRecord) -> int:
        async with self._db.write() as conn:
            cur = await conn.execute(
                "INSERT INTO bench_runs(run_group, label, started_at, finished_at, "
                "status, dataset_name, dataset_hash, subset_hash, system, "
                "profile_name, profile_hash, answer_model, judge_model, scope, "
                "trusted, calibration, config_json, metrics_json) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                _run_to_row(record),
            )
            return int(cur.lastrowid) if cur.lastrowid is not None else 0

    async def update_run(self, run_id: int, record: BenchRunRecord) -> bool:
        row = _run_to_row(record)
        # row[4] is started_at; we keep started_at stable, update the rest.
        async with self._db.write() as conn:
            cur = await conn.execute(
                "UPDATE bench_runs SET run_group=?, label=?, started_at=?, finished_at=?, "
                "status=?, dataset_name=?, dataset_hash=?, subset_hash=?, system=?, "
                "profile_name=?, profile_hash=?, answer_model=?, judge_model=?, scope=?, "
                "trusted=?, calibration=?, config_json=?, metrics_json=? WHERE id=?",
                (*row, run_id),
            )
            return bool(cur.rowcount > 0)

    async def get_run(self, run_id: int) -> BenchRunRecord | None:
        async with self._db.read() as conn:
            cur = await conn.execute("SELECT * FROM bench_runs WHERE id=?", (run_id,))
            row = await cur.fetchone()
        return _row_to_run(row) if row is not None else None

    async def list_runs(self, limit: int = 50) -> list[BenchRunRecord]:
        async with self._db.read() as conn:
            cur = await conn.execute("SELECT * FROM bench_runs ORDER BY id DESC LIMIT ?", (limit,))
            rows = await cur.fetchall()
        return [_row_to_run(r) for r in rows]

    async def delete_run(self, run_id: int) -> bool:
        async with self._db.write() as conn:
            cur = await conn.execute("DELETE FROM bench_runs WHERE id=?", (run_id,))
            return bool(cur.rowcount > 0)

    async def mark_aborted(self, run_id: int, reason: str) -> bool:
        """Mark a ``running`` run ``aborted`` (only touches still-running rows)."""
        now = now_iso()
        async with self._db.write() as conn:
            cur = await conn.execute(
                "UPDATE bench_runs SET status='aborted', finished_at=? "
                "WHERE id=? AND status='running'",
                (now, run_id),
            )
            updated = bool(cur.rowcount > 0)
        if updated:
            # Record the abort reason in config_json.
            with suppress(Exception):
                rec = await self.get_run(run_id)
                if rec is not None:
                    cfg = dict(rec.config_json)
                    cfg["abort_reason"] = reason
                    await self.update_run(
                        run_id, replace(rec, config_json=cfg, abort_reason=reason)
                    )
        return updated

    # ── question results ─────────────────────────────────────────────────

    async def insert_question_result(self, run_id: int, row: BenchQuestionResult) -> None:
        async with self._db.write() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO bench_question_results("
                "run_id, question_id, capability, difficulty, question_text, "
                "expected_answer, answer_text, abstained, verdict, verdict_reason, "
                "source_hit_rank, source_coverage, sub_fact_coverage, "
                "retrieved_paths, rounds, latency_ms, error, trace_json, "
                "input_tokens, output_tokens) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                _result_to_row(run_id, row),
            )

    async def update_question_result(
        self, run_id: int, question_id: str, row: BenchQuestionResult
    ) -> bool:
        async with self._db.write() as conn:
            cur = await conn.execute(
                "UPDATE bench_question_results SET capability=?, difficulty=?, "
                "question_text=?, expected_answer=?, answer_text=?, abstained=?, "
                "verdict=?, verdict_reason=?, source_hit_rank=?, source_coverage=?, "
                "sub_fact_coverage=?, retrieved_paths=?, rounds=?, latency_ms=?, "
                "error=?, trace_json=?, input_tokens=?, output_tokens=? "
                "WHERE run_id=? AND question_id=?",
                (
                    row.capability,
                    row.difficulty,
                    row.question_text,
                    row.expected_answer,
                    row.answer_text,
                    1 if row.abstained else 0,
                    row.verdict,
                    row.verdict_reason,
                    row.source_hit_rank,
                    row.source_coverage,
                    row.sub_fact_coverage,
                    json.dumps(list(row.retrieved_paths)),
                    row.rounds,
                    row.latency_ms,
                    row.error,
                    json.dumps(row.trace) if row.trace else None,
                    row.input_tokens,
                    row.output_tokens,
                    run_id,
                    question_id,
                ),
            )
            return bool(cur.rowcount > 0)

    async def update_verdict(
        self,
        run_id: int,
        question_id: str,
        verdict: str,
        reason: str,
        sub_fact_coverage: float | None,
    ) -> bool:
        async with self._db.write() as conn:
            cur = await conn.execute(
                "UPDATE bench_question_results SET verdict=?, verdict_reason=?, "
                "sub_fact_coverage=? WHERE run_id=? AND question_id=?",
                (verdict, reason, sub_fact_coverage, run_id, question_id),
            )
            return bool(cur.rowcount > 0)

    async def list_question_results(self, run_id: int) -> list[BenchQuestionResult]:
        async with self._db.read() as conn:
            cur = await conn.execute(
                "SELECT * FROM bench_question_results WHERE run_id=? ORDER BY question_id",
                (run_id,),
            )
            rows = await cur.fetchall()
        return [_row_to_result(r) for r in rows]

    async def list_pending_results(self, run_id: int) -> list[BenchQuestionResult]:
        async with self._db.read() as conn:
            cur = await conn.execute(
                "SELECT * FROM bench_question_results WHERE run_id=? AND verdict='pending' "
                "ORDER BY question_id",
                (run_id,),
            )
            rows = await cur.fetchall()
        return [_row_to_result(r) for r in rows]

    # ── judge cache ──────────────────────────────────────────────────────

    async def judge_cache_get(self, key: str) -> JudgeOutcome | None:
        async with self._db.read() as conn:
            cur = await conn.execute("SELECT payload FROM bench_judge_cache WHERE key=?", (key,))
            row = await cur.fetchone()
        if row is None:
            return None
        return _payload_to_outcome(str(row[0]))

    async def judge_cache_put(self, key: str, outcome: JudgeOutcome) -> None:
        payload = _outcome_to_payload(outcome)
        now = now_iso()
        async with self._db.write() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO bench_judge_cache(key, verdict, reason, "
                "payload, created_at) VALUES(?,?,?,?,?)",
                (key, outcome.verdict.value, outcome.reason, payload, now),
            )

    # ── maintenance ──────────────────────────────────────────────────────

    async def prune_traces(self, older_than_days: int) -> int:
        """Drop trace_json from runs older than N days (trap 10)."""
        if older_than_days <= 0:
            return 0
        cutoff = (_dt.datetime.now(_dt.UTC) - _dt.timedelta(days=older_than_days)).isoformat()
        async with self._db.write() as conn:
            cur = await conn.execute(
                "UPDATE bench_question_results SET trace_json=NULL "
                "WHERE trace_json IS NOT NULL AND run_id IN "
                "(SELECT id FROM bench_runs WHERE started_at < ?)",
                (cutoff,),
            )
            return int(cur.rowcount or 0)

    async def reconcile_stale(self) -> int:
        """Mark orphaned ``running`` rows as ``aborted`` at startup."""
        async with self._db.read() as conn:
            cur = await conn.execute("SELECT id FROM bench_runs WHERE status='running'")
            stale = [int(r[0]) for r in await cur.fetchall()]
        for run_id in stale:
            await self.mark_aborted(run_id, "process restarted mid-run; marked aborted at startup")
        return len(stale)


# ── Row serialization helpers ──────────────────────────────────────────────


def _run_to_row(record: BenchRunRecord) -> tuple[Any, ...]:
    return (
        record.run_group,
        record.label,
        record.started_at,
        record.finished_at,
        record.status,
        record.dataset_name,
        record.dataset_hash,
        record.subset_hash,
        record.system,
        record.profile_name,
        record.profile_hash,
        record.answer_model,
        record.judge_model,
        record.scope,
        1 if record.trusted else 0,
        record.calibration,
        json.dumps(record.config_json),
        json.dumps(record.metrics_json),
    )


def _row_to_run(row: aiosqlite.Row) -> BenchRunRecord:
    cal = row["calibration"]
    config = json.loads(row["config_json"]) if row["config_json"] else {}
    return BenchRunRecord(
        id=int(row["id"]),
        run_group=str(row["run_group"]),
        label=str(row["label"]),
        started_at=str(row["started_at"]),
        finished_at=str(row["finished_at"]) if row["finished_at"] else None,
        status=str(row["status"]),
        dataset_name=str(row["dataset_name"]),
        dataset_hash=str(row["dataset_hash"]),
        subset_hash=str(row["subset_hash"]),
        system=str(row["system"]),
        profile_name=str(row["profile_name"]),
        profile_hash=str(row["profile_hash"]),
        answer_model=str(row["answer_model"]),
        judge_model=str(row["judge_model"]),
        scope=str(row["scope"]),
        trusted=bool(row["trusted"]),
        calibration=float(cal) if cal is not None else None,
        config_json=config,
        metrics_json=json.loads(row["metrics_json"]) if row["metrics_json"] else {},
        # No dedicated columns: both live in config_json (the failed-cell path
        # and mark_aborted stash the reason there; _run_cell pins the flag).
        judge_shares_endpoint=bool(config.get("judge_shares_endpoint", False)),
        abort_reason=str(config.get("abort_reason", "")),
    )


def _result_to_row(run_id: int, row: BenchQuestionResult) -> tuple[Any, ...]:
    return (
        run_id,
        row.question_id,
        row.capability,
        row.difficulty,
        row.question_text,
        row.expected_answer,
        row.answer_text,
        1 if row.abstained else 0,
        row.verdict,
        row.verdict_reason,
        row.source_hit_rank,
        row.source_coverage,
        row.sub_fact_coverage,
        json.dumps(list(row.retrieved_paths)),
        row.rounds,
        row.latency_ms,
        row.error,
        json.dumps(row.trace) if row.trace else None,
        row.input_tokens,
        row.output_tokens,
    )


def _row_to_result(row: aiosqlite.Row) -> BenchQuestionResult:
    raw_paths = row["retrieved_paths"]
    try:
        paths: tuple[str, ...] = tuple(str(p) for p in json.loads(raw_paths)) if raw_paths else ()
    except (json.JSONDecodeError, TypeError):
        paths = ()
    raw_trace = row["trace_json"]
    trace: dict[str, object] | None = None
    if raw_trace:
        try:
            parsed = json.loads(raw_trace)
            if isinstance(parsed, dict):
                trace = parsed
        except (json.JSONDecodeError, TypeError):
            pass
    sfc = row["sub_fact_coverage"]
    return BenchQuestionResult(
        run_id=int(row["run_id"]),
        question_id=str(row["question_id"]),
        capability=str(row["capability"]),
        difficulty=str(row["difficulty"]),
        question_text=str(row["question_text"]),
        expected_answer=str(row["expected_answer"]),
        answer_text=str(row["answer_text"]),
        abstained=bool(row["abstained"]),
        verdict=str(row["verdict"]),
        verdict_reason=str(row["verdict_reason"]),
        source_hit_rank=int(row["source_hit_rank"]) if row["source_hit_rank"] is not None else None,
        source_coverage=float(row["source_coverage"]),
        sub_fact_coverage=float(sfc) if sfc is not None else None,
        retrieved_paths=paths,
        rounds=int(row["rounds"]),
        latency_ms=float(row["latency_ms"]),
        error=str(row["error"]) if row["error"] is not None else None,
        trace=trace,
        input_tokens=int(row["input_tokens"] or 0),
        output_tokens=int(row["output_tokens"] or 0),
    )


# ── reconcile_stale_bench_runs (called from main.py lifespan) ──────────────


async def reconcile_stale_bench_runs(db: Any) -> int:
    """Mark benchmark runs orphaned by a process death as ``aborted`` (startup).

    A benchmark run is job-shaped: the row is inserted with ``status == 'running'``
    and the orchestrator owns the rest of the lifecycle in this process's memory.
    After a restart every ``running`` row is by definition orphaned. Called once
    at startup alongside ``reconcile_stale_runs``.

    Returns how many runs were marked aborted.
    """
    store = SqliteBenchStore(db)
    return await store.reconcile_stale()


async def prune_stale_bench_traces(db: Any, older_than_days: int) -> int:
    """Drop per-question trace JSON from runs past retention (called from the
    ``main.py`` lifespan, alongside :func:`reconcile_stale_bench_runs`).

    Traces are a separate prunable column: verdicts + retrieval + answer text
    stay forever, so old runs remain comparable — only the bulky raw evidence
    is bounded by ``bench.trace_retention_days`` (resolved by the caller).
    Returns how many rows were pruned.
    """
    store = SqliteBenchStore(db)
    return await store.prune_traces(older_than_days)


# ── answer_runs import (one-shot) ───────────────────────────────────────────


async def import_answer_runs(db: Any) -> int:
    """Import ``answer_runs`` rows into ``bench_runs`` + ``bench_question_results``
    where the mapping is unambiguous. The old table is left read-only.

    Returns the number of runs imported.
    """
    store = SqliteBenchStore(db)
    imported = 0
    async with db.read() as conn:
        cur = await conn.execute(
            "SELECT label FROM bench_runs WHERE run_group = 'imported_answer_runs'"
        )
        existing_labels = {str(r[0]) for r in await cur.fetchall() if r[0] is not None}
        cur = await conn.execute("SELECT * FROM answer_runs ORDER BY id")
        rows = await cur.fetchall()
    for row in rows:
        config = json.loads(row["config_json"]) if row["config_json"] else {}
        results = json.loads(row["results_json"]) if row["results_json"] else {}
        pq = results.get("per_question", []) if isinstance(results, dict) else []
        if not pq:
            continue
        strategies = results.get("strategies", []) if isinstance(results, dict) else []
        # Map the first strategy's per-question results into bench_question_results.
        for strat in strategies if isinstance(strategies, list) else []:
            strat_name = str(strat.get("name", "")) if isinstance(strat, dict) else ""
            if not strat_name:
                continue
            label = f"imported:{row['id']}:{strat_name}"
            if label in existing_labels:
                break
            record = BenchRunRecord(
                run_group="imported_answer_runs",
                label=label,
                started_at=str(row["started_at"]),
                finished_at=str(row["finished_at"]) if row["finished_at"] else None,
                status="complete",
                dataset_name=str(results.get("dataset_name", "gap_questions")),
                dataset_hash=str(row["dataset_hash"]),
                subset_hash="",
                system=strat_name,
                profile_name=str(config.get("profile", "")),
                profile_hash="",
                answer_model=str(config.get("model", "")),
                judge_model=str(row["judge_model"]),
                scope=str(config.get("scope", "")),
                trusted=bool(config.get("trusted", False)),
                config_json=dict(config),
                metrics_json=dict(strat) if isinstance(strat, dict) else {},
            )
            run_id = await store.insert_run(record)
            for entry in pq:
                if not isinstance(entry, dict):
                    continue
                qid = str(entry.get("id") or entry.get("question_id") or "")
                if not qid:
                    continue
                result_row = BenchQuestionResult(
                    run_id=run_id,
                    question_id=qid,
                    capability=str(entry.get("capability", "")),
                    difficulty=str(entry.get("difficulty", "")),
                    question_text=str(entry.get("question", "")),
                    expected_answer=str(entry.get("answer", "")),
                    answer_text=str(entry.get("model_answer", "")),
                    abstained=bool(entry.get("abstained", False)),
                    verdict=str(entry.get("verdict", Verdict.UNJUDGED.value)),
                    verdict_reason=str(entry.get("verdict_reason", "")),
                    source_hit_rank=entry.get("retrieval_hit_rank")
                    if isinstance(entry.get("retrieval_hit_rank"), int)
                    else None,
                    source_coverage=float(entry.get("source_coverage", 0.0)),
                    retrieved_paths=tuple(entry.get("retrieved_paths", []))
                    if isinstance(entry.get("retrieved_paths"), list)
                    else (),
                    rounds=int(entry.get("rounds", 0)),
                    latency_ms=float(entry.get("latency_ms", 0.0)),
                    error=str(entry["error"]) if entry.get("error") else None,
                )
                await store.insert_question_result(run_id, result_row)
            existing_labels.add(label)
            imported += 1
            break  # one bench_runs row per answer_runs row (first strategy)
    return imported


class InMemoryBenchStore:
    """In-memory :class:`BenchStore` for ``vesta bench run --no-persist``.

    Every write is held in plain dicts and discarded at exit, so a report-only
    smoke run never touches ``bench_runs`` / ``bench_question_results`` /
    ``bench_judge_cache``. The run flow needs the standard methods (insert →
    update → question results → judge cache); the read/comparison methods are
    implemented with the same semantics so the object satisfies the Protocol.
    """

    def __init__(self) -> None:
        self._runs: dict[int, BenchRunRecord] = {}
        self._results: dict[int, dict[str, BenchQuestionResult]] = {}
        self._cache: dict[str, JudgeOutcome] = {}
        self._next_id = 1

    # ── runs ──────────────────────────────────────────────────────────────

    async def insert_run(self, record: BenchRunRecord) -> int:
        rid = self._next_id
        self._next_id += 1
        self._runs[rid] = replace(record, id=rid)
        return rid

    async def update_run(self, run_id: int, record: BenchRunRecord) -> bool:
        if run_id not in self._runs:
            return False
        self._runs[run_id] = replace(record, id=run_id)
        return True

    async def get_run(self, run_id: int) -> BenchRunRecord | None:
        return self._runs.get(run_id)

    async def list_runs(self, limit: int = 50) -> list[BenchRunRecord]:
        return sorted(self._runs.values(), key=lambda r: r.id, reverse=True)[:limit]

    async def delete_run(self, run_id: int) -> bool:
        return self._runs.pop(run_id, None) is not None

    async def mark_aborted(self, run_id: int, reason: str) -> bool:
        rec = self._runs.get(run_id)
        if rec is None or rec.status != "running":
            return False
        self._runs[run_id] = replace(rec, status="aborted", abort_reason=reason)
        return True

    # ── question results ──────────────────────────────────────────────────

    async def insert_question_result(self, run_id: int, row: BenchQuestionResult) -> None:
        self._results.setdefault(run_id, {})[row.question_id] = row

    async def update_question_result(
        self, run_id: int, question_id: str, row: BenchQuestionResult
    ) -> bool:
        bucket = self._results.get(run_id)
        if bucket is None:
            return False
        bucket[question_id] = row
        return True

    async def update_verdict(
        self,
        run_id: int,
        question_id: str,
        verdict: str,
        reason: str,
        sub_fact_coverage: float | None,
    ) -> bool:
        bucket = self._results.get(run_id)
        row = bucket.get(question_id) if bucket is not None else None
        if bucket is None or row is None:
            return False
        bucket[question_id] = replace(
            row, verdict=verdict, verdict_reason=reason, sub_fact_coverage=sub_fact_coverage
        )
        return True

    async def list_question_results(self, run_id: int) -> list[BenchQuestionResult]:
        return list(self._results.get(run_id, {}).values())

    async def list_pending_results(self, run_id: int) -> list[BenchQuestionResult]:
        return [row for row in self._results.get(run_id, {}).values() if row.verdict == "pending"]

    # ── judge cache ───────────────────────────────────────────────────────

    async def judge_cache_get(self, key: str) -> JudgeOutcome | None:
        return self._cache.get(key)

    async def judge_cache_put(self, key: str, outcome: JudgeOutcome) -> None:
        self._cache[key] = outcome

    # ── maintenance (no-ops: nothing is persisted) ────────────────────────

    async def prune_traces(self, older_than_days: int) -> int:
        return 0

    async def reconcile_stale(self) -> int:
        return 0


# ── HTTP API — /api/bench/* ────────────────────────────────────────────────

router = APIRouter(prefix="/api/bench", tags=["bench"])

# In-flight run groups (job-shaped, mirrors the old api/benchmark.py). Single
# user, one worker: the background task owns the completed lifecycle, the rows
# it drives were pre-inserted by POST /run so the response can return run ids
# immediately. Keyed by run_group (a request creates one group per matrix).
_tasks: dict[str, asyncio.Task[None]] = {}
_progress: dict[str, dict[int, dict[str, object]]] = {}

#: Serializes group execution across requests: ``_drive_answer_events``
#: temporarily swaps the process-global ``state.gateway`` for a per-question
#: ``UsageRecorder``, so two groups running concurrently would nest recorders
#: and attribute tokens to the wrong run. One worker, one event loop: a module
#: lock matches the ``_tasks`` registry; POST /run refuses with 409 while it
#: is held, and this gate is the backstop that keeps overlap unreachable.
_RUN_GATE = asyncio.Lock()


def _metric(metrics: dict[str, object], path: str) -> object:
    """Look a metric up in ``metrics_json`` by dotted path (``answer.strict_accuracy``)."""
    return metric_lookup(metrics, path)


class BenchRunRequest(BaseModel):
    """``POST /api/bench/run`` body — the matrix axes + dataset filter."""

    systems: list[str] = []
    profiles: list[str] = []
    models: list[str] = []
    dataset: str | None = None
    slice: str | None = None
    capabilities: list[str] = []
    level: int | None = None
    limit: int | None = None
    scope: str | None = None
    judge_model: str | None = None
    repeats: int | None = None
    label: str | None = None


class BenchRunResponse(BaseModel):
    """The created run group + its run ids (job-shaped: returned immediately)."""

    run_group: str
    run_ids: list[int]
    systems: list[str]
    profiles: list[str]
    models: list[str]
    repeats: int
    matrix_size: int
    dataset_name: str
    dataset_hash: str
    subset_hash: str
    judge_model: str
    status: str = "running"


class BenchRunSummary(BaseModel):
    """One row in ``GET /runs`` — the pins + headline score chips."""

    id: int
    run_group: str
    label: str
    started_at: str
    finished_at: str | None = None
    status: str
    dataset_name: str
    dataset_hash: str
    subset_hash: str
    system: str
    profile_name: str
    answer_model: str
    judge_model: str
    scope: str
    trusted: bool
    headroom: object | None = None
    strict_accuracy: object | None = None
    source_recall_at_10: object | None = None
    hallucination_rate: object | None = None
    unjudged: object | None = None
    complete: object | None = None


class BenchRunDetail(BaseModel):
    """Full detail for ``GET /runs/{id}`` — aggregates + capability breakdown +
    attribution matrix (all read from ``metrics_json``)."""

    id: int
    run_group: str
    label: str
    started_at: str
    finished_at: str | None = None
    status: str
    dataset_name: str
    dataset_hash: str
    subset_hash: str
    system: str
    profile_name: str
    profile_hash: str
    answer_model: str
    judge_model: str
    scope: str
    trusted: bool
    calibration: float | None = None
    judge_shares_endpoint: bool = False
    abort_reason: str = ""
    config_json: dict[str, object] = {}
    metrics: dict[str, object] = {}
    progress: dict[str, object] | None = None


class BenchResultRow(BaseModel):
    """One per-question row in ``GET /runs/{id}/results``. NEVER carries
    ``trace_json`` (trap 10 — the column is not selected)."""

    run_id: int
    question_id: str
    capability: str
    difficulty: str
    question_text: str
    expected_answer: str
    answer_text: str
    abstained: bool
    verdict: str
    verdict_reason: str = ""
    source_hit_rank: int | None = None
    source_coverage: float = 0.0
    sub_fact_coverage: float | None = None
    retrieved_paths: list[str] = []
    rounds: int = 0
    latency_ms: float = 0.0
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class BenchResultsPage(BaseModel):
    """Paginated per-question rows for a run."""

    items: list[BenchResultRow]
    total: int
    offset: int
    limit: int


class ComparePair(BaseModel):
    """One pairwise per-question diff in ``GET /compare``."""

    run_a: int
    run_b: int
    shared_denominator: int
    fixed: list[str]
    broken: list[str]
    both_correct: list[str]
    both_wrong: list[str]
    only_a: list[str]
    only_b: list[str]
    unjudged: list[str] = []
    deltas: dict[str, float] = {}


class BenchCompareResponse(BaseModel):
    """Per-question diff across the requested runs (pairwise)."""

    runs: list[int]
    pairs: list[ComparePair]


class BenchDatasetInfo(BaseModel):
    """``GET /api/bench/dataset`` — metadata + composition + reference scores."""

    name: str
    version: int
    hash: str
    generated: str
    total: int
    by_capability: dict[str, int]
    by_difficulty: dict[str, int]
    by_slice: dict[str, int]
    ceiling: dict[str, object]
    floor: dict[str, object]


# ── Matrix resolution helpers ───────────────────────────────────────────────


def _resolve_questions(body: BenchRunRequest, dataset: Any) -> list[Any]:
    """Apply slice / level / capabilities / limit filters from the request body."""
    return apply_flag_filters(
        dataset.questions,
        slice=body.slice,
        level=body.level,
        capabilities=body.capabilities,
        limit=body.limit,
    )


def _resolve_matrix(body: BenchRunRequest) -> tuple[list[str], list[str], list[str]]:
    """Resolve the systems / profiles / models matrix axes (empty → defaults)."""

    systems, profiles, models = resolve_matrix_axes(
        body.systems,
        body.profiles,
        body.models,
        default_systems=str(app_config.get(BENCH_SYSTEMS)),
        default_model=str(app_config.get(INFERENCE_LLM_MODEL)),
    )
    if not models:
        # The model default is inert on a fresh install ("") — say so instead
        # of silently running the matrix against an
        # empty model id.
        raise HTTPException(
            status_code=400,
            detail="no model configured; pass models=[...] or set inference.llm.model",
        )
    return systems, profiles, models


def _profile_hash(name: str) -> str:
    """Resolve a profile name to its content hash (empty for the unset axis).

    An explicit unknown profile is an error (404) — never silently pinned to
    the lexical hash.
    """
    if not name:
        return ""
    p = resolve_profile_from_settings(name, fallback_to_default=False)
    if p is None:
        raise HTTPException(status_code=404, detail=f"profile {name!r} not found")
    return str(getattr(p, "hash", "") or "")


def _placeholder_record(
    *,
    run_group: str,
    label: str,
    started_at: str,
    dataset: Any,
    subset_hash_val: str,
    sys_name: str,
    profile_name: str,
    profile_hash: str,
    answer_model: str,
    judge_model: str,
    scope: str,
) -> BenchRunRecord:
    """A ``running`` placeholder row for one matrix cell (reserved by POST /run)."""
    return BenchRunRecord(
        run_group=run_group,
        label=label,
        started_at=started_at,
        status="running",
        dataset_name=str(getattr(dataset, "name", "")),
        dataset_hash=str(getattr(dataset, "hash", "")),
        subset_hash=subset_hash_val,
        system=sys_name,
        profile_name=profile_name,
        profile_hash=profile_hash,
        answer_model=answer_model,
        judge_model=judge_model,
        scope=scope,
    )


class _PreSeededStore:
    """Wrap a :class:`BenchStore` so ``insert_run`` returns pre-seeded run ids.

    ``POST /run`` pre-inserts one placeholder ``running`` row per matrix cell so
    the response can return run ids immediately (job-shaped). The background task
    runs the real runner over this wrapper whose ``insert_run`` reuses those ids
    instead of creating fresh rows; ``_run_cell`` then overwrites each placeholder
    via ``update_run``. Any method the runner doesn't use just delegates.
    """

    def __init__(self, inner: BenchStore, ids: Sequence[int]) -> None:
        self._inner: BenchStore = inner
        self._ids = list(ids)
        self._pos = 0

    async def insert_run(self, record: BenchRunRecord) -> int:
        if self._pos < len(self._ids):
            rid = self._ids[self._pos]
            self._pos += 1
            return rid
        return await self._inner.insert_run(record)

    async def update_run(self, run_id: int, record: BenchRunRecord) -> bool:
        return await self._inner.update_run(run_id, record)

    async def get_run(self, run_id: int) -> BenchRunRecord | None:
        return await self._inner.get_run(run_id)

    async def list_runs(self, limit: int = 50) -> list[BenchRunRecord]:
        return await self._inner.list_runs(limit)

    async def delete_run(self, run_id: int) -> bool:
        return await self._inner.delete_run(run_id)

    async def mark_aborted(self, run_id: int, reason: str) -> bool:
        return await self._inner.mark_aborted(run_id, reason)

    async def insert_question_result(self, run_id: int, row: BenchQuestionResult) -> None:
        await self._inner.insert_question_result(run_id, row)

    async def update_question_result(
        self, run_id: int, question_id: str, row: BenchQuestionResult
    ) -> bool:
        return await self._inner.update_question_result(run_id, question_id, row)

    async def update_verdict(
        self,
        run_id: int,
        question_id: str,
        verdict: str,
        reason: str,
        sub_fact_coverage: float | None,
    ) -> bool:
        return await self._inner.update_verdict(
            run_id, question_id, verdict, reason, sub_fact_coverage
        )

    async def list_question_results(self, run_id: int) -> list[BenchQuestionResult]:
        return await self._inner.list_question_results(run_id)

    async def list_pending_results(self, run_id: int) -> list[BenchQuestionResult]:
        return await self._inner.list_pending_results(run_id)

    async def judge_cache_get(self, key: str) -> JudgeOutcome | None:
        return await self._inner.judge_cache_get(key)

    async def judge_cache_put(self, key: str, outcome: JudgeOutcome) -> None:
        await self._inner.judge_cache_put(key, outcome)

    async def prune_traces(self, older_than_days: int) -> int:
        return await self._inner.prune_traces(older_than_days)

    async def reconcile_stale(self) -> int:
        return await self._inner.reconcile_stale()


# ── Background orchestration (mirrors the old api/benchmark.py pattern) ─────


async def _run_to_completion(
    *,
    state: AppState,
    store: SqliteBenchStore,
    body: BenchRunRequest,
    dataset: Any,
    questions: Sequence[Any],
    sut_list: list[Any],
    judge_model: str,
    run_group: str,
    run_ids: list[int],
    repeats: int,
    judge_concurrency: int,
    judge_shares_endpoint: bool,
) -> None:
    """Run the real matrix over the pre-seeded placeholder rows and sweep them."""
    async with _RUN_GATE:
        judge_gateway: Any | None = None
        try:
            judge, judge_gateway = make_judge_llm(state, judge_model)
            store_wrapped = _PreSeededStore(store, run_ids)

            def _progress_cb(update: Any) -> None:
                prog = _progress.get(run_group)
                if prog is not None:
                    prog[update.run_id] = {
                        "system": update.system,
                        "stage": update.stage,
                        "done": update.done,
                        "total": update.total,
                    }

            await run_benchmark(
                dataset=dataset,
                questions=questions,
                systems=sut_list,
                store=store_wrapped,
                judge=judge,
                judge_model=judge_model,
                run_group=run_group,
                label=body.label or "",
                scope=body.scope or "",
                # Persisted pins get served back by run-detail endpoints:
                # never embed credential material in the snapshot.
                config_snapshot=app_config.strip_secret_values(app_config.snapshot().values),
                judge_concurrency=judge_concurrency,
                judge_shares_endpoint=judge_shares_endpoint,
                repeats=repeats,
                max_concurrent=int(app_config.get_or_default(BENCH_MAX_CONCURRENT)),
                progress=_progress_cb,
                level=body.level,
            )
        except Exception as exc:
            # Orchestration-level failure (e.g. judge construction): mark every
            # cell still ``running`` aborted. Completed cells are untouched.
            reason = f"{type(exc).__name__}: {exc}"
            for rid in run_ids:
                with suppress(Exception):
                    await store.mark_aborted(rid, reason)
        finally:
            if judge_gateway is not None:
                with suppress(Exception):
                    await judge_gateway.aclose()
            _tasks.pop(run_group, None)
            _progress.pop(run_group, None)


# ── Routes ──────────────────────────────────────────────────────────────────


@router.post("/run", response_model=BenchRunResponse)
async def run_bench_endpoint(request: Request, body: BenchRunRequest) -> BenchRunResponse:
    """Start a run group; returns ``run_group`` + run ids immediately (job-shaped)."""
    state: AppState = app_state(request)
    store = SqliteBenchStore(state.db)
    if _RUN_GATE.locked():
        raise HTTPException(
            status_code=409,
            detail="another bench run group is currently executing; wait for it to finish",
        )

    dataset_path = body.dataset or str(app_config.get(BENCH_DATASET))
    try:
        dataset = load_bench_dataset(dataset_path)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"could not load dataset {dataset_path!r}: {exc!r}"
        ) from exc

    qs = _resolve_questions(body, dataset)
    if not qs:
        raise HTTPException(
            status_code=400,
            detail="no questions selected (check --slice/--capabilities/--limit)",
        )

    systems, profiles, models = _resolve_matrix(body)
    for sys_name in systems:
        if sys_name not in SYSTEM_CLASSES:
            raise HTTPException(
                status_code=400,
                detail=f"unknown system {sys_name!r}; choose from {sorted(SYSTEM_CLASSES)}",
            )

    judge_model = body.judge_model or str(app_config.get(EVAL_JUDGE_MODEL))
    answer_endpoint = str(app_config.get(INFERENCE_LLM_ENDPOINT_URL))
    judge_endpoint = str(app_config.get(EVAL_JUDGE_ENDPOINT_URL))
    judge_concurrency, shares = resolve_judge_concurrency(
        int(app_config.get_or_default(BENCH_JUDGE_CONCURRENCY)),
        answer_endpoint=answer_endpoint,
        judge_endpoint=judge_endpoint,
    )

    # Flat system list = systems x profiles x models.
    sut_list: list[Any] = []
    for sys_name in systems:
        for prof in profiles:
            phash = _profile_hash(prof)
            for model in models:
                sut_list.append(
                    make_system(
                        sys_name,
                        state,
                        profile=prof or None,
                        scope=body.scope or None,
                        model_id=model,
                        endpoint=answer_endpoint,
                        api_key=str(app_config.get(INFERENCE_LLM_API_KEY)),
                        profile_hash=phash,
                    )
                )

    repeats = body.repeats or int(app_config.get_or_default(BENCH_REPEATS))
    run_group = str(uuid.uuid4())
    subset_val = subset_hash(list(qs))

    # Pre-seed one placeholder running row per cell so the response returns run
    # ids immediately and the run list shows the group the instant POST returns.
    run_ids: list[int] = []
    for repeat in range(repeats):
        for sut in sut_list:
            label = f"{body.label or ''}{' r' + str(repeat + 1) if repeats > 1 else ''}".strip()
            placeholder = _placeholder_record(
                run_group=run_group,
                label=label,
                started_at=now_iso(),
                dataset=dataset,
                subset_hash_val=subset_val,
                sys_name=sut.name,
                profile_name=getattr(sut, "profile_name", "") or "",
                profile_hash=getattr(sut, "profile_hash", "") or "",
                answer_model=getattr(sut, "answer_model", "") or "",
                judge_model=judge_model,
                scope=body.scope or "",
            )
            run_ids.append(await store.insert_run(placeholder))

    task = asyncio.create_task(
        _run_to_completion(
            state=state,
            store=store,
            body=body,
            dataset=dataset,
            questions=qs,
            sut_list=sut_list,
            judge_model=judge_model,
            run_group=run_group,
            run_ids=run_ids,
            repeats=repeats,
            judge_concurrency=judge_concurrency,
            judge_shares_endpoint=shares,
        ),
        name=f"bench-{run_group}",
    )
    _tasks[run_group] = task
    _progress[run_group] = {
        rid: {"system": "", "stage": "pending", "done": 0, "total": len(qs)} for rid in run_ids
    }
    return BenchRunResponse(
        run_group=run_group,
        run_ids=run_ids,
        systems=systems,
        profiles=profiles,
        models=models,
        repeats=repeats,
        matrix_size=len(run_ids),
        dataset_name=getattr(dataset, "name", ""),
        dataset_hash=getattr(dataset, "hash", ""),
        subset_hash=subset_val,
        judge_model=judge_model,
        status="running",
    )


@router.get("/runs", response_model=list[BenchRunSummary])
async def list_bench_runs(
    request: Request,
    group: str | None = Query(None, description="filter by run_group"),
    system: str | None = Query(None, description="filter by system name"),
    dataset: str | None = Query(None, description="filter by dataset hash"),
    limit: int = Query(100, ge=1, le=500),
) -> list[BenchRunSummary]:
    """List recent bench runs, newest first, filterable by group/system/dataset."""
    state: AppState = app_state(request)
    store = SqliteBenchStore(state.db)
    runs = await store.list_runs(limit)
    if group:
        runs = [r for r in runs if r.run_group == group]
    if system:
        runs = [r for r in runs if r.system == system]
    if dataset:
        runs = [r for r in runs if r.dataset_hash == dataset]
    return [_to_summary(r) for r in runs]


@router.get("/runs/{run_id}", response_model=BenchRunDetail)
async def get_bench_run(request: Request, run_id: int) -> BenchRunDetail:
    """One run's detail: aggregates + capability breakdown + attribution matrix."""
    state: AppState = app_state(request)
    store = SqliteBenchStore(state.db)
    record = await store.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    detail = _to_detail(record)
    task = _tasks.get(record.run_group)
    if task is not None and not task.done():
        detail.status = "running"
        detail.progress = _progress.get(record.run_group, {}).get(run_id)
    return detail


#: attribution-cell → WHERE clause for the per-question results filter.
_ATTRIBUTION_FILTERS: dict[str, str] = {
    "correct_source_found": "(verdict='correct' AND source_hit_rank IS NOT NULL AND capability != 'out_of_corpus')",
    "correct_source_missed": "(verdict='correct' AND source_hit_rank IS NULL AND capability != 'out_of_corpus')",
    "failed_source_found": "(verdict IN ('incorrect','partial') AND source_hit_rank IS NOT NULL AND capability != 'out_of_corpus')",
    "failed_source_missed": "(verdict IN ('incorrect','partial') AND source_hit_rank IS NULL AND capability != 'out_of_corpus')",
}

#: columns selected for per-question results — trace_json is deliberately absent
#: (trap 10: never select the prunable trace blob in the public results feed).
_RESULT_COLUMNS = (
    "run_id, question_id, capability, difficulty, question_text, expected_answer, "
    "answer_text, abstained, verdict, verdict_reason, source_hit_rank, "
    "source_coverage, sub_fact_coverage, retrieved_paths, rounds, latency_ms, "
    "error, input_tokens, output_tokens"
)


@router.get("/runs/{run_id}/results", response_model=BenchResultsPage)
async def get_bench_run_results(
    request: Request,
    run_id: int,
    verdict: str | None = Query(None, description="filter by verdict"),
    capability: str | None = Query(None, description="filter by capability"),
    attribution: str | None = Query(None, description="attribution 2x2 cell"),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
) -> BenchResultsPage:
    """Per-question rows, paginated + filterable. NEVER selects ``trace_json``."""
    state: AppState = app_state(request)
    store = SqliteBenchStore(state.db)
    if await store.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")

    where = ["run_id=?"]
    params: list[Any] = [run_id]
    if verdict:
        where.append("verdict=?")
        params.append(verdict)
    if capability:
        where.append("capability=?")
        params.append(capability)
    if attribution:
        cell_where = _ATTRIBUTION_FILTERS.get(attribution)
        if cell_where is None:
            raise HTTPException(status_code=400, detail=f"unknown attribution cell {attribution!r}")
        where.append(cell_where)
    where_sql = " AND ".join(where)

    db = state.db
    async with db.read() as conn:
        cur = await conn.execute(
            f"SELECT {_RESULT_COLUMNS} FROM bench_question_results "
            f"WHERE {where_sql} ORDER BY question_id LIMIT ? OFFSET ?",
            (*params, limit, offset),
        )
        rows = await cur.fetchall()
        cur_count = await conn.execute(
            f"SELECT COUNT(*) FROM bench_question_results WHERE {where_sql}", params
        )
        count_row = await cur_count.fetchone()
        total = int(count_row[0]) if count_row is not None else 0

    items = [_row_to_result_row(r) for r in rows]
    return BenchResultsPage(items=items, total=total, offset=offset, limit=limit)


@router.get("/compare", response_model=BenchCompareResponse)
async def compare_bench_runs(
    request: Request,
    runs: str = Query("", description="comma-separated run ids, e.g. 1,2,3"),
) -> BenchCompareResponse:
    """Per-question diff across runs: fixed/broken/both_correct/both_wrong/unjudged."""
    state: AppState = app_state(request)
    store = SqliteBenchStore(state.db)
    try:
        ids = [int(p) for p in runs.split(",") if p.strip()]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid run ids: {runs!r}") from exc
    if len(ids) < 2:
        raise HTTPException(
            status_code=400, detail="compare requires at least 2 run ids (?runs=a,b)"
        )
    for rid in ids:
        if await store.get_run(rid) is None:
            raise HTTPException(status_code=404, detail=f"run {rid} not found")
    # The dataset question map aligns the source_recall delta's denominator
    # with the headline metric (excludes abstain/out_of_corpus questions).
    # Tolerant: a missing/unloadable dataset file degrades to the legacy
    # all-shared-questions denominator instead of failing an otherwise valid
    # comparison.
    try:
        dataset = load_bench_dataset(str(app_config.get(BENCH_DATASET)))
        questions = {q.id: q for q in dataset.questions}
    except Exception:
        questions = None

    pairs: list[ComparePair] = []
    try:
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                cmp = await compare_runs(store, ids[i], ids[j], questions=questions)
                pairs.append(
                    ComparePair(
                        run_a=cmp.run_a,
                        run_b=cmp.run_b,
                        shared_denominator=cmp.shared_denominator,
                        fixed=list(cmp.fixed),
                        broken=list(cmp.broken),
                        both_correct=list(cmp.both_correct),
                        both_wrong=list(cmp.both_wrong),
                        only_a=list(cmp.only_a),
                        only_b=list(cmp.only_b),
                        unjudged=list(cmp.unjudged),
                        deltas=cmp.deltas,
                    )
                )
    except IncomparableRuns as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return BenchCompareResponse(runs=ids, pairs=pairs)


@router.get("/dataset", response_model=BenchDatasetInfo)
async def get_bench_dataset_info(
    request: Request,
    path: str | None = Query(None, description="dataset path (default bench.dataset)"),
) -> BenchDatasetInfo:
    """Dataset metadata: hash, composition, ceiling and floor scores."""
    del request
    dataset_path = path or str(app_config.get(BENCH_DATASET))
    try:
        dataset = load_bench_dataset(dataset_path)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"could not load dataset {dataset_path!r}: {exc!r}"
        ) from exc

    active = [q for q in dataset.questions if q.status == "active"]
    by_cap: dict[str, int] = {}
    by_diff: dict[str, int] = {}
    by_slice: dict[str, int] = {}
    ceiling_correct = 0
    floor_correct = 0
    for q in active:
        by_cap[q.capability] = by_cap.get(q.capability, 0) + 1
        by_diff[q.difficulty] = by_diff.get(q.difficulty, 0) + 1
        by_slice[q.slice] = by_slice.get(q.slice, 0) + 1
        if str(q.oracle.get("verdict", "")).strip().lower() == "correct":
            ceiling_correct += 1
        if str(q.closed_book.get("verdict", "")).strip().lower() == "correct":
            floor_correct += 1
    n = len(active)
    return BenchDatasetInfo(
        name=dataset.name,
        version=dataset.version,
        hash=dataset.hash,
        generated=dataset.generated,
        total=n,
        by_capability=by_cap,
        by_difficulty=by_diff,
        by_slice=by_slice,
        ceiling={
            "correct": ceiling_correct,
            "total": n,
            "score": (ceiling_correct / n) if n else 0.0,
        },
        floor={
            "correct": floor_correct,
            "total": n,
            "score": (floor_correct / n) if n else 0.0,
        },
    )


@router.delete("/runs/{run_id}")
async def delete_bench_run(request: Request, run_id: int) -> dict[str, object]:
    """Delete a run (cascades to its per-question rows via FK ON DELETE CASCADE)."""
    state: AppState = app_state(request)
    store = SqliteBenchStore(state.db)
    record = await store.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    task = _tasks.get(record.run_group)
    if task is not None and not task.done():
        raise HTTPException(
            status_code=409,
            detail=(
                f"run {run_id} belongs to a bench group that is currently executing; "
                "wait for it to finish or cancel it first"
            ),
        )
    await store.delete_run(run_id)
    return {"deleted": run_id}


# ── Row serializers ─────────────────────────────────────────────────────────


def _row_to_result_row(row: aiosqlite.Row) -> BenchResultRow:
    """Map a results-feed row (no trace_json column) to the public DTO."""
    raw_paths = row["retrieved_paths"]
    try:
        paths: list[str] = [str(p) for p in json.loads(raw_paths)] if raw_paths else []
    except (json.JSONDecodeError, TypeError):
        paths = []
    return BenchResultRow(
        run_id=int(row["run_id"]),
        question_id=str(row["question_id"]),
        capability=str(row["capability"]),
        difficulty=str(row["difficulty"]),
        question_text=str(row["question_text"]),
        expected_answer=str(row["expected_answer"]),
        answer_text=str(row["answer_text"]),
        abstained=bool(row["abstained"]),
        verdict=str(row["verdict"]),
        verdict_reason=str(row["verdict_reason"]),
        source_hit_rank=int(row["source_hit_rank"]) if row["source_hit_rank"] is not None else None,
        source_coverage=float(row["source_coverage"]),
        sub_fact_coverage=float(row["sub_fact_coverage"])
        if row["sub_fact_coverage"] is not None
        else None,
        retrieved_paths=paths,
        rounds=int(row["rounds"]),
        latency_ms=float(row["latency_ms"]),
        error=str(row["error"]) if row["error"] is not None else None,
        input_tokens=int(row["input_tokens"] or 0),
        output_tokens=int(row["output_tokens"] or 0),
    )


def _to_summary(record: BenchRunRecord) -> BenchRunSummary:
    m = record.metrics_json
    return BenchRunSummary(
        id=record.id,
        run_group=record.run_group,
        label=record.label,
        started_at=record.started_at,
        finished_at=record.finished_at,
        status=record.status,
        dataset_name=record.dataset_name,
        dataset_hash=record.dataset_hash,
        subset_hash=record.subset_hash,
        system=record.system,
        profile_name=record.profile_name,
        answer_model=record.answer_model,
        judge_model=record.judge_model,
        scope=record.scope,
        trusted=record.trusted,
        headroom=_metric(m, "reference.headroom_realised"),
        strict_accuracy=_metric(m, "answer.strict_accuracy"),
        source_recall_at_10=_metric(m, "source.recall_at_10"),
        hallucination_rate=_metric(m, "answer.hallucination_rate"),
        unjudged=_metric(m, "answer.unjudged"),
        complete=_metric(m, "answer.complete"),
    )


def _to_detail(record: BenchRunRecord) -> BenchRunDetail:
    return BenchRunDetail(
        id=record.id,
        run_group=record.run_group,
        label=record.label,
        started_at=record.started_at,
        finished_at=record.finished_at,
        status=record.status,
        dataset_name=record.dataset_name,
        dataset_hash=record.dataset_hash,
        subset_hash=record.subset_hash,
        system=record.system,
        profile_name=record.profile_name,
        profile_hash=record.profile_hash,
        answer_model=record.answer_model,
        judge_model=record.judge_model,
        scope=record.scope,
        trusted=record.trusted,
        calibration=record.calibration,
        judge_shares_endpoint=record.judge_shares_endpoint,
        abort_reason=record.abort_reason,
        config_json=dict(record.config_json),
        metrics=dict(record.metrics_json),
        progress=None,
    )


__all__ = [
    "SYSTEM_CLASSES",
    "AgenticPydanticSystem",
    "ClosedBookSystem",
    "GatewayJudgeLLM",
    "InMemoryBenchStore",
    "OracleSystem",
    "RetrievalOnlySystem",
    "SourcesOnlySystem",
    "SqliteBenchStore",
    "build_judge_gateway",
    "import_answer_runs",
    "make_judge_llm",
    "make_system",
    "reconcile_stale_bench_runs",
    "router",
]
