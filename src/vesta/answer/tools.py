"""Dependency-injection seam for the agent's ``search``/``read_article`` tools.

The pydantic-ai answer agent (``api/agent_chat.py``) exposes two tools to the
model — ``search`` and ``read_article`` — and dispatches each through the
injected callables held by :class:`ToolRuntime`. The agent registers its tools
natively with pydantic-ai (``@agent.tool_plain`` in
``_TurnContext.build_agent``); the model calls them through the provider's
tool-calling protocol, and pydantic-ai owns the call/execute/feed-back loop.
This module holds no tool-call parsing or dispatch — only the callables the
agent's closures invoke, and the rendering of a search result for the model.

``answer/`` cannot import ``zim/`` (the module dependency cap is 3:
retrieval, inference, config), so ``read_article`` accesses ZIMs through an
injected callable, the same DI pattern used for ``Deps.vectors``
(typed under ``TYPE_CHECKING``, constructed by the composition root
``api/answer.py``'s ``_build_tool_runtime``). The tool runtime never touches
``zim/`` or builds a ``Deps`` itself.

:func:`format_search_result` is the answer-layer rendering of a
:class:`RetrievalResult` as the string the ``search`` tool returns — kept here
(rather than in ``retrieval/``) because "how much to show the model, in what
order" is answer policy, not retrieval policy.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vesta.retrieval.contracts import (
        ConfidenceSignals,
        RetrievalResult,
        ScoredPassage,
        SourceCard,
    )

#: Absolute fallback cap on how much of a full article's text is used in a
#: deep-read regenerate. Shared by the agent path's ``read_article`` tool
#: (``api/answer.py``'s ``_build_tool_runtime``), which bounds its focused-view
#: read with this constant. Do NOT raise this past 32k: the answer model
#: (qwen3.5-4b) has a 32k-token context window and a documented "context cliff"
#: beyond ~2.5k tokens of assembled context where extraction quality degrades.
_MAX_FULL_ARTICLE_CHARS = 32_000


@dataclass(frozen=True)
class SearchToolResult:
    """The ``search`` tool's structured result.

    ``text`` is what the agent appends to the model's context (unchanged wire
    shape). ``passages``/``cards``/``confidence`` carry the underlying retrieval
    result so the agent can merge search-round evidence into source cards and
    citations instead of discarding everything but the formatted string.
    ``confidence`` lets a recovery round re-check confidence against what
    the query actually found. ``trace`` carries the retrieval pipeline trace
    (per-stage ``duration_ms``) so the answer trace can show where time went.
    ``candidates_text`` holds any extra candidate-article blocks (term-surfaced
    titles, reformulated-article visibility) that were appended to ``text`` —
    a harness that re-renders ``passages`` instead of using ``text`` verbatim
    (the agent's search driver) re-attaches this so surfaced articles still
    reach the model. Empty when no recovery block fired.

    A composition root's ``search`` callable MAY return a bare ``str`` instead
    (see :data:`SearchFn`) — every test fake does, and the agent's search driver
    handles both shapes via ``isinstance`` (no merge data when it's a plain
    string). Only ``api/answer.py``'s real callable returns this richer shape.
    """

    text: str
    passages: tuple[ScoredPassage, ...] = ()
    cards: tuple[SourceCard, ...] = ()
    confidence: ConfidenceSignals | None = None
    trace: dict[str, object] | None = None
    candidates_text: str = ""


#: Type aliases for the injected callables. Both are async. ``search`` returns
#: either a plain string (test-fake shape) or a :class:`SearchToolResult` (the
#: real composition root, unlocking evidence merge) — the tool runtime never
#: touches ``zim/`` or builds a ``Deps`` itself.
SearchFn = Callable[[str, str], Awaitable["str | SearchToolResult"]]
ReadArticleFn = Callable[[int, str], Awaitable[str]]


@dataclass
class ToolRuntime:
    """Holds the injected callables the agent's tools dispatch to.

    The composition root (``api/answer.py``) constructs one of these from the
    live archive registry + retrieval deps and passes it via
    :attr:`AnswerDeps.tools`. Each callable returns a string ready to append to
    the model's context — the runtime does no formatting beyond a light envelope.

    ``search`` takes ``(query, scope_str)`` and returns formatted passages. The
    scope string is the raw comma-separated zim_ids the model passed (or empty
    for "all enabled"), parsed by the injected callable.

    ``search_exact`` is an optional second search entry point with the same
    ``(query, scope_str) -> SearchToolResult`` shape as :attr:`search`, but
    without any prefix-shortening/term-surfacing recovery the composition root
    layers onto tool-driven queries. It exists for callers that need to run
    retrieval on a raw, full natural-language question (e.g. the agent's
    Round-0 pre-seed) where that recovery ladder — tuned for short, fact-shaped
    tool-call rephrases — would degrade a mid-sentence entity into a
    leading-stopword query. Defaults to ``None`` (falls back to :attr:`search`
    unchanged) so every existing test fake and call site keeps working as-is.
    """

    search: SearchFn
    read_article: ReadArticleFn
    search_exact: SearchFn | None = None


# ── Formatting helpers for the composition root ─────────────────────────────


def format_search_result(
    result: RetrievalResult,
    *,
    max_passages: int = 8,
    archive_labels: Mapping[int, str] | None = None,
) -> str:
    """Format a ``RetrievalResult`` as the string the ``search`` tool returns.

    Lives here (not in ``retrieval/``) because the rendering is answer-layer
    policy: how much to show the model, in what order. The composition root calls
    this from inside the injected ``search`` callable so ``retrieval/`` stays
    unaware of the tool protocol.

    ``archive_labels`` (``AnswerDeps.archive_labels``) replaces the opaque
    ``archive-{zim_id}`` label with a human-readable name when one is known, so
    a card looks the same wherever the model sees it. ``None``/missing falls
    back to ``archive-{zim_id}``.
    """
    if not result.passages:
        return "Search returned no passages."
    labels = archive_labels or {}
    lines = [f"Search returned {len(result.passages)} passages (showing top {max_passages}):"]
    for i, sp in enumerate(result.passages[:max_passages]):
        p = sp.passage
        parts = p.breadcrumb.split(">")
        title = parts[0].strip() if parts else p.path
        archive = labels.get(p.zim_id) or f"archive-{p.zim_id}"
        lines.append(f'[{i + 1}] {archive} · "{title}"')
        lines.append(p.text)
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "ReadArticleFn",
    "SearchFn",
    "SearchToolResult",
    "ToolRuntime",
    "format_search_result",
]
