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

The rendering of a search result for the model is answer policy, not
retrieval policy — the agent's search driver (``api/agent_chat.py``)
renders passages itself, card-numbered and budget-capped.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from vesta.retrieval.contracts import (
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

    ``text`` is what reaches the model when the search found nothing: the
    no-passages sentinel plus any appended candidate blocks (the only shape
    the agent reads back verbatim). When passages exist the agent's search
    driver re-renders them itself — card-numbered, budget-capped — and
    re-attaches ``candidates_text``. ``passages``/``cards`` carry the
    underlying retrieval result so the agent can merge search-round evidence
    into source cards and citations instead of discarding everything but the
    formatted string. ``trace`` carries the retrieval pipeline trace (per-stage
    ``duration_ms``) so the answer trace can show where time went.
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
    trace: dict[str, object] | None = None
    candidates_text: str = ""


#: Type aliases for the injected callables. Both are async. ``search`` returns
#: either a plain string (test-fake shape) or a :class:`SearchToolResult` (the
#: real composition root, unlocking evidence merge) — the tool runtime never
#: touches ``zim/`` or builds a ``Deps`` itself.
SearchFn = Callable[[str, str], Awaitable["str | SearchToolResult"]]


class ReadArticleFn(Protocol):
    """The injected ``read_article`` callable: ``(zim_id, path) -> text``.

    ``must_include`` (keyword-only, default empty) carries the source card's
    retrieval snippet (AUDIT_0824 N11). The composition root's focused-view
    read locates it in the FULL article text and forces the span in via
    ``must_include_spans``, so the snippet survives the 32k stage-1 elision and
    the harness's stage-2 ``find()`` succeeds on articles longer than that
    window. Empty means "no snippet to force" — test fakes keep working
    unchanged by accepting and ignoring it.
    """

    def __call__(self, zim_id: int, path: str, *, must_include: str = "") -> Awaitable[str]: ...


@dataclass
class ToolRuntime:
    """Holds the injected callables the agent's tools dispatch to.

    The composition root (``api/answer.py``) constructs one of these from the
    live archive registry + retrieval deps and hands it to the agent chat
    path. Each callable returns a string ready to append to the model's
    context — the runtime does no formatting beyond a light envelope.

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


__all__ = [
    "ReadArticleFn",
    "SearchFn",
    "SearchToolResult",
    "ToolRuntime",
]
