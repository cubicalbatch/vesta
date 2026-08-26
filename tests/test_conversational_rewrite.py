"""Conversational rewrite preparer tests.

Covers the foundational contract: turn 1 is a
no-op, turn ≥2 rewrites, and every failure mode (no rewriter, gateway error,
empty output) degrades to the raw query — never worse.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from vesta.retrieval.contracts import PreparedQuery
from vesta.retrieval.impls.conversational_rewrite import ConversationalRewrite
from vesta.retrieval.trace import Trace


def _pq(history: tuple[tuple[str, str], ...] = ()) -> PreparedQuery:
    return PreparedQuery(
        raw="who was his wife",
        terms=("who", "was", "his", "wife"),
        text="who was his wife",
        aliases=(),
        is_keyword_query=False,
        rung="initial",
        history=history,
    )


class _FakeRewriter:
    """A controllable stand-in for the QueryRewriter Protocol."""

    def __init__(
        self, result: str = "albert einstein wife", error: Exception | None = None
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[tuple[str, list[tuple[str, str]]]] = []

    async def rewrite(self, query: str, history: Sequence[tuple[str, str]]) -> str:
        self.calls.append((query, list(history)))
        if self._error is not None:
            raise self._error
        return self._result


@pytest.mark.asyncio
async def test_turn1_is_noop_without_history() -> None:
    """Turn 1 (no history) returns the query unchanged even with a rewriter."""
    rewriter = _FakeRewriter()
    prep = ConversationalRewrite(rewriter=rewriter)
    q = _pq(history=())
    out = await prep.prepare(q, Trace())
    assert out is q
    assert rewriter.calls == []


@pytest.mark.asyncio
async def test_turn2_rewrites_using_history() -> None:
    """Turn ≥2 (history present) rewrites and preserves raw/history."""
    rewriter = _FakeRewriter(result="albert einstein wife")
    prep = ConversationalRewrite(rewriter=rewriter)
    history = (("user", "tell me about einstein"), ("assistant", "Einstein was a physicist"))
    q = _pq(history=history)
    out = await prep.prepare(q, Trace())
    assert out is not q
    assert out.text == "albert einstein wife"
    assert out.terms == ("albert", "einstein", "wife")
    assert out.is_keyword_query is True
    assert out.raw == q.raw
    assert out.history == history
    assert rewriter.calls[0][0] == "who was his wife"
    assert rewriter.calls[0][1] == list(history)


@pytest.mark.parametrize(
    "prep",
    [
        ConversationalRewrite(),
        ConversationalRewrite(rewriter=_FakeRewriter(error=RuntimeError("boom"))),
        ConversationalRewrite(rewriter=_FakeRewriter(result="   ")),
        ConversationalRewrite(rewriter=_FakeRewriter(result='""')),
        ConversationalRewrite(rewriter=_FakeRewriter(result="???")),
    ],
)
async def test_fallback_modes_preserve_raw_query(prep: ConversationalRewrite) -> None:
    """Degrade, don't fail: missing rewriter, exceptions, empty or punctuation-only rewrites return raw."""
    q = _pq(history=(("user", "ctx"),))
    out = await prep.prepare(q, Trace())
    assert out is q


def test_requires_llm_capability() -> None:
    """The preparer requires Capability.LLM so profiles drop it on a no-LLM box
    (degrade, don't fail)."""
    from vesta.config.capabilities import Capability

    assert ConversationalRewrite.requires == frozenset({Capability.LLM})


@pytest.mark.asyncio
async def test_max_history_turns_truncates_to_most_recent() -> None:
    """max_history_turns caps how many recent entries reach the rewriter."""
    rewriter = _FakeRewriter()
    prep = ConversationalRewrite(rewriter=rewriter)  # default max_history_turns = 4
    history = tuple((role, str(i)) for i, role in enumerate(["user", "assistant"] * 10))
    q = _pq(history=history)
    await prep.prepare(q, Trace())
    assert len(rewriter.calls) == 1
    assert rewriter.calls[0][1] == list(history[-4:])


@pytest.mark.asyncio
async def test_gateway_backed_rewriter_maps_history_to_messages() -> None:
    """The concrete GatewayQueryRewriter maps history → chat messages and calls
    chat_once once; the system prompt is the preparer-owned template."""
    from vesta.answer.rewriter import GatewayQueryRewriter
    from vesta.inference.gateway import ChatDelta, ChatMessage, ChatResult
    from vesta.retrieval.impls.conversational_rewrite import REWRITE_SYSTEM_PROMPT

    captured: list[list[ChatMessage]] = []

    class _FakeGateway:
        async def chat_stream(self, messages, *, model, **kwargs):
            yield ChatDelta(text="", finish_reason=None)

        async def chat_once(
            self,
            messages: Sequence[ChatMessage],
            *,
            model: str,
            temperature: float = 0.2,
            max_tokens: int = 300,
            enable_thinking: bool | None = None,
            timeout: float | None = None,
        ) -> ChatResult:
            captured.append(list(messages))
            return ChatResult(text="standalone query", finish_reason="stop", latency_ms=1.0)

    rw = GatewayQueryRewriter(_FakeGateway(), model="m")
    out = await rw.rewrite("his wife", (("user", "about einstein"), ("assistant", "...")))
    assert out == "standalone query"
    # system + 2 history + 1 current = 4 messages
    assert len(captured[0]) == 4
    assert captured[0][0].role == "system"
    assert captured[0][0].content == REWRITE_SYSTEM_PROMPT
    assert captured[0][-1].role == "user"
    assert captured[0][-1].content == "his wife"


@pytest.mark.asyncio
async def test_preparer_constructs_via_ladder_with_rewriter() -> None:
    """The pipeline's TypeError ladder constructs the preparer with the rewriter
    at the top rung (params, archives, rewriter) — archives is accepted but
    unused, matching the vector_knn bundling convention."""
    rewriter = _FakeRewriter()
    prep = ConversationalRewrite(
        params=ConversationalRewrite.Params(),
        archives=None,  # would be deps.archives from the ladder
        rewriter=rewriter,
    )
    q = _pq(history=(("user", "ctx"),))
    out = await prep.prepare(q, Trace())
    assert out.text == "albert einstein wife"


@pytest.mark.asyncio
async def test_full_preparer_chain_preserves_history_to_rewriter() -> None:
    """normalize -> alias_expand -> conversational_rewrite, in profile order.

    Regression check: the rewriter must still receive the turn's history
    after the two preceding preparers run.
    """
    from vesta.retrieval.impls.alias_expand import AliasExpand
    from vesta.retrieval.impls.normalize import Normalize

    class _FakeArchives:
        async def lookup_aliases(self, terms: list[str], *, max_aliases: int) -> list[str]:
            return ["relativity"]

    history = (("user", "tell me about einstein"), ("assistant", "Einstein was a physicist"))
    q = PreparedQuery(
        raw="what was his wife's name",
        terms=(),
        text="what was his wife's name",
        aliases=(),
        is_keyword_query=False,
        rung="initial",
        history=history,
    )
    tr = Trace()
    q_norm = await Normalize().prepare(q, tr)
    assert q_norm.history == history
    q_alias = await AliasExpand(archives=_FakeArchives()).prepare(q_norm, tr)  # type: ignore[arg-type]
    assert q_alias.history == history
    assert q_alias.aliases == ("relativity",)

    rewriter = _FakeRewriter(result="einstein wife name")
    q_rewritten = await ConversationalRewrite(rewriter=rewriter).prepare(q_alias, tr)

    assert rewriter.calls, (
        "conversational_rewrite never invoked the rewriter — history was lost upstream"
    )
    assert rewriter.calls[0][1] == list(history)
    assert q_rewritten.text == "einstein wife name"


@pytest.mark.asyncio
async def test_rewritten_query_normalizes_casing_and_punctuation() -> None:
    """Rewritten queries are lowercased and have terms cleaned of quotes and punctuation."""
    rewriter = _FakeRewriter(result="  Albert Einstein's General Relativity  ")
    prep = ConversationalRewrite(rewriter=rewriter)
    q = _pq(history=(("user", "tell me about einstein"),))
    out = await prep.prepare(q, Trace())
    assert out.text == "albert einstein's general relativity"
    assert out.terms == ("albert", "einstein", "s", "general", "relativity")
    assert out.is_keyword_query is True

    # Quoted query: text is lowercased and stripped of outer whitespace, while terms strip quotes
    rewriter_quoted = _FakeRewriter(result=' "Albert Einstein" ')
    prep_quoted = ConversationalRewrite(rewriter=rewriter_quoted)
    out_quoted = await prep_quoted.prepare(q, Trace())
    assert out_quoted.text == '"albert einstein"'
    assert out_quoted.terms == ("albert", "einstein")
    assert out_quoted.is_keyword_query is True


@pytest.mark.parametrize(
    ("rewritten_text", "expected_terms", "expected_is_keyword"),
    [
        ("hotels near the grand canyon", ("hotels", "grand", "canyon"), True),
        ("what is photosynthesis not artificial", ("photosynthesis", "artificial"), False),
        ("How Do I Mount a USB Drive?", ("mount", "usb", "drive"), False),
        ("restaurants xor cafes", ("restaurants", "cafes"), True),
    ],
)
@pytest.mark.asyncio
async def test_rewritten_query_strips_stopwords_and_inert_tokens(
    rewritten_text: str,
    expected_terms: tuple[str, ...],
    expected_is_keyword: bool,
) -> None:
    """Rewritten queries apply stopword stripping and inert token removal."""
    rewriter = _FakeRewriter(result=rewritten_text)
    prep = ConversationalRewrite(rewriter=rewriter)
    q = _pq(history=(("user", "ctx"),))
    out = await prep.prepare(q, Trace())
    assert out.terms == expected_terms
    assert out.is_keyword_query is expected_is_keyword


@pytest.mark.asyncio
async def test_rewritten_query_updates_is_keyword_query_from_keyword_to_question() -> None:
    """A keyword lookup rewritten into a question updates is_keyword_query from True to False."""
    q = PreparedQuery(
        raw="einstein",
        terms=("einstein",),
        text="einstein",
        aliases=(),
        is_keyword_query=True,
        rung="initial",
        history=(("user", "einstein"), ("assistant", "Albert Einstein was a physicist.")),
    )
    rewriter = _FakeRewriter(result="what did albert einstein discover?")
    prep = ConversationalRewrite(rewriter=rewriter)
    out = await prep.prepare(q, Trace())
    assert out.text == "what did albert einstein discover?"
    assert out.terms == ("albert", "einstein", "discover")
    assert out.is_keyword_query is False
    assert out.raw == "einstein"


@pytest.mark.asyncio
async def test_rewritten_query_updates_is_keyword_query_from_question_to_keyword() -> None:
    """A question rewritten into a keyword phrase updates is_keyword_query from False to True."""
    q = PreparedQuery(
        raw="who was his first wife?",
        terms=("first", "wife"),
        text="who was his first wife?",
        aliases=(),
        is_keyword_query=False,
        rung="initial",
        history=(("user", "tell me about einstein"), ("assistant", "Einstein was a physicist.")),
    )
    rewriter = _FakeRewriter(result="albert einstein mileva maric")
    prep = ConversationalRewrite(rewriter=rewriter)
    out = await prep.prepare(q, Trace())
    assert out.text == "albert einstein mileva maric"
    assert out.terms == ("albert", "einstein", "mileva", "maric")
    assert out.is_keyword_query is True
    assert out.raw == "who was his first wife?"


@pytest.mark.asyncio
async def test_rewritten_query_all_inert_tokens_keeps_terms() -> None:
    """A rewritten query made entirely of inert tokens still yields non-empty terms."""
    rewriter = _FakeRewriter(result="not near xor")
    prep = ConversationalRewrite(rewriter=rewriter)
    q = _pq(history=(("user", "ctx"),))
    out = await prep.prepare(q, Trace())
    assert out.terms
