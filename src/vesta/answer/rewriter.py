"""Gateway-backed conversational query rewriter.

The concrete :class:`QueryRewriter` the composition root injects into
``Deps.rewriter``. Lives in ``answer/`` (not ``retrieval/``, which must not
import ``inference/``; not ``inference/``, which must not own retrieval policy)
because it is the one place that bridges the retrieval-side
:class:`~vesta.retrieval.contracts.QueryRewriter` Protocol with the
inference-side :class:`~vesta.inference.gateway.Gateway`.

It is a thin adapter: turn the conversation history into chat messages, hand the
system prompt (owned by the preparer) to ``gateway.chat_once``, return the text.
Every failure mode is the caller's problem: the preparer falls back to the raw
query on exception or empty output (never make things worse).
"""

from __future__ import annotations

from collections.abc import Sequence

from vesta.inference.gateway import ChatMessage, Gateway
from vesta.retrieval.impls.conversational_rewrite import REWRITE_SYSTEM_PROMPT


class GatewayQueryRewriter:
    """Rewrite a follow-up query via ``gateway.chat_once``.

    ``model`` is the configured chat model id. ``enable_thinking`` defaults to
    ``False`` — a rewrite is a quick, deterministic task and reasoning models
    would burn the token budget on hidden thinking (same default as
    the answer path's ``inference.llm.enable_thinking``).
    """

    def __init__(
        self,
        gateway: Gateway,
        *,
        model: str,
        max_tokens: int = 96,
        temperature: float = 0.0,
        enable_thinking: bool = False,
    ) -> None:
        self._gateway = gateway
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._enable_thinking = enable_thinking

    async def rewrite(self, query: str, history: Sequence[tuple[str, str]]) -> str:
        """Return a standalone search query for ``query`` given ``history``.

        ``history`` is ``(role, content)`` pairs straight off
        :attr:`~vesta.retrieval.contracts.PreparedQuery.history`. The current
        question is appended as the final ``user`` turn, so the model rewrites it
        in context. Raises only if the gateway call itself fails; the preparer
        catches that and falls back.
        """
        messages: list[ChatMessage] = [ChatMessage(role="system", content=REWRITE_SYSTEM_PROMPT)]
        for role, content in history:
            messages.append(ChatMessage(role=role, content=content))
        messages.append(ChatMessage(role="user", content=query))
        result = await self._gateway.chat_once(
            messages,
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            enable_thinking=self._enable_thinking,
        )
        return result.text


__all__ = ["GatewayQueryRewriter"]
