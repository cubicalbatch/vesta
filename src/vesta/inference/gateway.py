"""Inference gateway — one ``AsyncOpenAI`` client for local and remote.

The deciding argument for bundling ``llama-server`` is that it makes
the LLM client identical for both backends: one ``AsyncOpenAI(base_url=...)``
instance, local mode points at the supervised child on ``127.0.0.1``, remote mode
points at whatever the user typed. **Nothing upstream branches on backend** —
``gateway.chat_stream(...)`` is the entire surface.

Three independent roles (``llm``, ``embed``, ``rerank``) are conceptually distinct,
but only ``llm`` is exercised through this gateway. Embedding and
reranking stay in-process via ``encoders/`` (ONNX is ~3x
faster for encoder work and the embedder must be pinned for index integrity).

``inference/`` depends ONLY on ``config``.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from vesta.inference.local import LlamaServerSupervisor
    from vesta.inference.runtime import LlmRuntime


@dataclass(frozen=True)
class ChatMessage:
    """One message in a chat conversation (OpenAI ``role``/``content`` shape)."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class ChatDelta:
    """One streamed chunk from a chat completion.

    ``text`` is the incremental token text (may be empty on the final chunk).
    ``finish_reason`` is set on the last chunk (``"stop"`` or ``"length"``), else
    ``None``. ``input_tokens`` / ``output_tokens`` / ``total_tokens`` are set on
    the final usage chunk (when ``stream_options.include_usage`` is enabled);
    they stay ``0`` on intermediate chunks.
    """

    text: str
    finish_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @property
    def has_usage(self) -> bool:
        """True when this delta carries token usage (the final usage chunk)."""
        return self.total_tokens > 0


@dataclass(frozen=True)
class ChatResult:
    """Non-streaming chat result (used by the round-trip ``test`` probe).

    Token usage is populated from the endpoint's ``usage`` block when present
    (0 when the endpoint does not report it).
    """

    text: str
    finish_reason: str | None
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class Gateway(Protocol):
    """The inference surface every upstream caller uses.

    ``chat_stream`` is an async generator yielding :class:`ChatDelta` chunks as
    they arrive from the model. ``chat_once`` is the non-streaming variant used
    by the conversational query rewriter and the benchmark's closed-book system.

    ``enable_thinking`` forwards ``chat_template_kwargs.enable_thinking`` to the
    endpoint (Qwen3/Qwen3.5 reasoning models think by default). ``None`` sends
    nothing and leaves the model's default behaviour
    untouched; endpoints that don't know the parameter ignore it.
    ``timeout`` overrides the client's default per-call timeout (seconds); ``None``
    uses the gateway's configured default.
    """

    def chat_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 300,
        enable_thinking: bool | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[ChatDelta]: ...

    async def chat_once(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 300,
        enable_thinking: bool | None = None,
        timeout: float | None = None,
    ) -> ChatResult: ...

    async def aclose(self) -> None: ...


def _extract_choices(resp: Any) -> list[Any]:
    """Choices from a chat completion, tolerating a ``{"data": ...}`` envelope.

    Some OpenAI-compatible proxies (observed: ``api.cline.bot``) wrap the
    standard completion in ``{"data": {choices: [...]}, "success": true}``.
    The openai SDK then sees no top-level ``choices`` and leaves the payload in
    ``model_extra`` — without unwrapping, every call silently returns empty
    text (the caller falls back to a lexical estimate: judge verdicts graded
    "not judged"). Standard endpoints (local ``llama-server``) return
    ``choices`` directly and are returned unchanged.
    """

    if getattr(resp, "choices", None):
        return list(resp.choices)
    extra = getattr(resp, "model_extra", None)
    if isinstance(extra, dict):
        data = extra.get("data")
        if isinstance(data, dict):
            raw = data.get("choices")
            if isinstance(raw, list):
                return raw
    return []


def _extract_usage(resp_or_chunk: Any) -> tuple[int, int, int]:
    """Pull ``(input, output, total)`` token counts from a completion / chunk.

    Tolerates the same ``{"data": ...}`` envelope as :func:`_extract_choices`
    and the OpenAI SDK's attribute names (``prompt_tokens`` /
    ``completion_tokens``). Returns ``(0, 0, 0)`` when the endpoint does not
    report usage (some proxies and older ``llama-server`` builds omit it).
    """
    usage = getattr(resp_or_chunk, "usage", None)
    if usage is None and isinstance(resp_or_chunk, dict):
        usage = resp_or_chunk.get("usage")
    if usage is None:
        extra = getattr(resp_or_chunk, "model_extra", None)
        if isinstance(extra, dict):
            data = extra.get("data")
            if isinstance(data, dict):
                usage = data.get("usage")
        elif isinstance(resp_or_chunk, dict):
            data = resp_or_chunk.get("data")
            if isinstance(data, dict):
                usage = data.get("usage")
    if usage is None:
        return (0, 0, 0)

    def _get(obj: Any, *names: str, default: int = 0) -> int:
        for n in names:
            if isinstance(obj, dict):
                val = obj.get(n)
            else:
                val = getattr(obj, n, None)
                if val is None and hasattr(obj, "get") and callable(obj.get):
                    val = obj.get(n)
            if val is None:
                continue
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
        return default

    inp = _get(usage, "prompt_tokens", "input_tokens")
    out = _get(usage, "completion_tokens", "output_tokens")
    total = _get(usage, "total_tokens")
    if total == 0:
        total = inp + out
    return (inp, out, total)


class OpenAIGateway:
    """Concrete gateway backed by one ``AsyncOpenAI`` client.

    For the **local** source, the base URL points at the supervised
    ``llama-server`` child on ``127.0.0.1:<port>``. The supervisor is started
    lazily: only when an answer is actually requested, not at app startup.
    For the **remote** source, the base URL is whatever the user configured.

    Both paths go through identical OpenAI-compatible HTTP; there is zero
    branching in streaming, error mapping, or token handling — the surface where
    branching actually hurts.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        supervisor: LlamaServerSupervisor | None = None,
        runtime: LlmRuntime | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._supervisor = supervisor
        self._runtime = runtime
        self._timeout = timeout
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "not-set",
            timeout=timeout,
        )

    @property
    def base_url(self) -> str:
        """The OpenAI-compatible base URL this client points at."""
        return self._base_url

    @property
    def api_key(self) -> str:
        """The configured API key (or 'local' / empty for local/unauthenticated)."""
        return self._api_key

    @property
    def supervisor(self) -> LlamaServerSupervisor | None:
        """The supervised child's owner (``None`` for the remote source)."""
        return self._supervisor

    async def _ensure_ready(self) -> None:
        """For the local source, ensure the supervised ``llama-server`` is up.

        Lazy start: the supervisor is only spawned when the first answer is
        requested, not at app startup. A no-op for the remote source
        (``supervisor`` is ``None``).
        """
        if self._supervisor is not None:
            await self._supervisor.ensure_running()

    def attach_supervisor(self, supervisor: LlamaServerSupervisor | None) -> None:
        """Swap the lazy-start supervisor after a settings rebind.

        The local port is fixed (:data:`~vesta.inference.local.DEFAULT_PORT`),
        so the base URL never changes — only which supervised child (with
        which baked command line) ``_ensure_ready`` may spawn.
        """
        self._supervisor = supervisor

    def attach_runtime(self, runtime: LlmRuntime | None) -> None:
        """Attach or swap the bound LLM runtime for in-flight tracking (I5)."""
        self._runtime = runtime

    def _resolve_runtime(self) -> Any:
        if self._runtime is not None:
            return self._runtime
        try:
            from vesta.inference import get_runtime

            return get_runtime()
        except Exception:
            return None

    def reconfigure(
        self,
        *,
        base_url: str,
        api_key: str,
        supervisor: LlamaServerSupervisor | None = None,
        runtime: LlmRuntime | None = None,
        timeout: float | None = None,
    ) -> None:
        """Update gateway configuration (base URL, API key, supervisor, runtime).

        Rebuilds the underlying ``AsyncOpenAI`` client so requests route to
        the new endpoint with the new authentication.
        """
        if timeout is not None:
            self._timeout = timeout
        self._base_url = base_url
        self._api_key = api_key
        self._supervisor = supervisor
        if runtime is not None:
            self._runtime = runtime
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "not-set",
            timeout=self._timeout,
        )

    async def chat_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 300,
        enable_thinking: bool | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[ChatDelta]:
        """Stream chat completions as :class:`ChatDelta` chunks.

        Maps the OpenAI streaming shape to our ``ChatDelta`` so callers never
        import the OpenAI SDK. A mid-stream crash (killed ``llama-server``,
        remote timeout) raises; the answer strategy catches it and emits a clean
        ``error`` SSE event + triggers a restart.

        ``enable_thinking`` is forwarded as ``chat_template_kwargs`` in the
        request body (Qwen3/Qwen3.5 reasoning models think by default and burn
        ``max_tokens`` on hidden reasoning). ``None`` omits
        it entirely.
        """
        await self._ensure_ready()
        rt = self._resolve_runtime()
        ctx = (
            rt.in_flight()
            if (rt is not None and hasattr(rt, "in_flight"))
            else contextlib.nullcontext()
        )
        with ctx:
            openai_messages = [{"role": m.role, "content": m.content} for m in messages]
            kwargs: dict[str, Any] = {"stream_options": {"include_usage": True}}
            if enable_thinking is not None:
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": enable_thinking}
                }
            if timeout is not None:
                kwargs["timeout"] = timeout
            stream = await self._client.chat.completions.create(
                model=model,
                messages=cast("Any", openai_messages),
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )
            # The final chunk (choices empty, usage populated) carries the token
            # counts when include_usage is honoured; yield it so downstream token
            # accounting sees it. Endpoints that ignore stream_options simply never
            # populate usage — all deltas stay at 0, which is correct.
            # ``last_finish`` is carried onto the usage delta: answer strategies
            # read ``.finish_reason`` from the last delta to detect a
            # ``length``-truncated answer, and the usage chunk (empty choices,
            # finish None) must not overwrite the content chunk's real finish.
            last_finish: str | None = None
            async for chunk in stream:
                if rt is not None and hasattr(rt, "mark_used"):
                    rt.mark_used()
                usage = _extract_usage(chunk)
                if not chunk.choices:
                    if usage[2] > 0:
                        yield ChatDelta(
                            text="",
                            finish_reason=last_finish,
                            input_tokens=usage[0],
                            output_tokens=usage[1],
                            total_tokens=usage[2],
                        )
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                text = delta.content if delta and delta.content else ""
                finish = choice.finish_reason
                last_finish = finish
                yield ChatDelta(
                    text=text,
                    finish_reason=finish,
                    input_tokens=usage[0],
                    output_tokens=usage[1],
                    total_tokens=usage[2],
                )

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
        """Non-streaming chat — short calls (rewriter, benchmark).

        ``enable_thinking`` behaves exactly as in :meth:`chat_stream` (forwarded
        as ``chat_template_kwargs``; ``None`` omits it).
        """
        await self._ensure_ready()
        rt = self._resolve_runtime()
        ctx = (
            rt.in_flight()
            if (rt is not None and hasattr(rt, "in_flight"))
            else contextlib.nullcontext()
        )
        with ctx:
            openai_messages = [{"role": m.role, "content": m.content} for m in messages]
            kwargs: dict[str, Any] = {}
            if enable_thinking is not None:
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": enable_thinking}
                }
            if timeout is not None:
                kwargs["timeout"] = timeout
            t0 = time.perf_counter()
            resp = await self._client.chat.completions.create(
                model=model,
                messages=cast("Any", openai_messages),
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                **kwargs,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if rt is not None and hasattr(rt, "mark_used"):
                rt.mark_used()
            text = ""
            finish: str | None = None
            choices = _extract_choices(resp)
            if choices:
                first = choices[0]
                if isinstance(first, dict):
                    msg = first.get("message") or {}
                    text = str(msg.get("content") or "")
                    finish = first.get("finish_reason")
                else:
                    text = str(first.message.content or "")
                    finish = first.finish_reason
            in_tok, out_tok, total_tok = _extract_usage(resp)
            return ChatResult(
                text=text,
                finish_reason=finish,
                latency_ms=latency_ms,
                input_tokens=in_tok,
                output_tokens=out_tok,
                total_tokens=total_tok,
            )

    async def aclose(self) -> None:
        """Close the underlying ``AsyncOpenAI`` (httpx) client.

        Without this, every app instance (each test, each restart) leaks the
        client's open connection once any real chat call has been made — any
        real chat call (e.g. the conversational rewriter or an answer) is what
        surfaces this as an intermittent "unclosed socket"/"unclosed transport"
        teardown warning in the test suite.
        """
        await self._client.close()


class NullGateway:
    """A no-op gateway for when no LLM is configured (degrade-don't-fail).

    ``chat_stream`` raises :class:`NoLLMConfigured`; callers that depend on an
    LLM are capability-dropped before reaching this. ``sources_only`` never
    touches the gateway and works with ``NullGateway`` bound.
    """

    async def chat_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 300,
        enable_thinking: bool | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[ChatDelta]:
        raise NoLLMConfigured("no LLM gateway is configured")
        yield  # pragma: no cover — makes this an async generator for the type checker

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
        raise NoLLMConfigured("no LLM gateway is configured")

    async def aclose(self) -> None:
        return None


class UsageRecorder:
    """Transparent :class:`Gateway` proxy that accumulates token usage.

    Used by the benchmark to attribute per-question token costs across every
    LLM call the answer pipeline makes (generation, reformulation, rewrite,
    tool decisions) without modifying the production strategies. All calls
    delegate to the inner gateway; token counts accumulate on
    :attr:`input_tokens` / :attr:`output_tokens`. Call :meth:`reset` before a
    question and read after.

    The accumulation is safe under asyncio: ``+=`` on an int contains no
    ``await`` so the event loop never preempts it, and each :meth:`reset` →
    read pair brackets one :meth:`run_one` call with no concurrent caller
    (pipeline concurrency is 1 by design).
    """

    def __init__(self, inner: Gateway) -> None:
        self._inner = inner
        self.input_tokens = 0
        self.output_tokens = 0

    def reset(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    async def chat_stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 300,
        enable_thinking: bool | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[ChatDelta]:
        async for delta in self._inner.chat_stream(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            timeout=timeout,
        ):
            if delta.has_usage:
                self.input_tokens += delta.input_tokens
                self.output_tokens += delta.output_tokens
            yield delta

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
        res = await self._inner.chat_once(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            timeout=timeout,
        )
        self.input_tokens += res.input_tokens
        self.output_tokens += res.output_tokens
        return res

    async def aclose(self) -> None:
        await self._inner.aclose()


class NoLLMConfigured(RuntimeError):
    """Raised when an LLM call is attempted with no gateway configured."""


__all__ = [
    "ChatDelta",
    "ChatMessage",
    "ChatResult",
    "Gateway",
    "NoLLMConfigured",
    "NullGateway",
    "OpenAIGateway",
    "UsageRecorder",
]
