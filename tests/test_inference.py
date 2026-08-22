"""Tests for the inference gateway and local llama-server supervisor (07 spec).

The gateway must not require a real OpenAI endpoint; the supervisor must not
require a real ``llama-server`` binary. Tests mock both.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vesta.inference.gateway import (
    ChatMessage,
    NoLLMConfigured,
    NullGateway,
    OpenAIGateway,
)
from vesta.inference.local import (
    BinaryMissing,
    LlamaServerSupervisor,
)


class TestNullGateway:
    @pytest.mark.asyncio
    async def test_chat_stream_raises(self) -> None:
        gw = NullGateway()
        with pytest.raises(NoLLMConfigured):
            async for _ in gw.chat_stream([ChatMessage(role="user", content="hi")], model="test"):
                pass

    @pytest.mark.asyncio
    async def test_chat_once_raises(self) -> None:
        gw = NullGateway()
        with pytest.raises(NoLLMConfigured):
            await gw.chat_once([ChatMessage(role="user", content="hi")], model="test")

    @pytest.mark.asyncio
    async def test_aclose_is_a_noop(self) -> None:
        """NullGateway never opens a connection, so closing it is a no-op —
        but the method must exist (the Gateway protocol requires it)."""
        await NullGateway().aclose()


class TestOpenAIGateway:
    @pytest.mark.asyncio
    async def test_aclose_closes_underlying_client(self) -> None:
        """Regression (found via live verification): the gateway's
        underlying AsyncOpenAI/httpx client was never closed anywhere, so any
        app instance that made one real chat call (e.g. the startup
        capability probe) leaked an open connection until GC —
        surfacing as an intermittent 'unclosed socket' teardown warning in the
        test suite. ``aclose`` must close the real client."""
        gw = OpenAIGateway(base_url="http://localhost:8081/v1", api_key="test")
        with patch.object(gw._client, "close", new_callable=AsyncMock) as mock_close:
            await gw.aclose()
        mock_close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_chat_stream_yields_deltas(self) -> None:
        """The gateway maps OpenAI chunks to ChatDelta objects."""
        gw = OpenAIGateway(base_url="http://localhost:8081/v1", api_key="test")

        # Mock the OpenAI client's create method.
        mock_chunks = [
            _mock_chunk("Hello", None),
            _mock_chunk(" world", "stop"),
        ]

        mock_stream = AsyncMock()
        mock_stream.__aiter__ = MockAsyncIterator(mock_chunks)

        with patch.object(
            gw._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_stream
            deltas = []
            async for d in gw.chat_stream(
                [ChatMessage(role="user", content="hi")], model="test-model"
            ):
                deltas.append(d)

        assert len(deltas) == 2
        assert deltas[0].text == "Hello"
        assert deltas[0].finish_reason is None
        assert deltas[1].text == " world"
        assert deltas[1].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_chat_once_returns_result(self) -> None:
        gw = OpenAIGateway(base_url="http://localhost:8081/v1", api_key="test")

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Test answer"
        mock_resp.choices[0].finish_reason = "stop"

        with patch.object(
            gw._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_resp
            result = await gw.chat_once(
                [ChatMessage(role="user", content="hi")], model="test-model"
            )

        assert result.text == "Test answer"
        assert result.finish_reason == "stop"
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_chat_once_unwraps_data_envelope(self) -> None:
        """Some OpenAI-compatible proxies (observed: api.cline.bot) wrap the
        completion in ``{"data": {...}, "success": true}``. The SDK then leaves
        ``choices`` empty and stashes the payload in ``model_extra`` — without
        unwrapping, chat_once silently returns empty text and judge verdicts
        fall back to lexical ("not judged")."""
        gw = OpenAIGateway(base_url="http://localhost:8081/v1", api_key="test")

        mock_resp = MagicMock()
        mock_resp.choices = None
        mock_resp.model_extra = {
            "data": {
                "choices": [{"message": {"content": "correct | matches"}, "finish_reason": "stop"}]
            },
            "success": True,
        }

        with patch.object(
            gw._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_resp
            result = await gw.chat_once(
                [ChatMessage(role="user", content="judge this")], model="test-model"
            )

        assert result.text == "correct | matches"
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_chat_stream_forwards_enable_thinking(self) -> None:
        """Qwen3/Qwen3.5 reasoning models think by default; the gateway must
        forward ``chat_template_kwargs.enable_thinking`` via ``extra_body`` when
        told to, and omit the parameter entirely when left unset (None) so the
        model's default applies."""
        gw = OpenAIGateway(base_url="http://localhost:8081/v1", api_key="test")
        mock_stream = AsyncMock()
        mock_stream.__aiter__ = MockAsyncIterator([_mock_chunk("hi", "stop")])
        messages = [ChatMessage(role="user", content="hi")]

        with patch.object(
            gw._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_stream
            async for _ in gw.chat_stream(messages, model="test-model", enable_thinking=False):
                pass
        _, kwargs = mock_create.call_args
        assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}

        with patch.object(
            gw._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_stream
            async for _ in gw.chat_stream(messages, model="test-model", enable_thinking=True):
                pass
        _, kwargs = mock_create.call_args
        assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}

        # Unset (None) → no extra_body at all.
        with patch.object(
            gw._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_stream
            async for _ in gw.chat_stream(messages, model="test-model"):
                pass
        _, kwargs = mock_create.call_args
        assert "extra_body" not in kwargs

    @pytest.mark.asyncio
    async def test_chat_once_forwards_enable_thinking(self) -> None:
        """Same forwarding contract for the non-streaming path."""
        gw = OpenAIGateway(base_url="http://localhost:8081/v1", api_key="test")
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Test answer"
        mock_resp.choices[0].finish_reason = "stop"

        with patch.object(
            gw._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_resp
            await gw.chat_once(
                [ChatMessage(role="user", content="hi")],
                model="test-model",
                enable_thinking=False,
            )
        _, kwargs = mock_create.call_args
        assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}

        with patch.object(
            gw._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_resp
            await gw.chat_once([ChatMessage(role="user", content="hi")], model="test-model")
        _, kwargs = mock_create.call_args
        assert "extra_body" not in kwargs

    @pytest.mark.asyncio
    async def test_chat_stream_ensure_ready_called_for_local(self) -> None:
        """For local source, the supervisor's ensure_running is called first."""
        mock_supervisor = MagicMock()
        mock_supervisor.ensure_running = AsyncMock()

        gw = OpenAIGateway(
            base_url="http://127.0.0.1:8081/v1",
            api_key="local",
            supervisor=mock_supervisor,  # type: ignore[arg-type]
        )

        mock_stream = MockAsyncIterator([])
        with patch.object(
            gw._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_stream
            async for _ in gw.chat_stream([ChatMessage(role="user", content="hi")], model="test"):
                pass

        mock_supervisor.ensure_running.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_stream_no_supervisor_for_remote(self) -> None:
        """For remote source (no supervisor), ensure_ready is a no-op."""
        gw = OpenAIGateway(base_url="http://remote:1234/v1", api_key="key", supervisor=None)

        mock_stream = MockAsyncIterator([])
        with patch.object(
            gw._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_stream
            async for _ in gw.chat_stream([ChatMessage(role="user", content="hi")], model="test"):
                pass
        # No exception means no supervisor was called (which is correct).

    @pytest.mark.asyncio
    async def test_chat_once_captures_usage(self) -> None:
        """chat_once reads resp.usage into ChatResult."""
        gw = OpenAIGateway(base_url="http://localhost:8081/v1", api_key="test")
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "answer"
        mock_resp.choices[0].finish_reason = "stop"
        mock_resp.usage = MagicMock(prompt_tokens=120, completion_tokens=30, total_tokens=150)

        with patch.object(
            gw._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_resp
            result = await gw.chat_once(
                [ChatMessage(role="user", content="hi")], model="test-model"
            )
        assert result.input_tokens == 120
        assert result.output_tokens == 30
        assert result.total_tokens == 150

    @pytest.mark.asyncio
    async def test_chat_stream_captures_usage_from_final_chunk(self) -> None:
        """chat_stream yields the usage chunk (empty choices, usage set)."""
        gw = OpenAIGateway(base_url="http://localhost:8081/v1", api_key="test")
        usage_chunk = MagicMock()
        usage_chunk.choices = []
        usage_chunk.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        content_chunk = _mock_chunk("hi", "stop")
        content_chunk.usage = None  # real content chunks carry no usage
        mock_stream = MockAsyncIterator([content_chunk, usage_chunk])

        with patch.object(
            gw._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_stream
            deltas = [
                d
                async for d in gw.chat_stream(
                    [ChatMessage(role="user", content="hi")], model="test-model"
                )
            ]
        usage_deltas = [d for d in deltas if d.has_usage]
        assert len(usage_deltas) == 1
        assert usage_deltas[0].input_tokens == 100
        assert usage_deltas[0].output_tokens == 50
        assert usage_deltas[0].total_tokens == 150
        # The usage chunk must not overwrite the content chunk's finish_reason —
        # answer strategies read the last delta's finish to detect truncation.
        assert usage_deltas[0].finish_reason == "stop"
        assert deltas[-1].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_chat_once_usage_defaults_to_zero(self) -> None:
        """When the endpoint does not report usage, all fields stay 0."""
        gw = OpenAIGateway(base_url="http://localhost:8081/v1", api_key="test")
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "answer"
        mock_resp.choices[0].finish_reason = "stop"
        mock_resp.usage = None

        with patch.object(
            gw._client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_resp
            result = await gw.chat_once(
                [ChatMessage(role="user", content="hi")], model="test-model"
            )
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.total_tokens == 0


class TestUsageRecorder:
    @pytest.mark.asyncio
    async def test_accumulates_chat_once_usage(self) -> None:
        """UsageRecorder accumulates input/output from chat_once calls."""
        from vesta.inference.gateway import ChatResult, UsageRecorder

        class _FakeGateway:
            async def chat_once(self, messages, *, model, **kw):
                return ChatResult(
                    text="ok",
                    finish_reason="stop",
                    latency_ms=1.0,
                    input_tokens=100,
                    output_tokens=20,
                    total_tokens=120,
                )

            async def chat_stream(self, messages, *, model, **kw):
                return
                yield  # pragma: no cover

            async def aclose(self):
                pass

        rec = UsageRecorder(_FakeGateway())
        await rec.chat_once([ChatMessage(role="user", content="a")], model="m")
        await rec.chat_once([ChatMessage(role="user", content="b")], model="m")
        assert rec.input_tokens == 200
        assert rec.output_tokens == 40
        assert rec.total_tokens == 240

    @pytest.mark.asyncio
    async def test_reset_clears_accumulator(self) -> None:
        from vesta.inference.gateway import ChatResult, UsageRecorder

        class _FakeGateway:
            async def chat_once(self, messages, *, model, **kw):
                return ChatResult("ok", "stop", 1.0, 50, 10, 60)

            async def chat_stream(self, messages, *, model, **kw):
                return
                yield  # pragma: no cover

            async def aclose(self):
                pass

        rec = UsageRecorder(_FakeGateway())
        await rec.chat_once([ChatMessage(role="user", content="a")], model="m")
        assert rec.total_tokens == 60
        rec.reset()
        assert rec.total_tokens == 0
        assert rec.input_tokens == 0


class TestLlamaServerSupervisor:
    def test_binary_available_false_for_nonexistent(self, tmp_path: Path) -> None:
        sup = LlamaServerSupervisor(
            binary_path=str(tmp_path / "nonexistent-llama-server"),
            models_dir=tmp_path / "models",
            config_dir=tmp_path / "config",
        )
        assert not sup.binary_available()

    def test_binary_available_true_for_existing_file(self, tmp_path: Path) -> None:
        binary = tmp_path / "fake-llama-server"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        sup = LlamaServerSupervisor(
            binary_path=str(binary),
            models_dir=tmp_path / "models",
            config_dir=tmp_path / "config",
        )
        assert sup.binary_available()

    def test_base_url_uses_port(self) -> None:
        sup = LlamaServerSupervisor(
            binary_path="llama-server",
            models_dir=Path("/tmp/models"),
            config_dir=Path("/tmp/config"),
            port=9999,
        )
        assert "9999" in sup.base_url
        assert sup.base_url.startswith("http://127.0.0.1:")

    def test_writes_models_ini(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        models_dir = tmp_path / "models"
        sup = LlamaServerSupervisor(
            binary_path="llama-server",
            models_dir=models_dir,
            config_dir=config_dir,
            threads_gen=6,
            threads_prefill=8,
            context_size=16384,
        )
        sup._write_models_ini()
        ini_path = config_dir / "models.ini"
        assert ini_path.exists()
        content = ini_path.read_text()
        assert "threads = 6" in content
        assert "threads-batch = 8" in content
        assert "parallel = 1" in content
        assert "cache-reuse = 256" in content
        # isolation-tested against the real b10373 binary
        # (unknown INI keys abort startup, so these three are load-bearing).
        assert "c = 16384" in content
        assert "jinja = true" in content
        assert "reasoning-format = auto" in content

    def test_build_command_router_mode_via_no_model(self, tmp_path: Path) -> None:
        """Router mode activates by NOT specifying a model — upstream has no
        ``--router`` flag, and an unknown flag aborts ``llama-server`` startup.
        Regression: the command previously passed ``--router``, which would have
        made every local start fail against the real binary."""
        sup = LlamaServerSupervisor(
            binary_path="/usr/bin/llama-server",
            models_dir=tmp_path / "models",
            config_dir=tmp_path / "config",
        )
        cmd = sup._build_command()
        assert "--router" not in cmd
        assert "--model" not in cmd
        assert "-m" not in cmd
        assert "--models-dir" in cmd
        assert "--models-preset" in cmd
        assert "--models-max" in cmd
        assert "--sleep-idle-seconds" in cmd
        assert "--port" in cmd
        assert "--host" in cmd

    def test_build_command_omits_sleep_idle_when_zero(self, tmp_path: Path) -> None:
        """0 means "never" — the flag is OMITTED entirely, not passed as 0
        (don't pass 0 and hope upstream treats it as never)."""
        sup = LlamaServerSupervisor(
            binary_path="/usr/bin/llama-server",
            models_dir=tmp_path / "models",
            config_dir=tmp_path / "config",
            idle_unload_seconds=0,
        )
        cmd = sup._build_command()
        assert "--sleep-idle-seconds" not in cmd
        assert "0" not in cmd  # no stray bare 0 argument either

        sup_on = LlamaServerSupervisor(
            binary_path="/usr/bin/llama-server",
            models_dir=tmp_path / "models",
            config_dir=tmp_path / "config",
            idle_unload_seconds=900,
        )
        cmd_on = sup_on._build_command()
        i = cmd_on.index("--sleep-idle-seconds")
        assert cmd_on[i + 1] == "900"

    async def test_wait_for_health_fails_fast_when_stopped_concurrently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A concurrent ``stop()`` nulls ``_proc`` mid-startup — the health
        loop must abort instead of polling a dead port for the full timeout
        (found driving the real b10373 binary via LlmRuntime's watchdog)."""
        from vesta import inference
        from vesta.inference.local import LlamaServerError

        sup = LlamaServerSupervisor(
            binary_path="/usr/bin/llama-server",
            models_dir=tmp_path / "models",
            config_dir=tmp_path / "config",
        )

        class FakeProc:
            returncode: int | None = None

        proc = FakeProc()
        sup._proc = proc  # type: ignore[assignment]

        def _stop_midway(url: str) -> int:
            sup._proc = None  # exactly what stop() does before the loop notices
            return 0

        monkeypatch.setattr(inference.local, "_check_health", _stop_midway)
        with pytest.raises(LlamaServerError, match="stopped concurrently"):
            await asyncio.wait_for(sup._wait_for_health(proc), timeout=5.0)  # type: ignore[arg-type]

    def test_is_running_false_before_start_true_with_fake_proc(self, tmp_path: Path) -> None:
        sup = LlamaServerSupervisor(
            binary_path="llama-server",
            models_dir=tmp_path / "models",
            config_dir=tmp_path / "config",
        )
        assert sup.is_running() is False
        # A live-looking fake process handle: returncode None ⇒ alive.
        sup._proc = MagicMock()
        sup._proc.returncode = None
        assert sup.is_running() is True
        sup._proc.returncode = 0
        assert sup.is_running() is False

    @pytest.mark.asyncio
    async def test_ensure_running_raises_on_missing_binary(self, tmp_path: Path) -> None:
        sup = LlamaServerSupervisor(
            binary_path=str(tmp_path / "no-such-binary"),
            models_dir=tmp_path / "models",
            config_dir=tmp_path / "config",
        )
        with pytest.raises(BinaryMissing):
            await sup.ensure_running()

    @pytest.mark.asyncio
    async def test_stop_is_safe_when_not_started(self, tmp_path: Path) -> None:
        sup = LlamaServerSupervisor(
            binary_path="llama-server",
            models_dir=tmp_path / "models",
            config_dir=tmp_path / "config",
        )
        # Should not raise even if nothing was started.
        await sup.stop()

    @pytest.mark.asyncio
    async def test_ensure_running_no_deadlock_with_restart_task(self, tmp_path: Path) -> None:
        """Regression: ``ensure_running`` previously held ``self._lock`` while
        awaiting ``_restart_task``, but ``_restart_with_backoff`` needs the same
        lock — a deadlock (asyncio.Lock is not reentrant). The fix awaits the
        restart task OUTSIDE the lock."""
        import asyncio

        binary = tmp_path / "fake-llama-server"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)

        sup = LlamaServerSupervisor(
            binary_path=str(binary),
            models_dir=tmp_path / "models",
            config_dir=tmp_path / "config",
        )

        # Simulate a crash: plant a fake restart task that needs the lock.
        restart_done = asyncio.Event()

        async def fake_restart() -> None:
            async with sup._lock:
                sup._crashed = False
                restart_done.set()

        sup._restart_task = asyncio.create_task(fake_restart())

        # Mock _start_and_wait so we don't actually spawn a subprocess.
        started = False

        async def fake_start() -> None:
            nonlocal started
            started = True

        sup._start_and_wait = fake_start  # type: ignore[method-assign]

        # ensure_running must complete without deadlock.
        await asyncio.wait_for(sup.ensure_running(), timeout=5.0)

        # If we reach here, no deadlock occurred. The restart task finished
        # and _start_and_wait was called.
        assert restart_done.is_set()
        assert started


class TestInferenceCapabilityProbe:
    def test_probe_returns_empty_without_gateway(self) -> None:
        from vesta.config.capabilities import Capability

        # Without a bound gateway, the probe returns empty.
        from vesta.inference import _capability_probe, bind_gateway

        bind_gateway(None, None)
        caps = _capability_probe()
        assert Capability.LLM not in caps

    def test_probe_returns_llm_for_remote_configured(self) -> None:
        from vesta import config
        from vesta.config.capabilities import Capability
        from vesta.inference import _capability_probe, bind_gateway
        from vesta.inference.gateway import NullGateway

        config.configure()
        # Set remote endpoint + model.
        config.set_db_values(
            {
                "inference.llm.source": "remote",
                "inference.llm.endpoint_url": "http://test:1234/v1",
                "inference.llm.model": "test-model",
            }
        )
        bind_gateway(NullGateway(), None)  # type: ignore[arg-type]
        try:
            caps = _capability_probe()
            assert Capability.LLM in caps
        finally:
            bind_gateway(None, None)
            config.reset_for_test()

    def test_probe_local_requires_model_file_on_disk(self, tmp_path: Path) -> None:
        """D5: local probe = binary present AND the configured GGUF exists."""
        from vesta import config
        from vesta.config.capabilities import Capability
        from vesta.inference import _capability_probe, bind_gateway, bind_models_dir
        from vesta.inference.gateway import NullGateway

        binary = tmp_path / "fake-llama-server"
        binary.write_text("#!/bin/sh\n")
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "some-model.gguf").write_bytes(b"gguf")

        config.configure()
        config.set_db_values(
            {
                "inference.llm.source": "local",
                "inference.llm.model": "some-model.gguf",
                "inference.local.binary_path": str(binary),
            }
        )
        bind_models_dir(models_dir)
        bind_gateway(NullGateway(), None)  # type: ignore[arg-type]
        try:
            # File present → capability on.
            assert Capability.LLM in _capability_probe()
            # File gone → capability off (degrade to sources_only).
            (models_dir / "some-model.gguf").unlink()
            assert Capability.LLM not in _capability_probe()
        finally:
            bind_gateway(None, None)
            bind_models_dir(None)
            config.reset_for_test()


# ── Helpers ─────────────────────────────────────────────────────────────────


def _mock_chunk(text: str, finish: str | None) -> MagicMock:
    """Create a mock OpenAI streaming chunk."""
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = text
    chunk.choices[0].finish_reason = finish
    return chunk


class MockAsyncIterator:
    """A mock async iterator over a list of items."""

    def __init__(self, items: list) -> None:
        self._items = list(items)

    def __call__(self, *args, **kwargs) -> MockAsyncIterator:
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)
