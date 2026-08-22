"""Unit tests for the context-overflow ``ModelHTTPError`` matcher in ``agent_chat``.

``run_one_turn`` must recover from a context-window overflow (a hard 400 from the
model endpoint once the accumulated tool-call history exceeds the answer model's
context) the same way it recovers from ``UsageLimitExceeded`` — but it must NOT
swallow an unrelated 400/401/500, which is a real bug that should stay loud. These
tests exercise only the narrow matching predicate, ``_is_context_overflow_error``,
without needing a live model or the retrieval pipeline.
"""

from __future__ import annotations

import pytest
from pydantic_ai.exceptions import ModelHTTPError

from vesta.api.agent_chat import _is_context_overflow_error


def _http_error(status_code: int, body: object) -> ModelHTTPError:
    return ModelHTTPError(
        status_code=status_code, model_name="lmstudio/unsloth/qwen3.5-4b", body=body
    )


@pytest.mark.parametrize(
    "body",
    [
        {
            "message": (
                "litellm.BadRequestError: OpenAIException - Error code: 400 - "
                "{'error': 'Engine protocol predict stream returned an error: "
                '{"code":500,"message":"Context size has been exceeded.",'
                '"type":"server_error"}\'}'
            )
        },
        {"error": "This model's maximum context length is 8192 tokens, and you exceeded it."},
        {"error": "Request exceeded the context window for this model."},
        {"error": "CONTEXT SIZE HAS BEEN EXCEEDED."},
        "context length exceeded",
    ],
)
def test_context_overflow_recognized(body: object) -> None:
    """Real-world and near-variant context-overflow bodies must be treated as recoverable."""
    assert _is_context_overflow_error(_http_error(400, body)) is True


@pytest.mark.parametrize(
    ("status_code", "body"),
    [
        (400, {"error": "Invalid value for 'temperature': must be between 0 and 2."}),
        (401, {"error": "Invalid API key provided."}),
        (500, {"error": "Internal server error"}),
        (400, {"error": "Rate limit exceeded, please retry later."}),
        (400, {"error": "The context window setting is misconfigured."}),
        (400, None),
        (402, {"error": "Context size has been exceeded."}),
        (403, {"error": "Context size has been exceeded."}),
        (404, {"error": "Context size has been exceeded."}),
        (429, {"error": "Context size has been exceeded."}),
        (502, {"error": "Context size has been exceeded."}),
        (503, {"error": "Context size has been exceeded."}),
    ],
)
def test_unrelated_errors_rejected(status_code: int, body: object) -> None:
    """Unrelated errors and non-400 status codes must not be treated as context overflow."""
    assert _is_context_overflow_error(_http_error(status_code, body)) is False
