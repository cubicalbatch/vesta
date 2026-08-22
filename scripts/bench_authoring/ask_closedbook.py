#!/usr/bin/env python3
"""Ask qwen3.5-4b a question with NO context (closed-book), to check whether the
model already knows the fact from training.

Benchmark authoring uses this to REJECT facts the model can answer without
retrieval (which would raise the floor). Print the model's answer so the author
can judge: if it gives the correct fact, the fact is TOO FAMOUS — pick a more
obscure one.

Usage: uv run python scripts/bench_authoring/ask_closedbook.py "YOUR QUESTION"
"""

from __future__ import annotations

import asyncio
import sys

from vesta import cli
from vesta.inference import (
    INFERENCE_LLM_API_KEY,
    INFERENCE_LLM_ENDPOINT_URL,
    INFERENCE_LLM_MODEL,
)
from vesta.inference.gateway import ChatMessage

OVERRIDES = {
    INFERENCE_LLM_MODEL.key: "qwen3.5-4b@q4_k_s",
    INFERENCE_LLM_ENDPOINT_URL.key: "http://desktop.onoz.cc:1234/v1",
    INFERENCE_LLM_API_KEY.key: "",
}


async def _ask(question: str) -> str:
    async with cli._open_runtime("data", with_gateway=True, settings_overrides=OVERRIDES) as state:
        res = await state.gateway.chat_once(
            [ChatMessage(role="user", content=question)],
            model="qwen3.5-4b@q4_k_s",
            temperature=0.0,
            max_tokens=256,
            enable_thinking=False,
        )
        return str(res.text)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    question = " ".join(sys.argv[1:])
    print(asyncio.run(_ask(question)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
