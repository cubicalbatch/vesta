"""Token estimation for the answer path.

A pure chars→tokens estimator used by window-budget arithmetic to decide
whether a request fits a declared context window. It deliberately makes NO
network or DB access: budget decisions must not add an HTTP round trip per
call, and the module imports nothing internal so the ``answer/`` dependency
cap (tests/test_boundaries.py) is untouched.

Calibration: ground truth is measured against the frozen vesta_bench dataset on
the reference endpoint with reported ``input_tokens``. The Round-0 request text
(system prompt + pre-seeded user message) was evaluated across single-request
(one-shot) turns, pairing exact request characters with exact endpoint prompt
tokens.

The ratio is the MINIMUM observed chars-per-token over that calibration set
(the most token-dense text), then rounded DOWN: an over-estimate costs a bit
of headroom, an under-estimate is a hard 400 and a discarded transcript.
Char counting here also ignores per-message template overhead (role markers,
tool-call envelopes), which biases every estimate further upward — the safe
direction.

The seam is swappable by design: if a future runtime exposes llama-server's
``/tokenize``, an exact counter can replace :func:`estimate_tokens` behind
the same signature without touching any caller.
"""

from __future__ import annotations

import math

#: Characters per token. Calibrated against benchmark one-shot
#: turns (see module docstring): observed chars/token min 3.038, p50 3.748,
#: max 4.361. Rounded DOWN from the observed minimum to 3.0 so estimates err
#: high, never low (a low estimate is a context-window 400). At 3.0 the
#: estimator under-estimates 0/90 calibration turns (mean over-estimate
#: ~24 %).
CHARS_PER_TOKEN = 3.0


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text`` (ceil; never below the calibration).

    Pure, allocation-free, and conservative: ``ceil(len(text) /
    CHARS_PER_TOKEN)`` with the ratio rounded down, so the estimate never
    under-states the calibration set's true counts and carries the chat-
    template overhead as free safety margin.
    """
    if not text:
        return 0
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def estimate_tokens_for_chars(chars: int) -> int:
    """Token estimate for a known character count — the same conservative
    ceil as :func:`estimate_tokens`, without materializing text. Component
    char ceilings compose exactly under it: a total bounded in chars bounds
    the single ceil of the whole (splitting only over-counts, never under).
    """
    if chars <= 0:
        return 0
    return math.ceil(chars / CHARS_PER_TOKEN)


__all__ = ["CHARS_PER_TOKEN", "estimate_tokens", "estimate_tokens_for_chars"]
