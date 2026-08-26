"""``answer.tokens.estimate_tokens_for_chars`` safety properties.

The estimator backs window-budget arithmetic: an under-estimate is
a hard context-window 400 and a discarded transcript, so these tests pin the
floor and rounding direction, not just the arithmetic. The embedded
calibration pairs are a subset of the real calibration set (one-shot turns:
exact rebuilt request chars vs endpoint-reported input tokens) — the full set
lives in the calibration artifact, this subset keeps the shipped invariant
executable.
"""

from __future__ import annotations

import math

import pytest

from vesta.answer.tokens import CHARS_PER_TOKEN, estimate_tokens_for_chars

#: (request chars, endpoint-reported input tokens) from run 46 one-shot turns
#: (``phase19-4-off``, qwen3.5-4b@q4_k_s on the pinned endpoint). Chars are
#: system prompt + pre-seeded user message rebuilt via the production
_CALIBRATION_PAIRS: list[tuple[int, int]] = [
    (13_432, 4_421),  # densest observed (ratio 3.038) — the floor anchor
    (15_132, 4_855),
    (14_810, 4_548),
    (15_088, 4_600),
    (14_965, 4_478),
    (15_354, 3_705),
    (15_555, 3_567),  # sparsest observed (ratio 4.361)
    (14_126, 3_584),
]


@pytest.mark.parametrize(
    "n",
    [0, 1, 2, 3, 7, 100, 3_333, 90_000],
)
def test_estimate_tokens_for_chars_matches_ceil_ratio(n: int) -> None:
    """The estimate is exactly ceil(chars / ratio) and non-positive chars are 0."""
    expected = 0 if n == 0 else math.ceil(n / CHARS_PER_TOKEN)
    assert estimate_tokens_for_chars(n) == expected
    if n > 0:
        assert estimate_tokens_for_chars(n) >= 1


def test_never_under_estimates_calibration_envelope() -> None:
    """Acceptance check: ratio is floored at calibration minimum and never under-estimates."""
    observed_min = min(c / t for c, t in _CALIBRATION_PAIRS if t > 0)
    assert observed_min >= CHARS_PER_TOKEN

    for chars, tokens in _CALIBRATION_PAIRS:
        assert estimate_tokens_for_chars(chars) >= tokens, (chars, tokens)


def test_monotone_non_decreasing() -> None:
    prev = 0
    for n in range(0, 2_000, 97):
        est = estimate_tokens_for_chars(n)
        assert est >= prev
        prev = est
