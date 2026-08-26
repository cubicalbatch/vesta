"""Regression tests for the ``normalize`` query preparer's term extraction.

Covers AUDIT_0824 N39: the primary path must drop libzim-inert boolean/operator
tokens (``vesta.zim.query.INERT_TOKENS``) exactly like the ladder's
``normalize_terms``, so "hotels near the grand canyon" does not AND a literal
``near`` into every hit — while queries without such tokens keep byte-identical
term lists.
"""

from __future__ import annotations

import pytest

from vesta.retrieval.contracts import PreparedQuery
from vesta.retrieval.impls.normalize import Normalize
from vesta.retrieval.trace import Trace


@pytest.fixture
def preparer() -> Normalize:
    return Normalize()


def _raw_query(raw: str) -> PreparedQuery:
    """Bootstrap pipeline input: raw string only, everything else empty."""
    return PreparedQuery(
        raw=raw,
        terms=(),
        text=raw,
        aliases=(),
        is_keyword_query=False,
        rung="initial",
    )


async def test_inert_boolean_word_not_a_term(preparer: Normalize) -> None:
    """'near' must not survive as an AND-ed term on the primary path."""
    q = await preparer.prepare(_raw_query("hotels near the grand canyon"), Trace())
    assert q.terms == ("hotels", "grand", "canyon")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("what is photosynthesis not artificial", ("photosynthesis", "artificial")),
        ("restaurants xor cafes", ("restaurants", "cafes")),
    ],
)
async def test_each_inert_word_dropped(
    preparer: Normalize, raw: str, expected: tuple[str, ...]
) -> None:
    q = await preparer.prepare(_raw_query(raw), Trace())
    assert q.terms == expected


async def test_clean_query_terms_unchanged(preparer: Normalize) -> None:
    """A plain query keeps the exact pre-change term list."""
    q = await preparer.prepare(_raw_query("How Do I Mount a USB Drive?"), Trace())
    assert q.terms == ("mount", "usb", "drive")


async def test_all_inert_query_keeps_something(preparer: Normalize) -> None:
    """A query made entirely of inert tokens still yields terms."""
    q = await preparer.prepare(_raw_query("not near xor"), Trace())
    assert q.terms  # never stripped down to nothing
