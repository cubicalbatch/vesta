"""Tests for ``retrieval/snippets.py`` and ``retrieval/dedup.py``."""

from __future__ import annotations

from vesta.retrieval.contracts import ScoredPassage
from vesta.retrieval.dedup import is_near_duplicate
from vesta.retrieval.snippets import build_snippet
from vesta.zim.types import Passage


def _passage(text: str, *, path: str = "A") -> Passage:
    return Passage(
        zim_id=1,
        path=path,
        ordinal=0,
        char_start=0,
        char_end=len(text),
        breadcrumb="",
        text=text,
        is_lead=False,
    )


def _sp(text: str, *, path: str = "A", score: float = 1.0) -> ScoredPassage:
    return ScoredPassage(passage=_passage(text, path=path), score=score, source_info="x")


# ── snippets.build_snippet ───────────────────────────────────────────────────


def test_snippet_returns_full_text_when_shorter_than_window() -> None:
    assert build_snippet("short passage", ["short"]) == "short passage"


def test_snippet_empty_text_returns_empty() -> None:
    assert build_snippet("", ["term"]) == ""


def test_snippet_centres_on_earliest_term_match() -> None:
    lead = "x " * 200
    text = f"{lead}the target phrase appears here {'y ' * 200}"
    snippet = build_snippet(text, ["target"], window_chars=60)
    assert "target phrase" in snippet
    assert snippet.startswith("…")  # truncated at the front
    assert snippet.endswith("…")  # truncated at the back


def test_snippet_falls_back_to_lead_window_when_no_term_matches() -> None:
    text = "a " * 300
    snippet = build_snippet(text, ["nonexistent"], window_chars=50)
    assert text.strip().startswith(snippet.rstrip("…").strip())
    assert snippet.endswith("…")


def test_snippet_ignores_very_short_terms() -> None:
    # "of" is too short to anchor a snippet; falls back to the lead window.
    text = "Introduction paragraph. " + ("filler word " * 100) + "of interest is buried far in."
    snippet = build_snippet(text, ["of"], window_chars=40)
    assert snippet.startswith("Introduction")


def test_snippet_never_exceeds_window_by_much() -> None:
    text = "word " * 500 + "needle" + " word" * 500
    snippet = build_snippet(text, ["needle"], window_chars=100)
    # generous slack for word-boundary extension
    assert len(snippet) < 150


# ── dedup.is_near_duplicate ──────────────────────────────────────────────────


def test_identical_text_is_near_duplicate() -> None:
    a = _sp("The quick brown fox jumps over the lazy dog near the river bank today")
    b = _sp("The quick brown fox jumps over the lazy dog near the river bank today", path="B")
    assert is_near_duplicate(a, [b]) is True


def test_unrelated_text_is_not_near_duplicate() -> None:
    a = _sp("The quick brown fox jumps over the lazy dog near the river bank today")
    b = _sp("Quantum computing relies on superposition and entanglement of qubits", path="B")
    assert is_near_duplicate(a, [b]) is False


def test_near_duplicate_ignores_zim_id_catches_cross_archive() -> None:
    """The check never looks at zim_id — this IS the cross-archive-mirror case
    the module docstring describes."""
    text = "Albert Einstein developed the theory of general relativity in 1915"
    a_passage = Passage(
        zim_id=1,
        path="Einstein",
        ordinal=0,
        char_start=0,
        char_end=len(text),
        breadcrumb="",
        text=text,
        is_lead=False,
    )
    b_passage = Passage(
        zim_id=2,
        path="Einstein_mirror",
        ordinal=0,
        char_start=0,
        char_end=len(text),
        breadcrumb="",
        text=text,
        is_lead=False,
    )
    a = ScoredPassage(passage=a_passage, score=1.0, source_info="x")
    b = ScoredPassage(passage=b_passage, score=1.0, source_info="x")
    assert is_near_duplicate(a, [b]) is True


def test_short_passages_never_flagged_duplicate() -> None:
    a = _sp("hi there")
    b = _sp("hi there", path="B")
    assert is_near_duplicate(a, [b]) is False
