"""Tests for ``retrieval/snippets.py`` and ``retrieval/dedup.py``."""

from __future__ import annotations

import random
from itertools import pairwise

import pytest

from vesta.retrieval.contracts import ScoredPassage
from vesta.retrieval.dedup import DEFAULT_THRESHOLD, NearDuplicateGate, is_near_duplicate
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


# ── dedup.NearDuplicateGate ──────────────────────────────────────────────────
#
# The gate is the incremental form of ``is_near_duplicate`` the assembler
# selection loops use: each accepted passage's bigram set is computed once and
# reused, instead of re-tokenizing the whole selection per candidate. Verdicts
# must be indistinguishable from the one-shot function.


def test_gate_mirrors_function_verdicts_on_fixed_cases() -> None:
    dup = "The quick brown fox jumps over the lazy dog near the river bank today"
    unrelated = "Quantum computing relies on superposition and entanglement of qubits"
    gate = NearDuplicateGate(DEFAULT_THRESHOLD)
    first = _sp(dup)
    assert gate.is_near_duplicate(first) is False
    gate.accept(first)
    second = _sp(dup, path="B")
    assert gate.is_near_duplicate(second) is is_near_duplicate(second, [first])
    third = _sp(unrelated, path="C")
    assert gate.is_near_duplicate(third) is is_near_duplicate(third, [first])


def test_gate_rejected_candidates_stay_out() -> None:
    """Selection loops only ``accept`` what they append — a flagged duplicate
    must not influence later verdicts (it never entered ``selected``)."""
    text = "The quick brown fox jumps over the lazy dog near the river bank today"
    gate = NearDuplicateGate(DEFAULT_THRESHOLD)
    original = _sp(text)
    gate.accept(original)
    dup = _sp(text, path="B")
    assert gate.is_near_duplicate(dup) is True  # would be rejected, NOT accepted
    other = _sp("Completely different content about marine biology and coral reefs", path="C")
    assert gate.is_near_duplicate(other) is False


# ── equivalence vs naive reference (AUDIT_0822 P2 invariant) ─────────────────
#
# The caching restructure must be pure: identical verdicts to the old
# re-tokenize-everything logic. The reference below IS that old logic,
# verbatim. Randomized corpora mix exact duplicates, near-duplicates straddling
# the threshold, plain passages, and short/empty texts (which hit the
# min-bigram guard and the empty-bigram skip).


_NAIVE_MIN_BIGRAMS = 4

_WORDS = [
    "time",
    "person",
    "year",
    "way",
    "day",
    "thing",
    "man",
    "world",
    "life",
    "hand",
    "part",
    "child",
    "eye",
    "woman",
    "place",
    "work",
    "week",
    "case",
    "point",
    "government",
    "company",
    "number",
    "group",
    "problem",
    "fact",
]


def _naive_bigrams(text: str) -> frozenset[tuple[str, str]]:
    return frozenset(pairwise(text.lower().split()))


def _naive_is_near_duplicate(
    candidate: ScoredPassage,
    selected: list[ScoredPassage],
    *,
    threshold: float,
) -> bool:
    cand_bg = _naive_bigrams(candidate.passage.text)
    if len(cand_bg) < _NAIVE_MIN_BIGRAMS:
        return False
    for other in selected:
        other_bg = _naive_bigrams(other.passage.text)
        if not other_bg:
            continue
        union = cand_bg | other_bg
        if not union:
            continue
        if len(cand_bg & other_bg) / len(union) > threshold:
            return True
    return False


def _random_text(rng: random.Random) -> str:
    n_words = rng.choice([0, 1, 2, 3, 5, 8, 13, 21])
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


def _base_text(rng: random.Random) -> str:
    """Duplicate-source text long enough to clear the min-bigram guard —
    shorter bases could never be flagged, by design."""
    return " ".join(rng.choice(_WORDS) for _ in range(rng.randint(6, 25)))


def _mutate(rng: random.Random, text: str) -> str:
    """One or two word swaps — lands Jaccard around the default threshold."""
    words = text.split()
    if not words:
        return text
    out = list(words)
    for _ in range(rng.choice([1, 2])):
        out[rng.randrange(len(out))] = rng.choice(_WORDS)
    return " ".join(out)


def _random_corpus(rng: random.Random) -> list[ScoredPassage]:
    corpus: list[ScoredPassage] = []
    bases = [_base_text(rng) for _ in range(6)]
    for i in range(30):
        roll = rng.random()
        if roll < 0.3:
            text = rng.choice(bases)  # exact duplicate of a base
        elif roll < 0.65:
            text = _mutate(rng, rng.choice(bases))  # near-duplicate
        else:
            text = _random_text(rng)  # plain / short / empty
        corpus.append(_sp(text, path=f"P{i}", score=round(rng.uniform(0.0, 1.0), 2)))
    # Guaranteed edge shapes regardless of the rng draws: empty text (empty
    # bigram set — hits the selected-side skip), sub-min-bigram lengths
    # (candidate-side guard), and an exact-duplicate pair so BOTH verdicts are
    # exercised even when the random rolls and the drawn threshold wouldn't
    # produce any flags.
    corpus.append(_sp("", path="EDGE_empty", score=0.5))
    corpus.append(_sp("single", path="EDGE_one", score=0.5))
    corpus.append(_sp("two words", path="EDGE_two", score=0.5))
    anchor = _base_text(rng)
    corpus.append(_sp(anchor, path="EDGE_dup_a", score=0.5))
    corpus.append(_sp(anchor, path="EDGE_dup_b", score=0.5))
    return corpus


@pytest.mark.parametrize("seed", range(20))
def test_gate_and_function_match_naive_reference_randomized(seed: int) -> None:
    rng = random.Random(seed)
    corpus = _random_corpus(rng)
    threshold = rng.choice([DEFAULT_THRESHOLD, 0.5, 0.95])

    accepted: list[ScoredPassage] = []
    gate = NearDuplicateGate(threshold)
    flags: list[bool] = []
    for sp in corpus:
        expected = _naive_is_near_duplicate(sp, accepted, threshold=threshold)
        assert is_near_duplicate(sp, accepted, threshold=threshold) is expected
        assert gate.is_near_duplicate(sp) is expected
        flags.append(expected)
        if not expected:  # selection loops accept exactly the survivors
            accepted.append(sp)
            gate.accept(sp)

    # Sanity: the randomized corpus really exercised both verdicts and the
    # edge shapes reached the selection loop.
    assert any(flags)
    assert not all(flags)
    assert any(sp.passage.text == "" for sp in accepted)
