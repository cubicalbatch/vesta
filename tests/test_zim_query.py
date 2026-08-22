"""Query preprocessing ladder — the difference between a working and a
broken search box.

libzim passes only ``FLAG_CJK_NGRAM`` to ``parse_query`` and sets
``default_op = AND``, so a natural-language question like
``"how do i mount a usb drive"`` returns **0 results**. The ladder fixes it:
all-terms → stopword-stripped → OR-of-terms → title. These tests use a fake
AND-semantics corpus (no real ZIM) so the rung selection logic is exercised
deterministically and fast.
"""

from __future__ import annotations

import pytest

from vesta.retrieval.trace import Trace
from vesta.zim.query import QueryPreparer, SearchFn, SuggestFn, normalize_terms

DEFAULT_STOP = (
    "a,an,the,and,or,of,to,in,on,at,by,for,with,from,into,is,are,was,were,be,"
    "been,being,do,does,did,doing,have,has,had,i,you,he,she,it,we,they,this,that,"
    "these,those,my,your,me,him,her,us,them,how,what,why,who,whom,when,where,which,"
    "whose,explain,tell,about,there,here"
)


def _make(corpus: dict[str, list[str]]) -> tuple[SearchFn, SuggestFn]:
    """Build (search, suggest) fakes implementing libzim's AND semantics."""

    async def search(term_set: tuple[str, ...], limit: int) -> list[str]:
        ts = set(term_set)
        return [p for p, terms in corpus.items() if ts <= set(terms)][:limit]

    async def suggest(prefix: str, limit: int) -> list[str]:
        pl = prefix.lower()
        return [p for p in corpus if p.lower().startswith(pl)][:limit]

    return search, suggest


@pytest.fixture
def preparer() -> QueryPreparer:
    return QueryPreparer.from_settings(
        stopword_stripping=True, stopword_list=DEFAULT_STOP, ladder_enabled=True
    )


def test_normalize_strips_inert_syntax_and_punctuation() -> None:
    # Quotes / booleans / +/- / * are inert under libzim — dropped, not
    # sent as literal AND-ed terms that would force a zero-result.
    assert normalize_terms('"mount usb" AND -drive *') == ["mount", "usb", "drive"]
    assert normalize_terms("How Do I Mount a USB Drive?") == [
        "how",
        "do",
        "i",
        "mount",
        "usb",
        "drive",
    ]


async def test_natural_language_question_succeeds_via_stopword_rung(
    preparer: QueryPreparer,
) -> None:
    """The headline trap: the raw AND returns 0; the ladder rescues it."""
    corpus = {
        "USB_flash_drive": ["usb", "flash", "drive", "mount", "storage"],
        "Optical_disc_drive": ["optical", "disc", "drive"],
        "Mount_Everest": ["mount", "everest", "mountain"],
    }
    search, suggest = _make(corpus)

    # Raw all-terms AND over the *full* NL question matches nothing.
    raw = await search(("how", "do", "i", "mount", "usb", "drive"), 10)
    assert raw == []

    trace = Trace()
    hits = await preparer.execute(
        "how do i mount a usb drive", search, suggest, limit=10, trace=trace
    )
    assert hits
    assert "USB_flash_drive" in hits

    # The trace must show WHICH rung produced the result.
    stage = trace.to_dict()["stages"][0]
    assert stage["outputs"]["chosen_rung"] == "stopword_stripped"
    assert stage["outputs"]["rung.all_terms.hits"] == 0
    assert stage["outputs"]["rung.stopword_stripped.hits"] >= 1


async def test_or_of_terms_rung_used_when_no_conjunction_matches(preparer: QueryPreparer) -> None:
    # No single entry has BOTH "everest" and "storage"; the OR rung union finds
    # each by a single term.
    corpus = {
        "Mount_Everest": ["mount", "everest"],
        "Cloud_storage": ["cloud", "storage"],
    }
    search, suggest = _make(corpus)
    trace = Trace()
    hits = await preparer.execute("mount everest storage", search, suggest, limit=10, trace=trace)
    assert set(hits) == {"Mount_Everest", "Cloud_storage"}
    assert trace.to_dict()["stages"][0]["outputs"]["chosen_rung"] == "or_of_terms"


async def test_title_rung_is_the_universal_fallback(preparer: QueryPreparer) -> None:
    """When no fulltext rung matches, the title/suggestion index resolves it."""
    # Body terms deliberately exclude the title-ish prefix so every fulltext
    # rung (all-terms AND, or-of-terms) misses; the title prefix rung catches it.
    corpus: dict[str, list[str]] = {"Photosynthesis": ["plants", "chlorophyll"]}
    search, suggest = _make(corpus)
    trace = Trace()
    hits = await preparer.execute("photosyn", search, suggest, limit=10, trace=trace)
    assert hits == ["Photosynthesis"]
    assert trace.to_dict()["stages"][0]["outputs"]["chosen_rung"] == "title"


async def test_all_terms_rung_short_circuits_when_it_already_matches(
    preparer: QueryPreparer,
) -> None:
    corpus = {"Carbon_dioxide": ["carbon", "dioxide", "gas"]}
    search, suggest = _make(corpus)
    trace = Trace()
    hits = await preparer.execute("carbon dioxide", search, suggest, limit=10, trace=trace)
    assert hits == ["Carbon_dioxide"]
    assert trace.to_dict()["stages"][0]["outputs"]["chosen_rung"] == "all_terms"


async def test_ladder_disabled_keeps_only_all_terms(preparer: QueryPreparer) -> None:
    disabled = QueryPreparer.from_settings(
        stopword_stripping=True, stopword_list=DEFAULT_STOP, ladder_enabled=False
    )
    corpus = {"USB_flash_drive": ["usb", "flash", "drive", "mount"]}
    search, suggest = _make(corpus)
    # Only the all-terms rung runs; the NL question returns nothing (no fallback).
    hits = await disabled.execute("how do i mount a usb drive", search, suggest, limit=10)
    assert hits == []


async def test_execute_runs_without_a_trace(preparer: QueryPreparer) -> None:
    """The default tracer is a no-op; callers may pass nothing (self-contained)."""
    corpus = {"Carbon_dioxide": ["carbon", "dioxide"]}
    search, suggest = _make(corpus)
    hits = await preparer.execute("carbon dioxide", search, suggest, limit=5)
    assert hits == ["Carbon_dioxide"]
