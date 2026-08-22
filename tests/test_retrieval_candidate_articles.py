"""CandidateArticles passage builder — the title-fallback rule (media/SPA ZIMs).

The builder drops candidates whose extracted body is empty (the pre-2026-08
behaviour). That silently killed EVERY candidate from a media/SPA ZIM, whose
``text/html`` entries are meta-refresh redirect stubs: the ZIM's own Xapian
fulltext returns 34 good paths for "fire", and all 34 died here. The fix is a
generic rule — empty body + real title → synthesize a title passage — which
never branches on ZIM kind. These tests pin it without a real ZIM.
"""

from __future__ import annotations

from vesta.retrieval.contracts import Candidate
from vesta.retrieval.impls.candidate_articles import CandidateArticles
from vesta.zim.types import EntryFlags, ExtractedArticle


class _StubArchive:
    """A fake Archive whose extract() always returns an empty-body article with
    a title — the media-ZIM-stub shape."""

    def __init__(self, title: str) -> None:
        self.id = 7
        self._title = title

    async def extract(self, path: str) -> ExtractedArticle:
        return ExtractedArticle(
            path=path,
            title=self._title,
            text="",  # empty body — the stub case
            sections=(),
            flags=EntryFlags.SOFT_REDIRECT,
        )


class _BodyArchive:
    """A fake Archive returning a real-body article — the normal case."""

    def __init__(self) -> None:
        self.id = 8

    async def extract(self, path: str) -> ExtractedArticle:
        return ExtractedArticle(
            path=path,
            title="Real",
            text="A real article body with enough words to split.",
            sections=(),
            flags=EntryFlags.NONE,
        )


class _Registry:
    def __init__(self, *archives: object) -> None:
        self._by_id = {a.id: a for a in archives}  # type: ignore[attr-defined]

    def get(self, zim_id: int) -> object:
        return self._by_id[zim_id]


async def test_empty_body_with_title_yields_title_passage() -> None:
    reg = _Registry(_StubArchive("Building a Winter Survival Shelter"))
    builder = CandidateArticles(CandidateArticles.Params(), archives=reg)  # type: ignore[arg-type]
    cands = [Candidate(zim_id=7, path="index/x-3QSQ", source="xapian_fts", rank=0, score=None)]
    passages = await builder.build(cands, prepared_query_stub(), trace_stub())
    assert len(passages) == 1
    p = passages[0]
    assert p.text == "Building a Winter Survival Shelter"
    assert p.breadcrumb == "Building a Winter Survival Shelter"
    assert p.is_lead


async def test_empty_body_no_title_still_dropped() -> None:
    class _NoTitle(_StubArchive):
        async def extract(self, path: str) -> ExtractedArticle:
            return ExtractedArticle(
                path=path, title="", text="", sections=(), flags=EntryFlags.SOFT_REDIRECT
            )

    reg = _Registry(_NoTitle(""))
    builder = CandidateArticles(CandidateArticles.Params(), archives=reg)  # type: ignore[arg-type]
    passages = await builder.build(
        [Candidate(zim_id=7, path="x", source="s", rank=0, score=None)],
        prepared_query_stub(),
        trace_stub(),
    )
    assert passages == []


async def test_title_fallback_disabled_restores_drop_behaviour() -> None:
    """A profile can set title_fallback=false to restore the old drop behaviour."""
    reg = _Registry(_StubArchive("T"))
    params = CandidateArticles.Params(title_fallback=False)
    builder = CandidateArticles(params, archives=reg)  # type: ignore[arg-type]
    passages = await builder.build(
        [Candidate(zim_id=7, path="x", source="s", rank=0, score=None)],
        prepared_query_stub(),
        trace_stub(),
    )
    assert passages == []


async def test_real_body_uses_split_passages_not_title_fallback() -> None:
    """A normal-body article goes through split_passages (>=1 passage whose text
    is the body, not the title)."""
    reg = _Registry(_BodyArchive())
    builder = CandidateArticles(CandidateArticles.Params(), archives=reg)  # type: ignore[arg-type]
    passages = await builder.build(
        [Candidate(zim_id=8, path="A/Foo", source="s", rank=0, score=None)],
        prepared_query_stub(),
        trace_stub(),
    )
    assert passages, "real-body article must yield passages"
    assert all("real article body" in p.text.lower() for p in passages)


# ── minimal stubs for the PreparedQuery / Trace the contract needs ───────────


def prepared_query_stub() -> object:
    class _Q:
        terms = ()
        raw = ""
        text = ""

    return _Q()


def trace_stub() -> object:
    class _Tr:
        def degraded(self, *a: object, **k: object) -> None:
            pass

    return _Tr()
