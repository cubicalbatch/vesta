"""Tests for Stage B's two-pass scorer chain.

Fake ``Encoder``/``CrossEncoder`` doubles (satisfying the Protocols in
``encoders/contracts.py``) stand in for real ONNX models — these tests are
about the scorer's *chain and shortlist logic*, not embedding math (covered by
``test_encoders.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from vesta.config.capabilities import Capability
from vesta.retrieval.contracts import PreparedQuery, ScoredPassage
from vesta.retrieval.registry import resolve
from vesta.retrieval.scorers.cross_encoder import CrossEncoderScorer
from vesta.retrieval.scorers.static_pass import StaticPass
from vesta.zim.types import Passage


def _passage(
    path: str, text: str, *, ordinal: int = 0, breadcrumb: str = "", is_lead: bool = False
) -> Passage:
    return Passage(
        zim_id=1,
        path=path,
        ordinal=ordinal,
        char_start=0,
        char_end=len(text),
        breadcrumb=breadcrumb,
        text=text,
        is_lead=is_lead,
    )


def _sp(
    path: str,
    text: str,
    *,
    score: float = 0.0,
    ordinal: int = 0,
    breadcrumb: str = "",
    is_lead: bool = False,
) -> ScoredPassage:
    return ScoredPassage(
        passage=_passage(path, text, ordinal=ordinal, breadcrumb=breadcrumb, is_lead=is_lead),
        score=score,
        source_info="initial",
    )


def _pq(text: str = "capital of france") -> PreparedQuery:
    return PreparedQuery(
        raw=text,
        terms=tuple(text.split()),
        text=text,
        aliases=(),
        is_keyword_query=True,
        rung="initial",
    )


class FakeEncoder:
    """Returns a fixed similarity ranking: score = index of the keyword's
    position in a hand-picked relevance table, deterministic and order-checkable."""

    id = "fake-static"
    dim = 2
    max_tokens = 999
    query_prefix = ""
    passage_prefix = ""
    normalize = True
    pooling: Literal["cls", "mean"] = "mean"

    def __init__(self, relevance: dict[str, float]) -> None:
        self._relevance = relevance
        self.embed_calls = 0

    async def embed(
        self, texts: Sequence[str], *, kind: Literal["query", "passage"]
    ) -> NDArray[np.float32]:
        self.embed_calls += 1
        if kind == "query":
            return np.array([[1.0, 0.0]], dtype=np.float32)
        vecs = []
        for t in texts:
            r = next((v for k, v in self._relevance.items() if k in t), 0.0)
            vecs.append([r, (1.0 - r**2) ** 0.5 if r <= 1.0 else 0.0])
        return np.array(vecs, dtype=np.float32)


class FakeCrossEncoder:
    id = "fake-rerank"
    max_tokens = 256

    def __init__(self, relevance: dict[str, float]) -> None:
        self._relevance = relevance
        self.score_calls = 0
        self.last_passages: list[str] = []

    async def score(self, query: str, passages: Sequence[str]) -> Sequence[float]:
        self.score_calls += 1
        self.last_passages = list(passages)
        return [next((v for k, v in self._relevance.items() if k in p), 0.0) for p in passages]


class FakeManager:
    def __init__(
        self, static: FakeEncoder | None = None, rerank: FakeCrossEncoder | None = None
    ) -> None:
        self._static = static
        self._rerank = rerank

    async def get_static(self) -> FakeEncoder | None:
        return self._static

    async def get_embed(self) -> None:
        return None

    async def get_rerank(self) -> FakeCrossEncoder | None:
        return self._rerank


# ── Registration + capability contract ───────────────────────────────────────


def test_static_pass_and_cross_encoder_are_registered() -> None:
    assert resolve("passage_scorer", "static_pass") is StaticPass
    assert resolve("passage_scorer", "cross_encoder") is CrossEncoderScorer


@pytest.mark.parametrize(
    ("scorer_cls", "capability"),
    [
        (StaticPass, Capability.STATIC_ENCODER),
        (CrossEncoderScorer, Capability.CROSS_ENCODER),
    ],
)
def test_scorers_require_declared_capabilities(
    scorer_cls: type[StaticPass] | type[CrossEncoderScorer], capability: Capability
) -> None:
    assert scorer_cls.requires == frozenset({capability})


# ── StaticPass (Stage B1) ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_static_pass_shortlists_to_configured_size() -> None:
    passages = [_sp(f"P{i}", f"paris text {i}") for i in range(5)]
    encoder = FakeEncoder(relevance={"paris": 0.9})
    scorer = StaticPass(params=StaticPass.Params(shortlist=2), encoders=FakeManager(static=encoder))
    out = await scorer.score(passages, _pq(), tr=None)  # type: ignore[arg-type]
    assert len(out) == 2
    assert all(sp.source_info == "static_pass" for sp in out)


@pytest.mark.asyncio
async def test_static_pass_ranks_relevant_passage_first() -> None:
    passages = [_sp("Bananas", "bananas are yellow"), _sp("Paris", "paris is a city")]
    encoder = FakeEncoder(relevance={"paris": 1.0, "bananas": 0.1})
    scorer = StaticPass(
        params=StaticPass.Params(shortlist=10), encoders=FakeManager(static=encoder)
    )
    out = await scorer.score(passages, _pq(), tr=None)  # type: ignore[arg-type]
    assert out[0].passage.path == "Paris"
    assert out[0].score > out[1].score


@pytest.mark.asyncio
async def test_static_pass_respects_candidates_max_bound() -> None:
    passages = [_sp(f"P{i}", f"text {i}") for i in range(10)]
    encoder = FakeEncoder(relevance={})
    scorer = StaticPass(
        params=StaticPass.Params(candidates_max=3, shortlist=10),
        encoders=FakeManager(static=encoder),
    )
    await scorer.score(passages, _pq(), tr=None)  # type: ignore[arg-type]
    # embed() is called twice (query, passages); the passage call sees only 3 texts.
    assert encoder.embed_calls == 2


# ── Shared Scorer Guards (Empty Input & Passthroughs) ───────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scorer",
    [
        StaticPass(params=StaticPass.Params(), encoders=FakeManager(static=FakeEncoder({}))),
        CrossEncoderScorer(
            params=CrossEncoderScorer.Params(), encoders=FakeManager(rerank=FakeCrossEncoder({}))
        ),
    ],
)
async def test_scorers_handle_empty_input(scorer: StaticPass | CrossEncoderScorer) -> None:
    assert await scorer.score([], _pq(), tr=None) == []  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scorer", "check_calls_zero"),
    [
        (StaticPass(params=StaticPass.Params(), encoders=FakeManager(static=None)), False),
        (StaticPass(params=StaticPass.Params(), encoders=None), False),
        (
            CrossEncoderScorer(
                params=CrossEncoderScorer.Params(enabled=False),
                encoders=FakeManager(rerank=FakeCrossEncoder(relevance={"text": 0.9})),
            ),
            True,
        ),
        (
            CrossEncoderScorer(
                params=CrossEncoderScorer.Params(), encoders=FakeManager(rerank=None)
            ),
            False,
        ),
    ],
)
async def test_scorer_passthrough_guards(
    scorer: StaticPass | CrossEncoderScorer, check_calls_zero: bool
) -> None:
    passages = [_sp("A", "text")]
    out = await scorer.score(passages, _pq(), tr=None)  # type: ignore[arg-type]
    assert out == passages
    if check_calls_zero and isinstance(scorer, CrossEncoderScorer) and scorer._encoders is not None:
        rerank = await scorer._encoders.get_rerank()
        if rerank is not None:
            assert rerank.score_calls == 0


# ── exact_title_guarantee (Stage B1) ─────────────────────────────────────────
#
# Regression case (2026-08-04, live-reproduced): searching "fire" against an
# archive with an article titled exactly "Fire" ranked several of the
# article's own body sections (and other articles) ahead of its lead under
# cosine similarity — a bare keyword embeds more like a short, keyword-dense
# body section than the lead's broad prose. The lead was cut by the shortlist
# truncation and never reached `cross_encoder`, so `lead_boost`'s own
# exact-title guarantee downstream had nothing to promote (a passage that
# never reaches `scored` can't be resurrected later).


@pytest.mark.asyncio
async def test_static_pass_exact_title_lead_survives_shortlist_cut() -> None:
    # The lead scores 0.0 (no relevance keyword) while ten decoy body
    # passages all score 0.9 — with shortlist=5 the lead would ordinarily be
    # cut entirely, exactly the live-observed bug shape.
    fire_lead = _sp(
        "Fire", "fire is the rapid oxidation of a fuel", breadcrumb="Fire", is_lead=True
    )
    decoys = [
        _sp(f"Fire_decoy_{i}", f"keyword body passage {i}", breadcrumb=f"Fire decoy {i}")
        for i in range(10)
    ]
    encoder = FakeEncoder(relevance={"keyword": 0.9})
    scorer = StaticPass(params=StaticPass.Params(shortlist=5), encoders=FakeManager(static=encoder))
    out = await scorer.score([fire_lead, *decoys], _pq("fire"), tr=None)  # type: ignore[arg-type]

    assert len(out) == 6  # the 5-item shortlist plus the guaranteed extra
    assert any(sp.passage.path == "Fire" for sp in out)
    # nothing the shortlist already chose was evicted to make room.
    assert sum(1 for sp in out if sp.passage.path.startswith("Fire_decoy_")) == 5


@pytest.mark.asyncio
async def test_static_pass_exact_title_guarantee_is_noop_when_lead_already_shortlisted() -> None:
    """No duplicate is added when the exact-title lead already made the
    cosine-similarity cut on its own merits."""
    fire_lead = _sp(
        "Fire",
        "keyword fire is the rapid oxidation of a fuel",
        breadcrumb="Fire",
        is_lead=True,
        score=0.0,
    )
    other = _sp("Other", "unrelated passage text", breadcrumb="Other")
    encoder = FakeEncoder(relevance={"keyword": 0.9})
    scorer = StaticPass(params=StaticPass.Params(shortlist=5), encoders=FakeManager(static=encoder))
    out = await scorer.score([fire_lead, other], _pq("fire"), tr=None)  # type: ignore[arg-type]
    assert len(out) == 2  # not duplicated


@pytest.mark.asyncio
async def test_static_pass_exact_title_guarantee_can_be_disabled() -> None:
    fire_lead = _sp(
        "Fire", "fire is the rapid oxidation of a fuel", breadcrumb="Fire", is_lead=True
    )
    decoys = [
        _sp(f"Fire_decoy_{i}", f"keyword body passage {i}", breadcrumb=f"Fire decoy {i}")
        for i in range(10)
    ]
    encoder = FakeEncoder(relevance={"keyword": 0.9})
    scorer = StaticPass(
        params=StaticPass.Params(shortlist=5, exact_title_guarantee=False),
        encoders=FakeManager(static=encoder),
    )
    out = await scorer.score([fire_lead, *decoys], _pq("fire"), tr=None)  # type: ignore[arg-type]
    assert len(out) == 5
    assert not any(sp.passage.path == "Fire" for sp in out)


@pytest.mark.asyncio
async def test_static_pass_exact_title_guarantee_ignores_non_lead_and_non_matching() -> None:
    """A non-lead passage titled "Fire" and a lead passage titled something
    else are both ineligible — the guarantee must not fire for either."""
    fire_body = _sp("Fire", "fire body text", breadcrumb="Fire", is_lead=False, ordinal=1)
    other_lead = _sp("Other", "other lead text", breadcrumb="Other", is_lead=True)
    decoys = [_sp(f"D{i}", f"keyword body passage {i}", breadcrumb=f"D{i}") for i in range(10)]
    encoder = FakeEncoder(relevance={"keyword": 0.9})
    scorer = StaticPass(params=StaticPass.Params(shortlist=5), encoders=FakeManager(static=encoder))
    out = await scorer.score([fire_body, other_lead, *decoys], _pq("fire"), tr=None)  # type: ignore[arg-type]
    assert len(out) == 5  # no guarantee fired — shortlist size unchanged
    assert not any(sp.passage.path == "Fire" for sp in out)


# ── CrossEncoderScorer (Stage B2) ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cross_encoder_reranks_by_score() -> None:
    passages = [_sp("Bananas", "bananas are yellow"), _sp("Paris", "paris is a city")]
    rerank = FakeCrossEncoder(relevance={"paris": 0.95, "bananas": 0.05})
    scorer = CrossEncoderScorer(
        params=CrossEncoderScorer.Params(), encoders=FakeManager(rerank=rerank)
    )
    out = await scorer.score(passages, _pq(), tr=None)  # type: ignore[arg-type]
    assert out[0].passage.path == "Paris"
    assert all(sp.source_info == "cross_encoder" for sp in out)


@pytest.mark.asyncio
async def test_cross_encoder_respects_candidates_max_bound() -> None:
    passages = [_sp(f"P{i}", f"text {i}") for i in range(30)]
    rerank = FakeCrossEncoder(relevance={})
    scorer = CrossEncoderScorer(
        params=CrossEncoderScorer.Params(candidates_max=5), encoders=FakeManager(rerank=rerank)
    )
    await scorer.score(passages, _pq(), tr=None)  # type: ignore[arg-type]
    assert len(rerank.last_passages) == 5


# ── Chain semantics: B2's score replaces B1's (03 spec "N+1 replaces N") ────


@pytest.mark.asyncio
async def test_chain_cross_encoder_overrides_static_pass_score() -> None:
    passages = [_sp("Bananas", "bananas text"), _sp("Paris", "paris text")]
    static = FakeEncoder(relevance={"paris": 0.6, "bananas": 0.55})  # close call for B1
    rerank = FakeCrossEncoder(relevance={"paris": 0.99, "bananas": 0.01})  # decisive for B2

    b1 = StaticPass(params=StaticPass.Params(shortlist=10), encoders=FakeManager(static=static))
    after_b1 = await b1.score(passages, _pq(), tr=None)  # type: ignore[arg-type]

    b2 = CrossEncoderScorer(params=CrossEncoderScorer.Params(), encoders=FakeManager(rerank=rerank))
    after_b2 = await b2.score(after_b1, _pq(), tr=None)  # type: ignore[arg-type]

    assert after_b2[0].source_info == "cross_encoder"
    assert after_b2[0].passage.path == "Paris"
    assert after_b2[0].score > 0.9
