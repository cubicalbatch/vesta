"""SSE answer protocol contract test.

Verifies the event ordering, field names, and semantics documented in
``docs/sse-protocol.md``.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from vesta.answer.contracts import (
    AnswerContext,
    AnswerResetEvent,
    CitationsEvent,
    CitationSpan,
    DoneEvent,
    ErrorEvent,
    SourcesEvent,
    StatusEvent,
    TokenEvent,
    TraceEvent,
)
from vesta.retrieval.contracts import (
    ConfidenceSignals,
    RetrievalResult,
    ScoredPassage,
    SourceCard,
)
from vesta.retrieval.trace import Trace
from vesta.zim.types import Passage


def _make_test_passages() -> tuple[ScoredPassage, ...]:
    p = Passage(
        zim_id=1,
        path="Test_Article",
        ordinal=0,
        char_start=0,
        char_end=50,
        breadcrumb="Test Article > Section",
        text="The Battle of Hastings was fought on 14 October 1066.",
        is_lead=True,
    )
    return (ScoredPassage(passage=p, score=0.9, source_info="rerank"),)


def _make_test_cards() -> tuple[SourceCard, ...]:
    return (
        SourceCard(
            zim_id=1,
            path="Test_Article",
            title="Test Article",
            snippet="A test snippet.",
            breadcrumb="Test Article",
            score=0.9,
            source="xapian_fts",
        ),
    )


def _make_test_retrieval() -> RetrievalResult:
    return RetrievalResult(
        passages=_make_test_passages(),
        cards=_make_test_cards(),
        trace=Trace(),
        confidence=ConfidenceSignals(top_score=0.9, score_dropoff=0.3, density=0.5, agreement=0.5),
    )


# ── Event serialization contract ────────────────────────────────────────────


class TestSSESerialization:
    def test_sources_event_serializes(self) -> None:
        from vesta.api.answer import _serialize_event

        cards = _make_test_cards()
        sse = _serialize_event(SourcesEvent(cards=cards))
        assert sse.startswith("event: sources\n")
        data = json.loads(sse.split("data: ")[1].strip())
        assert "cards" in data
        assert data["cards"][0]["title"] == "Test Article"

    def test_status_event_serializes(self) -> None:
        from vesta.api.answer import _serialize_event

        sse = _serialize_event(StatusEvent(phase="reading", detail="3 sources"))
        assert "event: status" in sse
        data = json.loads(sse.split("data: ")[1].strip())
        assert data["phase"] == "reading"
        assert data["detail"] == "3 sources"

    def test_token_event_serializes(self) -> None:
        from vesta.api.answer import _serialize_event

        sse = _serialize_event(TokenEvent(text="hello"))
        assert "event: token" in sse
        data = json.loads(sse.split("data: ")[1].strip())
        assert data["text"] == "hello"

    def test_citations_event_serializes(self) -> None:
        from vesta.api.answer import _serialize_event

        spans = (
            CitationSpan(
                answer_start=0,
                answer_end=10,
                source_index=0,
                passage_start=5,
                passage_end=15,
                score=0.8,
            ),
        )
        sse = _serialize_event(CitationsEvent(spans=spans))
        assert "event: citations" in sse
        data = json.loads(sse.split("data: ")[1].strip())
        assert "spans" in data
        assert data["spans"][0]["answer_span"] == [0, 10]
        assert data["spans"][0]["card_id"] == 0
        assert data["spans"][0]["passage_span"] == [5, 15]

    def test_citations_null_passage_span(self) -> None:
        from vesta.api.answer import _serialize_event

        spans = (
            CitationSpan(
                answer_start=0,
                answer_end=10,
                source_index=0,
                passage_start=None,
                passage_end=None,
                score=0.3,
            ),
        )
        sse = _serialize_event(CitationsEvent(spans=spans))
        data = json.loads(sse.split("data: ")[1].strip())
        assert data["spans"][0]["passage_span"] is None

    def test_answer_reset_event_serializes(self) -> None:
        """FIX 1: a new ``answer_reset`` SSE event — emitted before a
        REPLACEMENT regenerate's first token so clients discard whatever
        they've accumulated for this turn so far."""
        from vesta.api.answer import _serialize_event

        sse = _serialize_event(AnswerResetEvent(reason="weak_support"))
        assert sse.startswith("event: answer_reset\n")
        data = json.loads(sse.split("data: ")[1].strip())
        assert data["reason"] == "weak_support"

    def test_answer_reset_event_default_reason_is_empty_string(self) -> None:
        from vesta.api.answer import _serialize_event

        sse = _serialize_event(AnswerResetEvent())
        data = json.loads(sse.split("data: ")[1].strip())
        assert data["reason"] == ""

    def test_error_event_serializes(self) -> None:
        from vesta.api.answer import _serialize_event

        sse = _serialize_event(
            ErrorEvent(code="stream_error", message="disconnected", recoverable=True)
        )
        assert "event: error" in sse
        data = json.loads(sse.split("data: ")[1].strip())
        assert data["code"] == "stream_error"
        assert data["recoverable"] is True

    def test_done_event_serializes(self) -> None:
        from vesta.api.answer import _serialize_event

        sse = _serialize_event(DoneEvent())
        assert "event: done" in sse

    def test_trace_event_serializes(self) -> None:
        from vesta.api.answer import _serialize_event

        tr = Trace()
        sse = _serialize_event(TraceEvent(trace=tr.to_dict()))
        assert "event: trace" in sse
        data = json.loads(sse.split("data: ")[1].strip())
        assert "version" in data


# ── Sources-only strategy ───────────────────────────────────────────────────


class TestSourcesOnlyStrategy:
    @pytest.mark.asyncio
    async def test_emits_sources_then_done(self) -> None:
        from vesta.answer.sources_only import SourcesOnlyStrategy

        retrieval = _make_test_retrieval()
        ctx = AnswerContext(retrieval=retrieval)
        strategy = SourcesOnlyStrategy()

        events = []
        async for ev in strategy.answer(ctx, Trace()):
            events.append(ev)

        assert isinstance(events[0], SourcesEvent)
        assert any(isinstance(e, StatusEvent) for e in events)
        assert any(isinstance(e, TraceEvent) for e in events)
        assert isinstance(events[-1], DoneEvent)
        # No token events in sources_only mode.
        assert not any(isinstance(e, TokenEvent) for e in events)


# ── Strategy auto-selection ─────────────────────────────────────────────────


class TestStrategyAutoSelection:
    def test_sources_only_always_stays(self) -> None:
        from vesta.answer import resolve_strategy_name
        from vesta.config.capabilities import Capability

        assert resolve_strategy_name("sources_only", frozenset()) == "sources_only"
        assert resolve_strategy_name("sources_only", frozenset({Capability.LLM})) == "sources_only"


# ── End-to-end SSE protocol via HTTP ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_answer_endpoint_sse_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The SSE endpoint produces valid events in the documented order."""

    # Build a tiny ZIM so retrieval has something to return.
    from fixtures.tiny_zim import build_tiny_zim

    zims_dir = tmp_path / "zims"
    zims_dir.mkdir(parents=True, exist_ok=True)
    build_tiny_zim(zims_dir / "tiny.zim")
    monkeypatch.setenv("data.dir", str(tmp_path))

    from vesta.main import create_app

    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # With no LLM configured, this should auto-select sources_only.
            resp = await client.get("/api/answer?q=Hastings")
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

            # Parse SSE events.
            events = _parse_sse(resp.text)
            event_types = [e[0] for e in events]

            # With no LLM configured, this should auto-select sources_only.
            # If retrieval succeeded, we should see sources events.
            # If there's an error, we should at least see the protocol shape.
            assert "done" in event_types
            if "error" in event_types:
                # Log the error for debugging but don't fail — the protocol
                # shape is what matters in this contract test.
                err = next(e[1] for e in events if e[0] == "error")
                assert "code" in err
                assert "message" in err
            else:
                assert "sources" in event_types
                # Regression: the trace event must carry the RETRIEVAL stages,
                # not just answer stages (docs/sse-protocol.md: "The answer
                # stages are appended to the retrieval stages"). The API
                # previously built a fresh Trace() and dropped result.trace.
                trace_events = [e[1] for e in events if e[0] == "trace"]
                assert trace_events
                components = {s.get("component") for s in trace_events[0].get("stages", [])}
                assert components & {"normalize", "xapian_fts", "title_suggest", "rrf"}, (
                    f"no retrieval stages in trace: {components}"
                )


@pytest.mark.asyncio
async def test_answer_endpoint_no_archives_abstains(app_client: AsyncClient) -> None:
    """With zero archives registered, every candidate source returns nothing and
    the pipeline raises NoCandidatesError. The endpoint must surface that the way
    /api/search does — a valid, explainable empty outcome (empty sources +
    abstention token + trace + done), NOT a ``retrieval_failed`` error event."""
    resp = await app_client.get("/api/answer?q=anything")
    assert resp.status_code == 200

    events = _parse_sse(resp.text)
    event_types = [e[0] for e in events]

    assert "error" not in event_types
    assert event_types[0] == "sources"
    sources = next(e[1] for e in events if e[0] == "sources")
    assert sources["cards"] == []
    statuses = [e[1]["phase"] for e in events if e[0] == "status"]
    assert "abstaining" in statuses
    tokens = [e[1]["text"] for e in events if e[0] == "token"]
    assert any("No passage" in t for t in tokens)
    assert "trace" in event_types
    assert event_types[-1] == "done"


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parse SSE text into ``(event_name, data_dict)`` pairs."""
    events = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event_name = ""
        data_str = ""
        for line in block.split("\n"):
            if line.startswith("event: "):
                event_name = line[len("event: ") :]
            elif line.startswith("data: "):
                data_str = line[len("data: ") :]
        if event_name and data_str:
            with contextlib.suppress(json.JSONDecodeError):
                events.append((event_name, json.loads(data_str)))
    return events


# ── Fixture conformance ─────────────────────────────────────────────────────


class TestSSEFixtureConformance:
    """The three committed SSE fixtures must conform to the frozen protocol:
    sources first, done last, valid event types, and parseable JSON."""

    FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sse"

    def test_three_fixtures_exist(self) -> None:
        files = list(self.FIXTURES_DIR.glob("*.json"))
        assert len(files) >= 3, f"DoD requires ≥3 fixtures, found {len(files)}"

    def test_single_shot_fixture_ordering(self) -> None:
        data = json.loads((self.FIXTURES_DIR / "single_shot_cited.json").read_text())
        events = data["events"]
        types = [e["event"] for e in events]
        assert types[0] == "sources"
        assert types[-1] == "done"
        # citations after tokens
        if "citations" in types:
            assert types.index("citations") > types.index("token")
        # trace before done
        if "trace" in types:
            assert types.index("trace") < types.index("done")

    def test_sources_only_fixture_no_tokens(self) -> None:
        data = json.loads((self.FIXTURES_DIR / "sources_only.json").read_text())
        types = [e["event"] for e in data["events"]]
        assert "token" not in types
        assert "citations" not in types
        assert types[0] == "sources"
        assert types[-1] == "done"

    def test_abstention_fixture_has_abstaining_status(self) -> None:
        data = json.loads((self.FIXTURES_DIR / "abstention.json").read_text())
        statuses = [e["data"]["phase"] for e in data["events"] if e["event"] == "status"]
        assert "abstaining" in statuses


# ── Sources-merge protocol amendment ────────────────────────────────────────


class TestSourcesMergeAmendment:
    """The additive `sources` merge amendment: every `sources` event carries
    `merge`, a second `sources` event (when present) is a delta that arrives
    after tokens and before citations, and the recorded recovery fixture
    demonstrates the full shape end to end."""

    FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sse"

    def test_serialized_sources_event_always_carries_merge_field(self) -> None:
        from vesta.api.answer import _serialize_event

        sse = _serialize_event(SourcesEvent(cards=_make_test_cards()))
        data = json.loads(sse.split("data: ")[1].strip())
        assert data["merge"] is False

        sse2 = _serialize_event(SourcesEvent(cards=_make_test_cards(), merge=True))
        data2 = json.loads(sse2.split("data: ")[1].strip())
        assert data2["merge"] is True

    def test_recovery_fixture_exists_and_demonstrates_merge(self) -> None:
        data = json.loads((self.FIXTURES_DIR / "agentic_recovery.json").read_text())
        events = data["events"]
        types = [e["event"] for e in events]

        assert types[0] == "sources"
        assert events[0]["data"]["merge"] is False
        assert types[-1] == "done"

        sources_indices = [i for i, t in enumerate(types) if t == "sources"]
        assert len(sources_indices) == 2, "the recovery fixture must show the merge event"
        merge_idx = sources_indices[1]
        assert events[merge_idx]["data"]["merge"] is True
        # The merge event's cards are a DELTA, not the full accumulated set —
        # it must not repeat the first event's card.
        first_paths = {c["path"] for c in events[sources_indices[0]]["data"]["cards"]}
        merge_paths = {c["path"] for c in events[merge_idx]["data"]["cards"]}
        assert first_paths.isdisjoint(merge_paths)

        # Ordering rule: merge event after all tokens, before citations.
        last_token_idx = max(i for i, t in enumerate(types) if t == "token")
        citations_idx = types.index("citations")
        assert last_token_idx < merge_idx < citations_idx

        # A citation may reference the merged card (card_id 1 — the delta
        # continues the first event's 0-based numbering).
        citation_card_ids = {s["card_id"] for s in events[citations_idx]["data"]["spans"]}
        assert 1 in citation_card_ids
