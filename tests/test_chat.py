"""Multi-turn chat: DB layer, history policy, and ``POST /api/chat`` (09
'Multi-turn'). Companion to ``test_answer_sse.py`` (single-turn protocol) and
``test_conversational_rewrite.py`` (the turn-≥2 rewrite itself) — this file
covers the piece neither of those exercises: conversation persistence and the
chat endpoint that threads history through the shared answer pipeline.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from vesta.answer.conversation import build_history, derive_title
from vesta.api.conversation_store import SqliteConversationStore, prune_old_traces
from vesta.db.connection import Database
from vesta.db.migrations import run_migrations

# ── build_history / derive_title (pure, no DB) ──────────────────────────────


class TestBuildHistory:
    def test_empty_prior_returns_empty(self) -> None:
        assert build_history((), max_turns=10) == ()

    def test_max_turns_zero_returns_empty(self) -> None:
        prior = (("user", "hi"), ("assistant", "hello"))
        assert build_history(prior, max_turns=0) == ()

    def test_negative_max_turns_returns_empty(self) -> None:
        prior = (("user", "hi"), ("assistant", "hello"))
        assert build_history(prior, max_turns=-1) == ()

    def test_keeps_most_recent_n_entries(self) -> None:
        prior = tuple(("user", f"msg{i}") for i in range(5))
        bounded = build_history(prior, max_turns=2)
        assert bounded == (("user", "msg3"), ("user", "msg4"))

    def test_fewer_entries_than_max_turns_kept_whole(self) -> None:
        prior = (("user", "hi"), ("assistant", "hello"))
        assert build_history(prior, max_turns=10) == prior


class TestDeriveTitle:
    def test_short_text_is_its_own_title(self) -> None:
        assert derive_title("What is the capital of France?") == "What is the capital of France?"

    def test_collapses_whitespace(self) -> None:
        assert derive_title("What   is\n\nthe capital?") == "What is the capital?"

    def test_truncates_long_text_with_ellipsis(self) -> None:
        text = "a" * 100
        title = derive_title(text, max_len=60)
        assert len(title) == 61  # 60 chars + ellipsis
        assert title.endswith("…")

    def test_exact_max_len_not_truncated(self) -> None:
        text = "a" * 60
        assert derive_title(text, max_len=60) == text


# ── SqliteConversationStore (real DB, no app) ───────────────────────────────


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(str(tmp_path / "vesta.db"), busy_timeout_ms=1000)
    await d.start()
    async with d.write() as conn:
        await run_migrations(conn)
    yield d
    await d.stop()


class TestSqliteConversationStore:
    @pytest.mark.asyncio
    async def test_create_and_list_conversations(self, db: Database) -> None:
        store = SqliteConversationStore(db)
        cid = await store.create_conversation("My chat")
        convs = await store.list_conversations()
        assert len(convs) == 1
        assert convs[0].id == cid
        assert convs[0].title == "My chat"

    @pytest.mark.asyncio
    async def test_create_with_none_title(self, db: Database) -> None:
        store = SqliteConversationStore(db)
        cid = await store.create_conversation(None)
        convs = await store.list_conversations()
        assert convs[0].id == cid
        assert convs[0].title is None

    @pytest.mark.asyncio
    async def test_append_message_roundtrips_optional_fields(self, db: Database) -> None:
        store = SqliteConversationStore(db)
        cid = await store.create_conversation("t")
        await store.append_message(
            cid,
            "assistant",
            "the answer",
            sources_json='[{"zim_id": 1}]',
            trace_json='{"version": 1}',
            tokens_in=10,
            tokens_out=20,
            latency_ms=500,
        )
        messages = await store.list_messages(cid)
        assert len(messages) == 1
        m = messages[0]
        assert m.role == "assistant"
        assert m.content == "the answer"
        assert m.sources_json == '[{"zim_id": 1}]'
        assert m.trace_json == '{"version": 1}'
        assert m.tokens_in == 10
        assert m.tokens_out == 20
        assert m.latency_ms == 500

    @pytest.mark.asyncio
    async def test_append_message_defaults_optional_fields_to_none(self, db: Database) -> None:
        store = SqliteConversationStore(db)
        cid = await store.create_conversation("t")
        await store.append_message(cid, "user", "hello")
        messages = await store.list_messages(cid)
        m = messages[0]
        assert m.sources_json is None
        assert m.trace_json is None
        assert m.tokens_in is None

    @pytest.mark.asyncio
    async def test_messages_ordered_oldest_first(self, db: Database) -> None:
        store = SqliteConversationStore(db)
        cid = await store.create_conversation("t")
        await store.append_message(cid, "user", "first")
        await store.append_message(cid, "assistant", "second")
        await store.append_message(cid, "user", "third")
        messages = await store.list_messages(cid)
        assert [m.content for m in messages] == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_list_messages_past_limit_keeps_newest_not_oldest(self, db: Database) -> None:
        """AUDIT_0822 C3: ``ORDER BY id ASC LIMIT ?`` pins the OLDEST window once
        a conversation outgrows ``limit`` — recent turns silently vanished from
        the detail view. DESC + Python-reverse keeps the newest while still
        returning them oldest-first."""
        store = SqliteConversationStore(db)
        cid = await store.create_conversation("t")
        for i in range(5):
            await store.append_message(cid, "user", f"msg{i}")
        messages = await store.list_messages(cid, limit=3)
        assert [m.content for m in messages] == ["msg2", "msg3", "msg4"]

    @pytest.mark.asyncio
    async def test_list_recent_messages_keeps_newest_window_chronological(
        self, db: Database
    ) -> None:
        """AUDIT_0822 C3 on the history path: past ``limit`` rows the window must
        slide — newest turns kept, oldest dropped — and come back oldest-first so
        the rewriter/model history stays chronological."""
        store = SqliteConversationStore(db)
        cid = await store.create_conversation("t")
        contents = [f"m{i}" for i in range(6)]
        for i, content in enumerate(contents):
            await store.append_message(cid, "user" if i % 2 == 0 else "assistant", content)
        messages = await store.list_recent_messages(cid, limit=4)
        assert [m.content for m in messages] == contents[-4:]

    @pytest.mark.asyncio
    async def test_list_recent_messages_returns_only_role_and_content(self, db: Database) -> None:
        """AUDIT_0822 C8: the chat-history read selects only ``(role, content)``;
        every payload column is left defaulted instead of deserialized and thrown
        away. Full rows remain available via ``list_messages``."""
        store = SqliteConversationStore(db)
        cid = await store.create_conversation("t")
        await store.append_message(
            cid,
            "assistant",
            "the answer",
            sources_json='[{"zim_id": 1}]',
            trace_json='{"version": 1, "stages": ["big"]}',
            tokens_in=10,
            tokens_out=20,
            latency_ms=500,
        )
        slim = await store.list_recent_messages(cid, limit=10)
        assert len(slim) == 1
        m = slim[0]
        assert m.role == "assistant"
        assert m.content == "the answer"
        assert m.id == 0
        assert m.sources_json is None
        assert m.trace_json is None
        assert m.tokens_in is None
        assert m.tokens_out is None
        assert m.latency_ms is None
        assert m.created_at is None

    @pytest.mark.asyncio
    async def test_append_message_touches_conversation_updated_at(self, db: Database) -> None:
        store = SqliteConversationStore(db)
        cid = await store.create_conversation("t")
        before = (await store.list_conversations())[0].updated_at
        await store.append_message(cid, "user", "hello")
        after = (await store.list_conversations())[0].updated_at
        assert after is not None
        assert before is not None
        assert after >= before

    @pytest.mark.asyncio
    async def test_delete_conversation_cascades_messages(self, db: Database) -> None:
        store = SqliteConversationStore(db)
        cid = await store.create_conversation("t")
        await store.append_message(cid, "user", "hello")
        deleted = await store.delete_conversation(cid)
        assert deleted is True
        assert await store.list_messages(cid) == []
        assert await store.list_conversations() == []

    @pytest.mark.asyncio
    async def test_delete_unknown_conversation_returns_false(self, db: Database) -> None:
        store = SqliteConversationStore(db)
        assert await store.delete_conversation(999) is False

    @pytest.mark.asyncio
    async def test_get_conversation_resolves_beyond_list_window(self, db: Database) -> None:
        """AUDIT_0824 C4: a conversation older than the list window still resolves
        by id — the old scan of the 500 newest rows 404'd it."""
        store = SqliteConversationStore(db)
        oldest = await store.create_conversation("oldest")
        for i in range(500):
            await store.create_conversation(f"c{i}")
        conv = await store.get_conversation(oldest)
        assert conv is not None
        assert conv.id == oldest
        assert conv.title == "oldest"
        # It sits outside what any 500-row listing can see.
        assert all(c.id != oldest for c in await store.list_conversations(limit=500))
        assert await store.get_conversation(999999) is None


class TestPruneOldTraces:
    @pytest.mark.asyncio
    async def test_retention_zero_is_noop(self, db: Database) -> None:
        store = SqliteConversationStore(db)
        cid = await store.create_conversation("t")
        await store.append_message(cid, "assistant", "a", trace_json='{"x": 1}')
        pruned = await prune_old_traces(db, 0)
        assert pruned == 0
        messages = await store.list_messages(cid)
        assert messages[0].trace_json == '{"x": 1}'

    @pytest.mark.asyncio
    async def test_old_trace_is_pruned(self, db: Database) -> None:
        store = SqliteConversationStore(db)
        cid = await store.create_conversation("t")
        msg_id = await store.append_message(cid, "assistant", "a", trace_json='{"x": 1}')
        # Backdate created_at well beyond the retention window.
        old = (
            (_dt.datetime.now(_dt.UTC) - _dt.timedelta(days=30)).replace(microsecond=0).isoformat()
        )
        async with db.write() as conn:
            await conn.execute("UPDATE messages SET created_at=? WHERE id=?", (old, msg_id))
        pruned = await prune_old_traces(db, 7)
        assert pruned == 1
        messages = await store.list_messages(cid)
        assert messages[0].trace_json is None
        # Content is untouched — only the trace is pruned.
        assert messages[0].content == "a"

    @pytest.mark.asyncio
    async def test_recent_trace_is_kept(self, db: Database) -> None:
        store = SqliteConversationStore(db)
        cid = await store.create_conversation("t")
        await store.append_message(cid, "assistant", "a", trace_json='{"x": 1}')
        pruned = await prune_old_traces(db, 7)
        assert pruned == 0
        messages = await store.list_messages(cid)
        assert messages[0].trace_json == '{"x": 1}'


# ── POST /api/chat + conversation CRUD (live app, no LLM → sources_only) ───


def _parse_sse(text: str) -> list[tuple[str, dict]]:
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


class TestChatEndpoint:
    @pytest.mark.asyncio
    async def test_empty_query_is_400(
        self, app_client_with_zim: tuple[httpx.AsyncClient, int]
    ) -> None:
        client, _ = app_client_with_zim
        resp = await client.post("/api/chat", json={"query": "   "})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_unknown_conversation_id_is_404(
        self, app_client_with_zim: tuple[httpx.AsyncClient, int]
    ) -> None:
        client, _ = app_client_with_zim
        resp = await client.post("/api/chat", json={"query": "hello", "conversation_id": 999999})
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_new_conversation_returns_header_and_sse_stream(
        self, app_client_with_zim: tuple[httpx.AsyncClient, int]
    ) -> None:
        client, _ = app_client_with_zim
        resp = await client.post("/api/chat", json={"query": "Hastings"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        conv_id = resp.headers.get("X-Conversation-Id")
        assert conv_id is not None and int(conv_id) > 0

        events = _parse_sse(resp.text)
        event_types = [e[0] for e in events]
        assert "done" in event_types
        # No LLM configured in the test app → sources_only auto-selection,
        # same protocol shape /api/answer already guarantees.
        assert "sources" in event_types or "error" in event_types

    @pytest.mark.asyncio
    async def test_conversation_persists_user_and_assistant_turns(
        self, app_client_with_zim: tuple[httpx.AsyncClient, int]
    ) -> None:
        client, _ = app_client_with_zim
        resp = await client.post("/api/chat", json={"query": "Hastings"})
        conv_id = int(resp.headers["X-Conversation-Id"])

        detail = await client.get(f"/api/conversations/{conv_id}")
        assert detail.status_code == 200
        body = detail.json()
        assert body["conversation"]["id"] == conv_id
        roles = [m["role"] for m in body["messages"]]
        assert roles == ["user", "assistant"]
        assert body["messages"][0]["content"] == "Hastings"

    @pytest.mark.asyncio
    async def test_second_turn_reuses_conversation_id(
        self, app_client_with_zim: tuple[httpx.AsyncClient, int]
    ) -> None:
        client, _ = app_client_with_zim
        first = await client.post("/api/chat", json={"query": "Hastings"})
        conv_id = int(first.headers["X-Conversation-Id"])

        second = await client.post(
            "/api/chat", json={"query": "when was it?", "conversation_id": conv_id}
        )
        assert second.status_code == 200
        assert int(second.headers["X-Conversation-Id"]) == conv_id

        detail = await client.get(f"/api/conversations/{conv_id}")
        roles_and_content = [(m["role"], m["content"]) for m in detail.json()["messages"]]
        assert ("user", "Hastings") in roles_and_content
        assert ("user", "when was it?") in roles_and_content
        # Exactly two user turns + two assistant turns.
        assert len([r for r, _ in roles_and_content if r == "user"]) == 2
        assert len([r for r, _ in roles_and_content if r == "assistant"]) == 2

    @pytest.mark.asyncio
    async def test_list_conversations_newest_first(
        self, app_client_with_zim: tuple[httpx.AsyncClient, int]
    ) -> None:
        client, _ = app_client_with_zim
        first = await client.post("/api/chat", json={"query": "first question"})
        second = await client.post("/api/chat", json={"query": "second question"})
        first_id = int(first.headers["X-Conversation-Id"])
        second_id = int(second.headers["X-Conversation-Id"])

        listing = await client.get("/api/conversations")
        assert listing.status_code == 200
        ids = [c["id"] for c in listing.json()]
        assert ids.index(second_id) < ids.index(first_id)

    @pytest.mark.asyncio
    async def test_conversation_auto_titled_from_first_message(
        self, app_client_with_zim: tuple[httpx.AsyncClient, int]
    ) -> None:
        client, _ = app_client_with_zim
        resp = await client.post("/api/chat", json={"query": "What is Hastings?"})
        conv_id = int(resp.headers["X-Conversation-Id"])
        detail = await client.get(f"/api/conversations/{conv_id}")
        assert detail.json()["conversation"]["title"] == "What is Hastings?"

    @pytest.mark.asyncio
    async def test_get_unknown_conversation_is_404(
        self, app_client_with_zim: tuple[httpx.AsyncClient, int]
    ) -> None:
        client, _ = app_client_with_zim
        resp = await client.get("/api/conversations/999999")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_conversation_removes_it_and_its_messages(
        self, app_client_with_zim: tuple[httpx.AsyncClient, int]
    ) -> None:
        client, _ = app_client_with_zim
        resp = await client.post("/api/chat", json={"query": "Hastings"})
        conv_id = int(resp.headers["X-Conversation-Id"])

        deleted = await client.delete(f"/api/conversations/{conv_id}")
        assert deleted.status_code == 200
        assert deleted.json() == {"deleted": True}

        after = await client.get(f"/api/conversations/{conv_id}")
        assert after.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_unknown_conversation_is_404(
        self, app_client_with_zim: tuple[httpx.AsyncClient, int]
    ) -> None:
        client, _ = app_client_with_zim
        resp = await client.delete("/api/conversations/999999")
        assert resp.status_code == 404


# ── AnswerResetEvent + citations.answer_text persistence (precision fixes) ──


class TestAnswerResetAndCitationRewritePersistence:
    """FIX 1 + FIX 2: ``_run_chat_turn``'s ``answer_parts`` accumulator must
    honour ``AnswerResetEvent`` (discard, not concatenate the doubled answer)
    and prefer the citations event's ``answer_text`` (the citation-renumbered
    final text) over the raw token join, when present. Drives the real
    ``/api/chat`` endpoint with ``iter_answer_events`` swapped for a fake that
    emits a controlled event sequence — the persistence logic under test lives
    entirely inside ``_run_chat_turn``, so this is the only way to exercise it
    without a real LLM."""

    @pytest.mark.asyncio
    async def test_answer_reset_discards_previously_accumulated_tokens(
        self, app_client_with_zim: tuple[httpx.AsyncClient, int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import vesta.api.chat as chat_module
        from vesta.answer.contracts import (
            AnswerResetEvent,
            DoneEvent,
            SourcesEvent,
            TokenEvent,
            TraceEvent,
        )

        async def fake_iter_answer_events(
            state: object,
            query: str,
            scope: str | None,
            profile: str | None,
            strategy: str | None,
            *,
            history: tuple[tuple[str, str], ...] = (),
        ) -> AsyncIterator[object]:
            yield SourcesEvent(cards=())
            yield TokenEvent(text="OLD ANSWER THAT GETS SUPERSEDED")
            yield AnswerResetEvent(reason="test")
            yield TokenEvent(text="new correct answer")
            yield TraceEvent(trace={"version": 1, "stages": []})
            yield DoneEvent()

        monkeypatch.setattr(chat_module, "iter_answer_events", fake_iter_answer_events)
        # Force the no-LLM (sources_only) branch so the fake generator above is
        # used — otherwise /api/chat would route to the pydantic-ai agent
        # whenever an LLM is configured in the environment.
        monkeypatch.setattr(chat_module, "compute_capabilities", frozenset)

        client, _ = app_client_with_zim
        resp = await client.post("/api/chat", json={"query": "test"})
        conv_id = int(resp.headers["X-Conversation-Id"])

        detail = await client.get(f"/api/conversations/{conv_id}")
        assistant = next(m for m in detail.json()["messages"] if m["role"] == "assistant")
        assert assistant["content"] == "new correct answer"
        assert "OLD ANSWER" not in assistant["content"]

    @pytest.mark.asyncio
    async def test_citations_answer_text_is_persisted_not_raw_token_join(
        self, app_client_with_zim: tuple[httpx.AsyncClient, int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FIX 2: when the citations event carries the citation-renumbered
        ``answer_text`` (passage-numbered markers rewritten to card numbers),
        ``_run_chat_turn`` must persist THAT — the raw token join still cites
        the passage number ``[7]`` the UI never displays. ``answer_text=None``
        (already covered by the other test: there is no citations event at all
        and the token join persists as before) is the no-rewrite fallback."""
        import vesta.api.chat as chat_module
        from vesta.answer.contracts import (
            CitationsEvent,
            DoneEvent,
            SourcesEvent,
            TokenEvent,
            TraceEvent,
        )

        async def fake_iter_answer_events(
            state: object,
            query: str,
            scope: str | None,
            profile: str | None,
            strategy: str | None,
            *,
            history: tuple[tuple[str, str], ...] = (),
        ) -> AsyncIterator[object]:
            yield SourcesEvent(cards=())
            yield TokenEvent(text="The battle happened in 1066 [7].")
            # spans=() is fine — persistence only reads answer_text; the empty
            # span tuple mirrors a citations event where alignment found nothing
            # but the renumbered text is still authoritative.
            yield CitationsEvent(spans=(), answer_text="The battle happened in 1066 [1].")
            yield TraceEvent(trace={"version": 1, "stages": []})
            yield DoneEvent()

        monkeypatch.setattr(chat_module, "iter_answer_events", fake_iter_answer_events)
        # Force the no-LLM (sources_only) branch so the fake generator above is
        # used — see the sibling FIX test for why.
        monkeypatch.setattr(chat_module, "compute_capabilities", frozenset)

        client, _ = app_client_with_zim
        resp = await client.post("/api/chat", json={"query": "test"})
        conv_id = int(resp.headers["X-Conversation-Id"])

        detail = await client.get(f"/api/conversations/{conv_id}")
        assistant = next(m for m in detail.json()["messages"] if m["role"] == "assistant")
        assert assistant["content"] == "The battle happened in 1066 [1]."
        assert assistant["content"] != "The battle happened in 1066 [7]."


class TestSourcesMergePersistence:
    """AUDIT_0824 M9: ``_run_chat_turn`` must accumulate cards across
    ``SourcesEvent``s before persisting. The agent chat path emits a final
    ``sources(merge=True)`` event carrying ONLY the delta cards its tool rounds
    discovered beyond the initial set (docs/sse-protocol.md "Sources merge");
    replacing the accumulator on every event persisted a row whose
    ``sources_json`` held just the late cards, so restored conversations lost
    the Round-0 sources and the persisted answer's [n] markers referenced card
    numbers that no longer existed. The live SPA appends on merge — the
    persisted JSON must match what the client displayed."""

    @staticmethod
    def _card(path: str) -> object:
        from vesta.retrieval.contracts import SourceCard

        return SourceCard(
            zim_id=1,
            path=path,
            title=path,
            snippet=f"snippet for {path}",
            breadcrumb=path,
            score=1.0,
            source="test",
        )

    @pytest.mark.asyncio
    async def test_merge_delta_is_appended_not_replaced(
        self, app_client_with_zim: tuple[httpx.AsyncClient, int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import vesta.api.chat as chat_module
        from vesta.answer.contracts import (
            CitationsEvent,
            DoneEvent,
            SourcesEvent,
            TokenEvent,
            TraceEvent,
        )

        async def fake_iter_answer_events(
            state: object,
            query: str,
            scope: str | None,
            profile: str | None,
            strategy: str | None,
            *,
            history: tuple[tuple[str, str], ...] = (),
        ) -> AsyncIterator[object]:
            yield SourcesEvent(cards=(self._card("round0/a"),))  # type: ignore[arg-type]
            yield TokenEvent(text="Answer citing [1] and [2].")
            yield SourcesEvent(cards=(self._card("recovered/b"),), merge=True)  # type: ignore[arg-type]
            yield CitationsEvent(spans=(), answer_text="Answer citing [0] and [1].")
            yield TraceEvent(trace={"version": 1, "stages": []})
            yield DoneEvent()

        monkeypatch.setattr(chat_module, "iter_answer_events", fake_iter_answer_events)
        # Force the no-LLM (sources_only) branch so the fake generator above is
        # used — see the sibling FIX tests for why.
        monkeypatch.setattr(chat_module, "compute_capabilities", frozenset)

        client, _ = app_client_with_zim
        resp = await client.post("/api/chat", json={"query": "test"})
        conv_id = int(resp.headers["X-Conversation-Id"])

        detail = await client.get(f"/api/conversations/{conv_id}")
        assistant = next(m for m in detail.json()["messages"] if m["role"] == "assistant")
        cards = assistant["sources"]
        assert [c["path"] for c in cards] == ["round0/a", "recovered/b"]

    @pytest.mark.asyncio
    async def test_single_non_merge_event_persists_exactly_its_cards(
        self, app_client_with_zim: tuple[httpx.AsyncClient, int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No tool rounds → exactly one non-merge sources event; the persisted
        row is unchanged from the pre-fix behaviour."""
        import vesta.api.chat as chat_module
        from vesta.answer.contracts import DoneEvent, SourcesEvent, TokenEvent, TraceEvent

        async def fake_iter_answer_events(
            state: object,
            query: str,
            scope: str | None,
            profile: str | None,
            strategy: str | None,
            *,
            history: tuple[tuple[str, str], ...] = (),
        ) -> AsyncIterator[object]:
            yield SourcesEvent(cards=(self._card("only/a"), self._card("only/b")))  # type: ignore[arg-type]
            yield TokenEvent(text="plain answer")
            yield TraceEvent(trace={"version": 1, "stages": []})
            yield DoneEvent()

        monkeypatch.setattr(chat_module, "iter_answer_events", fake_iter_answer_events)
        monkeypatch.setattr(chat_module, "compute_capabilities", frozenset)

        client, _ = app_client_with_zim
        resp = await client.post("/api/chat", json={"query": "test"})
        conv_id = int(resp.headers["X-Conversation-Id"])

        detail = await client.get(f"/api/conversations/{conv_id}")
        assistant = next(m for m in detail.json()["messages"] if m["role"] == "assistant")
        cards = assistant["sources"]
        assert [c["path"] for c in cards] == ["only/a", "only/b"]


class TestErrorTerminalStream:
    """AUDIT_0824 C1: ordering rule 8 — an ``error`` event terminates the
    stream (no ``done`` after it), and a failed turn still persists whatever
    of the exchange existed when it died."""

    @pytest.mark.asyncio
    async def test_upstream_crash_ends_stream_at_error_and_persists_partial(
        self, app_client_with_zim: tuple[httpx.AsyncClient, int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The upstream generator raises mid-stream: the client stream ends on
        the fatal ``error`` event (never a trailing ``done``), and the partial
        answer is persisted alongside the already-written user row."""
        import vesta.api.chat as chat_module
        from vesta.answer.contracts import TokenEvent

        async def fake_iter_answer_events(
            state: object,
            query: str,
            scope: str | None,
            profile: str | None,
            strategy: str | None,
            *,
            history: tuple[tuple[str, str], ...] = (),
        ) -> AsyncIterator[object]:
            yield TokenEvent(text="partial ans")
            raise RuntimeError("model exploded mid-turn")

        monkeypatch.setattr(chat_module, "iter_answer_events", fake_iter_answer_events)
        monkeypatch.setattr(chat_module, "compute_capabilities", frozenset)

        client, _ = app_client_with_zim
        resp = await client.post("/api/chat", json={"query": "test"})
        assert resp.status_code == 200
        names = [name for name, _ in _parse_sse(resp.text)]
        assert names[-1] == "error"
        assert "done" not in names

        detail = await client.get(f"/api/conversations/{int(resp.headers['X-Conversation-Id'])}")
        turns = [(m["role"], m["content"]) for m in detail.json()["messages"]]
        assert turns == [("user", "test"), ("assistant", "partial ans")]

    @pytest.mark.asyncio
    async def test_terminal_error_event_stops_consuming_and_persists_partial(
        self, app_client_with_zim: tuple[httpx.AsyncClient, int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A yielded ``ErrorEvent`` is terminal even if a misbehaving upstream
        keeps emitting afterwards: nothing may follow it on the wire, and the
        partial turn is still persisted."""
        import vesta.api.chat as chat_module
        from vesta.answer.contracts import DoneEvent, ErrorEvent, TokenEvent, TraceEvent

        async def fake_iter_answer_events(
            state: object,
            query: str,
            scope: str | None,
            profile: str | None,
            strategy: str | None,
            *,
            history: tuple[tuple[str, str], ...] = (),
        ) -> AsyncIterator[object]:
            yield TokenEvent(text="half an answer")
            yield ErrorEvent(code="fatal", message="boom", recoverable=False)
            # Misbehaving upstream: these must NEVER reach the wire.
            yield TraceEvent(trace={"version": 1, "stages": []})
            yield DoneEvent()

        monkeypatch.setattr(chat_module, "iter_answer_events", fake_iter_answer_events)
        monkeypatch.setattr(chat_module, "compute_capabilities", frozenset)

        client, _ = app_client_with_zim
        resp = await client.post("/api/chat", json={"query": "test"})
        assert resp.status_code == 200
        names = [name for name, _ in _parse_sse(resp.text)]
        assert names == ["token", "error"]

        detail = await client.get(f"/api/conversations/{int(resp.headers['X-Conversation-Id'])}")
        turns = [(m["role"], m["content"]) for m in detail.json()["messages"]]
        assert turns == [("user", "test"), ("assistant", "half an answer")]

    @pytest.mark.asyncio
    async def test_setup_failure_before_first_event_persists_assistant_row(
        self, app_client_with_zim: tuple[httpx.AsyncClient, int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-LLM exception raised while setting the stream up (capability
        probe, history reconstruction) — after the user row is written but
        before a single event streams — must still persist the assistant turn
        (empty content: nothing had accumulated) and end the wire on ``error``.
        Without this the question dangles with no answer row at all."""
        import vesta.api.chat as chat_module

        def boom() -> frozenset:
            raise RuntimeError("capability probe exploded")

        monkeypatch.setattr(chat_module, "compute_capabilities", boom)

        client, _ = app_client_with_zim
        resp = await client.post("/api/chat", json={"query": "test"})
        assert resp.status_code == 200
        names = [name for name, _ in _parse_sse(resp.text)]
        assert names[-1] == "error"
        assert "done" not in names

        detail = await client.get(f"/api/conversations/{int(resp.headers['X-Conversation-Id'])}")
        turns = [(m["role"], m["content"]) for m in detail.json()["messages"]]
        assert turns == [("user", "test"), ("assistant", "")]


# ── The local runtime in the answer path ─────────────────────────────────────


class TestLocalRuntimeWarmup:
    @pytest.mark.asyncio
    async def test_runtime_failure_yields_error_event_and_200(
        self, app_client_with_zim: tuple[httpx.AsyncClient, int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """(c) A runtime that cannot come up (missing binary, load failure,
        no-match model id) produces a clean terminal SSE ``error`` event with
        a human message — never a 500ing stream."""
        import vesta.api.chat as chat_module
        from fixtures.llm_runtime import FakeLlmRuntime
        from vesta.config.capabilities import Capability
        from vesta.inference.runtime import LlmRuntimeError

        fake = FakeLlmRuntime(
            error=LlmRuntimeError(
                "model 'missing.gguf' not found in the llama-server registry; "
                "router reports: ['other.gguf']"
            )
        )
        monkeypatch.setattr("vesta.inference.get_runtime", lambda: fake)
        monkeypatch.setattr(
            chat_module, "compute_capabilities", lambda: frozenset({Capability.LLM})
        )

        client, _ = app_client_with_zim
        resp = await client.post("/api/chat", json={"query": "Hastings"})
        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        names = [name for name, _ in events]
        assert "error" in names
        err = next(d for name, d in events if name == "error")
        assert err["code"] == "no_llm"
        assert err["recoverable"] is True
        assert "Settings" in err["message"]
        assert "missing.gguf" in err["message"]
        # Protocol ordering rule 8: the error event terminates the stream.
        assert names[-1] == "error"

    @pytest.mark.asyncio
    async def test_put_settings_inference_key_rebuilds_runtime_once(
        self, app_client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """(d) PUT /api/settings with an ``inference.*`` key triggers exactly
        one runtime rebuild with the fresh snapshot (D7); a non-inference key
        triggers none."""
        from fixtures.llm_runtime import FakeLlmRuntime
        from vesta.inference import INFERENCE_LLM_MODEL

        fake = FakeLlmRuntime()
        monkeypatch.setattr("vesta.inference.get_runtime", lambda: fake)

        resp = await app_client.put(
            "/api/settings", json={"values": {"inference.llm.model": "new-model.gguf"}}
        )
        assert resp.status_code == 200
        assert resp.json()["values"]["inference.llm.model"] == "new-model.gguf"
        assert len(fake.rebuild_snapshots) == 1
        assert str(fake.rebuild_snapshots[0].get(INFERENCE_LLM_MODEL)) == "new-model.gguf"

        # A non-inference write must NOT rebuild.
        resp = await app_client.put("/api/settings", json={"values": {"logging.level": "INFO"}})
        assert resp.status_code == 200
        assert len(fake.rebuild_snapshots) == 1

    @pytest.mark.asyncio
    async def test_rebuild_failure_is_non_fatal(
        self, app_client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """D7: a failing rebuild is logged, not raised — the saved values are
        still returned so a bad endpoint stays correctable from the UI."""
        from fixtures.llm_runtime import FakeLlmRuntime

        fake = FakeLlmRuntime()

        async def _boom(snapshot: object) -> None:
            raise RuntimeError("rebuild exploded")

        monkeypatch.setattr("vesta.inference.get_runtime", lambda: fake)
        monkeypatch.setattr(fake, "rebuild", _boom)
        resp = await app_client.put(
            "/api/settings", json={"values": {"inference.llm.model": "another.gguf"}}
        )
        assert resp.status_code == 200
        assert resp.json()["values"]["inference.llm.model"] == "another.gguf"
