"""Conversation persistence — the ``conversations``/``messages`` tables.

The migration ``0001_init.sql`` already declared these tables; this
module is the typed accessor. It lives in ``api/`` (not ``answer/``) because it
imports aiosqlite via the injected :class:`~vesta.db.connection.Database` —
keeping the DB dep out of ``answer/`` preserves the ≤3 dependency cap.
The composition root wires the concrete
:class:`SqliteConversationStore`.

``trace_json`` is prunable: multi-turn conversations with full traces
grow the DB, so :func:`prune_old_traces` nulls out traces older than the
``chat.trace_retention_days`` setting. Wired to run once at startup
(``main.py`` lifespan) — sufficient for a single-user appliance.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import aiosqlite

if TYPE_CHECKING:
    from vesta.db.connection import Database


@dataclass(frozen=True)
class StoredConversation:
    """One row of ``conversations`` for list views."""

    id: int
    title: str | None
    created_at: str | None
    updated_at: str | None


@dataclass(frozen=True)
class StoredMessage:
    """One row of ``messages``. ``sources_json``/``trace_json`` are nullable/prunable.

    The payload columns carry defaults so :meth:`SqliteConversationStore.
    list_recent_messages` can hand back slim (role, content) rows without
    materializing fields it never selected.
    """

    id: int
    role: str
    content: str | None
    sources_json: str | None = None
    trace_json: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_ms: int | None = None
    created_at: str | None = None


class SqliteConversationStore:
    """Persist conversations to the ``conversations``/``messages`` tables.

    Wraps the injected :class:`~vesta.db.connection.Database` and follows the
    same write()/read() context-manager pattern every DB accessor uses: the
    serialized writer commits on clean exit, readers borrow a pooled connection
    with ``sqlite3.Row`` dict-style access.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def create_conversation(self, title: str | None) -> int:
        now = _now()
        async with self._db.write() as conn:
            cur = await conn.execute(
                "INSERT INTO conversations(title, created_at, updated_at) VALUES(?,?,?)",
                (title, now, now),
            )
            return int(cur.lastrowid) if cur.lastrowid is not None else 0

    async def append_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        *,
        sources_json: str | None = None,
        trace_json: str | None = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        latency_ms: int | None = None,
    ) -> int:
        now = _now()
        async with self._db.write() as conn:
            cur = await conn.execute(
                "INSERT INTO messages(conversation_id, role, content, sources_json, "
                "trace_json, tokens_in, tokens_out, latency_ms, created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    conversation_id,
                    role,
                    content,
                    sources_json,
                    trace_json,
                    tokens_in,
                    tokens_out,
                    latency_ms,
                    now,
                ),
            )
            msg_id = int(cur.lastrowid) if cur.lastrowid is not None else 0
            await conn.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?",
                (now, conversation_id),
            )
            return msg_id

    async def list_messages(self, conversation_id: int, limit: int = 100) -> list[StoredMessage]:
        # DESC + Python-reverse keeps the NEWEST ``limit`` rows while returning
        # them oldest-first for display (AUDIT_0822 C3): an ASC LIMIT would pin
        # the oldest window once a conversation outgrows ``limit``, silently
        # dropping every recent turn from the detail view.
        async with self._db.read() as conn:
            cur = await conn.execute(
                "SELECT id, role, content, sources_json, trace_json, tokens_in, "
                "tokens_out, latency_ms, created_at FROM messages "
                "WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            )
            rows = list(await cur.fetchall())
        rows.reverse()
        return [_row_to_message(r) for r in rows]

    async def list_recent_messages(
        self, conversation_id: int, limit: int = 200
    ) -> list[StoredMessage]:
        """The most recent ``limit`` turns, oldest-first, with only role/content.

        Chat-history loading never reads source cards or traces — but full-row
        selects deserialized up to ``limit`` stored trace payloads per turn just
        to throw them away (AUDIT_0822 C8). This slim select touches only
        ``(role, content)``; every other ``StoredMessage`` field is left at its
        default. Newest-window semantics as in :meth:`list_messages`: ordered
        DESC then reversed so long conversations keep their most recent turns.
        """
        async with self._db.read() as conn:
            cur = await conn.execute(
                "SELECT role, content FROM messages "
                "WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            )
            rows = list(await cur.fetchall())
        rows.reverse()
        return [StoredMessage(id=0, role=str(row["role"]), content=row["content"]) for row in rows]

    async def list_conversations(self, limit: int = 50) -> list[StoredConversation]:
        async with self._db.read() as conn:
            cur = await conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = await cur.fetchall()
        return [_row_to_conversation(r) for r in rows]

    async def get_conversation(self, conversation_id: int) -> StoredConversation | None:
        async with self._db.read() as conn:
            cur = await conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id=?",
                (conversation_id,),
            )
            row = await cur.fetchone()
        return _row_to_conversation(row) if row is not None else None

    async def delete_conversation(self, conversation_id: int) -> bool:
        # messages cascade via ON DELETE CASCADE (0001_init.sql); foreign_keys
        # is ON on every connection (db/connection.py).
        async with self._db.write() as conn:
            cur = await conn.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
            return bool(cur.rowcount > 0)


def _row_to_message(row: aiosqlite.Row) -> StoredMessage:
    return StoredMessage(
        id=int(row["id"]),
        role=str(row["role"]),
        content=row["content"],
        sources_json=row["sources_json"],
        trace_json=row["trace_json"],
        tokens_in=_int_or_none(row["tokens_in"]),
        tokens_out=_int_or_none(row["tokens_out"]),
        latency_ms=_int_or_none(row["latency_ms"]),
        created_at=row["created_at"],
    )


def _row_to_conversation(row: aiosqlite.Row) -> StoredConversation:
    return StoredConversation(
        id=int(row["id"]),
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _int_or_none(v: Any) -> int | None:
    return int(v) if v is not None else None


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()


async def prune_old_traces(db: Database, retention_days: int) -> int:
    """Null ``trace_json`` on messages older than ``retention_days`` (09 Traps).

    Multi-turn conversations with full traces grow the DB; the retention setting
    bounds that. ``retention_days <= 0`` ⇒ keep indefinitely (no-op). Returns the
    number of rows pruned. Runs at startup — sufficient for a single-user
    appliance; a periodic job is overkill until conversation volume justifies it.
    """
    if retention_days <= 0:
        return 0
    cutoff = (
        (_dt.datetime.now(_dt.UTC) - _dt.timedelta(days=retention_days))
        .replace(microsecond=0)
        .isoformat()
    )
    async with db.write() as conn:
        cur = await conn.execute(
            "UPDATE messages SET trace_json=NULL WHERE trace_json IS NOT NULL AND created_at < ?",
            (cutoff,),
        )
        affected = cur.rowcount
        return affected if affected > 0 else 0


__all__ = [
    "SqliteConversationStore",
    "StoredConversation",
    "StoredMessage",
    "prune_old_traces",
]
