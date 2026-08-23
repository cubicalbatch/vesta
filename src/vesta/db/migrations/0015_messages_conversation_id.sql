-- Migration 0015 — index messages.conversation_id (AUDIT_0822 P1 / C7).
--
-- Every chat turn loads the conversation's recent messages
-- (api/conversation_store.py), but the FK alone gave SQLite no access path
-- other than a full scan of ``messages`` — a table that only grows (rows are
-- never deleted; retention nulls ``trace_json`` in place). The interactive
-- path pays that scan every turn. Naming follows the existing
-- <table>_<column> convention (articles_zim, chunks_zim, ...).

CREATE INDEX IF NOT EXISTS messages_conversation_id ON messages(conversation_id);
