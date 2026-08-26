"""DB layer: migrations are idempotent, schema is complete, vectors excluded."""

from __future__ import annotations

from pathlib import Path

import pytest

from vesta.db import migrations as db_migrations
from vesta.db.connection import Database
from vesta.db.migrations import (
    MigrationError,
    available_migrations,
    current_version,
    run_migrations,
)

EXPECTED_TABLES = {
    "zims",
    "articles",
    "aliases",  # migration 0002
    "chunks",  # migration 0004 — chunk metadata, the uniform index unit
    "index_meta",  # migration 0004 — per-archive embedder compat record
    "jobs",
    "models",
    "catalog_entries",
    "conversations",
    "messages",
    "settings",
    "eval_runs",
    "answer_runs",  # migration 0005 — answer benchmark runs
    "bench_runs",  # migration 0009 — unified benchmark runs
    "bench_question_results",  # migration 0009
    "bench_judge_cache",  # migration 0009
    "article_media",  # media-ZIM asset manifest (migration 0008)
    "article_documents",  # nautiluszim document-library catalog (migration 0013)
    "index_leases",  # cross-process build exclusion, one row per building archive (migration 0014)
    # catalog_fts is a virtual table (FTS5); it appears in sqlite_master too,
    # but FTS5 virtual tables + their shadow tables are excluded here to keep the
    # set to the logical base tables (asserted separately in test_catalog_fts).
}


async def _table_names(db: Database) -> set[str]:
    async with db.read() as conn:
        cur = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        rows = await cur.fetchall()
    return {row[0] for row in rows}


def _migration_sql(number: int) -> str:
    """Read a migration file's SQL by its numeric prefix."""
    for _number, _name, path in available_migrations():
        if _number == number:
            return path.read_text(encoding="utf-8")
    raise AssertionError(f"no migration {number}")


@pytest.mark.asyncio
async def test_migration_0001_exists_and_numbered() -> None:
    migrations = available_migrations()
    assert migrations[0][0] == 1
    assert migrations[0][1] == "init"


@pytest.mark.asyncio
async def test_fresh_db_reaches_full_schema(tmp_db_path: Path) -> None:
    db = Database(str(tmp_db_path), busy_timeout_ms=1000)
    await db.start()
    async with db.write() as conn:
        applied = await run_migrations(conn)
        assert (
            applied
            == [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
            ]
        )  # ...+bench_runs +token usage +retire_agentic_settings +retire_single_shot +article_documents +index_leases +messages_conversation_id +eval_runs_status
        assert await current_version(conn) == 16
    tables = await _table_names(db)
    await db.stop()
    assert tables >= EXPECTED_TABLES


@pytest.mark.asyncio
async def test_no_literal_vectors_table_only_dim_suffixed(tmp_db_path: Path) -> None:
    """The vec0 tables are ``vectors_d{N}`` (one per embedding dim),
    NEVER the literal ``vectors``. The vec0 tables themselves are created
    lazily by ``SqliteVecStore`` (not the migration), so right after migration the
    schema holds the plain ``chunks``/``index_meta`` tables and no vec0 table at all
    — this test guards the literal-name reservation and confirms the metadata tables
    landed. See migration 0004 for why vec0 DDL lives in the store, not here."""
    db = Database(str(tmp_db_path), busy_timeout_ms=1000)
    await db.start()
    async with db.write() as conn:
        await run_migrations(conn)
    tables = await _table_names(db)
    await db.stop()
    assert "vectors" not in tables
    assert "chunks" in tables
    assert "index_meta" in tables


@pytest.mark.asyncio
async def test_re_running_migrations_is_noop(tmp_db_path: Path) -> None:
    db = Database(str(tmp_db_path), busy_timeout_ms=1000)
    await db.start()
    async with db.write() as conn:
        await run_migrations(conn)
        again = await run_migrations(conn)
    await db.stop()
    assert again == []


@pytest.mark.asyncio
async def test_failed_migration_rolls_back_atomically(
    tmp_db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A migration that fails mid-script must leave NOTHING behind (AUDIT_0822
    M6): each script runs as one ``BEGIN IMMEDIATE … COMMIT`` transaction, so a
    failure rolls back the earlier DDL *and* the version bump — the next boot
    retries from a clean slate instead of dying forever on re-run against
    half-applied DDL. Proven here in four beats: raises ``MigrationError``;
    ``user_version`` unchanged; the script's own earlier ``CREATE TABLE`` is
    gone; a corrected retry applies cleanly on the same connection."""
    # Reach the real latest version first (real 0001-0014), THEN substitute
    # one extra pending migration living in tmp_path — no packaged .sql is
    # touched, and the runner's available_migrations() lookup resolves through
    # module globals at call time, so the patch takes effect for pending runs.
    db = Database(str(tmp_db_path), busy_timeout_ms=1000)
    await db.start()
    async with db.write() as conn:
        applied = await run_migrations(conn)
        assert len(applied) == 16
        assert await current_version(conn) == 16

        probe_sql = tmp_path / "0017_atomicity_probe.sql"
        monkeypatch.setattr(
            db_migrations,
            "available_migrations",
            lambda: [(17, "atomicity_probe", probe_sql)],
        )

        # First half succeeds (CREATE TABLE), second half errors mid-script.
        probe_sql.write_text(
            "CREATE TABLE m6_atomicity_probe (id INTEGER PRIMARY KEY);\n"
            "INSERT INTO no_such_table VALUES (1);\n",
            encoding="utf-8",
        )
        with pytest.raises(MigrationError):
            await run_migrations(conn)

        assert await current_version(conn) == 16  # bump rolled back
        async with conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='m6_atomicity_probe'"
        ) as cur:
            assert await cur.fetchone() is None  # partial DDL rolled back

        # Corrected migration applies cleanly on the same boot/connection.
        probe_sql.write_text(
            "CREATE TABLE m6_atomicity_probe (id INTEGER PRIMARY KEY);\n"
            "INSERT INTO m6_atomicity_probe VALUES (1);\n",
            encoding="utf-8",
        )
        assert await run_migrations(conn) == [17]
        assert await current_version(conn) == 17
        async with conn.execute("SELECT COUNT(*) FROM m6_atomicity_probe") as cur:
            row = await cur.fetchone()
        assert row is not None and row[0] == 1
    await db.stop()


@pytest.mark.asyncio
async def test_articles_zim_index_created(tmp_db_path: Path) -> None:
    db = Database(str(tmp_db_path), busy_timeout_ms=1000)
    await db.start()
    async with db.write() as conn:
        await run_migrations(conn)
        cur = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='articles'"
        )
        indexes = {row[0] for row in await cur.fetchall()}
    await db.stop()
    assert "articles_zim" in indexes


@pytest.mark.asyncio
async def test_messages_conversation_id_index_created(tmp_db_path: Path) -> None:
    """AUDIT_0822 P1/C7: every chat turn reads a conversation's recent messages,
    so ``messages.conversation_id`` must have an index — the bare FK left the
    per-turn history query a full scan of an unbounded table."""
    db = Database(str(tmp_db_path), busy_timeout_ms=1000)
    await db.start()
    async with db.write() as conn:
        await run_migrations(conn)
        cur = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='messages'"
        )
        indexes = {row[0] for row in await cur.fetchall()}
    await db.stop()
    assert "messages_conversation_id" in indexes


@pytest.mark.asyncio
async def test_previous_version_reaches_same_state(tmp_db_path: Path) -> None:
    """An empty DB and a DB 'from a previous version' must end in the same state
    (01-foundations Traps)."""
    db = Database(str(tmp_db_path), busy_timeout_ms=1000)
    await db.start()
    async with db.write() as conn:
        # Pretend we are at version 0 (no migrations applied) then run.
        await run_migrations(conn)
        v1 = await current_version(conn)
        first_tables = await _table_names(db)
        # second run keeps the same version + tables
        await run_migrations(conn)
        v2 = await current_version(conn)
    second_tables = await _table_names(db)
    await db.stop()
    assert v1 == v2 == 16  # 0001-0016: init+aliases+eval pins+vectors+answer_runs+catalog_fts+
    #                   zim_kind+article_media+bench_runs+token_usage+retire_agentic_settings+retire_single_shot
    #                   +article_documents+index_leases+messages_conversation_id+eval_runs_status
    assert first_tables == second_tables


@pytest.mark.asyncio
async def test_migration_0011_retires_agentic_settings(tmp_db_path: Path) -> None:
    """Migration 0011: the removed loop/gate/probe/reformulate settings are dropped
    from the settings table, and a stored ``answer.strategy='agentic'`` is reset
    to the default so the resolver's choices validation no longer fails."""
    db = Database(str(tmp_db_path), busy_timeout_ms=1000)
    await db.start()
    async with db.write() as conn:
        # Simulate a pre-17.4 DB: apply 0001-0010, then seed stale settings.
        for number in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
            sql = _migration_sql(number)
            await conn.executescript(sql)
            await conn.execute(f"PRAGMA user_version = {number}")
            await conn.commit()
        await conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?), (?, ?), (?, ?), (?, ?), (?, ?)",
            (
                "answer.strategy",
                "agentic",
                "answer.loop.max_rounds",
                "2",
                "answer.gate.rho_target",
                "0.25",
                "answer.probe.enabled",
                "1",
                "answer.reformulate.enabled",
                "1",
            ),
        )
        await conn.commit()
        # Apply 0011.
        sql = _migration_sql(11)
        await conn.executescript(sql)
        await conn.execute("PRAGMA user_version = 11")
        await conn.commit()
        cur = await conn.execute("SELECT key, value FROM settings ORDER BY key")
        rows = dict(await cur.fetchall())
    await db.stop()
    assert rows == {"answer.strategy": "single_shot"}  # stale keys gone, strategy repaired


@pytest.mark.asyncio
async def test_migration_0012_retires_single_shot_settings(tmp_db_path: Path) -> None:
    """The ``single_shot`` strategy is retired: a stored ``answer.strategy`` of
    ``single_shot`` (or a residual ``agentic``) is reset to ``sources_only``, and
    the seven single_shot-only ``answer.*`` knobs are dropped, so the resolver's
    ``choices`` validation no longer fails on the next snapshot."""
    db = Database(str(tmp_db_path), busy_timeout_ms=1000)
    await db.start()
    async with db.write() as conn:
        # Apply 0001-0011, then seed stale single_shot settings.
        for number in range(1, 12):
            sql = _migration_sql(number)
            await conn.executescript(sql)
            await conn.execute(f"PRAGMA user_version = {number}")
            await conn.commit()
        await conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?), (?, ?), (?, ?)",
            (
                "answer.strategy",
                "single_shot",
                "answer.abstention.top_score_floor",
                "0.25",
                "answer.context.min_score",
                "0.02",
            ),
        )
        await conn.commit()
        # Apply 0012.
        sql = _migration_sql(12)
        await conn.executescript(sql)
        await conn.execute("PRAGMA user_version = 12")
        await conn.commit()
        cur = await conn.execute("SELECT key, value FROM settings ORDER BY key")
        rows = dict(await cur.fetchall())
    await db.stop()
    assert rows == {"answer.strategy": "sources_only"}  # strategy repaired, dead knobs gone


@pytest.mark.asyncio
async def test_read_waits_when_pool_exhausted(tmp_db_path: Path) -> None:
    """A busy read pool must block on the semaphore, not raise
    ``Database.start() not called`` (B2 regression: a 1 MiB-chunk model
    download checkpoints faster than the pool drains while the UI polls jobs,
    which used to kill the download mid-flight)."""
    import asyncio

    db = Database(str(tmp_db_path), read_pool_size=1)
    await db.start()
    try:
        async with db.read():  # exhaust the single reader
            second = asyncio.create_task(_read_once(db))
            await asyncio.sleep(0.05)
            assert not second.done()  # still waiting, not raised
        await asyncio.wait_for(second, timeout=2.0)
    finally:
        await db.stop()


async def _read_once(db: Database) -> None:
    async with db.read():
        pass
