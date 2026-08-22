"""Tiny ``settings`` table accessors.

``db`` depends on nothing internal, so the settings-table
read/write that backs the config layer lives here as plain SQL — it returns a
``Mapping[str, str]`` that ``main`` hands to :mod:`vesta.config.resolution`, and
that ``PUT /api/settings`` writes through. No business logic.
"""

from __future__ import annotations

import aiosqlite


async def load_settings(conn: aiosqlite.Connection) -> dict[str, str]:
    """Return every row of the ``settings`` table as ``{key: value}``."""
    result: dict[str, str] = {}
    async with conn.execute("SELECT key, value FROM settings") as cur:
        rows = await cur.fetchall()
    for key, value in rows:
        result[key] = value
    return result


async def upsert_setting(conn: aiosqlite.Connection, key: str, value: str, now: str) -> None:
    """Insert or update one setting row. Caller holds a write transaction."""
    await conn.execute(
        "INSERT INTO settings(key, value, updated_at) VALUES(?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
        "updated_at = excluded.updated_at",
        (key, value, now),
    )
