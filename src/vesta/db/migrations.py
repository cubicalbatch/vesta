"""Forward-only, numbered SQL migrations applied at startup.

We track applied state with ``PRAGMA user_version`` (the SQLite-blessed integer
on the file header). Migrations are ``NNNN_name.sql`` files shipped beside this
module; each runs in its own transaction and bumps ``user_version`` to ``NNNN``.

Why not Alembic / an ORM: overkill for a single SQLite file. Why
forward-only: rollback of a schema migration on a system of record is a footgun
the user cannot afford, and we never need it in a single-binary appliance.

The invariant this protects: an empty DB *and* a DB
from a previous version must both reach the same state after the runner fires,
because migrations run before anything reads. Re-running the runner is a no-op.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import aiosqlite

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class MigrationError(RuntimeError):
    """Raised when migrations cannot be applied consistently."""


def available_migrations() -> list[tuple[int, str, Path]]:
    """Return ``(number, name, path)`` for every migration file, ordered."""
    found: list[tuple[int, str, Path]] = []
    for path in sorted(_MIGRATIONS_DIR.glob("*.sql")):
        stem = path.stem  # e.g. "0001_init"
        number_str, _, name = stem.partition("_")
        try:
            number = int(number_str)
        except ValueError as exc:
            raise MigrationError(f"bad migration filename {path.name}") from exc
        found.append((number, name, path))
    return found


async def current_version(write_conn: aiosqlite.Connection) -> int:
    async with write_conn.execute("PRAGMA user_version") as cur:
        row = await cur.fetchone()
    return int(row[0]) if row else 0


async def run_migrations(write_conn: aiosqlite.Connection) -> list[int]:
    """Apply every pending migration in order. Returns the numbers applied.

    Idempotent: calling it again applies nothing. Safe on a freshly created
    empty DB (user_version starts at 0).
    """
    version = await current_version(write_conn)
    applied: list[int] = []
    for number, _name, path in available_migrations():
        if number <= version:
            continue
        sql = path.read_text(encoding="utf-8")
        await write_conn.executescript(sql)
        # executescript issues an implicit commit; bump the version explicitly.
        await write_conn.execute(f"PRAGMA user_version = {number}")
        await write_conn.commit()
        applied.append(number)
        version = number
    return applied


def migrations_table_sql() -> str:
    """The init SQL, exposed for tests that assert schema completeness."""
    res = resources.files(__package__).joinpath("migrations/0001_init.sql")
    return res.read_text(encoding="utf-8")
