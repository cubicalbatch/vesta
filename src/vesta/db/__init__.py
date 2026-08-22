"""SQLite layer: schema, migrations, connection management.

``db`` depends on nothing internal. Every tunable arrives as a constructor
argument resolved by the composition root, which keeps the dependency arrow
one-way.
"""

from __future__ import annotations

from vesta.db.connection import Database
from vesta.db.migrations import (
    MigrationError,
    available_migrations,
    current_version,
    run_migrations,
)

__all__ = [
    "Database",
    "MigrationError",
    "available_migrations",
    "current_version",
    "run_migrations",
]
