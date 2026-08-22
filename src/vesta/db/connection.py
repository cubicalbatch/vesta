"""SQLite connection management for a single-writer / multi-reader async app.

Design considerations and guarantees:

* **WAL + one writer.** SQLite serializes writes; under WAL, readers don't block
  the writer and vice-versa. We keep a single dedicated write connection guarded
  by a lock, and a small pool of read connections. Connections are never shared
  across concurrent tasks — aiosqlite runs each connection on its own thread, and
  crossing them corrupts the serialization guarantee.
* ``busy_timeout`` is set so a reader holding a read transaction can't make a
  writer fail with SQLITE_BUSY mid-progress write (job progress, downloads).
* ``synchronous=NORMAL`` is safe under WAL and makes
  per-second job-progress writes cheap.
* **vec0 extension load.** ``sqlite-vec``'s ``vec0`` virtual table
  needs the extension loaded on a connection before it can be created *or*
  queried, so every connection (writer + each reader) loads it at open time. The
  load is gated try/except (graceful degradation): a build without the
  compiled extension stays healthy — ``Capability.VECTORS`` simply stays off and
  the store is a no-op. aiosqlite owns each connection on its own worker thread,
  so the load runs *on that thread* via ``enable_load_extension`` (aiosqlite's
  async wrapper) and ``_execute(sqlite_vec.load, _conn)``; touching the wrapped
  ``sqlite3.Connection`` from the event-loop thread trips sqlite3's thread
  affinity check.

``db`` depends on nothing internal; all tunables arrive as
constructor arguments resolved from settings by the composition root (``main``).
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiosqlite
import structlog

_PRAGMA_FOREIGN = "PRAGMA foreign_keys = ON"


async def _load_vec0(conn: aiosqlite.Connection) -> bool:
    """Load the ``vec0`` extension on ``conn``'s worker thread.

    Returns ``True`` if loaded, ``False`` if the extension is unavailable — the
    caller treats ``False`` as "VECTORS capability off", never as an error.
    Both calls dispatch onto aiosqlite's worker thread so the wrapped
    ``sqlite3.Connection`` is touched only from its owning thread (sqlite3
    objects reject cross-thread use; reaching ``_conn`` from the event-loop
    thread trips the affinity check).
    """
    try:
        import sqlite_vec
    except ImportError:
        return False
    try:
        await conn.enable_load_extension(True)
        await conn._execute(sqlite_vec.load, conn._conn)  # type: ignore[no-untyped-call]
        return True
    except Exception as exc:
        structlog.get_logger("vesta.db").warning("db.vec0_load_failed", error=repr(exc))
        return False


class Database:
    """Owns one write connection and a bounded read pool.

    The writer is serialized via :pyattr:`_write_lock`; the read pool is a fixed
    set of connections protected by a semaphore so callers block only when every
    reader is in use.
    """

    def __init__(
        self,
        path: str,
        *,
        busy_timeout_ms: int = 5000,
        read_pool_size: int = 4,
    ) -> None:
        self._path = path
        self._busy_timeout_ms = busy_timeout_ms
        self._read_pool_size = read_pool_size
        self._write: aiosqlite.Connection | None = None
        self._read_pool: list[aiosqlite.Connection] = []
        self._write_lock = asyncio.Lock()
        self._read_sem = asyncio.Semaphore(read_pool_size)
        #: Whether the vec0 extension loaded on every connection. Determined
        #: during ``start()``; ``False`` ⇒ ``Capability.VECTORS`` stays off and
        #: the store is a no-op.
        self._vec0_available: bool = False
        self._started: bool = False

    @property
    def path(self) -> str:
        return self._path

    def vec0_available(self) -> bool:
        """True iff the ``vec0`` extension loaded on every connection."""
        return self._vec0_available

    async def start(self) -> None:
        """Open the writer and the read pool, applying tuned pragmas."""
        loaded: list[bool] = []
        self._write = await aiosqlite.connect(self._path)
        loaded.append(await self._apply_pragmas(self._write, writer=True))
        for _ in range(self._read_pool_size):
            conn = await aiosqlite.connect(self._path)
            loaded.append(await self._apply_pragmas(conn, writer=False))
            self._read_pool.append(conn)
        # VECTORS is on only if vec0 loaded on the writer AND every reader: a
        # vec0 virtual table queried from a connection missing the extension
        # errors, so a half-loaded pool is not a usable semantic index.
        self._started = True
        self._vec0_available = all(loaded)

    async def _apply_pragmas(self, conn: aiosqlite.Connection, *, writer: bool) -> bool:
        # Dict-style rows so callers read columns by name (jobs, settings, ...).
        conn.row_factory = sqlite3.Row
        # Load vec0 before any query — the store creates/queries vec0 virtual
        # tables and needs the extension live on every connection.
        # Best-effort; the returned flag rolls up into ``vec0_available()``.
        vec0_ok = await _load_vec0(conn)
        # foreign_keys is a no-op off but we want it on for cascading deletes.
        await conn.execute(_PRAGMA_FOREIGN)
        if writer:
            # WAL is persistent on the DB file; synchronous=NORMAL is connection-
            # local and safe under WAL.
            await conn.execute("PRAGMA journal_mode = WAL")
            await conn.execute("PRAGMA synchronous = NORMAL")
        # How long a writer waits for a lock before returning SQLITE_BUSY.
        await conn.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        await conn.commit()
        return vec0_ok

    async def stop(self) -> None:
        """Close every connection. Safe to call once at shutdown."""
        if self._write is not None:
            await self._write.close()
            self._write = None
        for conn in self._read_pool:
            await conn.close()
        self._read_pool.clear()
        self._started = False

    @asynccontextmanager
    async def write(self) -> AsyncIterator[aiosqlite.Connection]:
        """Serialized access to the single writer. Commits on clean exit."""
        if self._write is None:
            raise RuntimeError("Database.start() not called")
        async with self._write_lock:
            try:
                yield self._write
                await self._write.commit()
            except Exception:
                await self._write.rollback()
                raise

    @asynccontextmanager
    async def read(self) -> AsyncIterator[aiosqlite.Connection]:
        """Borrow a read connection from the pool for the duration of the block.

        Waits on the pool semaphore when every reader is checked out — the
        pool list is legitimately empty under concurrency, which is not the
        same state as "start() never ran" (a 1 MiB-chunk model download plus
        jobs polling used to trip that conflation and kill the download with
        a spurious ``Database.start() not called``).
        """
        if not self._started:
            raise RuntimeError("Database.start() not called")
        await self._read_sem.acquire()
        conn = self._read_pool.pop()
        try:
            yield conn
        finally:
            self._read_pool.append(conn)
            self._read_sem.release()
