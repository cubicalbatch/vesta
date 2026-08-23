"""Cross-process mutual exclusion for index builds (AUDIT_0822 M7).

Exactly one indexer may write an archive's articles/chunks/vectors at a time.
The in-process guards (the runner's per-type semaphore, the API's pending-job
dedupe, the CLI's cancel of stranded job rows) cannot see across processes:
the detached ``vesta index`` CLI and the server's JobRunner both run
:class:`vesta.index.job.IndexZimJob` in different OS processes, and until now
nothing stopped both from rebuilding the same ``zim_id`` concurrently.

Deployment is one container on one host, so an OS pid is a valid liveness
signal: a lease row records the holder's pid and the next acquirer probes it
with ``os.kill(pid, 0)``. A dead pid — or a lease older than
:data:`STALE_AFTER`, the reboot/pid-reuse/wedged-holder escape hatch — is
taken over; a live foreign holder fails the acquire fast
(:class:`IndexLeaseHeld`). Both entry points claim through here: the job
acquires in ``IndexZimJob.run`` (so ANY build path is covered) and the API
trigger pre-checks for a clean 409.

Release is guarded by ``(owner_id, pid)``, so a late release from a superseded
holder can never delete a newer holder's lease — and a leaked lease (owner
killed between acquire and release) self-heals via the pid probe on the next
acquire.

Our own pid never contends: builds within one process are serialized by the
runner's ``jobs.max_concurrent.index_zim`` limit, so a same-pid row means our
own earlier state and is refreshed rather than refused.

No new package dependency: ``db`` is TYPE_CHECKING-only, already part of
``index/``'s declared dep set.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vesta.db.connection import Database

_log = logging.getLogger(__name__)

#: A live-looking lease older than this counts as abandoned and may be taken
#: over. Builds are multi-hour (full-Wikipedia scale), so this is deliberately
#: generous; it exists for what a pid probe cannot see: a wedged-but-alive
#: holder, or pid reuse after a reboot pointing at an unrelated process.
STALE_AFTER = _dt.timedelta(hours=12)


class IndexLeaseHeld(RuntimeError):
    """Raised by :func:`acquire_index_lease` when another live process holds
    the lease for the archive."""

    def __init__(self, holder: str) -> None:
        super().__init__(f"another index build is already running for this archive ({holder})")
        self.holder = holder


def describe(owner_id: str, pid: int) -> str:
    """Human-readable holder identity for errors and API 409 details."""
    return f"{owner_id}, pid {pid}"


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()


def _stale(acquired_at: str) -> bool:
    try:
        acquired = _dt.datetime.fromisoformat(acquired_at)
    except ValueError:
        return True  # unparseable ⇒ cannot vouch for the holder ⇒ treat as abandoned
    return (_dt.datetime.now(_dt.UTC) - acquired) >= STALE_AFTER


def _pid_alive(pid: int) -> bool:
    """Signal-0 probe: cheap cross-process liveness check."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    except OSError:
        return False
    return True


def _live_foreign_holder(owner_id: str, pid: int, acquired_at: str) -> str | None:
    """:func:`describe` of the holder iff it is a live OTHER process whose
    lease is not yet stale. Our own pid never contends (same-process builds
    are serialized upstream); dead or stale holders don't either."""
    if pid == os.getpid():
        return None
    if _pid_alive(pid) and not _stale(acquired_at):
        return describe(owner_id, pid)
    return None


async def active_holder(db: Database, zim_id: int) -> str | None:
    """Read-only check for entry points that must refuse fast (the API trigger
    answers 409 naming the holder instead of enqueueing a build destined to
    fail its own claim). Returns the live foreign holder's description, or
    ``None`` when the archive is free — or held only by a dead/stale/own-
    process lease, which :func:`acquire_index_lease` would take over."""
    async with (
        db.read() as conn,
        conn.execute(
            "SELECT owner_id, pid, acquired_at FROM index_leases WHERE zim_id=?", (zim_id,)
        ) as cur,
    ):
        row = await cur.fetchone()
    if row is None:
        return None
    return _live_foreign_holder(str(row["owner_id"]), int(row["pid"]), str(row["acquired_at"]))


async def acquire_index_lease(db: Database, zim_id: int, *, owner_id: str) -> None:
    """Claim the exclusive right to build ``zim_id``'s index in THIS process.

    Raises :class:`IndexLeaseHeld` when a live foreign process holds the
    lease. Takes over a free / own-process / dead / stale lease atomically:
    one ``BEGIN IMMEDIATE`` transaction holds the writer lock from the liveness
    read through the claim, so two simultaneous acquirers across processes
    resolve one-winner (the loser reads the winner's fresh row and refuses).
    """
    now = _now_iso()
    async with db.write() as conn:
        await conn.execute("BEGIN IMMEDIATE")
        cur = await conn.execute(
            "SELECT owner_id, pid, acquired_at FROM index_leases WHERE zim_id=?", (zim_id,)
        )
        row = await cur.fetchone()
        if row is not None:
            old_owner, old_pid, old_acquired = (
                str(row["owner_id"]),
                int(row["pid"]),
                str(row["acquired_at"]),
            )
            holder = _live_foreign_holder(old_owner, old_pid, old_acquired)
            if holder is not None:
                # Roll back out of the open transaction before raising —
                # Database.write()'s except arm would do it too, but being
                # explicit keeps the refusal self-contained.
                await conn.execute("ROLLBACK")
                raise IndexLeaseHeld(holder)
            # Take over (dead/stale/own): rewrite the row in place.
            await conn.execute(
                "UPDATE index_leases SET owner_id=?, pid=?, acquired_at=? WHERE zim_id=?",
                (owner_id, os.getpid(), now, zim_id),
            )
            return
        await conn.execute(
            "INSERT INTO index_leases(zim_id, owner_id, pid, acquired_at) VALUES(?,?,?,?)",
            (zim_id, owner_id, os.getpid(), now),
        )


async def release_index_lease(db: Database, zim_id: int, *, owner_id: str) -> None:
    """Drop our lease. Guarded by ``(owner_id, pid)`` so a superseded holder's
    late release cannot delete a successor's lease. Best-effort by design: a
    failed release self-heals via the pid liveness probe on the next acquire,
    so it must never mask the build outcome it runs under."""
    try:
        async with db.write() as conn:
            await conn.execute(
                "DELETE FROM index_leases WHERE zim_id=? AND owner_id=? AND pid=?",
                (zim_id, owner_id, os.getpid()),
            )
    except Exception:
        _log.exception("index.lease_release_failed", extra={"zim_id": zim_id})
