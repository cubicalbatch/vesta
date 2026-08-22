"""``vesta index`` multi-archive queueing regression (cli.py).

``--zim`` may be repeated or comma-separated to index several archives
back-to-back in one CLI invocation; with no ``--zim``, every registered archive
is queued. The queue runs each archive's build sequentially and stops at the
first failure; specs that don't resolve to exactly one archive are skipped
(``_resolve_zim`` prints the problem and raises ``SystemExit``) instead of
aborting the whole queue.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from vesta.cli import _resolve_zims, _run_index_many
from vesta.db.connection import Database
from vesta.db.migrations import run_migrations

pytestmark = pytest.mark.asyncio


async def _seed_zims(db: Database) -> None:
    async with db.write() as conn:
        await run_migrations(conn)
        for zim_id, name, filename in (
            (1, "wikipedia_en_top", "wikipedia_en_top_nopic_2026-06.zim"),
            (2, "wikivoyage_en_europe", "wikivoyage_en_europe_nopic_2026-06.zim"),
            (3, "history.stackexchange.com_en_all", "history.stackexchange.com_en_all_2026-02.zim"),
        ):
            await conn.execute(
                "INSERT INTO zims(id, path, status, name, filename, enabled) "
                "VALUES (?, '/fake.zim', 'ready', ?, ?, 1)",
                (zim_id, name, filename),
            )


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    """Seeded Database, always stopped on teardown — including on assertion
    failure. A bare trailing ``await db.stop()`` in each test body would be
    skipped whenever an earlier assert fails, leaking the aiosqlite worker
    thread; since that thread is non-daemon, the leak doesn't just fail the
    test, it hangs the whole pytest process on exit."""
    database = Database(str(tmp_path / "vesta.db"), busy_timeout_ms=2000)
    await database.start()
    await _seed_zims(database)
    try:
        yield database
    finally:
        await database.stop()


class _FakeRunIndex:
    """Records the queue order it was called with; ``fails_on`` raises."""

    def __init__(self) -> None:
        self.calls: list[int] = []
        self.fails_on: set[int] = set()

    async def __call__(self, _state: object, args: object) -> int:  # pragma: no cover - tracing
        zim_id = int(args.zim)  # type: ignore[attr-defined]
        self.calls.append(zim_id)
        return 1 if zim_id in self.fails_on else 0


def _args(*zims: str, fresh: bool = False) -> object:
    # ``action="append"`` (real argparse) always yields a list or None — match
    # that contract exactly rather than the tuple ``*zims`` collects, so these
    # tests exercise the same shape ``_run_index_many`` sees from the real CLI.
    return type(
        "Args", (), {"zim": list(zims) or None, "depth": 1, "fresh": fresh, "data_dir": None}
    )()


async def test_runs_selected_archives_in_order(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeRunIndex()
    monkeypatch.setattr("vesta.cli._run_index", fake)

    code = await _run_index_many(  # type: ignore[arg-type]
        type("State", (), {"db": db})(), _args("wikipedia", "3", "history")
    )

    assert code == 0
    assert fake.calls == [1, 3], f"expected [1, 3] in queue order, got {fake.calls}"


async def test_no_zim_queues_every_archive(db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRunIndex()
    monkeypatch.setattr("vesta.cli._run_index", fake)

    code = await _run_index_many(  # type: ignore[arg-type]
        type("State", (), {"db": db})(), _args()
    )

    assert code == 0
    assert fake.calls == [1, 2, 3]


async def test_failure_stops_the_queue(db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRunIndex()
    fake.fails_on = {2}
    monkeypatch.setattr("vesta.cli._run_index", fake)

    code = await _run_index_many(  # type: ignore[arg-type]
        type("State", (), {"db": db})(), _args("1", "2", "3")
    )

    assert code == 1
    assert fake.calls == [1, 2], "queue must stop at the first failing archive"


async def test_unresolvable_spec_is_skipped_not_aborting(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeRunIndex()
    monkeypatch.setattr("vesta.cli._run_index", fake)

    code = await _run_index_many(  # type: ignore[arg-type]
        type("State", (), {"db": db})(), _args("does-not-exist", "wikipedia")
    )

    assert code == 0
    assert fake.calls == [1], "unresolvable spec is skipped, later specs still run"


async def test_single_archive_delegates_directly(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeRunIndex()
    monkeypatch.setattr("vesta.cli._run_index", fake)

    code = await _run_index_many(  # type: ignore[arg-type]
        type("State", (), {"db": db})(), _args("wikipedia")
    )

    assert code == 0
    assert fake.calls == [1]


async def test_resolve_zims_skips_systemexit_specs(db: Database) -> None:
    resolved = await _resolve_zims(db, ["wikipedia", "nope", "2"])

    assert resolved == [1, 2]
