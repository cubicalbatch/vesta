"""``vesta index`` multi-archive queueing regression (cli.py).

``--zim`` may be repeated or comma-separated to index several archives
back-to-back in one CLI invocation; with no ``--zim``, every registered archive
is queued. The queue runs each archive's build sequentially and stops at the
first failure; specs that don't resolve to exactly one archive are skipped
(``_resolve_zim`` prints the problem and raises ``SystemExit``) instead of
aborting the whole queue.

Also covers the resume-sidecar lifecycle around ``_run_index`` (AUDIT_0822 M8):
``--fresh`` unlinks a stale sidecar before the job starts; a plain run resumes
from it; a paused run keeps it.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from vesta import config
from vesta.cli import _resolve_zims, _run_index, _run_index_many
from vesta.db.connection import Database
from vesta.db.migrations import run_migrations
from vesta.jobs.types import RESUME_CHECKPOINT_KEY

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


# ── resume-sidecar lifecycle around _run_index (AUDIT_0822 M8) ────────────────


class _FakeIndexJob:
    """Stands in for ``IndexZimJob``: records what ``_run_index`` handed it and
    whether the sidecar still existed when the job started."""

    def __init__(self) -> None:
        self.params: dict[str, Any] | None = None
        self.sidecar_exists_at_start: bool | None = None
        instances.append(self)

    async def run(self, handle: Any, params: Mapping[str, Any]) -> None:
        self.params = dict(params)
        cp_path = getattr(handle, "_cp", None)
        self.sidecar_exists_at_start = bool(cp_path.exists()) if cp_path is not None else None


instances: list[_FakeIndexJob] = []


def _seed_sidecar(data_dir: Path, zim_id: int = 1, *, depth: int = 1, done: int = 40) -> Path:
    cp = data_dir / f".index_progress_{zim_id}.json"
    cp.write_text(json.dumps({"done_count": done, "depth": depth}))
    return cp


def _index_args(zim: str, data_dir: Path, *, fresh: bool) -> argparse.Namespace:
    return argparse.Namespace(depth=1, zim=zim, fresh=fresh, data_dir=str(data_dir))


async def test_fresh_deletes_stale_sidecar_before_the_job_starts(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config.configure(env={})
    sidecar = _seed_sidecar(tmp_path)
    monkeypatch.setattr("vesta.index.job.IndexZimJob", _FakeIndexJob)

    code = await _run_index(
        type("State", (), {"db": db})(),  # type: ignore[arg-type]
        _index_args("wikipedia", tmp_path, fresh=True),
    )

    assert code == 0
    fake = instances[-1]
    # Gone BEFORE the job started — nothing stale survives into the rebuild…
    assert fake.sidecar_exists_at_start is False
    assert not sidecar.exists()
    # …and no resume cursor is offered to a --fresh run.
    assert fake.params is not None and RESUME_CHECKPOINT_KEY not in fake.params


async def test_plain_resume_still_honors_an_existing_sidecar(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config.configure(env={})
    sidecar = _seed_sidecar(tmp_path)
    # The zims row must still describe the lineage the sidecar was written
    # against: same depth, at least done_count articles committed.
    async with db.write() as conn:
        await conn.execute(
            "UPDATE zims SET index_status='error', index_depth=1, index_progress=40 WHERE id=1"
        )
    monkeypatch.setattr("vesta.index.job.IndexZimJob", _FakeIndexJob)

    code = await _run_index(
        type("State", (), {"db": db})(),  # type: ignore[arg-type]
        _index_args("wikipedia", tmp_path, fresh=False),
    )

    assert code == 0
    fake = instances[-1]
    assert fake.params is not None
    blob = fake.params.get(RESUME_CHECKPOINT_KEY)
    assert isinstance(blob, dict)
    assert blob["done_count"] == 40 and blob["depth"] == 1
    # A successful plain run still cleans the sidecar at the end (unchanged).
    assert not sidecar.exists()


async def test_paused_run_keeps_the_sidecar(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config.configure(env={})
    sidecar = _seed_sidecar(tmp_path)
    async with db.write() as conn:
        await conn.execute(
            "UPDATE zims SET index_status='paused', index_depth=1, index_progress=40 WHERE id=1"
        )
    monkeypatch.setattr("vesta.index.job.IndexZimJob", _FakeIndexJob)

    code = await _run_index(
        type("State", (), {"db": db})(),  # type: ignore[arg-type]
        _index_args("wikipedia", tmp_path, fresh=False),
    )

    assert code == 0
    assert sidecar.exists(), "a paused run keeps its resume sidecar"


async def test_server_wipe_leaves_no_stale_resume(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AUDIT_0824 M8: after DELETE /api/zims/{id}/index the row is wiped to
    depth 0 / progress 0, but the CLI sidecar survived (the API never touched
    it). A plain `vesta index --depth 1` must NOT resume at N into the emptied
    store — that would permanently skip articles 0..N."""
    config.configure(env={})
    sidecar = _seed_sidecar(tmp_path)
    async with db.write() as conn:
        await conn.execute(
            "UPDATE zims SET index_status='none', index_depth=0, index_progress=0 WHERE id=1"
        )
    monkeypatch.setattr("vesta.index.job.IndexZimJob", _FakeIndexJob)

    code = await _run_index(
        type("State", (), {"db": db})(),  # type: ignore[arg-type]
        _index_args("wikipedia", tmp_path, fresh=False),
    )

    assert code == 0
    fake = instances[-1]
    assert fake.params is not None
    assert RESUME_CHECKPOINT_KEY not in fake.params, "a wiped store must not be resumed into"
    assert not sidecar.exists(), "the stale cursor is dropped, not left for the next run"


async def test_same_depth_rebuild_drift_refuses_resume(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AUDIT_0824 M8: a server-side rebuild at the SAME depth wiped the store
    and re-committed only 10 articles before erroring. The CLI sidecar's N=40
    describes the destroyed lineage; resuming at 40 would skip articles 10..39
    of the new build forever."""
    config.configure(env={})
    sidecar = _seed_sidecar(tmp_path)
    async with db.write() as conn:
        await conn.execute(
            "UPDATE zims SET index_status='error', index_depth=1, index_progress=10 WHERE id=1"
        )
    monkeypatch.setattr("vesta.index.job.IndexZimJob", _FakeIndexJob)

    code = await _run_index(
        type("State", (), {"db": db})(),  # type: ignore[arg-type]
        _index_args("wikipedia", tmp_path, fresh=False),
    )

    assert code == 0
    fake = instances[-1]
    assert fake.params is not None
    assert RESUME_CHECKPOINT_KEY not in fake.params, "drifted lineage must not be resumed"
    assert not sidecar.exists()
