"""``main.py`` lifespan-ordering regression (jobs resume correctly after a
container kill mid-run).

Live verification found that ``JobRunner.start()`` (which
re-enqueues any job left ``running`` by a prior crash) ran *before*
``bind_store``/``bind_registry``/``bind_runtime`` in the lifespan. An ``await``
in between (``vector_store.ensure_default_table()``) yields control to the
newly-scheduled resume task before those bindings exist, so every resumed
``index_zim`` job crashed instantly with ``RuntimeError("index_zim: db/
registry/store not bound")`` instead of resuming. Fixed by moving
``runner.start()`` to run after every ``bind_*`` call, immediately before
``yield``. This test reproduces the crash scenario end to end through the real
``create_app()``/lifespan (not a bare ``JobRunner``, which doesn't touch this
ordering), across two separate app instances simulating a restart.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from pathlib import Path

import httpx
import pytest

from fixtures.tiny_zim import build_tiny_zim
from vesta.main import create_app
from vesta.vectors.sqlite_vec_store import SqliteVecStore

pytestmark = pytest.mark.asyncio


async def test_orphaned_running_index_job_resumes_without_lifespan_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zims_dir = tmp_path / "zims"
    zims_dir.mkdir(parents=True, exist_ok=True)
    build_tiny_zim(zims_dir / "tiny.zim")
    os.environ["data.dir"] = str(tmp_path)
    try:
        # First "process": register the tiny zim, enqueue an index build.
        app1 = create_app()
        async with app1.router.lifespan_context(app1):
            transport = httpx.ASGITransport(app=app1)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                zim_id = int((await client.get("/api/zims")).json()["archives"][0]["id"])
                resp = await client.post(f"/api/zims/{zim_id}/index", json={"depth": 1})
                job_id = int(resp.json()["job_id"])
                await asyncio.sleep(0.005)  # let it run at least one step

        # Simulate a hard crash mid-run: force the row back to 'running' with a
        # real (partial) checkpoint, regardless of how far app1 actually got.
        db_path = tmp_path / "vesta.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            "UPDATE jobs SET status='running', error=NULL, checkpoint=? WHERE id=?",
            (json.dumps({"done_count": 0, "depth": 1}), job_id),
        )
        conn.commit()
        conn.close()

        # Second "process" (the restart): JobRunner.start() re-enqueues this
        # job. Before the fix, this raced ahead of bind_store/bind_registry/
        # bind_runtime and crashed with "not bound". The race is timing-
        # dependent — a fast tmp-dir test may not naturally hit the same
        # interleaving a slower real box did — so force a real scheduling
        # yield at the point in the lifespan right after ``runner.start()``
        # used to sit (``ensure_default_table`` is the first ``await`` after
        # it), guaranteeing the resume task gets a chance to run before the
        # remaining ``bind_*`` calls, deterministically reproducing the race
        # against the old ordering.
        original_ensure = SqliteVecStore.ensure_default_table

        async def _ensure_default_table_with_yield(self: SqliteVecStore) -> None:
            await asyncio.sleep(0.005)
            await original_ensure(self)

        monkeypatch.setattr(
            SqliteVecStore, "ensure_default_table", _ensure_default_table_with_yield
        )

        app2 = create_app()
        async with app2.router.lifespan_context(app2):
            transport = httpx.ASGITransport(app=app2)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                job = None
                for _ in range(50):
                    await asyncio.sleep(0.005)
                    jobs = (await client.get("/api/jobs")).json()["jobs"]
                    job = next(j for j in jobs if j["id"] == job_id)
                    if job["status"] in {"done", "error", "cancelled"}:
                        break
                assert job is not None
                if job["error"] is not None:
                    assert "not bound" not in job["error"], (
                        "resumed job hit the lifespan race: bind_store/bind_registry/"
                        f"bind_runtime were not wired before resume — got {job['error']!r}"
                    )
    finally:
        os.environ.pop("data.dir", None)
