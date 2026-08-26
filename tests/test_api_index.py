"""Semantic-index API tests.

Covers the three new endpoints through the composition root, exactly as the
dev console calls them:

* ``POST /api/zims/{id}/index/estimate`` — the honest pre-commit cost;
* ``POST /api/zims/{id}/index`` — enqueue a resumable build (the job type is
  registered by ``main``'s import; the build itself then fails fast here
  because no embed model is present in the tmp data dir — the API contract is
  the enqueue, not the outcome);
* ``GET /api/zims/{id}/index`` — depth/status/progress projection;

plus the archive-removal cascade: deleting an archive must delete its vectors
(vec0 tables are not FK-cascaded, so the composition root wires it).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from vesta.vectors import get_store
from vesta.vectors.contracts import VectorRow

pytestmark = pytest.mark.asyncio


# ── estimate ──────────────────────────────────────────────────────────────────


async def test_estimate_returns_range_and_disk(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    client, zim_id = app_client_with_zim
    resp = await client.post(f"/api/zims/{zim_id}/index/estimate", json={"depth": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["depth"] == 1
    assert body["articles"] > 0
    assert body["calibrated"] is False  # nothing timed yet
    assert body["disk_bytes_low"] < body["disk_bytes_expected"] < body["disk_bytes_high"]


async def test_estimate_rejects_bad_depth_and_unknown_archive(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    client, zim_id = app_client_with_zim
    assert (
        await client.post(f"/api/zims/{zim_id}/index/estimate", json={"depth": 0})
    ).status_code == 400
    assert (
        await client.post(f"/api/zims/{zim_id}/index/estimate", json={"depth": 4})
    ).status_code == 400
    assert (
        await client.post("/api/zims/99999/index/estimate", json={"depth": 1})
    ).status_code == 404


# ── trigger ───────────────────────────────────────────────────────────────────


async def test_trigger_enqueues_index_job(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    client, zim_id = app_client_with_zim
    resp = await client.post(f"/api/zims/{zim_id}/index", json={"depth": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["zim_id"] == zim_id and body["depth"] == 1
    assert body["job_id"] > 0
    # The job is visible through the jobs API (type registered by main's import).
    jobs = (await client.get("/api/jobs")).json()["jobs"]
    assert any(j["type"] == "index_zim" and j["id"] == body["job_id"] for j in jobs)


async def test_trigger_rejects_bad_depth_and_unknown_archive(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    client, zim_id = app_client_with_zim
    assert (await client.post(f"/api/zims/{zim_id}/index", json={"depth": 0})).status_code == 400
    assert (await client.post(f"/api/zims/{zim_id}/index", json={"depth": 9})).status_code == 400
    assert (await client.post("/api/zims/99999/index", json={"depth": 1})).status_code == 404


async def test_trigger_is_idempotent_for_a_pending_job(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    """A double-click (or two tabs) must not queue a second build for the same
    archive while one is already queued/running/paused (Workstream A, Issue 1).

    Seeds a non-terminal ``index_zim`` job directly rather than racing the
    real one, since the fixture's build fails fast (no embed model in the tmp
    data dir) and could already be terminal by the time a second POST lands.
    """
    client, zim_id = app_client_with_zim
    app = client._transport.app  # type: ignore[attr-defined]
    db = app.state.vesta.db  # type: ignore[attr-defined]
    async with db.write() as conn:
        cur = await conn.execute(
            "INSERT INTO jobs(type, target, params, status, progress, total, "
            "created_at, updated_at) VALUES('index_zim', ?, ?, 'queued', 0, 0, "
            "'2026-01-01T00:00:00', '2026-01-01T00:00:00')",
            (str(zim_id), f'{{"zim_id": {zim_id}, "depth": 2}}'),
        )
        existing_job_id = int(cur.lastrowid)

    resp = await client.post(f"/api/zims/{zim_id}/index", json={"depth": 3})
    assert resp.status_code == 200
    body = resp.json()
    # References the existing queued job (and its depth) rather than creating one.
    assert body == {"zim_id": zim_id, "depth": 2, "job_id": existing_job_id}

    jobs = (await client.get("/api/jobs")).json()["jobs"]
    index_jobs = [j for j in jobs if j["type"] == "index_zim"]
    assert len(index_jobs) == 1 and index_jobs[0]["id"] == existing_job_id


async def test_trigger_creates_new_job_once_previous_is_terminal(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    """Once the pending job reaches a terminal status, a fresh trigger is free
    to enqueue a new one — dedupe only guards the non-terminal window."""
    client, zim_id = app_client_with_zim
    app = client._transport.app  # type: ignore[attr-defined]
    db = app.state.vesta.db  # type: ignore[attr-defined]
    async with db.write() as conn:
        await conn.execute(
            "INSERT INTO jobs(type, target, params, status, progress, total, "
            "created_at, updated_at, finished_at) VALUES('index_zim', ?, '{}', "
            "'done', 0, 0, '2026-01-01T00:00:00', '2026-01-01T00:00:00', "
            "'2026-01-01T00:00:00')",
            (str(zim_id),),
        )

    resp = await client.post(f"/api/zims/{zim_id}/index", json={"depth": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["depth"] == 1

    jobs = (await client.get("/api/jobs")).json()["jobs"]
    index_jobs = [j for j in jobs if j["type"] == "index_zim"]
    assert len(index_jobs) == 2


# ── status ────────────────────────────────────────────────────────────────────


async def test_index_status_defaults(app_client_with_zim: tuple[httpx.AsyncClient, int]) -> None:
    client, zim_id = app_client_with_zim
    resp = await client.get(f"/api/zims/{zim_id}/index")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "zim_id": zim_id,
        "depth": 0,
        "status": "none",
        "progress": 0,
        "total": 0,
        "embedding_model": None,
    }


# ── lowering to depth 0 ───────────────────────────────────────────────────────


async def test_delete_index_wipes_vectors_meta_and_row(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    client, zim_id = app_client_with_zim
    store = get_store()
    assert store is not None
    db = store._db
    async with db.write() as conn:
        await conn.execute(
            "INSERT INTO articles(zim_id, entry_path, title) VALUES (?, ?, ?)",
            (zim_id, "A/Wipe_Test", "Wipe Test"),
        )
        await conn.execute(
            "UPDATE zims SET index_depth=1, index_status='complete', "
            "embedding_model='m', embedding_dim=384 WHERE id=?",
            (zim_id,),
        )
    await store.upsert(
        [
            VectorRow(
                id=900002,
                zim_id=zim_id,
                article_id=1,
                embedding=np.array([1.0] * 384, dtype=np.float32),
                char_start=0,
                char_end=10,
            )
        ]
    )
    from vesta.vectors.contracts import IndexMeta

    await store.record_index_meta(zim_id, IndexMeta("m", 384, "", "", "mean", True))
    assert await store.describe(zim_id) is not None

    resp = await client.delete(f"/api/zims/{zim_id}/index")
    assert resp.status_code == 200 and resp.json()["depth"] == 0

    # Vectors, chunks, and the compat record are all gone (lowering
    # depth deletes vectors; the archive reads as genuinely un-indexed).
    assert (await store.stats()).per_zim.get(zim_id, 0) == 0
    assert await store.describe(zim_id) is None
    body = (await client.get(f"/api/zims/{zim_id}/index")).json()
    assert body["depth"] == 0 and body["status"] == "none"
    assert body["embedding_model"] is None


async def test_delete_index_drops_the_cli_resume_sidecar(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    """AUDIT_0824 M8: the wipe must also unlink the detached-CLI resume
    sidecar — a surviving ``done_count=N`` on disk would let a later plain
    ``vesta index --depth d`` "resume" into this emptied store."""
    from vesta import config

    client, zim_id = app_client_with_zim
    sidecar = Path(config.get(config.DATA_DIR)) / f".index_progress_{zim_id}.json"
    sidecar.write_text(json.dumps({"done_count": 40, "depth": 1}))

    resp = await client.delete(f"/api/zims/{zim_id}/index")
    assert resp.status_code == 200 and resp.json()["depth"] == 0
    assert not sidecar.exists(), "the wiped archive's resume sidecar must not survive"


async def test_status_and_delete_reject_unknown_archive_404(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    client, _ = app_client_with_zim
    assert (await client.get("/api/zims/99999/index")).status_code == 404
    assert (await client.delete("/api/zims/99999/index")).status_code == 404


async def test_archives_list_carries_index_fields(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    client, _ = app_client_with_zim
    arc = (await client.get("/api/zims")).json()["archives"][0]
    assert arc["index_depth"] == 0
    assert arc["index_status"] == "none"
    assert arc["embedding_model"] is None


# ── the archive-removal cascade (DoD) ─────────────────────────────────────────


async def test_deleting_archive_removes_its_vectors(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    client, zim_id = app_client_with_zim
    store = get_store()
    assert store is not None
    # Seed one vector for the archive. The store's db handle is the only route
    # to SQL from a test (the API exposes none); chunks FK-needs an articles row.
    db = store._db
    async with db.write() as conn:
        await conn.execute(
            "INSERT INTO articles(zim_id, entry_path, title) VALUES (?, ?, ?)",
            (zim_id, "A/Cascade_Test", "Cascade Test"),
        )
    await store.upsert(
        [
            VectorRow(
                id=900001,
                zim_id=zim_id,
                article_id=1,
                embedding=np.array([1.0] * 384, dtype=np.float32),
                char_start=0,
                char_end=10,
            )
        ]
    )
    assert (await store.stats()).per_zim.get(zim_id) == 1

    resp = await client.delete(f"/api/zims/{zim_id}")
    assert resp.json()["ok"] is True

    # DoD: stats() confirms the archive's vectors AND chunk rows are gone.
    stats = await store.stats()
    assert stats.per_zim.get(zim_id, 0) == 0
    hits = await store.search(np.array([1.0] * 384, dtype=np.float32), zim_ids=[zim_id], k=5)
    assert hits == []
