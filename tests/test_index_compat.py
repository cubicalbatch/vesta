"""Index-settings enforcement tests (embedder mismatch → stale).

A mismatched embedder is the worst failure mode for a grounded-answer product
(plausible-looking garbage), so the comparison and the stale-reconciliation are
pinned here field-by-field, and the startup reconcile flow is exercised against
a real DB + store.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from vesta.db.connection import Database
from vesta.db.migrations import run_migrations
from vesta.index.compat import (
    EmbedderFingerprint,
    fingerprint_from_meta,
    fingerprint_from_spec,
    is_compatible,
    mark_stale,
    reconcile_stale,
)
from vesta.vectors.contracts import IndexMeta
from vesta.vectors.sqlite_vec_store import SqliteVecStore

_GRANITE = IndexMeta(
    embedder_id="onnx-community/granite-embedding-small-english-r2-ONNX",
    dim=384,
    query_prefix="",
    passage_prefix="",
    pooling="mean",
    normalize=True,
)
_GRANITE_FP = EmbedderFingerprint(
    embedder_id="onnx-community/granite-embedding-small-english-r2-ONNX",
    dim=384,
    query_prefix="",
    passage_prefix="",
    pooling="mean",
    normalize=True,
)


@pytest_asyncio.fixture
async def db(tmp_db_path: Path) -> Database:
    database = Database(str(tmp_db_path), busy_timeout_ms=2000)
    await database.start()
    async with database.write() as conn:
        await run_migrations(conn)
        await conn.execute("INSERT INTO zims(id) VALUES (1)")
        await conn.execute("INSERT INTO zims(id) VALUES (2)")
    yield database
    await database.stop()


async def _status(db: Database, zim_id: int) -> str:
    async with (
        db.read() as conn,
        conn.execute("SELECT index_status FROM zims WHERE id=?", (zim_id,)) as cur,
    ):
        row = await cur.fetchone()
    assert row is not None
    # NULL index_status reads as "none" (mirrors api/zims.py's projection).
    return str(row["index_status"] or "none")


# ── the pure comparison ──────────────────────────────────────────────────────


def test_none_recorded_meta_is_never_compatible() -> None:
    # "Not indexed" → don't search (the dense source skips silently).
    assert not is_compatible(None, _GRANITE_FP)


def test_exact_match_is_compatible() -> None:
    assert is_compatible(_GRANITE, _GRANITE_FP)


@pytest.mark.parametrize(
    "field,value",
    [
        ("embedder_id", "onnx-community/gte-modernbert-base-ONNX"),
        ("dim", 768),
        ("query_prefix", "query: "),
        ("passage_prefix", "passage: "),
        ("pooling", "cls"),
        ("normalize", False),
    ],
)
def test_any_single_field_mismatch_refuses(field: str, value: object) -> None:
    meta = IndexMeta(**{**_GRANITE.__dict__, field: value})  # type: ignore[arg-type]
    assert not is_compatible(meta, _GRANITE_FP), f"mismatch on {field} must refuse"


def test_fingerprint_from_spec_duck_typed() -> None:
    class _Spec:
        repo_id = "some/model"
        dim = 512
        query_prefix = "q:"
        passage_prefix = "p:"
        pooling = "cls"
        normalize = False

    fp = fingerprint_from_spec(_Spec())
    assert fp.embedder_id == "some/model" and fp.dim == 512
    assert fp.pooling == "cls" and not fp.normalize
    assert fingerprint_from_meta(_GRANITE) == _GRANITE_FP


# ── mark_stale / reconcile_stale against a real DB ──────────────────────────


@pytest.mark.asyncio
async def test_stale_reconciliation_lifecycle(db: Database) -> None:
    """State-machine test covering direct marking, match preservation, mismatch
    stale marking, and unstaling back to complete when the embedder is restored."""
    # 1. Direct mark_stale sets status on target archive without affecting others.
    await mark_stale(db, 1)
    assert await _status(db, 1) == "stale"
    assert await _status(db, 2) == "none"

    # Reset zim 1 status for reconcile testing.
    async with db.write() as conn:
        await conn.execute("UPDATE zims SET index_status=NULL WHERE id=1")

    # 2. Matching fingerprint leaves archive untouched (status stays 'none').
    store = SqliteVecStore(db, default_dim=4)
    await store.record_index_meta(1, _GRANITE)
    stale = await reconcile_stale(db, store, [_GRANITE_FP])
    assert stale == []
    assert await _status(db, 1) == "none"

    # 3. Mismatched fingerprint transitions to 'stale'.
    other = EmbedderFingerprint(
        "onnx-community/gte-modernbert-base-ONNX", 768, "", "", "mean", True
    )
    stale = await reconcile_stale(db, store, [other])
    assert stale == [1]
    assert await _status(db, 1) == "stale"

    # 4. Changing back: a matching fingerprint clears stale → complete (no re-index).
    stale = await reconcile_stale(db, store, [_GRANITE_FP])
    assert stale == []
    assert await _status(db, 1) == "complete"


@pytest.mark.asyncio
async def test_reconcile_requires_all_configured_fingerprints(db: Database) -> None:
    # index.embedder AND encoders.embed.model must both agree with the index;
    # matching only one still marks stale (the query side would refuse, or a
    # re-index would silently change vector spaces).
    store = SqliteVecStore(db, default_dim=4)
    await store.record_index_meta(1, _GRANITE)
    other = EmbedderFingerprint(
        "onnx-community/gte-modernbert-base-ONNX", 768, "", "", "mean", True
    )
    stale = await reconcile_stale(db, store, [_GRANITE_FP, other])
    assert stale == [1]
    assert await _status(db, 1) == "stale"
