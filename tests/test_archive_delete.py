"""Archive deletion cascade — the user's "delete the file AND every DB
reference" requirement.

Deleting an archive must remove, in one action:

* the ``.zim`` file on disk (default) — with ``keep_file=true`` an opt-out;
* every DB reference — ``articles``, ``aliases``, ``chunks``, ``index_meta``
  (FK CASCADE) and the vec0 vectors (the registry ``on_remove`` callback wired in
  ``main`` — vec0 is not FK-cascaded).

This is what makes a deleted archive leave no trace. The cascade is exercised
end-to-end through the real API against the tiny fixture: dependent rows are
seeded directly, then ``DELETE /api/zims/{id}`` must wipe them all.
"""

from __future__ import annotations

from pathlib import Path

import httpx


async def _seed_dependents(app: object, zim_id: int) -> None:
    """Insert articles/aliases/chunks/index_meta rows owned by ``zim_id`` so the
    cascade has something to delete (the tiny fixture isn't indexed)."""
    db = app.state.vesta.db  # type: ignore[attr-defined]
    async with db.write() as conn:
        await conn.execute(
            "INSERT INTO articles(id, zim_id, entry_path, title, char_len, n_sections, flags) "
            "VALUES(?,?,?,?,?,?,?)",
            (9001, zim_id, "A/Test_Article", "Test Article", 100, 1, 0),
        )
        await conn.execute(
            "INSERT INTO aliases(zim_id, source, target) VALUES(?,?,?)",
            (zim_id, "TST", "A/Test_Article"),
        )
        await conn.execute(
            "INSERT INTO chunks(id, zim_id, article_id, ordinal, char_start, char_end, depth) "
            "VALUES(?,?,?,?,?,?,?)",
            (9001, zim_id, 9001, 0, 0, 100, 1),
        )
        await conn.execute(
            "INSERT INTO index_meta(zim_id, embedder_id, dim, query_prefix, passage_prefix, "
            "pooling, normalize) VALUES(?,?,?,?,?,?,?)",
            (zim_id, "test-embedder", 384, "q:", "p:", "cls", 1),
        )


async def _archive_path(app: object, zim_id: int) -> str:
    db = app.state.vesta.db  # type: ignore[attr-defined]
    async with db.read() as conn:
        cur = await conn.execute("SELECT path FROM zims WHERE id=?", (zim_id,))
        row = await cur.fetchone()
    return str(row["path"]) if row is not None else ""


async def test_delete_default_removes_file_and_every_db_reference(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    """Default ``DELETE`` removes the file AND every DB reference."""
    client, zim_id = app_client_with_zim
    app = client._transport.app  # type: ignore[attr-defined]
    await _seed_dependents(app, zim_id)
    file_path = await _archive_path(app, zim_id)
    assert Path(file_path).exists()  # sanity: file present before delete

    resp = await client.delete(f"/api/zims/{zim_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["file_removed"] is True

    # The file is gone from disk.
    assert not Path(file_path).exists()

    db = app.state.vesta.db  # type: ignore[attr-defined]
    async with db.read() as conn:
        # articles/aliases/chunks/index_meta key on zim_id; zims keys on id.
        for table in ("articles", "aliases", "chunks", "index_meta"):
            cur = await conn.execute(f"SELECT COUNT(*) FROM {table} WHERE zim_id=?", (zim_id,))
            assert (await cur.fetchone())[0] == 0, f"{table} still has rows"
        cur = await conn.execute("SELECT COUNT(*) FROM zims WHERE id=?", (zim_id,))
        assert (await cur.fetchone())[0] == 0, "zims row still present"


async def test_delete_keep_file_preserves_bytes_but_wipes_db(
    app_client_with_zim: tuple[httpx.AsyncClient, int],
) -> None:
    """``keep_file=true`` spares the bytes on disk but still wipes every DB
    reference (the user's 'an option to keep the file')."""
    client, zim_id = app_client_with_zim
    app = client._transport.app  # type: ignore[attr-defined]
    await _seed_dependents(app, zim_id)
    file_path = await _archive_path(app, zim_id)

    resp = await client.delete(f"/api/zims/{zim_id}?keep_file=true")
    assert resp.status_code == 200
    body = resp.json()
    assert body["file_removed"] is False

    # The file survives...
    assert Path(file_path).exists()
    # ...but every DB reference is gone.
    db = app.state.vesta.db  # type: ignore[attr-defined]
    async with db.read() as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM articles WHERE zim_id=?", (zim_id,))
        assert (await cur.fetchone())[0] == 0
        cur = await conn.execute("SELECT COUNT(*) FROM zims WHERE id=?", (zim_id,))
        assert (await cur.fetchone())[0] == 0


async def test_delete_unknown_archive_404(app_client: httpx.AsyncClient) -> None:
    resp = await app_client.delete("/api/zims/99999")
    assert resp.status_code == 404
