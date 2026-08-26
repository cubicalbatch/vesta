"""Archive registry over the fixture, end to end.

Exercises the owned ``ArchiveRegistry`` contract: scan registers archives,
probes ``has_fulltext_index`` at runtime (never the catalog tag), counts from
``Counter['text/html']``, mines aliases into the ``aliases`` table, marks
missing files, and the ``ZIM_FULLTEXT`` capability probe reflects the result.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from fixtures.tiny_media_zim import build_tiny_media_zim
from fixtures.tiny_zim import REDIRECT_TARGET, build_tiny_zim
from vesta.config.capabilities import Capability, compute_capabilities
from vesta.db.connection import Database
from vesta.db.migrations import run_migrations
from vesta.zim import bind_registry
from vesta.zim.registry import ArchiveRegistry, LocalArchive


@pytest.fixture
async def registry(tmp_path: Path) -> AsyncIterator[ArchiveRegistry]:
    zims = tmp_path / "zims"
    zims.mkdir(parents=True, exist_ok=True)
    build_tiny_zim(zims / "tiny.zim")
    db = Database(str(tmp_path / "vesta.db"), busy_timeout_ms=1000)
    await db.start()
    async with db.write() as conn:
        await run_migrations(conn)
    reg = ArchiveRegistry(db=db, zims_dir=zims, read_pool_size=2, cluster_cache_mb=32)
    bind_registry(reg)
    try:
        await reg.start()
        yield reg
    finally:
        bind_registry(None)
        await reg.stop()
        await db.stop()


async def test_scan_registers_and_probes_index(registry: ArchiveRegistry) -> None:
    enabled = registry.enabled()
    assert len(enabled) == 1
    archive = enabled[0]
    # has_fulltext_index is PROBED (a real bool), never read from the catalog.
    assert isinstance(archive.has_fulltext_index, bool)
    # article_count comes from Counter['text/html']; the fixture has no Counter
    # metadata, so the fallback to archive.article_count applies — still a real int.
    assert archive.article_count > 0
    assert archive.uuid
    assert archive.title


async def test_aliases_persisted_to_table(registry: ArchiveRegistry, tmp_path: Path) -> None:
    archive = registry.enabled()[0]
    async with (
        registry._db.read() as conn,
        conn.execute("SELECT source, target FROM aliases WHERE zim_id=?", (archive.id,)) as cur,
    ):
        rows = await cur.fetchall()
    pairs = {(r["source"], r["target"]) for r in rows}
    # The fixture's hard redirect is mined into the alias table.
    assert ("USA", REDIRECT_TARGET) in pairs


async def test_random_on_articles_kind_skips_shells_and_assets(
    registry: ArchiveRegistry,
) -> None:
    """Browse "Random article"/discover draws a real content article: no hard
    redirects, no soft-redirect shells, no non-HTML assets (the wikipedia-100
    archive draws ~74% redirects and ~24% ``#Section`` shells)."""
    from fixtures.tiny_zim import DISAMBIGUATION_PATH, LONG_ARTICLE_PATH, NESTED_PATH

    archive = registry.enabled()[0]
    content = {LONG_ARTICLE_PATH, REDIRECT_TARGET, DISAMBIGUATION_PATH, NESTED_PATH}
    for _ in range(20):
        path = await archive.random()
        assert path in content
        # The path must resolve to a real entry — extract() must not 404.
        article = await archive.extract(path)
        assert article.path == path


def test_random_articles_only_degrades_to_last_draw() -> None:
    """A shell-only archive never raises — the draw budget runs out and the
    last draw is returned (a shell is still a valid ``extract()`` input)."""
    import types as _types

    from vesta.zim.registry import _random_entry_path_sync

    shell_html = b'<html><head><meta http-equiv="refresh" content="0; url=A/Foo#Bar"></head></html>'
    shell = _types.SimpleNamespace(
        path="A/shell",
        is_redirect=False,
        get_item=lambda: _types.SimpleNamespace(
            mimetype="text/html", size=len(shell_html), content=shell_html
        ),
    )
    draws = iter([shell] * 500)
    fake = _types.SimpleNamespace(get_random_entry=lambda: next(draws))
    assert _random_entry_path_sync(fake, articles_only=True) == "A/shell"


async def test_missing_file_marks_row_missing(registry: ArchiveRegistry, tmp_path: Path) -> None:
    archive = registry.enabled()[0]
    # Delete the ZIM on disk and rescan → the row must go 'missing', not crash.
    (tmp_path / "zims" / "tiny.zim").unlink()
    result = await registry.rescan()
    assert archive.id in result.missing
    with pytest.raises(KeyError):
        registry.get(archive.id)


async def test_capability_probe_reflects_fulltext(registry: ArchiveRegistry) -> None:
    """ZIM_FULLTEXT is on iff an enabled archive has a probed fulltext index."""
    archive = registry.enabled()[0]
    caps = compute_capabilities()
    if archive.has_fulltext_index:
        assert Capability.ZIM_FULLTEXT in caps
    else:
        assert Capability.ZIM_FULLTEXT not in caps
    # Disabling the only archive drops the capability.
    await registry.set_enabled(archive.id, False)
    assert Capability.ZIM_FULLTEXT not in compute_capabilities()


async def test_resolve_alias_targets(registry: ArchiveRegistry) -> None:
    """Table-driven resolve_alias_targets: exact path, scope filters, degenerates, and capping."""
    archive = registry.enabled()[0]

    # Seed additional aliases for capping check.
    async with registry._db.write() as conn:
        await conn.execute(
            "INSERT INTO aliases(zim_id, source, target) VALUES (?,?,?)",
            (archive.id, "usa2", "A/Second_Target"),
        )
        await conn.execute(
            "INSERT INTO aliases(zim_id, source, target) VALUES (?,?,?)",
            (archive.id, "usa3", "A/Third_Target"),
        )

    other_id = archive.id + 1000

    # Table test cases: (terms, zim_ids, max_aliases, expected)
    cases: list[tuple[list[str], set[int] | None, int, list[tuple[int, str]]]] = [
        (["usa"], None, 3, [(archive.id, REDIRECT_TARGET)]),
        (["usa"], {archive.id}, 3, [(archive.id, REDIRECT_TARGET)]),
        (["usa"], {other_id}, 3, []),
        ([], None, 3, []),
        (["usa"], None, 0, []),
    ]

    for terms, zim_ids, max_aliases, expected in cases:
        got = await registry.resolve_alias_targets(terms, zim_ids=zim_ids, max_aliases=max_aliases)
        assert got == expected, f"failed for {terms=}, {zim_ids=}, {max_aliases=}: got {got}"

    # Capping check: max_aliases caps the number of rows returned.
    capped = await registry.resolve_alias_targets(["usa", "usa2", "usa3"], max_aliases=2)
    assert len(capped) == 2

    # Disabling the archive causes resolution to degrade to [].
    await registry.set_enabled(archive.id, False)
    assert await registry.resolve_alias_targets(["usa"], max_aliases=3) == []
    await registry.set_enabled(archive.id, True)


async def test_remove_drops_archive(registry: ArchiveRegistry) -> None:
    archive = registry.enabled()[0]
    ok = await registry.remove(archive.id)
    assert ok
    with pytest.raises(KeyError):
        registry.get(archive.id)
    assert registry.enabled() == []


async def test_media_kind_registers_manifest_even_when_registered_before(tmp_path: Path) -> None:
    """Regression: the media manifest must be built for media-kind archives that
    were registered BEFORE the feature, not only brand-new ones. Simulates the
    upgrade path — row declared kind='media' (from S1) but article_media empty
    (pre-S4) — then a rescan must populate it."""
    zims = tmp_path / "zims"
    zims.mkdir(parents=True, exist_ok=True)
    build_tiny_media_zim(zims / "media.zim")
    db = Database(str(tmp_path / "vesta.db"), busy_timeout_ms=1000)
    await db.start()
    async with db.write() as conn:
        await run_migrations(conn)

    reg = ArchiveRegistry(db=db, zims_dir=zims, read_pool_size=2, cluster_cache_mb=32)
    bind_registry(reg)
    try:
        await reg.start()  # registers fresh: manifest built via _register_new
        archive = reg.enabled()[0]
        assert archive.title == "Vesta Tiny Media Test Archive"

        # Simulate the upgrade path: wipe the manifest but keep the row, then
        # rescan — _refresh_existing must rebuild it.
        async with db.write() as conn:
            await conn.execute("DELETE FROM article_media WHERE zim_id=?", (archive.id,))
        await reg.rescan()

        async with (
            db.read() as conn,
            conn.execute(
                "SELECT video_path, poster_path, duration FROM article_media WHERE zim_id=?",
                (archive.id,),
            ) as cur,
        ):
            row = await cur.fetchone()
        assert row is not None, "rescan of a media archive must rebuild the manifest"
        assert row["video_path"] == "videos/v1abcdeXYZ/video.webm"
        assert row["poster_path"] == "videos/v1abcdeXYZ/video.webp"
        assert row["duration"] == 626  # ISO-8601 "PT10M26S" parsed to seconds
    finally:
        bind_registry(None)
        await db.stop()


async def test_concurrent_rescans_register_archive_once(tmp_path: Path) -> None:
    """Regression (AUDIT_0824 N17): two overlapping rescan() calls — startup
    scan, POST /api/zims/scan, and the post-download callback can race — must
    serialize: both complete without error, the archive gets exactly one
    ``zims`` row, and exactly one scan reports it as added. Without the lock,
    both scans read the empty ``zims`` table before either INSERT lands and the
    loser dies on UNIQUE(zims.uuid), aborting its whole loop with a leaked,
    never-stored archive handle."""
    zims = tmp_path / "zims"
    zims.mkdir(parents=True, exist_ok=True)
    db = Database(str(tmp_path / "vesta.db"), busy_timeout_ms=1000)
    await db.start()
    async with db.write() as conn:
        await run_migrations(conn)
    reg = ArchiveRegistry(db=db, zims_dir=zims, read_pool_size=2, cluster_cache_mb=32)
    bind_registry(reg)
    try:
        await reg.start()  # nothing on disk yet → no rows

        # Drop one new archive and force both scans to overlap inside
        # _register_new (both have already seen no existing row): without the
        # rescan lock the second INSERT deterministically hits UNIQUE(uuid).
        build_tiny_zim(zims / "tiny.zim")
        orig_register_new = reg._register_new

        async def slow_register_new(archive: Any, probe: Any, path: Path) -> int:
            await asyncio.sleep(0.05)
            return await orig_register_new(archive, probe, path)

        reg._register_new = slow_register_new  # type: ignore[method-assign]
        try:
            results = await asyncio.gather(reg.rescan(), reg.rescan())
        finally:
            reg._register_new = orig_register_new  # type: ignore[method-assign]

        # Both scans completed without error.
        assert len(results) == 2
        assert sum(len(r.added) for r in results) == 1
        assert all(r.total == 1 for r in results)

        # Exactly one zims row for the archive.
        async with db.read() as conn, conn.execute("SELECT id FROM zims") as cur:
            rows = await cur.fetchall()
        assert len(rows) == 1
        zim_id = int(rows[0]["id"])
        assert [zim_id] == list(results[0].added) or [zim_id] == list(results[1].added)

        # No leaked handle: the registry holds exactly one open archive for it.
        assert list(reg._archives.keys()) == [zim_id]
        stored = reg.get(zim_id)
        assert isinstance(stored, LocalArchive)
        assert stored._lz is not None  # close() sets this to None; handle is live
        assert zim_id in reg._enabled
    finally:
        bind_registry(None)
        await reg.stop()
        await db.stop()
