"""Alias dictionary mined from the redirect table.

At registration time the archive's redirect table is walked once: Simple
Wikipedia yields 108 571 redirects → 62 911 targets, including real acronym
expansions (``AFAICS → Internet_slang``). The fixture has one hard redirect
(``USA`` → ``A/United_States``) which exercises the same path.
"""

from __future__ import annotations

from libzim.reader import Archive

from fixtures.tiny_zim import REDIRECT_PATH, REDIRECT_TARGET, build_tiny_zim
from vesta.zim.aliases import mine_aliases


def test_fixture_redirects_mined_to_alias_pairs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    arc = Archive(str(build_tiny_zim(tmp_path / "tiny.zim")))
    pairs = mine_aliases(arc)
    # The fixture's one redirect: source title "USA" → target A/United_States.
    assert ("USA", REDIRECT_TARGET) in pairs


def test_list_page_targets_skipped(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Targets like ``List_of_…`` accrue hundreds of navigation aliases — skip."""
    arc = Archive(str(build_tiny_zim(tmp_path / "tiny.zim")))
    pairs = mine_aliases(arc)
    for _source, target in pairs:
        assert not target.startswith(("List_of_", "Index_of_", "Timeline_of_", "Glossary_of_"))


def test_alias_cap_enforced() -> None:
    """Per-target cap keeps junk clusters (248 TLDs → one list) out."""
    from unittest.mock import MagicMock

    # A fake archive with 40 redirects all pointing at one target.
    entries = []
    for i in range(40):
        e = MagicMock()
        e.is_redirect = True
        e.title = f"alias{i}"
        e.get_redirect_entry().path = "A/Target"
        entries.append(e)
    arc = MagicMock()
    arc.entry_count = 40
    arc._get_entry_by_id = lambda i: entries[i]
    pairs = mine_aliases(arc, max_aliases_per_target=5)
    targets = [t for _s, t in pairs]
    assert all(t == "A/Target" for t in targets)
    assert len(targets) == 5  # capped


def test_corrupt_entries_skipped_not_fatal() -> None:
    """AUDIT_0824 N18 regression: one corrupt entry among valid ones must not
    abort mining — per-entry tolerance matching ``registry.
    _text_entry_paths_sync`` ("a corrupt entry never aborts enumeration")."""
    from unittest.mock import MagicMock, PropertyMock

    def ok_entry(i: int) -> MagicMock:
        e = MagicMock()
        e.is_redirect = True
        e.title = f"alias{i}"
        e.get_redirect_entry().path = f"A/Target{i}"
        return e

    fetch_boom = MagicMock(side_effect=RuntimeError("corrupt entry"))
    redirect_boom = ok_entry(2)
    type(redirect_boom).is_redirect = PropertyMock(side_effect=RuntimeError("corrupt flag"))
    title_boom = ok_entry(3)
    type(title_boom).title = PropertyMock(side_effect=RuntimeError("corrupt title"))

    entries = [ok_entry(0), fetch_boom, redirect_boom, title_boom, ok_entry(4), ok_entry(5)]
    arc = MagicMock()
    arc.entry_count = len(entries)
    arc._get_entry_by_id = lambda i: entries[i]

    pairs = mine_aliases(arc)
    assert pairs == [
        ("alias0", "A/Target0"),
        ("alias4", "A/Target4"),
        ("alias5", "A/Target5"),
    ]


async def test_corrupt_entry_does_not_block_registration(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """AUDIT_0824 N18 regression end to end: a corrupt redirect entry raised
    out of ``mine_aliases`` used to abort registration AFTER the ``zims``
    INSERT committed — half-registered archive, every later rescan dying the
    same way. Mining must skip it; registration must complete."""
    from vesta.db.connection import Database
    from vesta.db.migrations import run_migrations
    from vesta.zim import aliases as aliases_mod
    from vesta.zim.registry import ArchiveRegistry

    zims = tmp_path / "zims"
    zims.mkdir(parents=True, exist_ok=True)
    build_tiny_zim(zims / "tiny.zim")
    db = Database(str(tmp_path / "vesta.db"), busy_timeout_ms=1000)
    await db.start()
    async with db.write() as conn:
        await run_migrations(conn)

    class _PoisonedArchive:
        """Archive view whose one redirect entry raises at read time."""

        def __init__(self, arc: Archive) -> None:
            self._arc = arc
            self.entry_count = arc.entry_count

        def _get_entry_by_id(self, i: int) -> object:
            entry = self._arc._get_entry_by_id(i)
            if entry.path == REDIRECT_PATH:
                raise RuntimeError("simulated corrupt entry")
            return entry

    real_mine = aliases_mod.mine_aliases

    def poisoned_mine(archive: Archive, **kwargs: int) -> list[tuple[str, str]]:
        # libzim's Archive is an immutable pybind type — poison via a proxy.
        return real_mine(_PoisonedArchive(archive), **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(aliases_mod, "mine_aliases", poisoned_mine)

    reg = ArchiveRegistry(db=db, zims_dir=zims, read_pool_size=2, cluster_cache_mb=32)
    try:
        result = await reg.start()
        assert len(result.added) == 1
        zim_id = result.added[0]
        async with (
            db.read() as conn,
            conn.execute("SELECT status FROM zims WHERE id=?", (zim_id,)) as cur,
        ):
            row = await cur.fetchone()
        assert row is not None and row["status"] == "known"
    finally:
        await reg.stop()
        await db.stop()
