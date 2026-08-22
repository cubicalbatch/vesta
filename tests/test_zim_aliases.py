"""Alias dictionary mined from the redirect table.

At registration time the archive's redirect table is walked once: Simple
Wikipedia yields 108 571 redirects → 62 911 targets, including real acronym
expansions (``AFAICS → Internet_slang``). The fixture has one hard redirect
(``USA`` → ``A/United_States``) which exercises the same path.
"""

from __future__ import annotations

from libzim.reader import Archive

from fixtures.tiny_zim import REDIRECT_TARGET, build_tiny_zim
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
