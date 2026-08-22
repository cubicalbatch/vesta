"""Reader passthrough primitives over the fixture.

Two load-bearing rules:

* **Serve ``item.mimetype`` from libzim, never infer from the extension.** The
  fixture's WebP asset is *named* ``.jpg`` but libzim stores ``image/webp`` —
  serving that verbatim is what makes images render without ``allow-scripts``.
* **Hard and soft redirects are reported, not followed.** The HTTP layer turns
  them into 302s (a sandboxed iframe without allow-scripts blocks meta refresh).
"""

from __future__ import annotations

import pytest
from libzim.reader import Archive

from fixtures.tiny_zim import (
    NESTED_PATH,
    REDIRECT_PATH,
    REDIRECT_TARGET,
    SOFT_REDIRECT_PATH,
    WEBP_ASSET_PATH,
    WEBP_BYTES,
    build_tiny_zim,
)
from vesta.zim.reader import EntryNotFound, main_entry_path, read_entry_sync


@pytest.fixture
def archive(tmp_path) -> Archive:  # type: ignore[no-untyped-def]
    return Archive(str(build_tiny_zim(tmp_path / "tiny.zim")))


def test_webp_asset_serves_true_mimetype_not_extension(archive: Archive) -> None:
    """MIME trap: ``.jpg``-named asset serves ``image/webp``."""
    raw = read_entry_sync(archive, WEBP_ASSET_PATH)
    assert raw.mimetype == "image/webp"  # libzim's true type, NOT the .jpg extension
    assert raw.is_redirect is False
    assert raw.soft_redirect_target is None
    # Bytes really are WebP (and the content is copied, not a memoryview alias).
    assert raw.content == WEBP_BYTES
    assert raw.content.startswith(b"RIFF") and raw.content[8:12] == b"WEBP"


def test_hard_redirect_reported_not_followed(archive: Archive) -> None:
    raw = read_entry_sync(archive, REDIRECT_PATH)
    assert raw.is_redirect is True
    assert raw.redirect_target == REDIRECT_TARGET
    # No body fetched for a redirect (the HTTP layer 302s).
    assert raw.content == b""


def test_soft_redirect_target_keeps_section_fragment(archive: Archive) -> None:
    """Modern mwoffliner soft redirects target ``./Article#Section`` — the
    fragment survives target extraction (the HTTP layer keeps it a fragment)."""
    raw = read_entry_sync(archive, SOFT_REDIRECT_PATH)
    assert raw.soft_redirect_target == f"{REDIRECT_TARGET}#Government"


def test_fragment_path_stripped_for_lookup(archive: Archive) -> None:
    """A ``#fragment`` suffix is a URL section anchor, never part of a ZIM
    entry path — lookup resolves the base entry (the 2026 wikipedia-100 bug:
    ``quote('A/Foo#Bar')`` produced ``A/Foo%23Bar`` which 404ed)."""
    raw = read_entry_sync(archive, f"{REDIRECT_TARGET}#Government")
    assert raw.is_redirect is False
    assert raw.path == REDIRECT_TARGET
    assert b"United States" in raw.content


def test_nested_path_resolves(archive: Archive) -> None:
    """Deep ZIM paths (``A/-/nested/deep/article``) read correctly."""
    raw = read_entry_sync(archive, NESTED_PATH)
    assert raw.is_redirect is False
    assert b"deep" in raw.content


def test_missing_path_raises_not_found(archive: Archive) -> None:
    with pytest.raises(EntryNotFound):
        read_entry_sync(archive, "A/Does_Not_Exist")


def test_main_entry_path_resolves(archive: Archive) -> None:
    # The fixture's main page is the long article; main_entry_path mirrors it.
    assert main_entry_path(archive) is not None
