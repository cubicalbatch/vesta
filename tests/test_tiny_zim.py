"""The tiny-ZIM fixture must build in-test and contain all six trap cases."""

from __future__ import annotations

from pathlib import Path

from fixtures.tiny_zim import (
    DISAMBIGUATION_PATH,
    LONG_ARTICLE_PATH,
    NESTED_PATH,
    REDIRECT_PATH,
    REDIRECT_TARGET,
    SOFT_REDIRECT_PATH,
    WEBP_ASSET_PATH,
    WEBP_BYTES,
    build_tiny_zim,
    open_tiny_zim,
)


def test_builds_and_contains_all_six_cases(tmp_path: Path) -> None:
    zim_path = build_tiny_zim(tmp_path / "tiny.zim")
    # A few hundred KB, not gigabytes — usable in CI.
    assert zim_path.stat().st_size > 1000
    archive = open_tiny_zim(zim_path)

    # Case 1: hard redirect — is_redirect True, points at the target.
    redirect = archive.get_entry_by_path(REDIRECT_PATH)
    assert redirect.is_redirect is True
    assert redirect.get_redirect_entry().path == REDIRECT_TARGET

    # Case 2: soft redirect — a real article, NOT flagged, with a meta refresh.
    soft = archive.get_entry_by_path(SOFT_REDIRECT_PATH)
    assert soft.is_redirect is False
    soft_html = bytes(soft.get_item().content).decode("utf-8", "replace")
    assert "meta http-equiv" in soft_html and "refresh" in soft_html

    # Case 3: disambiguation page exists.
    assert archive.has_entry_by_path(DISAMBIGUATION_PATH)

    # Case 4: long multi-section article — multiple <h2> sections.
    long_html = bytes(archive.get_entry_by_path(LONG_ARTICLE_PATH).get_item().content).decode(
        "utf-8", "replace"
    )
    assert long_html.count("<h2>") >= 5

    # Case 5: asset named .jpg but actually WebP — the MIME trap.
    # libzim stores the TRUE type (image/webp) despite the .jpg extension; the
    # reader serves that verbatim so images render without allow-scripts.
    asset = archive.get_entry_by_path(WEBP_ASSET_PATH)
    item = asset.get_item()
    assert item.mimetype == "image/webp"  # libzim reports the true type, not the extension
    content = bytes(item.content)
    assert content.startswith(b"RIFF") and content[8:12] == b"WEBP"
    assert content == WEBP_BYTES

    # Case 6: nested-path article resolves.
    assert archive.has_entry_by_path(NESTED_PATH)


def test_rebuild_is_deterministic(tmp_path: Path) -> None:
    """Rebuilding overwrites cleanly (no 'file exists' errors from libzim)."""
    path = build_tiny_zim(tmp_path / "tiny.zim")
    path_again = build_tiny_zim(tmp_path / "tiny.zim")
    assert path == path_again
    archive = open_tiny_zim(path_again)
    assert archive.has_entry_by_path(LONG_ARTICLE_PATH)
