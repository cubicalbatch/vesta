"""Entry classification over the fixture's trap cases.

The two traps the research measured:

* **Hard redirects** — ``entry.is_redirect``; cheap, libzim-flagged.
* **Soft redirects** — ``<meta http-equiv="refresh">`` pages that are *not*
  flagged ``is_redirect`` (~4 709 in Simple Wikipedia alone). Missing them puts
  thousands of ~280-byte near-duplicates in the index.

Plus disambiguation / list / stub heuristics. Classification writes
``articles.flags``; skipping is an indexing decision, so bits here only label.
"""

from __future__ import annotations

import pytest
from libzim.reader import Archive

from fixtures.tiny_zim import (
    REDIRECT_PATH,
    SOFT_REDIRECT_PATH,
    build_tiny_zim,
)
from vesta.zim.entries import (
    STUB_CHAR_THRESHOLD,
    classify_entry,
    extract_soft_redirect_target,
    is_soft_redirect,
)
from vesta.zim.types import EntryFlags


def _archive(tmp_path) -> Archive:  # type: ignore[no-untyped-def]
    return Archive(str(build_tiny_zim(tmp_path / "tiny.zim")))


def test_hard_redirect_is_flagged_redirect(tmp_path) -> None:  # type: ignore[no-untyped-def]
    arc = _archive(tmp_path)
    entry = arc.get_entry_by_path(REDIRECT_PATH)
    assert entry.is_redirect is True
    flags = classify_entry(entry.path, entry.title, None, is_redirect=True)
    assert EntryFlags.REDIRECT in flags
    # A hard redirect is not a soft one — the two stay distinct.
    assert EntryFlags.SOFT_REDIRECT not in flags


def test_soft_redirect_detected_distinctly_from_hard(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The trap: is_redirect is False, but it's still a redirect."""
    arc = _archive(tmp_path)
    entry = arc.get_entry_by_path(SOFT_REDIRECT_PATH)
    assert entry.is_redirect is False  # NOT flagged by libzim — the trap
    content = bytes(entry.get_item().content)
    assert is_soft_redirect(content)
    target = extract_soft_redirect_target(content)
    assert target is not None
    assert "United_States" in target
    flags = classify_entry(entry.path, entry.title, content, is_redirect=False)
    assert EntryFlags.SOFT_REDIRECT in flags
    # Distinct from hard redirects.
    assert EntryFlags.REDIRECT not in flags


def test_soft_redirect_detector_only_scans_small_payloads() -> None:
    """A real article body must never be mistaken for a soft redirect."""
    big = b"<html><body>" + (b"<p>real prose here. </p>" * 500)
    assert not is_soft_redirect(big)
    assert extract_soft_redirect_target(big) is None


@pytest.mark.parametrize(
    ("html", "expected_target"),
    [
        (b"<meta http-equiv='refresh' content='0; url=A/Foo'>", "A/Foo"),
        (b'<meta http-equiv="refresh" content="0;URL=A/Foo">', "A/Foo"),
        (
            b"<html><head><title>X</title>"
            b'<meta http-equiv="refresh" content="0;URL=\'../index.html#/watch/x-3QSQ?list=y-0Xxa\'" />'
            b"</head><body></body></html>",
            "index.html#/watch/x-3QSQ",
        ),
        (
            b'<meta http-equiv="refresh" content="0; url=A/United_States">',
            "A/United_States",
        ),
    ],
)
def test_soft_redirect_quoting_and_url_extraction(html: bytes, expected_target: str) -> None:
    assert is_soft_redirect(html)
    target = extract_soft_redirect_target(html)
    assert target is not None
    if expected_target.startswith("index.html"):
        assert target.startswith(expected_target)
        assert "URL=" not in target
    else:
        assert target == expected_target


def test_disambiguation_detected_by_marker_and_title() -> None:
    # The classifier keys off the MediaWiki ``mw-disambig`` class (or the title
    # suffix); the fixture's disambig page has neither, so test the heuristics
    # directly with synthetic HTML.
    marker_html = b'<html><body class="mw-disambig"><h1>Mercury</h1></body></html>'
    by_marker = classify_entry("A/Mercury", "Mercury", marker_html, is_redirect=False)
    assert EntryFlags.DISAMBIGUATION in by_marker
    by_title = classify_entry(
        "A/Foo_(disambiguation)", "Foo (disambiguation)", None, is_redirect=False
    )
    assert EntryFlags.DISAMBIGUATION in by_title


def test_list_page_detected_by_title_prefix() -> None:
    for title in ("List of countries", "Timeline of computing", "Glossary of terms"):
        flags = classify_entry("A/X", title, None, is_redirect=False)
        assert EntryFlags.LIST in flags, title


def test_stub_flagged_below_threshold() -> None:
    flags = classify_entry("A/Short", "Short", b"<html></html>", is_redirect=False, char_len=50)
    assert EntryFlags.STUB in flags
    flags = classify_entry(
        "A/Long", "Long", b"<html></html>", is_redirect=False, char_len=STUB_CHAR_THRESHOLD + 1
    )
    assert EntryFlags.STUB not in flags
