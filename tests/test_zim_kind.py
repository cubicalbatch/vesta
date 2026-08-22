"""ZIM kind classification (0007_zim_kind.sql) — signal-driven, never scraper-name-driven.

``_classify_kind`` decides ``articles`` | ``media`` | ``spa`` from the Counter
mime histogram and the Kiwix ``Tags`` ``_videos:yes``/``_spa:yes`` convention.
It must NOT branch on the scraper name, so it generalises across youtube2zim /
ted2zim / phet / future scrapers. These tests pin that contract with no ZIM I/O.
"""

from __future__ import annotations

from vesta.zim.registry import _classify_kind, _counter_dict


def test_classify_media_from_video_in_counter() -> None:
    """A Counter with a ``video/`` entry ⇒ media (youtube2zim / ted2zim)."""
    assert _classify_kind("preppers;_videos:yes", {"text/html": 54, "video/webm": 52}) == "media"


def test_classify_media_from_audio_in_counter() -> None:
    """``audio/`` entries (e.g. podcast ZIMs) are media too."""
    assert _classify_kind("", {"audio/mpeg": 10, "text/html": 10}) == "media"


def test_classify_media_from_videos_tag_without_counter_media() -> None:
    """The ``_videos:yes`` tag detects media ZIMs whose media lives outside the
    ZIM (so the Counter has no video/audio entry)."""
    assert _classify_kind("_category:category_videos;_videos:yes", {"text/html": 5}) == "media"


def test_classify_spa_from_tag() -> None:
    """``_spa:yes`` ⇒ a client-rendered SPA ZIM (no per-entry body text)."""
    assert _classify_kind("_spa:yes", {"text/html": 1}) == "spa"


def test_classify_articles_default() -> None:
    """Wikipedia / Stack Exchange / based.cooking — no media signals."""
    assert _classify_kind("_ftindex:yes", {"text/html": 281284, "image/webp": 50000}) == "articles"
    assert _classify_kind("", {}) == "articles"


def test_classify_never_uses_scraper_name() -> None:
    """The scraper name is deliberately NOT a signal — a ZIM named 'youtube2zim'
    but carrying no media signals classifies as articles. (The real youtube2zim
    always carries video/webm, so this only guards against name-sniffing.)"""
    assert _classify_kind("", {"text/html": 10}) == "articles"


def test_counter_dict_drops_non_mime_fragments() -> None:
    """Parameterized Counter entries like ``text/html; charset=iso-8859-1=1``
    split into a ``charset=…`` fragment with no ``/``; such fragments are
    dropped so the dict holds only real mimetypes."""

    # Build the dict the same way the registry parses the raw Counter string.
    class _FakeArchive:
        def get_metadata(self, key: str) -> bytes:
            return (
                b"text/html=285992;text/html; charset=iso-8859-1=1;video/webm=52;text/javascript=3"
            )

    d = _counter_dict(_FakeArchive())  # type: ignore[arg-type]
    assert d["text/html"] == 285992
    assert d["video/webm"] == 52
    assert d["text/javascript"] == 3
    # The charset fragment is not a mime and must not appear as a key.
    assert all("/" in k for k in d)
    assert "charset=iso-8859-1" not in d


def test_counter_dict_empty_when_no_counter() -> None:
    class _FakeArchive:
        def get_metadata(self, key: str) -> bytes:
            raise KeyError

    assert _counter_dict(_FakeArchive()) == {}  # type: ignore[arg-type]
