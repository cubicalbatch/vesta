"""Minimal media-kind ZIM fixture (0008): one browsable stub + one video JSON.

A tiny version of the youtube2zim shape: a ``text/html`` meta-refresh stub
(``index/<slug>``) alongside its ``application/json`` media descriptor
(``videos/<slug>.json`` carrying ``videoPath``/``thumbnailPath``/``duration``).
Used by the registry test that proves the manifest is built even for archives
that were registered before the feature (the refresh path), not just new ones.
"""

from __future__ import annotations

import json
from pathlib import Path

from libzim.writer import Creator, Hint, Item, StringProvider

# The stub (browsable) entry path — a candidate/search path.
STUB_PATH = "index/winter_shed-v1"
VIDEO_PATH = "videos/v1abcdeXYZ/video.webm"
POSTER_PATH = "videos/v1abcdeXYZ/video.webp"


class _Stub(Item):
    def get_path(self) -> str:
        return STUB_PATH

    def get_title(self) -> str:
        return "Winter Shed"

    def get_mimetype(self) -> str:
        return "text/html"

    def get_hints(self) -> dict[Hint, bool]:
        return {Hint.FRONT_ARTICLE: True}

    def get_contentprovider(self) -> StringProvider:
        return StringProvider(
            "<html><head><title>Winter Shed</title>"
            '<meta http-equiv="refresh" content="0;URL=\'../index.html#/watch/winter_shed-v1\'" />'
            "</head><body></body></html>"
        )


class _Json(Item):
    def get_path(self) -> str:
        return "videos/winter_shed-v1.json"

    def get_title(self) -> str:
        return "winter_shed-v1"

    def get_mimetype(self) -> str:
        return "application/json"

    def get_hints(self) -> dict[Hint, bool]:
        return {}

    def get_contentprovider(self) -> StringProvider:
        body = {
            "title": "Winter Shed",
            "videoPath": VIDEO_PATH,
            "thumbnailPath": POSTER_PATH,
            "duration": "PT10M26S",  # ISO-8601 — the youtube2zim shape
        }
        return StringProvider(json.dumps(body).encode("utf-8"))


class _Video(Item):
    def get_path(self) -> str:
        return VIDEO_PATH

    def get_title(self) -> str:
        return "winter_shed-v1"

    def get_mimetype(self) -> str:
        return "video/webm"

    def get_hints(self) -> dict[Hint, bool]:
        return {}

    def get_contentprovider(self) -> StringProvider:
        return StringProvider(b"\x1a\x45\xdf\xa3webmpartial")


def build_tiny_media_zim(path: str | Path) -> Path:
    """Write the fixture ZIM to ``path`` and return it."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with Creator(filename=str(out)) as creator:
        creator.add_metadata("Name", "vesta-tiny-media")
        creator.add_metadata("Title", "Vesta Tiny Media Test Archive")
        creator.add_metadata("Description", "One stub + one video JSON")
        creator.add_metadata("Language", "eng")
        creator.add_metadata("Date", "2026-08-04")
        creator.add_metadata("Creator", "vesta-tests")
        creator.add_metadata("Publisher", "vesta")
        # Do NOT add a "Counter" metadata item here: libzim computes the Counter
        # dir entry itself (add_metadata("Counter", ...) raises "existing
        # dirent's title is : Counter"), and its auto-histrogram includes the
        # video/webm item below — which is what drives kind='media' detection.
        creator.set_mainpath(STUB_PATH)
        creator.add_item(_Stub())
        creator.add_item(_Json())
        creator.add_item(_Video())
    return out
