"""Media manifest (0008) — resolve media-ZIM browsable stubs to playable assets.

``zim/media.py`` reads each ``application/json`` sidecar carrying a
``videoPath`` field and persists ``(entry_path) → (video_path, poster_path,
duration)`` into ``article_media``. Field-name-driven (not scraper-specific);
the stub-path derivation follows the youtube2zim ``videos/<slug>.json ↔
index/<slug>`` layout. These tests pin the mining + fetch with a fake archive
(no real ZIM needed).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from vesta.db.connection import Database
from vesta.db.migrations import run_migrations
from vesta.zim.media import (
    _coerce_duration,
    _mine_media_sync,
    build_media_manifest,
    fetch_media_for_paths,
)


class _Item:
    def __init__(self, path: str, mimetype: str, content: bytes) -> None:
        self.path = path
        self._mime = mimetype
        self._content = content

    def get_item(self) -> _Item:
        return self

    @property
    def mimetype(self) -> str:
        return self._mime

    @property
    def content(self) -> bytes:
        return self._content


class _FakeArchive:
    """Minimal stand-in for a libzim Archive over an in-memory item list."""

    def __init__(self, items: list[_Item]) -> None:
        self._items = items
        self._stub_paths = {it.path for it in items}

    @property
    def entry_count(self) -> int:
        return len(self._items)

    def _get_entry_by_id(self, i: int) -> _Item:
        return self._items[i]

    def has_entry_by_path(self, path: str) -> bool:
        return path in self._stub_paths


def _video_json(slug: str, *, video: str, poster: str | None, duration: object) -> _Item:
    import json

    body = {"title": slug, "videoPath": video, "duration": duration}
    if poster is not None:
        body["thumbnailPath"] = poster
    return _Item(f"videos/{slug}.json", "application/json", json.dumps(body).encode())


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(str(tmp_path / "vesta.db"), busy_timeout_ms=1000)
    await d.start()
    async with d.write() as conn:
        await run_migrations(conn)
        # A zims row for the FK on article_media.
        await conn.execute(
            "INSERT INTO zims(id, uuid, filename, path, name, title, kind, enabled, status) "
            "VALUES(1, 'u', 'f.zim', 'p', 'n', 't', 'media', 1, 'known')"
        )
    try:
        yield d
    finally:
        await d.stop()


def test_mine_media_descriptors_and_absent_stub_handling() -> None:
    """_mine_media_sync extracts videoPath descriptors, parses duration/poster, and drops orphans/non-video JSON."""
    archive = _FakeArchive(
        [
            _Item("index/alpha-a1", "text/html", b"<html></html>"),  # the stub
            _Item("index/beta-b2", "text/html", b"<html></html>"),
            _video_json(
                "alpha-a1",
                video="videos/a1/video.webm",
                poster="videos/a1/video.webp",
                duration=123.4,
            ),
            _video_json("beta-b2", video="videos/b2/video.webm", poster=None, duration=None),
            # orphan: JSON present, no matching stub entry -> skipped
            _video_json("has-not", video="videos/x/video.webm", poster=None, duration=1),
            _Item("channel.json", "application/json", b'{"id":"chan"}'),  # no videoPath → skipped
            _Item("config.json", "application/json", b'{"foo":1}'),  # no videoPath → skipped
        ]
    )
    records = _mine_media_sync(archive)  # type: ignore[arg-type]
    by_path = {r.entry_path: r for r in records}
    assert set(by_path) == {"index/alpha-a1", "index/beta-b2"}  # orphan/channel/config dropped
    assert by_path["index/alpha-a1"].video_path == "videos/a1/video.webm"
    assert by_path["index/alpha-a1"].poster_path == "videos/a1/video.webp"
    assert by_path["index/alpha-a1"].duration == 123  # float → int seconds
    assert by_path["index/beta-b2"].poster_path is None
    assert by_path["index/beta-b2"].duration is None


async def test_build_manifest_persists_and_fetch_round_trips(db: Database) -> None:
    archive = _FakeArchive(
        [
            _Item("index/x-x1", "text/html", b"<html></html>"),
            _video_json(
                "x-x1", video="videos/x1/video.webm", poster="videos/x1/video.webp", duration=90
            ),
        ]
    )
    n = await build_media_manifest(db, archive, zim_id=1)  # type: ignore[arg-type]
    assert n == 1
    fetched = await fetch_media_for_paths(db, 1, ["index/x-x1", "index/missing"])
    assert "index/missing" not in fetched  # absent → not in map
    ref = fetched["index/x-x1"]
    assert ref.video_path == "videos/x1/video.webm"
    assert ref.poster_path == "videos/x1/video.webp"
    assert ref.duration == 90


async def test_build_manifest_is_a_clean_refresh(db: Database) -> None:
    """Re-running wipes the archive's prior rows (no stale duplicates)."""
    archive = _FakeArchive(
        [
            _Item("index/x-x1", "text/html", b"<html></html>"),
            _video_json("x-x1", video="videos/x1/video.webm", poster=None, duration=1),
        ]
    )
    await build_media_manifest(db, archive, zim_id=1)  # type: ignore[arg-type]
    # Second build with the same archive → still one row, not two.
    await build_media_manifest(db, archive, zim_id=1)  # type: ignore[arg-type]
    async with (
        db.read() as conn,
        conn.execute("SELECT COUNT(*) AS n FROM article_media WHERE zim_id=1") as cur,
    ):
        row = await cur.fetchone()
    assert int(row["n"]) == 1


async def test_fetch_empty_paths_returns_empty(db: Database) -> None:
    assert await fetch_media_for_paths(db, 1, []) == {}


def test_coerce_duration_iso8601() -> None:
    """youtube2zim durations are ISO-8601 strings (PT10M26S), not numbers."""
    assert _coerce_duration("PT10M26S") == 626
    assert _coerce_duration("PT1H2M3S") == 3723
    assert _coerce_duration("PT45S") == 45
    assert _coerce_duration("P1DT2H3M4S") == 93784
    assert _coerce_duration(123.4) == 123  # numeric path still works
    assert _coerce_duration("PT0S") is None  # zero/empty → None
    assert _coerce_duration("not-a-duration") is None
    assert _coerce_duration(None) is None
    assert _coerce_duration(True) is None  # bool must not coerce to 1
