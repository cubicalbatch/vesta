"""Document manifest (0013) — catalog PDFs for nautiluszim document-library ZIMs.

``zim/documents.py`` parses a nautilus ``database.js`` manifest into
``(doc_path) → (title, description, author, doc_mime)`` rows in
``article_documents``. Detection is content-based: an archive classifies as
``"documents"`` iff a ``database.js`` entry parses to records carrying
``ti``/``fp`` fields — never the scraper name. These tests pin the parser,
detection, mining, and the build/fetch round-trip with a fake archive (no real
ZIM needed) plus a gated real-archive check against the water example.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from vesta.db.connection import Database
from vesta.db.migrations import run_migrations
from vesta.zim.documents import (
    DocumentRecord,
    _mine_documents_sync,
    _parse_database_js,
    _resolve_doc_path,
    build_documents_manifest,
    fetch_document_refs,
    fetch_document_refs_for_paths,
    fetch_documents,
    looks_like_nautilus_manifest,
)

DATA = Path("data/zims")
WATER = DATA / "zimgit-water_en_2024-08.zim"
ARTICLE = DATA / "devdocs_en_liquid_2026-07.zim"  # a small normal article ZIM

water = pytest.mark.skipif(not WATER.exists(), reason=f"{WATER} not present")
article = pytest.mark.skipif(not ARTICLE.exists(), reason=f"{ARTICLE} not present")


# A realistic ``database.js`` snippet — the water archive's manifest shape
# (single-quoted JS object literals, a ``var DATABASE = [...]`` array, trailing
# ``;``). Seven PDF records.
_DATABASE_JS = """var DATABASE = [
{'_id': '00000', 'ti': 'Distillation For Home Water Treatment', 'dsc': 'For people with a water quality problem ', 'aut': 'Michigan State University', 'fp': ['Water (1).pdf']},
{'_id': '00001', 'ti': 'Purifying Water During an Emergency', 'dsc': 'The treatments described work only to remove bacteria or viruses from water', 'aut': 'Washington State Dept of Health', 'fp': ['Water (2).pdf']},
{'_id': '00002', 'ti': 'Giardia: Drinking Water Factsheet', 'dsc': 'Giardia are parasites commonly found in untreated water', 'aut': 'US Environmental Protection Agency', 'fp': ['Water (3).pdf']},
{'_id': '00003', 'ti': 'Plants as Indicator of Ground Water', 'dsc': 'A field guide', 'aut': 'Oscar Edward MEinzer', 'fp': ['Water (4).pdf']},
{'_id': '00004', 'ti': 'Safe Water School', 'dsc': 'This manual, developed for primary schools in developing countries, is a working tool for school staff', 'aut': 'Various', 'fp': ['Water (5).pdf']},
{'_id': '00005', 'ti': 'Water', 'dsc': 'A handbook for finding and treating water', 'aut': 'Various', 'fp': ['Water (6).pdf']},
{'_id': '00006', 'ti': 'Water Treatment', 'dsc': 'A detailed and illustrated manual for water treatment', 'aut': 'Various', 'fp': ['Water (7).pdf']},
];
"""


class _Item:
    def __init__(self, path: str, mimetype: str, content: bytes) -> None:
        self.path = path
        self.is_redirect = False
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
    """Minimal stand-in for a libzim Archive over an in-memory item map.

    Documents mining keys off ``database.js`` (a single named entry) and
    resolves each ``fp`` to a real entry path, so only the path-based lookup API
    is exercised — no entry-id enumeration.
    """

    def __init__(self, items: list[_Item]) -> None:
        self._by_path = {it.path: it for it in items}

    def has_entry_by_path(self, path: str) -> bool:
        return path in self._by_path

    def get_entry_by_path(self, path: str) -> _Item:
        try:
            return self._by_path[path]
        except KeyError as exc:  # mirror libzim's KeyError on a missing path
            raise KeyError(path) from exc


def _water_fake_archive() -> _FakeArchive:
    """The water archive's shape: ``database.js`` + seven PDFs under ``files/``."""
    items: list[_Item] = [_Item("database.js", "application/javascript", _DATABASE_JS.encode())]
    for n in range(1, 8):
        items.append(_Item(f"files/Water ({n}).pdf", "application/pdf", b"%PDF-1.4..."))
    return _FakeArchive(items)


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(str(tmp_path / "vesta.db"), busy_timeout_ms=1000)
    await d.start()
    async with d.write() as conn:
        await run_migrations(conn)
        # A zims row for the FK on article_documents.
        await conn.execute(
            "INSERT INTO zims(id, uuid, filename, path, name, title, kind, enabled, status) "
            "VALUES(1, 'u', 'f.zim', 'p', 'n', 't', 'documents', 1, 'known')"
        )
    try:
        yield d
    finally:
        await d.stop()


# --- parser ------------------------------------------------------------------


def test_parse_database_js_extracts_seven_records() -> None:
    """The water manifest parses to seven records with the right fields."""
    records = _parse_database_js(_DATABASE_JS)
    assert len(records) == 7
    first = records[0]
    assert first["_id"] == "00000"
    assert first["ti"] == "Distillation For Home Water Treatment"
    assert first["aut"] == "Michigan State University"
    assert first["fp"] == ["Water (1).pdf"]
    assert records[-1]["ti"] == "Water Treatment"
    assert records[-1]["fp"] == ["Water (7).pdf"]


def test_parse_database_js_tolerates_double_quotes_and_dot_prefix() -> None:
    """The parser is field-name-driven and robust to quote style + ``./``."""
    js = """var DATABASE = [
    {"_id": "1", "ti": "Title One", "dsc": "Desc", "aut": "Author", "fp": ["./docs/a.pdf"]},
    ];"""
    records = _parse_database_js(js)
    assert len(records) == 1
    assert records[0]["ti"] == "Title One"
    assert records[0]["fp"] == ["./docs/a.pdf"]
    assert _resolve_doc_path("./docs/a.pdf", "files/") == "docs/a.pdf"


def test_parse_database_js_rejects_non_manifest() -> None:
    """A ``database.js`` that is not a nautilus manifest yields no records."""
    assert _parse_database_js("export default { foo: 1 };") == []
    assert _parse_database_js("") == []


def test_resolve_doc_path_bare_filename_uses_files_prefix() -> None:
    assert _resolve_doc_path("Water (1).pdf", "files/") == "files/Water (1).pdf"
    assert _resolve_doc_path("a/b.pdf", "files/") == "a/b.pdf"  # dir kept verbatim


# --- detection ---------------------------------------------------------------


def test_looks_like_nautilus_manifest() -> None:
    # True for nautilus manifest shape.
    assert looks_like_nautilus_manifest(_water_fake_archive()) is True
    # False when database.js is missing (normal article ZIM).
    no_db = _FakeArchive([_Item("home.html", "text/html", b"<html></html>")])
    assert looks_like_nautilus_manifest(no_db) is False
    # False on malformed database.js.
    malformed = _FakeArchive(
        [_Item("database.js", "application/javascript", b"not a manifest at all")]
    )
    assert looks_like_nautilus_manifest(malformed) is False


# --- mining + round-trip (fake archive, real DB) -----------------------------


def test_mine_documents_sync_resolves_paths_and_mime() -> None:
    records = _mine_documents_sync(_water_fake_archive())
    assert len(records) == 7
    by_path = {r.doc_path: r for r in records}
    assert set(by_path) == {f"files/Water ({n}).pdf" for n in range(1, 8)}
    first = by_path["files/Water (1).pdf"]
    assert first.title == "Distillation For Home Water Treatment"
    assert first.author == "Michigan State University"
    assert first.doc_mime == "application/pdf"


def test_mine_documents_skips_record_whose_path_is_absent() -> None:
    """A manifest entry whose ``fp`` does not resolve to a real entry is dropped
    — the catalog must never point at a non-entry (mirrors media.py)."""
    items = [
        _Item("database.js", "application/javascript", _DATABASE_JS.encode()),
        _Item("files/Water (1).pdf", "application/pdf", b"%PDF"),
    ]  # only the first PDF exists; the other six are absent
    records = _mine_documents_sync(_FakeArchive(items))
    assert {r.doc_path for r in records} == {"files/Water (1).pdf"}


async def test_build_documents_manifest_persists_and_fetch_round_trips(
    db: Database,
) -> None:
    n = await build_documents_manifest(db, _water_fake_archive(), zim_id=1)
    assert n == 7
    fetched = await fetch_documents(db, 1)
    assert len(fetched) == 7
    assert all(isinstance(r, DocumentRecord) for r in fetched)
    by_path = {r.doc_path: r for r in fetched}
    assert by_path["files/Water (1).pdf"].title == "Distillation For Home Water Treatment"
    assert by_path["files/Water (7).pdf"].title == "Water Treatment"
    assert by_path["files/Water (3).pdf"].author == "US Environmental Protection Agency"
    assert all(r.doc_mime == "application/pdf" for r in fetched)

    # Wire-facing refs carry the reader URL.
    refs = await fetch_document_refs(db, 1)
    assert len(refs) == 7
    ref_by_path = {r.doc_path: r for r in refs}
    first = ref_by_path["files/Water (1).pdf"]
    assert first.title == "Distillation For Home Water Treatment"
    assert first.description == "For people with a water quality problem"
    assert first.author == "Michigan State University"
    assert first.doc_mime == "application/pdf"
    assert first.url == "/api/zim/1/files/Water (1).pdf"
    assert all(r.url.startswith("/api/zim/1/files/") for r in refs)


async def test_build_documents_manifest_is_a_clean_refresh(db: Database) -> None:
    """Re-running wipes the archive's prior rows (no stale duplicates)."""
    archive = _water_fake_archive()
    await build_documents_manifest(db, archive, zim_id=1)
    await build_documents_manifest(db, archive, zim_id=1)
    async with (
        db.read() as conn,
        conn.execute("SELECT COUNT(*) AS n FROM article_documents WHERE zim_id=1") as cur,
    ):
        row = await cur.fetchone()
    assert int(row["n"]) == 7


async def test_fetch_document_refs_for_paths_is_path_keyed_and_drops_absent(
    db: Database,
) -> None:
    """``fetch_document_refs_for_paths`` is a path-keyed map (only rows present),
    with de-duped input and absent paths dropped — mirrors ``media.py``."""
    await build_documents_manifest(db, _water_fake_archive(), zim_id=1)
    paths = [
        "files/Water (1).pdf",
        "files/Water (1).pdf",  # duplicate input is de-duped, not a blow-up
        "files/Water (7).pdf",
        "files/nope.pdf",  # absent → dropped
    ]
    got = await fetch_document_refs_for_paths(db, 1, paths)
    assert set(got) == {"files/Water (1).pdf", "files/Water (7).pdf"}
    assert got["files/Water (1).pdf"].url == "/api/zim/1/files/Water (1).pdf"
    assert got["files/Water (7).pdf"].title == "Water Treatment"
    assert await fetch_document_refs_for_paths(db, 1, []) == {}


@water
def test_water_archive_probes_as_documents() -> None:
    """The real water archive classifies as ``"documents"`` via the content
    signal — never the scraper name."""

    from libzim.reader import Archive as LibzimArchive

    from vesta.zim.registry import _probe_archive

    assert looks_like_nautilus_manifest(LibzimArchive(str(WATER))) is True
    _archive, probe = _probe_archive(WATER)
    assert probe.kind == "documents"


@water
async def test_water_archive_builds_seven_row_manifest(db: Database) -> None:
    """``build_documents_manifest`` against the real water archive yields seven
    rows whose ``doc_path`` resolves to a real archive entry."""
    from libzim.reader import Archive as LibzimArchive

    archive = LibzimArchive(str(WATER))
    n = await build_documents_manifest(db, archive, zim_id=1)
    assert n == 7
    fetched = await fetch_documents(db, 1)
    assert len(fetched) == 7
    # Every catalogued doc_path is a real, resolvable archive entry.
    for rec in fetched:
        assert archive.has_entry_by_path(rec.doc_path), rec.doc_path
        assert rec.doc_mime == "application/pdf"
    titles = {r.title for r in fetched}
    assert "Distillation For Home Water Treatment" in titles
    assert "Water Treatment" in titles


@article
def test_normal_article_archive_is_not_documents() -> None:
    """A normal article ZIM has no ``database.js`` and probes as non-documents."""
    from libzim.reader import Archive as LibzimArchive

    from vesta.zim.registry import _probe_archive

    assert looks_like_nautilus_manifest(LibzimArchive(str(ARTICLE))) is False
    _archive, probe = _probe_archive(ARTICLE)
    assert probe.kind != "documents"
