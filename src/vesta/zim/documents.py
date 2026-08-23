"""Document manifest — catalog PDFs for nautiluszim document-library ZIMs.

For ``kind='documents'`` archives (openZIM ``nautiluszim``) the browsable
``text/html`` entry is a single-page-app viewer shell that renders the real
content — binary documents (PDFs) under ``files/`` — client-side from a
``database.js`` manifest:

.. code-block:: javascript

    var DATABASE = [
      {'_id':'00000','ti':'Distillation For Home Water Treatment',
       'dsc':'For people with a water quality problem',
       'aut':'Michigan State University','fp':['Water (1).pdf']},
      ...
    ];

So the indexable/browsable content is the PDFs referenced by ``database.js``
(``fp``, relative to a ``files_prefix`` defaulting to ``files/``), not the
viewer scaffolding. This module reads ``database.js`` and persists a
``(zim_id, doc_path) -> (title, description, author, doc_mime)`` catalog into
``article_documents`` (migration 0013), so the frontend can render a document
library and the indexer can title PDFs from the manifest.

Design rules (mirroring ``zim/media.py``):

* **Field-name-driven, not scraper-specific.** ``database.js`` is treated as a
  document manifest iff it matches the ``var DATABASE = [...]`` shape and yields
  records carrying ``ti``/``fp`` fields. ``_classify_kind`` keys off this
  content signal — never the scraper name.
* **Tolerant JS parser.** ``database.js`` is JavaScript object literals, not
  JSON: single OR double quotes, an optional trailing ``;``. Records are
  extracted by brace scanning and fields by name, robust to both quote styles
  and a ``./`` prefix on ``fp``.
* **Best-effort.** A malformed/absent manifest yields no records (detection
  falls through to ``"articles"``); a corrupt ``database.js`` entry never aborts
  the sweep. Only records whose resolved ``doc_path`` actually exists as a ZIM
  entry are kept, so the catalog never points at a non-entry.
* **Independent of the semantic index.** Populated at registration, never
  touched by a re-index; FK-cascades with the zims row.

``zim/`` depends on ``db`` and ``config`` only.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vesta.zim.types import DocumentRef, EntryPath, strip_entry_prefix

if TYPE_CHECKING:
    from libzim.reader import Archive as LibzimArchive

    from vesta.db.connection import Database

_log = logging.getLogger(__name__)

#: Default location of the binary documents inside a nautiluszim archive. The
#: ``fp`` entries in ``database.js`` are bare filenames (``Water (1).pdf``); they
#: resolve to ``files/<name>``. nautiluszim hardcodes this prefix.
_FILES_PREFIX = "files/"

#: A nautilus ``database.js`` declares its document array with this marker.
_DATABASE_MARKER = "DATABASE"


@dataclass(frozen=True)
class DocumentRecord:
    """One resolved document descriptor before it reaches ``article_documents``."""

    doc_path: EntryPath  # the ZIM entry path of the PDF, e.g. "files/Water (1).pdf"
    title: str | None  # manifest 'ti'
    description: str | None  # manifest 'dsc'
    author: str | None  # manifest 'aut'
    doc_mime: str  # the entry's true libzim mimetype, e.g. "application/pdf"


# --- parsing -----------------------------------------------------------------

#: One manifest record is a flat ``{ ... }`` object literal (no nested objects —
#: ``fp`` is a bracketed string list, which carries no braces). Non-greedy over
#: the record body so multiple records on one line (the real shape) all match.
_RECORD_RE = re.compile(r"\{[^{}]*\}")


def _quoted_field(record: str, key: str) -> str | None:
    """Read a quoted-string field ``'key': 'value'`` (single OR double quotes).

    Field-name-driven (not positional): the key may be quoted in either style
    and the value is the contents of the first quoted span after the ``:``.
    Returns ``None`` when the field is absent or unquoted (a non-string value).
    """
    m = re.search(rf"""['"]?{re.escape(key)}['"]?\s*:\s*['"]([^'"]*)['"]""", record)
    return m.group(1) if m else None


#: The ``fp`` (file paths) field is a bracketed list of quoted strings.
_FP_RE = re.compile(r"""['"]?fp['"]?\s*:\s*\[([^\]]*)\]""")
#: Individual quoted strings inside an ``fp`` list.
_LIST_ITEM_RE = re.compile(r"""['"]([^'"]+)['"]""")


def _parse_fp(record: str) -> list[str]:
    """Parse the ``fp`` field of one record into a list of bare file paths."""
    m = _FP_RE.search(record)
    if not m:
        return []
    return _LIST_ITEM_RE.findall(m.group(1))


def _parse_database_js(text: str) -> list[dict[str, object]]:
    """Tolerant parser for a nautilus ``database.js`` manifest.

    Returns one dict per ``{ ... }`` record, keyed by field name
    (``_id``/``ti``/``dsc``/``aut``/``fp``). Missing fields are ``None`` / ``[]``.
    Returns ``[]`` when the text is not a nautilus manifest (no ``DATABASE``
    marker) so detection never fires on an unrelated ``database.js``.
    """
    if not text or _DATABASE_MARKER not in text:
        return []
    out: list[dict[str, object]] = []
    for m in _RECORD_RE.finditer(text):
        rec = m.group(0)
        out.append(
            {
                "_id": _quoted_field(rec, "_id"),
                "ti": _quoted_field(rec, "ti"),
                "dsc": _quoted_field(rec, "dsc"),
                "aut": _quoted_field(rec, "aut"),
                "fp": _parse_fp(rec),
            }
        )
    return out


def _resolve_doc_path(fp: str, files_prefix: str) -> str:
    """Resolve a manifest ``fp`` to its ZIM entry path.

    Strips a leading ``./`` or ``/`` (the shared
    :func:`~vesta.zim.types.strip_entry_prefix` — not ``lstrip``, which would
    also eat leading dots of real filenames). A path already carrying a
    directory is kept verbatim; a bare filename is joined under
    ``files_prefix`` (the nautiluszim convention).
    """
    fp = strip_entry_prefix(fp.strip())
    if "/" in fp:
        return fp
    return f"{files_prefix}{fp}"


# --- archive reads -----------------------------------------------------------


def _read_manifest_text(archive: LibzimArchive) -> str | None:
    """Read ``database.js`` defensively. Returns ``None`` if absent/unreadable."""
    try:
        if not archive.has_entry_by_path("database.js"):
            return None
        entry = archive.get_entry_by_path("database.js")
        if entry.is_redirect:
            return None
        return bytes(entry.get_item().content).decode("utf-8", "replace")
    except Exception:  # a corrupt/unreadable manifest never aborts detection
        return None


def _entry_mime(archive: LibzimArchive, doc_path: str, fallback: str) -> str:
    """The entry's true libzim mimetype (never inferred from the extension)."""
    try:
        return str(archive.get_entry_by_path(doc_path).get_item().mimetype)
    except Exception:
        return fallback


def looks_like_nautilus_manifest(archive: LibzimArchive) -> bool:
    """Content-based detection of a nautiluszim document manifest.

    True iff a ``database.js`` entry exists and parses to at least one record
    carrying both a ``ti`` (title) and a non-empty ``fp`` (file path). Defensive:
    any read/parse failure or absent marker returns ``False``, so a normal
    article ZIM (which has no ``database.js``) never classifies as documents.
    """
    text = _read_manifest_text(archive)
    if text is None:
        return False
    return any(rec.get("ti") and rec.get("fp") for rec in _parse_database_js(text))


def _mine_documents_sync(
    archive: LibzimArchive, files_prefix: str = _FILES_PREFIX
) -> list[DocumentRecord]:
    """Parse ``database.js`` into resolved document records. Blocking; the
    registry dispatches it on the read pool."""
    text = _read_manifest_text(archive)
    if text is None:
        return []
    out: list[DocumentRecord] = []
    for rec in _parse_database_js(text):
        fps = rec.get("fp")
        if not isinstance(fps, list) or not fps:
            continue  # no file path → not a document record
        first = fps[0]
        if not isinstance(first, str) or not first:
            continue
        doc_path = _resolve_doc_path(first, files_prefix)
        try:
            exists = archive.has_entry_by_path(doc_path)
        except Exception:  # a corrupt path index never aborts the sweep
            exists = False
        if not exists:
            continue  # never point the catalog at a non-entry (mirrors media.py)
        out.append(
            DocumentRecord(
                doc_path=doc_path,
                title=_opt_str(rec.get("ti")),
                description=_opt_str(rec.get("dsc")),
                author=_opt_str(rec.get("aut")),
                doc_mime=_entry_mime(archive, doc_path, "application/pdf"),
            )
        )
    return out


def _opt_str(value: object) -> str | None:
    """Coerce a parsed field to ``str | None`` (empty string → None)."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


# --- persistence -------------------------------------------------------------


async def build_documents_manifest(db: Database, archive: LibzimArchive, zim_id: int) -> int:
    """Mine an archive's document catalog and persist it to ``article_documents``.

    Wipes the archive's prior rows first so a re-registration is a clean
    refresh. Returns the row count. Owned by ``zim/`` (called from the registry
    at registration, gated on ``kind='documents'``).
    """
    # The blocking libzim read runs off the event loop via to_thread — this
    # module stays executor-agnostic (the registry owns the pool, but to_thread
    # is the simpler, sufficient choice for a one-shot registration pass that
    # runs outside the hot search/read path).
    records = await asyncio.to_thread(_mine_documents_sync, archive)
    async with db.write() as conn:
        await conn.execute("DELETE FROM article_documents WHERE zim_id=?", (zim_id,))
        if records:
            await conn.executemany(
                "INSERT INTO article_documents"
                "(zim_id, doc_path, title, description, author, doc_mime) "
                "VALUES(?,?,?,?,?,?)",
                [
                    (zim_id, r.doc_path, r.title, r.description, r.author, r.doc_mime)
                    for r in records
                ],
            )
    _log.info("zim.documents_manifestBuilt zim_id=%d rows=%d", zim_id, len(records))
    return len(records)


async def fetch_documents(db: Database, zim_id: int) -> list[DocumentRecord]:
    """Batch-read an archive's document catalog (for the API browse surface)."""
    async with (
        db.read() as conn,
        conn.execute(
            "SELECT doc_path, title, description, author, doc_mime "
            "FROM article_documents WHERE zim_id=? ORDER BY doc_path",
            (zim_id,),
        ) as cur,
    ):
        rows = await cur.fetchall()
    return [
        DocumentRecord(
            doc_path=row["doc_path"],
            title=row["title"],
            description=row["description"],
            author=row["author"],
            doc_mime=row["doc_mime"],
        )
        for row in rows
    ]


def _ref_url(zim_id: int, doc_path: EntryPath) -> str:
    """The path-preserving reader URL for one document (``/api/zim/{id}/{path}``)."""
    return f"/api/zim/{zim_id}/{doc_path}"


def _to_ref(zim_id: int, record: DocumentRecord) -> DocumentRef:
    """Map a domain :class:`DocumentRecord` to the wire-facing :class:`DocumentRef`."""
    return DocumentRef(
        doc_path=record.doc_path,
        title=record.title,
        description=record.description,
        author=record.author,
        doc_mime=record.doc_mime,
        url=_ref_url(zim_id, record.doc_path),
    )


async def fetch_document_refs(db: Database, zim_id: int) -> list[DocumentRef]:
    """Batch-read an archive's document catalog as wire-facing refs (url filled).

    The API browse endpoint and search-card enrichment both consume this; it is
    the :class:`DocumentRecord` fetch plus the path-preserving reader ``url``.
    """
    records = await fetch_documents(db, zim_id)
    return [_to_ref(zim_id, r) for r in records]


async def fetch_document_refs_for_paths(
    db: Database, zim_id: int, paths: list[EntryPath]
) -> dict[EntryPath, DocumentRef]:
    """Batch-fetch ``DocumentRef`` for the given entry paths in one archive.

    Returns ``{doc_path: DocumentRef}`` only for paths that have a manifest row;
    paths without one are absent (callers treat absence as "not a document").
    Used by the api layer to enrich ArticleOut / search cards without a per-row
    round-trip.
    """
    if not paths:
        return {}
    # De-dup to avoid OR-list blowups; placeholders sized to the unique set.
    unique = list(dict.fromkeys(paths))
    placeholders = ",".join("?" for _ in unique)
    query = (
        "SELECT doc_path, title, description, author, doc_mime "
        f"FROM article_documents WHERE zim_id=? AND doc_path IN ({placeholders})"
    )
    async with db.read() as conn:
        cur = await conn.execute(query, (zim_id, *unique))
        rows = await cur.fetchall()
    return {
        str(r["doc_path"]): _to_ref(
            zim_id,
            DocumentRecord(
                doc_path=r["doc_path"],
                title=r["title"],
                description=r["description"],
                author=r["author"],
                doc_mime=r["doc_mime"],
            ),
        )
        for r in rows
    }


__all__ = [
    "DocumentRecord",
    "DocumentRef",
    "build_documents_manifest",
    "fetch_document_refs",
    "fetch_document_refs_for_paths",
    "fetch_documents",
    "looks_like_nautilus_manifest",
]
