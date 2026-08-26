"""Kiwix OPDS catalog client.

Fetches the Kiwix OPDS v2 acquisition feed, parses entries defensively into
``catalog_entries`` rows, and offers server-side FTS5
filtering so the client never receives the full ~3 000-row list.

Design considerations:

* **Parse defensively.** To handle catalog schema drift,
  unknown fields are ignored, missing fields defaulted to empty/zero. The parser
  matches child elements by *local name* (ignoring namespace prefix) so it
  survives Kiwix moving fields between the Atom default namespace and a Kiwix
  extension namespace — both shapes appear across catalog versions.
* **Never trust ``_ftindex``.** The tag has frequent false negatives; it is
  stored verbatim in ``tags`` for display only, never acted on. Fulltext
  indexing is probed at runtime after download.
* **The catalog is never a hard dependency.** A network/parse failure degrades to
  "catalog unavailable" and the library/search keep working off what's already
  installed.

The acquisition ``href`` points at a ``.meta4`` metalink; the download job
fetches it for the ranked mirror list + sha-256 + piece hashes. The
``length`` attribute on the acquisition link is the byte size (fallback when the
metalink's ``<size>`` is absent).

``catalog/`` depends on ``db`` and ``config`` only.
"""

from __future__ import annotations

import datetime as _dt
import logging
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from vesta.catalog.curated import curated_rank_for
from vesta.config import netguard

if TYPE_CHECKING:
    from vesta.db.connection import Database

_log = logging.getLogger(__name__)

#: Default Kiwix OPDS v2 acquisition feed.
DEFAULT_OPDS_URL = "https://library.kiwix.org/catalog/v2/entries?count=-1"

#: Request the full feed in one fetch.
_OPDS_PARAMS: dict[str, str] = {"count": "-1"}

#: Seconds to wait for the catalog endpoint before degrading to "unavailable".
_OPDS_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class CatalogEntry:
    """One OPDS acquisition entry, as persisted to ``catalog_entries``.

    Frozen domain object. The API layer maps this to a Pydantic model;
    everything here is plain data.
    """

    id: str  # OPDS entry id (urn:uuid); primary key
    name: str  # stable ZIM name, date-free (e.g. wikipedia_en_top_nopic)
    title: str
    description: str
    language: str  # ISO 639-3 (eng); stored verbatim
    flavour: str  # maxi | nopic | mini | ""
    tags: str  # ';'-joined, incl. _ftindex (display only — unverified by feed)
    size_bytes: int  # from the acquisition link length / metalink <size>
    article_count: int  # catalog figure — redirect-inflated, displayed as approx
    url: str  # the .meta4 acquisition URL (download job resolves mirrors)
    illustration_url: str
    zim_date: str  # dc:issued
    curated_rank: int | None


# ── parsing ─────────────────────────────────────────────────────────────────


def _local_name(tag: str) -> str:
    """Strip the ``{namespace}`` prefix or ``prefix:`` from an ElementTree tag.

    The Kiwix feed mixes the Atom default namespace with Kiwix extension
    elements; matching by local name survives either shape.
    """
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag.rsplit(":", 1)[-1]


def _find(elem: ET.Element, local: str) -> ET.Element | None:
    """First direct child of ``elem`` whose local name matches ``local``."""
    for child in elem:
        if _local_name(child.tag) == local:
            return child
    return None


def _findall_local(elem: ET.Element, local: str) -> list[ET.Element]:
    return [c for c in elem if _local_name(c.tag) == local]


def _text(elem: ET.Element | None, default: str = "") -> str:
    return (elem.text or "").strip() if elem is not None else default


def _int(elem: ET.Element | None, default: int = 0) -> int:
    if elem is None:
        return default
    raw = (elem.text or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _absolutize(href: str, base: str) -> str:
    """Resolve a possibly-relative catalog href against the feed URL."""
    if not href:
        return ""
    if href.startswith(("http://", "https://")):
        return href
    if href.startswith("//"):
        return "https:" + href
    # Root- or path-relative: resolve against the feed origin.
    from urllib.parse import urljoin, urlsplit

    parts = urlsplit(base)
    origin = f"{parts.scheme}://{parts.netloc}"
    if href.startswith("/"):
        return origin + href
    return urljoin(base, href)


def _parse_entry(entry: ET.Element, base_url: str) -> CatalogEntry | None:
    """Parse one ``<entry>`` defensively; ``None`` if it lacks an id/name/url.

    A malformed entry is skipped, never raises. Missing fields
    default to empty/zero. The acquisition URL prefers the ``open-access`` link
    (the ``.meta4``), falling back to any ``application/x-zim`` link.
    """
    eid = _text(_find(entry, "id"))
    name = _text(_find(entry, "name"))
    title = _text(_find(entry, "title")) or name
    if not eid or not name:
        return None

    summary = _text(_find(entry, "summary")) or _text(_find(entry, "subtitle"))
    language = _text(_find(entry, "language"))
    flavour = _text(_find(entry, "flavour"))
    tags = _text(_find(entry, "tags"))
    article_count = _int(_find(entry, "articleCount"))
    issued = _text(_find(entry, "issued")) or _text(_find(entry, "updated"))

    acquisition_url = ""
    size_bytes = 0
    illustration_url = ""
    for link in _findall_local(entry, "link"):
        rel = link.get("rel", "")
        link_type = link.get("type", "")
        href = _absolutize(link.get("href", ""), base_url)
        if "open-access" in rel or "application/x-zim" in link_type:
            # Prefer the open-access (metalink) link; remember the first zim link.
            if not acquisition_url or "open-access" in rel:
                acquisition_url = href
                length = link.get("length")
                if length and length.isdigit():
                    size_bytes = int(length)
        elif ("image/thumbnail" in rel or "image" in rel) and not illustration_url:
            illustration_url = href

    if not acquisition_url:
        return None  # nothing to download — not a usable catalog entry

    return CatalogEntry(
        id=eid,
        name=name,
        title=title,
        description=summary,
        language=language,
        flavour=flavour,
        tags=tags,
        size_bytes=size_bytes,
        article_count=article_count,
        url=acquisition_url,
        illustration_url=illustration_url,
        zim_date=issued,
        curated_rank=curated_rank_for(name, flavour),
    )


def parse_opds_feed(xml_text: str, base_url: str = DEFAULT_OPDS_URL) -> list[CatalogEntry]:
    """Parse an OPDS v2 acquisition feed into ``CatalogEntry`` records.

    Defensive: a feed with no ``<entry>`` children yields ``[]``;
    unparseable XML raises ``OPDSParseError`` so the caller can degrade cleanly.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise OPDSParseError(f"OPDS feed is not well-formed XML: {exc}") from exc
    # Entries may be direct children (acquisition feed) or nested one level.
    entries: list[ET.Element] = []
    if _local_name(root.tag) == "feed":
        entries = _findall_local(root, "entry")
    elif _local_name(root.tag) == "entry":
        entries = [root]
    out: list[CatalogEntry] = []
    for entry in entries:
        parsed = _parse_entry(entry, base_url)
        if parsed is not None:
            out.append(parsed)
    return out


class OPDSParseError(RuntimeError):
    """Raised when the OPDS feed cannot be parsed (caller degrades to unavailable)."""


# ── fetching ────────────────────────────────────────────────────────────────


async def fetch_opds_feed(
    url: str, *, client: httpx.AsyncClient | None = None, egress_guard: bool = False
) -> str:
    """Fetch the OPDS feed body at ``url`` as text. Raises on network/HTTP error.

    ``url`` is explicit (the job resolves it from the ``catalog.opds_url``
    setting and passes it here); this keeps the parser free of a config dep so it
    is unit-testable with a mock transport. The caller (``refresh_catalog_cache``)
    turns any error into "catalog unavailable" rather than letting it propagate to
    the request path. ``client`` is injectable so tests fetch
    against a recorded fixture.

    ``egress_guard=True`` (set by the job ONLY when the URL was supplied by the
    request rather than resolved from settings — audit AUDIT_0824 A1) routes the
    fetch through :mod:`vesta.config.netguard`, which validates scheme + host on
    every hop including redirects. Settings-resolved URLs stay unguarded: the
    owner's LAN catalog is legitimate.
    """
    own_client = client is None
    if client is None:
        client = (
            netguard.safe_client(timeout=_OPDS_TIMEOUT_S)
            if egress_guard
            else httpx.AsyncClient(
                timeout=_OPDS_TIMEOUT_S,
                follow_redirects=True,  # download.kiwix.org 301s
            )
        )
    try:
        params = None if "?" in url else _OPDS_PARAMS
        if egress_guard:
            resp = await netguard.guarded_request(client, "GET", url, params=params)
        else:
            resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.text
    finally:
        if own_client:
            await client.aclose()


# ── persistence ─────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()


async def _persist(db: Database, entries: Sequence[CatalogEntry]) -> int:
    """Replace the catalog cache + FTS index with ``entries`` (full
    replace). Returns the count written."""
    now = _now_iso()
    async with db.write() as conn:
        await conn.execute("DELETE FROM catalog_entries")
        await conn.execute("DELETE FROM catalog_fts")
        rows = [
            (
                e.id,
                e.name,
                e.title,
                e.description,
                e.language,
                e.flavour,
                e.tags,
                e.size_bytes,
                e.article_count,
                e.url,
                e.illustration_url,
                e.zim_date,
                e.curated_rank,
                now,
            )
            for e in entries
        ]
        await conn.executemany(
            "INSERT INTO catalog_entries(id, name, title, description, language, "
            "flavour, tags, size_bytes, article_count, url, illustration_url, "
            "zim_date, curated_rank, fetched_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        # Repopulate FTS in the same transaction so search never sees a half-state.
        await conn.executemany(
            "INSERT INTO catalog_fts(name, title, description, tags, entry_id) VALUES(?,?,?,?,?)",
            [(e.name, e.title, e.description, e.tags, e.id) for e in entries],
        )
    return len(entries)


async def refresh_catalog_cache(
    db: Database,
    *,
    url: str | None = None,
    client: httpx.AsyncClient | None = None,
    egress_guard: bool = False,
) -> int:
    """Fetch + parse + persist the catalog. Returns the entry count written.

    All failure modes degrade to a raised exception caught by the caller (never a
    half-written cache): a network failure leaves the existing cache untouched
    because ``_persist`` only runs after a successful parse — a catalog outage
    must not degrade what's already cached.

    ``egress_guard`` passes through to :func:`fetch_opds_feed`: on ONLY when
    ``url`` came from a request parameter rather than settings.
    """
    feed_url = url or DEFAULT_OPDS_URL
    xml = await fetch_opds_feed(feed_url, client=client, egress_guard=egress_guard)
    entries = parse_opds_feed(xml, base_url=feed_url)
    return await _persist(db, entries)


# ── search (server-side FTS5 filtering) ──────────────────────────────


@dataclass(frozen=True)
class CatalogFilter:
    """Server-side catalog query parameters (the API never ships all ~3 k rows)."""

    query: str = ""
    language: str = ""  # ISO 639-3 code, e.g. "eng"; matched as a comma-member
    recommended_only: bool = False  # curated_rank IS NOT NULL
    max_size_bytes: int | None = None  # exclude archives larger than this (bytes)
    sort: str = "default"  # default|relevance|size_asc|size_desc|date_asc|date_desc
    limit: int = 100
    offset: int = 0


@dataclass(frozen=True)
class CatalogSearchResult:
    entries: list[dict[str, Any]]
    total: int


def _fts_query(query: str) -> str:
    """Turn a free-text query into an FTS5 MATCH expression.

    FTS5's bareword tokenizer ANDs terms; we wrap each token in double quotes so
    a multi-word query matches all words in any order across the indexed columns,
    and strip characters that are FTS5 syntax (column filters, AND/OR/NOT, *).
    """
    cleaned = []
    for raw_token in query.replace('"', " ").split():
        token = raw_token.strip()
        if not token or token.upper() in {"AND", "OR", "NOT", "NEAR"}:
            continue
        cleaned.append(f'"{token}"')
    return " ".join(cleaned)


#: Explicit ORDER BY clauses for each non-default ``CatalogFilter.sort`` value.
#: ``zim_date`` is ISO text (dc:issued, "YYYY-MM" or "YYYY-MM-DD"), so it sorts
#: lexically. The leading boolean term sinks *dateless* entries — both NULL and
#: the empty string the parser writes when a feed entry carries no date — to the
#: bottom in both directions: the expression is 0 for a real date, 1 otherwise,
#: and ASC ordering puts the 0s first.
_SORT_ORDER: dict[str, str] = {
    "size_asc": "ce.size_bytes ASC",
    "size_desc": "ce.size_bytes DESC",
    "date_asc": "(ce.zim_date IS NULL OR ce.zim_date = '') ASC, ce.zim_date ASC",
    "date_desc": "(ce.zim_date IS NULL OR ce.zim_date = '') ASC, ce.zim_date DESC",
}


async def search_catalog(db: Database, flt: CatalogFilter) -> CatalogSearchResult:
    """FTS5-filtered, language/size/recommended-narrowed catalog listing.

    Ordering: an explicit ``size_*``/``date_*`` sort overrides the curated bias
    entirely. ``default``/``relevance`` resolve to FTS ``rank`` when a query
    MATCH is present, else curated-rank-then-size (the browse order). The join
    back to ``catalog_entries`` happens on the FTS ``entry_id`` column. With no
    catalog cached yet (offline first boot), this returns an empty result — the
    library page still lists installed archives (served from ``zims`` separately).
    """
    conditions: list[str] = []
    params: list[Any] = []
    fts_join = ""
    has_match = False
    if flt.query.strip():
        match = _fts_query(flt.query)
        if match:
            # FTS5's MATCH must reference the table by its real name (an alias is
            # rejected with "no such column"); the JOIN uses the real name too.
            fts_join = "JOIN catalog_fts ON catalog_fts.entry_id = ce.id"
            conditions.append("catalog_fts MATCH ?")
            params.append(match)
            has_match = True
    if flt.language:
        # Comma-member match: multilingual archives ship "eng,fra,deu", which an
        # exact ``=`` would drop. Bound the stored value in commas so the code is
        # matched as a member, not a substring ("eng" ≠ a future "eng_sm").
        conditions.append("(',' || ce.language || ',') LIKE ?")
        params.append(f"%,{flt.language},%")
    if flt.recommended_only:
        conditions.append("ce.curated_rank IS NOT NULL")
    if flt.max_size_bytes is not None:
        conditions.append("ce.size_bytes <= ?")
        params.append(flt.max_size_bytes)
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

    sort = flt.sort if flt.sort in _SORT_ORDER else "default"
    if sort in _SORT_ORDER:
        order = _SORT_ORDER[sort]
    elif has_match:
        order = "rank"  # FTS5 ``rank`` orders by relevance when a MATCH is present.
    else:
        order = "ce.curated_rank IS NULL, ce.curated_rank ASC, ce.size_bytes DESC"

    async with db.read() as conn:
        count_sql = f"SELECT COUNT(*) FROM catalog_entries ce {fts_join}{where}"
        cur = await conn.execute(count_sql, params)
        total_row = await cur.fetchone()
        total = int(total_row[0]) if total_row is not None else 0

        page_sql = (
            f"SELECT ce.* FROM catalog_entries ce {fts_join}{where} "
            f"ORDER BY {order} LIMIT ? OFFSET ?"
        )
        cur = await conn.execute(page_sql, [*params, flt.limit, flt.offset])
        rows = [dict(r) for r in await cur.fetchall()]
    return CatalogSearchResult(entries=rows, total=total)


async def get_entry(db: Database, entry_id: str) -> dict[str, Any] | None:
    """One catalog entry by OPDS id, or ``None`` if absent (manual URL fallback)."""
    async with (
        db.read() as conn,
        conn.execute("SELECT * FROM catalog_entries WHERE id=?", (entry_id,)) as cur,
    ):
        row = await cur.fetchone()
    return dict(row) if row is not None else None


async def catalog_state(db: Database) -> Mapping[str, Any]:
    """A cheap summary of the catalog cache for the library page header."""
    async with (
        db.read() as conn,
        conn.execute("SELECT COUNT(*) AS n, MAX(fetched_at) AS at FROM catalog_entries") as cur,
    ):
        row = await cur.fetchone()
    n = int(row["n"]) if row is not None and row["n"] is not None else 0
    at = str(row["at"]) if row is not None and row["at"] is not None else None
    return {"count": n, "fetched_at": at, "available": n > 0}


async def catalog_languages(db: Database) -> list[dict[str, Any]]:
    """Distinct languages in the cached catalog, each with its entry count.

    Counted by comma-member: a multilingual archive ("eng,fra") contributes one
    to each of ``eng`` and ``fra`` (the browse picker offers every language an
    archive is available in). Ordered by count desc then code, so the languages
    with the most archives surface first. Empty on an empty cache (offline first
    boot) — never raises.
    """
    counts: dict[str, int] = {}
    async with db.read() as conn:
        cur = await conn.execute("SELECT language FROM catalog_entries WHERE language != ''")
        for row in await cur.fetchall():
            for raw_code in str(row[0]).split(","):
                code = raw_code.strip()
                if code:
                    counts[code] = counts.get(code, 0) + 1
    return [
        {"code": code, "count": n}
        for code, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


__all__ = [
    "DEFAULT_OPDS_URL",
    "CatalogEntry",
    "CatalogFilter",
    "CatalogSearchResult",
    "OPDSParseError",
    "catalog_languages",
    "catalog_state",
    "fetch_opds_feed",
    "get_entry",
    "parse_opds_feed",
    "refresh_catalog_cache",
    "search_catalog",
]
