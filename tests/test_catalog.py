"""Catalog: OPDS parsing, FTS5 server-side filtering, persistence, refresh.

No network: parsing + FTS run against a recorded feed fixture
(``tests/fixtures/opds_sample.xml``). The refresh path uses an ``httpx`` mock
transport so the live fetch is faked too — a catalog outage must degrade, not
crash.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from vesta.catalog.curated import curated_entries, curated_rank_for
from vesta.catalog.opds import (
    CatalogFilter,
    OPDSParseError,
    catalog_languages,
    catalog_state,
    parse_opds_feed,
    refresh_catalog_cache,
    search_catalog,
)
from vesta.db.connection import Database
from vesta.db.migrations import run_migrations

FIXTURE = Path(__file__).parent / "fixtures" / "opds_sample.xml"


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    d = Database(str(tmp_path / "vesta.db"), busy_timeout_ms=1000)
    await d.start()
    async with d.write() as conn:
        await run_migrations(conn)
    yield d
    await d.stop()


# ── parsing ────────────────────────────────────────────────────────────────


def test_parse_opds_feed_comprehensive() -> None:
    """Comprehensive feed parsing: entry extraction, field defaults, curated ranking, and absolute URLs."""
    raw = FIXTURE.read_text(encoding="utf-8")
    entries = parse_opds_feed(raw)
    # 4 of 5 entries have an acquisition link; the broken one is skipped.
    assert len(entries) == 4
    by_name = {e.name: e for e in entries}
    assert "devdocs_en_lit" in by_name
    assert "wikipedia_en_top" in by_name
    assert "wikem_en_all" in by_name
    assert "wiktionary_fr_all" in by_name
    assert "broken_no_url_en" not in by_name

    e = by_name["devdocs_en_lit"]
    assert e.id == "urn:uuid:0002ed21-81ff-39eb-7274-d80240a8ea78"
    assert e.title == "Lit Docs"
    assert e.language == "eng"
    assert e.flavour == ""  # empty flavour defaulted, not None
    assert e.article_count == 57
    assert e.size_bytes == 739328  # from the acquisition link length attribute
    assert e.url.endswith("devdocs_en_lit_2026-07.zim.meta4")
    assert e.illustration_url.startswith("https://library.kiwix.org/")
    assert e.illustration_url.endswith("?size=48")  # absolutized
    assert e.zim_date.startswith("2026-07-06")
    # _ftindex is stored verbatim for DISPLAY ONLY, never acted on.
    assert "_ftindex:no" in e.tags

    # Curated rank stamped for recommended entries.
    assert by_name["wikipedia_en_top"].curated_rank == 2
    assert by_name["wikem_en_all"].curated_rank == 32
    assert by_name["devdocs_en_lit"].curated_rank is None


def test_parse_rejects_non_xml() -> None:
    with pytest.raises(OPDSParseError):
        parse_opds_feed("not xml at all <<<")


def test_parse_empty_feed_yields_empty_list() -> None:
    assert parse_opds_feed('<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"/>') == []


def test_curated_list_is_ordered_and_has_top_pick() -> None:
    cur = curated_entries()
    assert cur[0].key == "wikipedia_en_100"
    assert cur[0].rank == 1
    # Gutenberg carries a loud size warning.
    gutenberg = next(e for e in cur if e.name == "gutenberg_en_all")
    assert gutenberg.warning and "221" in gutenberg.warning
    assert curated_rank_for("not_a_real_name") is None
    assert curated_rank_for("wikipedia_en_100", "") == 1


def test_curated_keys_match_the_feeds_split_name_and_flavour() -> None:
    """Regression: the live feed's <name> excludes the flavour suffix.

    Every entry that carries a flavour must be looked up as the pair. Passing the
    joined stem as a bare name (what the code did before) matched nothing, which
    silently emptied the Recommended section for all but the flavourless picks.
    """
    assert curated_rank_for("wikipedia_en_top", "nopic") == 2
    assert curated_rank_for("mdwiki_en_all", "maxi") == 20
    # The flavour is load-bearing, not decoration: mdwiki's flavourless sibling
    # is the same 363 797 articles in 10.75 GB instead of 2.30 GB, and must not
    # inherit the recommendation.
    assert curated_rank_for("mdwiki_en_all") is None
    assert curated_rank_for("wikipedia_en_top", "maxi") is None
    # The pre-fix call shape — bare name, flavour dropped — matched nothing for
    # every flavoured entry, which is what emptied the Recommended section.
    assert curated_rank_for("wikipedia_en_top") is None
    assert curated_rank_for("wikem_en_all") is None


# ── persistence + refresh ───────────────────────────────────────────────────


def _mock_transport(body: str, status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body)

    return httpx.MockTransport(handler)


async def test_refresh_persists_entries_and_populates_fts(db: Database) -> None:
    xml = FIXTURE.read_text(encoding="utf-8")
    client = httpx.AsyncClient(transport=_mock_transport(xml))
    try:
        count = await refresh_catalog_cache(db, client=client)
    finally:
        await client.aclose()
    assert count == 4

    state = await catalog_state(db)
    assert state["count"] == 4
    assert state["available"] is True
    assert state["fetched_at"] is not None


async def test_refresh_leaves_existing_cache_untouched_on_failure(db: Database) -> None:
    # Seed a cache.
    xml = FIXTURE.read_text(encoding="utf-8")
    seed = httpx.AsyncClient(transport=_mock_transport(xml))
    try:
        await refresh_catalog_cache(db, client=seed)
    finally:
        await seed.aclose()
    # A later refresh that fails must NOT wipe the existing cache.
    failing = httpx.AsyncClient(transport=_mock_transport("oops", status=500))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await refresh_catalog_cache(db, client=failing)
    finally:
        await failing.aclose()
    state = await catalog_state(db)
    assert state["count"] == 4  # original entries survived


async def test_refresh_replaces_the_whole_cache(db: Database) -> None:
    # First refresh with the full fixture.
    xml = FIXTURE.read_text(encoding="utf-8")
    c1 = httpx.AsyncClient(transport=_mock_transport(xml))
    try:
        await refresh_catalog_cache(db, client=c1)
    finally:
        await c1.aclose()
    # Second refresh with a single-entry feed → the old 4 are gone.
    single = (
        '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">'
        "<entry><id>urn:uuid:x</id><name>only_one_en</name>"
        "<title>Only One</title>"
        '<link rel="http://opds-spec.org/acquisition/open-access" '
        'type="application/x-zim" href="https://x/only.zim.meta4" length="100"/>'
        "</entry></feed>"
    )
    c2 = httpx.AsyncClient(transport=_mock_transport(single))
    try:
        count = await refresh_catalog_cache(db, client=c2)
    finally:
        await c2.aclose()
    assert count == 1
    state = await catalog_state(db)
    assert state["count"] == 1


# ── server-side FTS5 search ─────────────────────────────────────────────────


async def _seed(db: Database) -> None:
    xml = FIXTURE.read_text(encoding="utf-8")
    client = httpx.AsyncClient(transport=_mock_transport(xml))
    try:
        await refresh_catalog_cache(db, client=client)
    finally:
        await client.aclose()


async def test_search_fts_filters_by_query(db: Database) -> None:
    await _seed(db)
    res = await search_catalog(db, CatalogFilter(query="emergency medicine"))
    names = {e["name"] for e in res.entries}
    assert names == {"wikem_en_all"}
    assert res.total == 1


async def test_search_language_filter_narrows(db: Database) -> None:
    await _seed(db)
    res = await search_catalog(db, CatalogFilter(language="fra"))
    assert {e["name"] for e in res.entries} == {"wiktionary_fr_all"}


async def test_search_recommended_only_filter(db: Database) -> None:
    await _seed(db)
    res = await search_catalog(db, CatalogFilter(recommended_only=True))
    names = {e["name"] for e in res.entries}
    assert "wikipedia_en_top" in names
    assert "devdocs_en_lit" not in names  # not curated


async def test_search_pagination(db: Database) -> None:
    await _seed(db)
    page1 = await search_catalog(db, CatalogFilter(limit=2, offset=0))
    page2 = await search_catalog(db, CatalogFilter(limit=2, offset=2))
    assert len(page1.entries) == 2
    assert len(page2.entries) == 2
    assert page1.total == 4
    # No overlap between pages.
    ids1 = {e["id"] for e in page1.entries}
    ids2 = {e["id"] for e in page2.entries}
    assert ids1.isdisjoint(ids2)


# ── size filter, sort, and language facets (catalog browse enhancements) ─────


async def _insert(
    db: Database,
    *,
    id: str,
    name: str,
    title: str = "T",
    description: str = "D",
    language: str = "eng",
    flavour: str = "",
    tags: str = "",
    size_bytes: int = 1000,
    article_count: int = 10,
    url: str = "https://x/x.meta4",
    illustration_url: str = "",
    zim_date: str = "",
    curated_rank: int | None = None,
) -> None:
    """Insert one catalog row + its FTS row directly (data-controlled tests)."""
    async with db.write() as conn:
        await conn.execute(
            "INSERT INTO catalog_entries(id, name, title, description, language, "
            "flavour, tags, size_bytes, article_count, url, illustration_url, "
            "zim_date, curated_rank, fetched_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                id,
                name,
                title,
                description,
                language,
                flavour,
                tags,
                size_bytes,
                article_count,
                url,
                illustration_url,
                zim_date,
                curated_rank,
                "2026-01-01T00:00:00+00:00",
            ),
        )
        await conn.execute(
            "INSERT INTO catalog_fts(name, title, description, tags, entry_id) VALUES(?,?,?,?,?)",
            (name, title, description, tags, id),
        )


async def test_search_max_size_filter_excludes_oversized(db: Database) -> None:
    await _insert(db, id="1", name="small", size_bytes=100)
    await _insert(db, id="2", name="big", size_bytes=10_000)
    res = await search_catalog(db, CatalogFilter(max_size_bytes=500))
    assert {e["name"] for e in res.entries} == {"small"}


async def test_search_sort_by_size(db: Database) -> None:
    await _insert(db, id="1", name="a", size_bytes=300)
    await _insert(db, id="2", name="b", size_bytes=100)
    await _insert(db, id="3", name="c", size_bytes=200)
    desc = await search_catalog(db, CatalogFilter(sort="size_desc", limit=10))
    assert [e["name"] for e in desc.entries] == ["a", "c", "b"]
    asc = await search_catalog(db, CatalogFilter(sort="size_asc", limit=10))
    assert [e["name"] for e in asc.entries] == ["b", "c", "a"]


async def test_search_sort_by_date_sinks_dateless_last(db: Database) -> None:
    """ISO dates sort lexically; entries with no date (NULL or '') sink last."""
    await _insert(db, id="1", name="jun", zim_date="2026-06-01")
    await _insert(db, id="2", name="jul", zim_date="2026-07-01")
    await _insert(db, id="3", name="undated", zim_date="")
    desc = await search_catalog(db, CatalogFilter(sort="date_desc", limit=10))
    assert [e["name"] for e in desc.entries] == ["jul", "jun", "undated"]
    asc = await search_catalog(db, CatalogFilter(sort="date_asc", limit=10))
    assert [e["name"] for e in asc.entries] == ["jun", "jul", "undated"]


async def test_search_and_sort_fallbacks(db: Database) -> None:
    """Fallbacks: empty cache, empty query, and sort modes (default, relevance without query, unknown sort)."""
    # Empty cache fallback: search and language facets return empty without errors.
    empty_res = await search_catalog(db, CatalogFilter(query="anything"))
    assert empty_res.entries == [] and empty_res.total == 0
    state = await catalog_state(db)
    assert state["available"] is False
    assert await catalog_languages(db) == []

    # Seeded search fallback: empty query lists curated entries first.
    await _seed(db)
    empty_q = await search_catalog(db, CatalogFilter(query="", limit=10))
    names = [e["name"] for e in empty_q.entries]
    assert names[0] == "wikipedia_en_top"
    assert names[1] == "wikem_en_all"
    assert empty_q.total == 4

    # Data-controlled sort fallbacks: default, relevance with no query, and unknown sort mode.
    async with db.write() as conn:
        await conn.execute("DELETE FROM catalog_entries")
        await conn.execute("DELETE FROM catalog_fts")
    await _insert(db, id="1", name="curated", size_bytes=100, curated_rank=1)
    await _insert(db, id="2", name="plain", size_bytes=999)

    for sort_mode in ("default", "relevance", "nonsense"):
        res = await search_catalog(db, CatalogFilter(sort=sort_mode, limit=10))
        assert res.entries[0]["name"] == "curated", f"failed fallback for sort={sort_mode}"

    # Explicit size sort drops curated bias.
    sized = await search_catalog(db, CatalogFilter(sort="size_desc", limit=10))
    assert sized.entries[0]["name"] == "plain"


async def test_search_language_match_is_a_comma_member(db: Database) -> None:
    """A multilingual archive ("eng,fra") must match each of its languages."""
    await _insert(db, id="1", name="mono_en", language="eng")
    await _insert(db, id="2", name="multi", language="eng,fra,deu")
    await _insert(db, id="3", name="mono_fr", language="fra")
    fra = await search_catalog(db, CatalogFilter(language="fra"))
    assert {e["name"] for e in fra.entries} == {"multi", "mono_fr"}
    eng = await search_catalog(db, CatalogFilter(language="eng"))
    assert {e["name"] for e in eng.entries} == {"mono_en", "multi"}
    # "eng" is matched as a member, not a substring: a code "eng_sm" wouldn't
    # match "eng" (defensive — Kiwix uses bare 3-letter codes today).
    await _insert(db, id="4", name="eng_sm", language="eng_sm")
    eng2 = await search_catalog(db, CatalogFilter(language="eng"))
    assert "eng_sm" not in {e["name"] for e in eng2.entries}


async def test_catalog_languages_facet_counts_comma_members(db: Database) -> None:
    await _insert(db, id="1", name="en", language="eng")
    await _insert(db, id="2", name="multi", language="eng,fra")
    await _insert(db, id="3", name="fr", language="fra")
    langs = await catalog_languages(db)
    # eng and fra each appear in two entries; ordered by count desc then code.
    assert langs == [{"code": "eng", "count": 2}, {"code": "fra", "count": 2}]
