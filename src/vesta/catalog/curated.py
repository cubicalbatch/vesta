"""The curated starter ZIM list.

Shipped as data in the repo so a fully-offline first boot still shows a
"Recommended" section even during a catalog outage.
Refreshed at release time against the live catalog; a missing entry degrades to
"unavailable" rather than erroring.

``curated_rank`` drives ordering in the Recommended section — lower = first
picked.

**Rank bands.** The welcome wizard reads three bands:
  - **ranks 1-2** — the two featured cards on ``/welcome`` ("Fastest start" +
    "Most useful"). Exactly two; a third is never offered.
  - **ranks 10-16** — the multi-select checkbox list of secondary archives.
  - **ranks >= 20** — additional curated entries surfaced in the Catalog page's
    "Recommended" filter but not on the welcome screen.

**Matching.** The OPDS feed splits an archive's identity across two fields:
``<name>`` (e.g. ``wikipedia_en_top``) and ``<flavour>`` (e.g. ``nopic``). Only
the *pair* identifies a downloadable archive — ``mdwiki_en_all`` is 10.75 GB with
an empty flavour and 2.30 GB at ``maxi``, so recommending a bare name is
ambiguous by an order of magnitude. Entries therefore carry both, and matching
goes through :func:`catalog_key`, which rebuilds the familiar ZIM filename stem
(``mdwiki_en_all_maxi``) that humans and ``.zim`` filenames use.

Both halves stay date-free, so ``wikipedia_en_top``+``nopic`` maps across
``_2026-03`` → ``_2026-06`` cleanly when detecting updates.

Real names/flavours/sizes/article counts from the live catalog, 2026-08-13.
Article counts are the catalog's (redirect-inflated; displayed as approximate and
replaced with the probed value after download).
"""

from __future__ import annotations

from dataclasses import dataclass


def catalog_key(name: str, flavour: str) -> str:
    """The ZIM filename stem for an OPDS ``name``/``flavour`` pair.

    ``wikipedia_en_top`` + ``nopic`` → ``wikipedia_en_top_nopic`` — the join key
    between a :class:`CuratedEntry` and the live catalog row, and the display
    name humans see in a ``.zim`` filename.
    """
    return f"{name}_{flavour}" if flavour else name


@dataclass(frozen=True)
class CuratedEntry:
    """A recommended starter archive. Matched to the live catalog by name+flavour.

    ``description`` and ``article_count`` ship in the repo so the welcome screen
    can render the checkbox list with real text even when the live OPDS feed is
    unreachable (catalog outage ≠ degraded first-run). When the feed *is*
    available, the matched :class:`~vesta.catalog.opds.CatalogEntry` provides the
    authoritative size/article_count/download URL.
    """

    name: str
    flavour: str
    rank: int
    size_note: str
    description: str = ""
    article_count: int = 0
    warning: str | None = None

    @property
    def key(self) -> str:
        """The ZIM filename stem — the display name and the catalog join key."""
        return catalog_key(self.name, self.flavour)


# ── Featured picks (ranks 1-2) — the two cards on /welcome ──────────────────
# Exactly two. The first is the smallest useful Wikipedia slice (top 100 most-
# read articles + everything they link to); the second is the top ~50k by
# importance — the backbone of a grounded-answer product. A third card is
# deliberately not offered: choice paralysis on the first screen is the worst
# UX failure for a self-hosted tool.
_CURATED: tuple[CuratedEntry, ...] = (
    # "Fastest start" — tiny, downloads in under a minute, enough to prove the
    # concept. The subtitle says "you'll probably want more" because 5 k articles
    # is a demo, not a corpus.
    CuratedEntry(
        name="wikipedia_en_100",
        flavour="",
        rank=1,
        size_note="~0.18 GB",
        article_count=5032,
        description=(
            "The 100 most-read Wikipedia articles and everything they link to. "
            "Small enough to download in under a minute — enough to see how Vesta works."
        ),
    ),
    # "Most useful" — the top ~50 000 Wikipedia articles by importance. Full-text,
    # no images, ~2.2 GB. This is the archive that makes Vesta genuinely useful.
    CuratedEntry(
        name="wikipedia_en_top",
        flavour="nopic",
        rank=2,
        size_note="~2.24 GB",
        article_count=875265,
        description=(
            "The top ~50,000 Wikipedia articles by importance — the backbone of "
            "a useful knowledge base. Answers real questions with real citations."
        ),
    ),
    # ── Secondary picks (ranks 10-16) — the /welcome checkbox list ──────────
    # The user picks any combination. Each is a focused corpus that complements
    # Wikipedia rather than duplicating it.
    CuratedEntry(
        name="wikivoyage_en_all",
        flavour="nopic",
        rank=10,
        size_note="~0.07 GB",
        article_count=10275,
        description="Travel guides, itineraries, and phrasebooks for destinations worldwide.",
    ),
    CuratedEntry(
        name="history.stackexchange.com_en_all",
        flavour="",
        rank=11,
        size_note="~0.30 GB",
        article_count=31860,
        description="Community Q&A on history — ancient civilizations to modern events.",
    ),
    CuratedEntry(
        name="appropedia_en_all",
        flavour="maxi",
        rank=12,
        size_note="~0.56 GB",
        article_count=22320,
        description="Practical knowledge for sustainability — appropriate technology, permaculture, and DIY projects.",
    ),
    CuratedEntry(
        name="nhs.uk_en_medicines",
        flavour="",
        rank=13,
        size_note="~0.02 GB",
        article_count=1996,
        description="Patient-facing medicine guides from the UK National Health Service — how and when to take a medicine.",
    ),
    CuratedEntry(
        name="restarters_en_all",
        flavour="maxi",
        rank=14,
        size_note="~0.02 GB",
        article_count=999,
        description="Community repair know-how — fixing electronics, appliances, and everyday items.",
    ),
    CuratedEntry(
        name="gardening.stackexchange.com_en_all",
        flavour="",
        rank=16,
        size_note="~0.88 GB",
        article_count=35148,
        description="Community Q&A on gardening — plants, soil, pests, and growing techniques.",
    ),
    # ── Additional curated entries for catalog browsing (ranks >= 20) ────────
    # Surfaced in the Catalog page's "Recommended" filter, but not on /welcome.
    CuratedEntry(
        name="mdwiki_en_all",
        flavour="maxi",
        rank=20,
        size_note="~2.30 GB",
        article_count=363797,
        description="Medical Wikipedia — full-text medical articles drawn from Wikipedia's medicine topics.",
    ),
    CuratedEntry(
        name="archlinux_en_all",
        flavour="maxi",
        rank=21,
        size_note="~0.04 GB",
        description="The Arch Linux wiki — package management, system configuration, and troubleshooting.",
    ),
    CuratedEntry(
        name="based.cooking_en_all",
        flavour="",
        rank=22,
        size_note="~0.02 GB",
        description="Practical recipes and cooking techniques — ingredient-focused, no fluff.",
    ),
    CuratedEntry(
        name="unix.stackexchange.com_en_all",
        flavour="",
        rank=23,
        size_note="~1.31 GB",
        description="Community Q&A on Unix and Linux — shell scripting, system administration, and command-line tools.",
    ),
    CuratedEntry(
        name="energypedia_en_all",
        flavour="nopic",
        rank=24,
        size_note="~0.72 GB",
        description="Knowledge platform on renewable energy, energy efficiency, and access to energy in developing countries.",
    ),
    CuratedEntry(
        name="wikibooks_en_all",
        flavour="nopic",
        rank=25,
        size_note="~3.51 GB",
        description="Open-content textbooks — mathematics, science, computing, and more.",
    ),
    CuratedEntry(
        name="openstreetmap-wiki_en_all",
        flavour="nopic",
        rank=26,
        size_note="~0.48 GB",
        description="Documentation for OpenStreetMap — tagging, editing tools, and data use.",
    ),
    CuratedEntry(
        name="wiktionary_en_all",
        flavour="nopic",
        rank=27,
        size_note="~9.12 GB",
        description="English dictionary and thesaurus — definitions, etymologies, and pronunciations.",
    ),
    CuratedEntry(
        name="diy.stackexchange.com_en_all",
        flavour="",
        rank=28,
        size_note="~2.05 GB",
        description="Community Q&A on home improvement and DIY — carpentry, electrical, plumbing, and more.",
    ),
    # Document bundles, not article-per-topic wikis: the catalog reports
    # articleCount=1 for both. They index, but the extractor sees one oversized
    # entry rather than a corpus, so retrieval quality is unverified.
    CuratedEntry(
        name="zimgit-medicine_en",
        flavour="",
        rank=29,
        size_note="~0.07 GB",
        description="A bundle of medical reference documents.",
        warning="Document bundle, not a wiki — catalog reports 1 article. Retrieval quality unverified.",
    ),
    CuratedEntry(
        name="zimgit-post-disaster_en",
        flavour="",
        rank=30,
        size_note="~0.65 GB",
        description="A bundle of post-disaster recovery reference documents.",
        warning="Document bundle, not a wiki — catalog reports 1 article. Retrieval quality unverified.",
    ),
    CuratedEntry(
        name="ifixit_en_all",
        flavour="",
        rank=31,
        size_note="~3.57 GB",
        description="Step-by-step repair guides for electronics, appliances, and vehicles.",
        warning="CC BY-NC-SA — non-commercial only; excluded from any published index set.",
    ),
    # WikiEM: dropped from the /welcome secondary list; still recommended in
    # the Catalog page's "Recommended" filter.
    CuratedEntry(
        name="wikem_en_all",
        flavour="maxi",
        rank=32,
        size_note="~0.36 GB",
        article_count=101394,
        description="Quick-reference clinical emergency medicine — dosing, procedures, and protocols.",
        warning="Clinician shorthand — terse and abbreviation-heavy. Prefer MDWiki for prose answers.",
    ),
    CuratedEntry(
        name="wikipedia_en_all",
        flavour="nopic",
        rank=40,
        size_note="~52.69 GB",
        description="Complete English Wikipedia — every article, no images.",
        warning="Complete English Wikipedia — large disk footprint.",
    ),
    # Explicitly-not-recommended entry surfaced with a loud size warning so the
    # user who goes looking for it is warned before committing 100+ GB.
    CuratedEntry(
        name="gutenberg_en_all",
        flavour="",
        rank=99,
        size_note="~221 GB",
        description="Project Gutenberg — all books, audiobooks, and images.",
        warning="221 GB — bundles audiobooks. Prefer a subset (e.g. gutenberg_en_lcc-pr).",
    ),
)

_BY_KEY: dict[str, CuratedEntry] = {e.key: e for e in _CURATED}


def curated_entries() -> tuple[CuratedEntry, ...]:
    """The ordered curated starter list (lower ``rank`` = recommended first)."""
    return _CURATED


def curated_rank_for(name: str, flavour: str = "") -> int | None:
    """The curated rank for a catalog ``name``/``flavour``, or ``None``.

    Used by the OPDS client to stamp ``curated_rank`` on refreshed entries so the
    "Recommended" filter is server-side too. A missing entry degrades to
    ``None`` (not in the Recommended section) — never an error.
    """
    entry = _BY_KEY.get(catalog_key(name, flavour))
    return entry.rank if entry else None


def curated_warning_for(name: str, flavour: str = "") -> str | None:
    """A loud size/content/licence warning for a curated entry, if any."""
    entry = _BY_KEY.get(catalog_key(name, flavour))
    return entry.warning if entry else None


__all__ = [
    "CuratedEntry",
    "catalog_key",
    "curated_entries",
    "curated_rank_for",
    "curated_warning_for",
]
