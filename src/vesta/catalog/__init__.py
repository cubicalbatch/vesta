"""Catalog & downloads package.

Owns getting archives onto the box without touching the filesystem by hand:
browse the Kiwix OPDS catalog, download multi-GB files resumably, verify them.
The catalog is cached aggressively and is **never a hard dependency** for
anything already installed — a catalog outage must not degrade
search, the library page, or indexing.

What lives here:

* **Settings** (``catalog.*`` / ``download.*``): declared at import so
  ``GET /api/settings/schema`` surfaces them with no frontend change.
* **Runtime bindings**: the download job reaches the zims dir + a post-download
  "register the archive" callback through singletons bound by ``main`` (the
  composition root). ``catalog/`` cannot import ``zim/``, so
  the rescan that registers a freshly-downloaded archive is injected as a
  callback — mirroring how ``index/`` receives its embedder provider.

``catalog/`` imports ``{config, db, jobs}`` — config is the universal settings
dep, ``db`` backs the catalog cache + FTS, and ``jobs`` registers the
``download_zim`` / ``refresh_catalog`` job types. That is a documented 3-dep
widened cap (dated note in ``tests/test_boundaries.py``), the same shape as
``answer`` and ``index``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vesta.config.settings import setting

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

# ── Settings ─────────────────────────────────────────────────────────────

CATALOG_ENABLED = setting(
    "catalog.enabled",
    bool,
    True,
    group="Catalog",
    help="Whether the catalog browse/download surface is offered. A catalog "
    "outage degrades to 'unavailable' regardless; this is the "
    "hard off-switch for the whole feature.",
    hot=True,
)
CATALOG_OPDS_URL = setting(
    "catalog.opds_url",
    str,
    "https://library.kiwix.org/catalog/v2/entries?count=-1",
    group="Catalog",
    help="Kiwix OPDS v2 acquisition feed URL. The "
    "default requests the full feed in one fetch (~3.6 k entries, 4.36 MB).",
    hot=True,
)

DOWNLOAD_BANDWIDTH_LIMIT_KBPS = setting(
    "download.bandwidth_limit_kbps",
    int,
    0,
    group="Downloads",
    help="Optional per-download throttle in KiB/s. 0 = unlimited. Best-effort: a "
    "simple rate gate so a download doesn't saturate a slow uplink.",
    min=0,
    max=10_000_000,
    hot=True,
)
DOWNLOAD_VERIFY_CHECKSUMS = setting(
    "download.verify_checksums",
    bool,
    True,
    group="Downloads",
    help="Verify the whole-file SHA-256 (from the .meta4 metalink) before the "
    "atomic rename. A truncated/corrupt ZIM that gets registered produces "
    "confusing libzim errors much later — keep this on.",
    hot=True,
)
DOWNLOAD_MIN_FREE_SPACE_GB = setting(
    "download.min_free_space_gb",
    int,
    5,
    group="Downloads",
    help="Minimum free space that must remain on the volume after a download. "
    "Disk exhaustion mid-download on a large file is a common failure; "
    "the guard fails the job with a clear message rather than filling "
    "the volume.",
    min=0,
    max=4096,
    hot=True,
)
DOWNLOAD_MIRROR_POLICY = setting(
    "download.mirror_policy",
    str,
    "metalink",
    group="Downloads",
    help="How mirrors are chosen: 'metalink' (follow the .meta4 priority order, "
    "recommended) or 'first' (use the catalog acquisition URL "
    "directly). Mirrors lie about Content-Length / range support; the job probes "
    "and falls back down the list.",
    choices=("metalink", "first"),
    hot=True,
)

# The download job's own concurrency slot (mirrors jobs.max_concurrent.index_zim).
JOBS_MAX_CONCURRENT_DOWNLOAD_ZIM = setting(
    "jobs.max_concurrent.download_zim",
    int,
    2,
    group="Jobs",
    help="Concurrent download_zim jobs. The job runner reads this key for its per-type semaphore.",
    min=1,
    max=8,
    hot=True,
)

# ── Runtime bindings (composition root wires these) ─────────────────────────
# The download + refresh jobs are JobTypes and get only ``(JobHandle, params)``;
# they reach the db, the zims directory, and the post-download "register archive"
# callback through these singletons, bound by ``main``'s lifespan. Each is
# ``None`` outside the lifespan so a unit test can run the job with fakes
# (mirrors index.bind_runtime).
_DB: Any = None
_ZIMS_DIR: Any = None
_REGISTER_ARCHIVE: Any = None


def bind_runtime(db: Any, zims_dir: Any, register_archive: Any) -> None:
    """Bind the db, zims directory + post-download register callback.

    ``register_archive`` is an ``async def(path: Path) -> None`` that the download
    job calls after a verified rename so the freshly-downloaded archive is
    registered (probe fulltext, count articles, mine aliases).
    ``catalog/`` cannot import ``zim/``, so the composition
    root injects ``registry.rescan`` (wrapped) here.
    """
    global _DB, _ZIMS_DIR, _REGISTER_ARCHIVE
    _DB = db
    _ZIMS_DIR = zims_dir
    _REGISTER_ARCHIVE = register_archive


def get_db() -> Any:
    return _DB


def get_zims_dir() -> Path | None:
    if _ZIMS_DIR is None:
        return None
    from pathlib import Path

    return Path(_ZIMS_DIR)


def get_register_archive() -> Callable[[Path], Awaitable[None]] | None:
    from typing import cast

    return cast("Callable[[Path], Awaitable[None]] | None", _REGISTER_ARCHIVE)


__all__ = [
    "CATALOG_ENABLED",
    "CATALOG_OPDS_URL",
    "DOWNLOAD_BANDWIDTH_LIMIT_KBPS",
    "DOWNLOAD_MIN_FREE_SPACE_GB",
    "DOWNLOAD_MIRROR_POLICY",
    "DOWNLOAD_VERIFY_CHECKSUMS",
    "JOBS_MAX_CONCURRENT_DOWNLOAD_ZIM",
    "bind_runtime",
    "get_db",
    "get_register_archive",
    "get_zims_dir",
]
