"""The ``refresh_catalog`` job — re-fetch + re-cache the Kiwix OPDS catalog.

A job (not an inline request) so a refresh is resumable, schedulable from the jobs
panel, and its progress/error lands on the job row + SSE like every other long
op. The actual fetch/parse/persist lives in ``catalog.opds``; this module is the
thin JobType wrapper that owns the runner integration and resolves the feed URL
from the ``catalog.opds_url`` setting.

On any failure (network down, parse error) the job records the error on its row
and leaves the existing cache untouched (a catalog outage must not
degrade what's already cached or break search/library).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vesta import config
from vesta.catalog import CATALOG_OPDS_URL, get_db
from vesta.catalog.opds import refresh_catalog_cache
from vesta.jobs.types import JobHandle, register_job_type


class RefreshCatalogJob:
    """Registered as job type ``refresh_catalog``. Params: optional ``url``."""

    name = "refresh_catalog"

    async def run(self, job: JobHandle, params: Mapping[str, Any]) -> None:
        db = get_db()
        if db is None:
            raise RuntimeError("refresh_catalog: db not bound (run inside the app lifespan)")
        # The job (not the parser) resolves the feed URL from settings; the parser
        # stays config-free so it's unit-testable with a mock transport.
        url = str(params.get("url") or "") or str(config.get(CATALOG_OPDS_URL))
        await job.progress(0, 1, "fetching catalog")
        try:
            count = await refresh_catalog_cache(db, url=url)
        except Exception as exc:
            # The cache is left untouched (refresh only persists after a clean
            # parse); surface the failure on the job row for the jobs panel.
            raise RuntimeError(f"catalog refresh failed: {exc!r}") from exc
        await job.progress(1, 1, f"cached {count} entries")


# Register the built-in refresh job type at import.
register_job_type(RefreshCatalogJob())


__all__ = ["RefreshCatalogJob"]
