"""Calibrated index estimator.

Shows **wall-clock time and disk cost before the user commits**. Indexing cost
is transparent rather than hidden, so two properties are load-bearing:

* **Calibrated from actual measured throughput on the running box**, not a
  projected number. The first ~1000 articles are timed; the estimate extrapolates
  from that and refines live as more samples arrive. A wrong estimate is worse
  than no estimate.
* **A range, never a single number.** Early articles skew short, leading to
  optimistic early estimates, so the estimate carries a low/expected/high band
  that narrows as data accumulates.

Pure logic: feed a :class:`ThroughputTracker` timed samples, then call
:meth:`ThroughputTracker.estimate`. No I/O; unit-testable with synthetic streams.
"""

from __future__ import annotations

from dataclasses import dataclass

#: How many articles to time before the estimate is considered "calibrated".
#: Below this the band is wide.
CALIBRATION_WINDOW = 1000

#: Depth → (low, expected, high) vectors per article, for the disk estimate.
#:
#: Measured 2026-08-19 by running ``depth.chunks_for_article`` over a 300-article
#: random sample from each of the nine installed archives (2407 articles total),
#: cross-checked against the live index (``chunks`` vs ``articles`` per zim).
#:
#: Per-archive means, by depth:
#:
#:     archive                      d1     d2     d3
#:     wikipedia_en_top            1.00  12.66  18.45
#:     wikivoyage_en_europe        1.00   9.03   5.08
#:     nhs.uk_en_medicines         0.87   6.36   2.16
#:     appropedia_en_all           1.00   5.96   3.67
#:     history.stackexchange       1.00   4.88   4.09
#:     based.cooking_en_all        1.00   3.12   1.05
#:     gardening.stackexchange     1.00   2.99   2.06
#:     restarters_en_all           1.00   2.48   1.90
#:     sample mean (2407 articles) 0.98   5.92   4.94
#:
#: Three things this replaces the old flat ``{1: 1.5, 2: 2.2, 3: 3.0}`` with:
#:
#: 1. **Depth 1 is exactly 1.0**, not 1.5 — depth 1 emits one title+lead chunk
#:    per article (``depth.py``: "1 — Article"), so chunk↔article is 1:1 by
#:    construction. The old 1.5 over-estimated every depth-1 run by 50 %, and
#:    depth 1 is the depth most users stay at.
#: 2. **Depth 2 is ~6, not 2.2** — it emits the lead chunk PLUS one chunk per
#:    H2 section (up to ``DEPTH2_MAX_SECTIONS``, long sections split at 400
#:    tokens). The old figure under-estimated disk by ~3x.
#: 3. **Depth 3 is not monotonically larger than depth 2.** Depth 3 re-splits
#:    the whole article into ~400-token passages with no section cap, so
#:    section-dense/short-section content (recipes, Q&A) yields FEWER chunks at
#:    depth 3 than at depth 2 (based.cooking 1.05 vs 3.12), while prose-heavy
#:    Wikipedia yields more (18.45 vs 12.66).
#:
#: The spread across archives is 2.5x-18.5x at depth 2, so a single expected
#: value cannot be accurate for a specific archive — hence the band. Callers get
#: low/expected/high and must not present the expected value alone as precise.
#: Documents-kind archives (nautiluszim PDF sets) can exceed the high band at
#: depth 3: ``zimgit-water_en`` measured 51.1, on a 7-article sample.
VECTORS_PER_ARTICLE_BAND: dict[int, tuple[float, float, float]] = {
    1: (0.9, 1.0, 1.0),
    2: (2.5, 6.0, 12.7),
    3: (1.0, 5.0, 18.5),
}

#: Bytes per stored vector (f32 dim 384 + chunk metadata + index overhead).
#:
#: Verified 2026-08-19 against the live index: the ``vectors_d384`` shadow tables
#: total 1606 MB and the ``chunks`` table plus its indexes 67 MB, over 1 074 717
#: vectors = **1632 B/vector**, so 1600 is accurate to 2 % for the current
#: sqlite-vec/vec0 storage.
#:
#: This figure is storage-specific and will need re-measuring if the vector
#: backend changes (a PQ-compressed store is several times smaller).
BYTES_PER_VECTOR = 1600

#: Storage-overhead uncertainty applied on top of the vectors-per-article band.
#: ``BYTES_PER_VECTOR`` is itself a measured average (page fill, index fanout and
#: chunk-metadata width all vary), so the disk band widens by these factors as
#: well as by the chunk-count band. Keeps the band strictly ordered at depth 1,
#: where the chunk count is 1.0 by construction and carries no upside spread.
BYTES_UNCERTAINTY_LOW = 0.9
BYTES_UNCERTAINTY_HIGH = 1.15


def disk_band(total_articles: int, depth: int) -> tuple[int, int, int]:
    """(low, expected, high) on-disk bytes for indexing ``total_articles``.

    Both uncertainties compose: the vectors-per-article band (which spans
    2.5x-18.5x across real archives at depth 2) and the per-vector storage
    overhead. An unknown depth falls back to the depth-1 band rather than
    guessing high.
    """
    low_vpa, exp_vpa, high_vpa = VECTORS_PER_ARTICLE_BAND.get(depth, VECTORS_PER_ARTICLE_BAND[1])
    expected = int(total_articles * exp_vpa * BYTES_PER_VECTOR)
    low = int(total_articles * low_vpa * BYTES_PER_VECTOR * BYTES_UNCERTAINTY_LOW)
    high = int(total_articles * high_vpa * BYTES_PER_VECTOR * BYTES_UNCERTAINTY_HIGH)
    return low, expected, high


#: Early-article optimism guard: short articles index faster, so the first
#: samples underestimate per-article cost. The low band scales the expected by
#: this factor until enough long articles are seen.
EARLY_PESSIMISM = 0.5

#: Pre-commit prior band for articles/sec, used ONLY before any throughput has
#: been measured on the running box (the estimate must still show an honest
#: cost before committing). Anchored to single-stream benchmarks
#: (Granite-small ≈ 14 chunks/s on a mid box) padded down for extraction +
#: upsert overhead and up for large-batch steady state; deliberately ~8x wide
#: so it can only be wrong in the honest direction. Replaced by the measured
#: rate as soon as the first batch lands, so the band stays wide and clearly
#: marked ``calibrated=False``.
PRIOR_RATE_LOW = 2.0
PRIOR_RATE_HIGH = 16.0


@dataclass(frozen=True)
class Estimate:
    """A time + disk estimate as a range (low/expected/high).

    ``calibrated`` is False until ``CALIBRATION_WINDOW`` articles are timed;
    callers should show a wider confidence band while it is False."""

    seconds_low: float
    seconds_expected: float
    seconds_high: float
    disk_bytes_low: int
    disk_bytes_expected: int
    disk_bytes_high: int
    calibrated: bool
    articles_done: int
    articles_total: int
    rate_articles_per_s: float


@dataclass
class ThroughputTracker:
    """Accumulates ``(articles, elapsed_s)`` samples and extrapolates a range.

    Refine live: call :meth:`record` after each batch; :meth:`estimate` returns
    a narrowing range. The expected value is the running per-article cost; the
    low band applies :data:`EARLY_PESSIMISM` (we may have only seen short
    articles) and the high band doubles it for tail variance, until calibrated.
    """

    total_articles: int
    depth: int = 1
    _done: int = 0
    _elapsed: float = 0.0

    def record(self, done: int, elapsed_s: float) -> None:
        """Record cumulative progress. ``done`` is articles completed so far;
        ``elapsed_s`` is wall-clock since the index started."""
        if done <= 0 or elapsed_s < 0:
            return
        self._done = done
        self._elapsed = elapsed_s

    @property
    def calibrated(self) -> bool:
        return self._done >= CALIBRATION_WINDOW

    @property
    def rate(self) -> float:
        """Articles per second from accumulated samples (0 before any data)."""
        if self._elapsed <= 0 or self._done <= 0:
            return 0.0
        return self._done / self._elapsed

    def estimate(self) -> Estimate:
        """Time + disk range for the remaining work, calibrated from samples."""
        remaining = max(self.total_articles - self._done, 0)
        rate = self.rate
        if rate > 0:
            per_article = 1.0 / rate
            expected = remaining * per_article
            # Before calibration the early-short-article bias makes us too
            # optimistic, so the LOW band is the pessimistic (slower) one.
            if self.calibrated:
                low, high = expected * 0.75, expected * 1.25
            else:
                low, high = expected / EARLY_PESSIMISM, expected / EARLY_PESSIMISM * 1.6
        else:
            # No samples yet: the honest pre-commit prior band (see constants).
            low, expected, high = (
                remaining / PRIOR_RATE_HIGH,
                remaining / ((PRIOR_RATE_LOW + PRIOR_RATE_HIGH) / 2),
                remaining / PRIOR_RATE_LOW,
            )
            disk_low, disk_expected, disk_high = disk_band(self.total_articles, self.depth)
            return Estimate(
                seconds_low=low,
                seconds_expected=expected,
                seconds_high=high,
                disk_bytes_low=disk_low,
                disk_bytes_expected=disk_expected,
                disk_bytes_high=disk_high,
                calibrated=False,
                articles_done=self._done,
                articles_total=self.total_articles,
                rate_articles_per_s=0.0,
            )

        disk_low, disk_expected, disk_high = disk_band(self.total_articles, self.depth)
        return Estimate(
            seconds_low=low,
            seconds_expected=expected,
            seconds_high=high,
            disk_bytes_low=disk_low,
            disk_bytes_expected=disk_expected,
            disk_bytes_high=disk_high,
            calibrated=self.calibrated,
            articles_done=self._done,
            articles_total=self.total_articles,
            rate_articles_per_s=rate,
        )


def initial_estimate(total_articles: int, depth: int) -> Estimate:
    """A pre-commit estimate before any timing data exists.

    Uses the honest prior band (``PRIOR_RATE_LOW``/``PRIOR_RATE_HIGH``, ~8x
    wide, ``calibrated=False``) so the wall-clock cost is visible BEFORE the
    user commits; the measured rate replaces it from the first batch.
    """
    t = ThroughputTracker(total_articles=total_articles, depth=depth)
    e = t.estimate()
    return e


__all__ = [
    "BYTES_PER_VECTOR",
    "CALIBRATION_WINDOW",
    "VECTORS_PER_ARTICLE_BAND",
    "Estimate",
    "ThroughputTracker",
    "disk_band",
    "initial_estimate",
]
