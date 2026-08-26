"""Calibrated-estimator tests (acceptance: within ±25% of actual).

The estimator's load-bearing properties:

* it is a RANGE, never a single number (disk always, time once calibrated);
* before calibration the band sits ABOVE the naive extrapolation — the
  early-articles-are-short bias must never produce a 3x optimistic number
  (08 Traps);
* after ``CALIBRATION_WINDOW`` articles the band narrows to ±25%;
* the extrapolated time-to-completion is the measured per-article cost times
  the remaining articles — the property the DoD's "within ±25% of actual"
  rests on;
* disk scales with depth via the documented vectors-per-article table.
"""

from __future__ import annotations

from vesta.index.estimate import (
    BYTES_PER_VECTOR,
    CALIBRATION_WINDOW,
    VECTORS_PER_ARTICLE_BAND,
    ThroughputTracker,
    disk_band,
    initial_estimate,
)


def test_no_samples_zero_rate_and_disk_estimate_present() -> None:
    est = initial_estimate(total_articles=10_000, depth=1)
    assert est.rate_articles_per_s == 0.0
    assert not est.calibrated
    # The pre-commit band is honest and non-zero: the cost is visible
    # BEFORE the user commits — a 0s placeholder would hide it.
    assert 0 < est.seconds_low < est.seconds_expected < est.seconds_high
    assert est.articles_total == 10_000
    assert est.articles_done == 0
    # Disk is known before any timing data: articles x vectors/article x bytes.
    assert est.disk_bytes_expected == int(
        10_000 * VECTORS_PER_ARTICLE_BAND[1][1] * BYTES_PER_VECTOR
    )
    assert est.disk_bytes_low < est.disk_bytes_expected < est.disk_bytes_high


def test_record_ignores_nonpositive_samples() -> None:
    t = ThroughputTracker(total_articles=100, depth=1)
    t.record(0, 5.0)
    t.record(10, -1.0)
    assert t.rate == 0.0
    assert t.estimate().articles_done == 0


def test_uncalibrated_band_sits_above_naive_extrapolation() -> None:
    # 100 articles in 10 s → 10/s naive → 990 s for the remaining 9900.
    t = ThroughputTracker(total_articles=10_000, depth=1)
    t.record(100, 10.0)
    est = t.estimate()
    naive = 9900 / 10.0
    assert not est.calibrated
    # The pessimism guard: the whole band must exceed the naive number, so an
    # early-short-article sample can't read as a 3x-optimistic promise.
    assert est.seconds_low > naive
    assert est.seconds_high > est.seconds_low
    # Expected time is the running per-article cost times remaining articles,
    # NOT inflated by the pessimistic band's midpoint (AUDIT_0824 S5).
    assert est.seconds_expected == naive


def test_uncalibrated_expected_not_inflated_by_band_midpoint() -> None:
    # 500 articles in 50 s -> 10 articles/s. Remaining 1500 articles -> expected = 150 s.
    t = ThroughputTracker(total_articles=2000, depth=1)
    t.record(500, 50.0)
    assert not t.calibrated
    est = t.estimate()
    expected = 1500 / 10.0  # 150.0 s
    assert est.seconds_expected == expected
    # The band midpoint would have been (low + high) / 2 = (300 + 480) / 2 = 390.0 s (2.6x inflated)
    band_midpoint = (est.seconds_low + est.seconds_high) / 2.0
    assert est.seconds_expected != band_midpoint
    assert est.seconds_expected == expected


def test_calibrated_band_narrows_to_plus_minus_25pct() -> None:
    t = ThroughputTracker(total_articles=CALIBRATION_WINDOW * 2, depth=1)
    # Exactly 5 articles/s throughout → remaining = 1000 → expected = 200 s.
    t.record(CALIBRATION_WINDOW, CALIBRATION_WINDOW / 5.0)
    est = t.estimate()
    assert est.calibrated
    expected = CALIBRATION_WINDOW / 5.0
    assert est.seconds_low == expected * 0.75
    assert est.seconds_high == expected * 1.25
    assert est.seconds_expected == expected


def test_estimate_tracks_true_completion_within_dod_tolerance() -> None:
    # Synthetic steady-state run: the job records at 40% and the projection of
    # the REMAINING time must land within the DoD's ±25% of the actual.
    total, rate = 5_000, 8.0  # articles/s
    done_at_40 = int(total * 0.4)
    t = ThroughputTracker(total_articles=total, depth=2)
    t.record(done_at_40, done_at_40 / rate)
    actual_remaining = (total - done_at_40) / rate
    est = t.estimate()
    assert abs(est.seconds_expected - actual_remaining) / actual_remaining <= 0.25
    assert est.seconds_low <= actual_remaining <= est.seconds_high


def test_disk_scales_with_depth() -> None:
    d1 = initial_estimate(1_000, 1)
    d3 = initial_estimate(1_000, 3)
    assert d3.disk_bytes_expected > d1.disk_bytes_expected
    ratio = d3.disk_bytes_expected / d1.disk_bytes_expected
    assert ratio == VECTORS_PER_ARTICLE_BAND[3][1] / VECTORS_PER_ARTICLE_BAND[1][1]


def test_completed_run_estimates_zero_remaining() -> None:
    t = ThroughputTracker(total_articles=500, depth=1)
    t.record(500, 100.0)
    est = t.estimate()
    assert est.articles_done == 500
    assert est.seconds_high == 0.0
    assert est.rate_articles_per_s == 5.0


# ── vectors-per-article and disk bands (measured 2026-08-19) ─────────────────


def test_static_vector_and_disk_bands() -> None:
    """Static band invariants (measured 2026-08-19):
    - Depth 1 emits 1 vector/article (no upside spread: low < expected == high == 1.0).
    - Depth 3 re-splits short sections, yielding fewer chunks at depth 3 than depth 2.
    - All bands (1, 2, 3) bracket expected values and disk bands are strictly ordered.
    - Disk band brackets live index ground truth and unknown depth falls back to depth 1.
    """
    # Depth 1 is exactly 1.0 with no upside spread.
    assert VECTORS_PER_ARTICLE_BAND[1][1] == 1.0
    low, expected, high = VECTORS_PER_ARTICLE_BAND[1]
    assert low < expected == high == 1.0

    # Depth 3 short-section inversion vs depth 2.
    assert VECTORS_PER_ARTICLE_BAND[3][0] < VECTORS_PER_ARTICLE_BAND[2][0]

    # Ordering across all valid depths.
    for depth in (1, 2, 3):
        v_low, v_exp, v_high = VECTORS_PER_ARTICLE_BAND[depth]
        assert v_low <= v_exp <= v_high

        d_low, d_exp, d_high = disk_band(10_000, depth)
        assert d_low < d_exp < d_high, f"depth {depth}: {d_low} {d_exp} {d_high}"

    # Ground truth: 9 installed archives hold 151 057 articles and 1 074 717 chunks.
    d_low, _d_exp, d_high = disk_band(151_057, 2)
    actual = 1_074_717 * BYTES_PER_VECTOR
    assert d_low < actual < d_high

    # Unknown depth falls back to depth 1.
    assert disk_band(10_000, 99) == disk_band(10_000, 1)
