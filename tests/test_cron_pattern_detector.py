"""Tests for modeling.cron_pattern_detector."""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta

from modeling.cron_pattern_detector import (
    CronPattern,
    HourlyCronPattern,
    build_cron_covariates,
    build_hourly_cron_covariates,
    detect_cron_patterns,
    detect_hourly_patterns,
    load_cron_patterns,
    load_hourly_cron_patterns,
    save_cron_patterns,
)


def _generate_daily_data(
    n_days: int,
    cron_slots: list[tuple[int, int]],
    cron_magnitude: float = 10.0,
    active_fraction: float = 1.0,
) -> tuple[list[datetime], list[float]]:
    """Generate minute-level event data with cron-like patterns.

    Args:
        n_days: Number of days of data to generate.
        cron_slots: List of (hour, minute) tuples for cron jobs.
        cron_magnitude: Event count at each cron slot.
        active_fraction: Fraction of days where the cron job fires (1.0 = every day).

    Returns:
        (timestamps, values) aligned minute-by-minute.
    """
    timestamps = []
    values = []
    base = datetime(2026, 1, 1, 0, 0)

    for day in range(n_days):
        for hour in range(24):
            for minute in range(60):
                ts = base + timedelta(days=day, hours=hour, minutes=minute)
                val = 0.0
                for ch, cm in cron_slots:
                    if hour == ch and minute == cm:
                        # fire on active_fraction of days
                        if day < int(n_days * active_fraction):
                            val = cron_magnitude
                timestamps.append(ts)
                values.append(val)

    return timestamps, values


class TestDetectCronPatterns(unittest.TestCase):
    """Tests for detect_cron_patterns()."""

    def test_detects_daily_cron(self):
        """Pattern at 09:00 every day for 30 days should be detected."""
        ts, vals = _generate_daily_data(30, [(9, 0)], cron_magnitude=5.0)
        patterns = detect_cron_patterns(ts, vals, min_frequency=0.8, min_occurrences=5)

        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].hour, 9)
        self.assertEqual(patterns[0].minute, 0)
        self.assertGreaterEqual(patterns[0].frequency, 0.8)
        self.assertAlmostEqual(patterns[0].mean_magnitude, 5.0, places=1)

    def test_sporadic_not_detected_at_default(self):
        """Pattern firing only 50% of days should not pass 0.8 threshold."""
        ts, vals = _generate_daily_data(30, [(14, 30)], active_fraction=0.5)
        patterns = detect_cron_patterns(ts, vals, min_frequency=0.8, min_occurrences=5)

        self.assertEqual(len(patterns), 0)

    def test_sporadic_detected_at_lower_threshold(self):
        """Same pattern detected when threshold is lowered to 0.4."""
        ts, vals = _generate_daily_data(30, [(14, 30)], active_fraction=0.5)
        patterns = detect_cron_patterns(ts, vals, min_frequency=0.4, min_occurrences=5)

        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].hour, 14)
        self.assertEqual(patterns[0].minute, 30)

    def test_multiple_patterns(self):
        """Two cron jobs at different times both detected."""
        ts, vals = _generate_daily_data(30, [(9, 0), (17, 15)], cron_magnitude=8.0)
        patterns = detect_cron_patterns(ts, vals, min_frequency=0.8, min_occurrences=5)

        self.assertEqual(len(patterns), 2)
        slots = [(p.hour, p.minute) for p in patterns]
        self.assertIn((9, 0), slots)
        self.assertIn((17, 15), slots)

    def test_no_patterns_in_empty_data(self):
        """No data produces no patterns."""
        patterns = detect_cron_patterns([], [], min_frequency=0.8)
        self.assertEqual(len(patterns), 0)

    def test_no_patterns_single_day(self):
        """Single day of data is not enough to establish patterns."""
        ts, vals = _generate_daily_data(1, [(9, 0)])
        patterns = detect_cron_patterns(ts, vals, min_frequency=0.8, min_occurrences=5)
        self.assertEqual(len(patterns), 0)

    def test_min_occurrences_filter(self):
        """Pattern on 3 of 4 days (75%) passes frequency but fails min_occurrences=5."""
        ts, vals = _generate_daily_data(4, [(10, 0)], active_fraction=0.75)
        patterns = detect_cron_patterns(
            ts, vals, min_frequency=0.7, min_occurrences=5
        )
        self.assertEqual(len(patterns), 0)

        # With lower min_occurrences it passes
        patterns = detect_cron_patterns(
            ts, vals, min_frequency=0.7, min_occurrences=2
        )
        self.assertEqual(len(patterns), 1)

    def test_max_cv_filter_rejects_high_variance(self):
        """High-variance slot rejected by max_cv filter."""
        # Build data where magnitude varies wildly across days
        base = datetime(2026, 1, 1, 0, 0)
        timestamps = []
        values = []
        for day in range(30):
            for hour in range(24):
                for minute in range(60):
                    ts = base + timedelta(days=day, hours=hour, minutes=minute)
                    val = 0.0
                    if hour == 9 and minute == 0:
                        # Alternate between very different magnitudes
                        val = 1.0 if day % 2 == 0 else 100.0
                    timestamps.append(ts)
                    values.append(val)

        # Without CV filter -> detected
        patterns = detect_cron_patterns(
            timestamps, values, min_frequency=0.8, min_occurrences=5, max_cv=None
        )
        self.assertEqual(len(patterns), 1)

        # With strict CV filter -> rejected (cv ~ 1.0)
        patterns = detect_cron_patterns(
            timestamps, values, min_frequency=0.8, min_occurrences=5, max_cv=0.3
        )
        self.assertEqual(len(patterns), 0)

    def test_patterns_sorted_by_time(self):
        """Output patterns are sorted by (hour, minute)."""
        ts, vals = _generate_daily_data(30, [(17, 0), (3, 30), (9, 15)])
        patterns = detect_cron_patterns(ts, vals, min_frequency=0.8, min_occurrences=5)

        times = [(p.hour, p.minute) for p in patterns]
        self.assertEqual(times, sorted(times))


class TestBuildCronCovariates(unittest.TestCase):
    """Tests for build_cron_covariates()."""

    def test_is_cron_window_binary(self):
        """is_cron_window is 1.0 at cron slot, 0.0 elsewhere."""
        patterns = [CronPattern(hour=9, minute=0, frequency=0.9,
                                mean_magnitude=10.0, std_magnitude=1.0)]
        timestamps = [
            datetime(2026, 1, 1, 8, 59),
            datetime(2026, 1, 1, 9, 0),
            datetime(2026, 1, 1, 9, 1),
        ]
        covs = build_cron_covariates(timestamps, patterns)

        self.assertEqual(covs["is_cron_window"], [0.0, 1.0, 0.0])

    def test_minutes_to_next_at_cron_slot(self):
        """Countdown is 0 at the cron slot itself."""
        patterns = [CronPattern(hour=9, minute=0, frequency=0.9,
                                mean_magnitude=10.0, std_magnitude=1.0)]
        timestamps = [datetime(2026, 1, 1, 9, 0)]
        covs = build_cron_covariates(timestamps, patterns)

        self.assertAlmostEqual(covs["minutes_to_next_cron"][0], 0.0)

    def test_minutes_to_next_decreases(self):
        """Countdown decreases as we approach the cron slot."""
        patterns = [CronPattern(hour=9, minute=0, frequency=0.9,
                                mean_magnitude=10.0, std_magnitude=1.0)]
        timestamps = [
            datetime(2026, 1, 1, 8, 50),
            datetime(2026, 1, 1, 8, 55),
            datetime(2026, 1, 1, 9, 0),
        ]
        covs = build_cron_covariates(timestamps, patterns)
        vals = covs["minutes_to_next_cron"]

        self.assertGreater(vals[0], vals[1])
        self.assertGreater(vals[1], vals[2])
        self.assertAlmostEqual(vals[2], 0.0)

    def test_minutes_to_next_wraps_midnight(self):
        """Countdown wraps around midnight correctly."""
        patterns = [CronPattern(hour=2, minute=0, frequency=0.9,
                                mean_magnitude=10.0, std_magnitude=1.0)]
        timestamps = [datetime(2026, 1, 1, 23, 0)]
        covs = build_cron_covariates(timestamps, patterns)

        # 23:00 -> 02:00 next day = 180 minutes = 180/1440 normalized
        expected = 180.0 / 1440.0
        self.assertAlmostEqual(covs["minutes_to_next_cron"][0], expected, places=5)

    def test_expected_magnitude_normalized(self):
        """Magnitude is normalized by max across all patterns."""
        patterns = [
            CronPattern(hour=9, minute=0, frequency=0.9,
                        mean_magnitude=20.0, std_magnitude=1.0),
            CronPattern(hour=14, minute=0, frequency=0.9,
                        mean_magnitude=10.0, std_magnitude=1.0),
        ]
        timestamps = [
            datetime(2026, 1, 1, 9, 0),
            datetime(2026, 1, 1, 14, 0),
            datetime(2026, 1, 1, 12, 0),  # not a cron slot
        ]
        covs = build_cron_covariates(timestamps, patterns)

        self.assertAlmostEqual(covs["cron_expected_magnitude"][0], 1.0)  # 20/20
        self.assertAlmostEqual(covs["cron_expected_magnitude"][1], 0.5)  # 10/20
        self.assertAlmostEqual(covs["cron_expected_magnitude"][2], 0.0)  # not a slot

    def test_no_patterns_gives_defaults(self):
        """Empty pattern list produces constant defaults."""
        timestamps = [datetime(2026, 1, 1, h, 0) for h in range(24)]
        covs = build_cron_covariates(timestamps, [])

        self.assertTrue(all(v == 0.0 for v in covs["is_cron_window"]))
        self.assertTrue(all(v == 1.0 for v in covs["minutes_to_next_cron"]))
        self.assertTrue(all(v == 0.0 for v in covs["cron_expected_magnitude"]))

    def test_output_lengths_match_input(self):
        """All covariate lists have same length as timestamps."""
        patterns = [CronPattern(hour=9, minute=0, frequency=0.9,
                                mean_magnitude=10.0, std_magnitude=1.0)]
        timestamps = [datetime(2026, 1, 1, h, m) for h in range(24) for m in range(60)]
        covs = build_cron_covariates(timestamps, patterns)

        for name, vals in covs.items():
            self.assertEqual(len(vals), len(timestamps), f"{name} length mismatch")

    def test_all_three_keys_present(self):
        """Output dict always contains all three covariate keys."""
        covs = build_cron_covariates([], [])
        self.assertIn("is_cron_window", covs)
        self.assertIn("minutes_to_next_cron", covs)
        self.assertIn("cron_expected_magnitude", covs)


class TestCronPatternSerialization(unittest.TestCase):
    """Tests for save_cron_patterns / load_cron_patterns."""

    def test_round_trip(self):
        """Patterns survive save/load round-trip."""
        patterns = [
            CronPattern(hour=9, minute=0, frequency=0.95,
                        mean_magnitude=12.3, std_magnitude=2.1),
            CronPattern(hour=17, minute=30, frequency=0.82,
                        mean_magnitude=5.5, std_magnitude=0.8),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cron_patterns.json")
            save_cron_patterns(patterns, path, detection_config={"min_frequency": 0.8})
            loaded = load_cron_patterns(path)

        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].hour, 9)
        self.assertEqual(loaded[0].minute, 0)
        self.assertAlmostEqual(loaded[0].frequency, 0.95)
        self.assertAlmostEqual(loaded[0].mean_magnitude, 12.3)
        self.assertEqual(loaded[1].hour, 17)
        self.assertEqual(loaded[1].minute, 30)

    def test_empty_patterns_round_trip(self):
        """Empty pattern list serializes and deserializes cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cron_patterns.json")
            save_cron_patterns([], path)
            loaded = load_cron_patterns(path)

        self.assertEqual(loaded, [])

    def test_detection_config_stored(self):
        """Detection config is stored alongside patterns in the JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cron_patterns.json")
            save_cron_patterns(
                [CronPattern(1, 0, 0.9, 5.0, 1.0)],
                path,
                detection_config={"min_frequency": 0.8, "max_cv": 0.5},
            )
            with open(path) as f:
                data = json.load(f)

        self.assertEqual(data["detection_config"]["min_frequency"], 0.8)
        self.assertEqual(data["detection_config"]["max_cv"], 0.5)


def _generate_hourly_data(
    n_days: int,
    burst_minutes: list[int],
    burst_magnitude: float = 5.0,
    active_fraction: float = 1.0,
) -> tuple[list[datetime], list[float]]:
    """Generate sparse event data with hourly bursts at specific minutes.

    Only minutes with events are included (mimics real sparse data).
    """
    timestamps = []
    values = []
    base = datetime(2026, 1, 1, 0, 0)

    for day in range(n_days):
        for hour in range(24):
            # Add burst minutes
            for m in burst_minutes:
                if day < int(n_days * active_fraction):
                    ts = base + timedelta(days=day, hours=hour, minutes=m)
                    timestamps.append(ts)
                    values.append(burst_magnitude)
            # Add a few background events at random other minutes
            for m in [15, 30, 45]:
                if day % 3 == 0:  # sporadic background
                    ts = base + timedelta(days=day, hours=hour, minutes=m)
                    timestamps.append(ts)
                    values.append(1.0)

    return timestamps, values


class TestDetectHourlyPatterns(unittest.TestCase):
    """Tests for detect_hourly_patterns()."""

    def test_detects_burst_minutes(self):
        """Minutes :01 and :02 firing every hour for 30 days should be detected."""
        ts, vals = _generate_hourly_data(30, [1, 2], burst_magnitude=5.0)
        patterns = detect_hourly_patterns(ts, vals, min_frequency=0.5, min_occurrences=10)

        detected_minutes = {p.minute for p in patterns}
        self.assertIn(1, detected_minutes)
        self.assertIn(2, detected_minutes)

    def test_sporadic_background_not_detected(self):
        """Background events at :15, :30, :45 only on 1/3 of days should not pass."""
        ts, vals = _generate_hourly_data(30, [1], burst_magnitude=5.0)
        patterns = detect_hourly_patterns(ts, vals, min_frequency=0.5, min_occurrences=10)

        detected_minutes = {p.minute for p in patterns}
        self.assertNotIn(15, detected_minutes)
        self.assertNotIn(30, detected_minutes)
        self.assertNotIn(45, detected_minutes)

    def test_frequency_threshold(self):
        """Pattern at 50% active_fraction should pass 0.3 but not 0.8."""
        ts, vals = _generate_hourly_data(30, [5], active_fraction=0.5)
        patterns_low = detect_hourly_patterns(ts, vals, min_frequency=0.3, min_occurrences=5)
        patterns_high = detect_hourly_patterns(ts, vals, min_frequency=0.8, min_occurrences=5)

        self.assertTrue(any(p.minute == 5 for p in patterns_low))
        self.assertFalse(any(p.minute == 5 for p in patterns_high))

    def test_no_patterns_empty_data(self):
        """Empty data produces no patterns."""
        patterns = detect_hourly_patterns([], [], min_frequency=0.5)
        self.assertEqual(len(patterns), 0)

    def test_min_occurrences_filter(self):
        """Pattern with too few active hours is filtered."""
        ts, vals = _generate_hourly_data(2, [10], burst_magnitude=3.0)
        patterns = detect_hourly_patterns(
            ts, vals, min_frequency=0.3, min_occurrences=100
        )
        self.assertEqual(len(patterns), 0)

        # With lower min_occurrences it passes
        patterns = detect_hourly_patterns(
            ts, vals, min_frequency=0.3, min_occurrences=5
        )
        self.assertTrue(any(p.minute == 10 for p in patterns))

    def test_max_cv_filter(self):
        """High-variance hourly pattern rejected by max_cv."""
        base = datetime(2026, 1, 1, 0, 0)
        timestamps = []
        values = []
        for day in range(30):
            for hour in range(24):
                ts = base + timedelta(days=day, hours=hour, minutes=1)
                timestamps.append(ts)
                # Alternate wildly
                values.append(1.0 if day % 2 == 0 else 100.0)

        patterns_no_cv = detect_hourly_patterns(
            timestamps, values, min_frequency=0.3, min_occurrences=5, max_cv=None,
        )
        self.assertTrue(any(p.minute == 1 for p in patterns_no_cv))

        patterns_strict_cv = detect_hourly_patterns(
            timestamps, values, min_frequency=0.3, min_occurrences=5, max_cv=0.3,
        )
        self.assertFalse(any(p.minute == 1 for p in patterns_strict_cv))

    def test_patterns_sorted_by_minute(self):
        """Output patterns are sorted by minute."""
        ts, vals = _generate_hourly_data(30, [10, 2, 5])
        patterns = detect_hourly_patterns(ts, vals, min_frequency=0.3, min_occurrences=5)
        minutes = [p.minute for p in patterns]
        self.assertEqual(minutes, sorted(minutes))

    def test_relative_frequency_detects_sparse_pattern(self):
        """Relative frequency catches overrepresented minutes in sparse data."""
        # Simulate sparse event data: events scattered across many minutes
        # but :01 is clearly overrepresented (appears in ~20% of active
        # hours vs ~3% baseline for typical minutes).
        import random
        rng = random.Random(42)
        base = datetime(2026, 1, 1, 0, 0)
        timestamps = []
        values = []
        for day in range(60):
            for hour in range(24):
                # :01 fires in 20% of hours
                if rng.random() < 0.20:
                    ts = base + timedelta(days=day, hours=hour, minutes=1)
                    timestamps.append(ts)
                    values.append(5.0)
                # Background: ~2 random events per hour at varied minutes
                for _ in range(2):
                    m = rng.randint(2, 59)
                    ts = base + timedelta(days=day, hours=hour, minutes=m)
                    timestamps.append(ts)
                    values.append(1.0)

        # Absolute threshold 0.5 misses :01 (it's only ~20%)
        patterns_abs = detect_hourly_patterns(
            timestamps, values,
            min_frequency=0.5, min_occurrences=5,
            min_relative_frequency=None,  # disable relative
        )
        self.assertFalse(any(p.minute == 1 for p in patterns_abs))

        # Relative threshold 3.0x catches it (20% vs ~3% baseline = ~7x)
        patterns_rel = detect_hourly_patterns(
            timestamps, values,
            min_frequency=0.5, min_occurrences=5,
            min_relative_frequency=3.0,
        )
        self.assertTrue(any(p.minute == 1 for p in patterns_rel))

    def test_magnitude_stats(self):
        """Mean and std magnitude are computed correctly."""
        ts, vals = _generate_hourly_data(30, [1], burst_magnitude=10.0)
        patterns = detect_hourly_patterns(ts, vals, min_frequency=0.3, min_occurrences=5)
        p = next(p for p in patterns if p.minute == 1)
        self.assertAlmostEqual(p.mean_magnitude, 10.0, places=1)


class TestBuildHourlyCronCovariates(unittest.TestCase):
    """Tests for build_hourly_cron_covariates()."""

    def test_is_window_binary(self):
        """is_hourly_cron_window is 1.0 at burst minute, 0.0 elsewhere."""
        patterns = [HourlyCronPattern(minute=1, frequency=0.9,
                                       mean_magnitude=5.0, std_magnitude=1.0)]
        timestamps = [
            datetime(2026, 1, 1, 14, 0),
            datetime(2026, 1, 1, 14, 1),
            datetime(2026, 1, 1, 14, 2),
        ]
        covs = build_hourly_cron_covariates(timestamps, patterns)
        self.assertEqual(covs["is_hourly_cron_window"], [0.0, 1.0, 0.0])

    def test_minutes_to_next_at_slot(self):
        """Countdown is 0 at the burst minute itself."""
        patterns = [HourlyCronPattern(minute=5, frequency=0.9,
                                       mean_magnitude=5.0, std_magnitude=1.0)]
        timestamps = [datetime(2026, 1, 1, 10, 5)]
        covs = build_hourly_cron_covariates(timestamps, patterns)
        self.assertAlmostEqual(covs["minutes_to_next_hourly_cron"][0], 0.0)

    def test_minutes_to_next_decreases(self):
        """Countdown decreases approaching the burst minute."""
        patterns = [HourlyCronPattern(minute=10, frequency=0.9,
                                       mean_magnitude=5.0, std_magnitude=1.0)]
        timestamps = [
            datetime(2026, 1, 1, 8, 5),
            datetime(2026, 1, 1, 8, 8),
            datetime(2026, 1, 1, 8, 10),
        ]
        covs = build_hourly_cron_covariates(timestamps, patterns)
        vals = covs["minutes_to_next_hourly_cron"]
        self.assertGreater(vals[0], vals[1])
        self.assertGreater(vals[1], vals[2])
        self.assertAlmostEqual(vals[2], 0.0)

    def test_minutes_to_next_wraps_hour(self):
        """Countdown wraps around the hour boundary."""
        patterns = [HourlyCronPattern(minute=5, frequency=0.9,
                                       mean_magnitude=5.0, std_magnitude=1.0)]
        timestamps = [datetime(2026, 1, 1, 10, 50)]
        covs = build_hourly_cron_covariates(timestamps, patterns)
        # :50 -> :05 next hour = 15 minutes = 15/60 normalized
        expected = 15.0 / 60.0
        self.assertAlmostEqual(
            covs["minutes_to_next_hourly_cron"][0], expected, places=5
        )

    def test_expected_magnitude_normalized(self):
        """Magnitude normalized by max across patterns."""
        patterns = [
            HourlyCronPattern(minute=1, frequency=0.9,
                              mean_magnitude=20.0, std_magnitude=1.0),
            HourlyCronPattern(minute=2, frequency=0.9,
                              mean_magnitude=10.0, std_magnitude=1.0),
        ]
        timestamps = [
            datetime(2026, 1, 1, 10, 1),
            datetime(2026, 1, 1, 10, 2),
            datetime(2026, 1, 1, 10, 30),  # not a burst minute
        ]
        covs = build_hourly_cron_covariates(timestamps, patterns)
        self.assertAlmostEqual(covs["hourly_cron_expected_magnitude"][0], 1.0)
        self.assertAlmostEqual(covs["hourly_cron_expected_magnitude"][1], 0.5)
        self.assertAlmostEqual(covs["hourly_cron_expected_magnitude"][2], 0.0)

    def test_no_patterns_gives_defaults(self):
        """Empty patterns produce constant defaults."""
        timestamps = [datetime(2026, 1, 1, h, 0) for h in range(24)]
        covs = build_hourly_cron_covariates(timestamps, [])
        self.assertTrue(all(v == 0.0 for v in covs["is_hourly_cron_window"]))
        self.assertTrue(all(v == 1.0 for v in covs["minutes_to_next_hourly_cron"]))
        self.assertTrue(all(v == 0.0 for v in covs["hourly_cron_expected_magnitude"]))

    def test_output_lengths_match(self):
        """All covariate lists have same length as timestamps."""
        patterns = [HourlyCronPattern(minute=1, frequency=0.9,
                                       mean_magnitude=5.0, std_magnitude=1.0)]
        timestamps = [datetime(2026, 1, 1, h, m) for h in range(24) for m in range(60)]
        covs = build_hourly_cron_covariates(timestamps, patterns)
        for name, vals in covs.items():
            self.assertEqual(len(vals), len(timestamps), f"{name} length mismatch")

    def test_all_three_keys_present(self):
        """Output dict always contains all three covariate keys."""
        covs = build_hourly_cron_covariates([], [])
        self.assertIn("is_hourly_cron_window", covs)
        self.assertIn("minutes_to_next_hourly_cron", covs)
        self.assertIn("hourly_cron_expected_magnitude", covs)


class TestHourlyCronSerialization(unittest.TestCase):
    """Tests for hourly patterns in save/load round-trip."""

    def test_hourly_round_trip(self):
        """Hourly patterns survive save/load round-trip."""
        daily = [CronPattern(hour=9, minute=0, frequency=0.95,
                             mean_magnitude=12.3, std_magnitude=2.1)]
        hourly = [
            HourlyCronPattern(minute=1, frequency=0.85,
                              mean_magnitude=5.0, std_magnitude=1.0),
            HourlyCronPattern(minute=2, frequency=0.78,
                              mean_magnitude=3.0, std_magnitude=0.5),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cron_patterns.json")
            save_cron_patterns(daily, path, hourly_patterns=hourly)
            loaded_daily = load_cron_patterns(path)
            loaded_hourly = load_hourly_cron_patterns(path)

        self.assertEqual(len(loaded_daily), 1)
        self.assertEqual(loaded_daily[0].hour, 9)
        self.assertEqual(len(loaded_hourly), 2)
        self.assertEqual(loaded_hourly[0].minute, 1)
        self.assertAlmostEqual(loaded_hourly[0].frequency, 0.85)
        self.assertEqual(loaded_hourly[1].minute, 2)

    def test_backward_compat_no_hourly(self):
        """Loading hourly patterns from old file without them returns empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cron_patterns.json")
            save_cron_patterns([], path)  # no hourly_patterns param
            loaded = load_hourly_cron_patterns(path)

        self.assertEqual(loaded, [])

    def test_empty_hourly_round_trip(self):
        """Empty hourly list serializes and deserializes cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cron_patterns.json")
            save_cron_patterns([], path, hourly_patterns=[])
            loaded = load_hourly_cron_patterns(path)

        self.assertEqual(loaded, [])


if __name__ == "__main__":
    unittest.main()
