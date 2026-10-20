"""Tests for the covariate builder module."""

import math
import unittest
from datetime import datetime

from modeling.covariate_builder import (
    AVAILABLE_COVARIATES,
    build_covariates,
    get_available_covariates,
    split_covariate_names,
    validate_covariate_names,
    _DATA_DERIVED_COVARIATES as DATA_DERIVED_COVARIATE_NAMES,
)
from modeling.cron_pattern_detector import (
    ALL_CRON_COVARIATE_NAMES,
    CRON_COVARIATE_NAMES,
    HOURLY_CRON_COVARIATE_NAMES,
)


class TestCovariateBuilder(unittest.TestCase):

    def setUp(self):
        # Monday 2024-03-11 14:30  (weekday=0)
        self.mon_afternoon = datetime(2024, 3, 11, 14, 30)
        # Saturday 2024-03-16 02:15 (weekday=5)
        self.sat_early = datetime(2024, 3, 16, 2, 15)
        self.timestamps = [self.mon_afternoon, self.sat_early]

    def test_get_available_covariates_returns_sorted(self):
        result = get_available_covariates()
        self.assertEqual(result, sorted(result))
        self.assertEqual(len(result), len(AVAILABLE_COVARIATES))

    def test_validate_accepts_valid_names(self):
        validate_covariate_names(["hour_of_day", "day_of_week"])

    def test_validate_rejects_unknown_names(self):
        with self.assertRaises(ValueError) as ctx:
            validate_covariate_names(["hour_of_day", "bogus_feature"])
        self.assertIn("bogus_feature", str(ctx.exception))

    def test_hour_of_day(self):
        result = build_covariates(self.timestamps, ["hour_of_day"])
        self.assertEqual(result["hour_of_day"], [14.0, 2.0])

    def test_day_of_week(self):
        result = build_covariates(self.timestamps, ["day_of_week"])
        self.assertEqual(result["day_of_week"], [0.0, 5.0])

    def test_is_weekend(self):
        result = build_covariates(self.timestamps, ["is_weekend"])
        self.assertEqual(result["is_weekend"], [0.0, 1.0])

    def test_day_of_month(self):
        result = build_covariates(self.timestamps, ["day_of_month"])
        self.assertEqual(result["day_of_month"], [11.0, 16.0])

    def test_month(self):
        result = build_covariates(self.timestamps, ["month"])
        self.assertEqual(result["month"], [3.0, 3.0])

    def test_quarter(self):
        ts = [datetime(2024, 1, 1), datetime(2024, 5, 1), datetime(2024, 10, 1)]
        result = build_covariates(ts, ["quarter"])
        self.assertEqual(result["quarter"], [1.0, 2.0, 4.0])

    def test_minute_of_hour(self):
        result = build_covariates(self.timestamps, ["minute_of_hour"])
        self.assertEqual(result["minute_of_hour"], [30.0, 15.0])

    def test_is_business_hours(self):
        result = build_covariates(self.timestamps, ["is_business_hours"])
        # Monday 14:30 = business hours, Saturday 02:15 = not
        self.assertEqual(result["is_business_hours"], [1.0, 0.0])

    def test_hours_since_midnight(self):
        result = build_covariates(self.timestamps, ["hours_since_midnight"])
        self.assertAlmostEqual(result["hours_since_midnight"][0], 14.5)
        self.assertAlmostEqual(result["hours_since_midnight"][1], 2.25)

    def test_sin_cos_hour(self):
        result = build_covariates(self.timestamps, ["sin_hour", "cos_hour"])
        two_pi = 2.0 * math.pi
        self.assertAlmostEqual(result["sin_hour"][0], math.sin(two_pi * 14 / 24))
        self.assertAlmostEqual(result["cos_hour"][0], math.cos(two_pi * 14 / 24))

    def test_sin_cos_day_of_week(self):
        result = build_covariates(self.timestamps, ["sin_day_of_week", "cos_day_of_week"])
        two_pi = 2.0 * math.pi
        self.assertAlmostEqual(result["sin_day_of_week"][0], math.sin(two_pi * 0 / 7))
        self.assertAlmostEqual(result["cos_day_of_week"][1], math.cos(two_pi * 5 / 7))

    def test_output_lengths_match(self):
        # Only timestamp-extractable covariates: cron covariates require pattern
        # data, and data-derived ones need the event series, so build_covariates
        # deliberately omits both from its output
        excluded = ALL_CRON_COVARIATE_NAMES | DATA_DERIVED_COVARIATE_NAMES
        names = [n for n in get_available_covariates() if n not in excluded]
        result = build_covariates(self.timestamps, names)
        self.assertEqual(sorted(result), sorted(names))
        for name in names:
            self.assertEqual(len(result[name]), 2, f"Mismatch for {name}")

    def test_data_derived_covariates_are_not_emitted(self):
        # elapsed_since_last_event is a valid name but cannot be computed from
        # timestamps alone; it is added downstream for TTE 1-min datasets
        names = ["hour_of_day", *DATA_DERIVED_COVARIATE_NAMES]
        result = build_covariates(self.timestamps, names)
        self.assertEqual(sorted(result), ["hour_of_day"])

    def test_empty_covariate_list_returns_empty_dict(self):
        result = build_covariates(self.timestamps, [])
        self.assertEqual(result, {})

    def test_empty_timestamps_returns_empty_lists(self):
        result = build_covariates([], ["hour_of_day"])
        self.assertEqual(result, {"hour_of_day": []})


class TestCronCovariateIntegration(unittest.TestCase):
    """Tests for cron covariate names in the covariate registry."""

    def test_cron_names_in_available_covariates(self):
        """Cron covariate names should be recognized as available."""
        for name in CRON_COVARIATE_NAMES:
            self.assertIn(name, AVAILABLE_COVARIATES)

    def test_validate_accepts_cron_names(self):
        """validate_covariate_names should accept cron covariate names."""
        validate_covariate_names(list(CRON_COVARIATE_NAMES))

    def test_build_covariates_raises_on_cron_names(self):
        """build_covariates cannot handle cron names — callers should split first."""
        ts = [datetime(2024, 1, 1)]
        with self.assertRaises(ValueError):
            build_covariates(ts, ["is_cron_window"])

    def test_split_covariate_names_separates_correctly(self):
        """split_covariate_names splits regular and cron names."""
        names = ["hour_of_day", "is_cron_window", "day_of_week", "minutes_to_next_cron"]
        regular, cron = split_covariate_names(names)
        self.assertEqual(regular, ["hour_of_day", "day_of_week"])
        self.assertEqual(cron, ["is_cron_window", "minutes_to_next_cron"])

    def test_split_no_cron_names(self):
        """split_covariate_names with no cron names returns empty cron list."""
        regular, cron = split_covariate_names(["hour_of_day", "is_weekend"])
        self.assertEqual(regular, ["hour_of_day", "is_weekend"])
        self.assertEqual(cron, [])

    def test_split_only_cron_names(self):
        """split_covariate_names with only cron names returns empty regular list."""
        regular, cron = split_covariate_names(["is_cron_window", "cron_expected_magnitude"])
        self.assertEqual(regular, [])
        self.assertEqual(cron, ["is_cron_window", "cron_expected_magnitude"])

    def test_split_rejects_unknown_names(self):
        """split_covariate_names raises on unknown names."""
        with self.assertRaises(ValueError):
            split_covariate_names(["hour_of_day", "bogus"])

    # --- Hourly cron covariate integration ---

    def test_hourly_cron_names_in_available(self):
        """Hourly cron covariate names should be recognized as available."""
        for name in HOURLY_CRON_COVARIATE_NAMES:
            self.assertIn(name, AVAILABLE_COVARIATES)

    def test_validate_accepts_hourly_cron_names(self):
        """validate_covariate_names should accept hourly cron names."""
        validate_covariate_names(list(HOURLY_CRON_COVARIATE_NAMES))

    def test_build_covariates_raises_on_hourly_cron_names(self):
        """build_covariates cannot handle hourly cron names."""
        ts = [datetime(2024, 1, 1)]
        with self.assertRaises(ValueError):
            build_covariates(ts, ["is_hourly_cron_window"])

    def test_split_includes_hourly_cron_in_cron_bucket(self):
        """split_covariate_names routes hourly cron names to the cron list."""
        names = [
            "hour_of_day",
            "is_cron_window",
            "is_hourly_cron_window",
            "day_of_week",
        ]
        regular, cron = split_covariate_names(names)
        self.assertEqual(regular, ["hour_of_day", "day_of_week"])
        self.assertIn("is_cron_window", cron)
        self.assertIn("is_hourly_cron_window", cron)

    def test_split_only_hourly_cron_names(self):
        """split_covariate_names with only hourly cron names."""
        regular, cron = split_covariate_names([
            "is_hourly_cron_window", "hourly_cron_expected_magnitude",
        ])
        self.assertEqual(regular, [])
        self.assertEqual(len(cron), 2)


if __name__ == "__main__":
    unittest.main()
