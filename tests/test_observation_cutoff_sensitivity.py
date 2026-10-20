"""Tests for observable-stream cutoff sensitivity metrics."""

import unittest

import pandas as pd

from analysis.observation_cutoff_sensitivity import (
    compute_lifecycle_metrics,
    compute_pattern_metrics,
    compute_split_coverage,
    compute_window_metrics,
)


class TestCutoffSensitivity(unittest.TestCase):
    """Validate filtering and metric calculations on a tiny synthetic stream."""

    def setUp(self):
        self.events = pd.DataFrame(
            {
                "minute": pd.to_datetime(
                    [
                        "2021-12-31T23:59:00Z",
                        "2022-01-01T00:00:00Z",
                        "2022-01-01T00:10:00Z",
                        "2022-01-02T00:00:00Z",
                    ]
                ),
                "events_per_minute": [1, 4, 2, 3],
            }
        )

    def test_cutoff_filters_before_computing_metrics(self):
        """Events before the requested start do not contribute to metrics."""
        result = compute_window_metrics(
            self.events, start="2022-01-01T00:00:00Z"
        )

        self.assertEqual(result["event_minutes"], 3)
        self.assertEqual(result["observed_events"], 9)
        self.assertEqual(result["effective_start"], "2022-01-01T00:00:00+00:00")
        self.assertAlmostEqual(result["minute_density_per_day"], 3.0)
        self.assertEqual(result["regime"], "very-sparse")

    def test_iat_and_minute_concentration(self):
        """IAT and minute-of-hour concentration use the filtered stream."""
        result = compute_window_metrics(
            self.events, start="2022-01-01T00:00:00Z"
        )

        self.assertEqual(result["median_iat_minutes"], 720.0)
        self.assertEqual(result["events_per_active_minute"], 3.0)
        self.assertEqual(result["peak_minute"], 0)
        self.assertEqual(result["top3_minutes"], "00,10")
        self.assertAlmostEqual(result["top3_minute_share"], 1.0)

    def test_naive_and_aware_timestamps_are_normalized_to_utc(self):
        """Naive artifact timestamps remain comparable with UTC cutoffs."""
        naive = self.events.copy()
        naive["minute"] = naive["minute"].dt.tz_localize(None)

        result = compute_window_metrics(naive, start="2022-01-01T00:00:00Z")

        self.assertEqual(result["event_minutes"], 3)

    def test_unsorted_timestamps_are_ordered_before_iat_calculation(self):
        """IAT remains positive when an input artifact is not chronological."""
        unsorted = self.events.iloc[[2, 0, 3, 1]]

        result = compute_window_metrics(unsorted)

        self.assertGreater(result["mean_iat_minutes"], 0)
        self.assertEqual(result["effective_start"], "2021-12-31T23:59:00+00:00")

    def test_missing_columns_are_rejected(self):
        """Malformed per-minute artifacts fail with a clear error."""
        with self.assertRaisesRegex(ValueError, "events_per_minute"):
            compute_window_metrics(pd.DataFrame({"minute": ["2022-01-01"]}))

    def test_empty_window_is_rejected(self):
        """A cutoff after the final event is rejected."""
        with self.assertRaisesRegex(ValueError, "No event-minutes"):
            compute_window_metrics(self.events, start="2025-01-01T00:00:00Z")

    def test_naive_cutoff_is_normalized_to_utc(self):
        """Naive caller bounds have the same UTC semantics as aware bounds."""
        result = compute_window_metrics(self.events, start="2022-01-01 00:00:00")

        self.assertEqual(result["event_minutes"], 3)

    def test_pattern_metrics_regularize_missing_minutes(self):
        """Detector presence is measured on a complete one-minute grid."""
        result = compute_pattern_metrics(
            self.events, start="2022-01-01T00:00:00Z"
        )

        self.assertEqual(result["regular_grid_minutes"], 1441)
        self.assertAlmostEqual(result["presence_ratio"], 3 / 1441)


class TestLifecycleCutoff(unittest.TestCase):
    """Keep only distinct record states at or after the exact UTC cutoff."""

    def test_filter_precedes_deduplication_and_record_state_counts(self):
        """An old multi-state record can become single-state after filtering."""
        states = pd.DataFrame({
            "id": ["old", "boundary", "changed", "changed", "changed", "two", "two"],
            "created_date": ["2023-08-01T00:00:00+00:00"] * 7,
            "modified_date": [
                "2023-08-29T23:59:00+00:00",
                "2023-08-30T02:00:00+02:00",  # Exactly the UTC cutoff; retained.
                "2023-08-29T23:59:00+00:00",
                "2023-08-30T00:01:00+00:00",
                "2023-08-30T00:01:00+00:00",  # Duplicate exported state.
                "2023-08-30T00:02:00+00:00",
                "2023-08-30T00:03:00+00:00",
            ],
        })
        result = compute_lifecycle_metrics(states, "2023-08-30T00:00:00Z")
        self.assertEqual(result["records_with_state"], 3)
        self.assertEqual(result["observed_states"], 4)
        self.assertAlmostEqual(result["pct_one_state_records"], 2 / 3)


class TestSplitCoverage(unittest.TestCase):
    """Validate connection overlap against inclusive minute-grid splits."""

    def setUp(self):
        self.split = {
            "data_start": "2024-01-01T00:00:00Z",
            "train_end": "2024-01-01T00:09:00Z",
            "val_end": "2024-01-01T00:14:00Z",
            "data_end": "2024-01-01T00:19:00Z",
        }

    def test_cutoff_can_leave_partial_validation_and_full_test(self):
        """A second-precision cutoff advances conservatively to the next minute."""
        result = compute_split_coverage(
            self.split, "2024-01-01T00:12:30Z"
        )

        self.assertEqual(result["train"]["classification"], "none")
        self.assertEqual(result["validation"]["post_cutoff_minutes"], 2)
        self.assertEqual(result["validation"]["classification"], "partial")
        self.assertEqual(result["test"]["classification"], "full")

    def test_cutoff_before_data_makes_every_split_full(self):
        """A cutoff before the artifact makes all split minutes observable."""
        result = compute_split_coverage(self.split, "2023-12-01T00:00:00Z")

        self.assertTrue(
            all(item["classification"] == "full" for item in result.values())
        )


if __name__ == "__main__":
    unittest.main()
