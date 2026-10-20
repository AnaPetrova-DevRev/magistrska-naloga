"""Tests for window-aware observable-stream characterization helpers."""

import unittest

import pandas as pd

from dataset_comparison.windowed import (
    AnalysisWindow,
    build_merged_event_profile,
    build_windowed_record_analysis,
    build_windowed_time_deltas,
    filter_timestamp_window,
    summarize_lifecycle,
    windows_for_unit,
)


class TestAnalysisWindows(unittest.TestCase):
    """Validate canonical and modeled window contracts."""

    def test_modeled_window_is_unit_specific(self):
        """Only the final Novacore Zendesk suite uses a modeled cutoff."""
        zendesk = windows_for_unit("novacore_zendesk_novacore_zendesk")[-1]
        other = windows_for_unit("novacore_jira_bugs")[-1]

        self.assertEqual(zendesk.start, "2022-01-01T00:00:00Z")
        self.assertIsNone(other.start)

    def test_bounds_are_inclusive_and_timezone_normalized(self):
        """Naive timestamps at either boundary remain in the result."""
        frame = pd.DataFrame(
            {"time": pd.to_datetime(["2021-01-01", "2022-01-01", "2023-01-01"])}
        )
        window = AnalysisWindow("bounded", "2021-01-01", "2022-01-01")

        result = filter_timestamp_window(frame, "time", window)

        self.assertEqual(len(result), 2)
        self.assertEqual(str(result["time"].dt.tz), "UTC")


class TestWindowedDeltas(unittest.TestCase):
    """Validate within-window IAT reconstruction."""

    def test_first_delta_has_no_pre_window_predecessor(self):
        """Filtering first and differencing second removes cross-boundary IATs."""
        epm = pd.DataFrame(
            {
                "minute": pd.to_datetime(
                    ["2021-12-31T23:00Z", "2022-01-01T00:00Z", "2022-01-01T00:10Z"]
                ),
                "events_per_minute": [1, 1, 1],
            }
        )
        filtered = filter_timestamp_window(
            epm, "minute", AnalysisWindow("post_2022", "2022-01-01T00:00Z")
        )

        result = build_windowed_time_deltas(filtered)

        self.assertTrue(pd.isna(result["events_delta"].iloc[0]))
        self.assertEqual(result["events_delta"].iloc[1], 10)

    def test_merged_profile_weights_creation_and_modification_events(self):
        """DOW and HOD profiles use merged-stream event magnitudes consistently."""
        epm = pd.DataFrame(
            {
                "minute": pd.to_datetime(
                    ["2022-01-03T09:00Z", "2022-01-08T20:00Z"]
                ),
                "events_per_minute": [9, 1],
            }
        )

        profile = build_merged_event_profile(epm).iloc[0]

        self.assertEqual(profile["pct_changes_monday"], 90.0)
        self.assertEqual(profile["pct_changes_saturday"], 10.0)
        self.assertEqual(profile["num_changes_hour_09"], 0.9)
        self.assertEqual(profile["num_changes_hour_20"], 0.1)


class TestWindowedLifecycle(unittest.TestCase):
    """Validate distinct exported-state lifecycle semantics."""

    def setUp(self):
        self.clean = pd.DataFrame(
            {
                "id": ["a", "a", "a", "a", "b", "b"],
                "created_date": pd.to_datetime(
                    ["2020-01-01"] * 4 + ["2022-01-01"] * 2, utc=True
                ),
                "modified_date": pd.to_datetime(
                    [
                        "2021-01-01", "2022-01-01", "2022-01-01",
                        "2022-01-03", "2022-01-01", "2022-01-02",
                    ],
                    utc=True,
                ),
            }
        )

    def test_duplicate_states_do_not_inflate_counts(self):
        """Repeated `(id, modified_date)` states count once."""
        lifecycle, _ = build_windowed_record_analysis(
            self.clean, AnalysisWindow("post_2022", "2022-01-01T00:00Z")
        )
        counts = lifecycle.set_index("id")["num_changes_total"].to_dict()

        self.assertEqual(counts, {"a": 2, "b": 2})

    def test_summary_uses_claim_safe_observed_state_metrics(self):
        """Lifecycle summary describes retained states and activity spans."""
        lifecycle, profile = build_windowed_record_analysis(
            self.clean, AnalysisWindow("post_2022", "2022-01-01T00:00Z")
        )

        summary = summarize_lifecycle(lifecycle)

        self.assertEqual(summary["records_with_state"], 2)
        self.assertEqual(summary["observed_states"], 4)
        self.assertEqual(summary["median_states_per_record"], 2.0)
        self.assertEqual(len(profile.columns), 31)

    def test_creation_span_and_observed_activity_span_are_separate(self):
        """Legacy creation span is not reused for within-window activity span."""
        lifecycle, _ = build_windowed_record_analysis(
            self.clean, AnalysisWindow("post_2022", "2022-01-01T00:00Z")
        )
        record = lifecycle.set_index("id").loc["a"]

        self.assertGreater(
            record["time_first_created_to_last_modified"],
            record["observed_activity_span_seconds"],
        )
        self.assertEqual(record["observed_activity_span_seconds"], 2 * 86400)


if __name__ == "__main__":
    unittest.main()
