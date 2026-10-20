"""Synthetic TTE assembly regressions; no real artifacts are read or written."""

from datetime import datetime, timedelta
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from data_preprocessing.calculate_time_delta import calculate_time_deltas_for_column
from modeling.core_types import DatasetConfig, TaskType
from modeling.dataset_builder import (
    _load_merged_time_deltas_dataframe,
    build_time_to_event_dataset,
)
from sync_unit import SyncUnit


START = datetime(2025, 1, 1, 9, 55)
UNIT = SyncUnit("test_org", "test_system", "test_project")


class TestTTEChunkBoundaries(unittest.TestCase):
    """Partitioning must not change the combined active-minute interval series."""

    def storage(self, parts):
        """Run per-part preprocessing and serve the frames from memory."""
        sa = MagicMock()
        frames = {}
        for i, offsets in enumerate(parts):
            frame = pd.DataFrame({
                "events": pd.to_datetime([START + timedelta(minutes=m) for m in offsets]),
            })
            frames[f"fixture_{i}"] = calculate_time_deltas_for_column(
                frame, False, "events")
        sa.list_files_in_folder.return_value = list(frames)
        sa.read_picklefile_as_dataframe.side_effect = lambda path: frames[path].copy()
        return sa

    def build(self, parts, **kwargs):
        """Build with real assembly logic but without snapshot writes."""
        kwargs.setdefault("start_time", START)
        kwargs.setdefault("end_time", START + timedelta(minutes=20))
        config = DatasetConfig(task_type=TaskType.TIME_TO_EVENT, **kwargs)
        with patch("modeling.dataset_builder._save_snapshot_metadata", return_value="fixture"):
            return build_time_to_event_dataset(UNIT, config, self.storage(parts))[0]

    def test_single_chunk_preserves_existing_targets_and_grid_start(self):
        """Retain the previous initial-row omission and strict-next targets."""
        parts = [[0, 5, 15, 20]]
        irregular = self.build(parts)
        self.assertEqual(irregular.minutes_to_next_event, [5, 10, 5])
        regular = self.build(parts, resample_freq="1min")
        self.assertEqual(regular.timestamps[0], START + timedelta(minutes=5))
        self.assertEqual(regular.minutes_to_next_event,
                         list(range(10, 0, -1)) + list(range(5, 0, -1)) + [1440])
        self.assertEqual(regular.covariates["elapsed_since_last_event"],
                         list(range(10)) + list(range(5)) + [0])

    def test_chronological_chunks_recover_cross_boundary_interval(self):
        """The first event of the second part has a global predecessor."""
        result = self.build([[0, 5], [15, 20]])
        self.assertEqual(result.timestamps,
                         [START + timedelta(minutes=m) for m in (5, 15, 20)])
        self.assertEqual(result.minutes_to_next_event, [5, 10, 5])

    def test_regular_grid_retains_boundary_event_and_elapsed_reset(self):
        """An internal boundary event contributes to waiting and elapsed times."""
        result = self.build([[0, 5], [15, 20]], resample_freq="1min")
        # At 10:00 the next event is 10:10, not 10:15.
        self.assertEqual(result.minutes_to_next_event[0], 10)
        self.assertEqual(result.minutes_to_next_event[5], 5)
        self.assertEqual(result.minutes_to_next_event[10], 5)
        self.assertEqual(result.covariates["elapsed_since_last_event"][10], 0)

    def test_partition_order_overlap_and_single_event_chunks_are_invariant(self):
        """Equivalent active-minute unions yield identical datasets and splits."""
        partitions = (
            [[15, 20], [0, 5]],
            [[0, 15], [5, 20]],
            [[0, 5, 15], [5, 15, 20]],
            [[0], [5], [15], [20]],
            [[], [0, 5], [], [15, 20]],
        )
        for frequency in (None, "1min", "5min"):
            expected = self.build([[0, 5, 15, 20]], resample_freq=frequency)
            for parts in partitions:
                with self.subTest(frequency=frequency, parts=parts):
                    result = self.build(parts, resample_freq=frequency)
                    self.assertEqual(result, expected)

    def test_reconstruction_precedes_time_and_min_delta_filters(self):
        """Filter a reconstructed interval, keeping its pre-window predecessor."""
        result = self.build([[0, 5], [15, 20]],
                            start_time=START + timedelta(minutes=15),
                            end_time=START + timedelta(minutes=20),
                            filters={"min_delta": 6})
        self.assertEqual(result.timestamps, [START + timedelta(minutes=15)])
        self.assertEqual(result.minutes_to_next_event, [10])

    def test_two_single_event_chunks_provide_one_completed_interval(self):
        """Locally undefined intervals can form a valid global interval."""
        result = self.build([[0], [5]])
        self.assertEqual(result.minutes_to_next_event, [5])
        regular = self.build([[0], [5]], resample_freq="1min")
        self.assertEqual(regular.minutes_to_next_event, [1440])

    def test_no_completed_interval_still_raises(self):
        """Empty traces or one distinct minute cannot yield interval samples."""
        for parts in ([[]], [[0]], [[0], [0]]):
            with self.subTest(parts=parts):
                with self.assertRaisesRegex(ValueError, "No time delta data"):
                    self.build(parts)

    def test_loader_normalizes_timezones_and_discards_missing_timestamps(self):
        """UTC ordering bridges parts without treating NaT as an event."""
        sa = MagicMock()
        sa.list_files_in_folder.return_value = ["first", "second"]
        sa.read_picklefile_as_dataframe.side_effect = [
            pd.DataFrame({"events": ["2025-01-01 10:00+00:00", None],
                          "events_delta": pd.array([None, 5], dtype="Int64")}),
            pd.DataFrame({"events": ["2025-01-01 11:10+01:00"],
                          "events_delta": pd.array([None], dtype="Int64")}),
        ]
        result = _load_merged_time_deltas_dataframe(sa, UNIT)
        self.assertEqual(result.timestamp.tolist(), [datetime(2025, 1, 1, 10, 10)])
        self.assertEqual(result.time_delta.tolist(), [10.0])


if __name__ == "__main__":
    unittest.main()
