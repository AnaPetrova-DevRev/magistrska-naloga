"""Tests for the DatasetBuilder (EVENT_RATE and TIME_TO_EVENT)."""

from datetime import datetime, timedelta
import os
from pathlib import Path
import shutil
import unittest

import pandas as pd

import global_vars
from modeling.core_types import DatasetConfig, TaskType
from modeling.dataset_builder import build_event_rate_dataset, build_time_to_event_dataset
from storage_access import StorageAccess
from sync_unit import SyncUnit


class TestEventRateDatasetBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.sa = StorageAccess()
        self.sync_unit = SyncUnit("test_org", "test_system", "test_project")
        base_folder = f"{self.sync_unit.get_path_for_sync_unit()}/{global_vars.DATA_PREPROCESSING_APP_FOLDER_NAME}"
        self.sa.create_folder(base_folder)

        # Also create a minimal initial_data file so that the dataset
        # builder can discover the chunk via the existing pipeline
        # conventions (work_*.json -> CALCULATE_MERGED_EVENTS_PER_MINUTE
        # outputs).
        initial_folder = f"{self.sync_unit.get_path_for_sync_unit()}/{global_vars.INITIAL_DATA_FOLDER_NAME}"
        self.sa.create_folder(initial_folder)
        initial_rel_path = f"{initial_folder}/{global_vars.INITIAL_DATA_FILE_NAME_FORMAT.format(int=1)}"
        # Content is irrelevant for the builder; it only needs the file
        # to exist and have the correct name pattern.
        self.sa.write_json_to_storage(initial_rel_path, {"dummy": True})

        # Create a small synthetic events-per-minute dataframe
        start = datetime(2024, 1, 1, 0, 0, 0)
        self.minutes = [start + timedelta(minutes=i) for i in range(3)]
        df = pd.DataFrame(
            {
                "minute": self.minutes,
                "events_per_minute": [1.0, 2.0, 3.0],
            }
        )

        events_rel_path = f"{base_folder}/{global_vars.MERGED_EVENTS_PER_MINUTE_FILE_NAME_FORMAT.format(int=1)}"
        self.sa.save_dataframe_as_picklefile(df, events_rel_path)

    def tearDown(self) -> None:
        """Clean up test data created during setUp."""
        test_data_path = self.sa.create_full_path_string(
            self.sync_unit.get_path_for_sync_unit()
        )
        # Only remove modeling-specific artifacts, not shared state files
        for subfolder in ["modeling", "initial_data", global_vars.DATA_PREPROCESSING_APP_FOLDER_NAME]:
            path = os.path.join(test_data_path, subfolder)
            if os.path.exists(path):
                shutil.rmtree(path)

    # --- helpers for writing extra test data ---

    def _write_events_pkl(self, df: pd.DataFrame, chunk: int = 2) -> None:
        base_folder = (
            f"{self.sync_unit.get_path_for_sync_unit()}/"
            f"{global_vars.DATA_PREPROCESSING_APP_FOLDER_NAME}"
        )
        path = (
            f"{base_folder}/"
            f"{global_vars.MERGED_EVENTS_PER_MINUTE_FILE_NAME_FORMAT.format(int=chunk)}"
        )
        self.sa.save_dataframe_as_picklefile(df, path)

    # Original test

    def test_build_event_rate_dataset_basic(self) -> None:
        cfg = DatasetConfig(
            task_type=TaskType.EVENT_RATE,
            start_time=self.minutes[0],
            end_time=self.minutes[-1],
            train_ratio=0.67,
            val_ratio=0.0,
        )

        dataset, meta = build_event_rate_dataset(self.sync_unit, cfg)

        # Dataset content
        self.assertEqual(len(dataset.timestamps), 3)
        self.assertEqual(dataset.events_per_minute, [1.0, 2.0, 3.0])

        # Indices should cover the full range without duplicates
        all_indices = set(dataset.train_indices + dataset.val_indices + dataset.test_indices)
        self.assertEqual(all_indices, {0, 1, 2})

        # Snapshot metadata JSON should have been written to storage
        full_meta_path = self.sa.create_full_path_string(meta.snapshot_metadata_path)
        self.assertTrue(Path(full_meta_path).is_file())
        self.assertTrue(full_meta_path.endswith(".json"))


    # Resampling tests

    def test_resample_fills_gaps_with_zeros(self) -> None:
        """Resampling at 1min should zero-fill missing minutes."""
        start = datetime(2024, 6, 1, 0, 0, 0)
        sparse_df = pd.DataFrame({
            "minute": [
                start,
                start + timedelta(minutes=5),
                start + timedelta(minutes=10),
            ],
            "events_per_minute": [3.0, 7.0, 2.0],
        })
        self._write_events_pkl(sparse_df, chunk=2)

        cfg = DatasetConfig(
            task_type=TaskType.EVENT_RATE,
            start_time=start,
            end_time=start + timedelta(minutes=10),
            train_ratio=0.5,
            val_ratio=0.2,
            resample_freq="1min",
        )
        dataset, _meta = build_event_rate_dataset(self.sync_unit, cfg)

        self.assertEqual(len(dataset.timestamps), 11)
        self.assertEqual(dataset.events_per_minute[0], 3.0)
        self.assertEqual(dataset.events_per_minute[5], 7.0)
        self.assertEqual(dataset.events_per_minute[10], 2.0)
        for i in [1, 2, 3, 4, 6, 7, 8, 9]:
            self.assertEqual(dataset.events_per_minute[i], 0.0)

    def test_resample_aggregates_within_bucket(self) -> None:
        """Events in multiple minutes should sum within a coarser bucket."""
        start = datetime(2024, 6, 1, 0, 0, 0)
        df = pd.DataFrame({
            "minute": [
                start,
                start + timedelta(minutes=1),
                start + timedelta(minutes=14),
                start + timedelta(minutes=15),
            ],
            "events_per_minute": [2.0, 3.0, 4.0, 1.0],
        })
        self._write_events_pkl(df, chunk=2)

        cfg = DatasetConfig(
            task_type=TaskType.EVENT_RATE,
            start_time=start,
            end_time=start + timedelta(minutes=15),
            train_ratio=0.5,
            val_ratio=0.5,
            resample_freq="15min",
        )
        dataset, _meta = build_event_rate_dataset(self.sync_unit, cfg)

        # Two 15-min buckets: [00:00–00:14] and [00:15–00:29]
        self.assertEqual(len(dataset.timestamps), 2)
        self.assertEqual(dataset.events_per_minute[0], 9.0)  # 2+3+4
        self.assertEqual(dataset.events_per_minute[1], 1.0)

    def test_resample_none_preserves_behavior(self) -> None:
        """resample_freq=None should give the same result as no resampling."""
        cfg = DatasetConfig(
            task_type=TaskType.EVENT_RATE,
            start_time=self.minutes[0],
            end_time=self.minutes[-1],
            train_ratio=0.67,
            val_ratio=0.0,
            resample_freq=None,
        )
        dataset, _meta = build_event_rate_dataset(self.sync_unit, cfg)

        self.assertEqual(len(dataset.timestamps), 3)
        self.assertEqual(dataset.events_per_minute, [1.0, 2.0, 3.0])

    def test_resample_invalid_freq_raises(self) -> None:
        """An invalid frequency string should raise ValueError."""
        cfg = DatasetConfig(
            task_type=TaskType.EVENT_RATE,
            start_time=self.minutes[0],
            end_time=self.minutes[-1],
            train_ratio=0.67,
            val_ratio=0.0,
            resample_freq="not_a_freq",
        )
        with self.assertRaises(ValueError):
            build_event_rate_dataset(self.sync_unit, cfg)


class TestTimeToEventDatasetBuilder(unittest.TestCase):
    """Tests for the TIME_TO_EVENT dataset builder, especially min_delta filtering."""

    def setUp(self) -> None:
        self.sa = StorageAccess()
        self.sync_unit = SyncUnit("test_org", "test_system", "test_project")
        base_folder = f"{self.sync_unit.get_path_for_sync_unit()}/{global_vars.DATA_PREPROCESSING_APP_FOLDER_NAME}"
        self.sa.create_folder(base_folder)

        initial_folder = f"{self.sync_unit.get_path_for_sync_unit()}/{global_vars.INITIAL_DATA_FOLDER_NAME}"
        self.sa.create_folder(initial_folder)
        initial_rel_path = f"{initial_folder}/{global_vars.INITIAL_DATA_FILE_NAME_FORMAT.format(int=1)}"
        self.sa.write_json_to_storage(initial_rel_path, {"dummy": True})

        # Synthetic TTE data: already preprocessed (minute-floored, deduplicated,
        # integer-minute deltas). First row has NaN delta (dropped on load).
        start = datetime(2024, 1, 1, 0, 0, 0)
        self.timestamps = [
            start,
            start + timedelta(minutes=2),
            start + timedelta(minutes=5),
            start + timedelta(minutes=8),
            start + timedelta(minutes=10),
            start + timedelta(minutes=15),
            start + timedelta(minutes=20),
            start + timedelta(minutes=30),
            start + timedelta(minutes=33),
            start + timedelta(minutes=40),
        ]
        # Integer-minute deltas (first is NaN → dropped by dataset_builder)
        self.deltas = [None, 2, 3, 3, 2, 5, 5, 10, 3, 7]
        df = pd.DataFrame({
            "id": [f"evt_{i}" for i in range(10)],
            "events": self.timestamps,
            "events_delta": pd.array(self.deltas, dtype="Int64"),
        })

        tte_rel_path = f"{base_folder}/{global_vars.MERGED_TIME_DELTA_FILE_NAME_FORMAT.format(int=1)}"
        self.sa.save_dataframe_as_picklefile(df, tte_rel_path)

    def tearDown(self) -> None:
        test_data_path = self.sa.create_full_path_string(
            self.sync_unit.get_path_for_sync_unit()
        )
        for subfolder in ["modeling", "initial_data", global_vars.DATA_PREPROCESSING_APP_FOLDER_NAME]:
            path = os.path.join(test_data_path, subfolder)
            if os.path.exists(path):
                shutil.rmtree(path)

    def test_build_tte_no_filter(self) -> None:
        """Without min_delta, all rows (after NaN drop) are included."""
        cfg = DatasetConfig(
            task_type=TaskType.TIME_TO_EVENT,
            start_time=self.timestamps[0],
            end_time=self.timestamps[-1],
            train_ratio=0.7,
            val_ratio=0.15,
        )
        dataset, _meta = build_time_to_event_dataset(self.sync_unit, cfg, self.sa)
        # 10 rows minus 1 NaN = 9
        self.assertEqual(len(dataset.timestamps), 9)
        self.assertEqual(
            dataset.minutes_to_next_event,
            [2, 3, 3, 2, 5, 5, 10, 3, 7],
        )

    def test_build_tte_min_delta_filter(self) -> None:
        """min_delta in filters removes rows with time_delta < threshold (minutes)."""
        cfg = DatasetConfig(
            task_type=TaskType.TIME_TO_EVENT,
            start_time=self.timestamps[0],
            end_time=self.timestamps[-1],
            train_ratio=0.7,
            val_ratio=0.15,
            filters={"min_delta": 5},
        )
        dataset, _meta = build_time_to_event_dataset(self.sync_unit, cfg, self.sa)
        # Only deltas >= 5: [5, 5, 10, 7] = 4 rows
        self.assertEqual(len(dataset.timestamps), 4)
        self.assertEqual(dataset.minutes_to_next_event, [5, 5, 10, 7])

    def test_build_tte_min_delta_zero_has_no_effect(self) -> None:
        """min_delta=0 should not filter anything."""
        cfg = DatasetConfig(
            task_type=TaskType.TIME_TO_EVENT,
            start_time=self.timestamps[0],
            end_time=self.timestamps[-1],
            train_ratio=0.7,
            val_ratio=0.15,
            filters={"min_delta": 0},
        )
        dataset, _meta = build_time_to_event_dataset(self.sync_unit, cfg, self.sa)
        self.assertEqual(len(dataset.timestamps), 9)

    def test_build_tte_min_delta_metadata_tracked(self) -> None:
        """Snapshot metadata should record filter statistics."""
        cfg = DatasetConfig(
            task_type=TaskType.TIME_TO_EVENT,
            start_time=self.timestamps[0],
            end_time=self.timestamps[-1],
            train_ratio=0.7,
            val_ratio=0.15,
            filters={"min_delta": 5},
        )
        _dataset, meta = build_time_to_event_dataset(self.sync_unit, cfg, self.sa)
        self.assertEqual(meta.filters["min_delta_applied"], 5)
        self.assertEqual(meta.filters["n_before_min_delta_filter"], 9)
        self.assertEqual(meta.filters["n_after_min_delta_filter"], 4)

    def test_build_tte_with_resample(self) -> None:
        """resample_freq converts irregular events to a regular 1-minute grid."""
        cfg = DatasetConfig(
            task_type=TaskType.TIME_TO_EVENT,
            start_time=self.timestamps[0],
            end_time=self.timestamps[-1],
            train_ratio=0.7,
            val_ratio=0.15,
            resample_freq="1min",
        )
        dataset, meta = build_time_to_event_dataset(self.sync_unit, cfg, self.sa)
        # Regular grid: every minute from 00:00 to 00:40 = 41 points
        self.assertGreater(len(dataset.timestamps), 9)
        # elapsed_since_last_event should be automatically added
        self.assertIn("elapsed_since_last_event", dataset.covariates)
        # All values should be non-negative integers
        for v in dataset.minutes_to_next_event:
            self.assertGreaterEqual(v, 0)
        self.assertEqual(meta.resample_freq, "1min")


if __name__ == "__main__":
    unittest.main()
