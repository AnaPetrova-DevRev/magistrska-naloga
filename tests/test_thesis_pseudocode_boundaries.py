"""Small behavioral fixtures for the implementation described in Appendix A.

These characterize current behavior, including limitations, not thesis wording.
No production datasets, model downloads or experiment reruns are used.
"""

# These tests deliberately exercise the helpers documented in the appendix.
# pylint: disable=protected-access

from datetime import datetime, timedelta
from types import SimpleNamespace
import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

import app_flow_definitions as flow
import global_vars
from app_actions_constants import CREATE_MERGED_DATASET
from data_cleanup.parse_json import parse_json_file
from data_preprocessing import invoke_data_preprocesser as invocation
from data_preprocessing.calculate_time_delta import calculate_merged_time_delta
from data_preprocessing.merge_columns import calculate_events_per_minute, calculate_merge
from file_metadata import TrainingReadinessStatus
from file_stats import Stats
from modeling.base_model import BaseModel
from modeling.core_types import DatasetConfig, TaskType
from modeling.dataset_builder import (
    _resample_time_to_event_data,
    build_event_rate_dataset, build_time_to_event_dataset,
    compute_time_series_cv_splits,
)
from orchestrator_message import OrchestratorMessage
from sync_policy_evaluation.policy_loader import (
    AlwaysSyncPolicy, NeverSyncPolicy, PeriodicSyncPolicy, ThresholdPolicy,
)
from sync_unit import SyncUnit


START = datetime(2025, 1, 1)
UNIT = SyncUnit("test_org", "test_system", "test_project")


class TestTraceBoundaries(unittest.TestCase):
    """Distinguish exact event multiplicity from active-minute presence."""

    def test_exact_dedup_precedes_flooring_and_tte_discards_multiplicity(self):
        """Two distinct timestamps within one minute still contribute to ER."""
        t = START
        frame = pd.DataFrame({
            "id": ["a", "a", "a", "b"],
            "created_date": [t] * 4,
            "modified_date": [t + timedelta(seconds=s) for s in (10, 50, 50, 120)],
        })
        merged = calculate_merge(frame)
        self.assertEqual(len(merged), 5)
        er = calculate_events_per_minute(merged)
        self.assertEqual(er.events_per_minute.tolist(), [4, 1])
        sa = MagicMock()
        sa.read_picklefile_as_dataframe.return_value = merged
        with patch("data_preprocessing.calculate_time_delta.storage_access.StorageAccess",
                   return_value=sa):
            calculate_merged_time_delta("output", "input")
        tte = sa.save_dataframe_as_picklefile.call_args.args[0]
        self.assertEqual(tte.events.tolist(), [t, t + timedelta(minutes=2)])
        self.assertTrue(pd.isna(tte.events_delta.iloc[0]))
        self.assertEqual(tte.events_delta.iloc[1], 2)

    def test_parser_retains_three_fields_and_requires_separate_closing_line(self):
        """Only the supported multiline format emits a parsed record."""
        item = {"id": "a", "created_date": START.isoformat(),
                "modified_date": START.isoformat(), "unused": "discard"}
        sa = MagicMock()
        with patch("data_cleanup.parse_json.storage_access.StorageAccess", return_value=sa):
            sa.read_lines_from_storage.return_value = json.dumps(item, indent=2).splitlines()
            items, _ = parse_json_file("fixture")
            self.assertEqual(len(items), 1)
            self.assertEqual(set(vars(items[0])), {"id", "created_date", "modified_date"})
            sa.read_lines_from_storage.return_value = [json.dumps(item)]
            self.assertEqual(parse_json_file("fixture")[0], [])


class TestPersistenceBoundary(unittest.TestCase):
    """Characterize saved counters separately from in-memory progress."""

    def run_chunks(self, fail_second=False):
        """Resume two in-memory chunks, optionally interrupting the second."""
        sa = MagicMock()
        app_state = invocation.DataPreprocessingState()
        state = app_state.create_merged_dataset_state
        state.last_processed_chunk_number = 1
        inputs = [flow.define_paths_for_input_output_files(
            CREATE_MERGED_DATASET, UNIT, k).input_filepath for k in (2, 3)]
        sa.check_if_file_exists.side_effect = lambda path: path in inputs
        seen = []

        def write_chunk(_output, source):
            sa.save_state.assert_not_called()
            seen.append(source)
            if fail_second and len(seen) == 2:
                raise OSError("fixture interruption")

        with patch.object(invocation.storage_access, "StorageAccess", return_value=sa), \
                patch.object(invocation, "get_stats_and_state",
                             return_value=(Stats(), app_state)), \
                patch.object(invocation.merge_columns, "merge_columns", side_effect=write_chunk):
            message = OrchestratorMessage(UNIT, CREATE_MERGED_DATASET)
            if fail_second:
                with self.assertRaisesRegex(OSError, "fixture interruption"):
                    invocation.receive_message(message)
            else:
                self.assertTrue(invocation.receive_message(message).is_done)
        self.assertEqual(seen, inputs)
        return sa, state

    def test_resume_from_saved_counter_and_persist_only_after_loop(self):
        """The receiver persists the final counter only after normal return."""
        sa, state = self.run_chunks()
        sa.save_state.assert_called_once()
        saved = sa.save_state.call_args.args[0]["create_merged_dataset_state"]
        self.assertEqual(saved["last_processed_chunk_number"], 3)
        self.assertEqual(saved["total_processed"], 2)
        self.assertTrue(state.is_done)

    def test_interrupted_loop_does_not_persist_in_memory_progress(self):
        """An interrupted loop leaves its latest counter unsaved."""
        sa, state = self.run_chunks(fail_second=True)
        self.assertEqual(state.last_processed_chunk_number, 2)
        sa.save_state.assert_not_called()


class TestReadinessBoundary(unittest.TestCase):
    """Verify readiness categories independently of storage and statistics."""

    def validate(self, initial, complete, stats_warning=False):
        """Validate a synthetic inventory of source and derived chunks."""
        formats = [global_vars.MERGED_DATASET_FILE_NAME_FORMAT,
                   global_vars.MERGED_TIME_DELTA_FILE_NAME_FORMAT,
                   global_vars.MERGED_EVENTS_PER_MINUTE_FILE_NAME_FORMAT]
        sa = MagicMock()
        sa.list_files_in_folder.return_value = [f"work_{k}.json" for k in initial]

        def paths(name, _unit):
            return {fmt: "present" if flow.extract_chunk_number(name) in complete else None
                    for fmt in formats}

        stats = {formats[0]: SimpleNamespace(issues=["fixture warning"])} if stats_warning else {}
        with patch.object(flow, "StorageAccess", return_value=sa), \
                patch.object(flow, "get_pipeline_file_paths", side_effect=paths), \
                patch.object(flow, "validate_pipeline_file_stats", return_value=stats):
            return flow.validate_training_readiness(UNIT)

    def test_intermediate_outputs_error_but_trailing_outputs_warning(self):
        """A missing interior output is more severe than unfinished tail work."""
        gap = self.validate([1, 2, 3], {1, 3})
        self.assertEqual(gap.status, TrainingReadinessStatus.ERROR)
        self.assertEqual(gap.gap_chunks, [2])
        tail = self.validate([1, 2, 3], {1, 2})
        self.assertEqual(tail.status, TrainingReadinessStatus.WARNING)
        self.assertEqual(tail.trailing_chunks, [3])

    def test_absent_source_number_is_not_an_output_gap(self):
        """Validation enumerates present sources, not all possible numbers."""
        result = self.validate([1, 3], {1, 3})
        self.assertEqual(result.status, TrainingReadinessStatus.READY)
        self.assertEqual(result.gap_chunks, [])

    def test_stats_warning_does_not_change_ready_status(self):
        """Diagnostic warnings alone do not change the readiness category."""
        result = self.validate([1], {1}, stats_warning=True)
        self.assertEqual(result.status, TrainingReadinessStatus.READY)
        self.assertTrue(result.warnings)


class TestDatasetBoundary(unittest.TestCase):
    """Test filtering, target construction and temporal split boundaries."""

    def build(self, task, frame, **kwargs):
        """Build a tiny dataset without reading or writing any real artifacts."""
        sa = MagicMock()
        sa.list_files_in_folder.return_value = ["fixture"]
        sa.read_picklefile_as_dataframe.return_value = frame
        config = DatasetConfig(task, START, START + timedelta(minutes=10), **kwargs)
        builder = (build_event_rate_dataset if task == TaskType.EVENT_RATE
                   else build_time_to_event_dataset)
        with patch("modeling.dataset_builder._save_snapshot_metadata", return_value="fixture"):
            return builder(UNIT, config, sa)[0]

    def test_er_inclusive_filter_then_partial_bins_and_zero_fill(self):
        """Inclusive filtering precedes ER resampling and zero filling."""
        frame = pd.DataFrame({
            "minute": [START + timedelta(minutes=k) for k in (-1, 2, 10, 11)],
            "events_per_minute": [100, 3, 7, 100],
        })
        result = self.build(TaskType.EVENT_RATE, frame, resample_freq="5min")
        self.assertEqual(result.events_per_minute, [3, 0, 7])
        self.assertEqual(result.timestamps, [START + timedelta(minutes=k) for k in (0, 5, 10)])

    def test_tte_drops_initial_delta_then_uses_strict_next_and_sentinel(self):
        """Grid targets use retained events and are not re-censored at splits."""
        frame = pd.DataFrame({
            "events": [START + timedelta(minutes=k) for k in (0, 2, 5)],
            "events_delta": [np.nan, 2, 3],
        })
        irregular = self.build(TaskType.TIME_TO_EVENT, frame)
        self.assertEqual(irregular.minutes_to_next_event, [2, 3])
        regular = self.build(TaskType.TIME_TO_EVENT, frame, resample_freq="1min",
                             train_ratio=0.5, val_ratio=0.25)
        self.assertEqual(regular.timestamps[0], START + timedelta(minutes=2))
        self.assertEqual(regular.minutes_to_next_event, [3, 2, 1, 1440])
        self.assertEqual(regular.covariates["elapsed_since_last_event"], [0, 1, 2, 0])
        self.assertEqual(regular.train_indices, [0, 1])
        # The final training label still reaches the event in the test split.
        self.assertEqual(regular.minutes_to_next_event[1], 2)

    def test_tte_coarse_grid_extends_to_ceil_and_does_not_cap_known_deltas(self):
        """Known waits may exceed the sentinel used for missing next events."""
        frame = pd.DataFrame({"timestamp": [START + timedelta(minutes=k) for k in (2, 2003)]})
        result = _resample_time_to_event_data(frame, "5min")
        self.assertEqual(result.minute.iloc[0], START)
        self.assertEqual(result.minute.iloc[-1], START + timedelta(minutes=2005))
        self.assertEqual(result.minutes_to_next_event.iloc[1], 1998)
        self.assertEqual(result.minutes_to_next_event.iloc[-1], 1440)

    def test_expanding_cv_assigns_remainder_to_last_validation(self):
        """The final expanding-window validation fold absorbs the remainder."""
        folds = compute_time_series_cv_splits(11, 2)
        self.assertEqual(folds, [(list(range(3)), list(range(3, 6))),
                                 (list(range(6)), list(range(6, 11)))])
        for n, count in ((2, 2), (10, 1)):
            with self.subTest(n=n, count=count), self.assertRaises(ValueError):
                compute_time_series_cv_splits(n, count)


class TestInterpolationBoundary(unittest.TestCase):
    """Check minute interpolation and its intentionally limited freq parser."""

    def test_midpoint_interpolation_and_constant_edges_er_vs_tte(self):
        """Only ER totals are divided by the native step before interpolation."""
        values = np.array([8.0, 24.0])
        er = BaseModel._interpolate_to_minutes(None, values, 4, 10, True)
        tte = BaseModel._interpolate_to_minutes(None, values, 4, 10, False)
        np.testing.assert_allclose(er, [2, 2, 2.5, 3.5, 4.5, 5.5, 6, 6, 6, 6])
        np.testing.assert_allclose(tte, er * 4)
        np.testing.assert_array_equal(values, [8, 24])

    def test_minute_resolution_truncates_or_pads_last_value(self):
        """Minute-native predictions are copied, truncated or edge-padded."""
        values = np.array([2.0, 7.0])
        short = BaseModel._interpolate_to_minutes(None, values, 1, 1)
        np.testing.assert_array_equal(short, [2])
        self.assertFalse(np.shares_memory(short, values))
        np.testing.assert_array_equal(
            BaseModel._interpolate_to_minutes(None, values, 1, 4), [2, 7, 7, 7])

    def test_frequency_uses_first_gap_and_unknown_string_defaults_to_minute(self):
        """Frequency inference does not validate the remaining timestamps."""
        times = [START, START + timedelta(minutes=90), START + timedelta(days=10)]
        self.assertEqual(BaseModel._infer_freq(times), "2h")
        self.assertEqual(BaseModel._freq_to_minutes("2h"), 120)
        self.assertEqual(BaseModel._freq_to_minutes("1D"), 1)
        self.assertEqual(BaseModel._freq_to_minutes("D"), 1440)


class TestBaselineBoundary(unittest.TestCase):
    """Verify baseline action timing, strict thresholds and reset behavior."""

    def test_periodic_counter_increments_before_comparison_and_reset(self):
        """The first synchronization occurs after a full interval of calls."""
        policy = PeriodicSyncPolicy(3)
        self.assertEqual([policy.predict(np.array([])) for _ in range(7)], [0, 0, 1, 0, 0, 1, 0])
        policy.reset()
        self.assertEqual([policy.predict(np.array([])) for _ in range(3)], [0, 0, 1])

    def test_threshold_strict_single_feature_and_unavailable_fallback(self):
        """Equality skips, and absent prediction features also skip."""
        policy = ThresholdPolicy(threshold=0.5, prediction_start_index=1)
        for values, expected in (([1, 0.5, 1], 0), ([0, 0.51, 0], 1), ([1], 0)):
            with self.subTest(values=values):
                policy.reset()
                self.assertEqual(policy.predict(np.array(values)), expected)

    def test_stateless_baselines_keep_action_after_reset(self):
        """Reset cannot change the constant action of either trivial baseline."""
        for policy, expected in ((AlwaysSyncPolicy(), 1), (NeverSyncPolicy(), 0)):
            policy.reset()
            self.assertEqual(policy.predict(np.array([])), expected)


if __name__ == "__main__":
    unittest.main()
