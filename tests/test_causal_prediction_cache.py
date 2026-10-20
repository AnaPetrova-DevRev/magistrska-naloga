"""Tests for causal rolling-origin cache construction."""

from datetime import datetime, timedelta
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from modeling.core_types import TaskType
from model_serving.model_serving_app import ModelServingApp
from rl_training.causal_prediction_cache import CausalContextBuilder, save_causal_cache


class TestCausalContextBuilder(unittest.TestCase):
    """Verify that contexts use only information before each origin."""

    def setUp(self):
        self.start = datetime(2025, 1, 1)
        self.minutes = [self.start + timedelta(minutes=i) for i in range(12)]

    def test_event_rate_ignores_origin_and_future_values(self):
        origin = self.minutes[6]
        values = np.arange(12, dtype=float)
        changed = values.copy()
        changed[6:] = 9999.0
        first = CausalContextBuilder(
            TaskType.EVENT_RATE, self.minutes, values, "1min", 4
        ).build([origin])[1]
        second = CausalContextBuilder(
            TaskType.EVENT_RATE, self.minutes, changed, "1min", 4
        ).build([origin])[1]
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[0], [2.0, 3.0, 4.0, 5.0])

    def test_tte_future_event_is_censored(self):
        origin = self.minutes[6]
        builder = CausalContextBuilder(
            TaskType.TIME_TO_EVENT,
            [self.minutes[1], self.minutes[4], self.minutes[9]],
            None,
            "1min",
            4,
        )
        _, targets, covariates = builder.build([origin])
        np.testing.assert_array_equal(targets[0], [2.0, 1.0, 1440.0, 1440.0])
        np.testing.assert_array_equal(
            covariates["elapsed_since_last_event"][0], [1.0, 2.0, 0.0, 1.0]
        )

    def test_coarse_event_rate_uses_only_partial_current_bin(self):
        origin = self.minutes[7]
        builder = CausalContextBuilder(
            TaskType.EVENT_RATE, self.minutes, np.ones(12), "5min", 2
        )
        _, targets, _ = builder.build([origin])
        np.testing.assert_array_equal(targets[0], [5.0, 2.0])


class TestSaveCausalCache(unittest.TestCase):
    """Verify non-overwriting cache persistence."""

    def test_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cache.npz")
            timestamps = [datetime(2025, 1, 1)]
            save_causal_cache(path, timestamps, np.ones((1, 2)), {"task_type": "event_rate"})
            with self.assertRaises(FileExistsError):
                save_causal_cache(
                    path, timestamps, np.ones((1, 2)), {"task_type": "event_rate"}
                )


class TestServingHistoryCutoff(unittest.TestCase):
    """Verify live serving cannot build history at or beyond the origin."""

    @patch("model_serving.model_serving_app.build_event_rate_dataset")
    def test_reference_time_sets_strict_history_cutoff(self, build_dataset):
        expected_dataset = MagicMock()
        build_dataset.return_value = (expected_dataset, {})
        origin = datetime(2025, 1, 2, 12, 0)
        app = ModelServingApp(MagicMock())

        result = app._build_history_dataset(
            MagicMock(),
            TaskType.EVENT_RATE,
            reference_time=origin,
            resample_freq="15min",
        )

        self.assertIs(result, expected_dataset)
        config = build_dataset.call_args.args[1]
        self.assertEqual(config.end_time, origin - timedelta(minutes=1))
        self.assertEqual(config.resample_freq, "15min")


if __name__ == "__main__":
    unittest.main()