"""Tests for rl_training.prediction_provider module."""

import unittest
from datetime import UTC, datetime, timedelta
import json
import os
import tempfile

import numpy as np

from rl_training.prediction_provider import (
    ActionAwarePredictionProvider,
    MaskedPredictionProvider,
    NoPredictionProvider,
    PrecomputedPredictionProvider,
)


class _RecordingForecaster:
    metadata = {"model_type": "synthetic"}

    def __init__(self):
        self.calls = []

    def forecast(
        self, observed_timestamps, observed_values,
        synthetic_timestamps, synthetic_values, first_target, horizon,
    ):
        self.calls.append({
            "observed": list(observed_values),
            "synthetic": list(synthetic_values),
            "first_target": first_target,
        })
        base = sum(observed_values)
        if synthetic_values:
            base = synthetic_values[-1]
        return [base + i + 1 for i in range(horizon)]


class TestActionAwarePredictionProvider(unittest.TestCase):

    def setUp(self):
        self.start = datetime(2025, 1, 1, tzinfo=UTC)
        self.forecaster = _RecordingForecaster()
        self.provider = ActionAwarePredictionProvider(
            self.forecaster, trajectory_horizon_steps=2,
        )

    def test_bootstrap_forecast_begins_after_cutoff(self):
        history = [self.start, self.start + timedelta(minutes=1)]
        self.provider.bootstrap(history[-1], history, [2.0, 0.0])
        target = history[-1] + timedelta(minutes=1)
        self.assertEqual(self.provider.get_predictions(target, 2), [3.0, 4.0])
        self.assertEqual(self.forecaster.calls[0]["first_target"], target)

    def test_hidden_values_are_unavailable_until_sync(self):
        self.provider.bootstrap(self.start, [self.start], [0.0])
        target = self.start + timedelta(minutes=1)
        before = self.provider.get_predictions(target, 2)
        self.assertEqual(self.forecaster.calls[-1]["observed"], [0.0])
        self.provider.on_sync(target, [target], [7.0])
        after = self.provider.get_predictions(target + timedelta(minutes=1), 2)
        self.assertEqual(before, [1.0, 2.0])
        self.assertEqual(after, [8.0, 9.0])

    def test_trajectory_recursively_extends_without_unknown_zeros(self):
        self.provider.bootstrap(self.start, [self.start], [1.0])
        target = self.start + timedelta(minutes=1)
        values = self.provider.get_predictions(target, 5)
        self.assertEqual(values, [2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(self.forecaster.calls[1]["synthetic"], [2.0, 3.0])
        self.assertNotIn(0.0, self.forecaster.calls[1]["synthetic"])

    def test_clone_has_independent_state(self):
        clone = self.provider.clone()
        self.provider.bootstrap(self.start, [self.start], [1.0])
        clone.bootstrap(self.start, [self.start], [9.0])
        target = self.start + timedelta(minutes=1)
        self.assertEqual(self.provider.get_predictions(target, 1), [2.0])
        self.assertEqual(clone.get_predictions(target, 1), [10.0])


class TestPrecomputedPredictionProvider(unittest.TestCase):

    def setUp(self):
        start = datetime(2025, 1, 1, tzinfo=UTC)
        self.timestamps = [start + timedelta(minutes=i) for i in range(3)]
        self.values = np.array([
            [1.0, 1.1, 1.2],
            [2.0, 2.1, 2.2],
            [3.0, 3.1, 3.2],
        ])

    def test_basic_get(self):
        provider = PrecomputedPredictionProvider(self.timestamps, self.values)
        preds = provider.get_predictions(self.timestamps[0], n_steps=3)
        np.testing.assert_allclose(preds, [1.0, 1.1, 1.2])

    def test_timestamp_lookup_is_not_positional(self):
        provider = PrecomputedPredictionProvider(self.timestamps, self.values)
        preds = provider.get_predictions(self.timestamps[2], n_steps=2)
        np.testing.assert_allclose(preds, [3.0, 3.1])

    def test_horizon_overrun_raises(self):
        provider = PrecomputedPredictionProvider(self.timestamps, self.values)
        with self.assertRaises(ValueError):
            provider.get_predictions(self.timestamps[0], n_steps=4)

    def test_missing_timestamp_raises(self):
        provider = PrecomputedPredictionProvider(self.timestamps, self.values)
        with self.assertRaises(KeyError):
            provider.get_predictions(self.timestamps[-1] + timedelta(minutes=1), 1)

    def test_duplicate_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            PrecomputedPredictionProvider(
                [self.timestamps[0], self.timestamps[0]], self.values[:2]
            )

    def test_reset_is_noop(self):
        provider = PrecomputedPredictionProvider(self.timestamps, self.values)
        provider.reset()
        preds = provider.get_predictions(self.timestamps[0], n_steps=2)
        np.testing.assert_allclose(preds, [1.0, 1.1])

    def test_legacy_timestamp_cache_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cache.npz")
            np.savez_compressed(
                path,
                timestamps=np.asarray(self.timestamps, dtype="datetime64[us]"),
                predictions=self.values,
            )
            with open(os.path.join(tmpdir, "cache.json"), "w") as f:
                json.dump({"protocol": "causal_timestamp_v1"}, f)
            with self.assertRaisesRegex(ValueError, "not information-causal"):
                PrecomputedPredictionProvider.from_cache(path)


class TestNoPredictionProvider(unittest.TestCase):

    def test_always_returns_zeros(self):
        provider = NoPredictionProvider()
        preds = provider.get_predictions(datetime(2025, 1, 1), n_steps=5)
        self.assertEqual(preds, [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_different_steps(self):
        provider = NoPredictionProvider()
        self.assertEqual(
            provider.get_predictions(datetime(2025, 1, 2), n_steps=3),
            [0.0, 0.0, 0.0],
        )


class TestMaskedPredictionProvider(unittest.TestCase):

    def test_event_rate_returns_raw_zeros(self):
        provider = MaskedPredictionProvider("event_rate")
        self.assertEqual(
            provider.get_predictions(datetime(2025, 1, 1), n_steps=3),
            [0.0, 0.0, 0.0],
        )

    def test_tte_returns_normalized_zero_raw_value(self):
        provider = MaskedPredictionProvider("time_to_event")
        self.assertEqual(
            provider.get_predictions(datetime(2025, 1, 1), n_steps=2),
            [1440.0, 1440.0],
        )
        self.assertEqual(provider.metadata["mask_rule"], "constant_normalized_zero")
        self.assertEqual(provider.metadata["information_content"], "none")

    def test_invalid_task_type_rejected(self):
        with self.assertRaisesRegex(ValueError, "prediction_task_type"):
            MaskedPredictionProvider("invalid")

    def test_non_positive_horizon_rejected(self):
        provider = MaskedPredictionProvider("event_rate")
        with self.assertRaisesRegex(ValueError, "positive"):
            provider.get_predictions(datetime(2025, 1, 1), n_steps=0)


if __name__ == "__main__":
    unittest.main()
