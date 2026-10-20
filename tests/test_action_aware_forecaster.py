"""Tests for explicit-history action-aware model forecasting."""

import unittest
from datetime import datetime, timedelta

from modeling.core_types import PredictionPoint, PredictionTrajectory, TaskType
from rl_training.action_aware_forecaster import ExplicitHistoryForecaster


class _FakeModel:
    """Capture forecast context and emit its requested minute index."""

    def __init__(self, task_type, model_config, freq=None):
        self.task_type = task_type
        self.model_config = model_config
        self._freq = freq
        self._covariate_names = []
        self.dataset = None

    def forecast(self, dataset, horizon_minutes, reference_time):
        self.dataset = dataset
        points = [
            PredictionPoint(reference_time + timedelta(minutes=i), float(i))
            for i in range(horizon_minutes)
        ]
        return PredictionTrajectory(
            "synthetic", self.task_type, horizon_minutes, points, "fake",
        )


class TestExplicitHistoryForecaster(unittest.TestCase):
    """Verify that only complete target history reaches fitted models."""

    def test_event_rate_drops_partial_native_bin(self):
        model = _FakeModel(TaskType.EVENT_RATE, {"input_size": 2}, "15min")
        adapter = ExplicitHistoryForecaster(
            model, TaskType.EVENT_RATE, "fake", {"model_type": "fake"},
        )
        start = datetime(2025, 1, 1)
        timestamps = [start + timedelta(minutes=i) for i in range(37)]
        values = [1.0] * len(timestamps)
        predictions = adapter.forecast(
            timestamps, values, [], [], start + timedelta(minutes=37), 3,
        )
        self.assertEqual(len(model.dataset.timestamps), 2)
        self.assertEqual(list(model.dataset.events_per_minute), [15.0, 15.0])
        self.assertEqual(predictions, [7.0, 8.0, 9.0])

    def test_tte_context_ends_before_latest_revealed_event(self):
        model = _FakeModel(TaskType.TIME_TO_EVENT, {"input_window": 2})
        adapter = ExplicitHistoryForecaster(
            model, TaskType.TIME_TO_EVENT, "fake", {"model_type": "fake"},
        )
        start = datetime(2025, 1, 1)
        timestamps = [start + timedelta(minutes=i) for i in range(16)]
        event_minutes = {0, 5, 12}
        values = [1.0 if i in event_minutes else 0.0 for i in range(16)]
        predictions = adapter.forecast(
            timestamps, values, [], [], start + timedelta(minutes=16), 2,
        )
        self.assertEqual(model.dataset.timestamps[-1], start + timedelta(minutes=11))
        self.assertEqual(model.dataset.minutes_to_next_event[-1], 1.0)
        self.assertNotIn(1440.0, model.dataset.minutes_to_next_event)
        self.assertEqual(predictions, [4.0, 5.0])

    def test_empty_tte_sync_does_not_regenerate(self):
        model = _FakeModel(TaskType.TIME_TO_EVENT, {"input_window": 2})
        adapter = ExplicitHistoryForecaster(
            model, TaskType.TIME_TO_EVENT, "fake", {"model_type": "fake"},
        )
        self.assertFalse(adapter.should_regenerate([0.0, 0.0]))
        self.assertTrue(adapter.should_regenerate([0.0, 1.0]))


if __name__ == "__main__":
    unittest.main()