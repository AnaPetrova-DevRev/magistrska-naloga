"""Causal covariate context and persistence for CPU recurrent forecasts."""

from datetime import datetime, timedelta
import tempfile
import unittest

import numpy as np
import torch

from modeling.core_types import EventRateDataset, TaskType, TimeToEventDataset
from modeling.models.rnn_gpu_model import RNNGPUForecastModel
from modeling.models.rnn_model import RNNForecastModel


class TestRNNForecastCovariates(unittest.TestCase):
    def _dataset(self, task, n=4):
        timestamps = [
            datetime(2024, 1, 1, 0, 55) + timedelta(minutes=i)
            for i in range(n)
        ]
        common = dict(
            timestamps=timestamps,
            covariates={
                "observed_status": [float(10 * (i + 1)) for i in range(n)],
                "minute_of_hour": [
                    float(timestamp.minute) for timestamp in timestamps
                ],
            },
            train_indices=list(range(n)), val_indices=[], test_indices=[],
        )
        values = [float(3 + 2 * i) for i in range(n)]
        if task == TaskType.EVENT_RATE:
            return EventRateDataset(events_per_minute=values, **common)
        return TimeToEventDataset(minutes_to_next_event=values, **common)

    def test_forecast_rolls_covariate_history_without_future_observations(self):
        class CaptureCore(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.windows = []

            def forward(self, values):
                self.windows.append(values.detach().cpu().numpy()[0])
                return values[:, -1, 0] + 1

        for model_class in (RNNForecastModel, RNNGPUForecastModel):
            for task in (TaskType.EVENT_RATE, TaskType.TIME_TO_EVENT):
                with self.subTest(adapter=model_class.MODEL_TYPE, task=task):
                    dataset = self._dataset(task)
                    model = model_class(
                        task,
                        {"input_window": 3, "use_covariates": True, "device": "cpu"},
                        {},
                    )
                    core = CaptureCore()
                    model._model = core
                    model._target_mean = 1.0
                    model._target_std = 2.0
                    model._n_features = 3
                    model._is_fitted = True

                    forecast = model.forecast(dataset, horizon_minutes=3)

                    np.testing.assert_allclose(
                        core.windows[0], [[2, 56, 20], [3, 57, 30], [4, 58, 40]],
                    )
                    np.testing.assert_allclose(core.windows[1][-1], [5, 59, 40])
                    np.testing.assert_allclose(core.windows[2][-1], [6, 0, 40])
                    np.testing.assert_allclose(
                        [point.expected_events for point in forecast.points],
                        [11, 13, 15],
                    )
                    self.assertEqual(
                        forecast.points[0].timestamp, datetime(2024, 1, 1, 0, 59),
                    )

    def test_small_covariate_models_forecast_after_save_load(self):
        for model_class in (RNNForecastModel, RNNGPUForecastModel):
            for task in (TaskType.EVENT_RATE, TaskType.TIME_TO_EVENT):
                with self.subTest(adapter=model_class.MODEL_TYPE, task=task):
                    torch.manual_seed(42)
                    dataset = self._dataset(task, n=16)
                    config = {
                        "input_window": 3, "use_covariates": True,
                        "cell_type": "GRU", "hidden_size": 4, "num_layers": 1,
                        "dropout": 0.0, "device": "cpu",
                    }
                    model = model_class(
                        task, config, {"max_epochs": 1, "batch_size": 16},
                    )
                    model.fit(dataset)
                    original = model.forecast(dataset, horizon_minutes=3)
                    self.assertTrue(all(
                        np.isfinite(point.expected_events)
                        for point in original.points
                    ))
                    with tempfile.TemporaryDirectory() as temporary:
                        model.save(temporary)
                        loaded = model_class.load(temporary, task, config)
                        roundtrip = loaded.forecast(dataset, horizon_minutes=3)
                    np.testing.assert_allclose(
                        [point.expected_events for point in original.points],
                        [point.expected_events for point in roundtrip.points],
                    )


if __name__ == "__main__":
    unittest.main()
