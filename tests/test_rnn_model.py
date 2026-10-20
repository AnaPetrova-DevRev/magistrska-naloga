"""Tests for the RNN/LSTM/GRU forecast model.

Heavy tests are gated behind RUN_HEAVY_TESTS=1.  Within each class,
models are trained once in setUpClass and shared across test methods.
"""

import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np

from modeling.core_types import EventRateDataset, TaskType, TimeToEventDataset

_SKIP_HEAVY = os.environ.get("RUN_HEAVY_TESTS", "0") != "1"

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _make_event_rate_dataset(n=80, seed=42):
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(minutes=i) for i in range(n)]
    events = (5.0 + np.sin(np.arange(n) * 0.1) + rng.normal(0, 0.3, n)).tolist()
    return EventRateDataset(
        timestamps=timestamps,
        events_per_minute=[max(0, e) for e in events],
        covariates={},
        train_indices=list(range(0, 50)),
        val_indices=list(range(50, 65)),
        test_indices=list(range(65, 80)),
    )


def _make_tte_dataset(n=80, seed=42):
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(minutes=i * 10) for i in range(n)]
    deltas = rng.lognormal(2.0, 0.3, n).tolist()
    return TimeToEventDataset(
        timestamps=timestamps,
        minutes_to_next_event=deltas,
        covariates={},
        train_indices=list(range(0, 50)),
        val_indices=list(range(50, 65)),
        test_indices=list(range(65, 80)),
    )


def _make_dataset_with_covariates(n=80, seed=42):
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(minutes=i) for i in range(n)]
    events = (5.0 + rng.normal(0, 0.3, n)).tolist()
    covariates = {
        "hour_of_day": [float(ts.hour) for ts in timestamps],
        "day_of_week": [float(ts.weekday()) for ts in timestamps],
    }
    return EventRateDataset(
        timestamps=timestamps,
        events_per_minute=[max(0, e) for e in events],
        covariates=covariates,
        train_indices=list(range(0, 50)),
        val_indices=list(range(50, 65)),
        test_indices=list(range(65, 80)),
    )


_SMALL_CONFIG = {
    "cell_type": "LSTM",
    "hidden_size": 16,
    "num_layers": 1,
    "input_window": 10,
    "dropout": 0.0,
    "use_covariates": False,
}

_SMALL_TRAIN = {
    "max_epochs": 5,
    "batch_size": 16,
    "lr": 0.01,
    "early_stop_patience": 3,
}


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
@unittest.skipIf(_SKIP_HEAVY, "Heavy test — set RUN_HEAVY_TESTS=1")
class TestRNNEventRate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from modeling.models.rnn_model import RNNForecastModel
        cls.dataset = _make_event_rate_dataset()

        cls.lstm_model = RNNForecastModel(
            task_type=TaskType.EVENT_RATE,
            model_config=_SMALL_CONFIG,
            training_config=_SMALL_TRAIN,
        )
        cls.lstm_info = cls.lstm_model.fit(cls.dataset)

        cls.gru_model = RNNForecastModel(
            task_type=TaskType.EVENT_RATE,
            model_config={**_SMALL_CONFIG, "cell_type": "GRU"},
            training_config=_SMALL_TRAIN,
        )
        cls.gru_info = cls.gru_model.fit(cls.dataset)

        cls.rnn_model = RNNForecastModel(
            task_type=TaskType.EVENT_RATE,
            model_config={**_SMALL_CONFIG, "cell_type": "RNN"},
            training_config=_SMALL_TRAIN,
        )
        cls.rnn_info = cls.rnn_model.fit(cls.dataset)

    def test_fit_lstm(self):
        self.assertTrue(self.lstm_model.is_fitted)
        self.assertIn("epochs_run", self.lstm_info)
        self.assertIn("final_train_loss", self.lstm_info)

    def test_fit_gru(self):
        self.assertTrue(self.gru_model.is_fitted)

    def test_fit_rnn(self):
        self.assertTrue(self.rnn_model.is_fitted)

    def test_predict_pointwise(self):
        preds = self.lstm_model.predict_pointwise(
            self.dataset, self.dataset.val_indices,
        )
        self.assertEqual(len(preds), len(self.dataset.val_indices))

    def test_forecast(self):
        traj = self.lstm_model.forecast(self.dataset, horizon_minutes=5)
        self.assertEqual(len(traj.points), 5)

    def test_save_and_load(self):
        from modeling.models.rnn_model import RNNForecastModel
        tmpdir = tempfile.mkdtemp()
        try:
            self.lstm_model.save(tmpdir)
            loaded = RNNForecastModel.load(
                tmpdir, TaskType.EVENT_RATE, _SMALL_CONFIG,
            )
            self.assertTrue(loaded.is_fitted)
            orig = self.lstm_model.predict_pointwise(self.dataset, [50, 51])
            reloaded = loaded.predict_pointwise(self.dataset, [50, 51])
            np.testing.assert_allclose(orig, reloaded, atol=1e-5)
        finally:
            shutil.rmtree(tmpdir)


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
@unittest.skipIf(_SKIP_HEAVY, "Heavy test — set RUN_HEAVY_TESTS=1")
class TestRNNTimeToEvent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from modeling.models.rnn_model import RNNForecastModel
        cls.dataset = _make_tte_dataset()
        cls.model = RNNForecastModel(
            task_type=TaskType.TIME_TO_EVENT,
            model_config=_SMALL_CONFIG,
            training_config=_SMALL_TRAIN,
        )
        cls.model.fit(cls.dataset)

    def test_fit_and_predict_tte(self):
        self.assertTrue(self.model.is_fitted)
        preds = self.model.predict_pointwise(
            self.dataset, self.dataset.val_indices,
        )
        self.assertEqual(len(preds), len(self.dataset.val_indices))


@unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
@unittest.skipIf(_SKIP_HEAVY, "Heavy test — set RUN_HEAVY_TESTS=1")
class TestRNNWithCovariates(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from modeling.models.rnn_model import RNNForecastModel
        cls.dataset = _make_dataset_with_covariates()
        cls.model = RNNForecastModel(
            task_type=TaskType.EVENT_RATE,
            model_config={**_SMALL_CONFIG, "use_covariates": True},
            training_config=_SMALL_TRAIN,
        )
        cls.fit_info = cls.model.fit(cls.dataset)

    def test_fit_with_covariates(self):
        self.assertTrue(self.model.is_fitted)
        self.assertEqual(self.fit_info["n_features"], 3)  # target + 2 covariates
        preds = self.model.predict_pointwise(
            self.dataset, self.dataset.val_indices,
        )
        self.assertEqual(len(preds), len(self.dataset.val_indices))


if __name__ == "__main__":
    unittest.main()
