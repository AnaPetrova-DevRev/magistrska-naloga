"""Tests for foundation and neural forecast models.

Each model's tests are guarded by @skipUnless so they only run if the
corresponding package is installed.  Heavy tests (neural network training)
are additionally gated behind RUN_HEAVY_TESTS=1 to keep routine test runs
fast and memory-light.

Within each class, models are trained once in setUpClass and shared across
test methods — reducing redundant training from ~25 to ~17 instances.
"""

import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np

from modeling.core_types import EventRateDataset, TaskType, TimeToEventDataset

# Skip heavy neural model tests unless explicitly enabled
_SKIP_HEAVY = os.environ.get("RUN_HEAVY_TESTS", "0") != "1"

# Probe for optional packages
try:
    from chronos import ChronosPipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False

try:
    from neuralforecast import NeuralForecast
    NEURALFORECAST_AVAILABLE = True
except ImportError:
    NEURALFORECAST_AVAILABLE = False


def _make_event_rate_dataset(n=120, seed=42):
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(hours=i) for i in range(n)]
    values = (5.0 + np.sin(np.arange(n) * 0.2) + rng.normal(0, 0.5, n)).tolist()
    return EventRateDataset(
        timestamps=timestamps,
        events_per_minute=[max(0, v) for v in values],
        covariates={},
        train_indices=list(range(0, 80)),
        val_indices=list(range(80, 100)),
        test_indices=list(range(100, 120)),
    )


def _make_tte_dataset(n=120, seed=42):
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(hours=i) for i in range(n)]
    deltas = rng.lognormal(2.0, 0.5, n).tolist()
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    return TimeToEventDataset(
        timestamps=timestamps,
        minutes_to_next_event=deltas,
        covariates={},
        train_indices=list(range(0, n_train)),
        val_indices=list(range(n_train, n_train + n_val)),
        test_indices=list(range(n_train + n_val, n)),
    )


def _make_er_dataset_with_covariates(n=120, seed=42):
    base = _make_event_rate_dataset(n=n, seed=seed)
    return EventRateDataset(
        timestamps=base.timestamps,
        events_per_minute=base.events_per_minute,
        covariates={
            "hour_of_day": [float(t.hour) for t in base.timestamps],
            "day_of_week": [float(t.weekday()) for t in base.timestamps],
        },
        train_indices=base.train_indices,
        val_indices=base.val_indices,
        test_indices=base.test_indices,
    )


def _make_tte_dataset_with_covariates(n=120, seed=42):
    base = _make_tte_dataset(n=n, seed=seed)
    return TimeToEventDataset(
        timestamps=base.timestamps,
        minutes_to_next_event=base.minutes_to_next_event,
        covariates={
            "hour_of_day": [float(t.hour) for t in base.timestamps],
            "day_of_week": [float(t.weekday()) for t in base.timestamps],
            "is_weekend": [float(t.weekday() >= 5) for t in base.timestamps],
        },
        train_indices=base.train_indices,
        val_indices=base.val_indices,
        test_indices=base.test_indices,
    )


class TestNHiTSFastInference(unittest.TestCase):
    """Exercise the lightweight single-series NeuralForecast path."""

    def test_fast_path_builds_expected_history_and_future_window(self):
        import pandas as pd
        import torch
        from types import SimpleNamespace
        from modeling.models.nhits_model import NHiTSForecastModel

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input_size = 3
                self.hist_exog_list = ["hour_of_day"]
                self.futr_exog_list = ["hour_of_day"]
                self.loss = SimpleNamespace(output_names=[""])
                self._test_size = 0
                self.seen_temporal = None

            @property
            def device(self):
                return torch.device("cpu")

            def get_test_size(self):
                return self._test_size

            def set_test_size(self, value):
                self._test_size = value

            def predict_step(self, batch, batch_idx):
                self.seen_temporal = batch["temporal"].detach().cpu().numpy()
                return torch.tensor([[[11.0], [12.0]]])

        native_model = FakeModel()
        training_dataset = SimpleNamespace(
            temporal_cols=pd.Index([
                "y", "hour_of_day", "available_mask",
            ]),
            scalers_=None,
            static=None,
        )
        model = NHiTSForecastModel(TaskType.EVENT_RATE, {}, {})
        model._nf = SimpleNamespace(
            dataset=training_dataset, models=[native_model], h=2,
        )

        result = model._predict_native_fast(
            np.asarray([1.0, 2.0, 3.0, 4.0]),
            {"hour_of_day": np.asarray([10.0, 20.0, 30.0, 40.0])},
            {"hour_of_day": np.asarray([50.0, 60.0])},
        )

        np.testing.assert_allclose(result, [11.0, 12.0])
        temporal = native_model.seen_temporal[0]
        np.testing.assert_allclose(temporal[0], [2, 3, 4, 0, 0])
        np.testing.assert_allclose(temporal[1], [20, 30, 40, 50, 60])
        np.testing.assert_allclose(temporal[2], [1, 1, 1, 1, 1])


# Chronos

@unittest.skipUnless(CHRONOS_AVAILABLE, "chronos-forecasting not installed")
@unittest.skipIf(_SKIP_HEAVY, "Heavy test — set RUN_HEAVY_TESTS=1")
class TestChronosModel(unittest.TestCase):

    _cfg = {"model_name": "amazon/chronos-t5-tiny", "num_samples": 5}

    @classmethod
    def setUpClass(cls):
        from modeling.models.chronos_model import ChronosForecastModel
        cls.er_dataset = _make_event_rate_dataset()
        cls.er_model = ChronosForecastModel(
            task_type=TaskType.EVENT_RATE,
            model_config=cls._cfg, training_config={},
        )
        cls.er_fit_info = cls.er_model.fit(cls.er_dataset)

        cls.tte_dataset = _make_tte_dataset()
        cls.tte_model = ChronosForecastModel(
            task_type=TaskType.TIME_TO_EVENT,
            model_config=cls._cfg, training_config={},
        )
        cls.tte_model.fit(cls.tte_dataset)

    def test_fit_event_rate(self):
        self.assertTrue(self.er_model.is_fitted)
        self.assertTrue(self.er_fit_info.get("zero_shot"))
        self.assertIn("warnings", self.er_fit_info)

    def test_predict_pointwise(self):
        preds = self.er_model.predict_pointwise(
            self.er_dataset, self.er_dataset.val_indices,
        )
        self.assertEqual(len(preds), len(self.er_dataset.val_indices))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_forecast(self):
        traj = self.er_model.forecast(self.er_dataset, horizon_minutes=5)
        self.assertEqual(len(traj.points), 5)

    def test_save_and_load(self):
        from modeling.models.chronos_model import ChronosForecastModel
        tmpdir = tempfile.mkdtemp()
        try:
            self.er_model.save(tmpdir)
            loaded = ChronosForecastModel.load(
                tmpdir, TaskType.EVENT_RATE, self._cfg,
            )
            self.assertTrue(loaded.is_fitted)
        finally:
            shutil.rmtree(tmpdir)

    def test_time_to_event(self):
        self.assertTrue(self.tte_model.is_fitted)
        preds = self.tte_model.predict_pointwise(
            self.tte_dataset, self.tte_dataset.val_indices,
        )
        self.assertEqual(len(preds), len(self.tte_dataset.val_indices))


# TFT

@unittest.skipUnless(NEURALFORECAST_AVAILABLE, "neuralforecast not installed")
@unittest.skipIf(_SKIP_HEAVY, "Heavy test — set RUN_HEAVY_TESTS=1")
class TestTFTModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from modeling.models.tft_model import TFTForecastModel

        cls.dataset = _make_event_rate_dataset()
        cls.model = TFTForecastModel(
            task_type=TaskType.EVENT_RATE,
            model_config={"horizon": 10, "input_size": 20,
                          "hidden_size": 16, "use_covariates": False},
            training_config={"max_steps": 5, "batch_size": 16},
        )
        cls.fit_info = cls.model.fit(cls.dataset)

        cls.cov_dataset = _make_er_dataset_with_covariates()
        cls.cov_model = TFTForecastModel(
            task_type=TaskType.EVENT_RATE,
            model_config={"horizon": 10, "input_size": 20,
                          "hidden_size": 16, "use_covariates": True},
            training_config={"max_steps": 5, "batch_size": 16},
        )
        cls.cov_model.fit(cls.cov_dataset)

    def test_fit_and_predict(self):
        self.assertTrue(self.model.is_fitted)
        self.assertIn("warnings", self.fit_info)

    def test_forecast(self):
        traj = self.model.forecast(self.dataset, horizon_minutes=5)
        self.assertEqual(len(traj.points), 5)

    def test_predict_with_covariates_and_chunking(self):
        """Test TFT with covariates enabled and prediction requiring multiple chunks."""
        preds = self.cov_model.predict_pointwise(
            self.cov_dataset, self.cov_dataset.val_indices,
        )
        self.assertEqual(len(preds), len(self.cov_dataset.val_indices))
        self.assertTrue(np.all(np.isfinite(preds)))


# N-HiTS

@unittest.skipUnless(NEURALFORECAST_AVAILABLE, "neuralforecast not installed")
@unittest.skipIf(_SKIP_HEAVY, "Heavy test — set RUN_HEAVY_TESTS=1")
class TestNHiTSModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from modeling.models.nhits_model import NHiTSForecastModel

        cls.dataset = _make_event_rate_dataset()
        cls.model = NHiTSForecastModel(
            task_type=TaskType.EVENT_RATE,
            model_config={"horizon": 10, "input_size": 20,
                          "n_stacks": 2, "use_covariates": False},
            training_config={"max_steps": 5, "batch_size": 16},
        )
        cls.fit_info = cls.model.fit(cls.dataset)

        cls.er_cov_dataset = _make_er_dataset_with_covariates()
        cls.er_cov_model = NHiTSForecastModel(
            task_type=TaskType.EVENT_RATE,
            model_config={"horizon": 10, "input_size": 20,
                          "n_stacks": 2, "use_covariates": True},
            training_config={"max_steps": 5, "batch_size": 16},
        )
        cls.er_cov_model.fit(cls.er_cov_dataset)

        cls.tte_cov_dataset = _make_tte_dataset_with_covariates()
        cls.tte_cov_model = NHiTSForecastModel(
            task_type=TaskType.TIME_TO_EVENT,
            model_config={"horizon": 12, "input_size": 30,
                          "n_stacks": 3, "use_covariates": True},
            training_config={"max_steps": 5, "batch_size": 16},
        )
        cls.tte_cov_model.fit(cls.tte_cov_dataset)

    def test_fit_and_predict(self):
        self.assertTrue(self.model.is_fitted)
        self.assertIn("warnings", self.fit_info)

    def test_forecast(self):
        traj = self.model.forecast(self.dataset, horizon_minutes=5)
        self.assertEqual(len(traj.points), 5)

    def test_predict_with_covariates_and_chunking(self):
        """Test NHiTS with covariates enabled and prediction requiring multiple chunks."""
        preds = self.er_cov_model.predict_pointwise(
            self.er_cov_dataset, self.er_cov_dataset.val_indices,
        )
        self.assertEqual(len(preds), len(self.er_cov_dataset.val_indices))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_predict_tte_with_covariates(self):
        """Test NHiTS on TTE task with covariates — the exact scenario that failed in Round 3."""
        preds = self.tte_cov_model.predict_pointwise(
            self.tte_cov_dataset, self.tte_cov_dataset.val_indices,
        )
        self.assertEqual(len(preds), len(self.tte_cov_dataset.val_indices))
        self.assertTrue(np.all(np.isfinite(preds)))
        traj = self.tte_cov_model.forecast(self.tte_cov_dataset, horizon_minutes=5)
        self.assertEqual(len(traj.points), 5)


# Ground-truth context predict_pointwise tests

@unittest.skipUnless(NEURALFORECAST_AVAILABLE, "neuralforecast not installed")
@unittest.skipIf(_SKIP_HEAVY, "Heavy test — set RUN_HEAVY_TESTS=1")
class TestGroundTruthContextPrediction(unittest.TestCase):
    """Test that predict_pointwise uses ground-truth context (not autoregressive).

    The key properties:
    - Each chunk uses real data as context, not model predictions
    - Context window is fixed-size (input_size), not growing
    - Output length always matches requested indices
    - Predictions are finite and non-negative
    - Works with multi-chunk scenarios (n_needed > horizon)

    Models are lazily cached by configuration tuple so each unique
    (model_type, task, horizon, input_size, covariates) is trained once.
    """

    @classmethod
    def setUpClass(cls):
        cls._model_cache = {}
        cls._er_dataset = _make_event_rate_dataset()
        cls._tte_dataset = _make_tte_dataset()
        cls._er_cov_dataset = _make_er_dataset_with_covariates()

    @classmethod
    def _get_model(cls, model_type, dataset_key, task_type,
                   horizon, input_size, covariates):
        key = (model_type, dataset_key, task_type, horizon, input_size, covariates)
        if key not in cls._model_cache:
            dataset = {
                'er': cls._er_dataset,
                'tte': cls._tte_dataset,
                'er_cov': cls._er_cov_dataset,
            }[dataset_key]
            if model_type == 'nhits':
                from modeling.models.nhits_model import NHiTSForecastModel
                model = NHiTSForecastModel(
                    task_type=task_type,
                    model_config={"horizon": horizon, "input_size": input_size,
                                  "n_stacks": 2, "use_covariates": covariates},
                    training_config={"max_steps": 5, "batch_size": 16},
                )
            else:
                from modeling.models.tft_model import TFTForecastModel
                model = TFTForecastModel(
                    task_type=task_type,
                    model_config={"horizon": horizon, "input_size": input_size,
                                  "hidden_size": 16, "use_covariates": covariates},
                    training_config={"max_steps": 5, "batch_size": 16},
                )
            model.fit(dataset)
            cls._model_cache[key] = model
        return cls._model_cache[key]

    def test_nhits_output_length_matches_indices(self):
        """Basic: output length == len(indices)."""
        model = self._get_model('nhits', 'er', TaskType.EVENT_RATE, 10, 20, False)
        preds = model.predict_pointwise(self._er_dataset, self._er_dataset.val_indices)
        self.assertEqual(len(preds), len(self._er_dataset.val_indices))

    def test_tft_output_length_matches_indices(self):
        """Basic: output length == len(indices)."""
        model = self._get_model('tft', 'er', TaskType.EVENT_RATE, 10, 20, False)
        preds = model.predict_pointwise(self._er_dataset, self._er_dataset.val_indices)
        self.assertEqual(len(preds), len(self._er_dataset.val_indices))

    def test_nhits_multi_chunk_er(self):
        """Event rate: val has 20 indices with horizon=5 -> 4 chunks."""
        model = self._get_model('nhits', 'er', TaskType.EVENT_RATE, 5, 15, False)
        preds = model.predict_pointwise(self._er_dataset, self._er_dataset.val_indices)
        self.assertEqual(len(preds), len(self._er_dataset.val_indices))
        self.assertTrue(np.all(np.isfinite(preds)))
        self.assertTrue(np.all(preds >= 0))

    def test_tft_multi_chunk_er(self):
        """Event rate: val has 20 indices with horizon=5 -> 4 chunks."""
        model = self._get_model('tft', 'er', TaskType.EVENT_RATE, 5, 15, False)
        preds = model.predict_pointwise(self._er_dataset, self._er_dataset.val_indices)
        self.assertEqual(len(preds), len(self._er_dataset.val_indices))
        self.assertTrue(np.all(np.isfinite(preds)))
        self.assertTrue(np.all(preds >= 0))

    def test_nhits_multi_chunk_tte(self):
        """TTE: val has 18 indices with horizon=5 -> 4 chunks."""
        model = self._get_model('nhits', 'tte', TaskType.TIME_TO_EVENT, 5, 15, False)
        preds = model.predict_pointwise(self._tte_dataset, self._tte_dataset.val_indices)
        self.assertEqual(len(preds), len(self._tte_dataset.val_indices))
        self.assertTrue(np.all(np.isfinite(preds)))
        self.assertTrue(np.all(preds >= 0))

    def test_tft_multi_chunk_tte(self):
        """TTE: val has 18 indices with horizon=5 -> 4 chunks."""
        model = self._get_model('tft', 'tte', TaskType.TIME_TO_EVENT, 5, 15, False)
        preds = model.predict_pointwise(self._tte_dataset, self._tte_dataset.val_indices)
        self.assertEqual(len(preds), len(self._tte_dataset.val_indices))
        self.assertTrue(np.all(np.isfinite(preds)))
        self.assertTrue(np.all(preds >= 0))

    def test_nhits_with_covariates_multi_chunk(self):
        """NHiTS with covariates and multi-chunk prediction."""
        model = self._get_model('nhits', 'er_cov', TaskType.EVENT_RATE, 5, 15, True)
        preds = model.predict_pointwise(self._er_cov_dataset, self._er_cov_dataset.val_indices)
        self.assertEqual(len(preds), len(self._er_cov_dataset.val_indices))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_tft_with_covariates_multi_chunk(self):
        """TFT with covariates and multi-chunk prediction."""
        model = self._get_model('tft', 'er_cov', TaskType.EVENT_RATE, 5, 15, True)
        preds = model.predict_pointwise(self._er_cov_dataset, self._er_cov_dataset.val_indices)
        self.assertEqual(len(preds), len(self._er_cov_dataset.val_indices))
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_nhits_early_indices_small_context(self):
        """Prediction at indices near start of data (context < input_size)."""
        model = self._get_model('nhits', 'er', TaskType.EVENT_RATE, 10, 20, False)
        early_indices = list(range(5, 15))
        preds = model.predict_pointwise(self._er_dataset, early_indices)
        self.assertEqual(len(preds), 10)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_predictions_independent_between_chunks(self):
        """Ground-truth context: predictions in chunk N should not depend on chunk N-1.

        We verify this by predicting the full val set, then predicting just the
        second chunk's indices. With ground-truth context the results should
        match; with autoregressive chaining they would diverge.
        """
        model = self._get_model('nhits', 'er', TaskType.EVENT_RATE, 5, 20, False)
        full_preds = model.predict_pointwise(self._er_dataset, self._er_dataset.val_indices)
        second_chunk_indices = self._er_dataset.val_indices[5:10]
        partial_preds = model.predict_pointwise(self._er_dataset, second_chunk_indices)
        np.testing.assert_array_almost_equal(
            full_preds[5:10], partial_preds, decimal=4,
            err_msg="Predictions differ — ground-truth context may not be working",
        )

    def test_exact_horizon_boundary(self):
        """n_needed is exact multiple of horizon — no partial final chunk."""
        model = self._get_model('nhits', 'er', TaskType.EVENT_RATE, 10, 20, False)
        indices = self._er_dataset.val_indices[:10]
        preds = model.predict_pointwise(self._er_dataset, indices)
        self.assertEqual(len(preds), 10)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_single_index(self):
        """Predict a single index — minimal case."""
        model = self._get_model('nhits', 'er', TaskType.EVENT_RATE, 10, 20, False)
        preds = model.predict_pointwise(
            self._er_dataset, [self._er_dataset.val_indices[0]],
        )
        self.assertEqual(len(preds), 1)
        self.assertTrue(np.isfinite(preds[0]))
        self.assertGreaterEqual(preds[0], 0)


# Registration sanity check (always runs — lightweight, no model loading)

class TestModelRegistration(unittest.TestCase):
    """Verify all new models are registered in the factory."""

    def test_all_new_models_registered(self):
        from modeling.model_factory import is_model_supported
        pairs = [
            ("Chronos", "event_rate"),
            ("Chronos", "time_to_event"),
            ("TFT", "event_rate"),
            ("TFT", "time_to_event"),
            ("NHiTS", "event_rate"),
            ("NHiTS", "time_to_event"),
        ]
        for model_type, task in pairs:
            self.assertTrue(
                is_model_supported(model_type, task),
                f"{model_type}/{task} not registered",
            )

    def test_ensemble_registered(self):
        from modeling.model_factory import is_model_supported
        for task in ["event_rate", "time_to_event"]:
            self.assertTrue(
                is_model_supported("Ensemble", task),
                f"Ensemble/{task} not registered",
            )


# Ensemble

@unittest.skipUnless(NEURALFORECAST_AVAILABLE, "neuralforecast not installed")
@unittest.skipIf(_SKIP_HEAVY, "Heavy test — set RUN_HEAVY_TESTS=1")
class TestEnsembleModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from modeling.models.ensemble_model import EnsembleModel
        n = 800
        base = _make_tte_dataset(n=n)
        cls.dataset = TimeToEventDataset(
            timestamps=base.timestamps,
            minutes_to_next_event=base.minutes_to_next_event,
            covariates={
                "hour_of_day": [float(t.hour) for t in base.timestamps],
                "day_of_week": [float(t.weekday()) for t in base.timestamps],
                "is_weekend": [float(t.weekday() >= 5) for t in base.timestamps],
            },
            train_indices=base.train_indices,
            val_indices=base.val_indices,
            test_indices=base.test_indices,
        )
        cls.model = EnsembleModel(
            task_type=TaskType.TIME_TO_EVENT,
            model_config={
                "base_models": [
                    {"model_type": "NHiTS", "template_name": "nhits_tte_v2"},
                    {"model_type": "TFT", "template_name": "tft_tte_v2"},
                ],
                "aggregation": "weighted",
            },
            training_config={},
        )
        cls.fit_info = cls.model.fit(cls.dataset)

    def test_ensemble_tte_fit_and_predict(self):
        """Test ensemble of NHiTS + TFT on TTE task."""
        self.assertTrue(self.model.is_fitted)
        self.assertEqual(self.fit_info["n_base_models"], 2)
        self.assertIsNotNone(self.fit_info["weights"])
        self.assertAlmostEqual(sum(self.fit_info["weights"]), 1.0, places=5)

        preds = self.model.predict_pointwise(self.dataset, self.dataset.val_indices)
        self.assertEqual(len(preds), len(self.dataset.val_indices))
        self.assertTrue(np.all(np.isfinite(preds)))
        self.assertTrue(np.all(preds >= 0))

        traj = self.model.forecast(self.dataset, horizon_minutes=5)
        self.assertEqual(len(traj.points), 5)

    @unittest.skip("neuralforecast load requires torch safe_globals workaround")
    def test_ensemble_save_and_load(self):
        """Test ensemble save/load round-trip."""
        from modeling.models.ensemble_model import EnsembleModel
        dataset = _make_tte_dataset(n=120)
        model = EnsembleModel(
            task_type=TaskType.TIME_TO_EVENT,
            model_config={
                "base_models": [
                    {"model_type": "NHiTS", "template_name": "nhits_tte_v1"},
                    {"model_type": "TFT", "template_name": "tft_tte_v1"},
                ],
                "aggregation": "mean",
            },
            training_config={},
        )
        model.fit(dataset)

        tmpdir = tempfile.mkdtemp()
        try:
            model.save(tmpdir)
            loaded = EnsembleModel.load(
                tmpdir, TaskType.TIME_TO_EVENT,
                model.model_config,
            )
            self.assertTrue(loaded.is_fitted)
            preds_orig = model.predict_pointwise(dataset, dataset.val_indices)
            preds_loaded = loaded.predict_pointwise(dataset, dataset.val_indices)
            np.testing.assert_allclose(preds_orig, preds_loaded, rtol=1e-3)
        finally:
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
