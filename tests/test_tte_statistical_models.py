"""Tests for LogNormal, Gamma, and LogLogistic time-to-event models."""

import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np

from modeling.core_types import EventRateDataset, TaskType, TimeToEventDataset
from modeling.models.lognormal_model import LogNormalTimeToEventModel
from modeling.models.gamma_model import GammaTimeToEventModel
from modeling.models.loglogistic_model import LogLogisticTimeToEventModel


def _make_tte_dataset(n=200, seed=42):
    """Create a synthetic TimeToEventDataset with log-normal-ish data."""
    rng = np.random.default_rng(seed)
    deltas = rng.lognormal(mean=3.0, sigma=0.5, size=n).tolist()
    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(seconds=sum(deltas[:i])) for i in range(n)]
    return TimeToEventDataset(
        timestamps=timestamps,
        minutes_to_next_event=deltas,
        covariates={},
        train_indices=list(range(0, 140)),
        val_indices=list(range(140, 170)),
        test_indices=list(range(170, 200)),
    )


def _make_event_rate_dataset():
    """Create a dummy EventRateDataset for rejection tests."""
    return EventRateDataset(
        timestamps=[datetime(2024, 1, 1)],
        events_per_minute=[1.0],
        covariates={},
        train_indices=[0],
        val_indices=[],
        test_indices=[],
    )


class _TTEModelTestMixin:
    """Shared tests for all TTE statistical models."""

    model_class = None
    model_type_name = None

    def _create_model(self, model_config=None):
        return self.model_class(
            task_type=TaskType.TIME_TO_EVENT,
            model_config=model_config or {"min_delta": 0.0},
            training_config={},
        )

    def test_fit_returns_expected_keys(self):
        model = self._create_model()
        dataset = _make_tte_dataset()
        info = model.fit(dataset)
        self.assertIn("mean_delta", info)
        self.assertIn("n_train", info)
        self.assertIn("log_likelihood", info)
        self.assertIn("aic", info)
        self.assertIn("bic", info)
        self.assertTrue(model.is_fitted)

    def test_predict_pointwise(self):
        model = self._create_model()
        dataset = _make_tte_dataset()
        model.fit(dataset)
        preds = model.predict_pointwise(dataset, dataset.val_indices)
        self.assertEqual(len(preds), len(dataset.val_indices))
        for p in preds:
            self.assertGreater(p, 0)

    def test_forecast(self):
        model = self._create_model()
        dataset = _make_tte_dataset()
        model.fit(dataset)
        traj = model.forecast(dataset, horizon_minutes=5)
        self.assertEqual(len(traj.points), 5)
        for pt in traj.points:
            self.assertIsNotNone(pt.expected_events)
            self.assertIsNotNone(pt.lower_quantile)
            self.assertIsNotNone(pt.upper_quantile)

    def test_save_and_load(self):
        model = self._create_model()
        dataset = _make_tte_dataset()
        model.fit(dataset)

        tmpdir = tempfile.mkdtemp()
        try:
            model.save(tmpdir)
            loaded = self.model_class.load(
                tmpdir, TaskType.TIME_TO_EVENT, {"min_delta": 0.0},
            )
            self.assertTrue(loaded.is_fitted)
            orig = model.predict_pointwise(dataset, [140])
            reloaded = loaded.predict_pointwise(dataset, [140])
            np.testing.assert_allclose(orig, reloaded)
        finally:
            shutil.rmtree(tmpdir)

    def test_rejects_event_rate_dataset(self):
        with self.assertRaises(ValueError):
            self.model_class(
                task_type=TaskType.EVENT_RATE,
                model_config={},
                training_config={},
            )

    def test_rejects_wrong_dataset_type_in_fit(self):
        model = self._create_model()
        er_dataset = _make_event_rate_dataset()
        with self.assertRaises(ValueError):
            model.fit(er_dataset)

    def test_min_delta_filtering(self):
        # lognormal(3, 0.5) has mean ~22, so min_delta=15 filters some
        model = self._create_model(model_config={"min_delta": 15.0})
        dataset = _make_tte_dataset()
        info = model.fit(dataset)
        self.assertGreater(info["n_filtered_out"], 0)


class TestLogNormalModel(_TTEModelTestMixin, unittest.TestCase):
    model_class = LogNormalTimeToEventModel
    model_type_name = "LogNormal"


class TestGammaModel(_TTEModelTestMixin, unittest.TestCase):
    model_class = GammaTimeToEventModel
    model_type_name = "Gamma"


class TestLogLogisticModel(_TTEModelTestMixin, unittest.TestCase):
    model_class = LogLogisticTimeToEventModel
    model_type_name = "LogLogistic"

    def test_infinite_mean_fallback_to_median(self):
        """When shape c <= 1, the mean is infinite; model should use median."""
        from scipy.stats import fisk as fisk_dist
        # Generate from fisk with c < 1 to guarantee infinite mean
        rng = np.random.default_rng(99)
        deltas = fisk_dist.rvs(c=0.7, scale=10, size=200, random_state=rng).tolist()
        start = datetime(2024, 1, 1)
        timestamps = [start + timedelta(minutes=i) for i in range(200)]
        dataset = TimeToEventDataset(
            timestamps=timestamps,
            minutes_to_next_event=deltas,
            covariates={},
            train_indices=list(range(0, 140)),
            val_indices=list(range(140, 170)),
            test_indices=list(range(170, 200)),
        )
        model = self._create_model()
        info = model.fit(dataset)
        # mean_delta should be finite (median fallback)
        self.assertTrue(np.isfinite(info["mean_delta"]))
        self.assertFalse(info["mean_is_finite"])
        # predictions should also be finite
        preds = model.predict_pointwise(dataset, dataset.val_indices)
        self.assertTrue(np.all(np.isfinite(preds)))
        # forecast should produce finite values
        traj = model.forecast(dataset, horizon_minutes=3)
        for pt in traj.points:
            self.assertTrue(np.isfinite(pt.expected_events))


if __name__ == "__main__":
    unittest.main()
