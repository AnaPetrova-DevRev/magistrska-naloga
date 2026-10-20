"""Tests for zero-inflated models: HurdleNB (event rate) and ZI_* (time-to-event)."""

import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np

from modeling.core_types import (
    EventRateDataset,
    TaskType,
    TimeToEventDataset,
)
from modeling.models.hurdle_event_rate_model import HurdleEventRateModel
from modeling.models.zero_inflated_tte_model import ZeroInflatedTTEModel


# Synthetic data helpers


def _make_zero_inflated_er_dataset(
    n=300, frac_zero=0.6, seed=42, with_covariates=True,
):
    """Create a synthetic EventRateDataset with configurable zero fraction.

    Non-zero values are drawn from a shifted Negative Binomial-like
    distribution (Poisson + noise) to mimic bursty event counts.
    """
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(minutes=15 * i) for i in range(n)]

    values = np.zeros(n)
    n_nonzero = int(n * (1 - frac_zero))
    nonzero_indices = rng.choice(n, size=n_nonzero, replace=False)
    values[nonzero_indices] = rng.poisson(lam=5, size=n_nonzero) + 1.0

    covariates = {}
    if with_covariates:
        covariates = {
            "hour_of_day": [float(ts.hour) for ts in timestamps],
            "day_of_week": [float(ts.weekday()) for ts in timestamps],
            "is_weekend": [1.0 if ts.weekday() >= 5 else 0.0 for ts in timestamps],
        }

    split1 = int(n * 0.7)
    split2 = int(n * 0.85)

    return EventRateDataset(
        timestamps=timestamps,
        events_per_minute=values.tolist(),
        covariates=covariates,
        train_indices=list(range(0, split1)),
        val_indices=list(range(split1, split2)),
        test_indices=list(range(split2, n)),
    )


def _make_zero_inflated_tte_dataset(n=300, frac_immediate=0.5, seed=42):
    """Create a synthetic TimeToEventDataset with a mixture of zeros and positives.

    Immediate arrivals have delta close to 0 (below threshold).
    Positive arrivals are drawn from a LogNormal distribution.
    """
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1)

    n_immediate = int(n * frac_immediate)
    n_positive = n - n_immediate

    # Immediate arrivals: very small deltas (seconds → minutes)
    immediate_deltas = rng.uniform(0.0, 0.005, size=n_immediate)  # < 0.3 seconds in minutes
    # Positive arrivals: LogNormal in minutes
    positive_deltas = rng.lognormal(mean=2.0, sigma=0.8, size=n_positive)

    all_deltas = np.concatenate([immediate_deltas, positive_deltas])
    rng.shuffle(all_deltas)

    timestamps = [start + timedelta(seconds=sum(all_deltas[:i] * 60)) for i in range(n)]

    split1 = int(n * 0.7)
    split2 = int(n * 0.85)

    return TimeToEventDataset(
        timestamps=timestamps,
        minutes_to_next_event=all_deltas.tolist(),
        covariates={},
        train_indices=list(range(0, split1)),
        val_indices=list(range(split1, split2)),
        test_indices=list(range(split2, n)),
    )


# HurdleNB Event Rate Tests


class TestHurdleEventRateModel(unittest.TestCase):
    """Tests for HurdleEventRateModel."""

    def _create_model(self, model_config=None):
        return HurdleEventRateModel(
            task_type=TaskType.EVENT_RATE,
            model_config=model_config or {"binary_regularization": 0.01},
            training_config={},
        )

    def test_fit_returns_expected_keys(self):
        model = self._create_model()
        dataset = _make_zero_inflated_er_dataset()
        info = model.fit(dataset)
        self.assertIn("marginal_p_nonzero", info)
        self.assertIn("frac_zeros", info)
        self.assertIn("nb_r", info)
        self.assertIn("nb_p", info)
        self.assertIn("nb_mean_positive", info)
        self.assertIn("n_train", info)
        self.assertIn("n_positive", info)
        self.assertIn("binary_log_likelihood", info)
        self.assertTrue(model.is_fitted)

    def test_fit_captures_zero_fraction(self):
        dataset = _make_zero_inflated_er_dataset(frac_zero=0.8)
        model = self._create_model()
        info = model.fit(dataset)
        # Should detect ~80% zeros in training split
        self.assertGreater(info["frac_zeros"], 0.5)

    def test_predict_pointwise(self):
        model = self._create_model()
        dataset = _make_zero_inflated_er_dataset()
        model.fit(dataset)
        preds = model.predict_pointwise(dataset, dataset.val_indices)
        self.assertEqual(len(preds), len(dataset.val_indices))
        # All predictions should be non-negative
        self.assertTrue(np.all(preds >= 0))

    def test_predictions_are_mixture(self):
        """Predictions should be between 0 and the mean of positive values."""
        model = self._create_model()
        dataset = _make_zero_inflated_er_dataset(frac_zero=0.6)
        info = model.fit(dataset)
        preds = model.predict_pointwise(dataset, dataset.val_indices)
        # Mixture mean should be less than the positive-only mean
        mean_pred = np.mean(preds)
        self.assertLess(mean_pred, info["nb_mean_positive"])
        self.assertGreater(mean_pred, 0)

    def test_forecast(self):
        model = self._create_model()
        dataset = _make_zero_inflated_er_dataset()
        model.fit(dataset)
        traj = model.forecast(dataset, horizon_minutes=10)
        self.assertEqual(len(traj.points), 10)
        for pt in traj.points:
            self.assertIsNotNone(pt.expected_events)
            self.assertGreaterEqual(pt.expected_events, 0)
            self.assertIsNotNone(pt.lower_quantile)
            self.assertIsNotNone(pt.upper_quantile)

    def test_save_and_load(self):
        model = self._create_model()
        dataset = _make_zero_inflated_er_dataset()
        model.fit(dataset)

        tmpdir = tempfile.mkdtemp()
        try:
            model.save(tmpdir)
            loaded = HurdleEventRateModel.load(
                tmpdir, TaskType.EVENT_RATE, {"binary_regularization": 0.01},
            )
            self.assertTrue(loaded.is_fitted)
            orig = model.predict_pointwise(dataset, dataset.val_indices)
            reloaded = loaded.predict_pointwise(dataset, dataset.val_indices)
            np.testing.assert_allclose(orig, reloaded)
        finally:
            shutil.rmtree(tmpdir)

    def test_rejects_time_to_event_task(self):
        with self.assertRaises(ValueError):
            HurdleEventRateModel(
                task_type=TaskType.TIME_TO_EVENT,
                model_config={},
                training_config={},
            )

    def test_rejects_wrong_dataset_type(self):
        model = self._create_model()
        tte_dataset = TimeToEventDataset(
            timestamps=[datetime(2024, 1, 1)],
            minutes_to_next_event=[1.0],
            covariates={},
            train_indices=[0],
            val_indices=[],
            test_indices=[],
        )
        with self.assertRaises(ValueError):
            model.fit(tte_dataset)

    def test_no_covariates_fallback(self):
        """Model should work without covariates using marginal P(Y>0)."""
        model = self._create_model()
        dataset = _make_zero_inflated_er_dataset(with_covariates=False)
        info = model.fit(dataset)
        self.assertTrue(model.is_fitted)
        preds = model.predict_pointwise(dataset, dataset.val_indices)
        # Without covariates, all predictions should be the same constant
        self.assertTrue(np.all(preds == preds[0]))

    def test_high_zero_fraction(self):
        """Model should handle 98% zeros without crashing."""
        model = self._create_model()
        dataset = _make_zero_inflated_er_dataset(frac_zero=0.98, n=500)
        info = model.fit(dataset)
        self.assertTrue(model.is_fitted)
        preds = model.predict_pointwise(dataset, dataset.val_indices)
        self.assertTrue(np.all(np.isfinite(preds)))

    def test_all_zeros(self):
        """Model should handle degenerate all-zero data."""
        start = datetime(2024, 1, 1)
        n = 100
        dataset = EventRateDataset(
            timestamps=[start + timedelta(minutes=15 * i) for i in range(n)],
            events_per_minute=[0.0] * n,
            covariates={},
            train_indices=list(range(70)),
            val_indices=list(range(70, 85)),
            test_indices=list(range(85, 100)),
        )
        model = self._create_model()
        info = model.fit(dataset)
        preds = model.predict_pointwise(dataset, dataset.val_indices)
        # All zeros → predictions should be zero
        np.testing.assert_allclose(preds, 0.0)


# ZeroInflated TTE Tests — shared mixin


class _ZITTEModelTestMixin:
    """Shared tests for ZI_LogNormal, ZI_Gamma, ZI_Weibull."""

    dist_name = None
    model_type_name = None

    def _create_model(self, model_config=None):
        config = model_config or {
            "positive_distribution": self.dist_name,
            "zero_threshold": 1.0 / 60.0,  # 1 second
        }
        return ZeroInflatedTTEModel(
            task_type=TaskType.TIME_TO_EVENT,
            model_config=config,
            training_config={},
        )

    def test_fit_returns_expected_keys(self):
        model = self._create_model()
        dataset = _make_zero_inflated_tte_dataset()
        info = model.fit(dataset)
        self.assertIn("p_immediate", info)
        self.assertIn("frac_above_threshold", info)
        self.assertIn("n_immediate", info)
        self.assertIn("n_positive", info)
        self.assertIn("mixture_mean_delta", info)
        self.assertIn("positive_distribution", info)
        self.assertEqual(info["positive_distribution"], self.dist_name)
        self.assertTrue(model.is_fitted)

    def test_fit_captures_immediate_fraction(self):
        dataset = _make_zero_inflated_tte_dataset(frac_immediate=0.7)
        model = self._create_model()
        info = model.fit(dataset)
        # Should detect ~70% immediate in training split
        self.assertGreater(info["p_immediate"], 0.4)

    def test_predict_pointwise(self):
        model = self._create_model()
        dataset = _make_zero_inflated_tte_dataset()
        model.fit(dataset)
        preds = model.predict_pointwise(dataset, dataset.val_indices)
        self.assertEqual(len(preds), len(dataset.val_indices))
        self.assertTrue(np.all(preds >= 0))

    def test_mixture_prediction_less_than_positive(self):
        """Mixture mean should be less than positive-only mean."""
        model = self._create_model()
        dataset = _make_zero_inflated_tte_dataset(frac_immediate=0.5)
        info = model.fit(dataset)
        self.assertLess(
            info["mixture_mean_delta"],
            info["positive_mean_delta"],
        )

    def test_forecast(self):
        model = self._create_model()
        dataset = _make_zero_inflated_tte_dataset()
        model.fit(dataset)
        traj = model.forecast(dataset, horizon_minutes=5)
        self.assertEqual(len(traj.points), 5)
        for pt in traj.points:
            self.assertIsNotNone(pt.expected_events)
            self.assertGreaterEqual(pt.expected_events, 0)

    def test_save_and_load(self):
        model = self._create_model()
        dataset = _make_zero_inflated_tte_dataset()
        model.fit(dataset)

        tmpdir = tempfile.mkdtemp()
        try:
            model.save(tmpdir)
            loaded = ZeroInflatedTTEModel.load(
                tmpdir,
                TaskType.TIME_TO_EVENT,
                {
                    "positive_distribution": self.dist_name,
                    "zero_threshold": 1.0 / 60.0,
                },
            )
            self.assertTrue(loaded.is_fitted)
            orig = model.predict_pointwise(dataset, [0])
            reloaded = loaded.predict_pointwise(dataset, [0])
            np.testing.assert_allclose(orig, reloaded)
        finally:
            shutil.rmtree(tmpdir)

    def test_rejects_event_rate_task(self):
        with self.assertRaises(ValueError):
            ZeroInflatedTTEModel(
                task_type=TaskType.EVENT_RATE,
                model_config={"positive_distribution": self.dist_name},
                training_config={},
            )

    def test_rejects_wrong_dataset_type(self):
        model = self._create_model()
        er_dataset = EventRateDataset(
            timestamps=[datetime(2024, 1, 1)],
            events_per_minute=[1.0],
            covariates={},
            train_indices=[0],
            val_indices=[],
            test_indices=[],
        )
        with self.assertRaises(ValueError):
            model.fit(er_dataset)

    def test_model_type_reflects_distribution(self):
        model = self._create_model()
        self.assertEqual(model.MODEL_TYPE, f"ZI_{self.dist_name}")

    def test_high_immediate_fraction(self):
        """Model should handle 95% immediate arrivals."""
        model = self._create_model()
        dataset = _make_zero_inflated_tte_dataset(frac_immediate=0.95, n=500)
        info = model.fit(dataset)
        self.assertTrue(model.is_fitted)
        preds = model.predict_pointwise(dataset, dataset.val_indices)
        self.assertTrue(np.all(np.isfinite(preds)))


class TestZILogNormal(_ZITTEModelTestMixin, unittest.TestCase):
    dist_name = "LogNormal"
    model_type_name = "ZI_LogNormal"


class TestZIGamma(_ZITTEModelTestMixin, unittest.TestCase):
    dist_name = "Gamma"
    model_type_name = "ZI_Gamma"


class TestZIWeibull(_ZITTEModelTestMixin, unittest.TestCase):
    dist_name = "Weibull"
    model_type_name = "ZI_Weibull"


# Model factory registration tests


class TestZeroInflatedRegistration(unittest.TestCase):
    """Verify the new models are registered in the factory."""

    def test_hurdle_nb_registered(self):
        from modeling.model_factory import is_model_supported
        self.assertTrue(is_model_supported("HurdleNB", TaskType.EVENT_RATE))

    def test_zi_lognormal_registered(self):
        from modeling.model_factory import is_model_supported
        self.assertTrue(is_model_supported("ZI_LogNormal", TaskType.TIME_TO_EVENT))

    def test_zi_gamma_registered(self):
        from modeling.model_factory import is_model_supported
        self.assertTrue(is_model_supported("ZI_Gamma", TaskType.TIME_TO_EVENT))

    def test_zi_weibull_registered(self):
        from modeling.model_factory import is_model_supported
        self.assertTrue(is_model_supported("ZI_Weibull", TaskType.TIME_TO_EVENT))

    def test_hurdle_nb_not_supported_for_tte(self):
        from modeling.model_factory import is_model_supported
        self.assertFalse(is_model_supported("HurdleNB", TaskType.TIME_TO_EVENT))

    def test_zi_models_not_supported_for_er(self):
        from modeling.model_factory import is_model_supported
        self.assertFalse(is_model_supported("ZI_LogNormal", TaskType.EVENT_RATE))

    def test_create_hurdle_via_factory(self):
        from modeling.model_factory import create_model_from_config
        model = create_model_from_config(
            "HurdleNB", TaskType.EVENT_RATE,
            model_config={"binary_regularization": 0.01},
        )
        self.assertIsInstance(model, HurdleEventRateModel)

    def test_create_zi_lognormal_via_factory(self):
        from modeling.model_factory import create_model_from_config
        model = create_model_from_config(
            "ZI_LogNormal", TaskType.TIME_TO_EVENT,
            model_config={"positive_distribution": "LogNormal", "zero_threshold": 0.016667},
        )
        self.assertIsInstance(model, ZeroInflatedTTEModel)


if __name__ == "__main__":
    unittest.main()
