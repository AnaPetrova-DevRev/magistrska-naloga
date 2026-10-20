"""Tests for the model validation layer.

Verifies that:
  - Universal pre-fit checks detect bad data (NaN, Inf, negatives, etc.)
  - Model-specific checks fire for the right conditions
  - The validation pipeline injects warnings into fit_info
  - All models produce finite predictions across diverse data shapes
"""

import unittest
from datetime import datetime, timedelta

import numpy as np

from modeling.core_types import EventRateDataset, TaskType, TimeToEventDataset
from modeling.validation import (
    FitWarning,
    ValidationResult,
    check_data_length_for_windows,
    check_excessive_zeros,
    check_fitted_params_finite,
    check_infinite_values,
    check_minimum_samples,
    check_nan_values,
    check_negative_values,
    check_zero_variance,
)


# Test individual check functions

class TestCheckFunctions(unittest.TestCase):

    def test_check_nan_values_clean(self):
        data = np.array([1.0, 2.0, 3.0])
        self.assertIsNone(check_nan_values(data))

    def test_check_nan_values_detected(self):
        data = np.array([1.0, np.nan, 3.0])
        w = check_nan_values(data)
        self.assertIsNotNone(w)
        self.assertEqual(w.code, "nan_in_data")
        self.assertEqual(w.level, "error")

    def test_check_infinite_values_clean(self):
        data = np.array([1.0, 2.0])
        self.assertIsNone(check_infinite_values(data))

    def test_check_infinite_values_detected(self):
        data = np.array([1.0, np.inf, -np.inf])
        w = check_infinite_values(data)
        self.assertIsNotNone(w)
        self.assertEqual(w.code, "inf_in_data")

    def test_check_negative_values_clean(self):
        self.assertIsNone(check_negative_values(np.array([0.0, 1.0, 2.0])))

    def test_check_negative_values_detected(self):
        w = check_negative_values(np.array([1.0, -0.5, 2.0]))
        self.assertIsNotNone(w)
        self.assertEqual(w.code, "negative_values")

    def test_check_minimum_samples_ok(self):
        self.assertIsNone(check_minimum_samples(np.arange(20)))

    def test_check_minimum_samples_warning(self):
        w = check_minimum_samples(np.arange(5), min_n=10)
        self.assertIsNotNone(w)
        self.assertEqual(w.level, "warning")

    def test_check_minimum_samples_error(self):
        w = check_minimum_samples(np.array([1.0, 2.0]), min_n=10)
        self.assertIsNotNone(w)
        self.assertEqual(w.level, "error")

    def test_check_zero_variance_ok(self):
        self.assertIsNone(check_zero_variance(np.array([1.0, 2.0, 3.0])))

    def test_check_zero_variance_detected(self):
        w = check_zero_variance(np.array([5.0, 5.0, 5.0]))
        self.assertIsNotNone(w)
        self.assertEqual(w.code, "zero_variance")
        self.assertEqual(w.level, "error")

    def test_check_excessive_zeros_ok(self):
        data = np.array([0.0, 1.0, 2.0, 3.0])
        self.assertIsNone(check_excessive_zeros(data))

    def test_check_excessive_zeros_detected(self):
        data = np.array([0.0, 0.0, 0.0, 1.0])
        w = check_excessive_zeros(data, threshold=0.5)
        self.assertIsNotNone(w)
        self.assertEqual(w.code, "excessive_zeros")

    def test_check_fitted_params_finite_ok(self):
        self.assertIsNone(check_fitted_params_finite({"a": 1.0, "b": 2.0}))

    def test_check_fitted_params_finite_detected(self):
        w = check_fitted_params_finite({"a": 1.0, "b": float("inf")})
        self.assertIsNotNone(w)
        self.assertEqual(w.code, "non_finite_params")

    def test_check_data_length_for_windows_ok(self):
        self.assertIsNone(check_data_length_for_windows(50, 10))

    def test_check_data_length_for_windows_warning(self):
        w = check_data_length_for_windows(12, 10)
        self.assertIsNotNone(w)
        self.assertEqual(w.code, "insufficient_data_for_windows")
        self.assertEqual(w.level, "warning")

    def test_check_data_length_for_windows_error(self):
        w = check_data_length_for_windows(5, 10)
        self.assertIsNotNone(w)
        self.assertEqual(w.level, "error")


class TestValidationResult(unittest.TestCase):

    def test_empty_result(self):
        r = ValidationResult()
        self.assertTrue(r.is_valid)
        self.assertEqual(len(r.warnings), 0)
        self.assertEqual(r.to_list(), [])

    def test_add_none(self):
        r = ValidationResult()
        r.add(None)
        self.assertTrue(r.is_valid)
        self.assertEqual(len(r.warnings), 0)

    def test_add_warning(self):
        r = ValidationResult()
        r.add(FitWarning("warning", "test", "msg"))
        self.assertTrue(r.is_valid)  # warnings don't invalidate
        self.assertEqual(len(r.warnings), 1)

    def test_add_error(self):
        r = ValidationResult()
        r.add(FitWarning("error", "test", "msg"))
        self.assertFalse(r.is_valid)

    def test_merge(self):
        r1 = ValidationResult()
        r1.add(FitWarning("warning", "a", "msg a"))
        r2 = ValidationResult()
        r2.add(FitWarning("error", "b", "msg b"))
        r1.merge(r2)
        self.assertFalse(r1.is_valid)
        self.assertEqual(len(r1.warnings), 2)

    def test_to_list(self):
        r = ValidationResult()
        r.add(FitWarning("warning", "code1", "message1"))
        lst = r.to_list()
        self.assertEqual(len(lst), 1)
        self.assertEqual(lst[0]["code"], "code1")
        self.assertEqual(lst[0]["level"], "warning")
        self.assertEqual(lst[0]["message"], "message1")


# Test validation integration in model fit()

def _make_tte_dataset(deltas, n=None):
    """Create a TTE dataset from a list/array of deltas."""
    if n is None:
        n = len(deltas)
    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(minutes=i) for i in range(n)]
    split = int(n * 0.7)
    val_end = int(n * 0.85)
    return TimeToEventDataset(
        timestamps=timestamps,
        minutes_to_next_event=list(deltas[:n]),
        covariates={},
        train_indices=list(range(0, split)),
        val_indices=list(range(split, val_end)),
        test_indices=list(range(val_end, n)),
    )


def _make_event_rate_dataset(values, n=None):
    """Create an EventRate dataset from a list/array of values."""
    if n is None:
        n = len(values)
    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(minutes=i) for i in range(n)]
    split = int(n * 0.7)
    val_end = int(n * 0.85)
    return EventRateDataset(
        timestamps=timestamps,
        events_per_minute=list(values[:n]),
        covariates={},
        train_indices=list(range(0, split)),
        val_indices=list(range(split, val_end)),
        test_indices=list(range(val_end, n)),
    )


class TestFitValidationIntegration(unittest.TestCase):
    """Test that fit() returns warnings in the training info dict."""

    def test_exponential_fit_has_warnings_key(self):
        from modeling.models.exponential_model import ExponentialTimeToEventModel
        rng = np.random.default_rng(42)
        deltas = rng.exponential(10, 100)
        dataset = _make_tte_dataset(deltas)
        model = ExponentialTimeToEventModel(
            task_type=TaskType.TIME_TO_EVENT,
            model_config={}, training_config={},
        )
        info = model.fit(dataset)
        self.assertIn("warnings", info)
        self.assertIn("validation_passed", info)

    def test_exponential_ks_test_fires_on_non_exponential_data(self):
        from modeling.models.exponential_model import ExponentialTimeToEventModel
        # Bimodal data won't fit exponential well
        rng = np.random.default_rng(42)
        deltas = np.concatenate([rng.exponential(1, 80), rng.exponential(100, 80)])
        rng.shuffle(deltas)
        dataset = _make_tte_dataset(deltas)
        model = ExponentialTimeToEventModel(
            task_type=TaskType.TIME_TO_EVENT,
            model_config={}, training_config={},
        )
        info = model.fit(dataset)
        codes = [w["code"] for w in info["warnings"]]
        self.assertIn("ks_test_poor_fit", codes)

    def test_lognormal_normality_test(self):
        from modeling.models.lognormal_model import LogNormalTimeToEventModel
        # Uniform data — log won't be normal
        rng = np.random.default_rng(42)
        deltas = rng.uniform(1, 100, 200)
        dataset = _make_tte_dataset(deltas)
        model = LogNormalTimeToEventModel(
            task_type=TaskType.TIME_TO_EVENT,
            model_config={}, training_config={},
        )
        info = model.fit(dataset)
        codes = [w["code"] for w in info["warnings"]]
        self.assertIn("log_not_normal", codes)

    def test_clean_data_no_warnings(self):
        from modeling.models.exponential_model import ExponentialTimeToEventModel
        rng = np.random.default_rng(42)
        deltas = rng.exponential(10, 200)
        dataset = _make_tte_dataset(deltas)
        model = ExponentialTimeToEventModel(
            task_type=TaskType.TIME_TO_EVENT,
            model_config={}, training_config={},
        )
        info = model.fit(dataset)
        self.assertTrue(info["validation_passed"])


# Robustness: all TTE models × diverse data shapes

_DATA_SCENARIOS = {
    "lognormal": lambda rng, n: rng.lognormal(3.0, 0.5, n),
    "exponential": lambda rng, n: rng.exponential(20, n),
    "heavy_tailed": lambda rng, n: (rng.pareto(1.5, n) + 1) * 5,
    "near_constant": lambda rng, n: 10.0 + rng.normal(0, 0.01, n),
    "bimodal": lambda rng, n: np.where(
        rng.random(n) < 0.5, rng.exponential(5, n), rng.exponential(100, n),
    ),
}

_TTE_MODELS = [
    ("Exponential", lambda: __import__(
        "modeling.models.exponential_model", fromlist=["ExponentialTimeToEventModel"]
    ).ExponentialTimeToEventModel),
    ("Weibull", lambda: __import__(
        "modeling.models.weibull_model", fromlist=["WeibullTimeToEventModel"]
    ).WeibullTimeToEventModel),
    ("LogNormal", lambda: __import__(
        "modeling.models.lognormal_model", fromlist=["LogNormalTimeToEventModel"]
    ).LogNormalTimeToEventModel),
    ("Gamma", lambda: __import__(
        "modeling.models.gamma_model", fromlist=["GammaTimeToEventModel"]
    ).GammaTimeToEventModel),
    ("LogLogistic", lambda: __import__(
        "modeling.models.loglogistic_model", fromlist=["LogLogisticTimeToEventModel"]
    ).LogLogisticTimeToEventModel),
]

try:
    import lifelines
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

if LIFELINES_AVAILABLE:
    _TTE_MODELS.append(
        ("KaplanMeier", lambda: __import__(
            "modeling.models.kaplan_meier_model", fromlist=["KaplanMeierTimeToEventModel"]
        ).KaplanMeierTimeToEventModel),
    )


class TestTTEModelRobustness(unittest.TestCase):
    """Every TTE model must produce finite predictions on every data shape."""

    def _run_model_on_data(self, model_name, get_class, scenario_name, gen_fn):
        rng = np.random.default_rng(42)
        deltas = np.abs(gen_fn(rng, 200))  # ensure positive
        deltas = np.maximum(deltas, 1e-6)  # no exact zeros
        dataset = _make_tte_dataset(deltas)

        cls = get_class()
        model = cls(
            task_type=TaskType.TIME_TO_EVENT,
            model_config={"min_delta": 0.0},
            training_config={},
        )
        info = model.fit(dataset)

        # Invariant 1: fit_info has validation fields
        self.assertIn("warnings", info, f"{model_name}/{scenario_name}")
        self.assertIn("validation_passed", info, f"{model_name}/{scenario_name}")

        # Invariant 2: predictions are finite and positive
        preds = model.predict_pointwise(dataset, dataset.val_indices)
        self.assertEqual(
            len(preds), len(dataset.val_indices),
            f"{model_name}/{scenario_name}: wrong number of predictions",
        )
        self.assertTrue(
            np.all(np.isfinite(preds)),
            f"{model_name}/{scenario_name}: non-finite predictions: {preds}",
        )
        self.assertTrue(
            np.all(preds > 0),
            f"{model_name}/{scenario_name}: non-positive predictions: {preds}",
        )

        # Invariant 3: forecast produces finite values
        traj = model.forecast(dataset, horizon_minutes=3)
        self.assertEqual(len(traj.points), 3, f"{model_name}/{scenario_name}")
        for pt in traj.points:
            self.assertTrue(
                np.isfinite(pt.expected_events),
                f"{model_name}/{scenario_name}: non-finite forecast point",
            )


def _make_robustness_test(model_name, get_class, scenario_name, gen_fn):
    def test_method(self):
        self._run_model_on_data(model_name, get_class, scenario_name, gen_fn)
    test_method.__name__ = f"test_{model_name}_{scenario_name}"
    test_method.__doc__ = f"{model_name} on {scenario_name} data"
    return test_method


# Dynamically generate test methods: one per (model × data_scenario)
for _model_name, _get_class in _TTE_MODELS:
    for _scenario_name, _gen_fn in _DATA_SCENARIOS.items():
        _test = _make_robustness_test(_model_name, _get_class, _scenario_name, _gen_fn)
        setattr(TestTTEModelRobustness, _test.__name__, _test)


if __name__ == "__main__":
    unittest.main()
