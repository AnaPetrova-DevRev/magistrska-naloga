"""Proof-of-concept tests for the model training system.

These tests demonstrate the full training pipeline from data loading
through model training, metrics evaluation, and artifact storage.
"""

from datetime import UTC, datetime, timedelta
import json
import os
import shutil
import unittest

import pandas as pd
import numpy as np

import global_vars
from modeling.core_types import TaskType
from modeling.metrics_registry import build_metric
from model_training.model_training_app import ModelTrainingApp
from storage_access import StorageAccess
from sync_unit import SyncUnit


class TestModelTrainingPipeline(unittest.TestCase):
    """End-to-end tests for the model training pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests."""
        cls._orig_mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME")
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "test_model_training"
        cls.sa = StorageAccess()
        cls.sync_unit = SyncUnit("test_org", "test_system", "test_project")
        cls._setup_test_data()

    @classmethod
    def _setup_test_data(cls):
        """Create synthetic events-per-minute data for testing."""
        base_folder = (
            f"{cls.sync_unit.get_path_for_sync_unit()}/"
            f"{global_vars.DATA_PREPROCESSING_APP_FOLDER_NAME}"
        )
        cls.sa.create_folder(base_folder)

        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        n_minutes = 100
        cls.timestamps = [start + timedelta(minutes=i) for i in range(n_minutes)]

        np.random.seed(42)
        base_rate = 5.0
        noise = np.random.normal(0, 1, n_minutes)
        trend = np.linspace(0, 2, n_minutes)
        cls.events = [max(0, base_rate + trend[i] + noise[i]) for i in range(n_minutes)]

        df = pd.DataFrame({
            "minute": cls.timestamps,
            "events_per_minute": cls.events,
        })

        events_path = (
            f"{base_folder}/"
            f"{global_vars.MERGED_EVENTS_PER_MINUTE_FILE_NAME_FORMAT.format(int=1)}"
        )
        cls.sa.save_dataframe_as_picklefile(df, events_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up only the artifacts created by this test class."""
        if cls._orig_mlflow_experiment is None:
            os.environ.pop("MLFLOW_EXPERIMENT_NAME", None)
        else:
            os.environ["MLFLOW_EXPERIMENT_NAME"] = cls._orig_mlflow_experiment
        test_data_path = cls.sa.create_full_path_string(
            cls.sync_unit.get_path_for_sync_unit()
        )
        for subfolder in ["models", "modeling", global_vars.DATA_PREPROCESSING_APP_FOLDER_NAME]:
            path = os.path.join(test_data_path, subfolder)
            if os.path.exists(path):
                shutil.rmtree(path)

    def test_train_arima_model_basic(self):
        """Test training a basic ARIMA model end-to-end."""
        app = ModelTrainingApp(sa=self.sa)

        parameters = {
            "task_type": "event_rate",
            "dataset": {
                "start_time": self.timestamps[0].isoformat(),
                "end_time": self.timestamps[-1].isoformat(),
                "train_ratio": 0.7,
                "val_ratio": 0.15,
            },
            "models": [
                {"model_type": "ARIMA", "template_name": "arima_small_v1"}
            ],
            "metrics": ["mae", "rmse"],
        }

        response = app.execute_action(
            self.sync_unit, "train_models", parameters
        )

        self.assertTrue(response.success, f"Training failed: {response.message}")
        self.assertIn("results", response.data)

        results = response.data["results"]
        self.assertEqual(len(results), 1)

        result = results[0]
        self.assertTrue(result["success"], f"Model training failed: {result.get('error')}")
        self.assertEqual(result["model_type"], "ARIMA")
        self.assertIn("metrics", result)
        self.assertIn("val_mae", result["metrics"])
        self.assertIn("val_rmse", result["metrics"])

        self.assertIsInstance(result["metrics"]["val_mae"], float)
        self.assertIsInstance(result["metrics"]["val_rmse"], float)
        self.assertGreaterEqual(result["metrics"]["val_mae"], 0)
        self.assertGreaterEqual(result["metrics"]["val_rmse"], 0)

        artifact_path = result["artifact_path"]
        self.assertTrue(os.path.exists(artifact_path), "Artifact directory not created")
        self.assertTrue(
            os.path.exists(os.path.join(artifact_path, "arima_model.pkl")),
            "Model pickle not saved"
        )
        self.assertTrue(
            os.path.exists(os.path.join(artifact_path, "config.json")),
            "Config not saved"
        )
        self.assertTrue(
            os.path.exists(os.path.join(artifact_path, "training_metadata.json")),
            "Training metadata not saved"
        )

        with open(os.path.join(artifact_path, "training_metadata.json"), encoding="utf-8") as f:
            metadata = json.load(f)

        self.assertEqual(metadata["model_type"], "ARIMA")
        self.assertEqual(metadata["task_type"], "event_rate")
        self.assertIn("aic", metadata["training_info"])
        self.assertIn("bic", metadata["training_info"])

    def test_train_multiple_models(self):
        """Test training multiple models in one call."""
        app = ModelTrainingApp(sa=self.sa)

        parameters = {
            "task_type": "event_rate",
            "dataset": {
                "train_ratio": 0.7,
                "val_ratio": 0.15,
            },
            "models": [
                {"model_type": "ARIMA", "template_name": "arima_small_v1"},
                {"model_type": "ARIMA", "template_name": "arima_ar2_v1"},
            ],
            "metrics": ["mae"],
        }

        response = app.execute_action(
            self.sync_unit, "train_models", parameters
        )

        self.assertTrue(response.success, f"Training failed: {response.message}")

        results = response.data["results"]
        self.assertEqual(len(results), 2)

        for result in results:
            self.assertTrue(result["success"], f"Failed: {result}")
            self.assertIn("val_mae", result["metrics"])

    def test_train_with_custom_metrics(self):
        """Test training with custom metric parameters."""
        app = ModelTrainingApp(sa=self.sa)

        parameters = {
            "task_type": "event_rate",
            "dataset": {
                "train_ratio": 0.7,
                "val_ratio": 0.15,
            },
            "models": [
                {"model_type": "ARIMA", "template_name": "arima_small_v1"}
            ],
            "metrics": [
                "mae",
                "rmse",
                {"name": "quantile_loss", "params": {"quantile": 0.9}},
            ],
        }

        response = app.execute_action(
            self.sync_unit, "train_models", parameters
        )

        self.assertTrue(response.success, f"Training failed: {response.message}")

        result = response.data["results"][0]
        self.assertIn("val_mae", result["metrics"])
        self.assertIn("val_rmse", result["metrics"])
        quantile_keys = [k for k in result["metrics"] if "quantile_loss" in k]
        self.assertTrue(len(quantile_keys) > 0, "Expected quantile_loss metric")

    def test_train_handles_unsupported_model_gracefully(self):
        """Test that unsupported model types are handled gracefully."""
        app = ModelTrainingApp(sa=self.sa)

        parameters = {
            "task_type": "event_rate",
            "dataset": {"train_ratio": 0.7, "val_ratio": 0.15},
            "models": [
                {"model_type": "NonExistentModel", "template_name": "fake"},
                {"model_type": "ARIMA", "template_name": "arima_small_v1"},
            ],
            "metrics": ["mae"],
        }

        response = app.execute_action(
            self.sync_unit, "train_models", parameters
        )

        self.assertFalse(response.success)

        results = response.data["results"]
        self.assertEqual(len(results), 2)

        self.assertFalse(results[0]["success"])
        self.assertIn("error", results[0])

        self.assertTrue(results[1]["success"])

    def test_train_with_resample_freq(self):
        """Test training with resample_freq records it in metadata."""
        app = ModelTrainingApp(sa=self.sa)

        parameters = {
            "task_type": "event_rate",
            "dataset": {
                "start_time": self.timestamps[0].isoformat(),
                "end_time": self.timestamps[-1].isoformat(),
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "resample_freq": "1min",
            },
            "models": [
                {"model_type": "ARIMA", "template_name": "arima_small_v1"}
            ],
            "metrics": ["mae"],
        }

        response = app.execute_action(
            self.sync_unit, "train_models", parameters
        )

        self.assertTrue(response.success, f"Training failed: {response.message}")

        result = response.data["results"][0]
        self.assertTrue(result["success"])

        metadata_path = os.path.join(
            result["artifact_path"], "training_metadata.json"
        )
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        self.assertEqual(metadata.get("resample_freq"), "1min")

    def test_train_with_inline_config(self):
        """Test training without a template file (inline config)."""
        app = ModelTrainingApp(sa=self.sa)

        parameters = {
            "task_type": "event_rate",
            "dataset": {"train_ratio": 0.7, "val_ratio": 0.15},
            "models": [
                {"model_type": "ARIMA", "template_name": None}
            ],
            "metrics": ["mae"],
        }

        response = app.execute_action(
            self.sync_unit, "train_models", parameters
        )

        self.assertTrue(response.success, f"Training failed: {response.message}")

        result = response.data["results"][0]
        self.assertTrue(result["success"])
        self.assertIn("val_mae", result["metrics"])


class TestARIMAModelDirectly(unittest.TestCase):
    """Direct tests for the ARIMA model wrapper."""

    def test_arima_fit_and_forecast(self):
        """Test ARIMA model fit and forecast directly."""
        from modeling.core_types import EventRateDataset
        from modeling.models.arima_model import ARIMAEventRateModel

        start = datetime(2024, 1, 1, 0, 0, 0)
        n_points = 50
        timestamps = [start + timedelta(minutes=i) for i in range(n_points)]


        np.random.seed(123)
        events = [max(0, 10 + np.random.normal(0, 2)) for _ in range(n_points)]

        dataset = EventRateDataset(
            timestamps=timestamps,
            events_per_minute=events,
            covariates={},
            train_indices=list(range(40)),
            val_indices=list(range(40, 45)),
            test_indices=list(range(45, 50)),
        )

        model = ARIMAEventRateModel(
            task_type=TaskType.EVENT_RATE,
            model_config={"order": [1, 0, 1]},
            training_config={},
        )

        training_info = model.fit(dataset)

        self.assertIn("aic", training_info)
        self.assertIn("bic", training_info)
        self.assertEqual(training_info["n_train"], 40)

        predictions = model.predict_pointwise(dataset, [40, 41, 42])

        self.assertEqual(len(predictions), 3)
        for pred in predictions:
            self.assertGreaterEqual(pred, 0)

        trajectory = model.forecast(dataset, horizon_minutes=10)

        self.assertEqual(len(trajectory.points), 10)
        self.assertEqual(trajectory.horizon_minutes, 10)
        for point in trajectory.points:
            self.assertIsNotNone(point.expected_events)
            self.assertGreaterEqual(point.expected_events, 0)


class TestResolutionAwareForecast(unittest.TestCase):
    """Tests for resolution-aware minute-level interpolation in forecast()."""

    def test_arima_hourly_forecast_produces_minute_level(self):
        """ARIMA trained on hourly data should produce minute-level forecasts."""
        from modeling.core_types import EventRateDataset
        from modeling.models.arima_model import ARIMAEventRateModel

        n_hours = 50
        start = datetime(2024, 1, 1, 0, 0, 0)
        timestamps = [start + timedelta(hours=i) for i in range(n_hours)]

        np.random.seed(42)
        events_per_hour = [max(0, 60 + np.random.normal(0, 10)) for _ in range(n_hours)]

        dataset = EventRateDataset(
            timestamps=timestamps,
            events_per_minute=events_per_hour,
            covariates={},
            train_indices=list(range(40)),
            val_indices=list(range(40, 45)),
            test_indices=list(range(45, 50)),
        )

        model = ARIMAEventRateModel(
            task_type=TaskType.EVENT_RATE,
            model_config={"order": [1, 0, 1]},
            training_config={},
        )
        model.fit(dataset)

        # Request 120 minute forecast
        traj = model.forecast(dataset, horizon_minutes=120)
        self.assertEqual(len(traj.points), 120)

        # Values should be per-minute (roughly 1.0 = 60 events/hour / 60)
        values = [p.expected_events for p in traj.points]
        mean_val = np.mean(values)
        self.assertGreater(mean_val, 0.1)
        self.assertLess(mean_val, 5.0)

        # Should have smooth transitions, not step-function
        diffs = np.abs(np.diff(values))
        max_jump = np.max(diffs)
        self.assertLess(max_jump, mean_val * 2, "Forecast should be smooth, not step-function")

    def test_arima_minute_forecast_unchanged(self):
        """ARIMA trained on minute data should still work normally."""
        from modeling.core_types import EventRateDataset
        from modeling.models.arima_model import ARIMAEventRateModel

        n = 50
        start = datetime(2024, 1, 1, 0, 0, 0)
        timestamps = [start + timedelta(minutes=i) for i in range(n)]
        np.random.seed(42)
        events = [max(0, 5 + np.random.normal(0, 1)) for _ in range(n)]

        dataset = EventRateDataset(
            timestamps=timestamps,
            events_per_minute=events,
            covariates={},
            train_indices=list(range(40)),
            val_indices=list(range(40, 45)),
            test_indices=list(range(45, 50)),
        )

        model = ARIMAEventRateModel(
            task_type=TaskType.EVENT_RATE,
            model_config={"order": [1, 0, 1]},
            training_config={},
        )
        model.fit(dataset)

        traj = model.forecast(dataset, horizon_minutes=10)
        self.assertEqual(len(traj.points), 10)
        values = [p.expected_events for p in traj.points]
        self.assertTrue(all(v >= 0 for v in values))

    def test_interpolation_helper_rate_conversion(self):
        """Verify _interpolate_to_minutes correctly converts rates."""
        from modeling.base_model import BaseModel

        class Dummy(BaseModel):
            MODEL_TYPE = "test"
            SUPPORTED_TASK_TYPES = {TaskType.EVENT_RATE}
            def fit(self, d): pass
            def predict_pointwise(self, d, i): pass
            def forecast(self, d, h, s="", r=None): pass
            def save(self, p): pass
            @classmethod
            def load(cls, p, t, m): pass

        m = Dummy(TaskType.EVENT_RATE, {}, {})
        # 120 events/hour should become ~2 events/minute
        native = np.array([120.0, 120.0])
        result = m._interpolate_to_minutes(native, 60.0, 60, is_cumulative_rate=True)
        self.assertEqual(len(result), 60)
        np.testing.assert_allclose(result, 2.0, atol=0.01)


class TestMetricsComputation(unittest.TestCase):
    """Tests for metric computation during training."""

    def test_metrics_registry_builds_all_metrics(self):
        """Test that all expected metrics can be built."""


        metric_defs = [
            "mae",
            "rmse",
            "mse",
            "mape",
            "smape",
            {"name": "quantile_loss", "params": {"quantile": 0.5}},
            {"name": "aoi_reward", "params": {"aoi_threshold_minutes": 5.0}},
        ]

        for defn in metric_defs:
            metric = build_metric(defn)
            self.assertIsNotNone(metric)

            y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
            y_pred = [1.1, 2.2, 2.9, 4.1, 5.0]
            value = metric(y_true, y_pred, context=None)
            self.assertIsInstance(value, float)

    def test_aoi_reward_norm_cross_resolution(self):
        """Test that resample_freq_minutes normalizes ER predictions to per-minute.

        Without normalization, a 1h model predicting 120 events/hour easily
        exceeds sync_threshold=50, while a 1min model predicting 2 events/min
        does not — even though they represent the same rate.  With normalization,
        both are compared as 2 events/min and get the same classification.
        """
        # Same underlying rate: 2 events/min
        y_true_1min = [2.0, 0.0, 3.0]
        y_pred_1min = [2.0, 0.0, 3.0]

        y_true_1h = [120.0, 0.0, 180.0]  # 2/min * 60, 0, 3/min * 60
        y_pred_1h = [120.0, 0.0, 180.0]

        # Unnormalized: threshold=2 at 1min vs 1h gives different results
        m_1min = build_metric({
            "name": "aoi_reward",
            "params": {"mode": "event_rate", "sync_threshold": 2,
                       "metric_name": "aoi_unnorm_1min"},
        })
        m_1h_unnorm = build_metric({
            "name": "aoi_reward",
            "params": {"mode": "event_rate", "sync_threshold": 2,
                       "metric_name": "aoi_unnorm_1h"},
        })
        score_1min = m_1min(y_true_1min, y_pred_1min)
        score_1h_unnorm = m_1h_unnorm(y_true_1h, y_pred_1h)
        # 1h values (120, 180) always exceed threshold=2 → different decision
        self.assertNotEqual(score_1min, score_1h_unnorm)

        # Normalized: both should produce the same result
        m_1min_norm = build_metric({
            "name": "aoi_reward",
            "params": {"mode": "event_rate", "sync_threshold": 2,
                       "resample_freq_minutes": 1,
                       "metric_name": "aoi_norm_1min"},
        })
        m_1h_norm = build_metric({
            "name": "aoi_reward",
            "params": {"mode": "event_rate", "sync_threshold": 2,
                       "resample_freq_minutes": 60,
                       "metric_name": "aoi_norm_1h"},
        })
        score_1min_norm = m_1min_norm(y_true_1min, y_pred_1min)
        score_1h_norm = m_1h_norm(y_true_1h, y_pred_1h)
        self.assertAlmostEqual(score_1min_norm, score_1h_norm, places=6)

    def test_metrics_handle_edge_cases(self):
        """Test metrics behavior with edge cases."""


        mae = build_metric("mae")

        value = mae([1.0], [1.0], context=None)
        self.assertEqual(value, 0.0)

        value = mae([0.0, 0.0], [0.0, 0.0], context=None)
        self.assertEqual(value, 0.0)


class TestZeroShotFastEval(unittest.TestCase):
    """Tests for zero-shot model fast evaluation (merged val+test, no CV)."""

    @classmethod
    def setUpClass(cls):
        cls._orig_mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME")
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "test_model_training"
        cls.sa = StorageAccess()
        cls.sync_unit = SyncUnit("test_org", "test_system", "test_project")
        cls._setup_test_data()

    @classmethod
    def _setup_test_data(cls):
        base_folder = (
            f"{cls.sync_unit.get_path_for_sync_unit()}/"
            f"{global_vars.DATA_PREPROCESSING_APP_FOLDER_NAME}"
        )
        cls.sa.create_folder(base_folder)

        from datetime import timedelta
        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        n_minutes = 100
        timestamps = [start + timedelta(minutes=i) for i in range(n_minutes)]

        np.random.seed(42)
        events = [max(0, 5.0 + np.random.normal(0, 1)) for _ in range(n_minutes)]

        df = pd.DataFrame({
            "minute": timestamps,
            "events_per_minute": events,
        })

        events_path = (
            f"{base_folder}/"
            f"{global_vars.MERGED_EVENTS_PER_MINUTE_FILE_NAME_FORMAT.format(int=1)}"
        )
        cls.sa.save_dataframe_as_picklefile(df, events_path)

    @classmethod
    def tearDownClass(cls):
        if cls._orig_mlflow_experiment is None:
            os.environ.pop("MLFLOW_EXPERIMENT_NAME", None)
        else:
            os.environ["MLFLOW_EXPERIMENT_NAME"] = cls._orig_mlflow_experiment
        test_data_path = cls.sa.create_full_path_string(
            cls.sync_unit.get_path_for_sync_unit()
        )
        models_path = os.path.join(test_data_path, "models")
        if os.path.exists(models_path):
            shutil.rmtree(models_path)

    def _train_with_zero_shot_mock(self, cv_folds=0):
        """Train ARIMA but patch fit() to return zero_shot=True, testing the fast-eval path."""
        from unittest.mock import patch

        app = ModelTrainingApp(sa=self.sa)

        parameters = {
            "task_type": "event_rate",
            "dataset": {"train_ratio": 0.7, "val_ratio": 0.15},
            "models": [
                {"model_type": "ARIMA", "template_name": "arima_small_v1"}
            ],
            "metrics": ["mae", "rmse"],
            "cv_folds": cv_folds,
        }

        from modeling.models.arima_model import ARIMAEventRateModel
        original_fit = ARIMAEventRateModel.fit

        def patched_fit(model_self, dataset):
            result = original_fit(model_self, dataset)
            result["zero_shot"] = True
            return result

        with patch.object(ARIMAEventRateModel, "fit", patched_fit):
            response = app.execute_action(
                self.sync_unit, "train_models", parameters
            )

        return response

    def test_zero_shot_merges_val_test_into_single_eval(self):
        """Zero-shot models should produce val_ metrics only (no test_ metrics)."""
        response = self._train_with_zero_shot_mock(cv_folds=0)

        self.assertTrue(response.success, f"Training failed: {response.message}")
        result = response.data["results"][0]
        self.assertTrue(result["success"])

        metrics = result["metrics"]
        val_keys = [k for k in metrics if k.startswith("val_")]
        test_keys = [k for k in metrics if k.startswith("test_")]

        self.assertTrue(len(val_keys) > 0, "Should have val_ metrics from merged eval")
        self.assertEqual(len(test_keys), 0, "Should have no test_ metrics (merged into val)")

    def test_zero_shot_skips_cv(self):
        """Zero-shot models should skip CV even when cv_folds >= 2."""
        response = self._train_with_zero_shot_mock(cv_folds=3)

        self.assertTrue(response.success, f"Training failed: {response.message}")
        result = response.data["results"][0]
        self.assertTrue(result["success"])
        self.assertIsNone(result["cv"], "CV should be skipped for zero-shot models")

    def test_zero_shot_metadata_has_no_cv(self):
        """Training metadata for zero-shot models should not contain CV results."""
        response = self._train_with_zero_shot_mock(cv_folds=3)

        result = response.data["results"][0]
        metadata_path = os.path.join(result["artifact_path"], "training_metadata.json")
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        self.assertNotIn("cv", metadata, "Metadata should not contain CV for zero-shot")
        self.assertTrue(metadata["training_info"].get("zero_shot"), "Should be marked zero-shot")

    def test_non_zero_shot_still_has_separate_val_test(self):
        """Non-zero-shot models should still produce separate val_ and test_ metrics."""
        app = ModelTrainingApp(sa=self.sa)

        parameters = {
            "task_type": "event_rate",
            "dataset": {"train_ratio": 0.7, "val_ratio": 0.15},
            "models": [
                {"model_type": "ARIMA", "template_name": "arima_small_v1"}
            ],
            "metrics": ["mae"],
        }

        response = app.execute_action(
            self.sync_unit, "train_models", parameters
        )

        self.assertTrue(response.success)
        result = response.data["results"][0]
        metrics = result["metrics"]

        val_keys = [k for k in metrics if k.startswith("val_")]
        test_keys = [k for k in metrics if k.startswith("test_")]

        self.assertTrue(len(val_keys) > 0, "Non-zero-shot should have val_ metrics")
        self.assertTrue(len(test_keys) > 0, "Non-zero-shot should have test_ metrics")

    def test_non_zero_shot_cv_still_works(self):
        """Non-zero-shot models should still run CV when requested."""
        app = ModelTrainingApp(sa=self.sa)

        parameters = {
            "task_type": "event_rate",
            "dataset": {"train_ratio": 0.7, "val_ratio": 0.15},
            "models": [
                {"model_type": "ARIMA", "template_name": "arima_small_v1"}
            ],
            "metrics": ["mae"],
            "cv_folds": 2,
        }

        response = app.execute_action(
            self.sync_unit, "train_models", parameters
        )

        self.assertTrue(response.success)
        result = response.data["results"][0]
        self.assertIsNotNone(result["cv"], "Non-zero-shot should run CV")
        self.assertEqual(result["cv"]["n_folds"], 2)

    def test_zero_shot_eval_indices_are_superset(self):
        """The merged eval should cover all val+test indices."""
        from unittest.mock import patch
        from modeling.models.arima_model import ARIMAEventRateModel

        app = ModelTrainingApp(sa=self.sa)

        parameters = {
            "task_type": "event_rate",
            "dataset": {"train_ratio": 0.7, "val_ratio": 0.15},
            "models": [
                {"model_type": "ARIMA", "template_name": "arima_small_v1"}
            ],
            "metrics": ["mae"],
        }

        captured_datasets = []
        original_evaluate = app._evaluate_metrics

        def capture_evaluate(training_ctx, model, dataset):
            captured_datasets.append({
                "val_indices": list(dataset.val_indices),
                "test_indices": list(dataset.test_indices),
            })
            return original_evaluate(training_ctx, model, dataset)

        original_fit = ARIMAEventRateModel.fit

        def patched_fit(model_self, dataset):
            result = original_fit(model_self, dataset)
            result["zero_shot"] = True
            return result

        with patch.object(ARIMAEventRateModel, "fit", patched_fit), \
             patch.object(app, "_evaluate_metrics", side_effect=capture_evaluate):
            response = app.execute_action(
                self.sync_unit, "train_models", parameters
            )

        self.assertTrue(response.success)
        self.assertEqual(len(captured_datasets), 1, "Should call _evaluate_metrics once")
        ds = captured_datasets[0]
        self.assertEqual(ds["test_indices"], [], "test_indices should be empty")
        self.assertTrue(len(ds["val_indices"]) > 0, "val_indices should contain merged indices")


if __name__ == "__main__":
    print("=" * 70)
    print("Model Training System - Proof of Concept Tests")
    print("=" * 70)
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestARIMAModelDirectly))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricsComputation))
    suite.addTests(loader.loadTestsFromTestCase(TestModelTrainingPipeline))

    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(suite)

    print()
    print("=" * 70)
    if test_result.wasSuccessful():
        print("All tests passed!")
    else:
        print(f"Tests failed: {len(test_result.failures)} failures, {len(test_result.errors)} errors")
    print("=" * 70)
