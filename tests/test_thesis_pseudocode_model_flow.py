"""Characterize training/serving control flow without fitting or loading models."""

# These tests deliberately exercise the helpers documented in the appendix.
# pylint: disable=protected-access

from datetime import datetime, timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from modeling.core_types import (
    DataSnapshotMetadata, EventRateDataset, PredictionPoint,
    PredictionTrajectory, TaskType,
)
from modeling.models.chronos_model import ChronosForecastModel
from model_serving.model_serving_app import ModelServingApp
from model_training.model_training_app import ModelTrainingApp
from sync_unit import SyncUnit


START = datetime(2025, 1, 1)
UNIT = SyncUnit("test_org", "test_system", "test_project")
ER = TaskType.EVENT_RATE


def dataset():
    """Return twelve minute-level samples with three chronological splits."""
    return EventRateDataset(
        [START + timedelta(minutes=i) for i in range(12)], list(range(12)),
        train_indices=list(range(6)), val_indices=[6, 7, 8], test_indices=[9, 10, 11],
    )


def trajectory(value=2.0):
    """Return a two-point forecast without invoking a real model."""
    return PredictionTrajectory(UNIT.get_name(), ER, 2, [
        PredictionPoint(START + timedelta(minutes=i), value) for i in (1, 2)
    ], "fixture")


class TestTrainingExecutionOrder(unittest.TestCase):
    """Keep the main fit, optional CV and zero-shot paths distinct."""

    def test_main_model_evaluated_before_cv_and_saved_afterwards(self):
        """Fold models never replace the saved main model."""
        app = ModelTrainingApp(sa=MagicMock())
        data = dataset()
        meta = DataSnapshotMetadata(None, ER, START, data.timestamps[-1],
                                    data.train_indices, data.val_indices, data.test_indices)
        events = []
        models = []

        def create(_context):
            name = f"model{len(models)}"
            model = MagicMock()
            model.fit.side_effect = lambda d: (
                events.append((name, "fit", list(d.train_indices))) or {})
            model.save.side_effect = lambda path: events.append((name, "save"))
            models.append(model)
            return model

        def evaluate(_context, model, d):
            name = f"model{models.index(model)}"
            events.append((name, "eval", list(d.val_indices), list(d.test_indices)))
            return {"val_mae": float(models.index(model))}

        tracker = MagicMock()
        tracker.__enter__.return_value = tracker
        tracker.log_model_artifacts.side_effect = lambda path: events.append(("track", "artifacts"))
        tracker.log_snapshot_metadata.side_effect = lambda: events.append(("track", "snapshot"))
        with patch("model_training.model_training_app.create_model", side_effect=create), \
                patch("model_training.model_training_app.MLflowTracker", return_value=tracker), \
                patch.object(app, "_evaluate_metrics", side_effect=evaluate), \
                patch.object(app, "_get_artifact_dir", return_value="fixture"), \
                patch.object(app, "_save_metadata",
                             side_effect=lambda *a, **k: events.append(("save", "metadata"))):
            result = app._train_single_model(UNIT, ER, data, meta, "ARIMA", None, [], cv_folds=2)
        self.assertEqual(events, [
            ("model0", "fit", list(range(6))), ("model0", "eval", [6, 7, 8], [9, 10, 11]),
            ("model1", "fit", [0, 1, 2]), ("model1", "eval", [3, 4, 5], []),
            ("model2", "fit", list(range(6))), ("model2", "eval", [6, 7, 8], []),
            ("model0", "save"), ("track", "artifacts"), ("track", "snapshot"), ("save", "metadata"),
        ])
        self.assertEqual(result["cv"]["combined_metrics"]["cv_mean_val_mae"], 1.5)
        self.assertEqual(data.test_indices, [9, 10, 11])
        for fold_model in models[1:]:
            fold_model.save.assert_not_called()

    def test_chronos_fit_keeps_context_without_pipeline_prediction_or_training(self):
        """Chronos fit loads its pipeline and stores only the training context."""
        model = ChronosForecastModel(ER, {}, {})
        with patch.object(model, "_load_pipeline") as load:
            info = model.fit(dataset())
        load.assert_called_once_with()
        self.assertTrue(info["zero_shot"])
        np.testing.assert_array_equal(model._context, np.arange(6))

    def test_zero_shot_merges_eval_and_never_calls_cv(self):
        """Zero-shot evaluation merges val/test without changing the input."""
        app = ModelTrainingApp(sa=MagicMock())
        data = dataset()
        meta = DataSnapshotMetadata(None, ER, START, data.timestamps[-1],
                                    data.train_indices, data.val_indices, data.test_indices)
        model = MagicMock()
        model.fit.return_value = {"zero_shot": True}
        with patch("model_training.model_training_app.create_model", return_value=model), \
                patch("model_training.model_training_app.MLflowTracker"), \
                patch.object(app, "_evaluate_metrics", return_value={}) as evaluate, \
                patch.object(app, "_run_cross_validation") as cv, \
                patch.object(app, "_get_artifact_dir", return_value="fixture"), \
                patch.object(app, "_save_metadata"):
            app._train_single_model(UNIT, ER, data, meta, "Chronos", None, [], cv_folds=3)
        evaluated = evaluate.call_args.args[2]
        self.assertEqual(evaluated.val_indices, list(range(6, 12)))
        self.assertEqual(evaluated.test_indices, [])
        self.assertEqual(data.test_indices, [9, 10, 11])
        cv.assert_not_called()


class TestServingSelectionBoundaries(unittest.TestCase):
    """Characterize per-mode history, origin, fallback and blending behavior."""

    def setUp(self):
        """Create a serving app with no real storage access."""
        self.app = ModelServingApp(sa=MagicMock())

    def test_single_model_forwards_origin_covariates_and_frequency(self):
        """The single-model path forwards all three history constraints."""
        model = SimpleNamespace(_covariate_names=["hour_of_day"], _freq="5min",
                                forecast=MagicMock(return_value=trajectory()))
        origin = START + timedelta(minutes=6)
        history = dataset()
        with patch.object(self.app, "_load_model_for_sync_unit",
                          return_value=(model, {"model_id": "chosen"})), \
                patch.object(self.app, "_build_history_dataset", return_value=history) as build:
            result = self.app._handle_get_prediction(UNIT, {
                "model_selection": {"mode": "explicit", "model_id": "fixture"},
                "reference_time": origin.isoformat(), "horizon_minutes": 2,
            })
        self.assertTrue(result.success)
        build.assert_called_once_with(UNIT, ER, covariates=["hour_of_day"],
                                      reference_time=origin, resample_freq="5min")
        model.forecast.assert_called_once_with(
            history, horizon_minutes=2, sync_unit_name=UNIT.get_name(), reference_time=origin)
        self.assertEqual(result.data["prediction_result"].prediction.model_id, "chosen")

    def test_heuristic_origin_not_forwarded_and_missing_group_uses_mean_of_means(self):
        """The heuristic ignores the origin and falls back to group means."""
        history = EventRateDataset(
            [START, START + timedelta(days=1), START + timedelta(days=1, minutes=1)],
            [10, 2, 2],
        )
        with patch.object(self.app, "_build_history_dataset", return_value=history) as build:
            result = self.app._handle_get_prediction(UNIT, {
                "model_selection": {"mode": "heuristic", "granularity": ["minute_of_hour"]},
                "reference_time": START.isoformat(), "horizon_minutes": 1,
            })
        build.assert_called_once_with(UNIT, ER)
        point = result.data["prediction_result"].prediction.points[0]
        self.assertEqual(point.timestamp, history.timestamps[-1] + timedelta(minutes=1))
        self.assertEqual(point.expected_events, 4)  # mean(group means 6 and 2), not 14/3

    def test_multi_uses_successful_models_and_unresampled_history_without_origin(self):
        """Multi tolerates partial load failures and uses shared raw history."""
        model = SimpleNamespace(_covariate_names=["hour_of_day"], _freq="5min",
                                forecast=MagicMock(return_value=trajectory()))
        with patch.object(self.app, "_load_model_for_sync_unit",
                          side_effect=[OSError("fixture failure"), (model, {"model_id": "ok"})]), \
                patch.object(self.app, "_build_history_dataset", return_value=dataset()) as build:
            result = self.app._handle_get_prediction(UNIT, {
                "reference_time": START, "horizon_minutes": 2,
                "model_selection": {"mode": "multi", "models": [
                    {"mode": "explicit", "model_id": "bad"},
                    {"mode": "explicit", "model_id": "ok"}]},
            })
        self.assertTrue(result.success)
        build.assert_called_once_with(UNIT, ER, covariates=["hour_of_day"])
        self.assertNotIn("reference_time", model.forecast.call_args.kwargs)

    def test_blending_uses_position_not_timestamp_join_and_always_mean(self):
        """Blending averages by index without checking timestamp alignment."""
        first, second = trajectory(2), trajectory(6)
        second.points[0].timestamp += timedelta(days=1)
        blended = self.app._blend_trajectories([first, second], UNIT.get_name(), ER, 2, "unused")
        self.assertEqual([p.expected_events for p in blended.points], [4, 4])
        self.assertEqual(blended.points[0].timestamp, first.points[0].timestamp)

    def test_stage_fallback_chooses_newest_metadata_without_retry(self):
        """Failure of the newest local candidate does not try older models."""
        with patch("model_serving.model_serving_app.os.path.exists", return_value=True), \
                patch("model_serving.model_serving_app.os.path.isdir", return_value=True), \
                patch("model_serving.model_serving_app.os.listdir",
                      return_value=["older", "newer"]), \
                patch("model_serving.model_serving_app.os.path.getmtime", side_effect=[1, 2]), \
                patch("model_serving.model_serving_app.is_mlflow_available", return_value=False), \
                patch.object(self.app, "_load_from_artifact_path",
                             side_effect=ValueError("wrong task")) as load:
            with self.assertRaisesRegex(ValueError, "wrong task"):
                self.app._load_model_for_sync_unit(UNIT, ER, {"mode": "stage"})
        load.assert_called_once()
        self.assertTrue(load.call_args.args[0].endswith("/newer"))


if __name__ == "__main__":
    unittest.main()
