"""Tests for core modeling types used by training and serving code."""

from datetime import UTC, datetime, timedelta
import unittest

from modeling.core_types import (
    TaskType,
    PredictionPoint,
    PredictionTrajectory,
    PredictionResult,
    MaeMetric,
)


class TestPredictionTypes(unittest.TestCase):
    def test_prediction_result_ok_to_dict(self):
        now = datetime.now(UTC)
        pt = PredictionPoint(timestamp=now, expected_events=1.0)
        traj = PredictionTrajectory(
            sync_unit_name="org/sys/proj",
            task_type=TaskType.EVENT_RATE,
            horizon_minutes=5,
            points=[pt],
            model_id="model-1",
            created_at=now,
        )
        result = PredictionResult(ok=True, prediction=traj)
        data = result.to_dict()

        self.assertTrue(data["ok"])
        self.assertIn("prediction", data)
        self.assertEqual(data["prediction"]["task_type"], TaskType.EVENT_RATE.value)
        self.assertEqual(len(data["prediction"]["points"]), 1)

    def test_prediction_result_error_to_dict(self):
        result = PredictionResult(
            ok=False,
            prediction=None,
            error_type="model_error",
            error_message="something went wrong",
        )
        data = result.to_dict()

        self.assertFalse(data["ok"])
        self.assertEqual(data["error_type"], "model_error")
        self.assertIn("error_message", data)


class TestMaeMetric(unittest.TestCase):
    def test_mae_metric_basic(self):
        metric = MaeMetric()
        y_true = [0.0, 2.0, 4.0]
        y_pred = [0.0, 0.0, 4.0]

        value = metric(y_true, y_pred, context=None)

        # |0-0| + |2-0| + |4-4| = 2, /3 = 0.666...
        self.assertAlmostEqual(value, 2.0 / 3.0, places=6)

    def test_mae_metric_raises_on_length_mismatch(self):
        metric = MaeMetric()
        with self.assertRaises(ValueError):
            metric([1.0, 2.0], [1.0], context=None)


if __name__ == "__main__":
    unittest.main()
