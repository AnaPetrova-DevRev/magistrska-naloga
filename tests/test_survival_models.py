"""Tests for Kaplan-Meier and Cox PH survival analysis models."""

import shutil
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np

from modeling.core_types import EventRateDataset, TaskType, TimeToEventDataset
from modeling.models.kaplan_meier_model import KaplanMeierTimeToEventModel
from modeling.models.cox_ph_model import CoxPHTimeToEventModel

try:
    import lifelines
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False


def _make_tte_dataset(n=200, seed=42, with_covariates=False):
    rng = np.random.default_rng(seed)
    deltas = rng.lognormal(mean=3.0, sigma=0.5, size=n).tolist()
    start = datetime(2024, 1, 1, 8, 0, 0)
    timestamps = [start + timedelta(minutes=i * 30) for i in range(n)]

    covariates = {}
    if with_covariates:
        covariates = {
            "hour_of_day": [float(ts.hour) for ts in timestamps],
            "day_of_week": [float(ts.weekday()) for ts in timestamps],
        }

    return TimeToEventDataset(
        timestamps=timestamps,
        minutes_to_next_event=deltas,
        covariates=covariates,
        train_indices=list(range(0, 140)),
        val_indices=list(range(140, 170)),
        test_indices=list(range(170, 200)),
    )


@unittest.skipUnless(LIFELINES_AVAILABLE, "lifelines not installed")
class TestKaplanMeierModel(unittest.TestCase):

    def test_fit_returns_expected_keys(self):
        model = KaplanMeierTimeToEventModel(
            task_type=TaskType.TIME_TO_EVENT, model_config={}, training_config={},
        )
        dataset = _make_tte_dataset()
        info = model.fit(dataset)
        self.assertIn("median_survival_time", info)
        self.assertIn("mean_delta", info)
        self.assertIn("n_train", info)
        self.assertTrue(model.is_fitted)

    def test_predict_pointwise(self):
        model = KaplanMeierTimeToEventModel(
            task_type=TaskType.TIME_TO_EVENT, model_config={}, training_config={},
        )
        dataset = _make_tte_dataset()
        model.fit(dataset)
        preds = model.predict_pointwise(dataset, dataset.val_indices)
        self.assertEqual(len(preds), len(dataset.val_indices))
        for p in preds:
            self.assertGreater(p, 0)

    def test_forecast(self):
        model = KaplanMeierTimeToEventModel(
            task_type=TaskType.TIME_TO_EVENT, model_config={}, training_config={},
        )
        dataset = _make_tte_dataset()
        model.fit(dataset)
        traj = model.forecast(dataset, horizon_minutes=5)
        self.assertEqual(len(traj.points), 5)

    def test_save_and_load(self):
        model = KaplanMeierTimeToEventModel(
            task_type=TaskType.TIME_TO_EVENT, model_config={}, training_config={},
        )
        dataset = _make_tte_dataset()
        model.fit(dataset)

        tmpdir = tempfile.mkdtemp()
        try:
            model.save(tmpdir)
            loaded = KaplanMeierTimeToEventModel.load(
                tmpdir, TaskType.TIME_TO_EVENT, {},
            )
            self.assertTrue(loaded.is_fitted)
            np.testing.assert_allclose(
                model.predict_pointwise(dataset, [140]),
                loaded.predict_pointwise(dataset, [140]),
            )
        finally:
            shutil.rmtree(tmpdir)

    def test_rejects_event_rate_task(self):
        with self.assertRaises(ValueError):
            KaplanMeierTimeToEventModel(
                task_type=TaskType.EVENT_RATE, model_config={}, training_config={},
            )


@unittest.skipUnless(LIFELINES_AVAILABLE, "lifelines not installed")
class TestCoxPHModel(unittest.TestCase):

    def test_fit_without_covariates(self):
        model = CoxPHTimeToEventModel(
            task_type=TaskType.TIME_TO_EVENT,
            model_config={"penalizer": 0.01},
            training_config={},
        )
        dataset = _make_tte_dataset(with_covariates=False)
        info = model.fit(dataset)
        self.assertIn("median_survival_time", info)
        self.assertEqual(info["n_covariates"], 0)
        self.assertTrue(model.is_fitted)

    def test_fit_with_covariates(self):
        model = CoxPHTimeToEventModel(
            task_type=TaskType.TIME_TO_EVENT,
            model_config={"penalizer": 0.01},
            training_config={},
        )
        dataset = _make_tte_dataset(with_covariates=True)
        info = model.fit(dataset)
        self.assertEqual(info["n_covariates"], 2)
        self.assertIn("concordance_index", info)
        self.assertIn("coefficients", info)

    def test_predict_pointwise(self):
        model = CoxPHTimeToEventModel(
            task_type=TaskType.TIME_TO_EVENT,
            model_config={"penalizer": 0.01},
            training_config={},
        )
        dataset = _make_tte_dataset(with_covariates=True)
        model.fit(dataset)
        preds = model.predict_pointwise(dataset, dataset.val_indices)
        self.assertEqual(len(preds), len(dataset.val_indices))

    def test_save_and_load(self):
        model = CoxPHTimeToEventModel(
            task_type=TaskType.TIME_TO_EVENT,
            model_config={"penalizer": 0.01},
            training_config={},
        )
        dataset = _make_tte_dataset(with_covariates=True)
        model.fit(dataset)

        tmpdir = tempfile.mkdtemp()
        try:
            model.save(tmpdir)
            loaded = CoxPHTimeToEventModel.load(
                tmpdir, TaskType.TIME_TO_EVENT, {"penalizer": 0.01},
            )
            self.assertTrue(loaded.is_fitted)
            np.testing.assert_allclose(
                model.predict_pointwise(dataset, [140]),
                loaded.predict_pointwise(dataset, [140]),
            )
        finally:
            shutil.rmtree(tmpdir)

    def test_rejects_event_rate_task(self):
        with self.assertRaises(ValueError):
            CoxPHTimeToEventModel(
                task_type=TaskType.EVENT_RATE, model_config={}, training_config={},
            )


if __name__ == "__main__":
    unittest.main()
