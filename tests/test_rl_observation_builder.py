"""Tests for rl_training.observation_builder module."""

import unittest

import numpy as np

from rl_training.observation_builder import (
    FEATURE_COVARIATES,
    FEATURE_CUMULATIVE_EMPTY_SYNCS,
    FEATURE_CURRENT_AOI,
    FEATURE_ITEMS_SYNCED_LAST,
    FEATURE_PREDICTIONS,
    FEATURE_TIME_SINCE_LAST_SYNC,
    FEATURE_WINDOWED_EMPTY_SYNCS,
    ObservationBuilder,
    ObservationSpec,
)


class TestObservationSpec(unittest.TestCase):

    def test_dimension_scalar_features_only(self):
        spec = ObservationSpec(
            features=[FEATURE_TIME_SINCE_LAST_SYNC, FEATURE_CURRENT_AOI],
        )
        self.assertEqual(spec.dimension, 2)

    def test_dimension_with_predictions(self):
        spec = ObservationSpec(
            features=[FEATURE_TIME_SINCE_LAST_SYNC, FEATURE_PREDICTIONS],
            prediction_horizon_steps=30,
        )
        self.assertEqual(spec.dimension, 31)  # 1 + 30

    def test_dimension_with_covariates(self):
        spec = ObservationSpec(
            features=[FEATURE_COVARIATES],
            covariate_names=["hour_of_day", "day_of_week", "is_weekend"],
        )
        self.assertEqual(spec.dimension, 3)

    def test_dimension_full(self):
        spec = ObservationSpec(
            features=[
                FEATURE_TIME_SINCE_LAST_SYNC,
                FEATURE_CURRENT_AOI,
                FEATURE_PREDICTIONS,
                FEATURE_COVARIATES,
                FEATURE_ITEMS_SYNCED_LAST,
                FEATURE_CUMULATIVE_EMPTY_SYNCS,
            ],
            prediction_horizon_steps=60,
            covariate_names=["hour_of_day", "day_of_week"],
        )
        # 1 + 1 + 60 + 2 + 1 + 1 = 66
        self.assertEqual(spec.dimension, 66)

    def test_feature_layout(self):
        spec = ObservationSpec(
            features=[FEATURE_TIME_SINCE_LAST_SYNC, FEATURE_PREDICTIONS],
            prediction_horizon_steps=10,
        )
        layout = spec.feature_layout()
        self.assertEqual(layout[FEATURE_TIME_SINCE_LAST_SYNC], (0, 1))
        self.assertEqual(layout[FEATURE_PREDICTIONS], (1, 11))


class TestObservationBuilder(unittest.TestCase):

    def test_build_shape(self):
        spec = ObservationSpec(
            features=[
                FEATURE_TIME_SINCE_LAST_SYNC,
                FEATURE_CURRENT_AOI,
                FEATURE_PREDICTIONS,
            ],
            prediction_horizon_steps=10,
        )
        builder = ObservationBuilder(spec)
        obs = builder.build(
            time_since_last_sync=5.0,
            current_aoi=3.0,
            predictions=[0.1] * 10,
            covariates={},
            items_synced_last=0.0,
            cumulative_empty_syncs=0,
        )
        self.assertEqual(obs.shape, (12,))
        self.assertEqual(obs.dtype, np.float32)

    def test_build_values_normalized(self):
        spec = ObservationSpec(
            features=[FEATURE_TIME_SINCE_LAST_SYNC, FEATURE_CURRENT_AOI],
        )
        builder = ObservationBuilder(spec)
        obs = builder.build(
            time_since_last_sync=720.0,  # Half of MAX_TIME (1440)
            current_aoi=360.0,
            predictions=[],
            covariates={},
            items_synced_last=0.0,
            cumulative_empty_syncs=0,
        )
        self.assertAlmostEqual(obs[0], 0.5, places=4)
        self.assertAlmostEqual(obs[1], 0.25, places=4)

    def test_build_clamps_high_values(self):
        spec = ObservationSpec(
            features=[FEATURE_TIME_SINCE_LAST_SYNC],
        )
        builder = ObservationBuilder(spec)
        obs = builder.build(
            time_since_last_sync=99999.0,  # Way over max
            current_aoi=0.0,
            predictions=[],
            covariates={},
            items_synced_last=0.0,
            cumulative_empty_syncs=0,
        )
        self.assertAlmostEqual(obs[0], 1.0)

    def test_predictions_padding(self):
        spec = ObservationSpec(
            features=[FEATURE_PREDICTIONS],
            prediction_horizon_steps=5,
        )
        builder = ObservationBuilder(spec)
        # Only 2 predictions provided, should pad to 5
        obs = builder.build(
            time_since_last_sync=0.0,
            current_aoi=0.0,
            predictions=[50.0, 50.0],  # 50/100 = 0.5 normalized
            covariates={},
            items_synced_last=0.0,
            cumulative_empty_syncs=0,
        )
        self.assertEqual(obs.shape, (5,))
        self.assertAlmostEqual(obs[0], 0.5)
        self.assertAlmostEqual(obs[1], 0.5)
        # Padded with last value (0.5)
        self.assertAlmostEqual(obs[2], 0.5)
        self.assertAlmostEqual(obs[3], 0.5)
        self.assertAlmostEqual(obs[4], 0.5)

    def test_covariates_normalized(self):
        spec = ObservationSpec(
            features=[FEATURE_COVARIATES],
            covariate_names=["hour_of_day", "day_of_week"],
        )
        builder = ObservationBuilder(spec)
        obs = builder.build(
            time_since_last_sync=0.0,
            current_aoi=0.0,
            predictions=[],
            covariates={"hour_of_day": 12.0, "day_of_week": 3.0},
            items_synced_last=0.0,
            cumulative_empty_syncs=0,
        )
        self.assertAlmostEqual(obs[0], 12.0 / 23.0, places=4)
        self.assertAlmostEqual(obs[1], 3.0 / 6.0, places=4)

    def test_covariates_sincos_normalized(self):
        spec = ObservationSpec(
            features=[FEATURE_COVARIATES],
            covariate_names=["sin_hour", "cos_hour"],
        )
        builder = ObservationBuilder(spec)
        obs = builder.build(
            time_since_last_sync=0.0,
            current_aoi=0.0,
            predictions=[],
            covariates={"sin_hour": -1.0, "cos_hour": 1.0},
            items_synced_last=0.0,
            cumulative_empty_syncs=0,
        )
        self.assertAlmostEqual(obs[0], 0.0)   # (-1+1)/2
        self.assertAlmostEqual(obs[1], 1.0)    # (1+1)/2

    def test_covariates_binary_unchanged(self):
        spec = ObservationSpec(
            features=[FEATURE_COVARIATES],
            covariate_names=["is_weekend", "is_business_hours"],
        )
        builder = ObservationBuilder(spec)
        obs = builder.build(
            time_since_last_sync=0.0,
            current_aoi=0.0,
            predictions=[],
            covariates={"is_weekend": 1.0, "is_business_hours": 0.0},
            items_synced_last=0.0,
            cumulative_empty_syncs=0,
        )
        self.assertAlmostEqual(obs[0], 1.0)
        self.assertAlmostEqual(obs[1], 0.0)

    def test_covariates_clamped_to_max(self):
        spec = ObservationSpec(
            features=[FEATURE_COVARIATES],
            covariate_names=["hour_of_day"],
        )
        builder = ObservationBuilder(spec)
        obs = builder.build(
            time_since_last_sync=0.0,
            current_aoi=0.0,
            predictions=[],
            covariates={"hour_of_day": 99.0},
            items_synced_last=0.0,
            cumulative_empty_syncs=0,
        )
        self.assertAlmostEqual(obs[0], 1.0)

    def test_all_values_in_0_1(self):
        """All features including covariates should be normalized to [0, 1]."""
        spec = ObservationSpec(
            features=[
                FEATURE_TIME_SINCE_LAST_SYNC,
                FEATURE_CURRENT_AOI,
                FEATURE_PREDICTIONS,
                FEATURE_COVARIATES,
                FEATURE_ITEMS_SYNCED_LAST,
                FEATURE_CUMULATIVE_EMPTY_SYNCS,
            ],
            prediction_horizon_steps=5,
            covariate_names=["hour_of_day", "day_of_week", "is_weekend"],
        )
        builder = ObservationBuilder(spec)
        obs = builder.build(
            time_since_last_sync=100.0,
            current_aoi=200.0,
            predictions=[10.0, 20.0, 30.0, 40.0, 50.0],
            covariates={"hour_of_day": 15.0, "day_of_week": 4.0, "is_weekend": 0.0},
            items_synced_last=500.0,
            cumulative_empty_syncs=720,
        )
        self.assertTrue(np.all(obs >= 0.0))
        self.assertTrue(np.all(obs <= 1.0))


class TestTTEPredictionNormalization(unittest.TestCase):

    def test_tte_predictions_inverted_normalization(self):
        """TTE predictions are inverted: 0min→1.0, 720min→0.5, 1440min→0.0."""
        spec = ObservationSpec(
            features=[FEATURE_PREDICTIONS],
            prediction_horizon_steps=3,
            prediction_task_type="time_to_event",
        )
        builder = ObservationBuilder(spec)
        obs = builder.build(
            time_since_last_sync=0.0,
            current_aoi=0.0,
            predictions=[0.0, 720.0, 1440.0],
            covariates={},
            items_synced_last=0.0,
            cumulative_empty_syncs=0,
        )
        self.assertAlmostEqual(obs[0], 1.0, places=4)
        self.assertAlmostEqual(obs[1], 0.5, places=4)
        self.assertAlmostEqual(obs[2], 0.0, places=4)

    def test_tte_predictions_clamp_above_1440(self):
        """TTE values above 1440 clamp to 1440, producing 0.0 after inversion."""
        spec = ObservationSpec(
            features=[FEATURE_PREDICTIONS],
            prediction_horizon_steps=2,
            prediction_task_type="time_to_event",
        )
        builder = ObservationBuilder(spec)
        obs = builder.build(
            time_since_last_sync=0.0,
            current_aoi=0.0,
            predictions=[2000.0, 5000.0],
            covariates={},
            items_synced_last=0.0,
            cumulative_empty_syncs=0,
        )
        self.assertAlmostEqual(obs[0], 0.0)
        self.assertAlmostEqual(obs[1], 0.0)

    def test_er_predictions_unchanged(self):
        """Default (ER) predictions still use MAX=100 without inversion."""
        spec = ObservationSpec(
            features=[FEATURE_PREDICTIONS],
            prediction_horizon_steps=2,
            prediction_task_type="event_rate",
        )
        builder = ObservationBuilder(spec)
        obs = builder.build(
            time_since_last_sync=0.0,
            current_aoi=0.0,
            predictions=[50.0, 200.0],
            covariates={},
            items_synced_last=0.0,
            cumulative_empty_syncs=0,
        )
        self.assertAlmostEqual(obs[0], 0.5)
        self.assertAlmostEqual(obs[1], 1.0)

    def test_default_prediction_task_type_is_event_rate(self):
        spec = ObservationSpec(features=[FEATURE_PREDICTIONS])
        self.assertEqual(spec.prediction_task_type, "event_rate")

    def test_tte_all_values_in_0_1(self):
        """TTE predictions should be normalized to [0, 1] range."""
        spec = ObservationSpec(
            features=[
                FEATURE_TIME_SINCE_LAST_SYNC,
                FEATURE_CURRENT_AOI,
                FEATURE_PREDICTIONS,
                FEATURE_ITEMS_SYNCED_LAST,
            ],
            prediction_horizon_steps=5,
            prediction_task_type="time_to_event",
        )
        builder = ObservationBuilder(spec)
        obs = builder.build(
            time_since_last_sync=100.0,
            current_aoi=200.0,
            predictions=[0.0, 100.0, 500.0, 1000.0, 2000.0],
            covariates={},
            items_synced_last=50.0,
            cumulative_empty_syncs=0,
        )
        self.assertTrue(np.all(obs >= 0.0))
        self.assertTrue(np.all(obs <= 1.0))


if __name__ == "__main__":
    unittest.main()
