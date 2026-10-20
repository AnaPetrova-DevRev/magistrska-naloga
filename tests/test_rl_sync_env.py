"""Tests for rl_training.sync_env module."""

import unittest
from datetime import datetime, timedelta

import numpy as np

from rl_training.prediction_provider import (
    ActionAwarePredictionProvider,
    PrecomputedPredictionProvider,
)
from rl_training.sync_env import SyncSchedulingEnv


def _make_env_config(
    n_steps=100,
    episode_length=None,
    random_start=False,
    task_type="event_rate",
    observation_features=None,
    prediction_horizon_steps=10,
    reward_config=None,
):
    """Create a minimal env config with synthetic data."""
    base_time = datetime(2025, 1, 1)
    timestamps = [base_time + timedelta(minutes=i) for i in range(n_steps)]

    # Synthetic events: every 10th minute has events
    actual_values = [0.0] * n_steps
    for i in range(0, n_steps, 10):
        actual_values[i] = 3.0

    predictions = np.random.rand(n_steps, prediction_horizon_steps).astype(np.float32)
    provider = PrecomputedPredictionProvider(timestamps, predictions)

    covariates = {
        "hour_of_day": [float(ts.hour) for ts in timestamps],
        "is_weekend": [1.0 if ts.weekday() >= 5 else 0.0 for ts in timestamps],
    }

    if observation_features is None:
        observation_features = [
            "time_since_last_sync",
            "current_aoi",
            "predictions",
            "covariates",
            "items_synced_last",
            "cumulative_empty_syncs",
        ]

    config = {
        "timestamps": timestamps,
        "actual_values": actual_values,
        "covariates": covariates,
        "prediction_provider": provider,
        "task_type": task_type,
        "episode_length": episode_length,
        "random_start": random_start,
        "prediction_horizon_steps": prediction_horizon_steps,
        "observation_features": observation_features,
        "reward_config": reward_config or {},
    }
    return config


class _RevealedSumForecaster:
    def forecast(
        self, observed_timestamps, observed_values,
        synthetic_timestamps, synthetic_values, first_target, horizon,
    ):
        return [sum(observed_values)] * horizon


def _make_action_aware_env(actual_values):
    n_steps = len(actual_values)
    config = _make_env_config(
        n_steps=n_steps,
        observation_features=["predictions"],
        prediction_horizon_steps=2,
    )
    config["actual_values"] = list(actual_values)
    config["prediction_provider"] = ActionAwarePredictionProvider(
        _RevealedSumForecaster(), trajectory_horizon_steps=2,
    )
    return SyncSchedulingEnv(config)


class TestSyncEnvInit(unittest.TestCase):

    def test_action_space(self):
        env = SyncSchedulingEnv(_make_env_config())
        self.assertEqual(env.action_space.n, 2)

    def test_observation_space_shape(self):
        env = SyncSchedulingEnv(_make_env_config(prediction_horizon_steps=10))
        # 1 + 1 + 10 + 2 (covariates) + 1 + 1 = 16
        self.assertEqual(env.observation_space.shape, (16,))

    def test_observation_space_bounds(self):
        env = SyncSchedulingEnv(_make_env_config())
        self.assertTrue(np.all(env.observation_space.low == 0.0))
        self.assertTrue(np.all(env.observation_space.high == 1.0))


class TestSyncEnvReset(unittest.TestCase):

    def test_reset_returns_obs_and_info(self):
        env = SyncSchedulingEnv(_make_env_config())
        obs, info = env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(info, dict)
        self.assertEqual(obs.shape, env.observation_space.shape)

    def test_reset_initial_state(self):
        env = SyncSchedulingEnv(_make_env_config())
        obs, info = env.reset()
        self.assertEqual(info["time_since_last_sync"], 0.0)
        self.assertEqual(info["current_aoi"], 0.0)
        self.assertEqual(info["cumulative_empty_syncs"], 0)

    def test_reset_with_seed(self):
        env = SyncSchedulingEnv(_make_env_config(episode_length=50, random_start=True))
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_reset_cycles_through_deterministic_episode_starts(self):
        config = _make_env_config(n_steps=100, episode_length=20)
        config["episode_start_indices"] = [0, 40, 80]
        env = SyncSchedulingEnv(config)
        starts = []
        for _ in range(4):
            env.reset()
            starts.append(env._episode_start)
        self.assertEqual(starts, [0, 40, 80, 0])

    def test_deterministic_episode_start_must_fit(self):
        config = _make_env_config(n_steps=100, episode_length=20)
        config["episode_start_indices"] = [81]
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            SyncSchedulingEnv(config)


class TestSyncEnvStep(unittest.TestCase):

    def test_step_returns_5_tuple(self):
        env = SyncSchedulingEnv(_make_env_config())
        env.reset()
        result = env.step(0)
        self.assertEqual(len(result), 5)
        obs, reward, terminated, truncated, info = result
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_no_sync_increases_aoi(self):
        env = SyncSchedulingEnv(_make_env_config())
        env.reset()
        _, _, _, _, info = env.step(0)
        self.assertEqual(info["current_aoi"], 1.0)
        _, _, _, _, info = env.step(0)
        self.assertEqual(info["current_aoi"], 2.0)

    def test_sync_resets_time_since_last_sync(self):
        env = SyncSchedulingEnv(_make_env_config())
        env.reset()
        env.step(0)
        env.step(0)
        _, _, _, _, info = env.step(1)  # Sync
        self.assertEqual(info["time_since_last_sync"], 0.0)

    def test_sync_with_events_resets_aoi(self):
        """Events at step 0 → sync at step 0 resets AoI."""
        config = _make_env_config()
        config["actual_values"][0] = 5.0  # Events at step 0
        env = SyncSchedulingEnv(config)
        env.reset()
        _, _, _, _, info = env.step(1)  # Sync at step 0 (has events)
        self.assertEqual(info["current_aoi"], 0.0)

    def test_sync_without_events_increments_empty_syncs(self):
        """Empty sync increments counter."""
        config = _make_env_config()
        config["actual_values"][0] = 0.0  # No events at step 0
        env = SyncSchedulingEnv(config)
        env.reset()
        _, _, _, _, info = env.step(1)  # Sync at step 0 (no events)
        self.assertEqual(info["cumulative_empty_syncs"], 1)

    def test_sync_without_events_still_resets_aoi(self):
        """AoI resets on every sync — data is up-to-date even if no new events."""
        config = _make_env_config()
        config["actual_values"][:3] = [0.0, 0.0, 0.0]  # No events
        env = SyncSchedulingEnv(config)
        env.reset()
        env.step(0)  # skip → AoI=1
        env.step(0)  # skip → AoI=2
        _, _, _, _, info = env.step(1)  # sync (empty) → AoI should reset
        self.assertEqual(info["current_aoi"], 0.0)

    def test_terminates_at_end_of_data(self):
        env = SyncSchedulingEnv(_make_env_config(n_steps=5))
        env.reset()
        for i in range(4):
            _, _, terminated, _, _ = env.step(0)
            self.assertFalse(terminated)
        _, _, terminated, _, _ = env.step(0)
        self.assertTrue(terminated)

    def test_truncates_at_episode_length(self):
        env = SyncSchedulingEnv(_make_env_config(n_steps=100, episode_length=10))
        env.reset()
        for i in range(9):
            _, _, _, truncated, _ = env.step(0)
            self.assertFalse(truncated)
        _, _, _, truncated, _ = env.step(0)
        self.assertTrue(truncated)

    def test_episode_reward_accumulates(self):
        env = SyncSchedulingEnv(_make_env_config())
        env.reset()
        total = 0.0
        for _ in range(10):
            _, reward, _, _, info = env.step(0)
            total += reward
        self.assertAlmostEqual(info["episode_reward"], total)

    def test_hidden_event_does_not_change_forecast_before_sync(self):
        without_event = _make_action_aware_env([0.0, 0.0, 0.0])
        with_hidden_event = _make_action_aware_env([9.0, 0.0, 0.0])
        obs_a, _ = without_event.reset()
        obs_b, _ = with_hidden_event.reset()
        np.testing.assert_array_equal(obs_a, obs_b)
        obs_a, *_ = without_event.step(0)
        obs_b, *_ = with_hidden_event.step(0)
        np.testing.assert_array_equal(obs_a, obs_b)

    def test_sync_revelation_affects_only_next_observation(self):
        without_event = _make_action_aware_env([0.0, 0.0, 0.0])
        with_event = _make_action_aware_env([9.0, 0.0, 0.0])
        initial_a, _ = without_event.reset()
        initial_b, _ = with_event.reset()
        np.testing.assert_array_equal(initial_a, initial_b)
        next_a, *_ = without_event.step(1)
        next_b, *_ = with_event.step(1)
        self.assertFalse(np.array_equal(next_a, next_b))


class TestSyncEnvRewardIntegration(unittest.TestCase):

    def test_reward_uses_reset_aoi_on_sync_and_preincrement_aoi_on_skip(self):
        """Lock the training objective's asymmetric reset/increment timing."""
        penalties = {
            "linear": [0.0, 0.0, -0.02],
            "quadratic": [0.0, -0.02, -0.08],
            "linear_no_threshold": [0.0, -0.02, -0.04],
        }
        for mode, skip_penalties in penalties.items():
            for reward_mode in ("full", "aoi_only"):
                for event_step in (None, 0, 3):
                    with self.subTest(mode=mode, reward_mode=reward_mode,
                                      event_step=event_step):
                        config = _make_env_config(n_steps=6, reward_config={
                            "aoi_mode": mode,
                            "reward_mode": reward_mode,
                            "aoi_threshold_minutes": 1.0,
                            "aoi_penalty_per_minute": 0.02,
                            "item_reward_per_unit": 0.1,
                        })
                        config["actual_values"] = [0.0] * 6
                        if event_step is not None:
                            config["actual_values"][event_step] = 5.0
                        env = SyncSchedulingEnv(config)
                        self.addCleanup(env.close)
                        env.reset(seed=7)
                        for step, expected in enumerate(skip_penalties):
                            _, reward, _, _, info = env.step(0)
                            parts = info["reward_components"]
                            self.assertAlmostEqual(parts["aoi_penalty"], expected)
                            self.assertAlmostEqual(parts["total_reward"], reward)
                            self.assertEqual(info["current_aoi"], step + 1)
                        _, reward, _, _, info = env.step(1)
                        parts = info["reward_components"]
                        self.assertEqual(parts["aoi_penalty"], 0.0)
                        self.assertEqual(info["current_aoi"], 0.0)
                        self.assertEqual(info["time_since_last_sync"], 0.0)
                        self.assertAlmostEqual(parts["total_reward"], reward)
                        self.assertAlmostEqual(
                            reward, parts["discrete_reward"] + parts["item_bonus"],
                        )
                        _, _, _, _, info = env.step(0)
                        self.assertEqual(info["reward_components"]["aoi_penalty"], 0.0)
                        self.assertEqual(info["current_aoi"], 1.0)

    def test_sync_with_events_gives_positive_relative_reward(self):
        """Sync with accumulated events should reward more than empty sync."""
        config = _make_env_config(n_steps=10)
        config["actual_values"] = [0.0] * 10
        config["actual_values"][0] = 5.0  # Events at step 0

        env = SyncSchedulingEnv(config)
        env.reset()
        _, r_event, _, _, _ = env.step(1)  # Sync — picked up 5.0 events

        # All-zero data: sync finds nothing accumulated
        config2 = _make_env_config(n_steps=10)
        config2["actual_values"] = [0.0] * 10

        env2 = SyncSchedulingEnv(config2)
        env2.reset()
        _, r_empty, _, _, _ = env2.step(1)  # Sync — 0 accumulated events

        self.assertGreater(r_event, r_empty)


class TestSyncEnvInfoContents(unittest.TestCase):

    def test_step_info_has_action_and_actual(self):
        env = SyncSchedulingEnv(_make_env_config())
        env.reset()
        _, _, _, _, info = env.step(1)
        self.assertIn("action", info)
        self.assertIn("actual_value", info)
        self.assertIn("reward_components", info)
        self.assertEqual(info["action"], 1)

    def test_reward_components_in_info(self):
        config = _make_env_config(reward_config={
            "aoi_mode": "linear_no_threshold",
            "aoi_penalty_per_minute": 1.0,
        })
        env = SyncSchedulingEnv(config)
        env.reset()
        _, reward, _, _, info = env.step(0)
        components = info["reward_components"]
        self.assertIn("discrete_reward", components)
        self.assertIn("aoi_penalty", components)
        self.assertIn("total_reward", components)
        self.assertEqual(components["aoi_penalty"], 0.0)
        self.assertAlmostEqual(components["total_reward"], reward)


class TestSyncEnvMinimalFeatures(unittest.TestCase):
    """Test with minimal observation features to ensure configurability."""

    def test_minimal_obs(self):
        config = _make_env_config(
            observation_features=["time_since_last_sync", "current_aoi"],
            prediction_horizon_steps=10,
        )
        env = SyncSchedulingEnv(config)
        self.assertEqual(env.observation_space.shape, (2,))
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (2,))


class TestPredictionTaskType(unittest.TestCase):

    def test_tte_prediction_task_type_flows_to_obs_spec(self):
        config = _make_env_config()
        config["prediction_task_type"] = "time_to_event"
        env = SyncSchedulingEnv(config)
        self.assertEqual(env.obs_spec.prediction_task_type, "time_to_event")

    def test_default_prediction_task_type_is_event_rate(self):
        config = _make_env_config()
        env = SyncSchedulingEnv(config)
        self.assertEqual(env.obs_spec.prediction_task_type, "event_rate")


if __name__ == "__main__":
    unittest.main()
