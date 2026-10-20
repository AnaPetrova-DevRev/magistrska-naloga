"""Integration tests for the RL training and evaluation pipeline.

These tests exercise end-to-end flows with tiny data and minimal
timesteps. They require stable_baselines3 to be installed.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, UTC
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

try:
    import stable_baselines3
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from rl_training.prediction_provider import (
    ActionAwarePredictionProvider,
    MaskedPredictionProvider,
    NoPredictionProvider,
    PrecomputedPredictionProvider,
)
from rl_training.sync_env import SyncSchedulingEnv
from sync_policy_evaluation.evaluation_engine import run_evaluation
from sync_policy_evaluation.policy_loader import (
    AlwaysSyncPolicy,
    PeriodicSyncPolicy,
)


def _make_synthetic_data(n_steps=200, event_interval=10):
    """Create synthetic minute-level data for integration tests."""
    base_time = datetime(2025, 1, 1, tzinfo=UTC)
    timestamps = [base_time + timedelta(minutes=i) for i in range(n_steps)]

    actual_values = [0.0] * n_steps
    for i in range(0, n_steps, event_interval):
        actual_values[i] = 3.0

    covariates = {
        "hour_of_day": [float(ts.hour) for ts in timestamps],
        "is_weekend": [1.0 if ts.weekday() >= 5 else 0.0 for ts in timestamps],
    }

    predictions = np.random.rand(n_steps, 10).astype(np.float32)
    provider = PrecomputedPredictionProvider(timestamps, predictions)

    return timestamps, actual_values, covariates, provider


def _make_env_config(n_steps=200, **overrides):
    """Create an env config dict from synthetic data."""
    timestamps, actual_values, covariates, provider = _make_synthetic_data(n_steps)
    config = {
        "timestamps": timestamps,
        "actual_values": actual_values,
        "covariates": covariates,
        "prediction_provider": provider,
        "task_type": "event_rate",
        "episode_length": None,
        "random_start": False,
        "prediction_horizon_steps": 10,
        "observation_features": [
            "time_since_last_sync",
            "current_aoi",
            "predictions",
            "covariates",
            "items_synced_last",
            "cumulative_empty_syncs",
        ],
        "reward_config": {},
    }
    config.update(overrides)
    return config


# Tiny SB3 training tests


@unittest.skipUnless(SB3_AVAILABLE, "stable_baselines3 not installed")
class TestTinyPPOTraining(unittest.TestCase):

    def test_train_and_predict(self):
        """Train PPO for a tiny number of steps and verify prediction works."""
        from stable_baselines3 import PPO

        env = SyncSchedulingEnv(_make_env_config(n_steps=100, episode_length=50))
        model = PPO(
            "MlpPolicy", env,
            n_steps=10, batch_size=5, n_epochs=1,
            seed=42, verbose=0,
        )
        model.learn(total_timesteps=20)

        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        self.assertIn(int(action), [0, 1])

    def test_save_and_load(self):
        """Train, save, load, and verify the loaded model can predict."""
        from stable_baselines3 import PPO

        env = SyncSchedulingEnv(_make_env_config(n_steps=100, episode_length=50))
        model = PPO(
            "MlpPolicy", env,
            n_steps=10, batch_size=5, n_epochs=1,
            seed=42, verbose=0,
        )
        model.learn(total_timesteps=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "agent")
            model.save(path)

            loaded = PPO.load(path)
            obs, _ = env.reset()
            action, _ = loaded.predict(obs, deterministic=True)
            self.assertIn(int(action), [0, 1])


@unittest.skipUnless(SB3_AVAILABLE, "stable_baselines3 not installed")
class TestTinyDQNTraining(unittest.TestCase):

    def test_train_and_predict(self):
        """Train DQN for a tiny number of steps."""
        from stable_baselines3 import DQN

        env = SyncSchedulingEnv(_make_env_config(n_steps=100))
        model = DQN(
            "MlpPolicy", env,
            learning_starts=5, buffer_size=50, batch_size=5,
            seed=42, verbose=0,
        )
        model.learn(total_timesteps=20)

        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        self.assertIn(int(action), [0, 1])


@unittest.skipUnless(SB3_AVAILABLE, "stable_baselines3 not installed")
class TestTinyA2CTraining(unittest.TestCase):

    def test_train_and_predict(self):
        """Train A2C for a tiny number of steps."""
        from stable_baselines3 import A2C

        env = SyncSchedulingEnv(_make_env_config(n_steps=100, episode_length=50))
        model = A2C(
            "MlpPolicy", env,
            n_steps=10, seed=42, verbose=0,
        )
        model.learn(total_timesteps=20)

        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        self.assertIn(int(action), [0, 1])


# Evaluation pipeline integration tests


class TestEvaluationPipelineIntegration(unittest.TestCase):
    """Test the full evaluation pipeline with heuristic policies."""

    def test_periodic_evaluation_produces_artifacts(self):
        """Run periodic policy evaluation and verify all artifacts."""
        env = SyncSchedulingEnv(_make_env_config(n_steps=100))
        policy = PeriodicSyncPolicy(interval_minutes=5)

        decisions_df, summary = run_evaluation(policy, env)

        self.assertEqual(len(decisions_df), 100)
        self.assertIn("true_aoi", decisions_df.columns)
        self.assertIn("action", decisions_df.columns)
        self.assertIn("reward", decisions_df.columns)
        self.assertIn("total_reward", summary)
        self.assertIn("avg_aoi", summary)
        self.assertIn("p95_aoi", summary)
        self.assertIn("median_aoi", summary)
        self.assertEqual(summary["total_steps"], 100)
        self.assertGreater(summary["sync_count"], 0)

    def test_evaluation_writes_to_disk(self):
        """Verify artifact_writer creates all expected files."""
        from sync_policy_evaluation.artifact_writer import write_evaluation_artifacts

        env = SyncSchedulingEnv(_make_env_config(n_steps=50))
        policy = AlwaysSyncPolicy()
        decisions_df, summary = run_evaluation(policy, env)

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = {
                "policy_name": "test_always",
                "task_type": "event_rate",
            }
            write_evaluation_artifacts(tmpdir, decisions_df, summary, metadata)

            self.assertTrue(os.path.exists(os.path.join(tmpdir, "decisions.parquet")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "summary_metrics.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "policy_metadata.json")))

            # Verify parquet roundtrip
            loaded = pd.read_parquet(os.path.join(tmpdir, "decisions.parquet"))
            self.assertEqual(len(loaded), 50)

            # Verify summary JSON
            with open(os.path.join(tmpdir, "summary_metrics.json")) as f:
                saved_summary = json.load(f)
            self.assertAlmostEqual(saved_summary["total_reward"], summary["total_reward"])


@unittest.skipUnless(SB3_AVAILABLE, "stable_baselines3 not installed")
class TestTrainedPolicyEvaluation(unittest.TestCase):
    """Train a tiny model and evaluate it through the evaluation engine."""

    def test_train_then_evaluate(self):
        from stable_baselines3 import PPO
        from sync_policy_evaluation.policy_loader import SB3Policy

        env_config = _make_env_config(n_steps=100, episode_length=50)
        train_env = SyncSchedulingEnv(env_config)
        model = PPO(
            "MlpPolicy", train_env,
            n_steps=10, batch_size=5, n_epochs=1,
            seed=42, verbose=0,
        )
        model.learn(total_timesteps=20)

        # Evaluate on fresh env (no episode_length for full run)
        eval_env = SyncSchedulingEnv(_make_env_config(n_steps=100))
        policy = SB3Policy(model, name="test_ppo")
        decisions_df, summary = run_evaluation(policy, eval_env)

        self.assertEqual(len(decisions_df), 100)
        self.assertIn("total_reward", summary)
        self.assertGreaterEqual(summary["sync_count"], 0)
        self.assertEqual(summary["total_steps"], 100)


# Build env config integration (tests Phase 1 fixes)


class TestBuildEnvConfigIntegration(unittest.TestCase):
    """Test _build_env_config behavior with Phase 1 fixes."""

    def _get_app(self):
        from rl_training.rl_training_app import RLTrainingApp
        return RLTrainingApp()

    def test_random_start_auto_enabled(self):
        """episode_length without random_start defaults to True."""
        app = self._get_app()
        timestamps, actual_values, covariates, provider = _make_synthetic_data(100)
        config = app._build_env_config(
            timestamps=timestamps,
            actual_values=actual_values,
            covariates=covariates,
            prediction_provider=provider,
            task_type="event_rate",
            env_params={"episode_length": 50},
            reward_params={},
        )
        self.assertTrue(config["random_start"])

    def test_random_start_explicit_false_honored(self):
        """Explicit random_start=False is respected."""
        app = self._get_app()
        timestamps, actual_values, covariates, provider = _make_synthetic_data(100)
        config = app._build_env_config(
            timestamps=timestamps,
            actual_values=actual_values,
            covariates=covariates,
            prediction_provider=provider,
            task_type="event_rate",
            env_params={"episode_length": 50, "random_start": False},
            reward_params={},
        )
        self.assertFalse(config["random_start"])

    def test_validation_defaults_to_full_split(self):
        """Existing runs retain full-split validation by default."""
        app = self._get_app()
        params = app._validation_env_params({"episode_length": 50})
        self.assertIsNone(params["episode_length"])
        self.assertFalse(params["random_start"])

    def test_validation_episode_length_is_capped_when_configured(self):
        """Pilots can bound deterministic validation runtime."""
        app = self._get_app()
        params = app._validation_env_params({
            "episode_length": 50,
            "validation_episode_length": 20,
            "random_start": True,
        })
        self.assertEqual(params["episode_length"], 20)
        self.assertFalse(params["random_start"])

    def test_validation_evaluation_count_controls_frequency(self):
        app = self._get_app()
        frequency = app._validation_eval_freq(
            6_000_000, {"validation_evaluations": 3},
        )
        self.assertEqual(frequency, 2_000_000)

    def test_validation_evaluation_count_must_be_positive(self):
        app = self._get_app()
        with self.assertRaisesRegex(ValueError, "must be positive"):
            app._validation_eval_freq(
                10_000, {"validation_evaluations": 0},
            )

    def test_validation_windows_span_entire_split(self):
        app = self._get_app()
        params = app._validation_env_params({
            "validation_episode_length": 20,
            "validation_window_count": 3,
        }, total_steps=100)
        self.assertEqual(params["episode_start_indices"], [0, 40, 80])

    def test_validation_window_count_must_be_positive(self):
        app = self._get_app()
        with self.assertRaisesRegex(ValueError, "window_count must be positive"):
            app._validation_env_params({
                "validation_episode_length": 20,
                "validation_window_count": 0,
            }, total_steps=100)

    def test_validation_episode_length_must_be_positive(self):
        """Non-positive validation caps fail before training."""
        app = self._get_app()
        with self.assertRaisesRegex(ValueError, "must be positive"):
            app._validation_env_params({"validation_episode_length": 0})

    def test_dead_predictions_stripped(self):
        """Predictions feature removed when using NoPredictionProvider."""
        app = self._get_app()
        timestamps, actual_values, covariates, _ = _make_synthetic_data(100)
        config = app._build_env_config(
            timestamps=timestamps,
            actual_values=actual_values,
            covariates=covariates,
            prediction_provider=NoPredictionProvider(),
            task_type="event_rate",
            env_params={
                "observation_features": [
                    "time_since_last_sync",
                    "predictions",
                    "current_aoi",
                ],
            },
            reward_params={},
        )
        self.assertNotIn("predictions", config["observation_features"])
        self.assertIn("time_since_last_sync", config["observation_features"])

    def test_predictions_kept_with_precomputed_provider(self):
        """Predictions stay when a real provider is used."""
        app = self._get_app()
        timestamps, actual_values, covariates, provider = _make_synthetic_data(100)
        config = app._build_env_config(
            timestamps=timestamps,
            actual_values=actual_values,
            covariates=covariates,
            prediction_provider=provider,
            task_type="event_rate",
            env_params={
                "observation_features": [
                    "time_since_last_sync",
                    "predictions",
                    "current_aoi",
                ],
            },
            reward_params={},
        )
        self.assertIn("predictions", config["observation_features"])

    def test_masked_tte_predictions_keep_zero_valued_dimensions(self):
        """Masked TTE inputs preserve dimensions without carrying information."""
        app = self._get_app()
        timestamps, actual_values, covariates, _ = _make_synthetic_data(100)
        config = app._build_env_config(
            timestamps=timestamps,
            actual_values=actual_values,
            covariates=covariates,
            prediction_provider=MaskedPredictionProvider("time_to_event"),
            task_type="event_rate",
            prediction_task_type="time_to_event",
            env_params={
                "prediction_horizon_steps": 10,
                "observation_features": ["predictions"],
            },
            reward_params={},
        )
        env = SyncSchedulingEnv(config)
        observation, _ = env.reset(seed=42)
        self.assertEqual(observation.shape, (10,))
        np.testing.assert_array_equal(observation, np.zeros(10, dtype=np.float32))

    def test_masked_mode_builds_masked_provider(self):
        app = self._get_app()
        provider = app._build_prediction_provider(
            MagicMock(), "time_to_event", {"mode": "masked"}, [], [],
        )
        self.assertIsInstance(provider, MaskedPredictionProvider)
        self.assertEqual(provider.metadata["protocol"], "masked_prediction_v1")

    @patch(
        "rl_training.action_aware_forecaster.ExplicitHistoryForecaster.from_model_path"
    )
    def test_action_aware_provider_requires_explicit_model(self, load_forecaster):
        """Action-aware mode constructs a stateful explicit-history provider."""
        app = self._get_app()
        load_forecaster.return_value = MagicMock()
        provider = app._build_prediction_provider(
            MagicMock(), "event_rate", {
                "mode": "action_aware",
                "trajectory_horizon_steps": 10,
                "model_selection": {"mode": "explicit", "model_id": "model"},
            }, [], [],
        )
        self.assertIsInstance(provider, ActionAwarePredictionProvider)
        load_forecaster.assert_called_once_with("model", "event_rate")


# Per-algorithm dataset passes


class TestAlgoDatasetPasses(unittest.TestCase):

    def test_dqn_default_passes_higher_than_ppo(self):
        from rl_training.rl_training_app import RLTrainingApp
        self.assertGreater(
            RLTrainingApp._ALGO_DATASET_PASSES["DQN"],
            RLTrainingApp._ALGO_DATASET_PASSES["PPO"],
        )


# Training metadata split recording


class TestTrainingMetadataSplit(unittest.TestCase):

    def test_metadata_includes_data_split(self):
        from rl_training.rl_training_app import RLTrainingApp

        app = RLTrainingApp()
        data_split = {
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "train_size": 140,
            "val_size": 30,
            "test_size": 30,
            "train_end_timestamp": "2025-06-01T00:00:00+00:00",
            "val_end_timestamp": "2025-07-01T00:00:00+00:00",
            "data_start_timestamp": "2025-01-01T00:00:00+00:00",
            "data_end_timestamp": "2025-08-01T00:00:00+00:00",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_su = MagicMock()
            mock_su.get_name.return_value = "test/unit"

            app._save_training_metadata(
                artifact_dir=tmpdir,
                sync_unit_obj=mock_su,
                task_type="event_rate",
                agent_params={"algorithm": "PPO"},
                env_params={},
                reward_params={},
                prediction_params={},
                agent_name="test_agent",
                total_timesteps=1000,
                mlflow_run_id=None,
                data_split=data_split,
            )

            meta_path = os.path.join(tmpdir, "training_metadata.json")
            self.assertTrue(os.path.exists(meta_path))

            with open(meta_path) as f:
                meta = json.load(f)

            self.assertIn("data_split", meta)
            self.assertEqual(meta["data_split"]["train_ratio"], 0.7)
            self.assertEqual(meta["data_split"]["val_ratio"], 0.15)
            self.assertEqual(meta["data_split"]["train_size"], 140)
            self.assertIn("train_end_timestamp", meta["data_split"])

    def test_metadata_without_data_split(self):
        """Backward compat: no data_split key when not provided."""
        from rl_training.rl_training_app import RLTrainingApp

        app = RLTrainingApp()

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_su = MagicMock()
            mock_su.get_name.return_value = "test/unit"

            app._save_training_metadata(
                artifact_dir=tmpdir,
                sync_unit_obj=mock_su,
                task_type="event_rate",
                agent_params={"algorithm": "PPO"},
                env_params={},
                reward_params={},
                prediction_params={},
                agent_name="test_agent",
                total_timesteps=1000,
                mlflow_run_id=None,
            )

            with open(os.path.join(tmpdir, "training_metadata.json")) as f:
                meta = json.load(f)

            self.assertNotIn("data_split", meta)


# Reward presets integration


class TestRewardPresetsIntegration(unittest.TestCase):
    """Verify presets work end-to-end with the environment."""

    def test_sparse_friendly_preset_in_env(self):
        """Build env with sparse_friendly preset and run a few steps."""
        from rl_training.reward import RewardConfig

        preset_cfg = RewardConfig.from_preset("sparse_friendly", mode="event_rate")
        config = _make_env_config(
            n_steps=20,
            reward_config=preset_cfg.to_dict(),
        )
        env = SyncSchedulingEnv(config)
        obs, _ = env.reset()

        # Run a few steps
        total_reward = 0.0
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(0)
            total_reward += reward
            if terminated or truncated:
                break

        # sparse_friendly has lower AoI penalty, so total reward
        # should be less negative than default for same skip behavior
        self.assertIsInstance(total_reward, float)

    def test_aoi_only_mode_in_env(self):
        """Verify aoi_only mode produces consistent rewards in env."""
        from rl_training.reward import RewardConfig

        cfg = RewardConfig(reward_mode="aoi_only", mode="event_rate")
        config = _make_env_config(
            n_steps=20,
            reward_config=cfg.to_dict(),
        )
        env = SyncSchedulingEnv(config)
        obs, _ = env.reset()

        # Sync at step 0 (events exist) — in aoi_only mode,
        # reward should be -sync_cost (1.0), not -sync_cost + successful_reward
        _, reward, _, _, info = env.step(1)
        self.assertAlmostEqual(reward, -1.0)


class TestPredictionTaskTypeIntegration(unittest.TestCase):
    """Test prediction_task_type flows through _build_env_config."""

    def _get_app(self):
        from rl_training.rl_training_app import RLTrainingApp
        return RLTrainingApp()

    def test_prediction_task_type_passed_to_env_config(self):
        app = self._get_app()
        timestamps, actual_values, covariates, provider = _make_synthetic_data(100)
        config = app._build_env_config(
            timestamps=timestamps,
            actual_values=actual_values,
            covariates=covariates,
            prediction_provider=provider,
            task_type="event_rate",
            env_params={},
            reward_params={},
            prediction_task_type="time_to_event",
        )
        self.assertEqual(config["prediction_task_type"], "time_to_event")

    def test_prediction_task_type_defaults_to_event_rate(self):
        app = self._get_app()
        timestamps, actual_values, covariates, provider = _make_synthetic_data(100)
        config = app._build_env_config(
            timestamps=timestamps,
            actual_values=actual_values,
            covariates=covariates,
            prediction_provider=provider,
            task_type="event_rate",
            env_params={},
            reward_params={},
        )
        self.assertEqual(config["prediction_task_type"], "event_rate")


if __name__ == "__main__":
    unittest.main()
