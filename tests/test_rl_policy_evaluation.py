"""Tests for sync_policy_evaluation components."""

import os
import json
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from rl_training.prediction_provider import PrecomputedPredictionProvider
from rl_training.sync_env import SyncSchedulingEnv
from sync_policy_evaluation.policy_loader import (
    AlwaysSyncPolicy,
    NeverSyncPolicy,
    PeriodicSyncPolicy,
    ThresholdPolicy,
    ModelPredictionPolicy,
    Policy,
    load_heuristic_policy,
    get_available_heuristics,
)
from sync_policy_evaluation.evaluation_engine import (
    _detect_event,
    TTE_MIN_JUMP,
    run_evaluation,
    compute_summary_metrics,
    compute_rolling_metrics,
    detect_problem_periods,
    DEFAULT_ROLLING_WINDOWS,
)
from sync_policy_evaluation.artifact_writer import write_evaluation_artifacts


def _make_test_env(n_steps=100):
    """Create a minimal test environment with synthetic data."""
    base_time = datetime(2025, 1, 1)
    timestamps = [base_time + timedelta(minutes=i) for i in range(n_steps)]

    actual_values = [0.0] * n_steps
    for i in range(0, n_steps, 10):
        actual_values[i] = 3.0

    predictions = np.random.rand(n_steps, 10).astype(np.float32)
    provider = PrecomputedPredictionProvider(timestamps, predictions)

    env_config = {
        "timestamps": timestamps,
        "actual_values": actual_values,
        "covariates": {"hour_of_day": [float(ts.hour) for ts in timestamps]},
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
        ],
        "reward_config": {},
    }
    return SyncSchedulingEnv(env_config)


# Policy Loader Tests


class TestHeuristicPolicies(unittest.TestCase):

    def test_periodic_sync(self):
        policy = PeriodicSyncPolicy(interval_minutes=5)
        obs = np.zeros(10, dtype=np.float32)

        actions = [policy.predict(obs) for _ in range(15)]
        # Should sync at steps 5, 10, 15 (0-indexed: 4, 9, 14)
        self.assertEqual(sum(actions), 3)

    def test_periodic_reset(self):
        policy = PeriodicSyncPolicy(interval_minutes=3)
        obs = np.zeros(10, dtype=np.float32)

        for _ in range(5):
            policy.predict(obs)
        policy.reset()

        # After reset, counter should be back at 0
        actions = [policy.predict(obs) for _ in range(3)]
        self.assertEqual(actions, [0, 0, 1])

    def test_always_sync(self):
        policy = AlwaysSyncPolicy()
        obs = np.zeros(10, dtype=np.float32)
        self.assertEqual(policy.predict(obs), 1)
        self.assertEqual(policy.get_name(), "always_sync")

    def test_never_sync(self):
        policy = NeverSyncPolicy()
        obs = np.zeros(10, dtype=np.float32)
        self.assertEqual(policy.predict(obs), 0)
        self.assertEqual(policy.get_name(), "never_sync")

    def test_threshold_policy(self):
        policy = ThresholdPolicy(threshold=0.5, prediction_start_index=2)
        # Observation with prediction above threshold
        obs = np.zeros(10, dtype=np.float32)
        obs[2] = 0.8
        self.assertEqual(policy.predict(obs), 1)

        # Below threshold
        obs[2] = 0.3
        self.assertEqual(policy.predict(obs), 0)

    def test_load_heuristic_periodic(self):
        policy = load_heuristic_policy({
            "name": "periodic",
            "interval_minutes": 10,
        })
        self.assertIsInstance(policy, PeriodicSyncPolicy)
        self.assertEqual(policy.get_name(), "periodic_10min")

    def test_load_heuristic_unknown_raises(self):
        with self.assertRaises(ValueError):
            load_heuristic_policy({"name": "nonexistent"})

    def test_available_heuristics(self):
        available = get_available_heuristics()
        self.assertIn("periodic", available)
        self.assertIn("always_sync", available)
        self.assertIn("never_sync", available)
        self.assertIn("threshold", available)


# Evaluation Engine Tests


class TestEvaluationEngine(unittest.TestCase):

    def test_run_evaluation_basic(self):
        env = _make_test_env(n_steps=50)
        policy = PeriodicSyncPolicy(interval_minutes=5)

        decisions_df, summary = run_evaluation(policy, env)

        self.assertIsInstance(decisions_df, pd.DataFrame)
        self.assertEqual(len(decisions_df), 50)
        self.assertIn("action", decisions_df.columns)
        self.assertIn("reward", decisions_df.columns)
        self.assertIn("true_aoi", decisions_df.columns)
        self.assertIn("current_aoi", decisions_df.columns)

        self.assertIn("total_reward", summary)
        self.assertIn("sync_count", summary)
        self.assertIn("empty_sync_rate", summary)
        self.assertIn("avg_aoi", summary)

    def test_always_sync_high_sync_count(self):
        env = _make_test_env(n_steps=20)
        policy = AlwaysSyncPolicy()

        _, summary = run_evaluation(policy, env)

        self.assertEqual(summary["sync_count"], 20)
        self.assertEqual(summary["sync_frequency"], 1.0)

    def test_never_sync_zero_sync_count(self):
        env = _make_test_env(n_steps=20)
        policy = NeverSyncPolicy()

        _, summary = run_evaluation(policy, env)

        self.assertEqual(summary["sync_count"], 0)
        self.assertEqual(summary["sync_frequency"], 0.0)

    def test_summary_metrics_complete(self):
        records = [
            {"action": 1, "reward": -1.0, "current_aoi": 0.0,
             "true_aoi": 0.0, "last_sync_items": 3.0},
            {"action": 0, "reward": 0.0, "current_aoi": 1.0,
             "true_aoi": 0.0, "last_sync_items": 3.0},
            {"action": 1, "reward": -1.0, "current_aoi": 0.0,
             "true_aoi": 0.0, "last_sync_items": 0.0},
        ]
        summary = compute_summary_metrics(records)

        self.assertEqual(summary["total_steps"], 3)
        self.assertEqual(summary["sync_count"], 2)
        self.assertEqual(summary["empty_sync_count"], 1)
        self.assertAlmostEqual(summary["empty_sync_rate"], 0.5)
        self.assertIn("p95_aoi", summary)
        self.assertIn("median_aoi", summary)

    def test_summary_metrics_prefers_true_aoi(self):
        """When true_aoi is present, summary AoI metrics use it."""
        records = [
            {"action": 0, "reward": 0.0, "current_aoi": 5.0,
             "true_aoi": 0.0, "last_sync_items": 0.0},
            {"action": 0, "reward": 0.0, "current_aoi": 6.0,
             "true_aoi": 0.0, "last_sync_items": 0.0},
        ]
        summary = compute_summary_metrics(records)
        # true_aoi is 0 (calm period, no events) even though
        # current_aoi (time since sync) is 5-6.
        self.assertAlmostEqual(summary["avg_aoi"], 0.0)

    def test_summary_metrics_falls_back_to_current_aoi(self):
        """Without true_aoi, summary uses current_aoi (backward compat)."""
        records = [
            {"action": 0, "reward": 0.0, "current_aoi": 5.0,
             "last_sync_items": 0.0},
            {"action": 0, "reward": 0.0, "current_aoi": 6.0,
             "last_sync_items": 0.0},
        ]
        summary = compute_summary_metrics(records)
        self.assertAlmostEqual(summary["avg_aoi"], 5.5)

    def test_true_aoi_semantics(self):
        """True AoI stays 0 during calm periods, grows only after events."""
        # Synthetic data: events at steps 0, 10, 20, ... (every 10th)
        # Same as _make_test_env pattern
        env = _make_test_env(n_steps=20)
        policy = NeverSyncPolicy()

        decisions_df, _ = run_evaluation(policy, env)
        true_aoi = decisions_df["true_aoi"].tolist()

        # Step 0: event (actual_value=3.0), no sync → true_aoi=0
        #   (event just appeared, 0 minutes stale)
        self.assertAlmostEqual(true_aoi[0], 0.0)

        # Steps 1-9: no events, no sync → true_aoi grows from the
        # unsynced event at step 0
        for i in range(1, 10):
            self.assertAlmostEqual(true_aoi[i], float(i))

        # Step 10: another event, no sync → first unsynced is still
        # step 0, so true_aoi = 10
        self.assertAlmostEqual(true_aoi[10], 10.0)

    def test_true_aoi_resets_on_sync(self):
        """After syncing, true AoI resets and stays 0 until next event."""
        env = _make_test_env(n_steps=20)
        policy = AlwaysSyncPolicy()

        decisions_df, _ = run_evaluation(policy, env)
        true_aoi = decisions_df["true_aoi"].tolist()

        # AlwaysSync syncs every step → true_aoi always 0
        for aoi in true_aoi:
            self.assertAlmostEqual(aoi, 0.0)


# Artifact Writer Tests


class TestArtifactWriter(unittest.TestCase):

    def test_write_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = os.path.join(tmpdir, "eval_output")

            decisions_df = pd.DataFrame({
                "timestamp": ["2025-01-01T00:00", "2025-01-01T00:01"],
                "action": [0, 1],
                "reward": [0.0, -1.0],
                "current_aoi": [1.0, 0.0],
            })
            summary = {"total_reward": -1.0, "sync_count": 1}
            metadata = {"policy_name": "test", "task_type": "event_rate"}

            write_evaluation_artifacts(
                eval_dir, decisions_df, summary, metadata
            )

            self.assertTrue(os.path.exists(
                os.path.join(eval_dir, "decisions.parquet")
            ))
            self.assertTrue(os.path.exists(
                os.path.join(eval_dir, "summary_metrics.json")
            ))
            self.assertTrue(os.path.exists(
                os.path.join(eval_dir, "policy_metadata.json")
            ))

            # Verify parquet is readable
            df = pd.read_parquet(os.path.join(eval_dir, "decisions.parquet"))
            self.assertEqual(len(df), 2)

            # Verify JSON
            with open(os.path.join(eval_dir, "summary_metrics.json")) as f:
                s = json.load(f)
            self.assertEqual(s["total_reward"], -1.0)


# Eval Split Slicing Tests


class TestApplyEvalSplitByTimestamp(unittest.TestCase):
    """Cover _apply_eval_split_by_timestamp.

    This replaced the older ratio-propagation path (and its
    _validate_split_consistency check): splits are now sliced by the exact
    timestamps recorded in the agent's training metadata, falling back to
    config ratios only when no metadata is available.
    """

    def setUp(self):
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)
        # 100 consecutive minutes, values 0..99
        self.timestamps = [base + timedelta(minutes=i) for i in range(100)]
        self.values = list(range(100))
        self.covariates = {"hour_of_day": [float(i % 24) for i in range(100)]}
        # 70 / 15 / 15 boundaries expressed as timestamps
        self.split_meta = {
            "train_end_timestamp": self.timestamps[69].isoformat(),
            "val_end_timestamp": self.timestamps[84].isoformat(),
        }

    def _slice(self, eval_split, split_meta=None, dataset_params=None):
        from sync_policy_evaluation.sync_policy_evaluation_app import (
            SyncPolicyEvaluationApp,
        )

        return SyncPolicyEvaluationApp._apply_eval_split_by_timestamp(
            eval_split,
            split_meta,
            dataset_params if dataset_params is not None else {},
            self.timestamps, self.values, self.covariates,
        )

    def test_all_returns_everything_unsliced(self):
        ts, vals, covs = self._slice("all", self.split_meta)
        self.assertEqual(len(ts), 100)
        self.assertEqual(vals, self.values)
        self.assertEqual(covs, self.covariates)

    def test_metadata_boundaries_partition_the_series(self):
        # train/val/test from metadata must tile the span with no overlap
        _, train, _ = self._slice("train", self.split_meta)
        _, val, _ = self._slice("val", self.split_meta)
        _, test, _ = self._slice("test", self.split_meta)

        self.assertEqual(train, list(range(70)))
        self.assertEqual(val, list(range(70, 85)))
        self.assertEqual(test, list(range(85, 100)))
        self.assertEqual(train + val + test, self.values)

    def test_covariates_are_sliced_with_the_values(self):
        ts, vals, covs = self._slice("val", self.split_meta)
        self.assertEqual(len(covs["hour_of_day"]), len(vals))
        self.assertEqual(
            covs["hour_of_day"], [float(i % 24) for i in range(70, 85)]
        )

    def test_falls_back_to_config_ratios_without_metadata(self):
        # Heuristic policies have no training metadata
        params = {"train_ratio": 0.7, "val_ratio": 0.15}
        _, train, _ = self._slice("train", None, params)
        _, val, _ = self._slice("val", None, params)
        _, test, _ = self._slice("test", None, params)

        self.assertEqual(train, list(range(70)))
        self.assertEqual(val, list(range(70, 85)))
        self.assertEqual(test, list(range(85, 100)))

    def test_val_end_used_as_test_start(self):
        # test must begin after val_end, not after train_end
        _, test, _ = self._slice("test", self.split_meta)
        self.assertEqual(test[0], 85)

    def test_train_end_only_metadata_warns_and_starts_test_there(self):
        import logging

        meta = {"train_end_timestamp": self.timestamps[69].isoformat()}
        with self.assertLogs(
            "sync_policy_evaluation.sync_policy_evaluation_app",
            level=logging.WARNING,
        ) as cm:
            _, test, _ = self._slice("test", meta)
        self.assertEqual(test[0], 70)
        self.assertTrue(any("val_end_timestamp" in m for m in cm.output))

    def test_invalid_eval_split_raises(self):
        with self.assertRaises(ValueError):
            self._slice("holdout", self.split_meta)

    def test_empty_split_raises(self):
        # A boundary at the very end leaves no test data. Regression: start_idx
        # used to default to 0 when nothing cleared the boundary, so the guard
        # never fired and the FULL span came back labelled as the test split
        meta = {
            "train_end_timestamp": self.timestamps[0].isoformat(),
            "val_end_timestamp": self.timestamps[-1].isoformat(),
        }
        with self.assertRaises(ValueError):
            self._slice("test", meta)

    def test_split_end_beyond_data_runs_to_the_end(self):
        # The mirror case: a val_end past the last timestamp is not an error —
        # the split legitimately extends to the end of the available data
        from datetime import timedelta

        meta = {
            "train_end_timestamp": self.timestamps[69].isoformat(),
            "val_end_timestamp": (
                self.timestamps[-1] + timedelta(days=30)
            ).isoformat(),
        }
        _, val, _ = self._slice("val", meta)
        self.assertEqual(val, list(range(70, 100)))

    def test_history_before_split_contains_only_prior_values(self):
        from sync_policy_evaluation.sync_policy_evaluation_app import (
            SyncPolicyEvaluationApp,
        )
        timestamps, values = SyncPolicyEvaluationApp._history_before_split(
            self.timestamps[85], self.timestamps, self.values,
        )
        self.assertEqual(timestamps[-1], self.timestamps[84])
        self.assertEqual(values, list(range(85)))
        self.assertNotIn(85, values)


class TestActionAwareEvaluationProvider(unittest.TestCase):
    class _Forecaster:
        def __init__(self):
            self.metadata = {"model_training_data_end": "9999-12-31"}

        def forecast(self, *args):
            return [0.0] * args[-1]

    def test_action_aware_provider_records_declared_fitting_cutoff(self):
        from rl_training.prediction_provider import ActionAwarePredictionProvider
        from sync_policy_evaluation.sync_policy_evaluation_app import (
            SyncPolicyEvaluationApp,
        )
        forecaster = self._Forecaster()
        params = {
            "mode": "action_aware",
            "forecasting_model_fitting_cutoff": "2024-01-02 03:04:00",
            "model_selection": {"mode": "explicit", "model_id": "model"},
        }
        with patch(
            "rl_training.action_aware_forecaster.ExplicitHistoryForecaster.from_model_path",
            return_value=forecaster,
        ):
            provider = SyncPolicyEvaluationApp()._build_prediction_provider(
                None, "event_rate", params, [], [],
            )
        self.assertIsInstance(provider, ActionAwarePredictionProvider)
        self.assertEqual(
            provider.metadata["forecasting_model_fitting_cutoff"],
            "2024-01-02 03:04:00",
        )
        self.assertEqual(provider.metadata["protocol"], "action_aware_trajectory_v1")

    def test_action_aware_provider_rejects_unknown_fitting_cutoff(self):
        from sync_policy_evaluation.sync_policy_evaluation_app import (
            SyncPolicyEvaluationApp,
        )
        params = {
            "mode": "action_aware",
            "model_selection": {"mode": "explicit", "model_id": "model"},
        }
        with patch(
            "rl_training.action_aware_forecaster.ExplicitHistoryForecaster.from_model_path",
            return_value=self._Forecaster(),
        ):
            with self.assertRaisesRegex(ValueError, "fitting_cutoff provenance"):
                SyncPolicyEvaluationApp()._build_prediction_provider(
                    None, "event_rate", params, [], [],
                )

    def test_masked_mode_builds_masked_provider(self):
        from rl_training.prediction_provider import MaskedPredictionProvider
        from sync_policy_evaluation.sync_policy_evaluation_app import (
            SyncPolicyEvaluationApp,
        )
        provider = SyncPolicyEvaluationApp()._build_prediction_provider(
            None, "time_to_event", {"mode": "masked"}, [], [],
        )
        self.assertIsInstance(provider, MaskedPredictionProvider)
        self.assertEqual(provider.metadata["protocol"], "masked_prediction_v1")


# Progress Callback MLflow Tests


class TestProgressCallback(unittest.TestCase):

    def test_callback_with_tracker(self):
        from unittest.mock import MagicMock
        from rl_training.progress_callback import ProgressLogCallback

        tracker = MagicMock()
        cb = ProgressLogCallback(
            log_interval=5,
            total_timesteps=100,
            agent_name="test",
            tracker=tracker,
        )
        cb.num_timesteps = 0
        cb._on_training_start()

        # Simulate steps with episode info
        cb.locals = {"infos": [{"episode": {"r": -5.0, "l": 10}}]}
        cb.n_envs = 1
        for step in range(1, 11):
            cb.num_timesteps = step
            cb._on_step()

        # Tracker should have been called at step 5 and 10
        self.assertTrue(tracker.log_metrics.called)

    def test_callback_logs_available_sb3_training_metrics(self):
        from unittest.mock import MagicMock
        from rl_training.progress_callback import ProgressLogCallback

        tracker = MagicMock()
        cb = ProgressLogCallback(log_interval=5, total_timesteps=10, tracker=tracker)
        cb.model = MagicMock()
        cb.model.logger.name_to_value = {
            "train/policy_gradient_loss": np.float32(-0.125),
            "train/value_loss": 3.5,
            "train/approx_kl": float("nan"),
            "rollout/ep_rew_mean": -10.0,
        }
        cb._on_training_start()
        cb.locals = {"infos": []}
        cb.n_envs = 1
        cb.num_timesteps = 5

        cb._on_step()

        metrics = tracker.log_metrics.call_args.args[0]
        self.assertAlmostEqual(metrics["train_policy_gradient_loss"], -0.125)
        self.assertEqual(metrics["train_value_loss"], 3.5)
        self.assertNotIn("train_approx_kl", metrics)
        self.assertNotIn("rollout_ep_rew_mean", metrics)

    def test_callback_without_tracker(self):
        from rl_training.progress_callback import ProgressLogCallback

        cb = ProgressLogCallback(
            log_interval=5,
            total_timesteps=100,
            agent_name="test",
        )
        cb.num_timesteps = 0
        cb._on_training_start()
        cb.locals = {"infos": []}
        cb.n_envs = 1

        # Should not raise
        cb.num_timesteps = 5
        cb._on_step()

    def test_callback_collects_episode_stats(self):
        from rl_training.progress_callback import ProgressLogCallback

        cb = ProgressLogCallback(
            log_interval=100, total_timesteps=100, agent_name="test"
        )
        cb.num_timesteps = 0
        cb._on_training_start()
        cb.n_envs = 1

        cb.locals = {"infos": [{"episode": {"r": -3.0, "l": 20}}]}
        cb.num_timesteps = 1
        cb._on_step()
        cb.locals = {"infos": [{"episode": {"r": -1.0, "l": 15}}]}
        cb.num_timesteps = 2
        cb._on_step()

        self.assertEqual(len(cb._episode_rewards), 2)
        self.assertAlmostEqual(cb._episode_rewards[0], -3.0)
        self.assertAlmostEqual(cb._episode_rewards[1], -1.0)


# TTE Event Detection Tests


class TestDetectEvent(unittest.TestCase):

    def test_event_rate_positive(self):
        self.assertTrue(_detect_event(3.0, 0.0, "event_rate"))

    def test_event_rate_zero(self):
        self.assertFalse(_detect_event(0.0, 0.0, "event_rate"))

    def test_event_rate_ignores_prev(self):
        self.assertTrue(_detect_event(1.0, None, "event_rate"))

    def test_tte_no_prev(self):
        self.assertFalse(_detect_event(10.0, None, "time_to_event"))

    def test_tte_large_jump_is_event(self):
        # Jump from 2.0 to 50.0 — clear countdown reset
        self.assertTrue(_detect_event(50.0, 2.0, "time_to_event"))

    def test_tte_small_noise_not_event(self):
        # Jump of 0.3 minutes — below TTE_MIN_JUMP threshold
        self.assertFalse(_detect_event(10.3, 10.0, "time_to_event"))

    def test_tte_exact_threshold_not_event(self):
        # Jump of exactly TTE_MIN_JUMP — not strictly greater
        self.assertFalse(
            _detect_event(10.0 + TTE_MIN_JUMP, 10.0, "time_to_event")
        )

    def test_tte_just_above_threshold_is_event(self):
        self.assertTrue(
            _detect_event(10.0 + TTE_MIN_JUMP + 0.1, 10.0, "time_to_event")
        )

    def test_tte_decreasing_not_event(self):
        # Normal countdown: 50 → 49 minutes
        self.assertFalse(_detect_event(49.0, 50.0, "time_to_event"))

    def test_tte_custom_threshold(self):
        # With a custom min_jump of 5.0
        self.assertFalse(
            _detect_event(13.0, 10.0, "time_to_event", tte_min_jump=5.0)
        )
        self.assertTrue(
            _detect_event(16.0, 10.0, "time_to_event", tte_min_jump=5.0)
        )


# Oracle Decisions Tests


class TestOracleDecisions(unittest.TestCase):

    def _make_decisions_df(self, actual_values, mode="event_rate"):
        """Helper to build a minimal decisions DataFrame."""
        base_time = datetime(2025, 1, 1)
        timestamps = [
            base_time + timedelta(minutes=i)
            for i in range(len(actual_values))
        ]
        return pd.DataFrame({
            "timestamp": timestamps,
            "actual_value": actual_values,
        })

    def test_event_rate_oracle_syncs_at_events(self):
        from sync_policy_dashboard import compute_oracle_decisions
        from rl_training.reward import RewardConfig

        values = [0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0]
        df = self._make_decisions_df(values)
        rc = RewardConfig(mode="event_rate")
        records = compute_oracle_decisions(df, rc)

        actions = [r["action"] for r in records]
        # Oracle should sync at steps 2 and 5 (where events are)
        self.assertEqual(actions[2], 1)
        self.assertEqual(actions[5], 1)
        # Should not sync at calm steps (unless maintenance kicks in)
        self.assertEqual(actions[0], 0)
        self.assertEqual(actions[1], 0)
        self.assertEqual(actions[3], 0)

    def test_event_rate_oracle_accumulates_pending_items(self):
        from sync_policy_dashboard import compute_oracle_decisions
        from rl_training.reward import RewardConfig

        # Events at steps 0, 1, 2 — oracle syncs each, last_sync_items
        # should reflect accumulated items, not just current step.
        values = [2.0, 3.0, 1.0, 0.0]
        df = self._make_decisions_df(values)
        rc = RewardConfig(mode="event_rate")
        records = compute_oracle_decisions(df, rc)

        # Step 0: event, sync. pending was 2.0 → last_items = 2.0
        self.assertAlmostEqual(records[0]["last_sync_items"], 2.0)
        # Step 1: event, sync. pending was 3.0 (reset after step 0)
        self.assertAlmostEqual(records[1]["last_sync_items"], 3.0)

    def test_tte_oracle_syncs_on_countdown_reset(self):
        from sync_policy_dashboard import compute_oracle_decisions
        from rl_training.reward import RewardConfig

        # TTE countdown: 5, 4, 3, 2, 1, 20 (reset = event), 19, 18
        values = [5.0, 4.0, 3.0, 2.0, 1.0, 20.0, 19.0, 18.0]
        df = self._make_decisions_df(values, mode="time_to_event")
        rc = RewardConfig(mode="time_to_event", aoi_threshold_minutes=5.0)
        # Use large maintenance interval so it doesn't interfere
        records = compute_oracle_decisions(
            df, rc, maintenance_interval_override=1e9
        )

        actions = [r["action"] for r in records]
        # Only step 5 has an event (jump from 1.0 to 20.0)
        self.assertEqual(actions[5], 1)
        # Other steps should not sync (no events, maintenance is 1e9)
        for i in [0, 1, 2, 3, 4, 6, 7]:
            self.assertEqual(actions[i], 0, f"step {i} should not sync")

    def test_tte_oracle_no_sync_on_small_jump(self):
        from sync_policy_dashboard import compute_oracle_decisions
        from rl_training.reward import RewardConfig

        # Small noise-level increase: 10.0 → 10.5 (below TTE_MIN_JUMP)
        values = [10.0, 10.5, 9.5, 8.5]
        df = self._make_decisions_df(values, mode="time_to_event")
        rc = RewardConfig(mode="time_to_event")
        records = compute_oracle_decisions(
            df, rc, maintenance_interval_override=1e9
        )

        actions = [r["action"] for r in records]
        # No real events — no syncs
        self.assertEqual(sum(actions), 0)


# Timing Quality Metrics Tests


class TestTimingQualityMetrics(unittest.TestCase):

    def test_reaction_time_immediate_sync(self):
        """Policy syncs at every event → reaction time = 0."""
        records = [
            {"action": 0, "reward": 0.0, "current_aoi": 1.0,
             "true_aoi": 0.0, "last_sync_items": 0.0, "actual_value": 0.0},
            {"action": 0, "reward": 0.0, "current_aoi": 2.0,
             "true_aoi": 0.0, "last_sync_items": 0.0, "actual_value": 0.0},
            {"action": 1, "reward": -1.0, "current_aoi": 0.0,
             "true_aoi": 0.0, "last_sync_items": 3.0, "actual_value": 3.0},
            {"action": 0, "reward": 0.0, "current_aoi": 1.0,
             "true_aoi": 0.0, "last_sync_items": 0.0, "actual_value": 0.0},
        ]
        summary = compute_summary_metrics(records, task_type="event_rate")
        # Event at step 2 (actual_value=3.0), sync also at step 2 → reaction = 0
        self.assertAlmostEqual(summary["avg_reaction_time"], 0.0)
        self.assertEqual(summary["total_events_detected"], 1)
        self.assertAlmostEqual(summary["event_capture_rate"], 1.0)

    def test_reaction_time_delayed_sync(self):
        """Policy syncs 3 steps after the event."""
        records = [
            {"action": 0, "reward": 0.0, "current_aoi": 1.0,
             "true_aoi": 0.0, "last_sync_items": 0.0, "actual_value": 5.0},
            {"action": 0, "reward": 0.0, "current_aoi": 2.0,
             "true_aoi": 1.0, "last_sync_items": 0.0, "actual_value": 0.0},
            {"action": 0, "reward": 0.0, "current_aoi": 3.0,
             "true_aoi": 2.0, "last_sync_items": 0.0, "actual_value": 0.0},
            {"action": 1, "reward": -1.0, "current_aoi": 0.0,
             "true_aoi": 0.0, "last_sync_items": 5.0, "actual_value": 0.0},
        ]
        summary = compute_summary_metrics(records, task_type="event_rate")
        # Event at step 0, first sync at step 3 → reaction = 3
        self.assertAlmostEqual(summary["avg_reaction_time"], 3.0)

    def test_event_capture_rate_partial(self):
        """Two events, only first captured before second arrives."""
        records = [
            {"action": 0, "reward": 0.0, "current_aoi": 1.0,
             "true_aoi": 0.0, "last_sync_items": 0.0, "actual_value": 2.0},
            {"action": 1, "reward": -1.0, "current_aoi": 0.0,
             "true_aoi": 0.0, "last_sync_items": 2.0, "actual_value": 0.0},
            {"action": 0, "reward": 0.0, "current_aoi": 1.0,
             "true_aoi": 0.0, "last_sync_items": 0.0, "actual_value": 3.0},
            {"action": 0, "reward": 0.0, "current_aoi": 2.0,
             "true_aoi": 1.0, "last_sync_items": 0.0, "actual_value": 0.0},
        ]
        summary = compute_summary_metrics(records, task_type="event_rate")
        # Event at step 0: sync at step 1 (before step 2 event) → captured
        # Event at step 2: no sync before end → not captured
        self.assertEqual(summary["total_events_detected"], 2)
        self.assertAlmostEqual(summary["event_capture_rate"], 0.5)

    def test_no_events_gives_zero_metrics(self):
        """All calm — no events detected."""
        records = [
            {"action": 0, "reward": 0.0, "current_aoi": 1.0,
             "true_aoi": 0.0, "last_sync_items": 0.0, "actual_value": 0.0},
            {"action": 0, "reward": 0.0, "current_aoi": 2.0,
             "true_aoi": 0.0, "last_sync_items": 0.0, "actual_value": 0.0},
        ]
        summary = compute_summary_metrics(records, task_type="event_rate")
        self.assertEqual(summary["total_events_detected"], 0)
        self.assertAlmostEqual(summary["avg_reaction_time"], 0.0)
        self.assertAlmostEqual(summary["event_capture_rate"], 0.0)

    def test_no_actual_value_gives_zero_metrics(self):
        """Records without actual_value (e.g. re-scored) → defaults to 0."""
        records = [
            {"action": 1, "reward": -1.0, "current_aoi": 0.0,
             "true_aoi": 0.0, "last_sync_items": 3.0},
            {"action": 0, "reward": 0.0, "current_aoi": 1.0,
             "true_aoi": 0.0, "last_sync_items": 0.0},
        ]
        summary = compute_summary_metrics(records)
        self.assertEqual(summary["total_events_detected"], 0)
        self.assertAlmostEqual(summary["avg_reaction_time"], 0.0)


# Comparison Compatibility Validation Tests


class TestComparisonCompatibility(unittest.TestCase):

    def test_compatible_evaluations_no_warnings(self):
        from sync_policy_evaluation.sync_policy_evaluation_app import (
            SyncPolicyEvaluationApp,
        )
        comparison = {
            "policy_a": {
                "metadata": {
                    "eval_split": "val",
                    "task_type": "event_rate",
                    "total_steps": 1000,
                    "dataset_config": {"start_time": "2025-01-01"},
                },
            },
            "policy_b": {
                "metadata": {
                    "eval_split": "val",
                    "task_type": "event_rate",
                    "total_steps": 1000,
                    "dataset_config": {"start_time": "2025-01-01"},
                },
            },
        }
        warnings = SyncPolicyEvaluationApp._validate_comparison_compatibility(
            comparison
        )
        self.assertEqual(len(warnings), 0)

    def test_eval_split_mismatch(self):
        from sync_policy_evaluation.sync_policy_evaluation_app import (
            SyncPolicyEvaluationApp,
        )
        comparison = {
            "policy_a": {"metadata": {"eval_split": "val"}},
            "policy_b": {"metadata": {"eval_split": "test"}},
        }
        warnings = SyncPolicyEvaluationApp._validate_comparison_compatibility(
            comparison
        )
        self.assertTrue(any("eval_split" in w for w in warnings))

    def test_task_type_mismatch(self):
        from sync_policy_evaluation.sync_policy_evaluation_app import (
            SyncPolicyEvaluationApp,
        )
        comparison = {
            "a": {"metadata": {"task_type": "event_rate"}},
            "b": {"metadata": {"task_type": "time_to_event"}},
        }
        warnings = SyncPolicyEvaluationApp._validate_comparison_compatibility(
            comparison
        )
        self.assertTrue(any("task_type" in w for w in warnings))

    def test_total_steps_mismatch(self):
        from sync_policy_evaluation.sync_policy_evaluation_app import (
            SyncPolicyEvaluationApp,
        )
        comparison = {
            "a": {"metadata": {"total_steps": 1000}},
            "b": {"metadata": {"total_steps": 2000}},
        }
        warnings = SyncPolicyEvaluationApp._validate_comparison_compatibility(
            comparison
        )
        self.assertTrue(any("total_steps" in w for w in warnings))

    def test_time_range_mismatch(self):
        from sync_policy_evaluation.sync_policy_evaluation_app import (
            SyncPolicyEvaluationApp,
        )
        comparison = {
            "a": {"metadata": {"dataset_config": {"start_time": "2025-01-01"}}},
            "b": {"metadata": {"dataset_config": {"start_time": "2025-06-01"}}},
        }
        warnings = SyncPolicyEvaluationApp._validate_comparison_compatibility(
            comparison
        )
        self.assertTrue(any("start_time" in w for w in warnings))

    def test_single_policy_no_warnings(self):
        from sync_policy_evaluation.sync_policy_evaluation_app import (
            SyncPolicyEvaluationApp,
        )
        comparison = {
            "only_one": {"metadata": {"eval_split": "val"}},
        }
        warnings = SyncPolicyEvaluationApp._validate_comparison_compatibility(
            comparison
        )
        self.assertEqual(len(warnings), 0)


# Rolling Metrics Tests


class TestRollingMetrics(unittest.TestCase):

    def _make_decisions_df(self, n=100):
        """Create a minimal decisions DataFrame."""
        np.random.seed(42)
        actions = np.random.choice([0, 1], size=n, p=[0.8, 0.2])
        rewards = np.where(actions == 1, -1.0, 0.0)
        true_aoi = np.zeros(n)
        last_sync_items = np.where(actions == 1, np.random.rand(n) * 5, 0.0)
        # Make some syncs empty
        empty_mask = (actions == 1) & (np.arange(n) % 3 == 0)
        last_sync_items[empty_mask] = 0.0

        return pd.DataFrame({
            "action": actions,
            "reward": rewards,
            "true_aoi": true_aoi,
            "current_aoi": true_aoi + 1.0,
            "last_sync_items": last_sync_items,
        })

    def test_columns_added(self):
        df = self._make_decisions_df()
        result = compute_rolling_metrics(df, windows=[10, 50])

        for w in [10, 50]:
            self.assertIn(f"rolling_{w}_reward", result.columns)
            self.assertIn(f"rolling_{w}_aoi", result.columns)
            self.assertIn(f"rolling_{w}_sync_rate", result.columns)
            self.assertIn(f"rolling_{w}_empty_rate", result.columns)

    def test_default_windows(self):
        df = self._make_decisions_df(n=600)
        result = compute_rolling_metrics(df)

        for w in DEFAULT_ROLLING_WINDOWS:
            self.assertIn(f"rolling_{w}_reward", result.columns)

    def test_rolling_values_are_reasonable(self):
        df = self._make_decisions_df(n=200)
        result = compute_rolling_metrics(df, windows=[20])

        # Rolling reward should be between min and max of raw reward
        raw_min = df["reward"].min()
        raw_max = df["reward"].max()
        rolling_vals = result["rolling_20_reward"].dropna()
        self.assertTrue((rolling_vals >= raw_min).all())
        self.assertTrue((rolling_vals <= raw_max).all())

        # Sync rate should be between 0 and 1
        sync_rate = result["rolling_20_sync_rate"].dropna()
        self.assertTrue((sync_rate >= 0).all())
        self.assertTrue((sync_rate <= 1).all())

    def test_empty_rate_only_counts_syncs(self):
        """Empty rate = empty syncs / total syncs, not / all steps."""
        df = pd.DataFrame({
            "action":          [0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
            "reward":          [0]*10,
            "true_aoi":        [0]*10,
            "last_sync_items": [0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
        })
        result = compute_rolling_metrics(df, windows=[10])
        # Syncs at indices 4, 5, 8. Items: 0, 3, 0 → 2 empty / 3 syncs
        val = result["rolling_10_empty_rate"].iloc[-1]
        self.assertAlmostEqual(val, 2.0 / 3.0, places=4)

    def test_no_syncs_gives_nan_empty_rate(self):
        df = pd.DataFrame({
            "action":          [0, 0, 0, 0, 0],
            "reward":          [0]*5,
            "true_aoi":        [0]*5,
            "last_sync_items": [0]*5,
        })
        result = compute_rolling_metrics(df, windows=[5])
        self.assertTrue(np.isnan(result["rolling_5_empty_rate"].iloc[-1]))

    def test_uses_true_aoi_over_current_aoi(self):
        df = pd.DataFrame({
            "action": [0, 0, 0],
            "reward": [0, 0, 0],
            "true_aoi": [0.0, 0.0, 0.0],
            "current_aoi": [10.0, 11.0, 12.0],
            "last_sync_items": [0, 0, 0],
        })
        result = compute_rolling_metrics(df, windows=[3])
        self.assertAlmostEqual(result["rolling_3_aoi"].iloc[-1], 0.0)

    def test_persisted_in_run_evaluation(self):
        """Rolling columns should be present in run_evaluation output."""
        env = _make_test_env(n_steps=100)
        policy = PeriodicSyncPolicy(interval_minutes=5)
        decisions_df, summary = run_evaluation(policy, env)

        for w in DEFAULT_ROLLING_WINDOWS:
            self.assertIn(f"rolling_{w}_reward", decisions_df.columns)


# Problem Period Detection Tests


class TestProblemPeriodDetection(unittest.TestCase):

    def _make_df_with_rolling(self, aoi_values, window=10):
        """Create a DataFrame with pre-computed rolling columns."""
        n = len(aoi_values)
        df = pd.DataFrame({
            "action": [0] * n,
            "reward": [-0.1] * n,
            "true_aoi": aoi_values,
            "last_sync_items": [0.0] * n,
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="min"),
        })
        return compute_rolling_metrics(df, windows=[window])

    def test_no_problems_returns_empty(self):
        aoi = [1.0] * 50
        df = self._make_df_with_rolling(aoi, window=10)
        periods = detect_problem_periods(df, window=10)
        self.assertEqual(len(periods), 0)

    def test_detects_high_aoi_period(self):
        aoi = [1.0] * 20 + [50.0] * 30 + [1.0] * 20
        df = self._make_df_with_rolling(aoi, window=10)
        periods = detect_problem_periods(
            df, window=10,
            thresholds={"aoi_max": 30.0, "empty_rate_max": 999, "reward_percentile": 0},
        )
        self.assertGreater(len(periods), 0)
        self.assertTrue(any("aoi" in p["triggers"] for p in periods))

    def test_period_has_required_fields(self):
        aoi = [1.0] * 10 + [50.0] * 20 + [1.0] * 10
        df = self._make_df_with_rolling(aoi, window=10)
        periods = detect_problem_periods(
            df, window=10,
            thresholds={"aoi_max": 20.0, "empty_rate_max": 999, "reward_percentile": 0},
        )
        if periods:
            p = periods[0]
            self.assertIn("start_idx", p)
            self.assertIn("end_idx", p)
            self.assertIn("duration_minutes", p)
            self.assertIn("triggers", p)
            self.assertIn("severity", p)
            self.assertIn("start_timestamp", p)
            self.assertIn("end_timestamp", p)
            self.assertGreater(p["duration_minutes"], 0)

    def test_missing_rolling_columns_returns_empty(self):
        df = pd.DataFrame({
            "action": [0, 0, 0],
            "reward": [0, 0, 0],
            "true_aoi": [0, 0, 0],
            "last_sync_items": [0, 0, 0],
        })
        periods = detect_problem_periods(df, window=10)
        self.assertEqual(len(periods), 0)

    def test_problem_periods_in_summary(self):
        """run_evaluation should embed problem periods in summary."""
        env = _make_test_env(n_steps=100)
        policy = PeriodicSyncPolicy(interval_minutes=5)
        _, summary = run_evaluation(policy, env)

        self.assertIn("problem_periods", summary)
        self.assertIn("problem_period_count", summary)
        self.assertIn("problem_period_total_minutes", summary)
        self.assertIsInstance(summary["problem_periods"], list)


# Temporal Heatmap Tests


class TestTemporalHeatmap(unittest.TestCase):

    def _make_decisions_df(self, n_days=3):
        """Create decisions spanning multiple days."""
        n = n_days * 1440  # minutes per day
        timestamps = pd.date_range("2025-01-06", periods=n, freq="min")  # Monday
        return pd.DataFrame({
            "timestamp": timestamps,
            "action": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
            "true_aoi": np.random.rand(n) * 10,
            "current_aoi": np.random.rand(n) * 10 + 1,
            "reward": np.random.randn(n),
            "last_sync_items": np.random.rand(n),
        })

    def test_heatmap_shape(self):
        from sync_policy_dashboard import compute_temporal_heatmap_data
        df = self._make_decisions_df(n_days=7)
        result = compute_temporal_heatmap_data(df, "true_aoi")

        self.assertEqual(result.shape[0], 24)  # 24 hours
        self.assertEqual(result.shape[1], 7)   # 7 days
        self.assertEqual(list(result.columns), [
            "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
        ])

    def test_heatmap_metrics(self):
        from sync_policy_dashboard import compute_temporal_heatmap_data
        df = self._make_decisions_df(n_days=3)

        for metric in ["true_aoi", "reward", "sync_rate", "empty_rate"]:
            result = compute_temporal_heatmap_data(df, metric)
            self.assertEqual(result.shape[0], 24)

    def test_heatmap_falls_back_to_current_aoi(self):
        from sync_policy_dashboard import compute_temporal_heatmap_data
        df = self._make_decisions_df(n_days=1)
        df = df.drop(columns=["true_aoi"])
        result = compute_temporal_heatmap_data(df, "true_aoi")
        self.assertEqual(result.shape[0], 24)
        # Should not be all NaN (used current_aoi)
        self.assertFalse(result.isna().all().all())


# ModelPredictionPolicy Tests


class TestModelPredictionPolicy(unittest.TestCase):

    def test_tte_syncs_below_threshold(self):
        policy = ModelPredictionPolicy(strategy="tte", threshold=3.0)
        policy.set_predictions(np.array([10.0, 5.0, 2.0, 8.0, 1.5]))
        obs = np.zeros(4, dtype=np.float32)

        actions = [policy.predict(obs) for _ in range(5)]
        # Steps 2 (2.0 < 3.0) and 4 (1.5 < 3.0) should sync
        self.assertEqual(actions, [0, 0, 1, 0, 1])

    def test_tte_no_sync_above_threshold(self):
        policy = ModelPredictionPolicy(strategy="tte", threshold=3.0)
        policy.set_predictions(np.array([10.0, 5.0, 8.0]))
        obs = np.zeros(4, dtype=np.float32)

        actions = [policy.predict(obs) for _ in range(3)]
        self.assertEqual(actions, [0, 0, 0])

    def test_er_cumulative_sync(self):
        policy = ModelPredictionPolicy(strategy="er", threshold=2.0)
        policy.set_predictions(np.array([0.5, 0.8, 0.9, 0.3, 1.5]))
        obs = np.zeros(4, dtype=np.float32)

        actions = [policy.predict(obs) for _ in range(5)]
        # Cumulative: 0.5, 1.3, 2.2→sync(reset), 0.3, 1.8
        self.assertEqual(actions[2], 1)
        self.assertEqual(sum(actions), 1)

    def test_er_resets_on_sync(self):
        policy = ModelPredictionPolicy(strategy="er", threshold=1.0)
        policy.set_predictions(np.array([0.6, 0.6, 0.6, 0.6]))
        obs = np.zeros(4, dtype=np.float32)

        actions = [policy.predict(obs) for _ in range(4)]
        # Cumulative: 0.6, 1.2→sync, 0.6, 1.2→sync
        self.assertEqual(actions, [0, 1, 0, 1])

    def test_reset_clears_state(self):
        policy = ModelPredictionPolicy(strategy="er", threshold=2.0)
        policy.set_predictions(np.array([1.0, 1.5, 0.5, 0.5, 2.0]))
        obs = np.zeros(4, dtype=np.float32)

        policy.predict(obs)  # step 0, cumulative=1.0
        policy.predict(obs)  # step 1, cumulative=2.5→sync
        policy.reset()

        # After reset, step=0 and accumulator=0
        action = policy.predict(obs)  # step 0 again, cumulative=1.0
        self.assertEqual(action, 0)

    def test_no_predictions_returns_skip(self):
        policy = ModelPredictionPolicy(strategy="tte", threshold=5.0)
        obs = np.zeros(4, dtype=np.float32)
        self.assertEqual(policy.predict(obs), 0)

    def test_out_of_bounds_returns_skip(self):
        policy = ModelPredictionPolicy(strategy="tte", threshold=100.0)
        policy.set_predictions(np.array([1.0, 1.0]))
        obs = np.zeros(4, dtype=np.float32)

        actions = [policy.predict(obs) for _ in range(4)]
        # Steps 0,1 have predictions (1.0 < 100 → sync), steps 2,3 out of bounds → skip
        self.assertEqual(actions, [1, 1, 0, 0])

    def test_get_name_default(self):
        policy = ModelPredictionPolicy(strategy="tte", threshold=3.0)
        self.assertEqual(policy.get_name(), "model_pred_tte_3.0")

    def test_get_name_custom(self):
        policy = ModelPredictionPolicy(
            strategy="er", threshold=2.0, name="my_policy"
        )
        self.assertEqual(policy.get_name(), "my_policy")

    def test_invalid_strategy_raises(self):
        with self.assertRaises(ValueError):
            ModelPredictionPolicy(strategy="invalid")

    def test_load_heuristic_model_prediction(self):
        policy = load_heuristic_policy({
            "name": "model_prediction",
            "strategy": "tte",
            "threshold": 5.0,
        })
        self.assertIsInstance(policy, ModelPredictionPolicy)
        self.assertEqual(policy._strategy, "tte")
        self.assertAlmostEqual(policy._threshold, 5.0)

    def test_with_run_evaluation(self):
        """Integration: ModelPredictionPolicy through the full eval loop."""
        env = _make_test_env(n_steps=50)
        policy = ModelPredictionPolicy(strategy="er", threshold=1.5)

        # Use same predictions the env has
        preds = np.random.rand(50).astype(np.float32) * 2
        policy.set_predictions(preds)

        decisions_df, summary = run_evaluation(policy, env)

        self.assertEqual(len(decisions_df), 50)
        self.assertIn("sync_count", summary)
        self.assertGreater(summary["sync_count"], 0)

    def test_available_heuristics_includes_model_prediction(self):
        self.assertIn("model_prediction", get_available_heuristics())


if __name__ == "__main__":
    unittest.main()
