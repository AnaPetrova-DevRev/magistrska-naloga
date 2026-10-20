"""RL training app for the orchestrator system.

This app handles training RL agents for sync scheduling decisions
and evaluating trained policies on held-out data.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app_base import App
from orchestrator_message import OrchestratorResponse
import storage_access
import sync_unit as sync_unit_module
from app_actions_constants import TRAIN_RL_AGENT, EVALUATE_RL_POLICY, EVALUATE_SYNC_POLICY
from modeling.core_types import (
    DatasetConfig,
    DataSnapshotMetadata,
    EventRateDataset,
    TaskType,
    TimeToEventDataset,
)
from modeling.dataset_builder import (
    build_event_rate_dataset,
    build_time_to_event_dataset,
)
from rl_training.reward import RewardConfig
from rl_training.sync_env import SyncSchedulingEnv
from rl_training.prediction_provider import (
    ActionAwarePredictionProvider,
    MaskedPredictionProvider,
    NoPredictionProvider,
    PrecomputedPredictionProvider,
)
from modeling.cron_pattern_detector import save_cron_patterns
from rl_training.rl_mlflow_tracking import RLMlflowTracker

logger = logging.getLogger(__name__)

RL_AGENTS_FOLDER = "rl_agents"


class RLTrainingApp(App):
    """App for training RL sync scheduling agents.

    Actions:
        train_rl_agent: Train a PPO agent on historical data.
        evaluate_rl_policy: Run a trained policy and save evaluation artifacts.
    """

    def __init__(self, sa: Optional[storage_access.StorageAccess] = None):
        self._sa = sa or storage_access.StorageAccess()

    def get_name(self) -> str:
        return "rl_training"

    def get_available_actions(self) -> list[str]:
        return [TRAIN_RL_AGENT, EVALUATE_RL_POLICY]

    def execute_action(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        action: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorResponse:
        if not self.supports_action(action):
            raise ValueError(
                f"Action '{action}' not supported by {self.get_name()}"
            )

        parameters = parameters or {}

        if action == TRAIN_RL_AGENT:
            return self._handle_train_rl_agent(sync_unit_obj, parameters)
        elif action == EVALUATE_RL_POLICY:
            return self._handle_evaluate_rl_policy(sync_unit_obj, parameters)

        raise ValueError(f"Unknown action: {action}")

    # train_rl_agent

    def _handle_train_rl_agent(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        parameters: Dict[str, Any],
    ) -> OrchestratorResponse:
        """Train one or more PPO agents on historical sync unit data.

        Supports two config styles:
        1. Single agent: top-level agent/env/reward/predictions/agent_name keys.
        2. Multiple agents: an "agents" list, each with its own
           agent/env/reward/predictions/agent_name overrides.
           Keys not specified in an agent spec fall back to top-level defaults.
        """
        task_type_str = parameters.get("task_type", "event_rate")
        dataset_params = parameters.get("dataset", {})
        prediction_params = parameters.get("predictions", {})
        prediction_task_type_str = prediction_params.get(
            "prediction_task_type", task_type_str
        )

        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            return OrchestratorResponse(
                success=False,
                message=f"Invalid task_type: {task_type_str}",
                is_done=True,
            )

        # Build dataset once (shared across all agent specs)
        try:
            dataset, snapshot_meta = self._build_dataset(
                sync_unit_obj, task_type, dataset_params
            )
        except Exception as e:
            logger.error("Failed to build dataset: %s", e, exc_info=True)
            return OrchestratorResponse(
                success=False,
                message=f"Failed to build dataset: {e}",
                is_done=True,
            )

        timestamps, actual_values, covariates = self._extract_data_arrays(
            dataset, task_type
        )
        train_end = len(dataset.train_indices)
        val_indices = getattr(dataset, "val_indices", [])
        test_indices = getattr(dataset, "test_indices", [])
        val_end = val_indices[-1] + 1 if val_indices else train_end

        # Record split boundaries for reproducible evaluation
        data_split = {
            "train_ratio": dataset_params.get("train_ratio", 0.8),
            "val_ratio": dataset_params.get("val_ratio", 0.2),
            "train_size": train_end,
            "val_size": len(val_indices),
            "test_size": len(test_indices),
            "train_end_timestamp": str(timestamps[train_end - 1]) if train_end > 0 else None,
            "val_end_timestamp": str(timestamps[val_end - 1]) if val_end > 0 else None,
            "data_start_timestamp": str(timestamps[0]) if timestamps else None,
            "data_end_timestamp": str(timestamps[-1]) if timestamps else None,
        }

        # Extract cron patterns if detected during dataset building
        cron_patterns = getattr(dataset, "cron_patterns", None)
        hourly_cron_patterns = getattr(dataset, "hourly_cron_patterns", None)

        # Prepare validation data for EvalCallback
        val_timestamps = timestamps[train_end:val_end] if val_indices else None
        val_actual_values = actual_values[train_end:val_end] if val_indices else None
        val_covariates = (
            {k: v[train_end:val_end] for k, v in covariates.items()}
            if val_indices else None
        )

        # Build list of agent specs (support single or multi)
        agent_specs = parameters.get("agents")
        if agent_specs is None:
            # Single-agent mode: wrap top-level params as one spec
            agent_specs = [{
                "agent": parameters.get("agent", {}),
                "env": parameters.get("env", {}),
                "reward": parameters.get("reward", {}),
                "agent_name": parameters.get("agent_name", "ppo_agent"),
            }]

        # Top-level defaults for multi-agent mode
        default_agent = parameters.get("agent", {})
        default_env = parameters.get("env", {})
        default_reward = parameters.get("reward", {})
        default_predictions = parameters.get("predictions", {})

        results: List[Dict[str, Any]] = []

        for spec in agent_specs:
            agent_params = {**default_agent, **spec.get("agent", {})}
            env_params = {**default_env, **spec.get("env", {})}
            reward_params = {**default_reward, **spec.get("reward", {})}
            per_agent_predictions = {**default_predictions, **spec.get("predictions", {})}
            agent_name = spec.get("agent_name", "ppo_agent")

            per_agent_pred_task_type = per_agent_predictions.get(
                "prediction_task_type", task_type_str
            )
            prediction_provider = self._build_prediction_provider(
                sync_unit_obj, per_agent_pred_task_type, per_agent_predictions,
                timestamps[:train_end], actual_values[:train_end],
            )

            # Resolve "auto" preset: calibrate from training data
            if reward_params.get("preset") == "auto":
                from rl_training.reward import RewardConfig
                overrides = {
                    k: v for k, v in reward_params.items()
                    if k != "preset" and k in RewardConfig.__dataclass_fields__
                }
                auto_cfg = RewardConfig.from_dataset(
                    actual_values[:train_end],
                    task_type=task_type_str,
                    **overrides,
                )
                reward_params = auto_cfg.to_dict()
                logger.info(
                    "Resolved 'auto' reward preset for '%s': %s",
                    agent_name, reward_params,
                )

            try:
                result = self._train_single_agent(
                    sync_unit_obj=sync_unit_obj,
                    task_type_str=task_type_str,
                    timestamps=timestamps[:train_end],
                    actual_values=actual_values[:train_end],
                    covariates={k: v[:train_end] for k, v in covariates.items()},
                    prediction_provider=prediction_provider,
                    agent_params=agent_params,
                    env_params=env_params,
                    reward_params=reward_params,
                    prediction_params=per_agent_predictions,
                    agent_name=agent_name,
                    cron_patterns=cron_patterns,
                    hourly_cron_patterns=hourly_cron_patterns,
                    data_split=data_split,
                    val_timestamps=val_timestamps,
                    val_actual_values=val_actual_values,
                    val_covariates=val_covariates,
                    prediction_task_type=per_agent_pred_task_type,
                )
                results.append({**result, "success": True})
            except Exception as e:
                logger.error(
                    "Error training agent '%s': %s", agent_name, e,
                    exc_info=True,
                )
                results.append({
                    "agent_name": agent_name,
                    "success": False,
                    "error": str(e),
                })

        successful_count = sum(1 for r in results if r.get("success"))
        total_count = len(results)
        overall_success = successful_count == total_count

        response = OrchestratorResponse(
            success=overall_success,
            message=(
                f"RL training complete: {successful_count}/{total_count} "
                f"agents succeeded"
            ),
            is_done=True,
        )
        response.data = {
            "results": results,
            "sync_unit": sync_unit_obj.get_name(),
            "task_type": task_type_str,
        }
        return response

    # Algorithm-specific hyperparameter keys accepted by each SB3 class.
    _ALGO_KWARGS: Dict[str, set] = {
        "PPO": {
            "learning_rate", "n_steps", "batch_size", "n_epochs",
            "gamma", "gae_lambda", "clip_range", "ent_coef",
        },
        "DQN": {
            "learning_rate", "buffer_size", "learning_starts",
            "batch_size", "tau", "gamma", "train_freq",
            "target_update_interval", "exploration_fraction",
            "exploration_initial_eps", "exploration_final_eps",
        },
        "A2C": {
            "learning_rate", "n_steps", "gamma", "gae_lambda",
            "ent_coef", "vf_coef", "max_grad_norm",
        },
    }

    # Default dataset passes per algorithm (used when total_timesteps="auto").
    _ALGO_DATASET_PASSES: Dict[str, int] = {
        "PPO": 3,
        "DQN": 8,
        "A2C": 5,
    }

    # Default hyperparameters per algorithm.
    _ALGO_DEFAULTS: Dict[str, Dict[str, Any]] = {
        "PPO": {
            "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,
            "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95,
            "clip_range": 0.2, "ent_coef": 0.01,
        },
        "DQN": {
            "learning_rate": 1e-4, "buffer_size": 100000,
            "learning_starts": 1000, "batch_size": 64, "tau": 1.0,
            "gamma": 0.99, "train_freq": 4, "target_update_interval": 1000,
            "exploration_fraction": 0.1, "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
        },
        "A2C": {
            "learning_rate": 7e-4, "n_steps": 5, "gamma": 0.99,
            "gae_lambda": 1.0, "ent_coef": 0.01, "vf_coef": 0.25,
            "max_grad_norm": 0.5,
        },
    }

    @staticmethod
    def _get_sb3_class(algorithm: str):
        """Import and return the SB3 algorithm class."""
        import stable_baselines3 as sb3
        algo_map = {
            "PPO": sb3.PPO,
            "DQN": sb3.DQN,
            "A2C": sb3.A2C,
        }
        if algorithm not in algo_map:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. "
                f"Supported: {list(algo_map.keys())}"
            )
        return algo_map[algorithm]

    def _train_single_agent(
        self,
        sync_unit_obj,
        task_type_str: str,
        timestamps: list,
        actual_values: list,
        covariates: dict,
        prediction_provider,
        agent_params: Dict[str, Any],
        env_params: Dict[str, Any],
        reward_params: Dict[str, Any],
        prediction_params: Dict[str, Any],
        agent_name: str,
        cron_patterns: Optional[list] = None,
        hourly_cron_patterns: Optional[list] = None,
        data_split: Optional[Dict[str, Any]] = None,
        val_timestamps: Optional[list] = None,
        val_actual_values: Optional[list] = None,
        val_covariates: Optional[dict] = None,
        prediction_task_type: str = "event_rate",
    ) -> Dict[str, Any]:
        """Train a single RL agent. Called for each spec in the agents list."""
        algorithm = agent_params.get("algorithm", "PPO").upper()
        AlgoClass = self._get_sb3_class(algorithm)
        seed = agent_params.get("seed", None)

        env_config = self._build_env_config(
            timestamps=timestamps,
            actual_values=actual_values,
            covariates=covariates,
            prediction_provider=prediction_provider,
            task_type=task_type_str,
            env_params=env_params,
            reward_params=reward_params,
            prediction_task_type=prediction_task_type,
        )

        artifact_dir = self._get_artifact_dir(sync_unit_obj, agent_name)
        os.makedirs(artifact_dir, exist_ok=True)

        # Vectorized environments: run n_envs in parallel for faster training.
        # SubprocVecEnv uses separate processes (bypasses GIL).
        # n_envs=1 (default) uses a single env with no overhead.
        n_envs = agent_params.get("n_envs", 1)

        if n_envs > 1:
            from stable_baselines3.common.vec_env import SubprocVecEnv

            def _make_env(env_seed: int):
                def _init():
                    e = SyncSchedulingEnv(env_config)
                    e.reset(seed=env_seed)
                    return e
                return _init

            base_seed = seed if seed is not None else 0
            env = SubprocVecEnv([_make_env(base_seed + i) for i in range(n_envs)])
            logger.info("Using %d parallel environments (SubprocVecEnv)", n_envs)
        else:
            env = SyncSchedulingEnv(env_config)

        # Resolve total_timesteps: explicit int, "auto", or default.
        # "auto" scales with training data: dataset_passes * data_length.
        raw_timesteps = agent_params.get("total_timesteps", "auto")
        default_passes = self._ALGO_DATASET_PASSES.get(algorithm, 3)
        dataset_passes = agent_params.get("dataset_passes", default_passes)
        train_data_length = len(timestamps)

        if raw_timesteps == "auto" or raw_timesteps is None:
            total_timesteps = dataset_passes * train_data_length
            logger.info(
                "Auto timesteps: %d passes × %d steps = %d total",
                dataset_passes, train_data_length, total_timesteps,
            )
        else:
            total_timesteps = int(raw_timesteps)

        policy_network = agent_params.get("policy_network", [64, 64])

        # Build algorithm-specific kwargs from config + defaults
        defaults = self._ALGO_DEFAULTS.get(algorithm, {})
        valid_keys = self._ALGO_KWARGS.get(algorithm, set())
        algo_kwargs = {}
        for key in valid_keys:
            if key in agent_params:
                algo_kwargs[key] = agent_params[key]
            elif key in defaults:
                algo_kwargs[key] = defaults[key]

        policy_kwargs = {"net_arch": policy_network}

        # Device: "auto" picks GPU if available (CUDA > MPS > CPU)
        device = agent_params.get("device", "auto")

        with RLMlflowTracker(
            sync_unit_name=sync_unit_obj.get_name(),
            task_type=task_type_str,
            agent_config={**agent_params, "algorithm": algorithm},
            env_config=env_params,
            reward_config=reward_params,
            agent_name=agent_name,
        ) as tracker:
            model = AlgoClass(
                "MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                device=device,
                seed=seed,
                verbose=0,
                **algo_kwargs,
            )

            from stable_baselines3.common.callbacks import CallbackList, EvalCallback
            from rl_training.progress_callback import ProgressLogCallback

            log_interval = max(total_timesteps // 20, 1000)
            progress_cb = ProgressLogCallback(
                log_interval=log_interval,
                total_timesteps=total_timesteps,
                agent_name=agent_name,
                tracker=tracker,
            )

            callbacks = [progress_cb]

            # Validation evaluation during training (saves best model)
            val_env = None
            if val_timestamps is not None and len(val_timestamps) > 0:
                validation_params = self._validation_env_params(
                    env_params, len(val_timestamps)
                )
                val_env_config = self._build_env_config(
                    timestamps=val_timestamps,
                    actual_values=val_actual_values,
                    covariates=val_covariates,
                    prediction_provider=prediction_provider,
                    task_type=task_type_str,
                    env_params=validation_params,
                    reward_params=reward_params,
                    prediction_task_type=prediction_task_type,
                    bootstrap_timestamps=timestamps,
                    bootstrap_actual_values=actual_values,
                )
                val_env = SyncSchedulingEnv(val_env_config)
                best_model_dir = os.path.join(artifact_dir, "best_model")
                eval_log_dir = os.path.join(artifact_dir, "eval_logs")
                eval_freq = self._validation_eval_freq(
                    total_timesteps, agent_params,
                )

                eval_cb = EvalCallback(
                    val_env,
                    best_model_save_path=best_model_dir,
                    log_path=eval_log_dir,
                    eval_freq=eval_freq,
                    n_eval_episodes=max(
                        1, len(validation_params.get("episode_start_indices", []))
                    ),
                    deterministic=True,
                    verbose=0,
                )
                callbacks.append(eval_cb)
                logger.info(
                    "EvalCallback enabled: eval every %d steps on %d val steps "
                    "(episode cap: %s; starts: %s)",
                    eval_freq, len(val_timestamps),
                    val_env_config["episode_length"] or "full split",
                    validation_params.get("episode_start_indices", [0]),
                )

            model.learn(
                total_timesteps=total_timesteps,
                callback=CallbackList(callbacks),
            )

            model_path = os.path.join(artifact_dir, "agent")
            model.save(model_path)

            self._save_training_metadata(
                artifact_dir, sync_unit_obj, task_type_str,
                {**agent_params, "algorithm": algorithm},
                env_params, reward_params,
                prediction_params, agent_name, total_timesteps,
                tracker.get_run_id(),
                cron_patterns=cron_patterns,
                hourly_cron_patterns=hourly_cron_patterns,
                data_split=data_split,
                prediction_cache_metadata=(
                    prediction_provider.metadata
                    if hasattr(prediction_provider, "metadata")
                    else None
                ),
            )

            tracker.log_artifacts(artifact_dir)

        # Clean up SubprocVecEnv processes
        if n_envs > 1:
            env.close()

        logger.info(
            "RL agent '%s' (%s) trained and saved to %s",
            agent_name, algorithm, artifact_dir,
        )

        return {
            "agent_name": agent_name,
            "algorithm": algorithm,
            "artifact_path": artifact_dir,
            "total_timesteps": total_timesteps,
            "mlflow_run_id": tracker.get_run_id(),
        }

    # evaluate_rl_policy (deprecated — redirects to SyncPolicyEvaluationApp)

    def _handle_evaluate_rl_policy(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        parameters: Dict[str, Any],
    ) -> OrchestratorResponse:
        """Evaluate a trained RL policy. Delegates to SyncPolicyEvaluationApp.

        .. deprecated::
            Use SyncPolicyEvaluationApp with action ``evaluate_sync_policy``
            instead — it supports true AoI, eval_split, and split boundary
            validation from training metadata.
        """
        import warnings
        warnings.warn(
            "RLTrainingApp.evaluate_rl_policy is deprecated. "
            "Use SyncPolicyEvaluationApp.evaluate_sync_policy instead, "
            "which supports true AoI, eval_split, and split validation.",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.warning(
            "evaluate_rl_policy is deprecated; redirecting to "
            "SyncPolicyEvaluationApp.evaluate_sync_policy"
        )

        from sync_policy_evaluation.sync_policy_evaluation_app import (
            SyncPolicyEvaluationApp,
        )
        eval_app = SyncPolicyEvaluationApp(self._sa)

        eval_params = {
            **parameters,
            "policy_type": "trained",
            "policy_name": parameters.get("policy_name", "rl_eval"),
        }
        return eval_app.execute_action(
            sync_unit_obj, EVALUATE_SYNC_POLICY, eval_params
        )

    # Helpers

    def _build_dataset(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        task_type: TaskType,
        dataset_params: Dict[str, Any],
    ) -> Tuple:
        start_time_str = dataset_params.get("start_time")
        end_time_str = dataset_params.get("end_time")

        if start_time_str:
            start_time = datetime.fromisoformat(
                start_time_str.replace("Z", "+00:00")
            )
        else:
            start_time = datetime.min.replace(tzinfo=UTC)

        if end_time_str:
            end_time = datetime.fromisoformat(
                end_time_str.replace("Z", "+00:00")
            )
        else:
            end_time = datetime.max.replace(tzinfo=UTC)

        # RL environments step minute-by-minute and assume every step is a
        # real calendar minute.  Without resampling the ER dataset is sparse
        # (only minutes with events), which silently breaks AoI tracking and
        # temporal features.  Force "1min" for event_rate datasets.
        resample_freq = dataset_params.get("resample_freq")
        if task_type == TaskType.EVENT_RATE:
            if resample_freq is None:
                resample_freq = "1min"
                logger.info(
                    "resample_freq not set — defaulting to '1min' for RL "
                    "event_rate dataset"
                )
            elif resample_freq != "1min":
                logger.warning(
                    "resample_freq='%s' overridden to '1min' — RL environment "
                    "requires minute-level event_rate data",
                    resample_freq,
                )
                resample_freq = "1min"

        config = DatasetConfig(
            task_type=task_type,
            start_time=start_time,
            end_time=end_time,
            train_ratio=dataset_params.get("train_ratio", 0.8),
            val_ratio=dataset_params.get("val_ratio", 0.2),
            filters=dataset_params.get("filters", {}),
            resample_freq=resample_freq,
            covariates=dataset_params.get("covariates", []),
        )

        if task_type == TaskType.EVENT_RATE:
            return build_event_rate_dataset(sync_unit_obj, config, self._sa)
        elif task_type == TaskType.TIME_TO_EVENT:
            return build_time_to_event_dataset(sync_unit_obj, config, self._sa)
        else:
            raise NotImplementedError(
                f"Dataset building for {task_type.value} not implemented"
            )

    def _extract_data_arrays(
        self, dataset, task_type: TaskType
    ) -> Tuple[list, list, dict]:
        """Extract timestamps, values, and covariates from a dataset."""
        timestamps = list(dataset.timestamps)
        if task_type == TaskType.EVENT_RATE:
            actual_values = list(dataset.events_per_minute)
        else:
            actual_values = list(dataset.minutes_to_next_event)
        covariates = {k: list(v) for k, v in dataset.covariates.items()}
        return timestamps, actual_values, covariates

    def _build_prediction_provider(
        self,
        sync_unit_obj,
        prediction_task_type: str,
        prediction_params: Dict[str, Any],
        timestamps: list,
        actual_values: list,
    ):
        """Build a PredictionProvider based on config."""
        mode = prediction_params.get("mode", "none")

        if mode == "masked":
            return MaskedPredictionProvider(prediction_task_type)

        if mode in ("precomputed", "causal_precomputed", "explicit"):
            cache_path = prediction_params.get("cache_path")
            if not cache_path:
                raise ValueError(
                    f"predictions.mode='{mode}' requires a causal cache_path; "
                    "long-horizon positional generation is disabled"
                )
            provider = PrecomputedPredictionProvider.from_cache(cache_path)
            provider.validate_timestamps(timestamps)
            cache_task = provider.metadata.get("task_type")
            if cache_task and cache_task != prediction_task_type:
                raise ValueError(
                    f"Prediction cache task_type={cache_task!r} does not match "
                    f"requested {prediction_task_type!r}"
                )
            logger.info(
                "Loaded causal prediction cache with %d requested timestamps from %s",
                len(timestamps), cache_path,
            )
            return provider

        elif mode == "action_aware":
            from rl_training.action_aware_forecaster import ExplicitHistoryForecaster

            model_selection = prediction_params.get("model_selection", {})
            if model_selection.get("mode") != "explicit":
                raise ValueError(
                    "action_aware predictions require explicit model_selection"
                )
            model_path = model_selection.get("model_id")
            if not model_path:
                raise ValueError("action_aware predictions require a model_id")
            forecaster = ExplicitHistoryForecaster.from_model_path(
                model_path, prediction_task_type,
            )
            fitting_cutoff = prediction_params.get(
                "forecasting_model_fitting_cutoff"
            )
            if fitting_cutoff is not None:
                forecaster.metadata["forecasting_model_fitting_cutoff"] = fitting_cutoff
                forecaster.metadata["fitting_cutoff_source"] = "prediction_config"
                forecaster.metadata["fitting_cutoff_semantics"] = (
                    prediction_params.get(
                        "fitting_cutoff_semantics",
                        "last_training_target_timestamp",
                    )
                )
            return ActionAwarePredictionProvider(
                forecaster,
                trajectory_horizon_steps=prediction_params.get(
                    "trajectory_horizon_steps", 60
                ),
            )

        elif mode == "live":
            from model_serving.model_serving_app import ModelServingApp
            from rl_training.prediction_provider import LivePredictionProvider

            serving_app = ModelServingApp(self._sa)
            return LivePredictionProvider(
                model_serving_app=serving_app,
                sync_unit_obj=sync_unit_obj,
                task_type=prediction_task_type,
                call_interval_minutes=prediction_params.get(
                    "call_interval_minutes", 15
                ),
                horizon_minutes=prediction_params.get("horizon_minutes", 60),
                model_selection=prediction_params.get("model_selection"),
            )

        return NoPredictionProvider()

    @staticmethod
    def _validation_eval_freq(
        total_timesteps: int, agent_params: Dict[str, Any],
    ) -> int:
        """Return callback frequency for the requested validation count."""
        evaluations = int(agent_params.get("validation_evaluations", 10))
        if evaluations <= 0:
            raise ValueError("validation_evaluations must be positive")
        return max(total_timesteps // evaluations, 5000)

    @staticmethod
    def _validation_env_params(
        env_params: Dict[str, Any], total_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build deterministic, split-spanning validation window settings."""
        episode_length = env_params.get("validation_episode_length")
        if episode_length is not None:
            episode_length = int(episode_length)
            if episode_length <= 0:
                raise ValueError("validation_episode_length must be positive")
        result = {
            **env_params,
            "episode_length": episode_length,
            "random_start": False,
        }
        if episode_length is not None and total_steps is not None:
            window_count = int(env_params.get("validation_window_count", 3))
            if window_count <= 0:
                raise ValueError("validation_window_count must be positive")
            max_start = max(0, total_steps - episode_length)
            count = min(window_count, max_start + 1)
            if count == 1:
                starts = [0]
            else:
                starts = sorted({
                    round(index * max_start / (count - 1))
                    for index in range(count)
                })
            result["episode_start_indices"] = starts
        return result

    def _build_env_config(
        self,
        timestamps,
        actual_values,
        covariates,
        prediction_provider,
        task_type,
        env_params,
        reward_params,
        prediction_task_type: str = "event_rate",
        bootstrap_timestamps=None,
        bootstrap_actual_values=None,
    ) -> Dict[str, Any]:
        episode_length = env_params.get("episode_length")
        random_start = env_params.get("random_start", False)

        if episode_length is not None and "random_start" not in env_params and not random_start:
            random_start = True
            logger.warning(
                "episode_length=%d set without explicit random_start. "
                "Defaulting random_start=True to avoid training only on "
                "the first %d minutes. Set random_start=False explicitly "
                "to suppress this warning.",
                episode_length, episode_length,
            )

        obs_features = list(env_params.get("observation_features", [
            "time_since_last_sync",
            "current_aoi",
            "predictions",
            "covariates",
            "items_synced_last",
            "cumulative_empty_syncs",
        ]))

        if isinstance(prediction_provider, NoPredictionProvider) and "predictions" in obs_features:
            obs_features.remove("predictions")
            logger.warning(
                "Removed 'predictions' from observation_features because "
                "prediction provider is NoPredictionProvider (mode='none'). "
                "This avoids %d wasted zero-valued features.",
                env_params.get("prediction_horizon_steps", 60),
            )

        return {
            "timestamps": timestamps,
            "actual_values": actual_values,
            "covariates": covariates,
            "prediction_provider": prediction_provider,
            "task_type": task_type,
            "prediction_task_type": prediction_task_type,
            "episode_length": episode_length,
            "random_start": random_start,
            "episode_start_indices": env_params.get("episode_start_indices", []),
            "bootstrap_history_steps": env_params.get(
                "bootstrap_history_steps", 0
            ),
            "bootstrap_timestamps": bootstrap_timestamps or [],
            "bootstrap_actual_values": bootstrap_actual_values or [],
            "prediction_horizon_steps": env_params.get(
                "prediction_horizon_steps", 60
            ),
            "observation_features": obs_features,
            "empty_sync_window": env_params.get("empty_sync_window", 60),
            "reward_config": reward_params,
        }

    def _get_artifact_dir(
        self, sync_unit_obj, agent_name: str
    ) -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%f")
        folder_name = f"{agent_name}_{timestamp}"
        base_path = os.path.join("data", sync_unit_obj.get_path_for_sync_unit())
        return os.path.join(base_path, RL_AGENTS_FOLDER, folder_name)

    def _save_training_metadata(
        self,
        artifact_dir: str,
        sync_unit_obj,
        task_type: str,
        agent_params: dict,
        env_params: dict,
        reward_params: dict,
        prediction_params: dict,
        agent_name: str,
        total_timesteps: int,
        mlflow_run_id: Optional[str],
        cron_patterns: Optional[list] = None,
        hourly_cron_patterns: Optional[list] = None,
        data_split: Optional[Dict[str, Any]] = None,
        prediction_cache_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        metadata = {
            "sync_unit": sync_unit_obj.get_name(),
            "task_type": task_type,
            "agent_name": agent_name,
            "total_timesteps": total_timesteps,
            "agent_config": agent_params,
            "env_config": env_params,
            "reward_config": reward_params,
            "prediction_config": prediction_params,
            "prediction_cache_metadata": prediction_cache_metadata,
            "mlflow_run_id": mlflow_run_id,
            "trained_at": datetime.now(UTC).isoformat(),
        }
        if data_split is not None:
            metadata["data_split"] = data_split
        if cron_patterns is not None:
            metadata["cron_patterns"] = cron_patterns
        if hourly_cron_patterns is not None:
            metadata["hourly_cron_patterns"] = hourly_cron_patterns

        with open(os.path.join(artifact_dir, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        with open(os.path.join(artifact_dir, "reward_config.json"), "w") as f:
            json.dump(reward_params, f, indent=2)

        with open(os.path.join(artifact_dir, "env_config.json"), "w") as f:
            json.dump(env_params, f, indent=2, default=str)

        has_any_patterns = (
            cron_patterns is not None or hourly_cron_patterns is not None
        )
        if has_any_patterns:
            from modeling.cron_pattern_detector import (
                CronPattern,
                HourlyCronPattern,
            )
            daily = (
                [CronPattern(**p) for p in cron_patterns]
                if cron_patterns else []
            )
            hourly = (
                [HourlyCronPattern(**p) for p in hourly_cron_patterns]
                if hourly_cron_patterns else []
            )
            save_cron_patterns(
                daily,
                os.path.join(artifact_dir, "cron_patterns.json"),
                hourly_patterns=hourly or None,
            )
