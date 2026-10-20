"""MLflow tracking utilities for RL agent training.

Adapted from model_training/mlflow_tracking.py for RL-specific
parameters (PPO hyperparams, reward config, env config).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from modeling.mlflow_utils import (
    MLFLOW_AVAILABLE,
    flatten_dict,
    is_mlflow_available,
    mlflow,
)

logger = logging.getLogger(__name__)


class RLMlflowTracker:
    """Context manager for MLflow tracking of RL training runs.

    Example:
        with RLMlflowTracker(run_config) as tracker:
            model = PPO("MlpPolicy", env)
            model.learn(total_timesteps=100000)
            tracker.log_metrics({"mean_reward": 0.5})
            tracker.log_artifacts(artifact_dir)
    """

    def __init__(
        self,
        sync_unit_name: str,
        task_type: str,
        agent_config: Dict[str, Any],
        env_config: Dict[str, Any],
        reward_config: Dict[str, Any],
        agent_name: str = "rl_agent",
        experiment_name: Optional[str] = None,
    ):
        self.sync_unit_name = sync_unit_name
        self.task_type = task_type
        self.agent_config = agent_config
        self.env_config = env_config
        self.reward_config = reward_config
        self.agent_name = agent_name
        self.experiment_name = (
            experiment_name
            or os.environ.get("MLFLOW_EXPERIMENT_NAME")
            or "rl_training"
        )
        self.run = None
        self.run_id = None
        self._active = False

    def __enter__(self) -> RLMlflowTracker:
        if not MLFLOW_AVAILABLE:
            logger.debug("MLflow not available - tracking disabled")
            return self

        try:
            mlflow.set_experiment(self.experiment_name)
            self.run = mlflow.start_run()
            self.run_id = self.run.info.run_id
            self._active = True

            self._log_initial_tags()
            self._log_params()
            logger.info("Started MLflow RL run: %s", self.run_id)

        except Exception as e:
            logger.warning("Failed to start MLflow run: %s", e)
            self._active = False

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self._active or not MLFLOW_AVAILABLE:
            return

        try:
            if exc_type is not None:
                mlflow.set_tag("run_status", "failed")
                mlflow.set_tag("error_type", exc_type.__name__)
                mlflow.set_tag("error_message", str(exc_val)[:500])
            else:
                mlflow.set_tag("run_status", "completed")

            mlflow.end_run()
            logger.info("Ended MLflow RL run: %s", self.run_id)

        except Exception as e:
            logger.warning("Error ending MLflow run: %s", e)

    def _log_initial_tags(self) -> None:
        if not self._active:
            return
        try:
            mlflow.set_tags({
                "sync_unit": self.sync_unit_name,
                "task_type": self.task_type,
                "agent_name": self.agent_name,
                "agent_type": "rl",
                "algorithm": self.agent_config.get("algorithm", "PPO"),
            })
        except Exception as e:
            logger.warning("Failed to log tags: %s", e)

    def _log_params(self) -> None:
        if not self._active:
            return
        try:
            params = {}
            params.update(flatten_dict(self.agent_config, "agent"))
            params.update(flatten_dict(self.reward_config, "reward"))

            # Log env config selectively (skip large arrays)
            env_safe = {
                k: v for k, v in self.env_config.items()
                if k in (
                    "episode_length", "prediction_horizon_steps",
                    "random_start", "observation_features",
                )
            }
            params.update(flatten_dict(env_safe, "env"))
            params = {k: str(v)[:500] for k, v in params.items()}

            if params:
                mlflow.log_params(params)
        except Exception as e:
            logger.warning("Failed to log params: %s", e)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        if not self._active or not MLFLOW_AVAILABLE:
            return
        try:
            numeric = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if numeric:
                mlflow.log_metrics(numeric, step=step)
        except Exception as e:
            logger.warning("Failed to log metrics: %s", e)

    def log_artifacts(self, artifact_dir: str) -> None:
        if not self._active or not MLFLOW_AVAILABLE:
            return
        try:
            if os.path.exists(artifact_dir):
                mlflow.log_artifacts(artifact_dir, artifact_path="rl_agent")
        except Exception as e:
            logger.warning("Failed to log artifacts: %s", e)

    def get_run_id(self) -> Optional[str]:
        return self.run_id
