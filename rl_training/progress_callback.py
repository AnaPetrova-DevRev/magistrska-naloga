"""Progress logging callback for SB3 training."""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Optional

from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    from rl_training.rl_mlflow_tracking import RLMlflowTracker

logger = logging.getLogger(__name__)


class ProgressLogCallback(BaseCallback):
    """Logs training progress and episode metrics at regular intervals.

    Optionally logs metrics to an MLflow tracker when provided.
    """

    _SB3_TRAIN_METRICS = {
        "train/approx_kl": "train_approx_kl",
        "train/clip_fraction": "train_clip_fraction",
        "train/entropy_loss": "train_entropy_loss",
        "train/explained_variance": "train_explained_variance",
        "train/learning_rate": "train_learning_rate",
        "train/loss": "train_loss",
        "train/n_updates": "train_n_updates",
        "train/policy_gradient_loss": "train_policy_gradient_loss",
        "train/value_loss": "train_value_loss",
    }

    def __init__(
        self,
        log_interval: int = 10000,
        total_timesteps: int = 0,
        agent_name: str = "",
        tracker: Optional[RLMlflowTracker] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.total_timesteps = total_timesteps
        self.agent_name = agent_name
        self.tracker = tracker
        self._start_time: float = 0.0
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _training_metrics(self) -> dict[str, float]:
        """Return finite optimizer metrics exposed by the SB3 logger."""
        model_logger = getattr(getattr(self, "model", None), "logger", None)
        values = getattr(model_logger, "name_to_value", {})
        metrics = {}
        for source, target in self._SB3_TRAIN_METRICS.items():
            if source not in values:
                continue
            try:
                value = float(values[source])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                metrics[target] = value
        return metrics

    def _on_step(self) -> bool:
        # Collect episode stats from SB3 info dicts
        if "infos" in self.locals:
            for info in self.locals["infos"]:
                if "episode" in info:
                    self._episode_rewards.append(info["episode"]["r"])
                    self._episode_lengths.append(info["episode"]["l"])

        n_envs = getattr(self, "n_envs", 1)
        if self.num_timesteps % self.log_interval < n_envs:
            elapsed = time.time() - self._start_time
            pct = self.num_timesteps / self.total_timesteps * 100 if self.total_timesteps else 0
            steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0
            remaining = (
                (self.total_timesteps - self.num_timesteps) / steps_per_sec
                if steps_per_sec > 0 else 0
            )

            # Compute recent episode stats
            mean_reward = (
                sum(self._episode_rewards[-100:]) / len(self._episode_rewards[-100:])
                if self._episode_rewards else 0.0
            )
            mean_length = (
                sum(self._episode_lengths[-100:]) / len(self._episode_lengths[-100:])
                if self._episode_lengths else 0.0
            )

            logger.info(
                "[%s] %d/%d steps (%.1f%%) | %.0f steps/s | ~%.0fs remaining"
                " | ep_reward=%.2f | episodes=%d",
                self.agent_name,
                self.num_timesteps,
                self.total_timesteps,
                pct,
                steps_per_sec,
                remaining,
                mean_reward,
                len(self._episode_rewards),
            )

            if self.tracker is not None:
                metrics = {
                    "mean_episode_reward": mean_reward,
                    "mean_episode_length": mean_length,
                    "episodes_completed": len(self._episode_rewards),
                    "steps_per_sec": steps_per_sec,
                }
                metrics.update(self._training_metrics())
                self.tracker.log_metrics(metrics, step=self.num_timesteps)

        return True
