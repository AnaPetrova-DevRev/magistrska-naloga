"""Gymnasium environment for sync scheduling over historical data.

Replays preprocessed time-series data minute-by-minute. The agent
decides each step whether to sync (action=1) or skip (action=0).
Observations, rewards, and episode boundaries are all configurable.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from collections import deque

from rl_training.observation_builder import (
    ALL_FEATURES,
    ObservationBuilder,
    ObservationSpec,
)
from rl_training.prediction_provider import (
    NoPredictionProvider,
    PredictionProvider,
)
from rl_training.reward import RewardConfig, RewardFunction

logger = logging.getLogger(__name__)


class SyncSchedulingEnv(gym.Env):
    """Gymnasium environment for training sync scheduling agents.

    The environment steps through historical time-series data one minute
    at a time. At each step the agent observes state features (AoI,
    predictions, covariates, etc.) and decides whether to sync.

    Supports configurable:
    - Observation features (via observation_features list)
    - Reward function (via reward_config dict)
    - Episode length (fixed windows or full dataset)
    - Prediction source (via PredictionProvider)

    ## AoI note (training proxy vs. true staleness)

    ``current_aoi`` in this environment is a **training proxy**: it
    increments every minute we don't sync and resets to 0 on every sync,
    regardless of whether events exist. This is intentional — as a
    reward-shaping signal it encourages the agent to sync proactively
    even before events appear, and it provides a dense gradient during
    calm periods.

    This is NOT the same as **true Age of Information**, which measures
    how stale our local data actually is relative to the external system.
    True AoI stays at 0 during calm periods (no unsynced events = no
    staleness) and only starts growing when an event appears that we
    haven't synced yet.

    For evaluation metrics the true AoI is computed in
    ``sync_policy_evaluation.evaluation_engine``. See the module
    docstring there and ``docs/sync_policy_evaluation.md`` §Age of
    Information for details.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env_config: Dict[str, Any]):
        """
        Args:
            env_config: Configuration dict with keys:
                timestamps: List[datetime] — minute-level timestamps.
                actual_values: List[float] — ground truth values per step.
                    For event_rate: events per minute.
                    For time_to_event: minutes to next event.
                covariates: Dict[str, List[float]] — covariate arrays.
                prediction_provider: PredictionProvider instance.
                reward_config: dict — passed to RewardConfig.from_dict().
                task_type: str — "event_rate" or "time_to_event".
                episode_length: Optional[int] — steps per episode.
                    None means use full dataset.
                prediction_horizon_steps: int — future prediction values
                    included in observation (default 60).
                observation_features: List[str] — which features to include.
                random_start: bool — random episode start offset (default False).
        """
        super().__init__()

        self._timestamps: List[datetime] = list(env_config["timestamps"])
        self._actual_values: np.ndarray = np.asarray(
            env_config["actual_values"], dtype=np.float32
        )
        self._covariates: Dict[str, List[float]] = env_config.get("covariates", {})
        self._task_type: str = env_config.get("task_type", "event_rate")
        self._total_steps: int = len(self._timestamps)

        source_provider: PredictionProvider = env_config.get(
            "prediction_provider", NoPredictionProvider()
        )
        self._prediction_provider = source_provider.clone()
        self._bootstrap_timestamps = list(env_config.get("bootstrap_timestamps", []))
        self._bootstrap_values = list(env_config.get("bootstrap_actual_values", []))
        if len(self._bootstrap_timestamps) != len(self._bootstrap_values):
            raise ValueError("bootstrap timestamps and values must have equal length")

        reward_cfg = env_config.get("reward_config", {})
        if not isinstance(reward_cfg, RewardConfig):
            reward_cfg = RewardConfig.from_dict({**reward_cfg, "mode": self._task_type})
        self._reward_fn = RewardFunction(reward_cfg)

        self._episode_length: Optional[int] = env_config.get("episode_length", None)
        self._random_start: bool = env_config.get("random_start", False)
        self._episode_start_indices = [
            int(value) for value in env_config.get("episode_start_indices", [])
        ]
        if self._episode_start_indices and self._episode_length is None:
            raise ValueError("episode_start_indices requires episode_length")
        if self._episode_start_indices and self._random_start:
            raise ValueError("episode_start_indices is incompatible with random_start")
        max_start = max(0, self._total_steps - (self._episode_length or self._total_steps))
        if any(value < 0 or value > max_start for value in self._episode_start_indices):
            raise ValueError("episode_start_indices contains an out-of-range start")
        self._episode_start_cursor = 0
        self._bootstrap_history_steps: int = int(
            env_config.get("bootstrap_history_steps", 0)
        )
        self._prediction_horizon: int = env_config.get(
            "prediction_horizon_steps", 60
        )
        self._empty_sync_window: int = env_config.get("empty_sync_window", 60)

        obs_features = env_config.get("observation_features", [
            "time_since_last_sync",
            "current_aoi",
            "predictions",
            "covariates",
            "items_synced_last",
            "cumulative_empty_syncs",
        ])
        invalid_features = set(obs_features) - set(ALL_FEATURES)
        if invalid_features:
            raise ValueError(
                f"Invalid observation features: {invalid_features}. "
                f"Valid features: {ALL_FEATURES}"
            )
        covariate_names = list(self._covariates.keys())
        self._obs_spec = ObservationSpec(
            features=obs_features,
            prediction_horizon_steps=self._prediction_horizon,
            covariate_names=covariate_names,
            prediction_task_type=env_config.get("prediction_task_type", "event_rate"),
        )
        self._obs_builder = ObservationBuilder(self._obs_spec)

        self.action_space = spaces.Discrete(2)  # 0=skip, 1=sync
        obs_dim = self._obs_spec.dimension
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._current_step: int = 0
        self._episode_start: int = 0
        self._time_since_last_sync: float = 0.0
        self._current_aoi: float = 0.0
        self._cumulative_empty_syncs: int = 0
        self._empty_sync_timestamps: deque = deque()  # step indices of recent empty syncs
        self._last_sync_items: float = 0.0
        self._pending_items: float = 0.0  # events accumulated since last sync
        self._episode_reward: float = 0.0
        self._step_count_in_episode: int = 0
        self._last_revealed_step: int = -1

    @property
    def obs_spec(self) -> ObservationSpec:
        return self._obs_spec

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new episode.

        Returns:
            Tuple of (observation, info_dict).
        """
        super().reset(seed=seed)

        if self._episode_length is not None:
            max_start = max(0, self._total_steps - self._episode_length)
            if self._episode_start_indices:
                index = self._episode_start_cursor % len(self._episode_start_indices)
                self._episode_start = self._episode_start_indices[index]
                self._episode_start_cursor += 1
            elif self._random_start and max_start > 0:
                min_start = min(self._bootstrap_history_steps, max_start)
                self._episode_start = self.np_random.integers(
                    min_start, max_start + 1
                )
            else:
                self._episode_start = 0
        else:
            self._episode_start = 0

        self._current_step = self._episode_start
        self._time_since_last_sync = 0.0
        self._current_aoi = 0.0
        self._cumulative_empty_syncs = 0
        self._empty_sync_timestamps.clear()
        self._last_sync_items = 0.0
        self._pending_items = 0.0
        self._episode_reward = 0.0
        self._step_count_in_episode = 0
        self._last_revealed_step = self._episode_start - 1

        self._prediction_provider.reset()
        cutoff = self._timestamps[self._episode_start] - timedelta(minutes=1)
        bootstrap_timestamps = (
            self._bootstrap_timestamps + self._timestamps[:self._episode_start]
        )
        bootstrap_values = (
            self._bootstrap_values + self._actual_values[:self._episode_start].tolist()
        )
        self._prediction_provider.bootstrap(
            cutoff, bootstrap_timestamps, bootstrap_values
        )

        obs = self._build_observation()
        info = self._build_info()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: 0 = don't sync, 1 = sync.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Guard against stepping past end of data
        if self._current_step >= self._total_steps:
            obs = np.zeros(self._obs_spec.dimension, dtype=np.float32)
            return obs, 0.0, True, False, self._build_info()

        actual_value = self._actual_values[self._current_step]

        # Accumulate events since last sync
        if self._task_type == "event_rate":
            self._pending_items += actual_value

        if action == 1:  # Sync
            self._time_since_last_sync = 0.0
            # Training proxy AoI — resets on every sync regardless of
            # events. See class docstring for why this differs from
            # true AoI used in evaluation.
            self._current_aoi = 0.0

            # Check accumulated items since last sync, not just this minute
            if self._task_type == "event_rate":
                has_events = self._pending_items > 0
            else:
                has_events = actual_value < self._reward_fn.config.aoi_threshold_minutes

            if has_events:
                self._last_sync_items = self._pending_items
            else:
                self._cumulative_empty_syncs += 1
                self._empty_sync_timestamps.append(self._step_count_in_episode)
                self._last_sync_items = 0.0

            self._pending_items = 0.0

        # For reward: when syncing, use accumulated items (not just this minute)
        # so the reward reflects whether the sync actually found anything.
        # For skip: use current minute's value (are we missing events now?)
        if action == 1 and self._task_type == "event_rate":
            reward_value = self._last_sync_items
        else:
            reward_value = actual_value

        items_for_reward = self._last_sync_items if action == 1 else 0.0
        reward = self._reward_fn.compute(
            action=action,
            actual_value=reward_value,
            current_aoi=self._current_aoi,
            items_found=items_for_reward,
        )
        reward_components = self._reward_fn.compute_components(
            action=action,
            actual_value=reward_value,
            current_aoi=self._current_aoi,
            items_found=items_for_reward,
        )

        # Increment training-proxy AoI and time-since-sync AFTER reward
        # computation so the reward reflects the state the agent acted on,
        # not the next state. Note: this increments unconditionally on
        # skip — true staleness is computed separately in evaluation.
        if action != 1:
            self._time_since_last_sync += 1.0
            self._current_aoi += 1.0
        self._episode_reward += reward

        # Reveal the complete pending interval only after the action and reward
        # have been determined. It can affect the next observation, never the
        # observation used to choose this synchronization.
        if action == 1:
            start = self._last_revealed_step + 1
            stop = self._current_step + 1
            self._prediction_provider.on_sync(
                self._timestamps[self._current_step],
                self._timestamps[start:stop],
                self._actual_values[start:stop].tolist(),
            )
            self._last_revealed_step = self._current_step

        self._current_step += 1
        self._step_count_in_episode += 1

        terminated = self._current_step >= self._total_steps
        truncated = False
        if self._episode_length is not None:
            truncated = self._step_count_in_episode >= self._episode_length

        if terminated or truncated:
            obs = np.zeros(self._obs_spec.dimension, dtype=np.float32)
        else:
            obs = self._build_observation()

        info = self._build_info()
        info["actual_value"] = float(actual_value)
        info["action"] = action
        info["reward_components"] = reward_components

        return obs, reward, terminated, truncated, info

    def _build_observation(self) -> np.ndarray:
        """Build observation vector for current step."""
        predictions = self._prediction_provider.get_predictions(
            self._timestamps[self._current_step], self._prediction_horizon
        )

        step_covariates = {}
        for cov_name, cov_values in self._covariates.items():
            if self._current_step < len(cov_values):
                step_covariates[cov_name] = cov_values[self._current_step]
            else:
                step_covariates[cov_name] = 0.0

        # Expire empty sync entries outside the rolling window
        cutoff = self._step_count_in_episode - self._empty_sync_window
        while self._empty_sync_timestamps and self._empty_sync_timestamps[0] < cutoff:
            self._empty_sync_timestamps.popleft()

        return self._obs_builder.build(
            time_since_last_sync=self._time_since_last_sync,
            current_aoi=self._current_aoi,
            predictions=predictions,
            covariates=step_covariates,
            items_synced_last=self._last_sync_items,
            cumulative_empty_syncs=self._cumulative_empty_syncs,
            windowed_empty_syncs=len(self._empty_sync_timestamps),
        )

    def _build_info(self) -> Dict[str, Any]:
        """Build info dict for current state."""
        return {
            "step": self._current_step,
            "step_in_episode": self._step_count_in_episode,
            "time_since_last_sync": self._time_since_last_sync,
            "current_aoi": self._current_aoi,
            "cumulative_empty_syncs": self._cumulative_empty_syncs,
            "windowed_empty_syncs": len(self._empty_sync_timestamps),
            "last_sync_items": self._last_sync_items,
            "episode_reward": self._episode_reward,
        }

    @property
    def current_timestamp(self) -> Optional[datetime]:
        """Return the timestamp for the current step, if available."""
        if 0 <= self._current_step < self._total_steps:
            return self._timestamps[self._current_step]
        return None
