"""Step-wise reward function for the sync scheduling RL agent.

Extends the AoIRewardMetric logic from modeling/metrics.py to work
as a stateful, per-step reward suitable for Gymnasium environments.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for the sync scheduling reward function.

    reward_mode controls how much ground truth leaks into the reward:
      - "full": uses actual_value to determine sync success/failure
        (default, provides strongest learning signal but peeks at
        ground truth the agent can't observe).
      - "aoi_only": ignores actual_value entirely — reward is only
        sync cost + continuous AoI penalty. Useful as an ablation to
        measure how much the ground-truth peek inflates performance.
    """

    sync_cost: float = 1.0
    late_penalty: float = 2.0
    successful_sync_reward: float = 0.5
    empty_sync_penalty: float = 1.0
    aoi_penalty_per_minute: float = 0.01
    aoi_threshold_minutes: float = 5.0
    mode: str = "event_rate"  # "event_rate" or "time_to_event"
    reward_mode: str = "full"  # "full" or "aoi_only"
    aoi_mode: str = "linear"  # "linear", "quadratic", "linear_no_threshold"
    item_reward_per_unit: float = 0.0

    PRESET_NAMES = ("default", "balanced", "sparse_friendly", "auto")
    _VALID_AOI_MODES = ("linear", "quadratic", "linear_no_threshold")

    # Canonical reward config for cross-policy evaluation comparisons.
    # All evaluation runs use this so that total_reward is comparable.
    STANDARD_EVAL_PARAMS: ClassVar[dict] = {
        "sync_cost": 1.0,
        "late_penalty": 2.0,
        "successful_sync_reward": 0.5,
        "empty_sync_penalty": 1.0,
        "aoi_penalty_per_minute": 0.02,
        "aoi_threshold_minutes": 0.0,
        "aoi_mode": "linear_no_threshold",
        "reward_mode": "full",
        "item_reward_per_unit": 0.0,
    }

    def __post_init__(self):
        if self.mode not in ("event_rate", "time_to_event"):
            raise ValueError(
                f"mode must be 'event_rate' or 'time_to_event', got '{self.mode}'"
            )
        if self.reward_mode not in ("full", "aoi_only"):
            raise ValueError(
                f"reward_mode must be 'full' or 'aoi_only', got '{self.reward_mode}'"
            )
        if self.aoi_mode not in self._VALID_AOI_MODES:
            raise ValueError(
                f"aoi_mode must be one of {self._VALID_AOI_MODES}, got '{self.aoi_mode}'"
            )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RewardConfig:
        d = dict(d)
        preset = d.pop("preset", None)
        if preset is not None:
            return cls.from_preset(preset, **{
                k: v for k, v in d.items() if k in cls.__dataclass_fields__
            })
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_preset(cls, name: str, **overrides) -> RewardConfig:
        presets = {
            "default": {
                "sync_cost": 1.0, "late_penalty": 2.0,
                "successful_sync_reward": 0.5, "empty_sync_penalty": 1.0,
                "aoi_penalty_per_minute": 0.01, "aoi_threshold_minutes": 5.0,
            },
            "balanced": {
                "sync_cost": 0.5, "late_penalty": 2.0,
                "successful_sync_reward": 1.5, "empty_sync_penalty": 0.5,
                "aoi_penalty_per_minute": 0.02, "aoi_threshold_minutes": 5.0,
            },
            "sparse_friendly": {
                "sync_cost": 0.3, "late_penalty": 3.0,
                "successful_sync_reward": 2.0, "empty_sync_penalty": 0.3,
                "aoi_penalty_per_minute": 0.005, "aoi_threshold_minutes": 10.0,
            },
        }
        if name == "auto":
            raise ValueError(
                "Preset 'auto' requires dataset statistics. Use "
                "RewardConfig.from_dataset(actual_values, task_type) "
                "or pass 'preset': 'auto' in a training config (the "
                "training app resolves it automatically)."
            )
        if name not in presets:
            raise ValueError(
                f"Unknown preset '{name}'. Available: {sorted(presets.keys())}"
            )
        params = {**presets[name], **overrides}
        return cls(**{k: v for k, v in params.items() if k in cls.__dataclass_fields__})

    @classmethod
    def standard_eval(cls, mode: str = "event_rate") -> RewardConfig:
        """Return the canonical reward config for evaluation comparisons."""
        return cls(**cls.STANDARD_EVAL_PARAMS, mode=mode)

    def is_standard_eval(self) -> bool:
        """True if this config matches the standard evaluation reward."""
        std = self.standard_eval(mode=self.mode)
        return self.to_dict() == std.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sync_cost": self.sync_cost,
            "late_penalty": self.late_penalty,
            "successful_sync_reward": self.successful_sync_reward,
            "empty_sync_penalty": self.empty_sync_penalty,
            "aoi_penalty_per_minute": self.aoi_penalty_per_minute,
            "aoi_threshold_minutes": self.aoi_threshold_minutes,
            "mode": self.mode,
            "reward_mode": self.reward_mode,
            "aoi_mode": self.aoi_mode,
            "item_reward_per_unit": self.item_reward_per_unit,
        }

    @classmethod
    def from_dataset(
        cls,
        actual_values: Sequence[float],
        task_type: str = "event_rate",
        gap_cost_ratio: float = 3.0,
        **overrides,
    ) -> RewardConfig:
        """Auto-calibrate reward parameters from dataset statistics.

        Sets ``aoi_threshold_minutes`` and ``aoi_penalty_per_minute`` so
        that the total AoI cost of ignoring one typical inter-event gap
        equals ``gap_cost_ratio`` empty syncs. Discrete rewards are scaled
        relative to event density.

        Args:
            actual_values: Training data values (events/min for ER,
                minutes-to-next-event for TTE).
            task_type: ``"event_rate"`` or ``"time_to_event"``.
            gap_cost_ratio: How many empty sync penalties one missed
                gap should cost. Higher = more aggressive AoI penalty.
            **overrides: Override any computed field.

        Returns:
            A RewardConfig with calibrated AoI parameters.
        """
        stats = compute_dataset_stats(actual_values, task_type)
        median_iet = stats["median_inter_event_time"]
        event_rate = stats["event_rate"]

        # --- AoI threshold: start penalizing at ~30% of median gap ---
        aoi_threshold = max(5.0, median_iet * 0.3)

        # --- AoI penalty: calibrate so total cost over one gap ≈ K * empty_sync_penalty ---
        # Total AoI cost over gap T with threshold Th (continuous approx):
        #   aoi_penalty * (T - Th)^2 / 2 = K * empty_sync_penalty
        # Solve for aoi_penalty:
        effective_gap = max(median_iet - aoi_threshold, 1.0)

        # Choose empty_sync_penalty based on density
        if event_rate > 0.1:
            empty_sync_penalty = 0.5
        elif event_rate > 0.01:
            empty_sync_penalty = 1.0
        else:
            empty_sync_penalty = 0.3

        aoi_penalty = (2.0 * gap_cost_ratio * empty_sync_penalty) / (effective_gap ** 2)
        aoi_penalty = max(aoi_penalty, 1e-6)

        # --- Discrete rewards: scale to density ---
        if event_rate > 0.1:
            sync_cost = 0.5
            late_penalty = 2.0
            successful_sync_reward = 1.0
        elif event_rate > 0.01:
            sync_cost = 0.5
            late_penalty = 2.0
            successful_sync_reward = 1.5
        else:
            sync_cost = 0.3
            late_penalty = 3.0
            successful_sync_reward = 2.0

        params = {
            "sync_cost": sync_cost,
            "late_penalty": late_penalty,
            "successful_sync_reward": successful_sync_reward,
            "empty_sync_penalty": empty_sync_penalty,
            "aoi_penalty_per_minute": round(aoi_penalty, 6),
            "aoi_threshold_minutes": round(aoi_threshold, 1),
            "mode": task_type,
        }
        params.update(overrides)

        config = cls(**{k: v for k, v in params.items() if k in cls.__dataclass_fields__})

        logger.info(
            "Auto-calibrated reward: median_iet=%.1f min, event_rate=%.4f, "
            "threshold=%.1f, penalty=%.6f, gap_cost=%.2f empty syncs",
            median_iet, event_rate, config.aoi_threshold_minutes,
            config.aoi_penalty_per_minute, gap_cost_ratio,
        )

        return config


def compute_dataset_stats(
    actual_values: Sequence[float],
    task_type: str = "event_rate",
) -> Dict[str, float]:
    """Compute dataset statistics relevant to reward calibration.

    Returns:
        Dict with keys: median_inter_event_time, mean_inter_event_time,
        event_rate, total_events, total_minutes.
    """
    values = np.asarray(actual_values, dtype=np.float64)
    total_minutes = len(values)

    if task_type == "event_rate":
        event_mask = values > 0
        event_indices = np.where(event_mask)[0]
        total_events = float(event_mask.sum())
    else:
        # TTE: event occurs when value jumps up (countdown reset)
        if len(values) < 2:
            return {
                "median_inter_event_time": float(total_minutes),
                "mean_inter_event_time": float(total_minutes),
                "event_rate": 0.0,
                "total_events": 0.0,
                "total_minutes": float(total_minutes),
            }
        diffs = np.diff(values)
        event_mask = diffs > 0
        event_indices = np.where(event_mask)[0] + 1
        total_events = float(event_mask.sum())

    if len(event_indices) < 2:
        median_iet = float(total_minutes)
        mean_iet = float(total_minutes)
    else:
        gaps = np.diff(event_indices).astype(np.float64)
        median_iet = float(np.median(gaps))
        mean_iet = float(np.mean(gaps))

    event_rate = total_events / total_minutes if total_minutes > 0 else 0.0

    return {
        "median_inter_event_time": median_iet,
        "mean_inter_event_time": mean_iet,
        "event_rate": event_rate,
        "total_events": total_events,
        "total_minutes": float(total_minutes),
    }


class RewardFunction:
    """Step-wise reward for sync scheduling decisions.

    Combines discrete sync-outcome rewards (from AoIRewardMetric logic)
    with a continuous AoI penalty that provides dense shaping signal.

    Discrete component (per sync decision):
        sync + events present   → -sync_cost + successful_sync_reward
        sync + no events        → -empty_sync_penalty
        no sync + events missed → -late_penalty
        no sync + calm          → 0

    Continuous component (every step):
        -aoi_penalty_per_minute * max(0, current_aoi - aoi_threshold)
    """

    def __init__(self, config: RewardConfig):
        self.config = config

    def _compute_aoi_penalty(self, current_aoi: float) -> float:
        c = self.config
        if c.aoi_mode == "quadratic":
            return -c.aoi_penalty_per_minute * (current_aoi ** 2)
        elif c.aoi_mode == "linear_no_threshold":
            return -c.aoi_penalty_per_minute * current_aoi
        # linear (default)
        aoi_excess = max(0.0, current_aoi - c.aoi_threshold_minutes)
        return -c.aoi_penalty_per_minute * aoi_excess

    def _compute_item_bonus(self, action: int, items_found: float) -> float:
        if action == 1 and items_found > 0:
            return self.config.item_reward_per_unit * items_found
        return 0.0

    def _is_urgent_event_rate(self, actual_value: float) -> bool:
        """For event_rate: urgent if there are actual events."""
        return actual_value > 0

    def _is_urgent_time_to_event(self, actual_value: float) -> bool:
        """For time_to_event: urgent if actual TTE is below threshold.

        actual_value is in minutes (consistent with dataset_builder output).
        """
        return actual_value < self.config.aoi_threshold_minutes

    def _compute_discrete(
        self, action: int, actual_value: float,
    ) -> tuple[float, bool]:
        """Compute discrete reward component. Returns (reward, is_urgent)."""
        c = self.config
        did_sync = action == 1

        if c.reward_mode == "aoi_only":
            return (-c.sync_cost if did_sync else 0.0), False

        if c.mode == "event_rate":
            is_urgent = self._is_urgent_event_rate(actual_value)
        else:
            is_urgent = self._is_urgent_time_to_event(actual_value)

        discrete_reward = 0.0
        if did_sync:
            if is_urgent:
                discrete_reward = -c.sync_cost + c.successful_sync_reward
            else:
                discrete_reward = -c.empty_sync_penalty
        else:
            if is_urgent:
                discrete_reward = -c.late_penalty

        return discrete_reward, is_urgent

    def compute(
        self,
        action: int,
        actual_value: float,
        current_aoi: float,
        items_found: float = 0.0,
    ) -> float:
        """Compute single-step reward.

        Args:
            action: 0 = don't sync, 1 = sync.
            actual_value: Ground truth for this timestep.
                For event_rate: events per minute.
                For time_to_event: minutes to next event.
            current_aoi: Current age of information in minutes
                (time since last meaningful sync).
            items_found: Number of items found when syncing (for
                item_reward_per_unit bonus). Ignored when action=0.

        Returns:
            Scalar reward (higher is better, typically <= 0).
        """
        discrete_reward, _ = self._compute_discrete(action, actual_value)
        aoi_penalty = self._compute_aoi_penalty(current_aoi)
        item_bonus = self._compute_item_bonus(action, items_found)

        return discrete_reward + aoi_penalty + item_bonus

    def compute_components(
        self,
        action: int,
        actual_value: float,
        current_aoi: float,
        items_found: float = 0.0,
    ) -> Dict[str, float]:
        """Like compute(), but returns a breakdown of reward components.

        Useful for logging and dashboard analysis.
        """
        discrete_reward, is_urgent = self._compute_discrete(action, actual_value)
        aoi_penalty = self._compute_aoi_penalty(current_aoi)
        item_bonus = self._compute_item_bonus(action, items_found)

        if self.config.aoi_mode == "linear":
            aoi_excess = max(0.0, current_aoi - self.config.aoi_threshold_minutes)
        else:
            aoi_excess = current_aoi

        return {
            "discrete_reward": discrete_reward,
            "aoi_penalty": aoi_penalty,
            "item_bonus": item_bonus,
            "total_reward": discrete_reward + aoi_penalty + item_bonus,
            "is_urgent": float(is_urgent),
            "did_sync": float(action == 1),
            "aoi_excess": aoi_excess,
        }
