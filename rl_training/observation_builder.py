"""Builds observation vectors for the sync scheduling RL environment.

Assembles configurable feature components (AoI, predictions, covariates, etc.)
into a flat numpy array suitable for Gymnasium Box observation spaces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# Observation feature component names (used in env_config.observation_features)
FEATURE_TIME_SINCE_LAST_SYNC = "time_since_last_sync"
FEATURE_CURRENT_AOI = "current_aoi"
FEATURE_PREDICTIONS = "predictions"
FEATURE_COVARIATES = "covariates"
FEATURE_ITEMS_SYNCED_LAST = "items_synced_last"
FEATURE_CUMULATIVE_EMPTY_SYNCS = "cumulative_empty_syncs"
FEATURE_WINDOWED_EMPTY_SYNCS = "windowed_empty_syncs"

ALL_FEATURES = [
    FEATURE_TIME_SINCE_LAST_SYNC,
    FEATURE_CURRENT_AOI,
    FEATURE_PREDICTIONS,
    FEATURE_COVARIATES,
    FEATURE_ITEMS_SYNCED_LAST,
    FEATURE_CUMULATIVE_EMPTY_SYNCS,
    FEATURE_WINDOWED_EMPTY_SYNCS,
]


@dataclass
class ObservationSpec:
    """Defines the observation space structure.

    Computed once at environment init from the config.
    """

    features: List[str]
    prediction_horizon_steps: int = 60
    covariate_names: List[str] = field(default_factory=list)
    prediction_task_type: str = "event_rate"

    @property
    def dimension(self) -> int:
        dim = 0
        for f in self.features:
            if f == FEATURE_PREDICTIONS:
                dim += self.prediction_horizon_steps
            elif f == FEATURE_COVARIATES:
                dim += len(self.covariate_names)
            else:
                dim += 1
        return dim

    def feature_layout(self) -> Dict[str, tuple]:
        """Return {feature_name: (start_idx, end_idx)} for each component."""
        layout = {}
        idx = 0
        for f in self.features:
            if f == FEATURE_PREDICTIONS:
                size = self.prediction_horizon_steps
            elif f == FEATURE_COVARIATES:
                size = len(self.covariate_names)
            else:
                size = 1
            layout[f] = (idx, idx + size)
            idx += size
        return layout


class ObservationBuilder:
    """Builds flat observation vectors from environment state.

    The observation is normalized to reasonable ranges for neural network
    consumption. All features are packed into a single float32 vector.
    """

    # Normalization constants
    MAX_TIME_MINUTES = 1440.0  # 24 hours
    MAX_ITEMS = 1000.0
    MAX_EMPTY_SYNCS = 1440.0
    MAX_PREDICTION_VALUE = 100.0  # ER: events/min ceiling
    MAX_TTE_PREDICTION_VALUE = 1440.0  # TTE: 24h ceiling in minutes

    # Covariate normalization: divisor that maps the raw value to [0, 1].
    # sin/cos covariates are in [-1, 1] and handled specially.
    _COVARIATE_DIVISORS: Dict[str, float] = {
        "hour_of_day": 23.0,
        "day_of_week": 6.0,
        "is_weekend": 1.0,
        "day_of_month": 31.0,
        "month": 12.0,
        "quarter": 4.0,
        "minute_of_hour": 59.0,
        "is_business_hours": 1.0,
        "hours_since_midnight": 24.0,
        # Cron covariates — already in [0, 1]
        "is_cron_window": 1.0,
        "minutes_to_next_cron": 1.0,
        "cron_expected_magnitude": 1.0,
        "is_hourly_cron_window": 1.0,
        "minutes_to_next_hourly_cron": 1.0,
        "hourly_cron_expected_magnitude": 1.0,
        # TTE resampling covariate
        "elapsed_since_last_event": 1440.0,
    }

    _SINCOS_COVARIATES = frozenset({
        "sin_hour", "cos_hour", "sin_day_of_week", "cos_day_of_week",
    })

    _warned_covariates: set = set()

    def __init__(self, spec: ObservationSpec):
        self.spec = spec

    def build(
        self,
        time_since_last_sync: float,
        current_aoi: float,
        predictions: Sequence[float],
        covariates: Dict[str, float],
        items_synced_last: float,
        cumulative_empty_syncs: int,
        windowed_empty_syncs: int = 0,
    ) -> np.ndarray:
        """Assemble a flat observation vector from environment state.

        Args:
            time_since_last_sync: Minutes since last sync action.
            current_aoi: Minutes since last meaningful sync.
            predictions: Next N minutes of model predictions.
            covariates: Current-step covariate values {name: value}.
            items_synced_last: Items found in most recent sync.
            cumulative_empty_syncs: Running count of empty syncs.
            windowed_empty_syncs: Empty syncs in the last N minutes.

        Returns:
            Float32 numpy array of shape (spec.dimension,).
        """
        obs = np.zeros(self.spec.dimension, dtype=np.float32)
        idx = 0

        for f in self.spec.features:
            if f == FEATURE_TIME_SINCE_LAST_SYNC:
                obs[idx] = min(time_since_last_sync, self.MAX_TIME_MINUTES) / self.MAX_TIME_MINUTES
                idx += 1

            elif f == FEATURE_CURRENT_AOI:
                obs[idx] = min(current_aoi, self.MAX_TIME_MINUTES) / self.MAX_TIME_MINUTES
                idx += 1

            elif f == FEATURE_PREDICTIONS:
                n = self.spec.prediction_horizon_steps
                pred_arr = np.array(predictions[:n], dtype=np.float32)
                # Pad if fewer predictions than expected
                if len(pred_arr) < n:
                    pad_val = pred_arr[-1] if len(pred_arr) > 0 else 0.0
                    pred_arr = np.pad(
                        pred_arr, (0, n - len(pred_arr)),
                        mode="constant", constant_values=pad_val,
                    )
                # Normalize predictions (polarity: high = event likely)
                if self.spec.prediction_task_type == "time_to_event":
                    # TTE: invert so 0min→1.0 (imminent) and 1440min→0.0 (distant)
                    np.clip(pred_arr, 0, self.MAX_TTE_PREDICTION_VALUE, out=pred_arr)
                    pred_arr = 1.0 - (pred_arr / self.MAX_TTE_PREDICTION_VALUE)
                else:
                    np.clip(pred_arr, 0, self.MAX_PREDICTION_VALUE, out=pred_arr)
                    pred_arr /= self.MAX_PREDICTION_VALUE
                obs[idx : idx + n] = pred_arr
                idx += n

            elif f == FEATURE_COVARIATES:
                for cov_name in self.spec.covariate_names:
                    raw_val = covariates.get(cov_name, 0.0)
                    if cov_name in self._SINCOS_COVARIATES:
                        obs[idx] = (raw_val + 1.0) / 2.0
                    elif cov_name in self._COVARIATE_DIVISORS:
                        divisor = self._COVARIATE_DIVISORS[cov_name]
                        obs[idx] = min(raw_val, divisor) / divisor if divisor else 0.0
                    else:
                        if cov_name not in self._warned_covariates:
                            logger.warning(
                                "No normalizer for covariate '%s'; "
                                "passing raw value (may exceed [0,1] bounds).",
                                cov_name,
                            )
                            self._warned_covariates.add(cov_name)
                        obs[idx] = raw_val
                    idx += 1

            elif f == FEATURE_ITEMS_SYNCED_LAST:
                obs[idx] = min(items_synced_last, self.MAX_ITEMS) / self.MAX_ITEMS
                idx += 1

            elif f == FEATURE_CUMULATIVE_EMPTY_SYNCS:
                obs[idx] = min(cumulative_empty_syncs, self.MAX_EMPTY_SYNCS) / self.MAX_EMPTY_SYNCS
                idx += 1

            elif f == FEATURE_WINDOWED_EMPTY_SYNCS:
                obs[idx] = min(windowed_empty_syncs, self.MAX_EMPTY_SYNCS) / self.MAX_EMPTY_SYNCS
                idx += 1

        return obs
