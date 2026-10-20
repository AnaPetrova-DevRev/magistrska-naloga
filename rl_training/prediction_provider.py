"""Prediction providers for the sync scheduling RL environment.

Abstracts how the RL agent obtains model predictions, supporting both
pre-computed (fast, deterministic) and live (realistic) modes.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
import copy
from datetime import UTC, datetime, timedelta
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class PredictionProvider(ABC):
    """Abstract interface for providing predictions to the RL environment."""

    @abstractmethod
    def get_predictions(self, timestamp: datetime, n_steps: int) -> List[float]:
        """Return predictions whose first target is ``timestamp``.

        Args:
            timestamp: Current decision timestamp.
            n_steps: Number of future steps to return predictions for.

        Returns:
            List of float prediction values, length n_steps.
            If fewer values are available, pads with the last value.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state (called on env.reset())."""

    def clone(self) -> "PredictionProvider":
        """Return an independently mutable provider instance."""
        return copy.deepcopy(self)

    def bootstrap(
        self,
        cutoff: datetime,
        timestamps: Sequence[datetime],
        values: Sequence[float],
    ) -> None:
        """Initialize state from target history fully revealed through cutoff."""

    def on_sync(
        self,
        timestamp: datetime,
        revealed_timestamps: Sequence[datetime],
        revealed_values: Sequence[float],
    ) -> None:
        """Reveal the complete interval ending at a completed synchronization."""


class ActionAwarePredictionProvider(PredictionProvider):
    """Stateful forecasts based only on synchronized target history.

    ``forecaster.forecast`` receives revealed minute counts, prior generated
    trajectory values, a first target timestamp, and a horizon. Unknown actual
    values are never passed to it.
    """

    PROTOCOL = "action_aware_trajectory_v1"

    def __init__(self, forecaster, trajectory_horizon_steps: int = 60):
        if trajectory_horizon_steps < 1:
            raise ValueError("trajectory_horizon_steps must be positive")
        self._forecaster = forecaster
        self._trajectory_horizon = int(trajectory_horizon_steps)
        self.reset()

    @staticmethod
    def _key(timestamp: datetime) -> datetime:
        if not isinstance(timestamp, datetime):
            raise TypeError("timestamps must be datetime instances")
        if timestamp.tzinfo is not None:
            return timestamp.astimezone(UTC).replace(tzinfo=None)
        return timestamp

    @classmethod
    def _validate_interval(
        cls,
        timestamps: Sequence[datetime],
        values: Sequence[float],
    ) -> None:
        if len(timestamps) != len(values):
            raise ValueError("timestamps and values must have equal length")
        if len(values) and not np.all(np.isfinite(np.asarray(values, dtype=float))):
            raise ValueError("revealed values must be finite")
        keys = [cls._key(ts) for ts in timestamps]
        if any(b - a != timedelta(minutes=1) for a, b in zip(keys, keys[1:])):
            raise ValueError("revealed history must be a contiguous minute interval")

    def bootstrap(
        self,
        cutoff: datetime,
        timestamps: Sequence[datetime],
        values: Sequence[float],
    ) -> None:
        self._validate_interval(timestamps, values)
        cutoff_key = self._key(cutoff)
        if timestamps and self._key(timestamps[-1]) != cutoff_key:
            raise ValueError("bootstrap history must end exactly at cutoff")
        self._observed_timestamps = list(timestamps)
        self._observed_values = [float(v) for v in values]
        self._last_revealed = cutoff
        self._regenerate(cutoff + timedelta(minutes=1))

    def on_sync(
        self,
        timestamp: datetime,
        revealed_timestamps: Sequence[datetime],
        revealed_values: Sequence[float],
    ) -> None:
        if self._last_revealed is None:
            raise RuntimeError("provider must be bootstrapped before synchronization")
        self._validate_interval(revealed_timestamps, revealed_values)
        if not revealed_timestamps:
            raise ValueError("a synchronization must reveal its complete interval")
        expected = self._key(self._last_revealed) + timedelta(minutes=1)
        if self._key(revealed_timestamps[0]) != expected:
            raise ValueError("revealed interval does not follow prior synchronization")
        if self._key(revealed_timestamps[-1]) != self._key(timestamp):
            raise ValueError("revealed interval must end at synchronization timestamp")
        self._observed_timestamps.extend(revealed_timestamps)
        self._observed_values.extend(float(v) for v in revealed_values)
        self._last_revealed = timestamp
        should_regenerate = getattr(self._forecaster, "should_regenerate", None)
        if should_regenerate is None or should_regenerate(revealed_values):
            self._regenerate(timestamp + timedelta(minutes=1))

    def get_predictions(self, timestamp: datetime, n_steps: int) -> List[float]:
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        if self._trajectory_start is None:
            raise RuntimeError("provider must be bootstrapped before prediction")
        delta = self._key(timestamp) - self._key(self._trajectory_start)
        offset_seconds = delta.total_seconds()
        if offset_seconds < 0 or offset_seconds % 60 != 0:
            raise KeyError("prediction timestamp is outside the active trajectory")
        offset = int(offset_seconds // 60)
        while len(self._trajectory_values) < offset + n_steps:
            self._extend()
        return self._trajectory_values[offset:offset + n_steps]

    def _regenerate(self, first_target: datetime) -> None:
        self._trajectory_start = first_target
        self._trajectory_values = []
        self._extend()

    def _extend(self) -> None:
        first_target = self._trajectory_start + timedelta(
            minutes=len(self._trajectory_values)
        )
        synthetic_timestamps = [
            self._trajectory_start + timedelta(minutes=i)
            for i in range(len(self._trajectory_values))
        ]
        values = self._forecaster.forecast(
            self._observed_timestamps,
            self._observed_values,
            synthetic_timestamps,
            self._trajectory_values,
            first_target,
            self._trajectory_horizon,
        )
        values = [float(v) for v in values]
        if len(values) != self._trajectory_horizon:
            raise ValueError("forecaster returned an unexpected trajectory length")
        if not np.all(np.isfinite(np.asarray(values))):
            raise ValueError("forecaster returned non-finite values")
        self._trajectory_values.extend(values)

    @property
    def metadata(self) -> Dict[str, Any]:
        details = dict(getattr(self._forecaster, "metadata", {}))
        return {
            **details,
            "protocol": self.PROTOCOL,
            "history_rule": "revealed_only_after_bootstrap_or_sync",
            "first_target_rule": "minute_after_bootstrap_or_sync",
            "unknown_interval_rule": "consume_or_recursively_extend_predictions",
        }

    def clone(self) -> "ActionAwarePredictionProvider":
        return ActionAwarePredictionProvider(
            self._forecaster, self._trajectory_horizon
        )

    def reset(self) -> None:
        self._observed_timestamps: List[datetime] = []
        self._observed_values: List[float] = []
        self._last_revealed: Optional[datetime] = None
        self._trajectory_start: Optional[datetime] = None
        self._trajectory_values: List[float] = []


class PrecomputedPredictionProvider(PredictionProvider):
    """Provides timestamp-keyed rolling-origin forecast rows."""

    def __init__(
        self,
        timestamps: Sequence[datetime],
        predictions: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            timestamps: Unique decision timestamps, one per forecast row.
            predictions: 2D array shaped ``(origins, horizon)``.
        """
        values = np.asarray(predictions, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("predictions must be a non-empty 2D array")
        if len(timestamps) != values.shape[0]:
            raise ValueError("timestamps and prediction rows must have equal length")
        if not np.all(np.isfinite(values)):
            raise ValueError("predictions contain non-finite values")

        keys = [self._timestamp_key(ts) for ts in timestamps]
        if len(set(keys)) != len(keys):
            raise ValueError("prediction timestamps must be unique")
        self._predictions = values
        self._row_by_timestamp = {key: i for i, key in enumerate(keys)}
        self._timestamps = list(timestamps)
        self._metadata = dict(metadata or {})

    @staticmethod
    def _timestamp_key(timestamp: datetime) -> datetime:
        if not isinstance(timestamp, datetime):
            raise TypeError("prediction lookup requires a datetime timestamp")
        if timestamp.tzinfo is not None:
            return timestamp.astimezone(UTC).replace(tzinfo=None)
        return timestamp

    @classmethod
    def from_cache(cls, cache_path: str) -> "PrecomputedPredictionProvider":
        """Reject timestamp-only caches invalidated for historical evaluation."""
        metadata_path = os.path.splitext(cache_path)[0] + ".json"
        if not os.path.isfile(cache_path) or not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                f"Causal prediction cache requires {cache_path} and {metadata_path}"
            )
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        protocol = metadata.get("protocol")
        raise ValueError(
            f"Prediction cache protocol {protocol!r} is not information-causal "
            "for historical policy evaluation; use action_aware_trajectory_v1"
        )

    def validate_timestamps(self, timestamps: Sequence[datetime]) -> None:
        missing = [ts for ts in timestamps if self._timestamp_key(ts) not in self._row_by_timestamp]
        if missing:
            raise KeyError(
                f"Prediction cache is missing {len(missing)} requested timestamps; "
                f"first missing timestamp: {missing[0].isoformat()}"
            )

    def get_predictions(self, timestamp: datetime, n_steps: int) -> List[float]:
        key = self._timestamp_key(timestamp)
        if key not in self._row_by_timestamp:
            raise KeyError(f"No causal prediction row for {timestamp.isoformat()}")
        row = self._predictions[self._row_by_timestamp[key]]
        if n_steps > len(row):
            raise ValueError(
                f"Requested horizon {n_steps} exceeds cached horizon {len(row)}"
            )
        return row[:n_steps].tolist()

    @property
    def raw_values(self) -> np.ndarray:
        """First-horizon values for scalar prediction policies."""
        return self._predictions[:, 0]

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    def reset(self) -> None:
        pass  # No state to reset

    def clone(self) -> "PrecomputedPredictionProvider":
        return self  # Immutable and stateless.


class NoPredictionProvider(PredictionProvider):
    """Returns zeros when no model predictions are available.

    Used as a fallback when no production model exists.
    """

    def get_predictions(self, timestamp: datetime, n_steps: int) -> List[float]:
        return [0.0] * n_steps

    def reset(self) -> None:
        pass

    def clone(self) -> "NoPredictionProvider":
        return self


class MaskedPredictionProvider(PredictionProvider):
    """Provides a constant, normalized-zero prediction channel.

    Unlike ``NoPredictionProvider``, this provider is used with the prediction
    feature enabled so control and treatment policies have identical input
    dimensions. Time-to-event values use the normalization ceiling so the
    resulting observation values are zero after polarity inversion.
    """

    PROTOCOL = "masked_prediction_v1"
    _RAW_MASK_VALUES = {
        "event_rate": 0.0,
        "time_to_event": 1440.0,
    }

    def __init__(self, prediction_task_type: str):
        if prediction_task_type not in self._RAW_MASK_VALUES:
            raise ValueError(
                "prediction_task_type must be 'event_rate' or 'time_to_event'"
            )
        self._prediction_task_type = prediction_task_type
        self._raw_mask_value = self._RAW_MASK_VALUES[prediction_task_type]

    def get_predictions(self, timestamp: datetime, n_steps: int) -> List[float]:
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        return [self._raw_mask_value] * n_steps

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "protocol": self.PROTOCOL,
            "prediction_task_type": self._prediction_task_type,
            "mask_rule": "constant_normalized_zero",
            "raw_mask_value": self._raw_mask_value,
            "information_content": "none",
        }

    def reset(self) -> None:
        pass

    def clone(self) -> "MaskedPredictionProvider":
        return self


class LivePredictionProvider(PredictionProvider):
    """Provides predictions by calling ModelServingApp at configurable intervals.

    Caches the last prediction trajectory and reuses it between calls.
    The call_interval_minutes parameter controls how often a fresh
    prediction is requested, typically matching the model's native
    resolution (e.g., 15 min for a 15-min model).
    """

    def __init__(
        self,
        model_serving_app,
        sync_unit_obj,
        task_type: str,
        call_interval_minutes: int = 15,
        horizon_minutes: int = 60,
        model_selection: Optional[dict] = None,
    ):
        """
        Args:
            model_serving_app: ModelServingApp instance.
            sync_unit_obj: SyncUnit to get predictions for.
            task_type: "event_rate" or "time_to_event".
            call_interval_minutes: How often to request fresh predictions.
            horizon_minutes: Prediction horizon per call.
            model_selection: Model selection config (mode, stage, etc.).
        """
        self._serving_app = model_serving_app
        self._sync_unit = sync_unit_obj
        self._task_type = task_type
        self._call_interval = call_interval_minutes
        self._horizon_minutes = horizon_minutes
        self._model_selection = model_selection or {
            "mode": "stage",
            "stage": "Production",
        }

        self._cached_values: Optional[List[float]] = None
        self._cache_step: int = -1
        self._prediction_available: bool = False

    def get_predictions(self, timestamp: datetime, n_steps: int) -> List[float]:
        step = int(timestamp.timestamp() // 60)
        if (
            self._cached_values is None
            or step - self._cache_step >= self._call_interval
        ):
            self._refresh_cache(step, timestamp)

        if self._cached_values is None:
            return [0.0] * n_steps

        offset = step - self._cache_step
        if offset < 0:
            offset = 0

        values = self._cached_values[offset : offset + n_steps]

        if len(values) < n_steps:
            pad_val = values[-1] if values else 0.0
            values.extend([pad_val] * (n_steps - len(values)))

        return values

    def _refresh_cache(self, step: int, timestamp: datetime) -> None:
        """Call ModelServingApp and cache the result."""
        try:
            from app_actions_constants import GET_PREDICTION

            params = {
                "task_type": self._task_type,
                "horizon_minutes": self._horizon_minutes,
                "model_selection": self._model_selection,
                "reference_time": timestamp.isoformat(),
            }
            response = self._serving_app.execute_action(
                self._sync_unit, GET_PREDICTION, params
            )

            if response.success and response.data:
                prediction_result = response.data.get("prediction_result")
                if prediction_result and prediction_result.ok:
                    trajectory = prediction_result.prediction
                    self._cached_values = [
                        p.expected_events for p in trajectory.points
                    ]
                    self._cache_step = step
                    self._prediction_available = True
                    return

            logger.warning(
                "ModelServingApp returned no prediction: %s",
                response.message if response else "no response",
            )
        except Exception as e:
            logger.warning("Failed to get live prediction: %s", e)

        # Fallback: keep old cache or use zeros
        if self._cached_values is None:
            self._cached_values = [0.0] * self._horizon_minutes
            self._cache_step = step
        self._prediction_available = False

    @property
    def prediction_available(self) -> bool:
        """Whether the last prediction call succeeded."""
        return self._prediction_available

    def reset(self) -> None:
        self._cached_values = None
        self._cache_step = -1
        self._prediction_available = False

    def clone(self) -> "LivePredictionProvider":
        return LivePredictionProvider(
            self._serving_app,
            self._sync_unit,
            self._task_type,
            self._call_interval,
            self._horizon_minutes,
            self._model_selection,
        )
