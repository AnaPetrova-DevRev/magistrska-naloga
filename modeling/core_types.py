"""Core modeling types for time-series training, evaluation, and serving.

These types are intentionally light-weight and free of ML/MLflow dependencies so
that both training and serving code (and tests) can share a stable contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class TaskType(str, Enum):
    """Supported modeling tasks.

    EVENT_RATE: predict events per minute over time.
    TIME_TO_EVENT: predict time until next event.
    """

    EVENT_RATE = "event_rate"
    TIME_TO_EVENT = "time_to_event"


# Dataset configuration and snapshots


@dataclass
class DatasetConfig:
    """Configuration for building a dataset for a given sync unit.

    The sync unit object itself is passed separately to the builder; this
    config only holds generic temporal and filtering options.
    """

    task_type: TaskType
    start_time: datetime
    end_time: datetime
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    filters: Dict[str, Any] = field(default_factory=dict)
    resample_freq: Optional[str] = None
    covariates: List[str] = field(default_factory=list)
    cron_detection: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSnapshotMetadata:
    """Metadata describing a frozen dataset snapshot on disk.

    Since raw preprocessed data is always available from the pipeline, we
    store only lightweight JSON metadata (not a full DataFrame pickle).
    The dataset can be recreated deterministically from the config params.
    """

    snapshot_metadata_path: Optional[str]
    task_type: TaskType
    start_time: datetime
    end_time: datetime
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    filters: Dict[str, Any] = field(default_factory=dict)
    resample_freq: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


# In-memory dataset representations


@dataclass
class EventRateDataset:
    """Event-rate time series for a single sync unit.

    Granularity depends on the resample_freq used when building the dataset.
    Without resampling the rows correspond to individual minutes that had events.
    """

    timestamps: Sequence[datetime]
    events_per_minute: Sequence[float]
    covariates: Dict[str, Sequence[float]] = field(default_factory=dict)
    train_indices: List[int] = field(default_factory=list)
    val_indices: List[int] = field(default_factory=list)
    test_indices: List[int] = field(default_factory=list)
    cron_patterns: Optional[List[Dict[str, Any]]] = None
    hourly_cron_patterns: Optional[List[Dict[str, Any]]] = None


@dataclass
class TimeToEventDataset:
    """Time-to-event (inter-arrival time) series for a single sync unit."""

    timestamps: Sequence[datetime]
    minutes_to_next_event: Sequence[float]
    covariates: Dict[str, Sequence[float]] = field(default_factory=dict)
    train_indices: List[int] = field(default_factory=list)
    val_indices: List[int] = field(default_factory=list)
    test_indices: List[int] = field(default_factory=list)


Dataset = EventRateDataset | TimeToEventDataset


# Prediction types shared with the RL agent


@dataclass
class PredictionPoint:
    """Prediction for a single timestamp/horizon step."""

    timestamp: datetime
    expected_events: float
    lower_quantile: Optional[float] = None
    upper_quantile: Optional[float] = None
    uncertainty: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "expected_events": self.expected_events,
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "uncertainty": self.uncertainty,
        }


@dataclass
class PredictionTrajectory:
    """A horizon of predictions for a sync unit and task."""

    sync_unit_name: str
    task_type: TaskType
    horizon_minutes: int
    points: List[PredictionPoint]
    model_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sync_unit_name": self.sync_unit_name,
            "task_type": self.task_type.value,
            "horizon_minutes": self.horizon_minutes,
            "points": [p.to_dict() for p in self.points],
            "model_id": self.model_id,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class PredictionResult:
    """Wrapper for returning predictions (or errors) via the orchestrator."""

    ok: bool
    prediction: Optional[PredictionTrajectory] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"ok": self.ok}
        if self.prediction is not None:
            data["prediction"] = self.prediction.to_dict()
        if self.error_type is not None:
            data["error_type"] = self.error_type
        if self.error_message is not None:
            data["error_message"] = self.error_message
        return data


# Metric base interface and a simple MAE metric


class Metric:
    """Base class for evaluation metrics.

    Metrics are simple callables taking true/predicted values plus optional
    context (e.g. sync unit, split name) and returning a scalar.
    """

    name: str = "metric"

    def __call__(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float],
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        raise NotImplementedError


class MaeMetric(Metric):
    """Mean Absolute Error (MAE)."""

    name: str = "mae"

    def __call__(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float],
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        if not y_true:
            return float("nan")
        total = 0.0
        for t, p in zip(y_true, y_pred):
            total += abs(t - p)
        return total / len(y_true)


# Training and Inference Contexts


@dataclass
class TrainingContext:
    """Context object describing how to train a model.

    This is built from a combination of sync unit info, task type,
    model template, and metric definitions from the training config.
    """

    sync_unit_name: str
    task_type: TaskType
    model_type: str
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    metric_defs: List[Dict[str, Any]] = field(default_factory=list)
    template_name: Optional[str] = None


@dataclass
class InferenceContext:
    """Context for serving/inference operations.

    Used by ModelServingApp to load and run predictions.
    """

    sync_unit_name: str
    task_type: TaskType
    horizon_minutes: int
    model_id: Optional[str] = None
    model_stage: str = "Production"
