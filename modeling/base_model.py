"""Base model interface for all forecasting models.

This module defines the abstract base class that all model wrappers must
implement. It provides a consistent interface for training, prediction,
and artifact management across different model types (ARIMA, N-HiTS, etc.).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Type

import numpy as np

logger = logging.getLogger(__name__)

from modeling.core_types import (
    Dataset,
    EventRateDataset,
    PredictionPoint,
    PredictionTrajectory,
    TaskType,
    TimeToEventDataset,
    TrainingContext,
)
from modeling.validation import (
    ValidationResult,
    check_excessive_zeros,
    check_infinite_values,
    check_minimum_samples,
    check_nan_values,
    check_negative_values,
    check_zero_variance,
)


class BaseModel(ABC):
    """Abstract base class for all forecasting models.

    Each concrete model implementation must:
    - Declare which task types it supports via MODEL_TYPE and SUPPORTED_TASK_TYPES
    - Implement fit() for training
    - Implement predict_pointwise() for metric evaluation
    - Implement forecast() for serving predictions to the RL agent
    - Implement save() and load() for artifact persistence

    Models can support either EVENT_RATE, TIME_TO_EVENT, or both tasks.
    """

    MODEL_TYPE: str = "base"
    SUPPORTED_TASK_TYPES: set = set()

    def __init__(
        self,
        task_type: TaskType,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
    ):
        """Initialize the model with configuration.

        Args:
            task_type: The task this model instance is configured for.
            model_config: Model-specific hyperparameters (e.g., hidden_size, lags).
            training_config: Generic training parameters (e.g., epochs, batch_size).
        """
        if task_type not in self.SUPPORTED_TASK_TYPES:
            raise ValueError(
                f"{self.__class__.__name__} does not support task type {task_type}. "
                f"Supported: {self.SUPPORTED_TASK_TYPES}"
            )
        self.task_type = task_type
        self.model_config = model_config
        self.training_config = training_config
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Return True if the model has been trained."""
        return self._is_fitted

    @abstractmethod
    def fit(self, dataset: Dataset) -> Dict[str, Any]:
        """Train the model on the dataset's training split.

        Args:
            dataset: An EventRateDataset or TimeToEventDataset with train_indices set.

        Returns:
            A dict containing training metadata (e.g., epochs run, final train loss).

        Raises:
            ValueError: If dataset type doesn't match task_type.
        """
        pass

    @abstractmethod
    def predict_pointwise(
        self,
        dataset: Dataset,
        indices: Sequence[int],
    ) -> np.ndarray:
        """Generate point predictions for specified indices.

        This is used by the metrics evaluation loop to compute MAE, RMSE, etc.

        Args:
            dataset: The dataset containing the input features.
            indices: Which indices to generate predictions for.

        Returns:
            A 1D numpy array of predictions, one per index.
        """
        pass

    @abstractmethod
    def forecast(
        self,
        dataset: Dataset,
        horizon_minutes: int,
        sync_unit_name: str = "",
        reference_time: Optional[datetime] = None,
    ) -> PredictionTrajectory:
        """Generate a full forecast trajectory for serving.

        This is called by ModelServingApp to produce predictions for the RL agent.
        The output includes uncertainty estimates when available.

        Args:
            dataset: Dataset with recent history for context.
            horizon_minutes: How many minutes into the future to forecast.
            sync_unit_name: Identifier for the sync unit (included in the trajectory).
            reference_time: The starting timestamp for the forecast. If None,
                           uses the last timestamp in the dataset + 1 minute.

        Returns:
            A PredictionTrajectory with per-minute predictions and uncertainty.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the trained model to disk.

        Args:
            path: Directory path where model artifacts should be saved.
                  The implementation should create files within this directory.
        """
        pass

    @classmethod
    @abstractmethod
    def load(
        cls,
        path: str,
        task_type: TaskType,
        model_config: Dict[str, Any],
    ) -> "BaseModel":
        """Load a trained model from disk.

        Args:
            path: Directory path containing the saved model artifacts.
            task_type: The task type this model was trained for.
            model_config: The model configuration used during training.

        Returns:
            A model instance ready for prediction.
        """
        pass

    def _validate_dataset_type(self, dataset: Dataset) -> None:
        """Validate that the dataset type matches the model's task type."""
        if self.task_type == TaskType.EVENT_RATE:
            if not isinstance(dataset, EventRateDataset):
                raise ValueError(
                    f"Expected EventRateDataset for task {self.task_type}, "
                    f"got {type(dataset).__name__}"
                )
        elif self.task_type == TaskType.TIME_TO_EVENT:
            if not isinstance(dataset, TimeToEventDataset):
                raise ValueError(
                    f"Expected TimeToEventDataset for task {self.task_type}, "
                    f"got {type(dataset).__name__}"
                )

    def _get_target_array(self, dataset: Dataset) -> np.ndarray:
        """Extract the target array from the dataset based on task type."""
        if self.task_type == TaskType.EVENT_RATE:
            return np.array(dataset.events_per_minute)
        else:
            return np.array(dataset.minutes_to_next_event)

    def _pre_fit_validate(self, dataset: Dataset) -> ValidationResult:
        """Run pre-fit validation checks on the dataset.

        Performs universal checks (NaN, Inf, negatives, sample size,
        variance, excessive zeros).  Subclasses should override to add
        model-specific checks, calling ``super()`` first.

        Returns:
            ValidationResult with any warnings found.
        """
        result = ValidationResult()
        target = self._get_target_array(dataset)
        train_data = target[dataset.train_indices]

        result.add(check_nan_values(train_data))
        result.add(check_infinite_values(train_data))
        result.add(check_negative_values(train_data))
        result.add(check_minimum_samples(train_data))
        result.add(check_zero_variance(train_data))
        result.add(check_excessive_zeros(train_data))
        return result

    def _post_fit_validate(
        self, dataset: Dataset, fit_info: Dict[str, Any],
    ) -> ValidationResult:
        """Run post-fit validation checks after model fitting.

        Subclasses should override to add goodness-of-fit tests
        (e.g. KS test for parametric models), calling ``super()`` first.

        Returns:
            ValidationResult with any warnings found.
        """
        return ValidationResult()

    def _run_validation(
        self, dataset: Dataset, fit_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run full pre+post validation and inject warnings into fit_info.

        Call this at the end of ``fit()`` to attach warnings to the
        returned training info dict.
        """
        result = ValidationResult()
        result.merge(self._pre_fit_validate(dataset))
        result.merge(self._post_fit_validate(dataset, fit_info))

        if result.warnings:
            fit_info["warnings"] = result.to_list()
            fit_info["validation_passed"] = result.is_valid
            for w in result.warnings:
                log_fn = logger.error if w.level == "error" else logger.warning
                log_fn("[%s] %s: %s", self.MODEL_TYPE, w.code, w.message)
        else:
            fit_info["warnings"] = []
            fit_info["validation_passed"] = True

        return fit_info

    def _create_prediction_points(
        self,
        timestamps: List[datetime],
        predictions: np.ndarray,
        lower_quantiles: Optional[np.ndarray] = None,
        upper_quantiles: Optional[np.ndarray] = None,
        uncertainties: Optional[np.ndarray] = None,
    ) -> List[PredictionPoint]:
        """Helper to create PredictionPoint objects from arrays."""
        points = []
        for i, ts in enumerate(timestamps):
            point = PredictionPoint(
                timestamp=ts,
                expected_events=float(predictions[i]),
                lower_quantile=float(lower_quantiles[i]) if lower_quantiles is not None else None,
                upper_quantile=float(upper_quantiles[i]) if upper_quantiles is not None else None,
                uncertainty=float(uncertainties[i]) if uncertainties is not None else None,
            )
            points.append(point)
        return points

    @staticmethod
    def _infer_freq(timestamps) -> str:
        """Infer pandas frequency string from timestamps."""
        if len(timestamps) < 2:
            return "min"
        delta = timestamps[1] - timestamps[0]
        seconds = delta.total_seconds()
        if seconds <= 60:
            return "min"
        elif seconds <= 3600:
            return f"{round(seconds / 60)}min"
        elif seconds <= 86400:
            return f"{round(seconds / 3600)}h"
        return "D"

    @staticmethod
    def _freq_to_minutes(freq: str) -> float:
        """Convert a pandas frequency string to minutes."""
        if freq in ("min", "T"):
            return 1.0
        if freq.endswith("min"):
            return float(freq.replace("min", ""))
        if freq.endswith("h"):
            return float(freq.replace("h", "")) * 60
        if freq == "D":
            return 1440.0
        return 1.0

    def _interpolate_to_minutes(
        self,
        native_values: np.ndarray,
        step_minutes: float,
        horizon_minutes: int,
        is_cumulative_rate: bool = False,
    ) -> np.ndarray:
        """Interpolate coarser-resolution forecasts to minute-level.

        Args:
            native_values: Forecast values at native resolution.
            step_minutes: Minutes per native step.
            horizon_minutes: Desired number of minute-level outputs.
            is_cumulative_rate: If True, values are bin totals (e.g. events/day)
                and will be divided by step_minutes to get per-minute rates.

        Returns:
            Minute-level interpolated array of length horizon_minutes.
        """
        if step_minutes <= 1.0:
            if len(native_values) >= horizon_minutes:
                return native_values[:horizon_minutes].copy()
            return np.pad(native_values, (0, horizon_minutes - len(native_values)), mode="edge")

        values = native_values / step_minutes if is_cumulative_rate else native_values.copy()

        # Place each native value at the midpoint of its bin
        midpoints = np.array([step_minutes * (i + 0.5) for i in range(len(values))])
        minute_points = np.arange(horizon_minutes, dtype=float) + 0.5

        return np.interp(minute_points, midpoints, values)

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata for logging/tracking."""
        return {
            "model_type": self.MODEL_TYPE,
            "task_type": self.task_type.value,
            "is_fitted": self._is_fitted,
            "model_config": self.model_config,
            "training_config": self.training_config,
        }
