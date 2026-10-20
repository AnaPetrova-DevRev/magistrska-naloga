"""Exponential distribution model for time-to-event forecasting.

The Exponential distribution is the simplest inter-arrival time model.
It assumes a constant hazard rate (memoryless property): the probability
of an event occurring in the next dt is independent of how long we've
already waited.  This makes it a strong baseline for time-to-event tasks.

The single parameter lambda (rate) is estimated via MLE as 1/mean.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from modeling.base_model import BaseModel
from modeling.core_types import (
    Dataset,
    PredictionTrajectory,
    TaskType,
    TimeToEventDataset,
)
from modeling.validation import ValidationResult, check_ks_test

logger = logging.getLogger(__name__)


class ExponentialTimeToEventModel(BaseModel):
    """Exponential distribution model for time-to-event forecasting.

    The model fits an Exponential(lambda) distribution to the training
    inter-arrival times via MLE.  Predictions are the distribution mean
    (1/lambda) for all future points — this is optimal under squared-error
    loss for a memoryless process.

    Configuration options (model_config):
        min_delta: float — minimum delta value to include in fitting
                   (filters out near-zero duplicates). Default: 0.0
        use_windowed_rate: bool — if True, estimate lambda from a
                          sliding window of recent events. Default: False
        window_size: int — number of recent events for windowed rate.
                     Default: 100

    Training configuration (training_config):
        (none required — MLE is closed-form)
    """

    MODEL_TYPE: str = "Exponential"
    SUPPORTED_TASK_TYPES: set = {TaskType.TIME_TO_EVENT}

    def __init__(
        self,
        task_type: TaskType,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
    ):
        super().__init__(task_type, model_config, training_config)
        self._rate: Optional[float] = None  # lambda
        self._mean_delta: Optional[float] = None  # 1/lambda
        self._train_deltas: Optional[np.ndarray] = None

    def fit(self, dataset: Dataset) -> Dict[str, Any]:
        """Fit the Exponential distribution via MLE.

        MLE for Exponential: lambda_hat = n / sum(x_i), i.e. 1/mean.
        """
        self._validate_dataset_type(dataset)

        target = self._get_target_array(dataset)
        train_data = target[dataset.train_indices]

        min_delta = self.model_config.get("min_delta", 0.0)
        if min_delta > 0:
            train_data = train_data[train_data >= min_delta]

        if len(train_data) == 0:
            raise ValueError(
                "No training samples remaining after filtering "
                f"(min_delta={min_delta})"
            )

        self._train_deltas = train_data
        self._mean_delta = float(np.mean(train_data))

        if self._mean_delta <= 0:
            raise ValueError(
                f"Mean delta is {self._mean_delta} — cannot fit "
                "Exponential with non-positive mean"
            )

        self._rate = 1.0 / self._mean_delta
        self._is_fitted = True

        # Log-likelihood: sum(log(lambda) - lambda * x_i)
        log_likelihood = float(
            len(train_data) * np.log(self._rate)
            - self._rate * np.sum(train_data)
        )

        training_info = {
            "rate_lambda": self._rate,
            "mean_delta": self._mean_delta,
            "std_delta": float(np.std(train_data)),
            "median_delta": float(np.median(train_data)),
            "n_train": len(train_data),
            "n_filtered_out": len(target[dataset.train_indices]) - len(train_data),
            "log_likelihood": log_likelihood,
            "aic": -2 * log_likelihood + 2,  # 1 parameter
            "bic": -2 * log_likelihood + np.log(len(train_data)),
        }

        logger.info(
            "Exponential model fitted. rate=%.6f, mean_delta=%.2f, n=%d",
            self._rate, self._mean_delta, len(train_data),
        )

        return self._run_validation(dataset, training_info)

    def _post_fit_validate(self, dataset, fit_info):
        result = super()._post_fit_validate(dataset, fit_info)
        target = self._get_target_array(dataset)
        train_data = target[dataset.train_indices]
        min_delta = self.model_config.get("min_delta", 0.0)
        if min_delta > 0:
            train_data = train_data[train_data >= min_delta]
        result.add(check_ks_test(
            train_data, "expon", (0, self._mean_delta),
        ))
        return result

    def predict_pointwise(
        self,
        dataset: Dataset,
        indices: Sequence[int],
    ) -> np.ndarray:
        """Generate point predictions for specified indices.

        For a basic Exponential model, the prediction is always the mean
        (1/lambda). If use_windowed_rate is enabled, predictions use
        a local rate estimate from recent events.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        self._validate_dataset_type(dataset)

        use_windowed = self.model_config.get("use_windowed_rate", False)

        if not use_windowed:
            return np.full(len(indices), self._mean_delta)

        # Windowed: estimate rate from the last `window_size` events
        window_size = self.model_config.get("window_size", 100)
        target = self._get_target_array(dataset)
        min_delta = self.model_config.get("min_delta", 0.0)

        predictions = np.zeros(len(indices))
        for i, idx in enumerate(indices):
            # Look back from this index
            start = max(0, idx - window_size)
            window = target[start:idx]
            if min_delta > 0:
                window = window[window >= min_delta]
            if len(window) > 0:
                predictions[i] = float(np.mean(window))
            else:
                predictions[i] = self._mean_delta

        return np.maximum(predictions, 0.0)

    def forecast(
        self,
        dataset: Dataset,
        horizon_minutes: int,
        sync_unit_name: str = "",
        reference_time: Optional[datetime] = None,
    ) -> PredictionTrajectory:
        """Generate a forecast trajectory.

        For the Exponential model, each future step predicts the same
        expected inter-arrival time (memoryless), with uncertainty from
        the Exponential distribution's variance (1/lambda^2).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before forecasting")

        self._validate_dataset_type(dataset)

        n_steps = horizon_minutes
        point_forecast = np.full(n_steps, self._mean_delta)

        # Exponential variance = 1/lambda^2 = mean^2
        std_dev = self._mean_delta
        lower = np.maximum(point_forecast - 1.645 * std_dev, 0.0)  # 90% CI
        upper = point_forecast + 1.645 * std_dev
        uncertainties = np.full(n_steps, std_dev)

        if reference_time is None:
            last_ts = dataset.timestamps[-1] if dataset.timestamps else datetime.now(UTC)
            reference_time = last_ts + timedelta(minutes=1)

        timestamps = [
            reference_time + timedelta(minutes=i)
            for i in range(n_steps)
        ]

        points = self._create_prediction_points(
            timestamps=timestamps,
            predictions=point_forecast,
            lower_quantiles=lower,
            upper_quantiles=upper,
            uncertainties=uncertainties,
        )

        return PredictionTrajectory(
            sync_unit_name=sync_unit_name,
            task_type=self.task_type,
            horizon_minutes=horizon_minutes,
            points=points,
            model_id=f"exponential_{self.MODEL_TYPE}",
            created_at=datetime.now(UTC),
        )

    def save(self, path: str) -> None:
        """Save the fitted model to disk."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")

        os.makedirs(path, exist_ok=True)

        model_state = {
            "rate": self._rate,
            "mean_delta": self._mean_delta,
        }
        with open(os.path.join(path, "exponential_model.pkl"), "wb") as f:
            pickle.dump(model_state, f)

        config = {
            "model_type": self.MODEL_TYPE,
            "task_type": self.task_type.value,
            "model_config": self.model_config,
            "training_config": self.training_config,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Saved Exponential model to %s", path)

    @classmethod
    def load(
        cls,
        path: str,
        task_type: TaskType,
        model_config: Dict[str, Any],
    ) -> "ExponentialTimeToEventModel":
        """Load a fitted model from disk."""
        config_path = os.path.join(path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                saved_config = json.load(f)
            training_config = saved_config.get("training_config", {})
        else:
            training_config = {}

        model = cls(
            task_type=task_type,
            model_config=model_config,
            training_config=training_config,
        )

        with open(os.path.join(path, "exponential_model.pkl"), "rb") as f:
            state = pickle.load(f)

        model._rate = state["rate"]
        model._mean_delta = state["mean_delta"]
        model._is_fitted = True

        logger.info("Loaded Exponential model from %s", path)
        return model
