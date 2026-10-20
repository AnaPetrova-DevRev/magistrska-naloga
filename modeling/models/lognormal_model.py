"""Log-Normal distribution model for time-to-event forecasting.

The Log-Normal distribution is a common choice for modelling positive-valued
inter-arrival times that exhibit right skew.  If X ~ LogNormal(mu, sigma),
then log(X) ~ Normal(mu, sigma).  MLE is available in closed form via the
sample mean and standard deviation of log(X).
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, Optional, Sequence

import numpy as np

from modeling.base_model import BaseModel
from modeling.core_types import (
    Dataset,
    PredictionTrajectory,
    TaskType,
)
from modeling.validation import ValidationResult, check_normality_of_log

logger = logging.getLogger(__name__)


class LogNormalTimeToEventModel(BaseModel):
    """Log-Normal distribution model for time-to-event forecasting.

    Configuration options (model_config):
        min_delta: float - minimum delta to include in fitting (default 0.0).
    """

    MODEL_TYPE: str = "LogNormal"
    SUPPORTED_TASK_TYPES: set = {TaskType.TIME_TO_EVENT}

    def __init__(
        self,
        task_type: TaskType,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
    ):
        super().__init__(task_type, model_config, training_config)
        self._mu: Optional[float] = None
        self._sigma: Optional[float] = None
        self._mean_delta: Optional[float] = None

    def fit(self, dataset: Dataset) -> Dict[str, Any]:
        self._validate_dataset_type(dataset)
        target = self._get_target_array(dataset)
        train_data = target[dataset.train_indices]

        min_delta = self.model_config.get("min_delta", 0.0)
        if min_delta > 0:
            train_data = train_data[train_data >= min_delta]

        if len(train_data) == 0:
            raise ValueError(
                f"No training samples after filtering (min_delta={min_delta})"
            )

        log_data = np.log(train_data)
        self._mu = float(np.mean(log_data))
        self._sigma = float(np.std(log_data, ddof=1)) if len(log_data) > 1 else 1e-6

        # Mean of LogNormal = exp(mu + sigma^2/2)
        self._mean_delta = float(np.exp(self._mu + self._sigma ** 2 / 2))
        self._is_fitted = True

        n = len(train_data)
        log_likelihood = float(
            -n / 2 * np.log(2 * np.pi)
            - n * np.log(self._sigma)
            - np.sum(log_data)
            - np.sum((log_data - self._mu) ** 2) / (2 * self._sigma ** 2)
        )

        fit_info = {
            "mu": self._mu,
            "sigma": self._sigma,
            "mean_delta": self._mean_delta,
            "std_delta": float(np.std(train_data)),
            "median_delta": float(np.exp(self._mu)),
            "n_train": n,
            "n_filtered_out": len(target[dataset.train_indices]) - n,
            "log_likelihood": log_likelihood,
            "aic": -2 * log_likelihood + 4,  # 2 parameters
            "bic": -2 * log_likelihood + 2 * np.log(n),
        }
        return self._run_validation(dataset, fit_info)

    def _post_fit_validate(self, dataset, fit_info):
        result = super()._post_fit_validate(dataset, fit_info)
        target = self._get_target_array(dataset)
        train_data = target[dataset.train_indices]
        min_delta = self.model_config.get("min_delta", 0.0)
        if min_delta > 0:
            train_data = train_data[train_data >= min_delta]
        result.add(check_normality_of_log(train_data))
        return result

    def predict_pointwise(
        self, dataset: Dataset, indices: Sequence[int],
    ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        self._validate_dataset_type(dataset)
        return np.full(len(indices), self._mean_delta)

    def forecast(
        self,
        dataset: Dataset,
        horizon_minutes: int,
        sync_unit_name: str = "",
        reference_time: Optional[datetime] = None,
    ) -> PredictionTrajectory:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before forecasting")
        self._validate_dataset_type(dataset)

        from scipy.stats import lognorm

        n_steps = horizon_minutes
        point_forecast = np.full(n_steps, self._mean_delta)

        dist = lognorm(s=self._sigma, scale=np.exp(self._mu))
        lower = np.full(n_steps, float(dist.ppf(0.05)))
        upper = np.full(n_steps, float(dist.ppf(0.95)))
        std = float(dist.std())
        uncertainties = np.full(n_steps, std)

        if reference_time is None:
            last_ts = dataset.timestamps[-1] if dataset.timestamps else datetime.now(UTC)
            reference_time = last_ts + timedelta(minutes=1)

        timestamps = [reference_time + timedelta(minutes=i) for i in range(n_steps)]
        points = self._create_prediction_points(
            timestamps, point_forecast, lower, upper, uncertainties,
        )
        return PredictionTrajectory(
            sync_unit_name=sync_unit_name,
            task_type=self.task_type,
            horizon_minutes=horizon_minutes,
            points=points,
            model_id=f"lognormal_{self.MODEL_TYPE}",
        )

    def save(self, path: str) -> None:
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        os.makedirs(path, exist_ok=True)

        state = {"mu": self._mu, "sigma": self._sigma, "mean_delta": self._mean_delta}
        with open(os.path.join(path, "lognormal_model.pkl"), "wb") as f:
            pickle.dump(state, f)

        config = {
            "model_type": self.MODEL_TYPE,
            "task_type": self.task_type.value,
            "model_config": self.model_config,
            "training_config": self.training_config,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str, task_type: TaskType, model_config: Dict[str, Any]):
        config_path = os.path.join(path, "config.json")
        training_config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                saved = json.load(f)
            training_config = saved.get("training_config", {})

        model = cls(task_type=task_type, model_config=model_config, training_config=training_config)
        with open(os.path.join(path, "lognormal_model.pkl"), "rb") as f:
            state = pickle.load(f)

        model._mu = state["mu"]
        model._sigma = state["sigma"]
        model._mean_delta = state["mean_delta"]
        model._is_fitted = True
        return model
