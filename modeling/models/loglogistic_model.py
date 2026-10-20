"""Log-Logistic distribution model for time-to-event forecasting.

The Log-Logistic (also known as Fisk) distribution is popular in survival
analysis because it has a closed-form survival function and can model
non-monotone hazard rates.  scipy implements it as ``scipy.stats.fisk``.
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
from modeling.validation import (
    ValidationResult,
    check_fitted_params_finite,
    check_ks_test,
)

logger = logging.getLogger(__name__)


class LogLogisticTimeToEventModel(BaseModel):
    """Log-Logistic (Fisk) distribution model for time-to-event forecasting.

    Configuration options (model_config):
        min_delta: float - minimum delta to include in fitting (default 0.0).
    """

    MODEL_TYPE: str = "LogLogistic"
    SUPPORTED_TASK_TYPES: set = {TaskType.TIME_TO_EVENT}

    def __init__(
        self,
        task_type: TaskType,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
    ):
        super().__init__(task_type, model_config, training_config)
        self._c: Optional[float] = None  # shape (scipy's c parameter)
        self._scale: Optional[float] = None
        self._mean_delta: Optional[float] = None

    def fit(self, dataset: Dataset) -> Dict[str, Any]:
        from scipy.stats import fisk

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

        # fisk.fit returns (c, loc, scale).  Fix loc=0.
        c, loc, scale = fisk.fit(train_data, floc=0)
        self._c = float(c)
        self._scale = float(scale)

        # The log-logistic mean is infinite when shape c <= 1.
        # Fall back to the median (always finite) in that case.
        raw_mean = float(fisk.mean(c, loc=0, scale=scale))
        median = float(fisk.median(c, loc=0, scale=scale))
        if not np.isfinite(raw_mean):
            logger.warning(
                "LogLogistic mean is non-finite (shape c=%.4f <= 1). "
                "Using median (%.2f) as point prediction instead.",
                c, median,
            )
            self._mean_delta = median
        else:
            self._mean_delta = raw_mean
        self._is_fitted = True

        n = len(train_data)
        log_likelihood = float(fisk.logpdf(train_data, c, loc=0, scale=scale).sum())

        fit_info = {
            "shape_c": self._c,
            "scale": self._scale,
            "mean_delta": self._mean_delta,
            "mean_is_finite": bool(np.isfinite(raw_mean)),
            "std_delta": float(np.std(train_data)),
            "median_delta": median,
            "n_train": n,
            "n_filtered_out": len(target[dataset.train_indices]) - n,
            "log_likelihood": log_likelihood,
            "aic": -2 * log_likelihood + 4,  # 2 parameters
            "bic": -2 * log_likelihood + 2 * np.log(n),
        }
        return self._run_validation(dataset, fit_info)

    def _post_fit_validate(self, dataset, fit_info):
        result = super()._post_fit_validate(dataset, fit_info)
        # Check fitted params
        result.add(check_fitted_params_finite({
            "mean": float(fit_info.get("mean_delta", 0)),
        }))
        # KS test against fisk distribution
        target = self._get_target_array(dataset)
        train_data = target[dataset.train_indices]
        min_delta = self.model_config.get("min_delta", 0.0)
        if min_delta > 0:
            train_data = train_data[train_data >= min_delta]
        result.add(check_ks_test(
            train_data, "fisk", (self._c, 0, self._scale),
        ))
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

        from scipy.stats import fisk

        n_steps = horizon_minutes
        point_forecast = np.full(n_steps, self._mean_delta)

        dist = fisk(c=self._c, loc=0, scale=self._scale)
        lower = np.full(n_steps, float(dist.ppf(0.05)))
        upper = np.full(n_steps, float(dist.ppf(0.95)))
        raw_std = float(dist.std())
        uncertainties = np.full(
            n_steps,
            raw_std if np.isfinite(raw_std) else float(upper[0] - lower[0]) / 2,
        )

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
            model_id=f"loglogistic_{self.MODEL_TYPE}",
        )

    def save(self, path: str) -> None:
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        os.makedirs(path, exist_ok=True)

        state = {"c": self._c, "scale": self._scale, "mean_delta": self._mean_delta}
        with open(os.path.join(path, "loglogistic_model.pkl"), "wb") as f:
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
        with open(os.path.join(path, "loglogistic_model.pkl"), "rb") as f:
            state = pickle.load(f)

        model._c = state["c"]
        model._scale = state["scale"]
        model._mean_delta = state["mean_delta"]
        model._is_fitted = True
        return model
