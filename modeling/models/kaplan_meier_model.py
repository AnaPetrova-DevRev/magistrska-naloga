"""Kaplan-Meier non-parametric model for time-to-event forecasting.

The Kaplan-Meier estimator provides a non-parametric estimate of the
survival function.  It makes no distributional assumptions and serves
as a strong baseline for time-to-event tasks.
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

logger = logging.getLogger(__name__)


class KaplanMeierTimeToEventModel(BaseModel):
    """Kaplan-Meier non-parametric survival estimator.

    Uses the median survival time as the point prediction.
    Does not use covariates.

    Configuration options (model_config):
        min_delta: float - minimum delta to include (default 0.0).
    """

    MODEL_TYPE: str = "KaplanMeier"
    SUPPORTED_TASK_TYPES: set = {TaskType.TIME_TO_EVENT}

    def __init__(
        self,
        task_type: TaskType,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
    ):
        super().__init__(task_type, model_config, training_config)
        self._median: Optional[float] = None
        self._mean: Optional[float] = None
        self._ci_lower: Optional[float] = None
        self._ci_upper: Optional[float] = None

    def fit(self, dataset: Dataset) -> Dict[str, Any]:
        from lifelines import KaplanMeierFitter

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

        kmf = KaplanMeierFitter()
        # All observations are uncensored (event observed)
        kmf.fit(train_data, event_observed=np.ones(len(train_data)))

        self._median = float(kmf.median_survival_time_)
        self._mean = float(np.mean(train_data))

        # Confidence interval at median
        ci = kmf.confidence_interval_survival_function_
        # Use 25th and 75th percentile of the data as simple CI
        self._ci_lower = float(np.percentile(train_data, 5))
        self._ci_upper = float(np.percentile(train_data, 95))
        self._is_fitted = True

        n = len(train_data)
        fit_info = {
            "median_survival_time": self._median,
            "mean_delta": self._mean,
            "std_delta": float(np.std(train_data)),
            "n_train": n,
            "n_filtered_out": len(target[dataset.train_indices]) - n,
            "ci_lower_5pct": self._ci_lower,
            "ci_upper_95pct": self._ci_upper,
        }
        return self._run_validation(dataset, fit_info)

    def predict_pointwise(
        self, dataset: Dataset, indices: Sequence[int],
    ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        self._validate_dataset_type(dataset)
        return np.full(len(indices), self._median)

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

        n_steps = horizon_minutes
        point_forecast = np.full(n_steps, self._median)
        lower = np.full(n_steps, self._ci_lower)
        upper = np.full(n_steps, self._ci_upper)
        uncertainties = np.full(n_steps, (self._ci_upper - self._ci_lower) / 2)

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
            model_id=f"kaplan_meier_{self.MODEL_TYPE}",
        )

    def save(self, path: str) -> None:
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        os.makedirs(path, exist_ok=True)

        state = {
            "median": self._median,
            "mean": self._mean,
            "ci_lower": self._ci_lower,
            "ci_upper": self._ci_upper,
        }
        with open(os.path.join(path, "kaplan_meier_model.pkl"), "wb") as f:
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
        with open(os.path.join(path, "kaplan_meier_model.pkl"), "rb") as f:
            state = pickle.load(f)

        model._median = state["median"]
        model._mean = state["mean"]
        model._ci_lower = state["ci_lower"]
        model._ci_upper = state["ci_upper"]
        model._is_fitted = True
        return model
