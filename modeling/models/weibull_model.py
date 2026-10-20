"""Weibull distribution model for time-to-event forecasting.

The Weibull distribution generalises the Exponential by adding a shape
parameter (k) that controls whether the hazard rate increases (k > 1),
decreases (k < 1), or stays constant (k = 1, reducing to Exponential).

Parameters:
    k (shape): Controls hazard trend.  k > 1 means events become more
               likely the longer you wait (e.g. aging systems).
               k < 1 means events become less likely (e.g. burn-in).
    lambda (scale): Controls the overall time scale.

MLE is performed via scipy.stats.weibull_min.fit().
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


class WeibullTimeToEventModel(BaseModel):
    """Weibull distribution model for time-to-event forecasting.

    Fits a Weibull(k, lambda) distribution to training inter-arrival
    times via MLE.  Point predictions use the distribution mean:

        E[X] = lambda * Gamma(1 + 1/k)

    Configuration options (model_config):
        min_delta: float — minimum delta to include (filters near-zero
                   duplicates). Default: 0.0
        fix_shape: float or null — if set, fixes the shape parameter k
                   to this value instead of estimating it. Useful for
                   testing k=1 (Exponential) or domain-informed values.
                   Default: null (estimate from data)

    Training configuration (training_config):
        (none required — MLE via scipy)
    """

    MODEL_TYPE: str = "Weibull"
    SUPPORTED_TASK_TYPES: set = {TaskType.TIME_TO_EVENT}

    def __init__(
        self,
        task_type: TaskType,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
    ):
        super().__init__(task_type, model_config, training_config)
        self._shape: Optional[float] = None  # k
        self._scale: Optional[float] = None  # lambda
        self._loc: float = 0.0
        self._mean_delta: Optional[float] = None

    def fit(self, dataset: Dataset) -> Dict[str, Any]:
        """Fit the Weibull distribution via MLE using scipy."""
        self._validate_dataset_type(dataset)

        try:
            from scipy.stats import weibull_min
            from scipy.special import gamma as gamma_fn
        except ImportError:
            raise ImportError(
                "scipy is required for the Weibull model. "
                "Install with: pip install scipy"
            )

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

        n_original = len(target[dataset.train_indices])
        n_filtered = len(train_data)

        fix_shape = self.model_config.get("fix_shape")

        if fix_shape is not None:
            # Fix shape, only estimate scale (and loc=0)
            shape, loc, scale = weibull_min.fit(
                train_data, f0=float(fix_shape), floc=0
            )
        else:
            shape, loc, scale = weibull_min.fit(train_data, floc=0)

        self._shape = float(shape)
        self._scale = float(scale)
        self._loc = float(loc)

        # Weibull mean: scale * Gamma(1 + 1/shape)
        self._mean_delta = float(self._scale * gamma_fn(1.0 + 1.0 / self._shape))

        self._is_fitted = True

        # Log-likelihood
        log_likelihood = float(np.sum(
            weibull_min.logpdf(train_data, self._shape, loc=self._loc, scale=self._scale)
        ))

        n_params = 1 if fix_shape is not None else 2  # scale only, or shape+scale
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(n_filtered)

        # Weibull variance: scale^2 * [Gamma(1+2/k) - Gamma(1+1/k)^2]
        variance = self._scale ** 2 * (
            gamma_fn(1.0 + 2.0 / self._shape)
            - gamma_fn(1.0 + 1.0 / self._shape) ** 2
        )

        training_info = {
            "shape_k": self._shape,
            "scale_lambda": self._scale,
            "mean_delta": self._mean_delta,
            "std_delta": float(np.sqrt(variance)),
            "median_delta": float(
                self._scale * (np.log(2) ** (1.0 / self._shape))
            ),
            "n_train": n_filtered,
            "n_filtered_out": n_original - n_filtered,
            "log_likelihood": log_likelihood,
            "aic": float(aic),
            "bic": float(bic),
            "hazard_interpretation": (
                "increasing" if self._shape > 1
                else "decreasing" if self._shape < 1
                else "constant (exponential)"
            ),
        }

        logger.info(
            "Weibull model fitted. shape=%.4f, scale=%.2f, mean=%.2f, n=%d (%s hazard)",
            self._shape, self._scale, self._mean_delta, n_filtered,
            training_info["hazard_interpretation"],
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
            train_data, "weibull_min",
            (self._shape, self._loc, self._scale),
        ))
        return result

    def predict_pointwise(
        self,
        dataset: Dataset,
        indices: Sequence[int],
    ) -> np.ndarray:
        """Generate point predictions for specified indices.

        Uses the conditional expectation E[T | T > age] where age is
        estimated from the position in the sequence. For the basic mode,
        predictions are the unconditional mean (same as Exponential
        baseline behaviour).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        self._validate_dataset_type(dataset)

        # Basic mode: predict the distribution mean for all points
        return np.full(len(indices), self._mean_delta)

    def forecast(
        self,
        dataset: Dataset,
        horizon_minutes: int,
        sync_unit_name: str = "",
        reference_time: Optional[datetime] = None,
    ) -> PredictionTrajectory:
        """Generate a forecast trajectory."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before forecasting")

        self._validate_dataset_type(dataset)

        try:
            from scipy.stats import weibull_min
        except ImportError:
            raise ImportError("scipy is required for Weibull forecasting")

        n_steps = horizon_minutes
        point_forecast = np.full(n_steps, self._mean_delta)

        # Confidence intervals from Weibull quantiles
        lower = np.full(n_steps, float(
            weibull_min.ppf(0.05, self._shape, loc=self._loc, scale=self._scale)
        ))
        upper = np.full(n_steps, float(
            weibull_min.ppf(0.95, self._shape, loc=self._loc, scale=self._scale)
        ))
        lower = np.maximum(lower, 0.0)

        uncertainties = (upper - lower) / 2.0

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
            model_id=f"weibull_{self.MODEL_TYPE}",
            created_at=datetime.now(UTC),
        )

    def save(self, path: str) -> None:
        """Save the fitted model to disk."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")

        os.makedirs(path, exist_ok=True)

        model_state = {
            "shape": self._shape,
            "scale": self._scale,
            "loc": self._loc,
            "mean_delta": self._mean_delta,
        }
        with open(os.path.join(path, "weibull_model.pkl"), "wb") as f:
            pickle.dump(model_state, f)

        config = {
            "model_type": self.MODEL_TYPE,
            "task_type": self.task_type.value,
            "model_config": self.model_config,
            "training_config": self.training_config,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Saved Weibull model to %s", path)

    @classmethod
    def load(
        cls,
        path: str,
        task_type: TaskType,
        model_config: Dict[str, Any],
    ) -> "WeibullTimeToEventModel":
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

        with open(os.path.join(path, "weibull_model.pkl"), "rb") as f:
            state = pickle.load(f)

        model._shape = state["shape"]
        model._scale = state["scale"]
        model._loc = state["loc"]
        model._mean_delta = state["mean_delta"]
        model._is_fitted = True

        logger.info("Loaded Weibull model from %s", path)
        return model
