"""Zero-inflated parametric models for time-to-event forecasting.

Handles datasets where a large fraction of inter-arrival times are zero or
near-zero (e.g. bulk API batch arrivals).  The model is a mixture:

    P(immediate) * delta(0)  +  P(wait) * f(t | t >= threshold)

where f() is a standard parametric distribution (LogNormal, Gamma, or Weibull)
fitted on the positive-only subset.

The positive component delegates to the existing parametric model classes,
reusing their MLE fitting and quantile computation.  This composition
pattern follows EnsembleModel.

Configuration:
    positive_distribution: str — "LogNormal", "Gamma", or "Weibull"
    zero_threshold: int — boundary in minutes between "immediate" and
                    "genuine wait" (default 0: events with 0-minute delta are immediate)
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
    TimeToEventDataset,
)

logger = logging.getLogger(__name__)

# Map from config name to (model class import path, model module, pickle filename)
_DISTRIBUTION_REGISTRY = {
    "LogNormal": (
        "modeling.models.lognormal_model",
        "LogNormalTimeToEventModel",
    ),
    "Gamma": (
        "modeling.models.gamma_model",
        "GammaTimeToEventModel",
    ),
    "Weibull": (
        "modeling.models.weibull_model",
        "WeibullTimeToEventModel",
    ),
}


def _get_positive_model_class(dist_name: str):
    """Dynamically import and return the positive-component model class."""
    if dist_name not in _DISTRIBUTION_REGISTRY:
        raise ValueError(
            f"Unknown positive_distribution '{dist_name}'. "
            f"Available: {sorted(_DISTRIBUTION_REGISTRY.keys())}"
        )
    module_path, class_name = _DISTRIBUTION_REGISTRY[dist_name]
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class ZeroInflatedTTEModel(BaseModel):
    """Zero-inflated parametric model for time-to-event forecasting.

    Configuration options (model_config):
        positive_distribution: str — "LogNormal", "Gamma", or "Weibull".
            Default: "LogNormal".
        zero_threshold: int — values at or below this threshold (in minutes)
            are treated as "immediate" arrivals.  Default: 0.

    The positive component inherits any model_config keys understood by
    the underlying distribution (e.g. min_delta, fix_shape for Weibull).
    """

    MODEL_TYPE: str = "ZI_LogNormal"  # Overridden per-instance based on config
    SUPPORTED_TASK_TYPES: set = {TaskType.TIME_TO_EVENT}

    def __init__(
        self,
        task_type: TaskType,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
    ):
        # Resolve MODEL_TYPE before super().__init__ for correct logging
        dist_name = model_config.get("positive_distribution", "LogNormal")
        # Temporarily set to allow super().__init__ validation
        super().__init__(task_type, model_config, training_config)
        self.MODEL_TYPE = f"ZI_{dist_name}"

        self._dist_name: str = dist_name
        self._zero_threshold: float = model_config.get("zero_threshold", 0)
        self._p_immediate: Optional[float] = None
        self._positive_model: Optional[BaseModel] = None

    def _create_positive_model(self) -> BaseModel:
        """Instantiate the positive-component parametric model."""
        model_class = _get_positive_model_class(self._dist_name)

        # Build config for the inner model: pass through relevant keys,
        # set min_delta to zero_threshold so it doesn't re-filter
        inner_config = dict(self.model_config)
        inner_config.pop("positive_distribution", None)
        inner_config.pop("zero_threshold", None)
        inner_config["min_delta"] = 0.0  # We pre-filter ourselves

        return model_class(
            task_type=TaskType.TIME_TO_EVENT,
            model_config=inner_config,
            training_config=self.training_config,
        )

    def _make_positive_dataset(
        self, dataset: Dataset, train_data: np.ndarray,
    ) -> TimeToEventDataset:
        """Create a filtered dataset containing only positive-threshold values.

        Preserves the train/val/test split structure on the filtered data.
        """
        target = self._get_target_array(dataset)
        threshold = self._zero_threshold

        # Filter each split independently
        new_timestamps = []
        new_values = []
        new_train = []
        new_val = []
        new_test = []

        for split_name, indices in [
            ("train", dataset.train_indices),
            ("val", dataset.val_indices),
            ("test", dataset.test_indices),
        ]:
            for idx in indices:
                if target[idx] >= threshold:
                    new_idx = len(new_timestamps)
                    new_timestamps.append(dataset.timestamps[idx])
                    new_values.append(float(target[idx]))
                    if split_name == "train":
                        new_train.append(new_idx)
                    elif split_name == "val":
                        new_val.append(new_idx)
                    else:
                        new_test.append(new_idx)

        if not new_train:
            raise ValueError(
                f"No training samples above zero_threshold={threshold:.4f}. "
                "All inter-arrival times are 'immediate'."
            )

        return TimeToEventDataset(
            timestamps=new_timestamps,
            minutes_to_next_event=new_values,
            covariates={},  # Positive component uses no covariates
            train_indices=new_train,
            val_indices=new_val,
            test_indices=new_test,
        )

    def fit(self, dataset: Dataset) -> Dict[str, Any]:
        """Fit the zero-inflated mixture model.

        1. Compute P(immediate) from fraction of training data below threshold.
        2. Fit the positive-component distribution on filtered data.
        """
        self._validate_dataset_type(dataset)

        target = self._get_target_array(dataset)
        train_data = target[dataset.train_indices]

        # Stage 1: Estimate P(immediate)
        n_immediate = int(np.sum(train_data < self._zero_threshold))
        n_total = len(train_data)
        self._p_immediate = n_immediate / n_total if n_total > 0 else 0.0

        # Stage 2: Fit positive-component distribution
        self._positive_model = self._create_positive_model()
        positive_dataset = self._make_positive_dataset(dataset, train_data)

        positive_info = self._positive_model.fit(positive_dataset)

        self._is_fitted = True

        # Mixture expected value
        positive_mean = positive_info.get("mean_delta", 0.0)
        mixture_mean = (1.0 - self._p_immediate) * positive_mean

        fit_info = {
            "p_immediate": self._p_immediate,
            "frac_above_threshold": 1.0 - self._p_immediate,
            "zero_threshold": self._zero_threshold,
            "n_train_total": n_total,
            "n_immediate": n_immediate,
            "n_positive": n_total - n_immediate,
            "positive_distribution": self._dist_name,
            "mixture_mean_delta": mixture_mean,
            "positive_mean_delta": positive_mean,
            "positive_fit_info": {
                k: v for k, v in positive_info.items()
                if k not in ("warnings",)
            },
        }

        logger.info(
            "ZI_%s fitted. P(immediate)=%.3f, positive_mean=%.2f, "
            "mixture_mean=%.2f, n=%d (%d immediate)",
            self._dist_name, self._p_immediate, positive_mean,
            mixture_mean, n_total, n_immediate,
        )

        return self._run_validation(dataset, fit_info)

    def predict_pointwise(
        self,
        dataset: Dataset,
        indices: Sequence[int],
    ) -> np.ndarray:
        """Generate point predictions: (1 - P(immediate)) * E[delta | positive].

        For observations below threshold, the true value is ~0 and the
        mixture prediction is appropriately small.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        self._validate_dataset_type(dataset)

        # The mixture expectation is constant (parametric model)
        positive_preds = self._positive_model.predict_pointwise(
            dataset, [0],  # Parametric models return constant anyway
        )
        positive_mean = float(positive_preds[0])
        mixture_pred = (1.0 - self._p_immediate) * positive_mean

        return np.full(len(indices), max(mixture_pred, 0.0))

    def forecast(
        self,
        dataset: Dataset,
        horizon_minutes: int,
        sync_unit_name: str = "",
        reference_time: Optional[datetime] = None,
    ) -> PredictionTrajectory:
        """Generate a mixture forecast trajectory."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before forecasting")
        self._validate_dataset_type(dataset)

        # Get positive-component forecast for quantiles
        positive_traj = self._positive_model.forecast(
            dataset, horizon_minutes, sync_unit_name, reference_time,
        )

        # Scale by mixture probability
        p_wait = 1.0 - self._p_immediate
        n_steps = horizon_minutes

        point_forecast = np.array([
            p_wait * pt.expected_events for pt in positive_traj.points[:n_steps]
        ])
        point_forecast = np.maximum(point_forecast, 0.0)

        # Scale quantiles by mixture weight
        lower = np.array([
            p_wait * (pt.lower_quantile or 0.0)
            for pt in positive_traj.points[:n_steps]
        ])
        upper = np.array([
            p_wait * (pt.upper_quantile or 0.0)
            for pt in positive_traj.points[:n_steps]
        ])
        lower = np.maximum(lower, 0.0)

        uncertainties = np.array([
            p_wait * (pt.uncertainty or 0.0)
            for pt in positive_traj.points[:n_steps]
        ])

        if reference_time is None:
            last_ts = dataset.timestamps[-1] if dataset.timestamps else datetime.now(UTC)
            reference_time = last_ts + timedelta(minutes=1)

        timestamps = [
            reference_time + timedelta(minutes=i) for i in range(n_steps)
        ]

        points = self._create_prediction_points(
            timestamps, point_forecast, lower, upper, uncertainties,
        )
        return PredictionTrajectory(
            sync_unit_name=sync_unit_name,
            task_type=self.task_type,
            horizon_minutes=horizon_minutes,
            points=points,
            model_id=f"zi_{self._dist_name.lower()}",
            created_at=datetime.now(UTC),
        )

    def save(self, path: str) -> None:
        """Save the zero-inflated model and its positive component."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        os.makedirs(path, exist_ok=True)

        # Save positive component in a subdirectory
        positive_path = os.path.join(path, "positive_model")
        self._positive_model.save(positive_path)

        # Save ZI-specific state
        state = {
            "p_immediate": self._p_immediate,
            "dist_name": self._dist_name,
            "zero_threshold": self._zero_threshold,
        }
        with open(os.path.join(path, "zi_state.pkl"), "wb") as f:
            pickle.dump(state, f)

        config = {
            "model_type": self.MODEL_TYPE,
            "task_type": self.task_type.value,
            "model_config": self.model_config,
            "training_config": self.training_config,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Saved ZI_%s model to %s", self._dist_name, path)

    @classmethod
    def load(
        cls,
        path: str,
        task_type: TaskType,
        model_config: Dict[str, Any],
    ) -> "ZeroInflatedTTEModel":
        """Load a fitted ZI model from disk."""
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

        # Load ZI-specific state
        with open(os.path.join(path, "zi_state.pkl"), "rb") as f:
            state = pickle.load(f)

        model._p_immediate = state["p_immediate"]
        model._dist_name = state["dist_name"]
        model._zero_threshold = state["zero_threshold"]
        model.MODEL_TYPE = f"ZI_{model._dist_name}"

        # Load positive component
        positive_path = os.path.join(path, "positive_model")
        positive_class = _get_positive_model_class(model._dist_name)

        inner_config = dict(model_config)
        inner_config.pop("positive_distribution", None)
        inner_config.pop("zero_threshold", None)
        inner_config["min_delta"] = 0.0

        model._positive_model = positive_class.load(
            positive_path, task_type, inner_config,
        )
        model._is_fitted = True

        logger.info("Loaded ZI_%s model from %s", model._dist_name, path)
        return model
