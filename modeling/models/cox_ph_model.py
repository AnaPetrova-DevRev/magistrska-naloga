"""Cox Proportional Hazards model for time-to-event forecasting.

The Cox PH model is a semi-parametric survival model that can leverage
covariates (e.g., hour_of_day, day_of_week) to predict time-to-event.
It estimates hazard ratios without assuming a specific baseline hazard form.
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
from modeling.validation import ValidationResult, check_covariates_for_cox

logger = logging.getLogger(__name__)


class CoxPHTimeToEventModel(BaseModel):
    """Cox Proportional Hazards model for time-to-event forecasting.

    Requires covariates in the dataset to be useful.  Falls back to a
    baseline hazard estimate (similar to KM) if no covariates are present.

    Configuration options (model_config):
        min_delta: float - minimum delta to include (default 0.0).
        penalizer: float - L2 regularisation strength (default 0.01).
    """

    MODEL_TYPE: str = "CoxPH"
    SUPPORTED_TASK_TYPES: set = {TaskType.TIME_TO_EVENT}

    def __init__(
        self,
        task_type: TaskType,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
    ):
        super().__init__(task_type, model_config, training_config)
        self._median_survival: Optional[float] = None
        self._mean_survival: Optional[float] = None
        self._cph_summary: Optional[Dict[str, Any]] = None
        self._covariate_names: list = []

    def fit(self, dataset: Dataset) -> Dict[str, Any]:
        import pandas as pd
        from lifelines import CoxPHFitter

        self._validate_dataset_type(dataset)
        target = self._get_target_array(dataset)
        train_data = target[dataset.train_indices]

        min_delta = self.model_config.get("min_delta", 0.0)
        penalizer = self.model_config.get("penalizer", 0.01)

        # Build DataFrame with duration + covariates
        train_idx = dataset.train_indices
        df = pd.DataFrame({"duration": train_data, "event": 1})

        self._cov_means: Dict[str, float] = {}
        self._cov_stds: Dict[str, float] = {}

        if dataset.covariates:
            self._covariate_names = sorted(dataset.covariates.keys())
            for cov_name in self._covariate_names:
                cov_values = np.array(dataset.covariates[cov_name], dtype=float)
                raw = cov_values[train_idx]
                mean, std = float(raw.mean()), float(raw.std())
                if std < 1e-12:
                    std = 1.0
                self._cov_means[cov_name] = mean
                self._cov_stds[cov_name] = std
                df[cov_name] = (raw - mean) / std

        if min_delta > 0:
            df = df[df["duration"] >= min_delta].copy()

        if len(df) == 0:
            raise ValueError(
                f"No training samples after filtering (min_delta={min_delta})"
            )

        # Add tiny jitter to break tied durations — Cox partial likelihood
        # becomes singular with massive ties (common in bursty event data).
        n_unique = df["duration"].nunique()
        if n_unique < len(df) * 0.1:
            rng = np.random.default_rng(42)
            jitter = rng.uniform(0, 1e-4, size=len(df))
            df["duration"] = df["duration"] + jitter
            logger.info(f"Added jitter to break ties: {n_unique} unique "
                        f"values in {len(df)} rows")

        n_original = len(train_data)
        n_used = len(df)

        step_size = self.model_config.get("step_size", None)
        fit_opts = {"step_size": step_size} if step_size else {}

        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(df, duration_col="duration", event_col="event",
                fit_options=fit_opts)

        self._median_survival = float(np.median(df["duration"]))
        self._mean_survival = float(df["duration"].mean())
        self._is_fitted = True

        # Store summary for metadata
        summary_df = cph.summary
        self._cph_summary = {
            "coefficients": {col: float(summary_df.loc[col, "coef"])
                             for col in summary_df.index},
            "hazard_ratios": {col: float(summary_df.loc[col, "exp(coef)"])
                              for col in summary_df.index},
        }

        fit_info = {
            "median_survival_time": self._median_survival,
            "mean_delta": self._mean_survival,
            "std_delta": float(df["duration"].std()),
            "n_train": n_used,
            "n_filtered_out": n_original - n_used,
            "n_covariates": len(self._covariate_names),
            "covariate_names": self._covariate_names,
            "concordance_index": float(cph.concordance_index_),
            "coefficients": self._cph_summary["coefficients"],
            "penalizer": penalizer,
        }
        return self._run_validation(dataset, fit_info)

    def _pre_fit_validate(self, dataset):
        result = super()._pre_fit_validate(dataset)
        result.add(check_covariates_for_cox(bool(dataset.covariates)))
        return result

    def predict_pointwise(
        self, dataset: Dataset, indices: Sequence[int],
    ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        self._validate_dataset_type(dataset)
        return np.full(len(indices), self._median_survival)

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
        point_forecast = np.full(n_steps, self._median_survival)
        std = abs(self._mean_survival - self._median_survival) + 1e-6
        lower = np.maximum(point_forecast - 1.645 * std, 0.0)
        upper = point_forecast + 1.645 * std
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
            model_id=f"cox_ph_{self.MODEL_TYPE}",
        )

    def save(self, path: str) -> None:
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        os.makedirs(path, exist_ok=True)

        state = {
            "median_survival": self._median_survival,
            "mean_survival": self._mean_survival,
            "cph_summary": self._cph_summary,
            "covariate_names": self._covariate_names,
            "cov_means": self._cov_means,
            "cov_stds": self._cov_stds,
        }
        with open(os.path.join(path, "cox_ph_model.pkl"), "wb") as f:
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
        with open(os.path.join(path, "cox_ph_model.pkl"), "rb") as f:
            state = pickle.load(f)

        model._median_survival = state["median_survival"]
        model._mean_survival = state["mean_survival"]
        model._cph_summary = state["cph_summary"]
        model._covariate_names = state["covariate_names"]
        model._cov_means = state.get("cov_means", {})
        model._cov_stds = state.get("cov_stds", {})
        model._is_fitted = True
        return model
