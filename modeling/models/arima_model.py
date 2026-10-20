"""ARIMA model wrapper for event-rate forecasting.

This module provides an ARIMA-based model for EVENT_RATE forecasting tasks.
It uses statsmodels ARIMA/SARIMAX implementations and is well-suited for
small to medium-sized univariate time series.
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
    EventRateDataset,
    PredictionPoint,
    PredictionTrajectory,
    TaskType,
)
from modeling.validation import ValidationResult, check_stationarity

logger = logging.getLogger(__name__)


class ARIMAEventRateModel(BaseModel):
    """ARIMA model for event-rate (events per minute) forecasting.

    Configuration options (model_config):
        order: tuple (p, d, q) for ARIMA. Default: (1, 0, 1)
        seasonal_order: tuple (P, D, Q, s) for SARIMAX. Default: None (no seasonality)
        enforce_stationarity: bool. Default: True
        enforce_invertibility: bool. Default: True
        trend: str, one of 'n', 'c', 't', 'ct'. Default: 'c'

    Training configuration (training_config):
        max_iter: Maximum iterations for optimization. Default: 50
    """

    MODEL_TYPE: str = "ARIMA"
    SUPPORTED_TASK_TYPES: set = {TaskType.EVENT_RATE}

    def __init__(
        self,
        task_type: TaskType,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
    ):
        super().__init__(task_type, model_config, training_config)
        self._model = None
        self._fitted_model = None
        self._last_train_values = None
        self._freq: str = "min"

    def fit(self, dataset: Dataset) -> Dict[str, Any]:
        """Train the ARIMA model on the training split.

        Args:
            dataset: An EventRateDataset with train_indices set.

        Returns:
            Dict with training metadata including AIC, BIC, and convergence info.
        """
        self._validate_dataset_type(dataset)

        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            raise ImportError(
                "statsmodels is required for ARIMA models. "
                "Install with: pip install statsmodels"
            )

        target = self._get_target_array(dataset)
        train_indices = dataset.train_indices

        if not train_indices:
            raise ValueError("No training indices provided in dataset")

        train_data = target[train_indices]

        order = tuple(self.model_config.get("order", (1, 0, 1)))
        seasonal_order = self.model_config.get("seasonal_order")
        trend = self.model_config.get("trend", "c")
        enforce_stationarity = self.model_config.get("enforce_stationarity", True)
        enforce_invertibility = self.model_config.get("enforce_invertibility", True)
        max_iter = self.training_config.get("max_iter", 50)

        logger.info(
            "Fitting ARIMA model with order=%s, seasonal_order=%s, trend=%s",
            order, seasonal_order, trend
        )

        if seasonal_order is not None:
            seasonal_order = tuple(seasonal_order)
            model = SARIMAX(
                train_data,
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
                enforce_stationarity=enforce_stationarity,
                enforce_invertibility=enforce_invertibility,
            )
            self._fitted_model = model.fit(maxiter=max_iter, disp=False)
        else:
            model = ARIMA(
                train_data,
                order=order,
                trend=trend,
                enforce_stationarity=enforce_stationarity,
                enforce_invertibility=enforce_invertibility,
            )
            method_kwargs = {"maxiter": max_iter, "disp": False}
            self._fitted_model = model.fit(method_kwargs=method_kwargs)
        self._model = model
        self._last_train_values = train_data
        train_ts = [dataset.timestamps[i] for i in train_indices]
        self._freq = self._infer_freq(train_ts)
        self._is_fitted = True

        training_info = {
            "aic": float(self._fitted_model.aic),
            "bic": float(self._fitted_model.bic),
            "converged": self._fitted_model.mle_retvals.get("converged", False),
            "n_train": len(train_data),
            "order": order,
            "seasonal_order": seasonal_order,
        }

        logger.info("ARIMA model fitted. AIC=%.2f, BIC=%.2f", training_info["aic"], training_info["bic"])

        return self._run_validation(dataset, training_info)

    def _pre_fit_validate(self, dataset):
        result = super()._pre_fit_validate(dataset)
        target = self._get_target_array(dataset)
        train_data = target[dataset.train_indices]
        d = self.model_config.get("order", (1, 0, 1))
        if isinstance(d, (list, tuple)) and len(d) >= 2 and d[1] == 0:
            result.add(check_stationarity(train_data))
        return result

    def predict_pointwise(
        self,
        dataset: Dataset,
        indices: Sequence[int],
    ) -> np.ndarray:
        """Generate point predictions for specified indices.

        For ARIMA, this performs in-sample or out-of-sample prediction
        depending on where the indices fall relative to the training data.

        Args:
            dataset: The dataset containing the time series.
            indices: Which indices to generate predictions for.

        Returns:
            A 1D numpy array of predictions.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        self._validate_dataset_type(dataset)

        target = self._get_target_array(dataset)
        predictions = np.zeros(len(indices))

        train_end = max(dataset.train_indices) if dataset.train_indices else 0

        # Split indices into in-sample and out-of-sample
        oos_max_steps = 0
        for idx in indices:
            if idx > train_end:
                oos_max_steps = max(oos_max_steps, idx - train_end)

        # Pre-compute a single forecast covering all out-of-sample indices
        oos_forecast = None
        if oos_max_steps > 0:
            oos_forecast = self._fitted_model.forecast(steps=oos_max_steps)

        for i, idx in enumerate(indices):
            if idx <= train_end:
                pred = self._fitted_model.predict(start=idx, end=idx)
                predictions[i] = pred[0] if len(pred) > 0 else target[idx]
            else:
                step_idx = idx - train_end - 1  # 0-based index into forecast
                predictions[i] = oos_forecast[step_idx] if oos_forecast is not None else 0.0

        predictions = np.maximum(predictions, 0.0)

        return predictions

    def forecast(
        self,
        dataset: Dataset,
        horizon_minutes: int,
        sync_unit_name: str = "",
        reference_time: Optional[datetime] = None,
    ) -> PredictionTrajectory:
        """Generate a forecast trajectory for the RL agent.

        Args:
            dataset: Dataset with recent history for context.
            horizon_minutes: How many minutes into the future to forecast.
            reference_time: Starting timestamp for the forecast.

        Returns:
            A PredictionTrajectory with per-minute predictions and confidence intervals.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before forecasting")

        self._validate_dataset_type(dataset)

        step_minutes = self._freq_to_minutes(self._freq)
        n_native_steps = int(np.ceil(horizon_minutes / step_minutes)) if step_minutes > 1 else horizon_minutes
        n_native_steps = max(n_native_steps, 1)

        forecast_result = self._fitted_model.get_forecast(steps=n_native_steps)
        predicted_mean = forecast_result.predicted_mean
        native_forecast = predicted_mean.values if hasattr(predicted_mean, 'values') else np.asarray(predicted_mean)

        try:
            conf_int = forecast_result.conf_int(alpha=0.1)
            if hasattr(conf_int, 'iloc'):
                native_lower = conf_int.iloc[:, 0].values
                native_upper = conf_int.iloc[:, 1].values
            else:
                native_lower = conf_int[:, 0]
                native_upper = conf_int[:, 1]
        except Exception:
            native_lower = native_forecast - np.std(native_forecast)
            native_upper = native_forecast + np.std(native_forecast)

        is_rate = step_minutes > 1.0
        point_forecast = self._interpolate_to_minutes(native_forecast, step_minutes, horizon_minutes, is_cumulative_rate=is_rate)
        lower = self._interpolate_to_minutes(native_lower, step_minutes, horizon_minutes, is_cumulative_rate=is_rate)
        upper = self._interpolate_to_minutes(native_upper, step_minutes, horizon_minutes, is_cumulative_rate=is_rate)

        point_forecast = np.maximum(point_forecast, 0.0)
        lower = np.maximum(lower, 0.0)
        upper = np.maximum(upper, 0.0)
        uncertainties = (upper - lower) / 2.0

        if reference_time is None:
            last_ts = dataset.timestamps[-1] if dataset.timestamps else datetime.now(UTC)
            reference_time = last_ts + timedelta(minutes=1)

        timestamps = [
            reference_time + timedelta(minutes=i)
            for i in range(horizon_minutes)
        ]

        points = self._create_prediction_points(
            timestamps=timestamps,
            predictions=point_forecast,
            lower_quantiles=lower,
            upper_quantiles=upper,
            uncertainties=uncertainties,
        )

        trajectory = PredictionTrajectory(
            sync_unit_name=sync_unit_name,
            task_type=self.task_type,
            horizon_minutes=horizon_minutes,
            points=points,
            model_id=f"arima_{self.MODEL_TYPE}",
            created_at=datetime.now(UTC),
        )

        return trajectory

    def save(self, path: str) -> None:
        """Save the trained model to disk.

        Args:
            path: Directory path where model artifacts should be saved.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")

        os.makedirs(path, exist_ok=True)

        model_path = os.path.join(path, "arima_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self._fitted_model, f)

        config = {
            "model_type": self.MODEL_TYPE,
            "task_type": self.task_type.value,
            "model_config": self.model_config,
            "training_config": self.training_config,
            "freq": self._freq,
        }
        config_path = os.path.join(path, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Saved ARIMA model to %s", path)

    @classmethod
    def load(
        cls,
        path: str,
        task_type: TaskType,
        model_config: Dict[str, Any],
    ) -> "ARIMAEventRateModel":
        """Load a trained model from disk.

        Args:
            path: Directory path containing the saved model artifacts.
            task_type: The task type this model was trained for.
            model_config: The model configuration.

        Returns:
            A model instance ready for prediction.
        """
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

        model_path = os.path.join(path, "arima_model.pkl")
        with open(model_path, "rb") as f:
            model._fitted_model = pickle.load(f)

        model._freq = saved_config.get("freq", "min") if os.path.exists(config_path) else "min"
        model._is_fitted = True

        logger.info("Loaded ARIMA model from %s", path)

        return model
