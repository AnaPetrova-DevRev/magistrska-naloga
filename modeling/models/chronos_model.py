"""Chronos foundation model for time-series forecasting.

Amazon's Chronos is a pretrained T5-based model for zero-shot time-series
forecasting.  It generates probabilistic forecasts by tokenising time-series
values and sampling from the language-model head.

Requires: ``pip install chronos-forecasting``
"""

from __future__ import annotations

import json
import logging
import os
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


class ChronosForecastModel(BaseModel):
    """Chronos zero-shot foundation model for time-series forecasting.

    Supports both EVENT_RATE and TIME_TO_EVENT tasks.

    Configuration options (model_config):
        model_name: str - HuggingFace model ID
            (default "amazon/chronos-t5-tiny").
        device: str - "cpu" or "cuda" (default "cpu").
        context_length: int - max context tokens (default 512).
        num_samples: int - number of forecast samples (default 20).
        temperature: float - sampling temperature (default 1.0).
        top_k: int - top-k sampling (default 50).
        top_p: float - nucleus sampling (default 1.0).
    """

    MODEL_TYPE: str = "Chronos"
    SUPPORTED_TASK_TYPES: set = {TaskType.EVENT_RATE, TaskType.TIME_TO_EVENT}

    def __init__(
        self,
        task_type: TaskType,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
    ):
        super().__init__(task_type, model_config, training_config)
        self._pipeline = None
        self._context: Optional[np.ndarray] = None
        self._num_samples: int = model_config.get("num_samples", 20)

    def _load_pipeline(self):
        import torch
        from chronos import ChronosPipeline

        model_name = self.model_config.get("model_name", "amazon/chronos-t5-tiny")
        device = self.model_config.get("device", "cpu")

        self._pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32,
        )

    def fit(self, dataset: Dataset) -> Dict[str, Any]:
        self._validate_dataset_type(dataset)
        target = self._get_target_array(dataset)
        train_data = target[dataset.train_indices]

        self._load_pipeline()
        self._context = train_data.copy()
        self._is_fitted = True

        model_name = self.model_config.get("model_name", "amazon/chronos-t5-tiny")
        fit_info = {
            "model_name": model_name,
            "context_length": len(self._context),
            "n_train": len(train_data),
            "num_samples": self._num_samples,
            "zero_shot": True,
        }
        return self._run_validation(dataset, fit_info)

    def predict_pointwise(
        self, dataset: Dataset, indices: Sequence[int],
    ) -> np.ndarray:
        import torch

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        self._validate_dataset_type(dataset)

        target = self._get_target_array(dataset)
        ctx_len = self.model_config.get("context_length", 512)

        sorted_idx = sorted(indices)
        is_contiguous = (
            len(sorted_idx) > 1
            and sorted_idx[-1] - sorted_idx[0] == len(sorted_idx) - 1
        )

        if is_contiguous:
            first = sorted_idx[0]
            ctx_start = max(0, first - ctx_len)
            context = torch.tensor(target[ctx_start:first], dtype=torch.float32)
            if len(context) == 0:
                return np.full(len(indices), float(np.mean(self._context)))

            forecast = self._pipeline.predict(
                context.unsqueeze(0),
                prediction_length=len(sorted_idx),
                num_samples=self._num_samples,
            )
            preds = np.median(forecast[0].numpy(), axis=0)
            idx_map = {idx: i for i, idx in enumerate(sorted_idx)}
            predictions = np.array([preds[idx_map[idx]] for idx in indices])
            return np.maximum(predictions, 0.0)

        # Non-contiguous: per-index prediction
        predictions = np.zeros(len(indices))
        for i, idx in enumerate(indices):
            ctx_start = max(0, idx - ctx_len)
            context = torch.tensor(target[ctx_start:idx], dtype=torch.float32)
            if len(context) == 0:
                predictions[i] = float(np.mean(self._context))
                continue
            forecast = self._pipeline.predict(
                context.unsqueeze(0), prediction_length=1,
                num_samples=self._num_samples,
            )
            predictions[i] = max(float(np.median(forecast[0].numpy())), 0.0)

        return predictions

    def forecast(
        self,
        dataset: Dataset,
        horizon_minutes: int,
        sync_unit_name: str = "",
        reference_time: Optional[datetime] = None,
    ) -> PredictionTrajectory:
        import torch

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before forecasting")
        self._validate_dataset_type(dataset)

        ctx_len = self.model_config.get("context_length", 512)
        context = torch.tensor(
            self._context[-ctx_len:], dtype=torch.float32,
        )

        freq = self._infer_freq(dataset.timestamps)
        step_minutes = self._freq_to_minutes(freq)
        n_native_steps = int(np.ceil(horizon_minutes / step_minutes)) if step_minutes > 1 else horizon_minutes
        n_native_steps = max(n_native_steps, 1)

        forecast = self._pipeline.predict(
            context.unsqueeze(0),
            prediction_length=n_native_steps,
            num_samples=self._num_samples,
        )

        samples = forecast[0].numpy()
        native_forecast = np.median(samples, axis=0)
        native_lower = np.quantile(samples, 0.05, axis=0)
        native_upper = np.quantile(samples, 0.95, axis=0)

        is_rate = step_minutes > 1.0 and self.task_type == TaskType.EVENT_RATE
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

        timestamps = [reference_time + timedelta(minutes=i) for i in range(horizon_minutes)]
        points = self._create_prediction_points(
            timestamps, point_forecast, lower, upper, uncertainties,
        )
        return PredictionTrajectory(
            sync_unit_name=sync_unit_name,
            task_type=self.task_type,
            horizon_minutes=horizon_minutes,
            points=points,
            model_id=f"chronos_{self.model_config.get('model_name', 'tiny').split('/')[-1]}",
        )

    def save(self, path: str) -> None:
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        os.makedirs(path, exist_ok=True)

        np.save(os.path.join(path, "context.npy"), self._context)

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

        model = cls(task_type, model_config, training_config)
        model._context = np.load(os.path.join(path, "context.npy"))
        model._load_pipeline()
        model._is_fitted = True
        return model
