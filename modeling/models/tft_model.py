"""Temporal Fusion Transformer (TFT) for time-series forecasting.

Google's TFT architecture combines LSTM encoding with multi-head attention
and variable selection networks.  It naturally leverages covariates and
produces multi-horizon probabilistic forecasts.

Supports GPU acceleration via MPS (Apple Silicon) or CUDA, with automatic
fallback to CPU when no accelerator is available.

Requires: ``pip install neuralforecast``
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


def _select_accelerator() -> tuple[str, int]:
    """Pick the best available accelerator for NeuralForecast/PyTorch Lightning.

    Returns (accelerator_string, devices) suitable for passing to
    NeuralForecast model constructors.

    Set env FORCE_CPU_TRAINING=1 to bypass MPS (avoids MPS memory leak).
    """
    import os
    import torch

    if os.environ.get("FORCE_CPU_TRAINING"):
        return "cpu", 1
    if torch.backends.mps.is_available():
        return "mps", 1
    if torch.cuda.is_available():
        return "gpu", 1
    return "cpu", 1


class TFTForecastModel(BaseModel):
    """Temporal Fusion Transformer for time-series forecasting.

    Supports both EVENT_RATE and TIME_TO_EVENT tasks.

    Configuration options (model_config):
        horizon: int - prediction horizon (default 24).
        input_size: int - lookback window (default 48).
        hidden_size: int - hidden layer size (default 64).
        n_head: int - number of attention heads (default 4).
        dropout: float (default 0.1).
        use_covariates: bool (default false).
        accelerator: str - force accelerator ("mps", "gpu", "cpu"). Auto-detected if omitted.

    Training configuration (training_config):
        max_steps: int - max training steps (default 200).
        learning_rate: float (default 0.001).
        batch_size: int (default 32).
        early_stop_patience: int - patience for early stopping (default 10).
    """

    MODEL_TYPE: str = "TFT"
    SUPPORTED_TASK_TYPES: set = {TaskType.EVENT_RATE, TaskType.TIME_TO_EVENT}

    @staticmethod
    def _allowlist_torch_globals():
        """Allowlist neuralforecast loss classes for PyTorch 2.6+ weights_only loading."""
        import torch
        from neuralforecast.losses.pytorch import MAE, MSE, MQLoss, SMAPE, MAPE
        torch.serialization.add_safe_globals([MAE, MSE, MQLoss, SMAPE, MAPE])

    def __init__(
        self,
        task_type: TaskType,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
    ):
        super().__init__(task_type, model_config, training_config)
        self._nf = None  # NeuralForecast object
        self._freq: str = "min"  # pandas freq string
        self._train_target: Optional[np.ndarray] = None
        self._train_timestamps: Optional[list] = None
        self._covariate_names: List[str] = []

    def _to_nf_dataframe(self, target, timestamps, covariates=None):
        """Convert arrays to neuralforecast DataFrame format."""
        import pandas as pd

        ds = pd.to_datetime(timestamps)
        if ds.tz is not None:
            ds = ds.tz_localize(None)
        df = pd.DataFrame({
            "unique_id": ["series"] * len(target),
            "ds": ds,
            "y": target.astype(float),
        })
        if covariates:
            for name, values in sorted(covariates.items()):
                arr = np.array(values)
                df[name] = arr[:len(target)].astype(float)
        return df

    def _infer_freq(self, timestamps):
        """Infer pandas frequency string from timestamps."""
        if len(timestamps) < 2:
            return "min"
        delta = timestamps[1] - timestamps[0]
        seconds = delta.total_seconds()
        if seconds <= 60:
            return "min"
        elif seconds <= 3600:
            return f"{int(seconds // 60)}min"
        elif seconds <= 86400:
            return f"{int(seconds // 3600)}h"
        return "D"

    def _get_covariate_names(self, dataset):
        if not self.model_config.get("use_covariates", False):
            return []
        if not dataset.covariates:
            return []
        return sorted(dataset.covariates.keys())

    def fit(self, dataset: Dataset) -> Dict[str, Any]:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import TFT

        self._validate_dataset_type(dataset)
        target = self._get_target_array(dataset)
        train_data = target[dataset.train_indices]
        val_data = target[dataset.val_indices] if dataset.val_indices else np.array([])
        train_ts = [dataset.timestamps[i] for i in dataset.train_indices]
        val_ts = [dataset.timestamps[i] for i in dataset.val_indices] if dataset.val_indices else []

        self._freq = self._infer_freq(train_ts)
        cov_names = self._get_covariate_names(dataset)
        self._covariate_names = cov_names

        horizon = self.model_config.get("horizon", 24)
        input_size = self.model_config.get("input_size", 48)
        hidden_size = self.model_config.get("hidden_size", 64)
        n_head = self.model_config.get("n_head", 4)
        dropout = self.model_config.get("dropout", 0.1)
        max_steps = self.training_config.get("max_steps", 200)
        learning_rate = self.training_config.get("learning_rate", 0.001)
        batch_size = self.training_config.get("batch_size", 32)
        early_stop_patience = self.training_config.get("early_stop_patience", 10)

        # Disable early stopping if no validation data
        val_size = len(val_data)
        if val_size == 0:
            early_stop_patience = -1

        # Resolve accelerator: explicit config > auto-detect
        forced_accel = self.model_config.get("accelerator")
        if forced_accel:
            accelerator, devices = forced_accel, 1
        else:
            accelerator, devices = _select_accelerator()
        logger.info("TFT using accelerator: %s (devices=%d)", accelerator, devices)

        tft_kwargs = dict(
            h=horizon,
            input_size=input_size,
            hidden_size=hidden_size,
            n_head=n_head,
            dropout=dropout,
            max_steps=max_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            early_stop_patience_steps=early_stop_patience,
            scaler_type="standard",
            accelerator=accelerator,
            devices=devices,
            logger=False,
        )
        if cov_names:
            tft_kwargs["futr_exog_list"] = cov_names
            tft_kwargs["hist_exog_list"] = cov_names

        tft_model = TFT(**tft_kwargs)

        # Combine train + val data; neuralforecast splits off val_size from end
        combined_data = np.concatenate([train_data, val_data]) if val_size > 0 else train_data
        combined_ts = train_ts + val_ts
        covariates = dataset.covariates if cov_names else None
        combined_indices = list(dataset.train_indices) + (list(dataset.val_indices) if val_size > 0 else [])
        fit_df = self._to_nf_dataframe(combined_data, combined_ts, covariates={
            k: np.array(v)[combined_indices] for k, v in covariates.items()
        } if covariates else None)

        self._nf = NeuralForecast(models=[tft_model], freq=self._freq)
        self._nf.fit(df=fit_df, val_size=val_size)

        self._train_target = train_data.copy()
        self._train_timestamps = list(train_ts)
        self._is_fitted = True

        fit_info = {
            "horizon": horizon,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "n_head": n_head,
            "max_steps": max_steps,
            "n_train": len(train_data),
            "n_val": val_size,
            "n_covariates": len(cov_names),
            "covariate_names": cov_names,
            "accelerator": accelerator,
        }
        return self._run_validation(dataset, fit_info)

    def predict_pointwise(
        self, dataset: Dataset, indices: Sequence[int],
    ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        self._validate_dataset_type(dataset)

        import pandas as pd

        target = self._get_target_array(dataset)
        cov_names = self._get_covariate_names(dataset)
        n_needed = len(indices)
        horizon = self.model_config.get("horizon", 24)
        input_size = self.model_config.get("input_size", 48)

        # Pre-compute full covariate arrays
        full_futr_covs = {}
        if cov_names and dataset.covariates:
            for name in cov_names:
                full_futr_covs[name] = np.array(dataset.covariates[name])

        freq_offset = pd.tseries.frequencies.to_offset(self._freq)

        # Ground-truth context prediction: for each chunk, use actual data
        # as context (last input_size values) rather than chaining predictions.
        # This avoids O(n²) DataFrame growth and error accumulation.
        all_preds = []
        first_idx = min(indices)
        remaining = n_needed
        n_emitted = 0

        while remaining > 0:
            chunk_start = first_idx + n_emitted
            # Use ground-truth context: last input_size values before chunk
            ctx_start = max(0, chunk_start - input_size)
            ctx_target = target[ctx_start:chunk_start]
            ctx_ts = dataset.timestamps[ctx_start:chunk_start]

            covariates = None
            if cov_names and dataset.covariates:
                covariates = {
                    k: np.array(v)[ctx_start:chunk_start]
                    for k, v in dataset.covariates.items()
                }

            ctx_df = self._to_nf_dataframe(ctx_target, ctx_ts, covariates)

            # Build futr_df for covariates
            chunk_futr_df = None
            if cov_names:
                chunk_end = chunk_start + horizon
                last_ds = ctx_df["ds"].iloc[-1]
                futr_timestamps = [last_ds + freq_offset * (i + 1)
                                   for i in range(horizon)]
                chunk_futr_df = pd.DataFrame({
                    "unique_id": ["series"] * horizon,
                    "ds": futr_timestamps,
                })
                for name in cov_names:
                    vals = full_futr_covs[name]
                    chunk = vals[chunk_start:chunk_end]
                    if len(chunk) < horizon:
                        chunk = np.pad(chunk, (0, horizon - len(chunk)),
                                       mode='edge')
                    chunk_futr_df[name] = chunk.astype(float)

            pred_df = self._nf.predict(df=ctx_df, futr_df=chunk_futr_df)
            preds = pred_df["TFT"].values
            take = min(len(preds), remaining)
            all_preds.extend(preds[:take])
            remaining -= take
            n_emitted += take

        predictions = np.array(all_preds[:n_needed])
        return np.maximum(predictions, 0.0)

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

        target = self._get_target_array(dataset)
        cov_names = self._get_covariate_names(dataset) or self._covariate_names
        covariates = None
        if cov_names and dataset.covariates:
            covariates = {k: np.array(v) for k, v in dataset.covariates.items()}
        ctx_df = self._to_nf_dataframe(target, dataset.timestamps, covariates)

        # Build futr_df for the forecast horizon using covariate_builder
        import pandas as pd
        futr_df = None
        if cov_names:
            from modeling.covariate_builder import build_covariates as _build_covs, _TEMPORAL_COVARIATES
            horizon = self.model_config.get("horizon", 24)
            last_ds = ctx_df["ds"].iloc[-1]
            freq = pd.tseries.frequencies.to_offset(self._freq)
            futr_timestamps = [last_ds + freq * (i + 1) for i in range(horizon)]
            futr_df = pd.DataFrame({
                "unique_id": ["series"] * horizon,
                "ds": futr_timestamps,
            })
            temporal_names = [n for n in cov_names if n in _TEMPORAL_COVARIATES]
            if temporal_names:
                futr_covs = _build_covs(futr_timestamps, temporal_names)
                for name, vals in futr_covs.items():
                    futr_df[name] = [float(v) for v in vals]
            for name in cov_names:
                if name not in _TEMPORAL_COVARIATES and name not in futr_df.columns:
                    last_val = float(covariates[name][-1]) if covariates and name in covariates else 0.0
                    futr_df[name] = [last_val] * horizon

        pred_df = self._nf.predict(df=ctx_df, futr_df=futr_df)
        native_forecast = pred_df["TFT"].values

        step_minutes = self._freq_to_minutes(self._freq)
        is_rate = step_minutes > 1.0 and self.task_type == TaskType.EVENT_RATE
        point_forecast = self._interpolate_to_minutes(native_forecast, step_minutes, horizon_minutes, is_cumulative_rate=is_rate)
        point_forecast = np.maximum(point_forecast, 0.0)

        std = float(np.std(self._train_target)) if self._train_target is not None else 1.0
        if is_rate:
            std = std / step_minutes
        lower = np.maximum(point_forecast - 1.645 * std, 0.0)
        upper = point_forecast + 1.645 * std
        uncertainties = np.full(horizon_minutes, std)

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
            model_id="tft",
        )

    def save(self, path: str) -> None:
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        os.makedirs(path, exist_ok=True)

        nf_path = os.path.join(path, "nf_model")
        self._nf.save(nf_path)

        np.save(os.path.join(path, "train_target.npy"), self._train_target)

        config = {
            "model_type": self.MODEL_TYPE,
            "task_type": self.task_type.value,
            "model_config": self.model_config,
            "training_config": self.training_config,
            "freq": self._freq,
            "covariate_names": self._covariate_names,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str, task_type: TaskType, model_config: Dict[str, Any]):
        from neuralforecast import NeuralForecast
        cls._allowlist_torch_globals()

        config_path = os.path.join(path, "config.json")
        training_config = {}
        freq = "min"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                saved = json.load(f)
            training_config = saved.get("training_config", {})
            freq = saved.get("freq", "min")

        model = cls(task_type, model_config, training_config)
        model._nf = NeuralForecast.load(os.path.join(path, "nf_model"))
        model._train_target = np.load(os.path.join(path, "train_target.npy"))
        model._freq = freq
        model._covariate_names = saved.get("covariate_names", [])
        # Backward compat: extract from NeuralForecast model if not saved
        if not model._covariate_names and model._nf.models:
            nf_model = model._nf.models[0]
            futr = getattr(nf_model, "futr_exog_list", None)
            if futr:
                model._covariate_names = sorted(futr)
        model._is_fitted = True
        return model
