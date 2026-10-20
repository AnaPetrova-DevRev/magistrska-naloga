"""N-HiTS (Neural Hierarchical Interpolation for Time Series) model.

N-HiTS uses multi-rate signal decomposition with hierarchical interpolation
to produce accurate multi-horizon forecasts.  Each stack operates at a
different temporal resolution.

Supports GPU acceleration via MPS (Apple Silicon) or CUDA, with automatic
fallback to CPU when no accelerator is available.

Requires: ``pip install neuralforecast``
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

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


class NHiTSForecastModel(BaseModel):
    """N-HiTS model for time-series forecasting.

    Supports both EVENT_RATE and TIME_TO_EVENT tasks.

    Configuration options (model_config):
        horizon: int - prediction horizon (default 24).
        input_size: int - lookback window (default 48).
        n_stacks: int - number of stacks (default 3).
        n_blocks: list[int] - blocks per stack (default [1,1,1]).
        mlp_units: list[list[int]] - MLP units per stack
            (default [[512,512],[512,512],[512,512]]).
        n_pool_kernel_size: list[int] - pooling sizes
            (default [16,8,1]).
        dropout: float (default 0.1).
        use_covariates: bool (default false).
        accelerator: str - force accelerator ("mps", "gpu", "cpu"). Auto-detected if omitted.

    Training configuration (training_config):
        max_steps: int (default 200).
        learning_rate: float (default 0.001).
        batch_size: int (default 32).
        early_stop_patience: int (default 10).
    """

    MODEL_TYPE: str = "NHiTS"
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
        self._nf = None
        self._freq: str = "min"
        self._train_target: Optional[np.ndarray] = None
        self._train_timestamps: Optional[list] = None
        self._covariate_names: List[str] = []

    def _to_nf_dataframe(self, target, timestamps, covariates=None):
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

    def _predict_native_fast(
        self,
        target: np.ndarray,
        covariates: Optional[Dict[str, np.ndarray]],
        future_covariates: Dict[str, np.ndarray],
    ) -> Optional[np.ndarray]:
        """Predict one series without constructing a Lightning Trainer.

        NeuralForecast's public ``predict`` path creates a Trainer and
        DataLoader for every call. Action-aware RL performs thousands of
        single-series calls, so use the model's prediction step directly when
        the loaded NeuralForecast internals have the expected safe shape.
        """
        nf = self._nf
        training_dataset = getattr(nf, "dataset", None)
        models = list(getattr(nf, "models", [])) if nf is not None else []
        if (
            training_dataset is None
            or len(models) != 1
            or getattr(training_dataset, "scalers_", None) is not None
            or getattr(training_dataset, "static", None) is not None
        ):
            return None

        model = models[0]
        required = ("predict_step", "get_test_size", "set_test_size", "loss")
        if not all(hasattr(model, name) for name in required):
            return None
        temporal_index = getattr(training_dataset, "temporal_cols", None)
        if temporal_index is None:
            return None
        temporal_cols = temporal_index.tolist()
        if "y" not in temporal_cols or "available_mask" not in temporal_cols:
            return None

        native_horizon = int(getattr(nf, "h", 0))
        input_size = int(getattr(model, "input_size", 0))
        history_size = min(len(target), input_size)
        if native_horizon <= 0 or input_size <= 0 or history_size <= 0:
            return None

        import torch

        device = getattr(model, "device", "cpu")
        temporal = torch.zeros(
            (1, len(temporal_cols), input_size + native_horizon),
            dtype=torch.float32,
            device=device,
        )
        history_start = input_size - history_size
        temporal[0, temporal_cols.index("y"), history_start:input_size] = (
            torch.as_tensor(target[-history_size:], dtype=torch.float32, device=device)
        )

        hist_exog = set(getattr(model, "hist_exog_list", []))
        futr_exog = set(getattr(model, "futr_exog_list", []))
        for name in temporal_cols:
            if name in {"y", "available_mask"}:
                continue
            column = temporal_cols.index(name)
            if covariates is not None and name in covariates:
                values = np.asarray(covariates[name])
                if len(values) < history_size:
                    return None
                temporal[0, column, history_start:input_size] = torch.as_tensor(
                    values[-history_size:], dtype=torch.float32, device=device,
                )
            elif name in hist_exog:
                return None
            if name in future_covariates:
                values = np.asarray(future_covariates[name])
                if len(values) < native_horizon:
                    return None
                temporal[0, column, input_size:] = torch.as_tensor(
                    values[:native_horizon], dtype=torch.float32, device=device,
                )
            elif name in futr_exog:
                return None

        available = temporal_cols.index("available_mask")
        temporal[0, available, history_start:] = 1.0
        batch = {
            "temporal": temporal,
            "temporal_cols": temporal_index,
            "static": None,
            "static_cols": None,
        }

        old_test_size = model.get_test_size()
        old_step_size = getattr(model, "predict_step_size", 1)
        old_decompose = getattr(model, "decompose_forecast", False)
        was_training = model.training
        try:
            model.set_test_size(native_horizon)
            model.predict_step_size = 1
            model.decompose_forecast = False
            model.eval()
            with torch.inference_mode():
                output = model.predict_step(batch, 0)
        finally:
            model.set_test_size(old_test_size)
            model.predict_step_size = old_step_size
            model.decompose_forecast = old_decompose
            model.train(was_training)

        output_count = len(model.loss.output_names)
        values = output.detach().cpu().numpy().reshape(-1, output_count)
        return values[:, 0]

    def fit(self, dataset: Dataset) -> Dict[str, Any]:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS

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
        n_stacks = self.model_config.get("n_stacks", 3)
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
        logger.info("NHiTS using accelerator: %s (devices=%d)", accelerator, devices)

        nhits_kwargs = dict(
            h=horizon,
            input_size=input_size,
            stack_types=n_stacks * ["identity"],
            dropout_prob_theta=dropout,
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
            nhits_kwargs["futr_exog_list"] = cov_names
            nhits_kwargs["hist_exog_list"] = cov_names

        nhits_model = NHITS(**nhits_kwargs)

        # Combine train + val data; neuralforecast splits off val_size from end
        combined_data = np.concatenate([train_data, val_data]) if val_size > 0 else train_data
        combined_ts = train_ts + val_ts
        covariates = dataset.covariates if cov_names else None
        combined_indices = list(dataset.train_indices) + (list(dataset.val_indices) if val_size > 0 else [])
        fit_df = self._to_nf_dataframe(combined_data, combined_ts, covariates={
            k: np.array(v)[combined_indices] for k, v in covariates.items()
        } if covariates else None)

        self._nf = NeuralForecast(models=[nhits_model], freq=self._freq)
        self._nf.fit(df=fit_df, val_size=val_size)

        self._train_target = train_data.copy()
        self._train_timestamps = list(train_ts)
        self._is_fitted = True

        fit_info = {
            "horizon": horizon,
            "input_size": input_size,
            "n_stacks": n_stacks,
            "max_steps": max_steps,
            "n_train": len(train_data),
            "n_val": val_size,
            "n_covariates": len(cov_names),
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
            preds = pred_df["NHITS"].values
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
        # Use stored covariate names when dataset doesn't carry them
        # (e.g. when called from ModelServingApp which builds history
        # dataset with covariates based on what the model needs).
        cov_names = self._get_covariate_names(dataset) or self._covariate_names
        covariates = None
        if cov_names and dataset.covariates:
            covariates = {k: np.array(v) for k, v in dataset.covariates.items()}

        # Synthesize elapsed_since_last_event if the model needs it
        if cov_names and "elapsed_since_last_event" in cov_names:
            if covariates is None:
                covariates = {}
            if "elapsed_since_last_event" not in covariates:
                elapsed = np.zeros(len(target), dtype=np.float64)
                last_event_idx = -1
                for i in range(len(target)):
                    if target[i] > 0:
                        last_event_idx = i
                    elapsed[i] = (i - last_event_idx) if last_event_idx >= 0 else 0
                covariates["elapsed_since_last_event"] = elapsed

        # Limit context to last N rows to avoid pandas argsort bug on
        # large (>400K) tz-aware datetime arrays in neuralforecast internals.
        input_size = self.model_config.get("input_size", 30)
        ctx_len = min(len(target), max(input_size * 10, 1000))
        if len(target) > ctx_len:
            target = target[-ctx_len:]
            timestamps_slice = dataset.timestamps[-ctx_len:]
            if covariates:
                covariates = {k: v[-ctx_len:] for k, v in covariates.items()}
        else:
            timestamps_slice = dataset.timestamps
        # Build future covariates at the model's native horizon.
        import pandas as pd
        horizon = self.model_config.get("horizon", 24)
        last_ds = pd.Timestamp(timestamps_slice[-1])
        freq = pd.tseries.frequencies.to_offset(self._freq)
        futr_timestamps = [last_ds + freq * (i + 1) for i in range(horizon)]
        future_covariates: Dict[str, np.ndarray] = {}
        if cov_names:
            from modeling.covariate_builder import (
                _TEMPORAL_COVARIATES,
                build_covariates as _build_covs,
            )
            temporal_names = [n for n in cov_names if n in _TEMPORAL_COVARIATES]
            if temporal_names:
                futr_covs = _build_covs(futr_timestamps, temporal_names)
                for name, vals in futr_covs.items():
                    future_covariates[name] = np.asarray(vals, dtype=float)
            # Non-temporal covariates: repeat last known value
            for name in cov_names:
                if name not in _TEMPORAL_COVARIATES:
                    last_val = (
                        float(covariates[name][-1])
                        if covariates and name in covariates else 0.0
                    )
                    future_covariates[name] = np.full(horizon, last_val)

        native_forecast = self._predict_native_fast(
            target, covariates, future_covariates,
        )
        if native_forecast is None:
            ctx_df = self._to_nf_dataframe(target, timestamps_slice, covariates)
            futr_df = None
            if cov_names:
                futr_df = pd.DataFrame({
                    "unique_id": ["series"] * horizon,
                    "ds": futr_timestamps,
                    **future_covariates,
                })
            pred_df = self._nf.predict(df=ctx_df, futr_df=futr_df, sort_df=False)
            native_forecast = pred_df["NHITS"].values

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
            model_id="nhits",
        )

    def save(self, path: str) -> None:
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        os.makedirs(path, exist_ok=True)

        self._nf.save(os.path.join(path, "nf_model"))
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
