"""RNN/LSTM/GRU model for time-series forecasting.

A single model class that supports RNN, LSTM, and GRU cell types for both
EVENT_RATE and TIME_TO_EVENT tasks.  Uses sliding windows over the target
series (and optionally covariates) to predict the next value.
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
from modeling.validation import ValidationResult, check_data_length_for_windows

logger = logging.getLogger(__name__)


class RNNForecastModel(BaseModel):
    """RNN/LSTM/GRU model for time-series forecasting.

    Configuration options (model_config):
        cell_type: "RNN", "LSTM", or "GRU" (default "LSTM").
        hidden_size: int (default 64).
        num_layers: int (default 2).
        input_window: int - lookback window size (default 30).
        dropout: float (default 0.1).
        use_covariates: bool (default false).

    Training configuration (training_config):
        max_epochs: int (default 100).
        batch_size: int (default 32).
        lr: float (default 0.001).
        early_stop_patience: int (default 10).
    """

    MODEL_TYPE: str = "RNN"
    SUPPORTED_TASK_TYPES: set = {TaskType.EVENT_RATE, TaskType.TIME_TO_EVENT}

    def __init__(
        self,
        task_type: TaskType,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
    ):
        super().__init__(task_type, model_config, training_config)
        self._model = None  # torch nn.Module
        self._target_mean: Optional[float] = None
        self._target_std: Optional[float] = None
        self._n_features: int = 1

    def _pre_fit_validate(self, dataset):
        result = super()._pre_fit_validate(dataset)
        window_size = self.model_config.get("input_window", 30)
        n_train = len(dataset.train_indices)
        result.add(check_data_length_for_windows(n_train, window_size))
        return result

    def _build_model(self):
        import torch
        import torch.nn as nn

        cell_type = self.model_config.get("cell_type", "LSTM")
        hidden_size = self.model_config.get("hidden_size", 64)
        num_layers = self.model_config.get("num_layers", 2)
        dropout = self.model_config.get("dropout", 0.1)

        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}
        if cell_type not in rnn_cls:
            raise ValueError(f"Unknown cell_type: {cell_type}. Use RNN, LSTM, or GRU.")

        class _RNNCore(nn.Module):
            def __init__(self_, n_features, hidden, n_layers, drop, rnn_class):
                super().__init__()
                self_.rnn = rnn_class(
                    input_size=n_features,
                    hidden_size=hidden,
                    num_layers=n_layers,
                    batch_first=True,
                    dropout=drop if n_layers > 1 else 0.0,
                )
                self_.fc = nn.Linear(hidden, 1)
                self_.drop = nn.Dropout(drop)

            def forward(self_, x):
                # x: (batch, seq_len, n_features)
                out, _ = self_.rnn(x)
                out = self_.drop(out[:, -1, :])  # last timestep
                return self_.fc(out).squeeze(-1)

        model = _RNNCore(
            self._n_features, hidden_size, num_layers, dropout, rnn_cls[cell_type],
        )
        return model

    def _prepare_windows(self, target, covariates_arrays, indices, window_size):
        """Create sliding windows of (X, y) pairs from contiguous indices."""
        import torch

        X_list, y_list = [], []
        for idx in indices:
            start = idx - window_size
            if start < 0:
                continue
            window_target = target[start:idx]
            if covariates_arrays is not None:
                window_covs = covariates_arrays[start:idx]  # (window, n_cov)
                window = np.column_stack([window_target, window_covs])
            else:
                window = window_target.reshape(-1, 1)
            X_list.append(window)
            y_list.append(target[idx])

        if not X_list:
            raise ValueError(
                f"No valid windows created (window_size={window_size}, "
                f"min index={min(indices) if indices else 'N/A'})"
            )

        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        y = torch.tensor(np.array(y_list), dtype=torch.float32)
        return X, y

    def _get_covariates_array(self, dataset: Dataset) -> Optional[np.ndarray]:
        """Stack covariates into (n_samples, n_covariates) array if requested."""
        if not self.model_config.get("use_covariates", False):
            return None
        if not dataset.covariates:
            return None
        cov_names = sorted(dataset.covariates.keys())
        arrays = [np.array(dataset.covariates[name]) for name in cov_names]
        return np.column_stack(arrays)

    def fit(self, dataset: Dataset) -> Dict[str, Any]:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._validate_dataset_type(dataset)
        target = self._get_target_array(dataset)

        # Normalise target
        self._target_mean = float(np.mean(target[dataset.train_indices]))
        self._target_std = float(np.std(target[dataset.train_indices]))
        if self._target_std < 1e-8:
            self._target_std = 1.0
        target_norm = (target - self._target_mean) / self._target_std

        covariates_arr = self._get_covariates_array(dataset)
        self._n_features = 1 + (covariates_arr.shape[1] if covariates_arr is not None else 0)

        window_size = self.model_config.get("input_window", 30)
        max_epochs = self.training_config.get("max_epochs", 100)
        batch_size = self.training_config.get("batch_size", 32)
        lr = self.training_config.get("lr", 0.001)
        patience = self.training_config.get("early_stop_patience", 10)

        X_train, y_train = self._prepare_windows(
            target_norm, covariates_arr, dataset.train_indices, window_size,
        )

        # Validation data (if available)
        X_val, y_val = None, None
        if dataset.val_indices:
            try:
                X_val, y_val = self._prepare_windows(
                    target_norm, covariates_arr, dataset.val_indices, window_size,
                )
            except ValueError:
                pass  # Not enough data for val windows

        self._model = self._build_model()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state = None

        for epoch in range(max_epochs):
            self._model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = self._model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(X_train)

            # Early stopping on val loss
            if X_val is not None:
                self._model.eval()
                with torch.no_grad():
                    val_pred = self._model(X_val)
                    val_loss = criterion(val_pred, y_val).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        logger.info("Early stopping at epoch %d", epoch + 1)
                        break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._is_fitted = True
        epochs_run = epoch + 1

        fit_info = {
            "epochs_run": epochs_run,
            "final_train_loss": train_loss,
            "best_val_loss": best_val_loss if X_val is not None else None,
            "n_train_windows": len(X_train),
            "n_val_windows": len(X_val) if X_val is not None else 0,
            "cell_type": self.model_config.get("cell_type", "LSTM"),
            "n_features": self._n_features,
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
        target_norm = (target - self._target_mean) / self._target_std
        covariates_arr = self._get_covariates_array(dataset)
        window_size = self.model_config.get("input_window", 30)

        self._model.eval()
        predictions = []
        with torch.no_grad():
            for idx in indices:
                start = idx - window_size
                if start < 0:
                    # Not enough history; use mean
                    predictions.append(self._target_mean)
                    continue

                window_target = target_norm[start:idx]
                if covariates_arr is not None:
                    window = np.column_stack([window_target, covariates_arr[start:idx]])
                else:
                    window = window_target.reshape(-1, 1)

                x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                pred_norm = self._model(x).item()
                pred = pred_norm * self._target_std + self._target_mean
                predictions.append(max(pred, 0.0))

        return np.array(predictions)

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

        target = self._get_target_array(dataset)
        target_norm = (target - self._target_mean) / self._target_std

        window_size = self.model_config.get("input_window", 30)
        n_steps = horizon_minutes

        # Autoregressive forecasting
        context = list(target_norm[-window_size:])
        point_forecasts = []

        self._model.eval()
        with torch.no_grad():
            for _ in range(n_steps):
                window = np.array(context[-window_size:]).reshape(-1, 1)
                x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                pred_norm = self._model(x).item()
                pred = pred_norm * self._target_std + self._target_mean
                point_forecasts.append(max(pred, 0.0))
                context.append(pred_norm)

        point_forecast = np.array(point_forecasts)
        std = self._target_std
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
            model_id=f"rnn_{self.model_config.get('cell_type', 'LSTM')}",
        )

    def save(self, path: str) -> None:
        import torch

        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        os.makedirs(path, exist_ok=True)

        torch.save(self._model.state_dict(), os.path.join(path, "rnn_model.pt"))

        state = {
            "target_mean": self._target_mean,
            "target_std": self._target_std,
            "n_features": self._n_features,
        }
        with open(os.path.join(path, "rnn_state.json"), "w") as f:
            json.dump(state, f, indent=2)

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
        import torch

        config_path = os.path.join(path, "config.json")
        training_config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                saved = json.load(f)
            training_config = saved.get("training_config", {})

        model = cls(task_type=task_type, model_config=model_config, training_config=training_config)

        with open(os.path.join(path, "rnn_state.json"), "r") as f:
            state = json.load(f)

        model._target_mean = state["target_mean"]
        model._target_std = state["target_std"]
        model._n_features = state["n_features"]
        model._model = model._build_model()
        model._model.load_state_dict(
            torch.load(os.path.join(path, "rnn_model.pt"), weights_only=True)
        )
        model._model.eval()
        model._is_fitted = True
        return model
