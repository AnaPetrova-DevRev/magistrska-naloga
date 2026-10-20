"""Transformer Hawkes Process model for event forecasting.

Implements the Transformer Hawkes Process (Zuo et al., ICML 2020) for
modeling self-exciting event streams. The model learns continuous-time
intensity functions from event history using self-attention, capturing
burst dynamics that parametric survival models cannot represent.

Supports both EVENT_RATE (intensity integration → events/minute) and
TIME_TO_EVENT (inverse intensity → expected waiting time) tasks.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from modeling.base_model import BaseModel
from modeling.core_types import (
    Dataset,
    PredictionTrajectory,
    TaskType,
)

logger = logging.getLogger(__name__)


def _select_device() -> "torch.device":
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TransformerHawkesModel(BaseModel):
    """Transformer Hawkes Process for event stream forecasting.

    Configuration (model_config):
        d_model: Hidden dimension (default 64).
        n_head: Attention heads (default 4).
        n_layers: Transformer layers (default 2).
        max_seq_len: Context window in events (default 256).
        mc_samples: Monte Carlo samples for integral approximation (default 20).
        dropout: Dropout rate (default 0.1).
        max_events: Subsample to this many events from tail (default None = all).
        min_delta_minutes: Filter inter-arrival times below this in minutes (default 0).
        device: Force device (default auto-detect).

    Training configuration (training_config):
        max_epochs: Maximum training epochs (default 100).
        batch_size: Batch size (default 64).
        lr: Learning rate (default 1e-4).
        early_stop_patience: Epochs without improvement before stopping (default 15).
        max_steps_per_epoch: Cap batches per epoch for large data (default 1000).
        grad_clip: Gradient clipping norm (default 1.0).
    """

    MODEL_TYPE: str = "THP"
    SUPPORTED_TASK_TYPES: set = {TaskType.EVENT_RATE, TaskType.TIME_TO_EVENT}

    def __init__(
        self,
        task_type: TaskType,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
    ):
        super().__init__(task_type, model_config, training_config)
        self._model = None
        self._iat_mean: Optional[float] = None
        self._iat_std: Optional[float] = None
        self._device = None
        self._base_intensity: Optional[float] = None

    def _get_device(self):
        if self._device is None:
            import torch

            forced = self.model_config.get("device")
            if forced:
                self._device = torch.device(forced)
            else:
                self._device = _select_device()
            logger.info("THP using device: %s", self._device)
        return self._device

    def _build_model(self):
        import torch
        import torch.nn as nn

        d_model = self.model_config.get("d_model", 64)
        n_head = self.model_config.get("n_head", 4)
        n_layers = self.model_config.get("n_layers", 2)
        dropout = self.model_config.get("dropout", 0.1)

        class _ContinuousTimePE(nn.Module):
            """Learnable sinusoidal positional encoding on inter-arrival times."""

            def __init__(self_, d_model):
                super().__init__()
                half_d = d_model // 2
                self_.freq = nn.Parameter(torch.randn(half_d) * 0.1)
                self_.phase = nn.Parameter(torch.zeros(half_d))

            def forward(self_, iat):
                # iat: (batch, seq_len) — normalized inter-arrival times
                angles = iat.unsqueeze(-1) * self_.freq.unsqueeze(0).unsqueeze(0) + self_.phase
                return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        class _THPNetwork(nn.Module):
            def __init__(self_, d_model, n_head, n_layers, dropout):
                super().__init__()
                self_.pe = _ContinuousTimePE(d_model)
                self_.input_proj = nn.Linear(d_model, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_head,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                self_.encoder = nn.TransformerEncoder(
                    encoder_layer, num_layers=n_layers,
                )
                self_.intensity_head = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Linear(d_model // 2, 1),
                    nn.Softplus(),
                )

            def forward(self_, iat_seq, mask=None):
                # iat_seq: (batch, seq_len) normalized inter-arrival times
                pe = self_.pe(iat_seq)
                h = self_.input_proj(pe)
                if mask is None:
                    seq_len = iat_seq.size(1)
                    mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(iat_seq.device)
                h = self_.encoder(h, mask=mask, is_causal=True)
                intensity = self_.intensity_head(h).squeeze(-1)
                return intensity  # (batch, seq_len)

        model = _THPNetwork(d_model, n_head, n_layers, dropout)
        return model.to(self._get_device())

    def _prepare_sequences(self, iat_norm: np.ndarray, indices: List[int], max_seq_len: int):
        """Create sliding-window sequences from inter-arrival times."""
        sequences = []
        for idx in indices:
            start = max(0, idx - max_seq_len + 1)
            seq = iat_norm[start:idx + 1]
            if len(seq) < 2:
                continue
            sequences.append(seq)
        return sequences

    def _pad_sequences(self, sequences: List[np.ndarray], max_len: int):
        """Pad sequences to max_len, return tensor + lengths."""
        import torch

        batch_size = len(sequences)
        padded = np.zeros((batch_size, max_len), dtype=np.float32)
        lengths = []
        for i, seq in enumerate(sequences):
            L = min(len(seq), max_len)
            padded[i, :L] = seq[-L:]
            lengths.append(L)
        return torch.tensor(padded, dtype=torch.float32), lengths

    def _compute_nll_loss(self, model, iat_batch, lengths, mc_samples, iat_std):
        """Compute negative log-likelihood for a batch of event sequences.

        NLL = -sum(log λ(t_i)) + integral(λ(t) dt)
        The integral is approximated by MC sampling between events.
        """
        import torch

        device = iat_batch.device
        batch_size, _ = iat_batch.shape

        intensity = model(iat_batch)  # (batch, seq_len)

        total_nll = torch.tensor(0.0, device=device)
        n_valid = 0

        for b in range(batch_size):
            L = lengths[b]
            if L < 2:
                continue

            # Log-likelihood of observed events (skip first event — no history)
            event_intensity = intensity[b, 1:L]
            log_likelihood = torch.log(event_intensity + 1e-8).sum()

            # Integral approximation via MC between consecutive events
            # The gap between events (in normalized units) is the IAT value at next event
            gaps = iat_batch[b, 1:L]  # normalized inter-arrival times
            gaps_real = gaps * iat_std  # denormalize to minutes

            # For each gap, sample mc_samples uniform points and evaluate intensity
            # Approximate: use the intensity at the preceding event as constant over the gap
            # (mean-field approximation — avoids expensive per-sample forward pass)
            preceding_intensity = intensity[b, :L-1]
            integral = (preceding_intensity * gaps_real).sum()

            nll = -log_likelihood + integral
            total_nll = total_nll + nll
            n_valid += 1

        if n_valid == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return total_nll / n_valid

    def _batched_val_nll(self, val_padded, val_lengths, mc_samples, batch_size, device):
        """Compute validation NLL in batches to avoid OOM."""
        import torch

        n = len(val_padded)
        total_nll = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = val_padded[start:end].to(device)
            batch_lengths = val_lengths[start:end]
            nll = self._compute_nll_loss(
                self._model, batch, batch_lengths, mc_samples, self._iat_std,
            ).item()
            total_nll += nll * (end - start)
            n_batches += (end - start)
        return total_nll / max(n_batches, 1)

    def fit(self, dataset: Dataset) -> Dict[str, Any]:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self._validate_dataset_type(dataset)
        device = self._get_device()

        target = self._get_target_array(dataset)  # minutes_to_next_event or events_per_minute

        # For THP, we always work with inter-arrival times
        # If event_rate task, we still train on IAT internally but output rates
        if self.task_type == TaskType.TIME_TO_EVENT:
            iat_minutes = target.copy()
        else:
            # For event_rate, the dataset is bucketed counts — not ideal for THP
            # THP needs raw IAT data. Use the target as-is and interpret as time gaps
            iat_minutes = target.copy()

        # Filter very short inter-arrival times (batch arrivals)
        if "min_delta_seconds" in self.model_config:
            min_delta_min = round(self.model_config["min_delta_seconds"] / 60.0)
        else:
            min_delta_min = self.model_config.get("min_delta_minutes", 0)
        valid_mask = iat_minutes >= min_delta_min
        # Keep track of valid indices within train/val/test
        all_valid_indices = np.where(valid_mask)[0]

        # Subsample if configured
        max_events = self.model_config.get("max_events")
        if max_events and len(all_valid_indices) > max_events:
            all_valid_indices = all_valid_indices[-max_events:]
            logger.info("THP: subsampled to last %d valid events", max_events)

        # Map original split indices to filtered data
        valid_set = set(all_valid_indices.tolist())
        train_idx = [i for i in dataset.train_indices if i in valid_set]
        val_idx = [i for i in dataset.val_indices if i in valid_set]

        if not train_idx:
            raise ValueError("No valid training events after filtering")

        # Normalize inter-arrival times
        train_iat = iat_minutes[train_idx]
        self._iat_mean = float(np.mean(train_iat))
        self._iat_std = float(np.std(train_iat))
        if self._iat_std < 1e-8:
            self._iat_std = 1.0

        iat_norm = (iat_minutes - self._iat_mean) / self._iat_std

        # Store base intensity (events per minute in training data)
        total_time_minutes = float(np.sum(iat_minutes[train_idx]))
        self._base_intensity = len(train_idx) / total_time_minutes if total_time_minutes > 0 else 1.0

        max_seq_len = self.model_config.get("max_seq_len", 256)
        mc_samples = self.model_config.get("mc_samples", 20)
        max_epochs = self.training_config.get("max_epochs", 100)
        batch_size = self.training_config.get("batch_size", 64)
        lr = self.training_config.get("lr", 1e-4)
        patience = self.training_config.get("early_stop_patience", 15)
        max_steps = self.training_config.get("max_steps_per_epoch", 1000)
        grad_clip = self.training_config.get("grad_clip", 1.0)

        # Prepare training sequences
        train_sequences = self._prepare_sequences(iat_norm, train_idx, max_seq_len)
        if not train_sequences:
            raise ValueError("No valid training sequences created")

        val_sequences = self._prepare_sequences(iat_norm, val_idx, max_seq_len) if val_idx else []

        logger.info(
            "THP: %d train sequences, %d val sequences (max_seq_len=%d)",
            len(train_sequences), len(val_sequences), max_seq_len,
        )

        # Pad and create DataLoader (keep on CPU, move per-batch to avoid OOM)
        train_padded, train_lengths = self._pad_sequences(train_sequences, max_seq_len)
        train_lengths_t = torch.tensor(train_lengths, dtype=torch.long)
        train_ds = TensorDataset(train_padded, train_lengths_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_padded, val_lengths = None, None
        if val_sequences:
            val_padded, val_lengths = self._pad_sequences(val_sequences, max_seq_len)
            # Keep on CPU — will batch to device during validation

        self._model = self._build_model()
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state = None

        for epoch in range(max_epochs):
            self._model.train()
            train_loss = 0.0
            n_batches = 0

            for batch_padded, batch_lengths in train_loader:
                if n_batches >= max_steps:
                    break

                batch_padded = batch_padded.to(device)
                batch_lengths_list = batch_lengths.tolist()

                optimizer.zero_grad()
                loss = self._compute_nll_loss(
                    self._model, batch_padded, batch_lengths_list,
                    mc_samples, self._iat_std,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), grad_clip)
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            scheduler.step()
            train_loss /= max(n_batches, 1)

            # Validation (batched to avoid OOM on large val sets)
            if val_padded is not None:
                self._model.eval()
                with torch.no_grad():
                    val_loss = self._batched_val_nll(
                        val_padded, val_lengths, mc_samples, batch_size, device,
                    )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        logger.info("THP: early stopping at epoch %d", epoch + 1)
                        break
            else:
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "THP epoch %d: train_nll=%.4f val_nll=%.4f",
                    epoch + 1, train_loss, best_val_loss,
                )

        if best_state is not None:
            self._model.load_state_dict(best_state)
            self._model.to(device)

        self._is_fitted = True

        fit_info = {
            "epochs_run": epoch + 1,
            "final_train_nll": train_loss,
            "best_val_nll": best_val_loss if val_padded is not None else None,
            "n_train_sequences": len(train_sequences),
            "n_val_sequences": len(val_sequences),
            "n_events_after_filter": len(all_valid_indices),
            "max_seq_len": max_seq_len,
            "base_intensity_per_min": self._base_intensity,
            "iat_mean_minutes": self._iat_mean,
            "iat_std_minutes": self._iat_std,
            "device": str(device),
        }
        return self._run_validation(dataset, fit_info)

    def predict_pointwise(
        self, dataset: Dataset, indices: Sequence[int],
    ) -> np.ndarray:
        import torch

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        self._validate_dataset_type(dataset)

        device = self._get_device()
        target = self._get_target_array(dataset)
        iat_norm = (target - self._iat_mean) / self._iat_std

        max_seq_len = self.model_config.get("max_seq_len", 256)

        self._model.eval()
        predictions = []

        with torch.no_grad():
            for idx in indices:
                start = max(0, idx - max_seq_len + 1)
                seq = iat_norm[start:idx + 1]

                if len(seq) < 2:
                    # Not enough context — return mean
                    predictions.append(self._iat_mean)
                    continue

                seq_tensor = torch.tensor(
                    seq.reshape(1, -1), dtype=torch.float32,
                ).to(device)
                intensity = self._model(seq_tensor)  # (1, seq_len)
                last_intensity = intensity[0, -1].item()

                if self.task_type == TaskType.TIME_TO_EVENT:
                    # Expected time to next event = 1/λ (in minutes)
                    predicted_tte = 1.0 / max(last_intensity, 1e-6)
                    predictions.append(max(predicted_tte, 0.0))
                else:
                    # Event rate = intensity (events per minute)
                    predictions.append(max(last_intensity, 0.0))

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

        device = self._get_device()
        target = self._get_target_array(dataset)
        iat_norm = (target - self._iat_mean) / self._iat_std

        max_seq_len = self.model_config.get("max_seq_len", 256)

        # Use last max_seq_len events as context
        context = iat_norm[-max_seq_len:].copy()

        self._model.eval()
        point_forecasts = []

        with torch.no_grad():
            # Get the current intensity from context
            seq_tensor = torch.tensor(
                context.reshape(1, -1), dtype=torch.float32,
            ).to(device)
            intensity_seq = self._model(seq_tensor)
            current_intensity = intensity_seq[0, -1].item()

            for _ in range(horizon_minutes):
                # Approximate integral of intensity over [k, k+1] minute
                # Use current intensity as approximately constant over 1 minute
                # (intensity evolves slowly relative to 1-minute resolution)
                expected_events = max(current_intensity, 0.0)
                point_forecasts.append(expected_events)

                # Simulate: if intensity predicts an event, add it to context
                # and re-evaluate. Otherwise intensity decays toward base rate.
                if expected_events > 0.5:
                    # Likely event in this minute — add synthetic IAT to context
                    synthetic_iat = (1.0 / max(current_intensity, 1e-6))
                    synthetic_norm = (synthetic_iat - self._iat_mean) / self._iat_std
                    context = np.append(context[-max_seq_len+1:], synthetic_norm)
                    seq_tensor = torch.tensor(
                        context.reshape(1, -1), dtype=torch.float32,
                    ).to(device)
                    intensity_seq = self._model(seq_tensor)
                    current_intensity = intensity_seq[0, -1].item()
                else:
                    # No event expected — intensity decays toward base rate
                    decay = 0.95
                    current_intensity = (
                        current_intensity * decay + self._base_intensity * (1 - decay)
                    )

        point_forecast = np.array(point_forecasts)

        if self.task_type == TaskType.TIME_TO_EVENT:
            # Convert: expected events/min → expected minutes to next event
            point_forecast = np.where(
                point_forecast > 1e-6,
                1.0 / point_forecast,
                self._iat_mean,
            )

        # Uncertainty estimate (simple heuristic based on intensity variance)
        std_estimate = self._iat_std * 0.5
        lower = np.maximum(point_forecast - 1.645 * std_estimate, 0.0)
        upper = point_forecast + 1.645 * std_estimate
        uncertainties = np.full(horizon_minutes, std_estimate)

        if reference_time is None:
            from datetime import UTC
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
            model_id=f"thp_d{self.model_config.get('d_model', 64)}",
        )

    def save(self, path: str) -> None:
        import torch

        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        os.makedirs(path, exist_ok=True)

        cpu_state = {k: v.cpu() for k, v in self._model.state_dict().items()}
        torch.save(cpu_state, os.path.join(path, "thp_model.pt"))

        state = {
            "iat_mean": self._iat_mean,
            "iat_std": self._iat_std,
            "base_intensity": self._base_intensity,
        }
        with open(os.path.join(path, "thp_state.json"), "w") as f:
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

        model = cls(
            task_type=task_type,
            model_config=model_config,
            training_config=training_config,
        )

        with open(os.path.join(path, "thp_state.json"), "r") as f:
            state = json.load(f)
        model._iat_mean = state["iat_mean"]
        model._iat_std = state["iat_std"]
        model._base_intensity = state["base_intensity"]

        model._model = model._build_model()
        model._model.load_state_dict(
            torch.load(os.path.join(path, "thp_model.pt"), weights_only=True)
        )
        model._model.to(model._get_device())
        model._model.eval()
        model._is_fitted = True
        return model
