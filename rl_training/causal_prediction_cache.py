"""Generate timestamp-keyed rolling-origin forecast caches for RL."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from modeling.core_types import TaskType


PROTOCOL = "causal_timestamp_v1"
TTE_CENSORED_VALUE = 1440.0


def _naive_utc(values: Sequence[datetime]) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(values))
    if index.tz is not None:
        index = index.tz_convert(UTC).tz_localize(None)
    return index


def save_causal_cache(
    cache_path: str,
    timestamps: Sequence[datetime],
    predictions: np.ndarray,
    metadata: Dict[str, Any],
) -> None:
    """Write a new NPZ cache and JSON provenance without overwriting."""
    metadata_path = os.path.splitext(cache_path)[0] + ".json"
    if os.path.exists(cache_path) or os.path.exists(metadata_path):
        raise FileExistsError(f"Refusing to overwrite causal cache {cache_path}")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    timestamp_values = _naive_utc(timestamps).to_numpy(dtype="datetime64[us]")
    values = np.asarray(predictions, dtype=np.float32)
    np.savez_compressed(cache_path, timestamps=timestamp_values, predictions=values)
    digest = hashlib.sha256()
    digest.update(timestamp_values.tobytes())
    digest.update(values.tobytes())
    provenance = {
        **metadata,
        "protocol": PROTOCOL,
        "created_at": datetime.now(UTC).isoformat(),
        "row_count": len(timestamp_values),
        "horizon_steps": int(values.shape[1]),
        "content_sha256": digest.hexdigest(),
        "history_rule": "strictly_before_origin",
        "first_target_rule": "decision_timestamp",
    }
    with open(metadata_path, "x", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, default=str)


class CausalContextBuilder:
    """Build fixed-size model contexts using information available before origin."""

    def __init__(
        self,
        task_type: TaskType,
        source_timestamps: Sequence[datetime],
        source_values: Sequence[float] | None,
        freq: str,
        input_size: int,
    ):
        self.task_type = task_type
        self.source_timestamps = _naive_utc(source_timestamps)
        self.source_values = (
            np.asarray(source_values, dtype=np.float64)
            if source_values is not None else None
        )
        self.freq = freq
        self.input_size = input_size
        offset = pd.tseries.frequencies.to_offset(freq)
        self.step_minutes = int(pd.Timedelta(offset).total_seconds() // 60)
        if self.step_minutes < 1:
            raise ValueError(f"Cache generation requires minute-based frequency: {freq}")
        self._source_start = self.source_timestamps[0]
        if task_type == TaskType.EVENT_RATE:
            if self.source_values is None:
                raise ValueError("Event-rate contexts require source values")
            self._cumsum = np.concatenate([[0.0], np.cumsum(self.source_values)])
        else:
            self._event_ns = self.source_timestamps.asi8

    def build(self, origins: Sequence[datetime]) -> tuple[pd.DatetimeIndex, np.ndarray, Dict[str, np.ndarray]]:
        """Return flattened history dates, target matrix, and causal covariates."""
        origin_index = _naive_utc(origins)
        last_history = (origin_index - pd.Timedelta(minutes=1)).floor(self.freq)
        offsets = np.arange(self.input_size - 1, -1, -1) * self.step_minutes
        history_ns = last_history.asi8[:, None] - offsets[None, :] * 60_000_000_000
        history = pd.DatetimeIndex(history_ns.ravel())
        if self.task_type == TaskType.EVENT_RATE:
            targets = self._event_rate_targets(history_ns, origin_index)
            covariates: Dict[str, np.ndarray] = {}
        else:
            targets, elapsed = self._tte_targets(history_ns, origin_index)
            covariates = {"elapsed_since_last_event": elapsed}
        return history, targets, covariates

    def _event_rate_targets(
        self, history_ns: np.ndarray, origins: pd.DatetimeIndex,
    ) -> np.ndarray:
        minute_ns = 60_000_000_000
        source_start_ns = self._source_start.value
        starts = np.floor_divide(history_ns - source_start_ns, minute_ns)
        origin_ends = np.floor_divide(origins.asi8 - source_start_ns, minute_ns)
        ends = np.minimum(starts + self.step_minutes, origin_ends[:, None])
        valid = (ends > 0) & (starts < len(self.source_values))
        clipped_starts = np.clip(starts, 0, len(self.source_values))
        clipped_ends = np.clip(ends, 0, len(self.source_values))
        values = self._cumsum[clipped_ends] - self._cumsum[clipped_starts]
        return np.where(valid, values, 0.0)

    def _tte_targets(
        self, history_ns: np.ndarray, origins: pd.DatetimeIndex,
    ) -> tuple[np.ndarray, np.ndarray]:
        next_indices = np.searchsorted(self._event_ns, history_ns, side="right")
        visible_counts = np.searchsorted(self._event_ns, origins.asi8, side="left")
        next_visible = next_indices < visible_counts[:, None]
        safe_next = np.clip(next_indices, 0, len(self._event_ns) - 1)
        deltas = (self._event_ns[safe_next] - history_ns) / 60_000_000_000
        targets = np.where(next_visible, np.rint(deltas), TTE_CENSORED_VALUE)
        previous_indices = next_indices - 1
        has_previous = previous_indices >= 0
        safe_previous = np.clip(previous_indices, 0, len(self._event_ns) - 1)
        elapsed = (history_ns - self._event_ns[safe_previous]) / 60_000_000_000
        return targets, np.where(has_previous, np.rint(elapsed), 0.0)


class CausalForecastGenerator:
    """Run fitted RNN or NeuralForecast models over causal context batches."""

    def __init__(self, model: Any, context_builder: CausalContextBuilder, horizon: int):
        self.model = model
        self.context_builder = context_builder
        self.horizon = horizon

    def generate(
        self, origins: Sequence[datetime], batch_size: int = 2048,
    ) -> np.ndarray:
        """Generate one horizon row per origin."""
        result = np.empty((len(origins), self.horizon), dtype=np.float32)
        for start in range(0, len(origins), batch_size):
            stop = min(start + batch_size, len(origins))
            batch = origins[start:stop]
            if hasattr(self.model, "_nf") and self.model._nf is not None:
                result[start:stop] = self._predict_neuralforecast(batch)
            else:
                result[start:stop] = self._predict_rnn(batch)
        return result

    @staticmethod
    def _temporal_covariate(name: str, dates: pd.DatetimeIndex) -> np.ndarray:
        if name == "hour_of_day":
            return dates.hour.to_numpy(dtype=float)
        if name == "day_of_week":
            return dates.dayofweek.to_numpy(dtype=float)
        if name == "is_weekend":
            return (dates.dayofweek >= 5).astype(float)
        if name == "minute_of_hour":
            return dates.minute.to_numpy(dtype=float)
        raise ValueError(f"Unsupported causal temporal covariate: {name}")

    def _predict_neuralforecast(self, origins: Sequence[datetime]) -> np.ndarray:
        history, targets, causal_covs = self.context_builder.build(origins)
        n_origins, input_size = targets.shape
        origin_ids = np.array([f"{i:08d}" for i in range(n_origins)])
        ids = np.repeat(origin_ids, input_size)
        context_df = pd.DataFrame({
            "unique_id": ids,
            "ds": history,
            "y": targets.ravel(),
        })
        covariate_names = list(getattr(self.model, "_covariate_names", []))
        for name in covariate_names:
            if name in causal_covs:
                context_df[name] = causal_covs[name].ravel()
            else:
                context_df[name] = self._temporal_covariate(name, history)

        native_horizon = int(self.model.model_config.get("horizon", self.horizon))
        last_history = history.to_numpy().reshape(n_origins, input_size)[:, -1]
        steps = np.arange(1, native_horizon + 1) * self.context_builder.step_minutes
        future_ns = last_history.astype("datetime64[ns]").astype(np.int64)[:, None]
        future_ns = future_ns + steps[None, :] * 60_000_000_000
        future_dates = pd.DatetimeIndex(future_ns.ravel())
        future_df = pd.DataFrame({
            "unique_id": np.repeat(origin_ids, native_horizon),
            "ds": future_dates,
        })
        for name in covariate_names:
            if name == "elapsed_since_last_event":
                last_elapsed = causal_covs[name][:, -1]
                values = last_elapsed[:, None] + steps[None, :]
                future_df[name] = values.ravel()
            else:
                future_df[name] = self._temporal_covariate(name, future_dates)

        predicted = self.model._nf.predict(df=context_df, futr_df=future_df)
        model_column = next(c for c in predicted.columns if c not in {"unique_id", "ds"})
        native = predicted[model_column].to_numpy().reshape(n_origins, native_horizon)
        rows = [
            self.model._interpolate_to_minutes(
                row,
                self.context_builder.step_minutes,
                self.horizon,
                is_cumulative_rate=(
                    self.context_builder.task_type == TaskType.EVENT_RATE
                    and self.context_builder.step_minutes > 1
                ),
            )
            for row in native
        ]
        return np.maximum(np.asarray(rows), 0.0)

    def _predict_rnn(self, origins: Sequence[datetime]) -> np.ndarray:
        import torch

        _, targets, _ = self.context_builder.build(origins)
        normalized = (targets - self.model._target_mean) / self.model._target_std
        context = torch.tensor(normalized, dtype=torch.float32).unsqueeze(-1)
        outputs = []
        self.model._model.eval()
        with torch.no_grad():
            for _ in range(self.horizon):
                prediction = self.model._model(context[:, -self.context_builder.input_size:, :])
                outputs.append(prediction.cpu().numpy())
                context = torch.cat([context, prediction[:, None, None]], dim=1)
        values = np.column_stack(outputs) * self.model._target_std + self.model._target_mean
        return np.maximum(values, 0.0)