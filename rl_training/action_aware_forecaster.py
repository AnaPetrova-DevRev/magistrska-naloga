"""Forecast fitted models from synchronization-revealed minute counts only."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from typing import Sequence

import numpy as np
import pandas as pd

from modeling.core_types import EventRateDataset, TaskType, TimeToEventDataset
from modeling.covariate_builder import build_covariates
from modeling.model_factory import load_model_from_artifacts


class ExplicitHistoryForecaster:
    """Adapt a fitted model to complete, caller-owned event history."""

    def __init__(self, model, task_type: TaskType, model_path: str, metadata: dict):
        self.model = model
        self.task_type = task_type
        self.model_path = model_path
        self._model_metadata = metadata
        self.metadata = {
            "model_path": model_path,
            "model_type": metadata.get("model_type"),
            "prediction_task_type": task_type.value,
            "model_created_at": metadata.get("created_at"),
            "model_training_data_end": metadata.get("data_end_time"),
            "model_artifact_sha256": self._artifact_digest(model_path),
        }

    @staticmethod
    def _artifact_digest(model_path: str) -> str:
        digest = hashlib.sha256()
        for root, dirnames, filenames in os.walk(model_path):
            dirnames.sort()
            for filename in sorted(filenames):
                path = os.path.join(root, filename)
                digest.update(os.path.relpath(path, model_path).encode())
                with open(path, "rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def from_model_path(
        cls, model_path: str, task_type: str | TaskType,
    ) -> "ExplicitHistoryForecaster":
        """Load an explicit fitted-model artifact."""
        task = TaskType(task_type)
        metadata_path = os.path.join(model_path, "training_metadata.json")
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"Missing model metadata: {metadata_path}")
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("task_type") != task.value:
            raise ValueError("model task type does not match requested prediction task")
        model = load_model_from_artifacts(
            metadata.get("model_type"), task, model_path,
            metadata.get("model_config", {}),
        )
        return cls(model, task, model_path, metadata)

    def should_regenerate(self, revealed_values: Sequence[float]) -> bool:
        """Return whether newly revealed events complete target labels."""
        # An empty interval only adds right-censoring for TTE. An unmasked TTE
        # model cannot consume that fact, so retaining its trajectory is exact.
        return self.task_type == TaskType.EVENT_RATE or any(
            float(value) > 0 for value in revealed_values
        )

    def forecast(
        self,
        observed_timestamps: Sequence[datetime],
        observed_values: Sequence[float],
        synthetic_timestamps: Sequence[datetime],
        synthetic_values: Sequence[float],
        first_target: datetime,
        horizon: int,
    ) -> list[float]:
        """Forecast from revealed counts and prior model-generated values."""
        if self.task_type == TaskType.EVENT_RATE:
            dataset, natural_start = self._event_rate_context(
                observed_timestamps, observed_values,
                synthetic_timestamps, synthetic_values, first_target,
            )
        else:
            dataset, natural_start = self._tte_context(
                observed_timestamps, observed_values,
            )
        bridge = int((first_target - natural_start).total_seconds() // 60)
        if bridge < 0:
            raise ValueError("model context extends beyond requested target")
        trajectory = self.model.forecast(
            dataset,
            horizon_minutes=bridge + horizon,
            reference_time=natural_start,
        )
        values = [point.expected_events for point in trajectory.points]
        if len(values) < bridge + horizon:
            raise ValueError("model returned an incomplete recursive trajectory")
        return values[bridge:bridge + horizon]

    def _event_rate_context(
        self, observed_timestamps, observed_values,
        synthetic_timestamps, synthetic_values, first_target,
    ):
        timestamps = list(observed_timestamps) + list(synthetic_timestamps)
        values = list(observed_values) + list(synthetic_values)
        if not timestamps:
            raise ValueError("event-rate forecasting requires bootstrap history")
        series = pd.Series(values, index=pd.DatetimeIndex(timestamps), dtype=float)
        freq = getattr(self.model, "_freq", None) or "1min"
        offset = pd.tseries.frequencies.to_offset(freq)
        native = series.resample(freq).sum()
        complete = native.index + offset <= pd.Timestamp(first_target)
        native = native.loc[complete]
        input_size = int(self.model.model_config.get("input_size", 30))
        if len(native) < input_size:
            raise ValueError("insufficient complete event-rate bins for model context")
        context_size = max(input_size * 10, 1000)
        native = native.iloc[-context_size:]
        timestamps = native.index.to_pydatetime().tolist()
        cov_names = list(getattr(self.model, "_covariate_names", []))
        covariates = build_covariates(timestamps, cov_names) if cov_names else {}
        dataset = EventRateDataset(timestamps, native.to_numpy(), covariates)
        return dataset, timestamps[-1] + offset

    def _tte_context(self, observed_timestamps, observed_values):
        events = pd.DatetimeIndex([
            timestamp for timestamp, value in zip(observed_timestamps, observed_values)
            if float(value) > 0
        ]).sort_values()
        if len(events) < 2:
            raise ValueError("TTE forecasting requires at least two revealed events")
        window = int(self.model.model_config.get("input_window", 30))
        last_event = events[-1]
        grid_start = max(events[0], last_event - pd.Timedelta(minutes=max(window * 10, 1000)))
        grid = pd.date_range(grid_start, last_event - pd.Timedelta(minutes=1), freq="1min")
        event_ns, grid_ns = events.asi8, grid.asi8
        next_idx = np.searchsorted(event_ns, grid_ns, side="right")
        targets = (event_ns[next_idx] - grid_ns) / 60_000_000_000
        previous = np.maximum(next_idx - 1, 0)
        elapsed = (grid_ns - event_ns[previous]) / 60_000_000_000
        if len(targets) < window:
            raise ValueError("insufficient completed TTE labels for model context")
        dataset = TimeToEventDataset(
            [timestamp.to_pydatetime() for timestamp in grid], targets,
            {"elapsed_since_last_event": elapsed},
        )
        return dataset, last_event.to_pydatetime()
