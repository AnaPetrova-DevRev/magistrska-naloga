"""Extended metrics for model evaluation.

This module provides additional metrics beyond the base MAE metric defined
in core_types.py. All metrics follow the same interface for consistency.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

from modeling.core_types import Metric


class RmseMetric(Metric):
    """Root Mean Squared Error (RMSE)."""

    name: str = "rmse"

    def __call__(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float],
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        if not y_true:
            return float("nan")
        total = 0.0
        for t, p in zip(y_true, y_pred):
            total += (t - p) ** 2
        return math.sqrt(total / len(y_true))


class MapeMetric(Metric):
    """Mean Absolute Percentage Error (MAPE).

    Note: Returns NaN if any true value is zero to avoid division by zero.
    For sparse data with many zeros, consider using SMAPE or MAE instead.
    """

    name: str = "mape"

    def __call__(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float],
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        if not y_true:
            return float("nan")
        total = 0.0
        for t, p in zip(y_true, y_pred):
            if t == 0:
                return float("nan")
            total += abs((t - p) / t)
        return (total / len(y_true)) * 100.0


class SmapeMetric(Metric):
    """Symmetric Mean Absolute Percentage Error (SMAPE).

    More robust than MAPE for data with zeros or near-zero values.
    Range is 0-200%.
    """

    name: str = "smape"

    def __call__(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float],
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        if not y_true:
            return float("nan")
        total = 0.0
        count = 0
        for t, p in zip(y_true, y_pred):
            denom = abs(t) + abs(p)
            if denom == 0:
                continue
            total += abs(t - p) / denom
            count += 1
        if count == 0:
            return 0.0
        return (total / count) * 200.0


class QuantileLossMetric(Metric):
    """Quantile Loss (Pinball Loss) for a specific quantile.

    Useful for evaluating probabilistic forecasts. Lower quantiles (e.g., 0.1)
    penalize over-prediction more; higher quantiles (e.g., 0.9) penalize
    under-prediction more.
    """

    def __init__(self, quantile: float = 0.5):
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be between 0 and 1 (exclusive)")
        self.quantile = quantile
        self.name = f"quantile_loss_{quantile}"

    def __call__(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float],
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        if not y_true:
            return float("nan")
        total = 0.0
        for t, p in zip(y_true, y_pred):
            error = t - p
            if error >= 0:
                total += self.quantile * error
            else:
                total += (self.quantile - 1) * error
        return total / len(y_true)


class AoIRewardMetric(Metric):
    """Age of Information (AoI) aware reward metric.

    A proxy metric that approximates the RL agent's sync-scheduling reward.
    It simulates a simple policy — "sync when the model says it's urgent" —
    and scores the outcome against what actually happened.

    Supports two modes determined by ``mode``:

    **event_rate** (default):
        y_true / y_pred are events-per-minute.
        Sync decision: predicted events > ``sync_threshold``.
        Urgent reality: actual events > 0.

    **time_to_event**:
        y_true / y_pred are inter-arrival times (minutes).
        Sync decision: predicted TTE < ``sync_threshold`` (event imminent).
        Urgent reality: actual TTE < ``aoi_threshold`` (data would go
        stale beyond the accepted age of information).

    Reward table (per data point):

    +-----------+--------+----------------------------------------------+
    | Decision  | Reality| Reward                                       |
    +-----------+--------+----------------------------------------------+
    | Sync      | Urgent | -sync_cost + sync_cost * 0.5 (good sync)     |
    | Sync      | Calm   | -sync_cost             (wasted bandwidth)    |
    | Don't sync| Urgent | -late_penalty          (stale data)          |
    | Don't sync| Calm   |  0                     (correct wait)        |
    +-----------+--------+----------------------------------------------+

    The final score is the mean reward across all points (higher = better,
    typically negative).
    """

    def __init__(
        self,
        aoi_threshold_minutes: float = 5.0,
        sync_cost: float = 1.0,
        late_penalty: float = 2.0,
        sync_threshold: float = 0.5,
        mode: str = "event_rate",
        metric_name: str = "aoi_reward",
        resample_freq_minutes: Optional[float] = None,
    ):
        if mode not in ("event_rate", "time_to_event"):
            raise ValueError(
                f"AoIRewardMetric mode must be 'event_rate' or "
                f"'time_to_event', got '{mode}'"
            )
        self.aoi_threshold = aoi_threshold_minutes
        self.sync_cost = sync_cost
        self.late_penalty = late_penalty
        self.sync_threshold = sync_threshold
        self.mode = mode
        self.name = metric_name
        self.resample_freq_minutes = resample_freq_minutes

    def _classify_event_rate(
        self, y_true: float, y_pred: float
    ) -> tuple:
        """Return (should_sync, is_urgent) for event-rate data.

        When ``resample_freq_minutes`` is set, both true and predicted values
        are normalized to per-minute rates before comparison so that the
        sync_threshold has the same meaning regardless of bucket size.
        """
        pred = y_pred
        true = y_true
        if self.resample_freq_minutes and self.resample_freq_minutes > 1:
            pred = y_pred / self.resample_freq_minutes
            true = y_true / self.resample_freq_minutes
        should_sync = pred > self.sync_threshold
        is_urgent = true > 0
        return should_sync, is_urgent

    def _classify_time_to_event(
        self, y_true: float, y_pred: float
    ) -> tuple:
        """Return (should_sync, is_urgent) for time-to-event data.

        Sync when predicted TTE is short (event imminent).
        Urgent when actual TTE is below the AoI threshold (data would
        become unacceptably stale).
        """
        should_sync = y_pred < self.sync_threshold
        is_urgent = y_true < self.aoi_threshold
        return should_sync, is_urgent

    def __call__(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float],
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute the AoI-based reward.

        Automatically picks event_rate or time_to_event logic based on
        ``self.mode``.  If ``context`` contains a ``task_type`` key that
        differs from the configured mode, the context value takes
        precedence (so a single metric instance works across tasks when
        the mode is left at default).
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        if not y_true:
            return float("nan")

        mode = self.mode
        if context and "task_type" in context:
            ctx_task = context["task_type"]
            if ctx_task in ("event_rate", "time_to_event"):
                mode = ctx_task

        classify = (
            self._classify_event_rate
            if mode == "event_rate"
            else self._classify_time_to_event
        )

        total_reward = 0.0
        for t, p in zip(y_true, y_pred):
            should_sync, is_urgent = classify(t, p)

            if should_sync:
                total_reward -= self.sync_cost
                if is_urgent:
                    total_reward += self.sync_cost * 0.5
            else:
                if is_urgent:
                    total_reward -= self.late_penalty

        return total_reward / len(y_true)


class MseMetric(Metric):
    """Mean Squared Error (MSE)."""

    name: str = "mse"

    def __call__(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float],
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        if not y_true:
            return float("nan")
        total = 0.0
        for t, p in zip(y_true, y_pred):
            total += (t - p) ** 2
        return total / len(y_true)


class RmsleMetric(Metric):
    """Root Mean Squared Logarithmic Error (RMSLE).

    Computes RMSE in log(1+x) space, which treats relative errors equally
    across scales. Ideal for heavy-tailed, non-negative data like
    inter-arrival times where values span several orders of magnitude.

    Both true and predicted values are clipped to >= 0 before the log
    transform (negative predictions become 0).
    """

    name: str = "rmsle"

    def __call__(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float],
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        if not y_true:
            return float("nan")
        total = 0.0
        for t, p in zip(y_true, y_pred):
            log_t = math.log1p(max(t, 0.0))
            log_p = math.log1p(max(p, 0.0))
            total += (log_t - log_p) ** 2
        return math.sqrt(total / len(y_true))


class MedianAeMetric(Metric):
    """Median Absolute Error (MdAE).

    The median of absolute errors — robust to extreme outliers.
    Reports the "typical" prediction error without being distorted
    by heavy tails.
    """

    name: str = "median_ae"

    def __call__(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float],
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        if not y_true:
            return float("nan")
        errors = sorted(abs(t - p) for t, p in zip(y_true, y_pred))
        n = len(errors)
        mid = n // 2
        if n % 2 == 1:
            return errors[mid]
        return (errors[mid - 1] + errors[mid]) / 2.0
