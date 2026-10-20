"""Metrics registry for building metrics from configuration.

This module provides a factory function to create metric instances from
JSON-serializable definitions, supporting both simple string names and
parameterized metric configurations.
"""

from __future__ import annotations

from typing import Any, Dict, Union

from modeling.core_types import MaeMetric, Metric
from modeling.metrics import (
    AoIRewardMetric,
    MapeMetric,
    MedianAeMetric,
    MseMetric,
    QuantileLossMetric,
    RmseMetric,
    RmsleMetric,
    SmapeMetric,
)


def build_metric(defn: Union[str, Dict[str, Any]]) -> Metric:
    """Build a Metric instance from a definition.

    Args:
        defn: Either a string metric name (e.g., "mae", "rmse") or a dict
              with "name" and optional "params" keys.

    Returns:
        A Metric instance ready to be called with (y_true, y_pred, context).

    Raises:
        ValueError: If the metric name is unknown or params are invalid.

    Examples:
        >>> build_metric("mae")
        <MaeMetric>

        >>> build_metric({"name": "rmse"})
        <RmseMetric>

        >>> build_metric({
        ...     "name": "quantile_loss",
        ...     "params": {"quantile": 0.9}
        ... })
        <QuantileLossMetric quantile=0.9>

        >>> build_metric({
        ...     "name": "aoi_reward",
        ...     "params": {
        ...         "aoi_threshold_minutes": 10,
        ...         "sync_cost": 1.5,
        ...         "late_penalty": 3.0
        ...     }
        ... })
        <AoIRewardMetric>
    """
    if isinstance(defn, str):
        defn = {"name": defn}

    name = defn.get("name")
    if not name:
        raise ValueError("Metric definition must include a 'name' field")

    params = defn.get("params", {})

    name_lower = name.lower()

    if name_lower == "mae":
        return MaeMetric()

    if name_lower == "rmse":
        return RmseMetric()

    if name_lower == "mse":
        return MseMetric()

    if name_lower == "mape":
        return MapeMetric()

    if name_lower == "smape":
        return SmapeMetric()

    if name_lower in ("quantile_loss", "pinball_loss", "quantile"):
        quantile = params.get("quantile", 0.5)
        return QuantileLossMetric(quantile=quantile)

    if name_lower in ("rmsle", "root_mean_squared_log_error"):
        return RmsleMetric()

    if name_lower in ("median_ae", "mdae", "median_absolute_error"):
        return MedianAeMetric()

    if name_lower in ("aoi_reward", "custom_aoi_reward", "aoi"):
        return AoIRewardMetric(
            aoi_threshold_minutes=params.get("aoi_threshold_minutes", 5.0),
            sync_cost=params.get("sync_cost", 1.0),
            late_penalty=params.get("late_penalty", 2.0),
            sync_threshold=params.get("sync_threshold", 0.5),
            mode=params.get("mode", "event_rate"),
            metric_name=params.get("metric_name", name),
            resample_freq_minutes=params.get("resample_freq_minutes"),
        )

    raise ValueError(f"Unknown metric: {name}")


def get_available_metrics() -> Dict[str, str]:
    """Return a dict of available metric names and their descriptions."""
    return {
        "mae": "Mean Absolute Error",
        "rmse": "Root Mean Squared Error",
        "mse": "Mean Squared Error",
        "mape": "Mean Absolute Percentage Error",
        "smape": "Symmetric Mean Absolute Percentage Error",
        "quantile_loss": "Quantile (Pinball) Loss - requires 'quantile' param",
        "rmsle": "Root Mean Squared Logarithmic Error - ideal for heavy-tailed non-negative data",
        "median_ae": "Median Absolute Error - robust to extreme outliers",
        "aoi_reward": "AoI-aware reward metric - supports 'mode' (event_rate|time_to_event), 'sync_threshold', 'aoi_threshold_minutes', 'sync_cost', 'late_penalty', 'resample_freq_minutes' params. Set resample_freq_minutes for cross-resolution ER comparison (normalizes to per-minute rate).",
    }


def normalize_metric_defs(raw_metrics: list) -> list:
    """Normalize a list of metric definitions to dict form.

    Converts string metric names to {"name": name} dicts for consistent
    processing downstream.

    Args:
        raw_metrics: List of strings or dicts from JSON config

    Returns:
        List of dicts, each with at least a "name" key
    """
    return [
        m if isinstance(m, dict) else {"name": m}
        for m in raw_metrics
    ]
