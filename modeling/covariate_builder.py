"""Covariate extraction from timestamps for dataset enrichment.

Provides configurable temporal features that can be added as covariates
to EventRateDataset or TimeToEventDataset.  Feature names are specified
in the dataset config and extracted at dataset-build time.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

from modeling.cron_pattern_detector import ALL_CRON_COVARIATE_NAMES, CRON_COVARIATE_NAMES

_TWO_PI = 2.0 * math.pi

_TEMPORAL_COVARIATES = frozenset({
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "day_of_month",
    "month",
    "quarter",
    "minute_of_hour",
    "is_business_hours",
    "hours_since_midnight",
    "sin_hour",
    "cos_hour",
    "sin_day_of_week",
    "cos_day_of_week",
})

_DATA_DERIVED_COVARIATES = frozenset({
    "elapsed_since_last_event",
})

AVAILABLE_COVARIATES = _TEMPORAL_COVARIATES | ALL_CRON_COVARIATE_NAMES | _DATA_DERIVED_COVARIATES

_DEFAULT_BUSINESS_HOURS: Tuple[int, int] = (9, 17)


def get_available_covariates() -> List[str]:
    """Return sorted list of available covariate names."""
    return sorted(AVAILABLE_COVARIATES)


def validate_covariate_names(names: List[str]) -> None:
    """Raise ValueError if any name is not a recognised covariate."""
    unknown = set(names) - AVAILABLE_COVARIATES
    if unknown:
        raise ValueError(
            f"Unknown covariate(s): {sorted(unknown)}. "
            f"Available: {sorted(AVAILABLE_COVARIATES)}"
        )


def _extract_single(
    ts: datetime,
    name: str,
    business_hours: Tuple[int, int],
) -> float:
    """Extract a single covariate value from a timestamp."""
    if name == "hour_of_day":
        return float(ts.hour)
    if name == "day_of_week":
        return float(ts.weekday())
    if name == "is_weekend":
        return 1.0 if ts.weekday() >= 5 else 0.0
    if name == "day_of_month":
        return float(ts.day)
    if name == "month":
        return float(ts.month)
    if name == "quarter":
        return float((ts.month - 1) // 3 + 1)
    if name == "minute_of_hour":
        return float(ts.minute)
    if name == "is_business_hours":
        bh_start, bh_end = business_hours
        return 1.0 if (bh_start <= ts.hour < bh_end and ts.weekday() < 5) else 0.0
    if name == "hours_since_midnight":
        return ts.hour + ts.minute / 60.0
    if name == "sin_hour":
        return math.sin(_TWO_PI * ts.hour / 24.0)
    if name == "cos_hour":
        return math.cos(_TWO_PI * ts.hour / 24.0)
    if name == "sin_day_of_week":
        return math.sin(_TWO_PI * ts.weekday() / 7.0)
    if name == "cos_day_of_week":
        return math.cos(_TWO_PI * ts.weekday() / 7.0)
    raise ValueError(f"Unknown covariate: {name}")


def build_covariates(
    timestamps: Sequence[datetime],
    covariate_names: List[str],
    business_hours: Tuple[int, int] = _DEFAULT_BUSINESS_HOURS,
) -> Dict[str, List[float]]:
    """Build covariate columns from a sequence of timestamps.

    Args:
        timestamps: The timestamps to extract features from.
        covariate_names: Which covariates to compute.
        business_hours: (start_hour, end_hour) for is_business_hours.

    Returns:
        Dict mapping covariate name to list of float values,
        one per timestamp.
    """
    validate_covariate_names(covariate_names)

    extractable = [n for n in covariate_names if n not in _DATA_DERIVED_COVARIATES]
    result: Dict[str, List[float]] = {name: [] for name in extractable}
    for ts in timestamps:
        for name in extractable:
            result[name].append(_extract_single(ts, name, business_hours))
    return result


def split_covariate_names(
    names: List[str],
) -> Tuple[List[str], List[str]]:
    """Separate regular temporal covariates from cron-derived covariates.

    Args:
        names: Full list of requested covariate names.

    Returns:
        Tuple of (regular_names, cron_names).  Both lists preserve the
        original ordering.

    Raises:
        ValueError: If any name is not in :data:`AVAILABLE_COVARIATES`.
    """
    validate_covariate_names(names)
    regular = [n for n in names if n not in ALL_CRON_COVARIATE_NAMES]
    cron = [n for n in names if n in ALL_CRON_COVARIATE_NAMES]
    return regular, cron
