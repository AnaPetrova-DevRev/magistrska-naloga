"""Detect and encode strict periodic (cron-like) patterns in event data.

Identifies time slots (hour, minute) that fire on a high fraction of days,
suggesting an automated/cron process in the external system.  Detected
patterns are encoded as covariates for the RL sync scheduling agent.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CronPattern:
    """A single detected periodic pattern at a fixed (hour, minute) slot."""

    hour: int
    minute: int
    frequency: float  # fraction of days the slot fired (0.0 - 1.0)
    mean_magnitude: float  # average event count when active
    std_magnitude: float  # standard deviation of magnitude

    def slot_key(self) -> Tuple[int, int]:
        return (self.hour, self.minute)


@dataclass
class HourlyCronPattern:
    """A detected periodic pattern at a fixed minute-of-hour.

    Unlike :class:`CronPattern` (daily, fixed hour+minute), this captures
    patterns that repeat every hour — e.g. events consistently arriving at
    :01 and :02 past each hour due to hourly polling/sync.
    """

    minute: int  # minute-of-hour (0-59)
    frequency: float  # fraction of active hours that had events at this minute
    mean_magnitude: float
    std_magnitude: float


# Detection


def detect_cron_patterns(
    timestamps: Sequence[datetime],
    values: Sequence[float],
    min_frequency: float = 0.8,
    min_occurrences: int = 5,
    max_cv: Optional[float] = None,
) -> List[CronPattern]:
    """Detect strict periodic patterns from event-rate time series.

    Groups data by ``(hour, minute)`` slot and checks what fraction of
    distinct days each slot had non-zero events.  Slots exceeding
    *min_frequency* are flagged as cron patterns.

    Args:
        timestamps: Minute-level datetime timestamps.
        values: Event counts (or rates) aligned with *timestamps*.
        min_frequency: Minimum fraction of days a slot must fire to
            qualify (0.0 - 1.0).
        min_occurrences: Minimum absolute number of active days required.
            Guards against false positives on short datasets.
        max_cv: Optional maximum coefficient of variation
            (``std / mean``) for the slot's magnitude.  Slots with
            higher variability are rejected.  ``None`` disables this
            filter.

    Returns:
        Sorted list of :class:`CronPattern` objects, ordered by
        ``(hour, minute)``.
    """
    if len(timestamps) < 2:
        return []

    # Build per-slot, per-date aggregation
    # slot_date_values: {(hour, minute): {date: [values]}}
    slot_date_values: Dict[Tuple[int, int], Dict[Any, List[float]]] = {}
    all_dates: Set = set()

    for ts, val in zip(timestamps, values):
        date_key = ts.date() if hasattr(ts, "date") else ts
        all_dates.add(date_key)
        slot = (ts.hour, ts.minute)
        slot_date_values.setdefault(slot, {}).setdefault(date_key, []).append(val)

    total_days = len(all_dates)
    if total_days < 2:
        logger.info("Only %d day(s) of data — skipping cron detection", total_days)
        return []

    patterns: List[CronPattern] = []

    for (hour, minute), date_vals in slot_date_values.items():
        # Count days where the slot had non-zero events
        active_days = sum(
            1 for vals in date_vals.values() if any(v > 0 for v in vals)
        )

        frequency = active_days / total_days
        if frequency < min_frequency:
            continue
        if active_days < min_occurrences:
            continue

        # Compute magnitude stats from active days only
        day_magnitudes = []
        for vals in date_vals.values():
            day_total = sum(v for v in vals if v > 0)
            if day_total > 0:
                day_magnitudes.append(day_total)

        if not day_magnitudes:
            continue

        mean_mag = sum(day_magnitudes) / len(day_magnitudes)
        if len(day_magnitudes) > 1:
            variance = sum((m - mean_mag) ** 2 for m in day_magnitudes) / (
                len(day_magnitudes) - 1
            )
            std_mag = variance**0.5
        else:
            std_mag = 0.0

        # Optional CV filter
        if max_cv is not None and mean_mag > 0:
            cv = std_mag / mean_mag
            if cv > max_cv:
                continue

        patterns.append(
            CronPattern(
                hour=hour,
                minute=minute,
                frequency=frequency,
                mean_magnitude=mean_mag,
                std_magnitude=std_mag,
            )
        )

    patterns.sort(key=lambda p: (p.hour, p.minute))

    if patterns:
        logger.info(
            "Detected %d cron pattern(s) at min_frequency=%.2f: %s",
            len(patterns),
            min_frequency,
            [(p.hour, p.minute, f"{p.frequency:.0%}") for p in patterns],
        )
    else:
        logger.info(
            "No cron patterns detected (min_frequency=%.2f, %d days of data)",
            min_frequency,
            total_days,
        )

    return patterns


def detect_hourly_patterns(
    timestamps: Sequence[datetime],
    values: Sequence[float],
    min_frequency: float = 0.5,
    min_occurrences: int = 10,
    max_cv: Optional[float] = None,
    min_relative_frequency: Optional[float] = 3.0,
) -> List[HourlyCronPattern]:
    """Detect minute-of-hour slots that fire in many active hours.

    Groups data by minute-of-hour and checks what fraction of *active*
    hours (hours that had any event) also had an event at that minute.
    Minutes significantly overrepresented compared to the baseline
    indicate hourly polling / sync patterns.

    The detector supports two complementary filtering modes:

    1. **Absolute frequency** (``min_frequency``): the minute must fire
       in at least this fraction of all active hours.
    2. **Relative frequency** (``min_relative_frequency``): the minute's
       frequency must be at least N times the *median* frequency across
       all 60 minutes.  This is especially useful for sparse data where
       absolute frequencies are low but cron minutes are clearly
       overrepresented.

    When *both* are set, a minute must pass **either** threshold (OR
    logic) to be considered, then still must pass ``min_occurrences``
    and ``max_cv``.

    Args:
        timestamps: Event timestamps.
        values: Event counts aligned with *timestamps*.
        min_frequency: Minimum absolute fraction of active hours
            (0.0 - 1.0).
        min_occurrences: Minimum absolute number of active-hour hits.
        max_cv: Optional maximum coefficient of variation for magnitude
            consistency.  ``None`` disables the filter.
        min_relative_frequency: Minimum ratio of a minute's frequency
            to the median frequency.  E.g. 3.0 means "must be 3x the
            typical minute."  ``None`` disables this filter.

    Returns:
        Sorted list of :class:`HourlyCronPattern` objects, ordered by
        ``minute``.
    """
    if len(timestamps) < 2:
        return []

    # Build (date, hour) active-hour sets per minute-of-hour
    all_active_hours: Set[Tuple[Any, int]] = set()
    minute_active_hours: Dict[int, Set[Tuple[Any, int]]] = {}
    minute_magnitudes: Dict[int, Dict[Tuple[Any, int], List[float]]] = {}

    for ts, val in zip(timestamps, values):
        if val <= 0:
            continue
        date_key = ts.date() if hasattr(ts, "date") else ts
        hour_key = (date_key, ts.hour)
        all_active_hours.add(hour_key)

        minute_active_hours.setdefault(ts.minute, set()).add(hour_key)
        minute_magnitudes.setdefault(ts.minute, {}).setdefault(
            hour_key, []
        ).append(val)

    total_active = len(all_active_hours)
    if total_active < 2:
        logger.info(
            "Only %d active hour(s) — skipping hourly pattern detection",
            total_active,
        )
        return []

    # Compute per-minute frequencies
    minute_freqs: List[float] = []
    for m in range(60):
        cnt = len(minute_active_hours.get(m, set()))
        minute_freqs.append(cnt / total_active)

    # Median frequency (for relative threshold)
    sorted_freqs = sorted(minute_freqs)
    median_freq = sorted_freqs[len(sorted_freqs) // 2]

    patterns: List[HourlyCronPattern] = []

    for minute in range(60):
        active_set = minute_active_hours.get(minute, set())
        active_count = len(active_set)
        frequency = minute_freqs[minute]

        # Must pass EITHER absolute or relative threshold
        passes_absolute = frequency >= min_frequency
        passes_relative = (
            min_relative_frequency is not None
            and median_freq > 0
            and frequency >= median_freq * min_relative_frequency
        )
        if not passes_absolute and not passes_relative:
            continue
        if active_count < min_occurrences:
            continue

        # Magnitude stats: one value per active (date, hour)
        mag_dict = minute_magnitudes.get(minute, {})
        hour_mags = [sum(v for v in vals if v > 0) for vals in mag_dict.values()]
        hour_mags = [m for m in hour_mags if m > 0]

        if not hour_mags:
            continue

        mean_mag = sum(hour_mags) / len(hour_mags)
        if len(hour_mags) > 1:
            variance = sum((m - mean_mag) ** 2 for m in hour_mags) / (
                len(hour_mags) - 1
            )
            std_mag = variance ** 0.5
        else:
            std_mag = 0.0

        if max_cv is not None and mean_mag > 0:
            cv = std_mag / mean_mag
            if cv > max_cv:
                continue

        patterns.append(
            HourlyCronPattern(
                minute=minute,
                frequency=frequency,
                mean_magnitude=mean_mag,
                std_magnitude=std_mag,
            )
        )

    patterns.sort(key=lambda p: p.minute)

    if patterns:
        logger.info(
            "Detected %d hourly pattern(s): %s",
            len(patterns),
            [
                (f":{p.minute:02d}", f"{p.frequency:.1%}",
                 f"{p.frequency / median_freq:.1f}x" if median_freq > 0 else "")
                for p in patterns
            ],
        )
    else:
        logger.info(
            "No hourly patterns detected (min_frequency=%.2f, "
            "min_relative=%.1fx, %d active hours, median_freq=%.3f)",
            min_frequency,
            min_relative_frequency or 0,
            total_active,
            median_freq,
        )

    return patterns


# Covariate encoding

# Covariate names produced by this module — daily cron
CRON_COV_IS_WINDOW = "is_cron_window"
CRON_COV_MINUTES_TO_NEXT = "minutes_to_next_cron"
CRON_COV_EXPECTED_MAG = "cron_expected_magnitude"
CRON_COVARIATE_NAMES = frozenset({
    CRON_COV_IS_WINDOW,
    CRON_COV_MINUTES_TO_NEXT,
    CRON_COV_EXPECTED_MAG,
})

# Covariate names — hourly cron
HOURLY_COV_IS_WINDOW = "is_hourly_cron_window"
HOURLY_COV_MINUTES_TO_NEXT = "minutes_to_next_hourly_cron"
HOURLY_COV_EXPECTED_MAG = "hourly_cron_expected_magnitude"
HOURLY_CRON_COVARIATE_NAMES = frozenset({
    HOURLY_COV_IS_WINDOW,
    HOURLY_COV_MINUTES_TO_NEXT,
    HOURLY_COV_EXPECTED_MAG,
})

# Union of all cron-derived covariate names
ALL_CRON_COVARIATE_NAMES = CRON_COVARIATE_NAMES | HOURLY_CRON_COVARIATE_NAMES

_MINUTES_IN_DAY = 1440
_MINUTES_IN_HOUR = 60


def _slot_to_minutes(hour: int, minute: int) -> int:
    """Convert (hour, minute) to minutes since midnight."""
    return hour * 60 + minute


def build_cron_covariates(
    timestamps: Sequence[datetime],
    patterns: List[CronPattern],
) -> Dict[str, List[float]]:
    """Build cron-derived covariates for each timestamp.

    Args:
        timestamps: The timestamps to compute covariates for.
        patterns: Detected cron patterns (may be empty).

    Returns:
        Dict with keys ``is_cron_window``, ``minutes_to_next_cron``,
        ``cron_expected_magnitude``, each mapping to a list of floats
        aligned with *timestamps*.
    """
    n = len(timestamps)

    # Pre-compute lookup structures
    slot_set: Set[Tuple[int, int]] = set()
    slot_magnitude: Dict[Tuple[int, int], float] = {}
    slot_minutes: List[int] = []
    max_magnitude = 0.0

    for p in patterns:
        key = p.slot_key()
        slot_set.add(key)
        slot_magnitude[key] = p.mean_magnitude
        slot_minutes.append(_slot_to_minutes(p.hour, p.minute))
        if p.mean_magnitude > max_magnitude:
            max_magnitude = p.mean_magnitude

    slot_minutes.sort()

    is_window: List[float] = []
    mins_to_next: List[float] = []
    expected_mag: List[float] = []

    for ts in timestamps:
        key = (ts.hour, ts.minute)

        # is_cron_window
        if key in slot_set:
            is_window.append(1.0)
        else:
            is_window.append(0.0)

        # minutes_to_next_cron (normalized to [0, 1])
        if not slot_minutes:
            mins_to_next.append(1.0)  # no patterns => max distance
        else:
            current_min = _slot_to_minutes(ts.hour, ts.minute)
            dist = _minutes_to_next_slot(current_min, slot_minutes)
            mins_to_next.append(dist / _MINUTES_IN_DAY)

        # cron_expected_magnitude (normalized by max magnitude)
        if key in slot_magnitude and max_magnitude > 0:
            expected_mag.append(slot_magnitude[key] / max_magnitude)
        else:
            expected_mag.append(0.0)

    return {
        CRON_COV_IS_WINDOW: is_window,
        CRON_COV_MINUTES_TO_NEXT: mins_to_next,
        CRON_COV_EXPECTED_MAG: expected_mag,
    }


def _minutes_to_next_slot(
    current_minute: int, sorted_slot_minutes: List[int]
) -> int:
    """Compute minutes until the next cron slot, wrapping at midnight.

    Args:
        current_minute: Minutes since midnight for the current timestamp.
        sorted_slot_minutes: Sorted list of cron slot minutes-since-midnight.

    Returns:
        Non-negative integer of minutes until the next cron slot.
        Returns 0 if *current_minute* is exactly on a cron slot.
    """
    # Binary search for the first slot >= current_minute
    lo, hi = 0, len(sorted_slot_minutes)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_slot_minutes[mid] < current_minute:
            lo = mid + 1
        else:
            hi = mid

    if lo < len(sorted_slot_minutes):
        return sorted_slot_minutes[lo] - current_minute

    # Wrap: next slot is tomorrow's first slot
    return (_MINUTES_IN_DAY - current_minute) + sorted_slot_minutes[0]


def build_hourly_cron_covariates(
    timestamps: Sequence[datetime],
    patterns: List[HourlyCronPattern],
) -> Dict[str, List[float]]:
    """Build hourly-cron-derived covariates for each timestamp.

    Args:
        timestamps: The timestamps to compute covariates for.
        patterns: Detected hourly patterns (may be empty).

    Returns:
        Dict with keys ``is_hourly_cron_window``,
        ``minutes_to_next_hourly_cron``,
        ``hourly_cron_expected_magnitude``, each mapping to a list of
        floats aligned with *timestamps*.
    """
    # Pre-compute lookup structures
    minute_set: Set[int] = set()
    minute_magnitude: Dict[int, float] = {}
    sorted_minutes: List[int] = []
    max_magnitude = 0.0

    for p in patterns:
        minute_set.add(p.minute)
        minute_magnitude[p.minute] = p.mean_magnitude
        sorted_minutes.append(p.minute)
        max_magnitude = max(max_magnitude, p.mean_magnitude)

    sorted_minutes.sort()

    is_window: List[float] = []
    mins_to_next: List[float] = []
    expected_mag: List[float] = []

    for ts in timestamps:
        m = ts.minute

        # is_hourly_cron_window
        is_window.append(1.0 if m in minute_set else 0.0)

        # minutes_to_next_hourly_cron (normalized to [0, 1] within the hour)
        if not sorted_minutes:
            mins_to_next.append(1.0)
        else:
            dist = _minutes_to_next_in_hour(m, sorted_minutes)
            mins_to_next.append(dist / _MINUTES_IN_HOUR)

        # hourly_cron_expected_magnitude (normalized by max)
        if m in minute_magnitude and max_magnitude > 0:
            expected_mag.append(minute_magnitude[m] / max_magnitude)
        else:
            expected_mag.append(0.0)

    return {
        HOURLY_COV_IS_WINDOW: is_window,
        HOURLY_COV_MINUTES_TO_NEXT: mins_to_next,
        HOURLY_COV_EXPECTED_MAG: expected_mag,
    }


def _minutes_to_next_in_hour(
    current_minute: int, sorted_slot_minutes: List[int]
) -> int:
    """Compute minutes until the next hourly cron slot, wrapping at 60.

    Same binary-search logic as :func:`_minutes_to_next_slot` but wraps
    at 60 minutes instead of 1440.
    """
    lo, hi = 0, len(sorted_slot_minutes)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_slot_minutes[mid] < current_minute:
            lo = mid + 1
        else:
            hi = mid

    if lo < len(sorted_slot_minutes):
        return sorted_slot_minutes[lo] - current_minute

    # Wrap: next slot is in the next hour
    return (_MINUTES_IN_HOUR - current_minute) + sorted_slot_minutes[0]


# Serialization


def save_cron_patterns(
    patterns: List[CronPattern],
    path: str,
    detection_config: Optional[Dict[str, Any]] = None,
    hourly_patterns: Optional[List[HourlyCronPattern]] = None,
) -> None:
    """Save detected patterns to a JSON file.

    Args:
        patterns: Daily cron patterns to persist.
        path: Filesystem path for the JSON file.
        detection_config: Optional dict of detection parameters
            (``min_frequency``, ``min_occurrences``, ``max_cv``) for
            reproducibility tracking.
        hourly_patterns: Optional list of hourly cron patterns.
    """
    data: Dict[str, Any] = {
        "patterns": [asdict(p) for p in patterns],
        "detection_config": detection_config or {},
    }
    if hourly_patterns is not None:
        data["hourly_patterns"] = [asdict(p) for p in hourly_patterns]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_cron_patterns(path: str) -> List[CronPattern]:
    """Load daily patterns from a JSON file.

    Args:
        path: Filesystem path to the JSON file.

    Returns:
        List of :class:`CronPattern` objects.
    """
    with open(path) as f:
        data = json.load(f)

    return [
        CronPattern(
            hour=p["hour"],
            minute=p["minute"],
            frequency=p["frequency"],
            mean_magnitude=p["mean_magnitude"],
            std_magnitude=p["std_magnitude"],
        )
        for p in data.get("patterns", [])
    ]


def load_hourly_cron_patterns(path: str) -> List[HourlyCronPattern]:
    """Load hourly patterns from a JSON file.

    Args:
        path: Filesystem path to the JSON file.

    Returns:
        List of :class:`HourlyCronPattern` objects.  Empty list if
        the file contains no ``hourly_patterns`` key (backward compat).
    """
    with open(path) as f:
        data = json.load(f)

    return [
        HourlyCronPattern(
            minute=p["minute"],
            frequency=p["frequency"],
            mean_magnitude=p["mean_magnitude"],
            std_magnitude=p["std_magnitude"],
        )
        for p in data.get("hourly_patterns", [])
    ]
