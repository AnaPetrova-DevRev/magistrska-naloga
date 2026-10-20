"""Window-aware data-characterization helpers for observable-stream analysis."""

# pylint: disable=too-many-locals

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from sync_unit import SyncUnit
from .fingerprint import DatasetFingerprint, compute_fingerprint_from_artifacts


DAY_NAMES = [
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
]


@dataclass(frozen=True)
class AnalysisWindow:
    """UTC-inclusive time window used to derive separate analysis artifacts."""

    window_id: str
    start: Optional[str] = None
    end: Optional[str] = None

    def to_dict(self) -> dict[str, Optional[str]]:
        """Return JSON-serializable requested bounds."""
        return {"window_id": self.window_id, "start": self.start, "end": self.end}


STANDARD_WINDOWS = (
    AnalysisWindow("full"),
    AnalysisWindow("post_2021", "2021-01-01T00:00:00Z"),
    AnalysisWindow("post_2022", "2022-01-01T00:00:00Z"),
    AnalysisWindow("post_2023", "2023-01-01T00:00:00Z"),
)

MODELED_STARTS = {
    "novacore_zendesk_novacore_zendesk": "2022-01-01T00:00:00Z",
}


def windows_for_unit(sync_unit_name: str) -> list[AnalysisWindow]:
    """Return standard scenarios plus the unit's final modeled window."""
    windows = list(STANDARD_WINDOWS)
    windows.append(
        AnalysisWindow("modeled", MODELED_STARTS.get(sync_unit_name))
    )
    return windows


def _utc_bound(value: str) -> pd.Timestamp:
    """Parse a naive or aware bound as UTC."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def filter_timestamp_window(
    frame: pd.DataFrame,
    timestamp_col: str,
    window: AnalysisWindow,
) -> pd.DataFrame:
    """Normalize timestamps to UTC and apply inclusive window bounds."""
    if timestamp_col not in frame.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")
    result = frame.copy()
    result[timestamp_col] = pd.to_datetime(result[timestamp_col], utc=True)
    result = result.dropna(subset=[timestamp_col])
    if window.start:
        result = result[result[timestamp_col] >= _utc_bound(window.start)]
    if window.end:
        result = result[result[timestamp_col] <= _utc_bound(window.end)]
    return result


def build_windowed_time_deltas(epm_df: pd.DataFrame) -> pd.DataFrame:
    """Recompute active-minute IATs without a cross-boundary predecessor."""
    if epm_df.empty:
        raise ValueError("Cannot compute deltas for an empty window")
    timestamps = pd.to_datetime(epm_df["minute"], utc=True).dt.floor("min")
    timestamps = timestamps.drop_duplicates().reset_index(drop=True)
    if not timestamps.is_monotonic_increasing:
        order = np.argsort(timestamps.astype("int64").to_numpy(), kind="stable")
        timestamps = timestamps.iloc[order].reset_index(drop=True)
    deltas = (timestamps.diff().dt.total_seconds() / 60.0).round().astype("Int64")
    return pd.DataFrame({"events": timestamps, "events_delta": deltas})


def build_merged_event_profile(epm_df: pd.DataFrame) -> pd.DataFrame:
    """Build event-weighted DOW/HOD profile fields from the merged stream."""
    required = {"minute", "events_per_minute"}
    missing = required.difference(epm_df.columns)
    if missing:
        raise ValueError(f"Missing events-per-minute columns: {sorted(missing)}")
    timestamps = pd.to_datetime(epm_df["minute"], utc=True)
    weights = epm_df["events_per_minute"].astype(float)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Merged event profile requires positive event magnitude")

    dow = weights.groupby(timestamps.dt.dayofweek).sum().reindex(
        range(7), fill_value=0.0
    ) / total
    hod = weights.groupby(timestamps.dt.hour).sum().reindex(
        range(24), fill_value=0.0
    ) / total
    profile = {
        f"pct_changes_{name}": [float(dow.iloc[index] * 100.0)]
        for index, name in enumerate(DAY_NAMES)
    }
    profile.update(
        {f"num_changes_hour_{hour:02d}": [float(hod.iloc[hour])]
         for hour in range(24)}
    )
    return pd.DataFrame(profile)


def build_windowed_record_analysis(
    clean_df: pd.DataFrame,
    window: AnalysisWindow,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build compact lifecycle and profile frames from distinct exported states."""
    required = {"id", "created_date", "modified_date"}
    missing = required.difference(clean_df.columns)
    if missing:
        raise ValueError(f"Missing clean-data columns: {sorted(missing)}")

    states = filter_timestamp_window(
        clean_df.loc[:, sorted(required)], "modified_date", window
    )
    states["created_date"] = pd.to_datetime(states["created_date"], utc=True)
    states = states.dropna(subset=["id", "created_date"])
    states = states.drop_duplicates(subset=["id", "modified_date"])
    if states.empty:
        raise ValueError("No exported record states remain in the requested window")

    grouped = states.groupby("id", sort=False)
    first_state = grouped["modified_date"].min()
    last_state = grouped["modified_date"].max()
    first_created = grouped["created_date"].min()
    state_counts = grouped.size()

    later = states[states["modified_date"] != states["created_date"]]
    first_later = later.groupby("id")["modified_date"].min().reindex(first_state.index)
    effective_end = states["modified_date"].max()

    lifecycle = pd.DataFrame(index=first_state.index)
    lifecycle["first_created_date"] = first_created
    lifecycle["first_modified_date"] = first_later
    lifecycle["last_modified_date"] = last_state
    lifecycle["num_changes_total"] = state_counts.astype(int)
    lifecycle["time_first_created_to_last_modified"] = (
        last_state - first_created
    ).dt.total_seconds()
    lifecycle["observed_activity_span_seconds"] = (
        last_state - first_state
    ).dt.total_seconds()
    lifecycle["time_to_first_modification"] = (
        first_later - first_created
    ).dt.total_seconds()
    lifecycle["was_created_3mo_before_cutoff"] = (
        first_created <= effective_end - pd.Timedelta(days=90)
    )
    lifecycle = lifecycle.reset_index()

    per_record_total = state_counts.astype(float)
    dow_counts = states.groupby(["id", states["modified_date"].dt.dayofweek]).size()
    record_totals = per_record_total.reindex(
        dow_counts.index.get_level_values(0)
    ).to_numpy()
    dow_shares = dow_counts.to_numpy() / record_totals
    dow_profile = pd.Series(
        dow_shares, index=dow_counts.index
    ).groupby(level=1).sum().reindex(range(7), fill_value=0.0)
    dow_profile = dow_profile / len(state_counts)

    hod_profile = states["modified_date"].dt.hour.value_counts().reindex(
        range(24), fill_value=0
    ).sort_index().astype(float)
    hod_profile = hod_profile / hod_profile.sum()

    profile = {
        f"pct_changes_{name}": [float(dow_profile.iloc[index] * 100.0)]
        for index, name in enumerate(DAY_NAMES)
    }
    profile.update(
        {f"num_changes_hour_{hour:02d}": [float(hod_profile.iloc[hour])]
         for hour in range(24)}
    )
    return lifecycle, pd.DataFrame(profile)


def summarize_lifecycle(apr_df: pd.DataFrame) -> dict[str, float]:
    """Summarize observed record-state counts with claim-safe metric names."""
    counts = apr_df["num_changes_total"].astype(int)
    descending = np.sort(counts.to_numpy())[::-1]
    top_n = max(int(np.ceil(len(descending) * 0.10)), 1)
    total = int(counts.sum())
    spans = apr_df["observed_activity_span_seconds"].astype(float)
    return {
        "records_with_state": int(len(counts)),
        "observed_states": total,
        "median_states_per_record": float(counts.median()),
        "pct_one_state_records": float((counts == 1).mean()),
        "pct_records_gt_10_states": float((counts > 10).mean()),
        "max_states_single_record": int(counts.max()),
        "top_10pct_state_share": float(descending[:top_n].sum() / total),
        "median_observed_activity_span_seconds": float(spans.median()),
    }


def compute_windowed_fingerprint(
    sync_unit: SyncUnit,
    clean_df: pd.DataFrame,
    epm_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    window: AnalysisWindow,
) -> tuple[DatasetFingerprint, dict[str, float], pd.DataFrame]:
    """Filter source artifacts and compute one complete windowed fingerprint."""
    window_epm = filter_timestamp_window(epm_df, "minute", window)
    window_merged = filter_timestamp_window(merged_df, "events", window)
    if window_epm.empty or window_merged.empty:
        raise ValueError("No stream events remain in the requested window")
    lifecycle, _ = build_windowed_record_analysis(clean_df, window)
    profile = build_merged_event_profile(window_epm)
    deltas = build_windowed_time_deltas(window_epm)
    fingerprint = compute_fingerprint_from_artifacts(
        sync_unit,
        lifecycle,
        window_epm,
        deltas,
        window_merged,
        profile_apr_df=profile,
    )
    return fingerprint, summarize_lifecycle(lifecycle), lifecycle
