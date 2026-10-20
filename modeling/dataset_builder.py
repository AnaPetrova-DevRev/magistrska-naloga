"""Dataset building utilities for modeling.

Supports building EVENT_RATE and TIME_TO_EVENT datasets for a single sync unit
from precomputed files produced by the data_preprocessing app.

Uses app_flow_definitions to locate the appropriate output files.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import storage_access
import sync_unit
from app_actions_constants import (
    CALCULATE_MERGED_EVENTS_PER_MINUTE,
    CALCULATE_MERGED_TIME_DELTAS,
)
from app_flow_definitions import define_input_output_files
from global_vars import (
    MERGED_EVENTS_PER_MINUTE_FILE_NAME_FORMAT,
    MERGED_TIME_DELTA_FILE_NAME_FORMAT,
)
from modeling.core_types import (
    DataSnapshotMetadata,
    DatasetConfig,
    EventRateDataset,
    TaskType,
    TimeToEventDataset,
)
import logging

from modeling.covariate_builder import build_covariates, split_covariate_names
from modeling.cron_pattern_detector import (
    CRON_COVARIATE_NAMES,
    HOURLY_CRON_COVARIATE_NAMES,
    CronPattern,
    HourlyCronPattern,
    build_cron_covariates,
    build_hourly_cron_covariates,
    detect_cron_patterns,
    detect_hourly_patterns,
)

logger = logging.getLogger(__name__)


SNAPSHOT_FOLDER_NAME = "modeling"


def _list_output_files_for_action(
    sa: storage_access.StorageAccess,
    sync_unit_obj: sync_unit.SyncUnit,
    action: str,
    file_format: str,
) -> List[str]:
    """List all output files for a given action in the sync unit.

    Uses app_flow_definitions to determine the correct output folder,
    then lists files matching the expected pattern.

    Args:
        sa: StorageAccess instance.
        sync_unit_obj: The sync unit.
        action: The action that produces these files (e.g., CALCULATE_MERGED_EVENTS_PER_MINUTE).
        file_format: The file name format (e.g., MERGED_EVENTS_PER_MINUTE_FILE_NAME_FORMAT).

    Returns:
        List of full file paths for existing output files.
    """
    io_files = define_input_output_files(action)
    if isinstance(io_files, ValueError):
        raise io_files

    sync_unit_path = sync_unit_obj.get_path_for_sync_unit()
    output_folder = f"{sync_unit_path}/{io_files.output_file_folder}"

    base_pattern = file_format.replace("{int}", "*")

    files = sa.list_files_in_folder(output_folder, base_pattern)
    return files


def _load_dataframes_from_files(
    sa: storage_access.StorageAccess,
    file_paths: List[str],
) -> pd.DataFrame:
    """Load and concatenate dataframes from multiple pickle files.

    Args:
        sa: StorageAccess instance.
        file_paths: List of paths to pickle files.

    Returns:
        Combined DataFrame from all files.
    """
    if not file_paths:
        raise FileNotFoundError("No files provided to load")

    dfs: List[pd.DataFrame] = []
    for path in file_paths:
        df_chunk = sa.read_picklefile_as_dataframe(path)
        dfs.append(df_chunk)

    return pd.concat(dfs, ignore_index=True)


def _load_events_per_minute_dataframe(
    sa: storage_access.StorageAccess,
    sync_unit_obj: sync_unit.SyncUnit,
) -> pd.DataFrame:
    """Load and aggregate merged events-per-minute data for a sync unit.

    Uses app_flow_definitions to locate the output files from the
    CALCULATE_MERGED_EVENTS_PER_MINUTE action.
    """
    files = _list_output_files_for_action(
        sa,
        sync_unit_obj,
        CALCULATE_MERGED_EVENTS_PER_MINUTE,
        MERGED_EVENTS_PER_MINUTE_FILE_NAME_FORMAT,
    )

    if not files:
        raise FileNotFoundError(
            f"No merged events-per-minute files found for sync unit "
            f"{sync_unit_obj.get_name()}"
        )

    df_all = _load_dataframes_from_files(sa, files)

    if "minute" not in df_all.columns or "events_per_minute" not in df_all.columns:
        raise ValueError(
            "Expected columns 'minute' and 'events_per_minute' in merged events-per-minute data"
        )

    df_all = df_all.copy()
    df_all["minute"] = pd.to_datetime(df_all["minute"], utc=True).dt.tz_localize(None)

    # Aggregate duplicate minutes using int64 groupby to avoid pandas argsort bug
    # on large arrays (pandas 2.3.3 + Python 3.14).  Use numpy argsort instead
    # of df.sort_values because nargsort crashes on large frames.
    df_all["_minute_ns"] = df_all["minute"].astype("datetime64[ns]").astype(np.int64)
    agg = df_all.groupby("_minute_ns", as_index=False)["events_per_minute"].sum()
    agg["minute"] = pd.to_datetime(agg["_minute_ns"])
    df_all = agg[["minute", "events_per_minute", "_minute_ns"]]
    order = np.argsort(df_all["_minute_ns"].values)
    df_all = df_all.iloc[order].reset_index(drop=True)
    df_all = df_all[["minute", "events_per_minute"]]

    return df_all


def _validate_resample_freq(freq: str) -> None:
    """Validate that *freq* is a valid pandas frequency string.

    Raises ``ValueError`` if the string cannot be parsed.
    """
    try:
        pd.tseries.frequencies.to_offset(freq)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid resample_freq '{freq}'. Must be a valid pandas "
            f"frequency string (e.g., '1min', '15min', '1h', '1D'). Error: {e}"
        ) from e


def _resample_event_rate_data(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample events-per-minute data to a regular time grid.

    Sets the ``minute`` column as the index, resamples to the given
    frequency (summing events within each bucket), and zero-fills any
    missing periods.

    Args:
        df: DataFrame with ``minute`` and ``events_per_minute`` columns.
        freq: Pandas frequency string (e.g. ``'1min'``, ``'15min'``, ``'1h'``).

    Returns:
        Resampled DataFrame with ``minute`` and ``events_per_minute`` columns,
        sorted by time, with no gaps.
    """
    df = df.set_index("minute")
    resampled = df.resample(freq)["events_per_minute"].sum().fillna(0.0)
    return resampled.reset_index()


def _resample_time_to_event_data(
    df: pd.DataFrame,
    freq: str = "1min",
) -> pd.DataFrame:
    """Resample irregular TTE data to a regular time grid.

    Converts event-based (irregular) inter-arrival data into a regular
    grid where each row is a fixed-interval time point and the target is
    "integer minutes until the next event from this point."

    Args:
        df: DataFrame with ``timestamp`` column (event occurrence times,
            already floored to minute granularity).
        freq: Pandas frequency string for output grid (default ``'1min'``).

    Returns:
        DataFrame with columns:
            - ``minute``: regular-grid timestamps
            - ``minutes_to_next_event``: integer minutes to next event
            - ``elapsed_since_last_event``: integer minutes since last event
    """
    if df.empty:
        return pd.DataFrame(
            columns=["minute", "minutes_to_next_event", "elapsed_since_last_event"]
        )

    event_times = pd.to_datetime(df["timestamp"]).sort_values().reset_index(drop=True)

    grid_start = event_times.iloc[0].floor(freq)
    grid_end = event_times.iloc[-1].ceil(freq)
    grid = pd.date_range(start=grid_start, end=grid_end, freq=freq)

    event_arr = event_times.values.astype("datetime64[ns]")
    grid_arr = grid.values.astype("datetime64[ns]")

    next_idx = np.searchsorted(event_arr, grid_arr, side="right")

    minutes_to_next = np.full(len(grid), 1440, dtype=np.int64)
    elapsed_since_last = np.zeros(len(grid), dtype=np.int64)

    for i in range(len(grid)):
        if next_idx[i] < len(event_arr):
            delta_ns = (event_arr[next_idx[i]] - grid_arr[i]).astype(np.int64)
            minutes_to_next[i] = int(round(delta_ns / 1e9 / 60.0))
        if next_idx[i] > 0:
            elapsed_ns = (grid_arr[i] - event_arr[next_idx[i] - 1]).astype(np.int64)
            elapsed_since_last[i] = int(round(elapsed_ns / 1e9 / 60.0))

    return pd.DataFrame({
        "minute": grid,
        "minutes_to_next_event": minutes_to_next,
        "elapsed_since_last_event": elapsed_since_last,
    })


def _load_merged_time_deltas_dataframe(
    sa: storage_access.StorageAccess,
    sync_unit_obj: sync_unit.SyncUnit,
) -> pd.DataFrame:
    """Reconstruct inter-event deltas across all chunks of a sync unit.

    Uses app_flow_definitions to locate the output files from the
    CALCULATE_MERGED_TIME_DELTAS action. Stored deltas are chunk-local, so
    recompute them from the sorted union of active-minute timestamps before
    applying dataset filters. Only the first event of the combined trace has
    no preceding interval and is dropped, preserving single-chunk behavior.
    """
    files = _list_output_files_for_action(
        sa,
        sync_unit_obj,
        CALCULATE_MERGED_TIME_DELTAS,
        MERGED_TIME_DELTA_FILE_NAME_FORMAT,
    )

    if not files:
        raise FileNotFoundError(
            f"No merged time delta files found for sync unit "
            f"{sync_unit_obj.get_name()}"
        )

    df_all = _load_dataframes_from_files(sa, files)

    # The preprocessing step (calculate_merged_time_delta) produces:
    #   "events"       – the event timestamp (floored to minute)
    #   "events_delta" – inter-event time in integer minutes
    # Verify both are present.
    timestamp_col = "events"
    delta_col = "events_delta"

    if timestamp_col not in df_all.columns:
        raise ValueError(
            f"Expected column '{timestamp_col}' in merged time delta data. "
            f"Available columns: {list(df_all.columns)}"
        )

    if delta_col not in df_all.columns:
        raise ValueError(
            f"Expected column '{delta_col}' in merged time delta data. "
            f"Available columns: {list(df_all.columns)}"
        )

    df_all = df_all[[timestamp_col]].copy()
    df_all["timestamp"] = pd.to_datetime(df_all[timestamp_col], utc=True).dt.tz_localize(None)
    df_all = df_all[["timestamp"]].dropna().drop_duplicates(subset=["timestamp"])
    # Use numpy argsort to avoid pandas nargsort bug on large arrays
    order = np.argsort(df_all["timestamp"].values)
    df_all = df_all.iloc[order].reset_index(drop=True)

    # Keep chunk-first timestamps until after differencing: their predecessor
    # may be in another chunk. Recompute all deltas to handle overlaps too.
    df_all["time_delta"] = (df_all["timestamp"].diff().dt.total_seconds() / 60.0).round()
    return df_all.dropna(subset=["time_delta"]).reset_index(drop=True)


def _compute_splits(
    n: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[int], List[int], List[int]]:
    """Compute train/val/test split indices."""
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    if n_train < 0 or n_val < 0 or n_train + n_val > n:
        raise ValueError(
            f"Invalid train/val ratios: train_ratio={train_ratio}, "
            f"val_ratio={val_ratio}, n={n}"
        )

    train_indices = list(range(0, n_train))
    val_indices = list(range(n_train, n_train + n_val))
    test_indices = list(range(n_train + n_val, n))

    return train_indices, val_indices, test_indices


def compute_time_series_cv_splits(
    n_trainval: int,
    n_folds: int,
) -> List[Tuple[List[int], List[int]]]:
    """Compute expanding-window time-series cross-validation splits.

    Divides the first ``n_trainval`` data points into ``n_folds + 1``
    equal segments.  Fold *i* trains on segments ``0..i`` and validates
    on segment ``i + 1``, producing an expanding training window.

    Args:
        n_trainval: Number of data points available for train+val
                    (test indices should already be excluded).
        n_folds: Number of CV folds.

    Returns:
        List of ``(train_indices, val_indices)`` tuples, one per fold.

    Raises:
        ValueError: If n_folds < 2 or too few data points.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")

    segment_size = n_trainval // (n_folds + 1)
    if segment_size < 1:
        raise ValueError(
            f"Not enough data ({n_trainval} points) for {n_folds} folds"
        )

    folds: List[Tuple[List[int], List[int]]] = []
    for i in range(n_folds):
        train_end = segment_size * (i + 1)
        val_start = train_end
        val_end = segment_size * (i + 2) if i < n_folds - 1 else n_trainval
        folds.append(
            (list(range(0, train_end)), list(range(val_start, val_end)))
        )

    return folds


def _to_python_datetime(ts: object) -> datetime:
    """Convert pandas Timestamp to Python datetime."""
    return ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts


def _to_naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Strip timezone info for comparison with tz-naive DataFrame columns."""
    if dt is None:
        return None
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def _save_snapshot_metadata(
    sa: storage_access.StorageAccess,
    sync_unit_obj: sync_unit.SyncUnit,
    prefix: str,
    task_type: str,
    start_time: Optional[datetime],
    end_time: Optional[datetime],
    n_total: int,
    n_train: int,
    n_val: int,
    n_test: int,
    resample_freq: Optional[str] = None,
    filters: Optional[dict] = None,
) -> str:
    """Save lightweight snapshot metadata as JSON and return the path."""
    import json

    snapshot_folder_rel = f"{sync_unit_obj.get_path_for_sync_unit()}/{SNAPSHOT_FOLDER_NAME}"
    sa.create_folder(snapshot_folder_rel)
    ts_str = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    meta_rel_path = f"{snapshot_folder_rel}/{prefix}_snapshot_{ts_str}.json"

    meta = {
        "task_type": task_type,
        "start_time": start_time.isoformat() if start_time else None,
        "end_time": end_time.isoformat() if end_time else None,
        "n_total": n_total,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "resample_freq": resample_freq,
        "filters": filters or {},
        "created_at": datetime.now(UTC).isoformat(),
    }

    full_path = sa.create_full_path_string(meta_rel_path)
    import os
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta_rel_path


def build_event_rate_dataset(
    sync_unit_obj: sync_unit.SyncUnit,
    config: DatasetConfig,
    sa: Optional[storage_access.StorageAccess] = None,
    cron_patterns: Optional[List[dict]] = None,
    hourly_cron_patterns: Optional[List[dict]] = None,
) -> Tuple[EventRateDataset, DataSnapshotMetadata]:
    """Build an EventRateDataset and corresponding snapshot metadata.

    This assumes the data_preprocessing pipeline has already produced
    merged events-per-minute files for the given sync unit.

    Args:
        sync_unit_obj: The sync unit to build dataset for.
        config: Dataset configuration with time range and split ratios.
        sa: Optional storage access instance.
        cron_patterns: Pre-detected daily cron patterns (as dicts).
            When provided, skips detection and uses these directly.
        hourly_cron_patterns: Pre-detected hourly cron patterns (as dicts).
            When provided, skips detection and uses these directly.

    Returns:
        Tuple of (EventRateDataset, DataSnapshotMetadata).
    """
    if config.task_type != TaskType.EVENT_RATE:
        raise ValueError(
            f"build_event_rate_dataset called with non-EVENT_RATE task_type: {config.task_type}"
        )

    if sa is None:
        sa = storage_access.StorageAccess()

    df = _load_events_per_minute_dataframe(sa, sync_unit_obj)

    if config.start_time is not None:
        df = df[df["minute"] >= _to_naive_utc(config.start_time)]
    if config.end_time is not None:
        df = df[df["minute"] <= _to_naive_utc(config.end_time)]

    if config.resample_freq is not None:
        _validate_resample_freq(config.resample_freq)
        df = _resample_event_rate_data(df, config.resample_freq)

    if df.empty:
        raise ValueError(
            "No events-per-minute data found for the requested time range "
            f"[{config.start_time}, {config.end_time}]"
        )

    n = len(df)
    train_indices, val_indices, test_indices = _compute_splits(
        n, config.train_ratio, config.val_ratio
    )

    timestamps: List[datetime] = [_to_python_datetime(ts) for ts in df["minute"]]
    events_values: List[float] = [float(v) for v in df["events_per_minute"]]

    covariates: dict = {}
    detected_patterns: Optional[List[dict]] = None
    detected_hourly: Optional[List[dict]] = None

    if config.covariates:
        regular_names, cron_names = split_covariate_names(config.covariates)

        if regular_names:
            covariates = build_covariates(timestamps, regular_names)

        # Separate daily vs hourly cron covariate requests
        daily_cron_names = [n for n in cron_names if n in CRON_COVARIATE_NAMES]
        hourly_cron_names = [
            n for n in cron_names if n in HOURLY_CRON_COVARIATE_NAMES
        ]

        # Lazily compute training data only when needed
        train_ts = None
        train_vals = None
        detection_cfg = config.cron_detection or {}

        if daily_cron_names:
            if cron_patterns is not None:
                patterns = [CronPattern(**p) for p in cron_patterns]
                logger.info(
                    "Using %d pre-supplied daily cron patterns", len(patterns)
                )
            else:
                if train_ts is None:
                    train_ts = [timestamps[i] for i in train_indices]
                    train_vals = [events_values[i] for i in train_indices]
                patterns = detect_cron_patterns(
                    train_ts,
                    train_vals,
                    min_frequency=detection_cfg.get("min_frequency", 0.8),
                    min_occurrences=detection_cfg.get("min_occurrences", 5),
                    max_cv=detection_cfg.get("max_cv"),
                )

            cron_covs = build_cron_covariates(timestamps, patterns)
            covariates.update(cron_covs)
            detected_patterns = [
                {"hour": p.hour, "minute": p.minute,
                 "frequency": p.frequency,
                 "mean_magnitude": p.mean_magnitude,
                 "std_magnitude": p.std_magnitude}
                for p in patterns
            ]

        if hourly_cron_names:
            if hourly_cron_patterns is not None:
                h_patterns = [
                    HourlyCronPattern(**p) for p in hourly_cron_patterns
                ]
                logger.info(
                    "Using %d pre-supplied hourly cron patterns",
                    len(h_patterns),
                )
            else:
                if train_ts is None:
                    train_ts = [timestamps[i] for i in train_indices]
                    train_vals = [events_values[i] for i in train_indices]
                hourly_cfg = detection_cfg.get("hourly", {})
                h_patterns = detect_hourly_patterns(
                    train_ts,
                    train_vals,
                    min_frequency=hourly_cfg.get("min_frequency", 0.5),
                    min_occurrences=hourly_cfg.get("min_occurrences", 10),
                    max_cv=hourly_cfg.get("max_cv"),
                    min_relative_frequency=hourly_cfg.get(
                        "min_relative_frequency", 3.0
                    ),
                )

            hourly_covs = build_hourly_cron_covariates(timestamps, h_patterns)
            covariates.update(hourly_covs)
            detected_hourly = [
                {"minute": p.minute,
                 "frequency": p.frequency,
                 "mean_magnitude": p.mean_magnitude,
                 "std_magnitude": p.std_magnitude}
                for p in h_patterns
            ]

    dataset = EventRateDataset(
        timestamps=timestamps,
        events_per_minute=events_values,
        covariates=covariates,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        cron_patterns=detected_patterns,
        hourly_cron_patterns=detected_hourly,
    )

    meta_path = _save_snapshot_metadata(
        sa, sync_unit_obj, "event_rate",
        task_type=config.task_type.value,
        start_time=config.start_time,
        end_time=config.end_time,
        n_total=n,
        n_train=len(train_indices),
        n_val=len(val_indices),
        n_test=len(test_indices),
        resample_freq=config.resample_freq,
        filters=config.filters,
    )

    metadata = DataSnapshotMetadata(
        snapshot_metadata_path=meta_path,
        task_type=config.task_type,
        start_time=config.start_time,
        end_time=config.end_time,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        filters=config.filters,
        resample_freq=config.resample_freq,
    )

    return dataset, metadata


def build_time_to_event_dataset(
    sync_unit_obj: sync_unit.SyncUnit,
    config: DatasetConfig,
    sa: Optional[storage_access.StorageAccess] = None,
) -> Tuple[TimeToEventDataset, DataSnapshotMetadata]:
    """Build a TimeToEventDataset and corresponding snapshot metadata.

    This assumes the data_preprocessing pipeline has already produced
    merged time delta files for the given sync unit.

    Args:
        sync_unit_obj: The sync unit to build dataset for.
        config: Dataset configuration with time range and split ratios.
        sa: Optional storage access instance.

    Returns:
        Tuple of (TimeToEventDataset, DataSnapshotMetadata).
    """
    if config.task_type != TaskType.TIME_TO_EVENT:
        raise ValueError(
            f"build_time_to_event_dataset called with non-TIME_TO_EVENT task_type: {config.task_type}"
        )

    if sa is None:
        sa = storage_access.StorageAccess()

    df = _load_merged_time_deltas_dataframe(sa, sync_unit_obj)

    if config.start_time is not None:
        df = df[df["timestamp"] >= _to_naive_utc(config.start_time)]
    if config.end_time is not None:
        df = df[df["timestamp"] <= _to_naive_utc(config.end_time)]

    # min_delta filter: reconstructed intervals use the preprocessed minute
    # timestamps. Filter rows below threshold (in minutes).
    min_delta = config.filters.get("min_delta")
    n_before_filter = len(df)
    if min_delta is not None and min_delta > 0:
        df = df[df["time_delta"] >= min_delta].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No time delta data found for the requested time range "
            f"[{config.start_time}, {config.end_time}]"
            + (f" after min_delta={min_delta} filter" if min_delta else "")
        )

    # Resample to regular grid for neural models, or keep irregular
    # format for parametric models.
    covariates: dict = {}
    if config.resample_freq is not None:
        _validate_resample_freq(config.resample_freq)
        resampled = _resample_time_to_event_data(df, config.resample_freq)

        timestamps = [_to_python_datetime(ts) for ts in resampled["minute"]]
        time_deltas = [int(v) for v in resampled["minutes_to_next_event"]]
        covariates["elapsed_since_last_event"] = list(
            resampled["elapsed_since_last_event"].values
        )
    else:
        timestamps = [_to_python_datetime(ts) for ts in df["timestamp"]]
        time_deltas = [int(v) for v in df["time_delta"]]

    n = len(timestamps)
    train_indices, val_indices, test_indices = _compute_splits(
        n, config.train_ratio, config.val_ratio
    )

    if config.covariates:
        regular_names, cron_names = split_covariate_names(config.covariates)
        if cron_names:
            logger.warning(
                "Cron covariates %s are not applicable to time-to-event "
                "datasets and will be ignored",
                cron_names,
            )
        if regular_names:
            covariates.update(build_covariates(timestamps, regular_names))

    dataset = TimeToEventDataset(
        timestamps=timestamps,
        minutes_to_next_event=time_deltas,
        covariates=covariates,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )

    effective_filters = dict(config.filters)
    if min_delta is not None and min_delta > 0:
        effective_filters["min_delta_applied"] = min_delta
        effective_filters["n_before_min_delta_filter"] = n_before_filter
        effective_filters["n_after_min_delta_filter"] = n

    meta_path = _save_snapshot_metadata(
        sa, sync_unit_obj, "time_to_event",
        task_type=config.task_type.value,
        start_time=config.start_time,
        end_time=config.end_time,
        n_total=n,
        n_train=len(train_indices),
        n_val=len(val_indices),
        n_test=len(test_indices),
        resample_freq=config.resample_freq,
        filters=effective_filters,
    )

    metadata = DataSnapshotMetadata(
        snapshot_metadata_path=meta_path,
        task_type=config.task_type,
        start_time=config.start_time,
        end_time=config.end_time,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        filters=effective_filters,
        resample_freq=config.resample_freq,
    )

    return dataset, metadata
