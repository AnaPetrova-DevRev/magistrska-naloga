"""
Analysis Per Record Module

Provides functionality to calculate per-record statistics from clean data files.
Each unique ID gets aggregated statistics about its modification patterns.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from storage_access import StorageAccess
from sync_unit import SyncUnit
from global_vars import (
    CLEAN_DATA_FOLDER_NAME,
    CLEANED_ITEMS_FILE_NAME_FORMAT,
    PIPELINE_EXPLORER_APP_FOLDER_NAME,
    ANALYSIS_PER_RECORD_FILE_NAME_FORMAT
)

logger = logging.getLogger(__name__)

# Time bucket thresholds in seconds
ONE_SECOND = 1
TEN_SECONDS = 10
ONE_MINUTE = 60
ONE_HOUR = 3600
FOUR_HOURS = 4 * 3600
ONE_DAY = 24 * 3600
ONE_WEEK = 7 * ONE_DAY
ONE_MONTH = 30 * ONE_DAY  # Approximation
THREE_MONTHS = 90 * ONE_DAY  # Approximation

# Day of week names (Monday=0, Sunday=6)
DAY_NAMES = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']


def load_all_clean_data(sync_unit: SyncUnit) -> Optional[pd.DataFrame]:
    """
    Load all clean data pickle files for a sync unit and combine them.

    Args:
        sync_unit: The sync unit to load data for

    Returns:
        Combined DataFrame with all clean data, or None if no files found
    """
    sa = StorageAccess()
    sync_unit_path = sync_unit.get_path_for_sync_unit()
    clean_data_path = f"{sync_unit_path}/{CLEAN_DATA_FOLDER_NAME}"

    # Find all clean data files
    all_dfs = []
    chunk_number = 1

    while True:
        file_name = CLEANED_ITEMS_FILE_NAME_FORMAT.replace("{int}", str(chunk_number))
        file_path = f"{clean_data_path}/{file_name}"

        if not sa.check_if_file_exists(file_path):
            break

        logger.info("Loading clean data file: %s", file_path)
        df = sa.read_picklefile_as_dataframe(file_path)
        all_dfs.append(df)
        chunk_number += 1

    if not all_dfs:
        logger.warning("No clean data files found in: %s", clean_data_path)
        return None

    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(
        "Loaded %d clean data files with %d total rows",
        len(all_dfs), len(combined_df)
    )

    return combined_df


def calculate_day_of_week_stats(group: pd.DataFrame) -> dict:
    """Calculate day of week statistics for a group of modifications."""
    total = len(group)
    day_of_week = group['modified_date'].dt.dayofweek

    stats = {}
    for day_num, day_name in enumerate(DAY_NAMES):
        count = (day_of_week == day_num).sum()
        stats[f'num_changes_{day_name}'] = count
        stats[f'pct_changes_{day_name}'] = (count / total * 100) if total > 0 else 0.0

    # Weekday (Mon-Fri) and Weekend (Sat-Sun)
    stats['num_changes_weekday'] = (day_of_week < 5).sum()
    stats['num_changes_weekend'] = (day_of_week >= 5).sum()

    return stats


def calculate_hour_of_day_stats(group: pd.DataFrame) -> dict:
    """Calculate hour of day statistics for a group of modifications."""
    hour_of_day = group['modified_date'].dt.hour

    stats = {}
    for hour in range(24):
        stats[f'num_changes_hour_{hour:02d}'] = (hour_of_day == hour).sum()

    return stats


def calculate_time_delta_stats(deltas_seconds: np.ndarray) -> dict:
    """
    Calculate statistics on time deltas between modifications.

    Args:
        deltas_seconds: Array of time deltas in seconds

    Returns:
        Dictionary with delta statistics
    """
    stats = {}

    if len(deltas_seconds) == 0:
        # No deltas (only one modification or none)
        stats['avg_time_between_modifications'] = None
        stats['min_time_between_modifications'] = None
        stats['max_time_between_modifications'] = None
        stats['std_time_between_modifications'] = None
        stats['q1_time_between_modifications'] = None
        stats['q3_time_between_modifications'] = None
        return stats

    stats['avg_time_between_modifications'] = float(np.mean(deltas_seconds))
    stats['min_time_between_modifications'] = float(np.min(deltas_seconds))
    stats['max_time_between_modifications'] = float(np.max(deltas_seconds))
    stats['std_time_between_modifications'] = float(np.std(deltas_seconds))
    stats['q1_time_between_modifications'] = float(np.percentile(deltas_seconds, 25))
    stats['q3_time_between_modifications'] = float(np.percentile(deltas_seconds, 75))

    return stats


def calculate_delta_bucket_stats(deltas_seconds: np.ndarray) -> dict:
    """
    Categorize time deltas into buckets.

    Buckets:
    - < 1 second
    - 1s to 10s
    - 10s to 1 minute
    - 1 minute to 1 hour
    - 1 hour to 4 hours
    - 4 hours to 1 day
    - > 1 day
    - > 1 week
    - > 1 month

    Args:
        deltas_seconds: Array of time deltas in seconds

    Returns:
        Dictionary with bucket counts
    """
    stats = {
        'num_deltas_lt_1s': 0,
        'num_deltas_1s_to_10s': 0,
        'num_deltas_10s_to_1min': 0,
        'num_deltas_1min_to_1h': 0,
        'num_deltas_1h_to_4h': 0,
        'num_deltas_4h_to_1d': 0,
        'num_deltas_gt_1d': 0,
        'num_deltas_gt_1w': 0,
        'num_deltas_gt_1mo': 0,
    }

    if len(deltas_seconds) == 0:
        return stats

    for delta in deltas_seconds:
        if delta < ONE_SECOND:
            stats['num_deltas_lt_1s'] += 1
        elif delta < TEN_SECONDS:
            stats['num_deltas_1s_to_10s'] += 1
        elif delta < ONE_MINUTE:
            stats['num_deltas_10s_to_1min'] += 1
        elif delta < ONE_HOUR:
            stats['num_deltas_1min_to_1h'] += 1
        elif delta < FOUR_HOURS:
            stats['num_deltas_1h_to_4h'] += 1
        elif delta < ONE_DAY:
            stats['num_deltas_4h_to_1d'] += 1
        else:
            stats['num_deltas_gt_1d'] += 1

        # Additional checks for > 1 week and > 1 month (these overlap with > 1 day)
        if delta > ONE_WEEK:
            stats['num_deltas_gt_1w'] += 1
        if delta > ONE_MONTH:
            stats['num_deltas_gt_1mo'] += 1

    return stats


def analyze_single_record(group: pd.DataFrame, cutoff_date: pd.Timestamp) -> dict:
    """
    Calculate all statistics for a single record (ID).

    Args:
        group: DataFrame containing all rows for a single ID
        cutoff_date: The latest date in the entire dataset (for 3-month check)

    Returns:
        Dictionary with all calculated statistics
    """
    # Sort by modified_date to calculate deltas correctly
    group = group.sort_values('modified_date')

    stats = {'id': group['id'].iloc[0]}

    # Date statistics
    first_created_date = group['created_date'].min()
    last_modified_date = group['modified_date'].max()

    stats['first_created_date'] = first_created_date
    stats['last_modified_date'] = last_modified_date

    # First modified date that is different from created date
    modified_dates = group['modified_date']
    different_from_created = modified_dates[modified_dates != first_created_date]
    if len(different_from_created) > 0:
        stats['first_modified_date'] = different_from_created.min()
    else:
        stats['first_modified_date'] = None

    # Time spans
    stats['time_first_created_to_last_modified'] = (
        (last_modified_date - first_created_date).total_seconds()
        if pd.notna(first_created_date) and pd.notna(last_modified_date)
        else None
    )

    stats['time_to_first_modification'] = (
        (stats['first_modified_date'] - first_created_date).total_seconds()
        if stats['first_modified_date'] is not None and pd.notna(first_created_date)
        else None
    )

    # Day of week stats
    stats.update(calculate_day_of_week_stats(group))

    # Hour of day stats
    stats.update(calculate_hour_of_day_stats(group))

    # Calculate time deltas between consecutive modifications
    sorted_dates = group['modified_date'].sort_values()
    deltas = sorted_dates.diff().dropna()
    deltas_seconds = deltas.dt.total_seconds().values

    # Delta statistics
    stats.update(calculate_time_delta_stats(deltas_seconds))

    # Delta bucket stats
    stats.update(calculate_delta_bucket_stats(deltas_seconds))

    # Was created 3 months before cutoff
    three_months_before_cutoff = cutoff_date - pd.Timedelta(seconds=THREE_MONTHS)
    stats['was_created_3mo_before_cutoff'] = (
        first_created_date <= three_months_before_cutoff
        if pd.notna(first_created_date)
        else False
    )

    # Total changes
    stats['num_changes_total'] = len(group)

    return stats


def calculate_analysis_per_record(sync_unit: SyncUnit) -> Optional[pd.DataFrame]:
    """
    Calculate per-record analysis for all clean data.

    This function:
    1. Loads all clean data pickle files
    2. Groups by ID
    3. Calculates all statistics for each ID
    4. Returns a DataFrame with one row per ID

    Args:
        sync_unit: The sync unit to analyze

    Returns:
        DataFrame with per-record statistics, or None if no data
    """
    # Load all clean data
    df = load_all_clean_data(sync_unit)

    if df is None or len(df) == 0:
        logger.warning("No clean data available for analysis")
        return None

    logger.info("Starting analysis per record for %d rows", len(df))

    # Determine cutoff date (latest date in dataset)
    cutoff_date = df['modified_date'].max()
    logger.info("Cutoff date for 3-month check: %s", cutoff_date)

    # Group by ID and calculate stats
    results = []
    grouped = df.groupby('id')
    total_groups = len(grouped)

    for i, (record_id, group) in enumerate(grouped):
        if (i + 1) % 10000 == 0:
            logger.info("Processing record %d of %d", i + 1, total_groups)

        try:
            record_stats = analyze_single_record(group, cutoff_date)
            results.append(record_stats)
        except Exception as e:
            logger.error("Error processing record %s: %s", record_id, str(e))
            continue

    if not results:
        logger.warning("No records could be processed")
        return None

    # Convert to DataFrame
    result_df = pd.DataFrame(results)
    logger.info(
        "Analysis complete: %d unique records with %d columns",
        len(result_df), len(result_df.columns)
    )

    return result_df


def save_analysis_results(
    sync_unit: SyncUnit,
    result_df: pd.DataFrame,
    output_filename: str = ANALYSIS_PER_RECORD_FILE_NAME_FORMAT
) -> str:
    """
    Save the analysis results to a pickle file.

    Args:
        sync_unit: The sync unit
        result_df: DataFrame with analysis results
        output_filename: Name of the output file

    Returns:
        Path to the saved file
    """
    sa = StorageAccess()
    sync_unit_path = sync_unit.get_path_for_sync_unit()

    # Save to pipeline_explorer folder
    output_path = f"{sync_unit_path}/{PIPELINE_EXPLORER_APP_FOLDER_NAME}/{output_filename}"

    # Ensure folder exists
    folder_path = f"{sync_unit_path}/{PIPELINE_EXPLORER_APP_FOLDER_NAME}"
    if not sa.check_if_file_exists(folder_path):
        sa.create_folder(folder_path)

    sa.save_dataframe_as_picklefile(result_df, output_path)
    logger.info("Saved analysis results to: %s", output_path)

    return output_path
