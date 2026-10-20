"""
Outlier detection module for Pipeline Explorer.

Provides functions to detect outliers using various statistical methods:
- Z-score
- Modified Z-score (MAD-based)
- IQR (Interquartile Range)
- Seasonal decomposition
"""

from enum import Enum
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Optional
from global_vars import (
    CLEANED_ITEMS_FILE_NAME_FORMAT,
    CREATED_TIME_DELTA_FILE_NAME_FORMAT,
    MODIFIED_TIME_DELTA_FILE_NAME_FORMAT,
    MERGED_TIME_DELTA_FILE_NAME_FORMAT,
    MERGED_EVENTS_PER_MINUTE_FILE_NAME_FORMAT
)


class OutlierMethod(Enum):
    """Enumeration of available outlier detection methods."""
    ZSCORE = "Z-score"
    MODIFIED_ZSCORE = "Modified Z-score"
    IQR = "IQR"
    SEASONAL = "Seasonal Decompose"


def detect_zscore_outliers(
    values: pd.Series,
    threshold: float = 3.0
) -> pd.Series:
    """
    Detect outliers using Z-score method.

    Args:
        values: Series of numerical values
        threshold: Z-score threshold (default 3.0)

    Returns:
        Boolean Series where True indicates an outlier
    """
    if values.empty or values.std() == 0:
        return pd.Series([False] * len(values), index=values.index)

    z_scores = np.abs((values - values.mean()) / values.std())
    return z_scores > threshold


def detect_modified_zscore_outliers(
    values: pd.Series,
    threshold: float = 3.5
) -> pd.Series:
    """
    Detect outliers using Modified Z-score (MAD-based) method.

    More robust than standard Z-score for non-normal distributions.
    Uses Median Absolute Deviation (MAD) instead of standard deviation.

    Args:
        values: Series of numerical values
        threshold: Modified Z-score threshold (default 3.5)

    Returns:
        Boolean Series where True indicates an outlier
    """
    if values.empty:
        return pd.Series([False] * len(values), index=values.index)

    median = values.median()
    mad = np.abs(values - median).median()

    # Handle zero MAD (all values are the same or nearly the same)
    if mad == 0 or np.isnan(mad):
        # Fall back to using mean absolute deviation from median
        mad = np.abs(values - median).mean()
        if mad == 0 or np.isnan(mad):
            return pd.Series([False] * len(values), index=values.index)

    # Modified Z-score formula: 0.6745 is the 0.75th quantile of standard normal
    modified_z = 0.6745 * (values - median) / mad
    return np.abs(modified_z) > threshold


def detect_iqr_outliers(
    values: pd.Series,
    multiplier: float = 1.5
) -> pd.Series:
    """
    Detect outliers using IQR (Interquartile Range) method.

    Args:
        values: Series of numerical values
        multiplier: IQR multiplier (default 1.5 for standard, 3.0 for extreme)

    Returns:
        Boolean Series where True indicates an outlier
    """
    if values.empty:
        return pd.Series([False] * len(values), index=values.index)

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    return (values < lower_bound) | (values > upper_bound)


def detect_seasonal_outliers(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    threshold: float = 3.0
) -> pd.Series:
    """
    Detect outliers using seasonal decomposition residuals.

    Groups data by hour of day and detects outliers in the residuals
    after removing the hourly pattern.

    Args:
        df: DataFrame containing the data
        date_column: Column name for datetime values
        value_column: Column name for numerical values to analyze
        threshold: Z-score threshold for residuals (default 3.0)

    Returns:
        Boolean Series where True indicates an outlier
    """
    if df.empty:
        return pd.Series([False] * len(df), index=df.index)

    # Extract hour from date column for seasonal pattern
    hours = pd.to_datetime(df[date_column]).dt.hour

    # Calculate hourly means
    hourly_means = df.groupby(hours)[value_column].transform('mean')

    # Calculate residuals (deviation from hourly pattern)
    residuals = df[value_column] - hourly_means

    # Apply Z-score to residuals
    return detect_zscore_outliers(residuals, threshold=threshold)


def detect_outliers(
    df: pd.DataFrame,
    value_column: str,
    method: OutlierMethod,
    date_column: Optional[str] = None,
    threshold: float = 3.0,
) -> pd.Series:
    """
    Detect outliers using the specified method.

    Args:
        df: DataFrame containing the data
        value_column: Column name for values to analyze
        method: Outlier detection method to use
        date_column: Column name for dates (required for seasonal method)
        threshold: Detection threshold

    Returns:
        Boolean Series where True indicates an outlier
    """
    values = df[value_column]

    if method == OutlierMethod.ZSCORE:
        return detect_zscore_outliers(values, threshold=threshold)
    elif method == OutlierMethod.MODIFIED_ZSCORE:
        return detect_modified_zscore_outliers(values, threshold=threshold)
    elif method == OutlierMethod.IQR:
        return detect_iqr_outliers(values, multiplier=threshold / 2)
    elif method == OutlierMethod.SEASONAL:
        if date_column is None:
            # Fall back to Z-score if no date column
            return detect_zscore_outliers(values, threshold=threshold)
        return detect_seasonal_outliers(df, date_column, value_column, threshold)
    else:
        return pd.Series([False] * len(df), index=df.index)


def create_outlier_scatter(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    outlier_mask: pd.Series,
    title: str,
    x_label: str = "Date",
    y_label: str = "Value",
) -> go.Figure:
    """
    Create a scatter plot highlighting outliers.

    Args:
        df: DataFrame containing the data
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        outlier_mask: Boolean Series where True indicates outliers
        title: Plot title
        x_label: Label for x-axis
        y_label: Label for y-axis

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Normal points
    normal_df = df[~outlier_mask]
    if not normal_df.empty:
        fig.add_trace(
            go.Scattergl(
                x=normal_df[x_column],
                y=normal_df[y_column],
                mode='markers',
                name='Normal',
                marker=dict(color='#3b82f6', size=5, opacity=0.6),
                hovertemplate=(
                    f'{x_label}: %{{x}}<br>'
                    f'{y_label}: %{{y}}<extra></extra>'
                ),
            )
        )

    # Outlier points
    outlier_df = df[outlier_mask]
    if not outlier_df.empty:
        fig.add_trace(
            go.Scattergl(
                x=outlier_df[x_column],
                y=outlier_df[y_column],
                mode='markers',
                name='Outliers',
                marker=dict(color='#ef4444', size=8, symbol='x'),
                hovertemplate=(
                    f'{x_label}: %{{x}}<br>'
                    f'{y_label}: %{{y}}<br>'
                    '<b>OUTLIER</b><extra></extra>'
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis=dict(
            title=x_label,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date" if pd.api.types.is_datetime64_any_dtype(df[x_column]) else None
        ),
        yaxis=dict(title=y_label),
        hovermode='closest',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def get_outlier_columns(file_path: str) -> list[tuple[str, str, str]]:
    """
    Get the columns to analyze for outliers based on file type.

    Args:
        file_path: Path to the file

    Returns:
        List of (x_column, y_column, title) tuples for outlier analysis
    """
    from pipeline_explorer.timeline_viewer import get_file_format_from_path

    file_format = get_file_format_from_path(file_path)
    columns = []

    if file_format == CLEANED_ITEMS_FILE_NAME_FORMAT:
        # For cleaned items, analyze event counts per minute
        columns.append(("created_date", "count", "Created Events Outliers"))
        columns.append(("modified_date", "count", "Modified Events Outliers"))
    elif file_format == CREATED_TIME_DELTA_FILE_NAME_FORMAT:
        columns.append((
            "created_date", "created_date_delta",
            "Created Time Delta Outliers"
        ))
    elif file_format == MODIFIED_TIME_DELTA_FILE_NAME_FORMAT:
        columns.append((
            "modified_date", "modified_date_delta",
            "Modified Time Delta Outliers"
        ))
    elif file_format == MERGED_TIME_DELTA_FILE_NAME_FORMAT:
        columns.append(("events", "events_delta", "Events Delta Outliers"))
    elif file_format == MERGED_EVENTS_PER_MINUTE_FILE_NAME_FORMAT:
        columns.append(("minute", "events_per_minute", "Events per Minute Outliers"))

    return columns


def get_outlier_plots(
    df: pd.DataFrame,
    file_path: str,
    method: OutlierMethod,
    threshold: float = 3.0,
) -> list[tuple[str, go.Figure, int, int]]:
    """
    Generate outlier detection plots for the given file.

    Args:
        df: DataFrame containing the file data
        file_path: Path to the file (used to determine columns)
        method: Outlier detection method to use
        threshold: Detection threshold

    Returns:
        List of (plot_name, figure, outlier_count, total_count) tuples
    """
    from pipeline_explorer.timeline_viewer import get_file_format_from_path

    file_format = get_file_format_from_path(file_path)
    plots = []

    if file_format == CLEANED_ITEMS_FILE_NAME_FORMAT:
        # Need to aggregate by minute first
        for date_col, color, title in [
            ("created_date", "#3b82f6", "Created Events Outliers"),
            ("modified_date", "#10b981", "Modified Events Outliers")
        ]:
            if date_col not in df.columns:
                continue

            # Aggregate by minute - use Series directly to avoid DataFrame copy
            rounded_dates = df[date_col].dt.round('min')
            grouped = rounded_dates.groupby(rounded_dates).size().reset_index(name='count')
            grouped.columns = [date_col, 'count']

            # Detect outliers in counts
            outlier_mask = detect_outliers(
                grouped, "count", method,
                date_column=date_col, threshold=threshold
            )

            fig = create_outlier_scatter(
                grouped, date_col, "count", outlier_mask,
                title=title, x_label=date_col, y_label="Events per minute"
            )

            plots.append((title, fig, outlier_mask.sum(), len(grouped)))

    elif file_format == CREATED_TIME_DELTA_FILE_NAME_FORMAT:
        if "created_date" in df.columns and "created_date_delta" in df.columns:
            # Use view instead of copy - no modification needed
            df_subset = df[["created_date", "created_date_delta"]]
            outlier_mask = detect_outliers(
                df_subset, "created_date_delta", method,
                date_column="created_date", threshold=threshold
            )
            fig = create_outlier_scatter(
                df_subset, "created_date", "created_date_delta", outlier_mask,
                title="Created Time Delta Outliers",
                x_label="Created Date", y_label="Time Delta"
            )
            plots.append((
                "Created Time Delta Outliers", fig,
                outlier_mask.sum(), len(df_subset)
            ))

    elif file_format == MODIFIED_TIME_DELTA_FILE_NAME_FORMAT:
        if "modified_date" in df.columns and "modified_date_delta" in df.columns:
            # Use view instead of copy - no modification needed
            df_subset = df[["modified_date", "modified_date_delta"]]
            outlier_mask = detect_outliers(
                df_subset, "modified_date_delta", method,
                date_column="modified_date", threshold=threshold
            )
            fig = create_outlier_scatter(
                df_subset, "modified_date", "modified_date_delta", outlier_mask,
                title="Modified Time Delta Outliers",
                x_label="Modified Date", y_label="Time Delta"
            )
            plots.append((
                "Modified Time Delta Outliers", fig,
                outlier_mask.sum(), len(df_subset)
            ))

    elif file_format == MERGED_TIME_DELTA_FILE_NAME_FORMAT:
        if "events" in df.columns and "events_delta" in df.columns:
            # Use view instead of copy - no modification needed
            df_subset = df[["events", "events_delta"]]
            outlier_mask = detect_outliers(
                df_subset, "events_delta", method,
                date_column="events", threshold=threshold
            )
            fig = create_outlier_scatter(
                df_subset, "events", "events_delta", outlier_mask,
                title="Events Delta Outliers",
                x_label="Events", y_label="Time Delta"
            )
            plots.append((
                "Events Delta Outliers", fig,
                outlier_mask.sum(), len(df_subset)
            ))

    elif file_format == MERGED_EVENTS_PER_MINUTE_FILE_NAME_FORMAT:
        if "minute" in df.columns and "events_per_minute" in df.columns:
            # Use view instead of copy - no modification needed
            df_subset = df[["minute", "events_per_minute"]]
            outlier_mask = detect_outliers(
                df_subset, "events_per_minute", method,
                date_column="minute", threshold=threshold
            )
            fig = create_outlier_scatter(
                df_subset, "minute", "events_per_minute", outlier_mask,
                title="Events per Minute Outliers",
                x_label="Time (Minute)", y_label="Event Count"
            )
            plots.append((
                "Events per Minute Outliers", fig,
                outlier_mask.sum(), len(df_subset)
            ))

    return plots


def can_detect_outliers(file_path: str) -> bool:
    """
    Check if outlier detection is available for the given file type.

    Args:
        file_path: Path to the file

    Returns:
        True if outlier detection is supported
    """
    from pipeline_explorer.timeline_viewer import get_file_format_from_path

    file_format = get_file_format_from_path(file_path)
    return file_format in [
        CLEANED_ITEMS_FILE_NAME_FORMAT,
        CREATED_TIME_DELTA_FILE_NAME_FORMAT,
        MODIFIED_TIME_DELTA_FILE_NAME_FORMAT,
        MERGED_TIME_DELTA_FILE_NAME_FORMAT,
        MERGED_EVENTS_PER_MINUTE_FILE_NAME_FORMAT
    ]
