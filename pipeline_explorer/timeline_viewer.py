"""
Timeline visualization module for Pipeline Explorer.

Provides functions to create interactive timeline scatter plots using Plotly.
"""

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


def create_timeline_scatter(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    x_label: str = "Date",
    y_label: str = "Value",
    marker_color: str = "blue",
) -> go.Figure:
    """
    Create an interactive timeline scatter plot.

    Args:
        df: DataFrame containing the data
        x_column: Column name for x-axis (typically a date column)
        y_column: Column name for y-axis
        title: Plot title
        x_label: Label for x-axis
        y_label: Label for y-axis
        marker_color: Color for scatter points

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=df[x_column],
            y=df[y_column],
            mode='markers',
            name=title,
            marker=dict(color=marker_color, size=5),
            opacity=0.7,
            hovertemplate=(
                f"<b>{x_label}:</b> %{{x}}<br>"
                f"<b>{y_label}:</b> %{{y}}<extra></extra>"
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
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis=dict(title=y_label),
        hovermode='closest',
        height=500,
    )

    return fig


def create_event_count_scatter(
    df: pd.DataFrame,
    date_column: str,
    title: str,
    marker_color: str = "blue",
) -> go.Figure:
    """
    Create a scatter plot showing event counts per minute.

    Groups data by minute and creates a scatter plot of count per minute.

    Args:
        df: DataFrame containing the data
        date_column: Column name for the date column
        title: Plot title
        marker_color: Color for scatter points

    Returns:
        Plotly Figure object
    """
    # Round dates to nearest minute and group by minute
    # Use Series directly to avoid DataFrame copy overhead
    rounded_dates = df[date_column].dt.round('min')
    grouped = rounded_dates.groupby(rounded_dates).size().reset_index(name='count')
    grouped.columns = [date_column, 'count']

    return create_timeline_scatter(
        df=grouped,
        x_column=date_column,
        y_column='count',
        title=title,
        x_label=date_column,
        y_label="Events per minute",
        marker_color=marker_color,
    )


def get_file_format_from_path(file_path: str) -> Optional[str]:
    """
    Extract the file format constant from a file path.

    Args:
        file_path: Full path to the file

    Returns:
        The matching file format constant, or None if not found
    """
    import re
    file_name = file_path.split("/")[-1]

    # Check each format
    formats = [
        CLEANED_ITEMS_FILE_NAME_FORMAT,
        CREATED_TIME_DELTA_FILE_NAME_FORMAT,
        MODIFIED_TIME_DELTA_FILE_NAME_FORMAT,
        MERGED_TIME_DELTA_FILE_NAME_FORMAT,
        MERGED_EVENTS_PER_MINUTE_FILE_NAME_FORMAT,
    ]

    for fmt in formats:
        # Convert format to regex pattern
        pattern = fmt.replace("{int}", r"\d+")
        if re.match(pattern, file_name):
            return fmt

    return None


def get_required_columns(file_path: str) -> list[str]:
    """
    Get the columns required for timeline visualization of a given file.

    Args:
        file_path: Path to the file

    Returns:
        List of column names needed for visualization
    """
    file_format = get_file_format_from_path(file_path)

    if file_format == CLEANED_ITEMS_FILE_NAME_FORMAT:
        return ["created_date", "modified_date"]
    elif file_format == CREATED_TIME_DELTA_FILE_NAME_FORMAT:
        return ["created_date", "created_date_delta"]
    elif file_format == MODIFIED_TIME_DELTA_FILE_NAME_FORMAT:
        return ["modified_date", "modified_date_delta"]
    elif file_format == MERGED_TIME_DELTA_FILE_NAME_FORMAT:
        return ["events", "events_delta"]
    elif file_format == MERGED_EVENTS_PER_MINUTE_FILE_NAME_FORMAT:
        return ["minute", "events_per_minute"]
    return []


def get_timeline_plots(df: pd.DataFrame, file_path: str) -> list[tuple[str, go.Figure]]:
    """
    Generate timeline plots appropriate for the given file type.

    Uses only the required columns for memory efficiency.

    Args:
        df: DataFrame containing the file data
        file_path: Path to the file (used to determine file type)

    Returns:
        List of (plot_name, figure) tuples
    """
    file_format = get_file_format_from_path(file_path)
    plots = []

    # Get only required columns to reduce memory usage
    required_cols = get_required_columns(file_path)
    available_cols = [c for c in required_cols if c in df.columns]
    if not available_cols:
        return plots

    # Work with only the required columns
    df_subset = df[available_cols]

    if file_format == CLEANED_ITEMS_FILE_NAME_FORMAT:
        # Two count plots: created_date and modified_date
        if "created_date" in df_subset.columns:
            fig = create_event_count_scatter(
                df=df_subset[["created_date"]],
                date_column="created_date",
                title="Created Events per Minute",
                marker_color="#3b82f6",  # Blue
            )
            plots.append(("Created Date Events", fig))

        if "modified_date" in df_subset.columns:
            fig = create_event_count_scatter(
                df=df_subset[["modified_date"]],
                date_column="modified_date",
                title="Modified Events per Minute",
                marker_color="#10b981",  # Green
            )
            plots.append(("Modified Date Events", fig))

    elif file_format == CREATED_TIME_DELTA_FILE_NAME_FORMAT:
        # Scatter: created_date vs created_date_delta
        if "created_date" in df_subset.columns and "created_date_delta" in df_subset.columns:
            fig = create_timeline_scatter(
                df=df_subset[["created_date", "created_date_delta"]],
                x_column="created_date",
                y_column="created_date_delta",
                title="Created Date vs Time Delta",
                x_label="Created Date",
                y_label="Time Delta",
                marker_color="#8b5cf6",  # Purple
            )
            plots.append(("Created Time Delta", fig))

    elif file_format == MODIFIED_TIME_DELTA_FILE_NAME_FORMAT:
        # Scatter: modified_date vs modified_date_delta
        if "modified_date" in df_subset.columns and "modified_date_delta" in df_subset.columns:
            fig = create_timeline_scatter(
                df=df_subset[["modified_date", "modified_date_delta"]],
                x_column="modified_date",
                y_column="modified_date_delta",
                title="Modified Date vs Time Delta",
                x_label="Modified Date",
                y_label="Time Delta",
                marker_color="#f59e0b",  # Amber
            )
            plots.append(("Modified Time Delta", fig))

    elif file_format == MERGED_TIME_DELTA_FILE_NAME_FORMAT:
        # Scatter: events vs events_delta
        if "events" in df_subset.columns and "events_delta" in df_subset.columns:
            fig = create_timeline_scatter(
                df=df_subset[["events", "events_delta"]],
                x_column="events",
                y_column="events_delta",
                title="Events vs Time Delta",
                x_label="Event Time",
                y_label="Time Delta",
                marker_color="#ef4444",  # Red
            )
            plots.append(("Events Time Delta", fig))

    elif file_format == MERGED_EVENTS_PER_MINUTE_FILE_NAME_FORMAT:
        # Scatter: minute vs events_per_minute
        if "minute" in df_subset.columns and "events_per_minute" in df_subset.columns:
            fig = create_timeline_scatter(
                df=df_subset[["minute", "events_per_minute"]],
                x_column="minute",
                y_column="events_per_minute",
                title="Events per Minute",
                x_label="Time (Minute)",
                y_label="Event Count",
                marker_color="#06b6d4",  # Cyan
            )
            plots.append(("Events per Minute", fig))

    return plots


def can_visualize_file(file_path: str) -> bool:
    """
    Check if timeline visualization is available for the given file.

    Args:
        file_path: Path to the file

    Returns:
        True if the file can be visualized, False otherwise
    """
    return get_file_format_from_path(file_path) is not None
