"""
File Viewer Module

Provides functionality to load, paginate, filter, and sort pickle files
for exploration in the Pipeline Explorer app.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import pandas as pd

from storage_access import StorageAccess
from app_flow_definitions import extract_file_name_format, file_format_to_columns_map

logger = logging.getLogger(__name__)


class ColumnType(Enum):
    """Types of columns for filtering/sorting purposes"""
    ID = "id"
    DATE = "date"
    TIME_DELTA = "time_delta"
    OTHER = "other"


@dataclass
class ColumnInfo:
    """Information about a column for filtering/sorting"""
    name: str
    column_type: ColumnType
    dtype: str


@dataclass
class PageResult:
    """Result of paginating a dataframe"""
    data: pd.DataFrame
    total_rows: int
    total_pages: int
    current_page: int
    page_size: int


def load_pickle_file(file_path: str) -> pd.DataFrame:
    """
    Load a pickle file as a pandas DataFrame.

    Args:
        file_path: Path relative to the data bucket (e.g., "org/sys/proj/data_cleanup/cleaned_items_1.pkl")

    Returns:
        DataFrame loaded from the pickle file

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    sa = StorageAccess()
    return sa.read_picklefile_as_dataframe(file_path)


def get_column_info(df: pd.DataFrame, file_path: str) -> list[ColumnInfo]:
    """
    Get column information including types for filtering/sorting.

    Uses the file format to determine which columns are IDs, dates, and time deltas.

    Args:
        df: The DataFrame to analyze
        file_path: Path to the file (used to determine file format)

    Returns:
        List of ColumnInfo objects describing each column
    """
    file_format = extract_file_name_format(file_path)

    # Get expected columns from the format definition
    format_columns = file_format_to_columns_map.get(file_format)

    result = []
    for col in df.columns:
        col_str = str(col)

        # Determine column type
        if format_columns:
            if col_str == format_columns.id_column:
                col_type = ColumnType.ID
            elif col_str in format_columns.date_columns:
                col_type = ColumnType.DATE
            elif col_str in format_columns.other_columns and "delta" in col_str.lower():
                col_type = ColumnType.TIME_DELTA
            else:
                col_type = ColumnType.OTHER
        else:
            # Fallback: infer from column name
            if col_str.lower() == "id" or col_str.endswith("_id"):
                col_type = ColumnType.ID
            elif "date" in col_str.lower() or "time" in col_str.lower():
                if "delta" in col_str.lower():
                    col_type = ColumnType.TIME_DELTA
                else:
                    col_type = ColumnType.DATE
            else:
                col_type = ColumnType.OTHER

        result.append(ColumnInfo(
            name=col_str,
            column_type=col_type,
            dtype=str(df[col].dtype)
        ))

    return result


def get_page(df: pd.DataFrame, page_num: int, page_size: int = 50) -> PageResult:
    """
    Get a specific page of data from a DataFrame.

    Args:
        df: The DataFrame to paginate
        page_num: 1-based page number
        page_size: Number of rows per page (default: 50)

    Returns:
        PageResult with the page data and pagination info
    """
    total_rows = len(df)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)

    # Clamp page number to valid range
    page_num = max(1, min(page_num, total_pages))

    start_idx = (page_num - 1) * page_size
    end_idx = start_idx + page_size

    return PageResult(
        data=df.iloc[start_idx:end_idx].copy(),
        total_rows=total_rows,
        total_pages=total_pages,
        current_page=page_num,
        page_size=page_size
    )


def apply_text_filter(df: pd.DataFrame, column: str, search_text: str) -> pd.DataFrame:
    """
    Apply a text-based filter to a column.

    Args:
        df: The DataFrame to filter
        column: Column name to filter on
        search_text: Text to search for (case-insensitive)

    Returns:
        Filtered DataFrame
    """
    if not search_text or column not in df.columns:
        return df

    # Convert column to string and search case-insensitively
    mask = df[column].astype(str).str.contains(search_text, case=False, na=False)
    return df[mask].copy()


def apply_date_range_filter(
    df: pd.DataFrame,
    column: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Apply a date range filter to a column.

    Args:
        df: The DataFrame to filter
        column: Column name to filter on (must contain datetime values)
        start_date: Minimum date (inclusive), None for no lower bound
        end_date: Maximum date (inclusive), None for no upper bound

    Returns:
        Filtered DataFrame
    """
    if column not in df.columns:
        return df

    if start_date is None and end_date is None:
        return df

    result = df.copy()

    # Try to convert column to datetime if needed
    try:
        date_col = pd.to_datetime(result[column], errors='coerce')
    except (ValueError, TypeError):
        logger.warning("Could not convert column %s to datetime for filtering", column)
        return df

    if start_date is not None:
        result = result[date_col >= pd.to_datetime(start_date)]
        date_col = date_col[date_col >= pd.to_datetime(start_date)]

    if end_date is not None:
        result = result[date_col <= pd.to_datetime(end_date)]

    return result


def apply_sort(df: pd.DataFrame, column: str, ascending: bool = True) -> pd.DataFrame:
    """
    Sort a DataFrame by a column.

    Args:
        df: The DataFrame to sort
        column: Column name to sort by
        ascending: Sort order (True for ascending, False for descending)

    Returns:
        Sorted DataFrame
    """
    if column not in df.columns:
        return df

    return df.sort_values(by=column, ascending=ascending, na_position='last').copy()


def apply_filters_and_sort(
    df: pd.DataFrame,
    filters: dict[str, dict],
    sort_column: Optional[str] = None,
    sort_ascending: bool = True
) -> pd.DataFrame:
    """
    Apply multiple filters and sorting to a DataFrame.

    Args:
        df: The DataFrame to process
        filters: Dictionary mapping column names to filter configs:
            - For text filters: {"text": "search_string"}
            - For date filters: {"start_date": datetime, "end_date": datetime, "text": "search"}
        sort_column: Column to sort by (None for no sorting)
        sort_ascending: Sort direction

    Returns:
        Filtered and sorted DataFrame
    """
    result = df.copy()

    for column, filter_config in filters.items():
        if column not in result.columns:
            continue

        # Apply text filter if present
        if "text" in filter_config and filter_config["text"]:
            result = apply_text_filter(result, column, filter_config["text"])

        # Apply date range filter if present
        if "start_date" in filter_config or "end_date" in filter_config:
            result = apply_date_range_filter(
                result,
                column,
                filter_config.get("start_date"),
                filter_config.get("end_date")
            )

    # Apply sorting
    if sort_column:
        result = apply_sort(result, sort_column, sort_ascending)

    return result
