"""Calculate and save time deltas for a given file"""
import pandas as pd

import storage_access
from app_registry import logger

created_date_column = "created_date"
modified_date_column = "modified_date"
events_column = "events"

def calculate_time_deltas_for_column(df: pd.DataFrame, unique_id: bool, time_column: str, id_column: str = "id") -> pd.DataFrame:
    """
    Calculate inter-event time deltas in integer minutes.

    Sorts by time_column, floors timestamps to minute granularity,
    deduplicates (one row per minute), and computes the delta between
    consecutive rows in rounded minutes.

    If unique_id is True, only keep the first row for each id (useful for
    created_date where we want one entry per entity).
    """
    df = df.sort_values(time_column)
    new_col_name = time_column + "_delta"
    if unique_id:
        df = pd.DataFrame(df.drop_duplicates(id_column, keep='first'))

    df = df.copy()
    df[time_column] = df[time_column].dt.floor("min")
    df = df.drop_duplicates(subset=[time_column], keep="last").reset_index(drop=True)

    df[new_col_name] = (
        df[time_column].diff().dt.total_seconds() / 60.0
    ).round().astype("Int64")

    return df


def calculate_time_delta(output_file_paths: list[str], file_path: str, id_column: str = "id") -> tuple[str, str]:
    """
    Calculate time delta for a given file and return the file paths for the created and modified time delta files.
    """
    sa = storage_access.StorageAccess()
    df = sa.read_picklefile_as_dataframe(file_path)

    created_time_delta_file_path = ""
    modified_time_delta_file_path = ""

    for output_file_path in output_file_paths:
        if "created" in output_file_path:
            created_time_delta_file_path = output_file_path
        if "modified" in output_file_path:
            modified_time_delta_file_path = output_file_path

    if created_time_delta_file_path == "":
        logger.warning("No created time delta file found for file %s", file_path)
        return "", ""

    if modified_time_delta_file_path == "":
        logger.warning("No modified time delta file found for file %s", file_path)
        return "", ""


    # Calculate and save the time deltas for the created date
    df_created_time_delta = calculate_time_deltas_for_column(df, True, created_date_column, id_column)
    #created_time_delta_file_path = f"{sync_unit_obj.get_path_for_sync_unit()}/{global_vars.DATA_PREPROCESSING_APP_FOLDER_NAME}/{global_vars.CREATED_TIME_DELTA_FILE_NAME_FORMAT.format(int=chunk_number)}"
    save_time_deltas(sa, created_time_delta_file_path, df_created_time_delta)

    # Calculate and save the time deltas for the modified date
    df_modified_time_delta = calculate_time_deltas_for_column(df, False, modified_date_column, id_column)
    #modified_time_delta_file_path = f"{sync_unit_obj.get_path_for_sync_unit()}/{global_vars.DATA_PREPROCESSING_APP_FOLDER_NAME}/{global_vars.MODIFIED_TIME_DELTA_FILE_NAME_FORMAT.format(int=chunk_number)}"
    save_time_deltas(sa, modified_time_delta_file_path, df_modified_time_delta)

    return created_time_delta_file_path, modified_time_delta_file_path

def save_time_deltas(
    sa: storage_access.StorageAccess,
    file_path: str,
    df: pd.DataFrame,
) -> str:
    """Save time deltas as a pickle file and return the file path"""
    sa.save_dataframe_as_picklefile(df, file_path)
    return file_path


def calculate_merged_time_delta(merged_delta_filepath: str, file_path: str) -> None:
    """
    Calculate time delta for the merged dataset.

    The merged dataset already has one event per row (id, events columns).
    This function calculates time deltas between consecutive events.
    """
    sa = storage_access.StorageAccess()
    df = sa.read_picklefile_as_dataframe(file_path)

    # Remove rows where events is empty/NaN
    df = df[df[events_column].notna()]

    # Reset index
    df = df.reset_index(drop=True)

    # Calculate time deltas on the events column
    df = calculate_time_deltas_for_column(df, False, events_column, id_column="id")

    save_time_deltas(sa, merged_delta_filepath, df)