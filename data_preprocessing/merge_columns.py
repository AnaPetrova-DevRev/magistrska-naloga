"""
Merge the created and modified columns into one column of events
"""

import pandas as pd

import storage_access

created_date_column = "created_date"
modified_date_column = "modified_date"
events_column = "events"
events_per_minute_column = "events_per_minute"

def merge_columns(merged_file_path: str, file_path: str) -> None:
    """
    Uses the clean data file to merge the created and modified columns into one column of events.
    Drop the created_date and modified_date columns.
    Save the merged dataset as a pickle file and return the file path.
    """
    sa = storage_access.StorageAccess()
    df = sa.read_picklefile_as_dataframe(file_path)

    # Merge the columns
    df = calculate_merge(df)

    # Save the merged dataset
    # sync_unit_path = sync_unit_obj.get_path_for_sync_unit()
    # app_folder = global_vars.DATA_PREPROCESSING_APP_FOLDER_NAME
    # merged_file_path = f"{sync_unit_path}/{app_folder}/{global_vars.MERGED_DATASET_FILE_NAME_FORMAT.format(int=chunk_number)}"
    sa.save_dataframe_as_picklefile(df, merged_file_path)


def calculate_merge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the dataframe and merge the created and modified columns into one column
    of events. Each distinct event becomes its own row.

    A cleaned row is a modification snapshot, not a record: the source returns one
    {id, created_date, modified_date} object per change, so created_date is a repeated
    attribute of the record rather than a per-row occurrence. The events of a record are
    therefore its single creation plus each of its distinct modification timestamps.
    Emitting created_date once per row would inflate the count at the creation minute by
    the number of times that record was later revised.

    The result is deduplicated on (id, events) and sorted by events in ascending order.

    Returns:
        DataFrame with 'id' and 'events' columns (one distinct event per row)
    """
    # Fall back to row identity when there is no id column, so that rows of different
    # records cannot collapse into each other
    ids = df['id'] if 'id' in df.columns else pd.Series(df.index, index=df.index)

    creations = pd.DataFrame({'id': ids, events_column: df[created_date_column]})
    modifications = pd.DataFrame({'id': ids, events_column: df[modified_date_column]})

    result_df = pd.concat([creations, modifications], ignore_index=True)
    result_df = result_df.dropna(subset=[events_column])

    # One row per distinct (record, timestamp). This collapses the repeated created_date,
    # a snapshot taken at the creation instant, and any redundant source objects
    result_df = result_df.drop_duplicates(subset=['id', events_column])

    # Sort by events in ascending order, keeping creations before modifications on ties
    result_df = result_df.sort_values(events_column, kind='stable').reset_index(drop=True)

    return result_df


def calculate_events_per_minute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the merged dataframe and calculate the number of events per minute.
    Returns a dataframe with 'minute' and 'events_per_minute' columns.
    """
    result_df = pd.DataFrame()
    # Ensure events column is in datetime format
    result_df[events_column] = pd.to_datetime(df[events_column])

    # Create a new column for the minute (truncate seconds)
    result_df['minute'] = result_df[events_column].dt.floor('min')

    # Group by minute and count events
    events_per_minute_df = result_df.groupby('minute').size().reset_index(name=events_per_minute_column)

    return events_per_minute_df


def read_and_save_merged_events_per_minute(merged_events_per_minute_file_path: str, merged_dataset_file_path: str) -> None:
    """
    Read the merged dataset, calculate events per minute, and save the result as a pickle file.
    """
    sa = storage_access.StorageAccess()
    merged_df = sa.read_picklefile_as_dataframe(merged_dataset_file_path)

    events_per_minute_df = calculate_events_per_minute(merged_df)

    sa.save_dataframe_as_picklefile(events_per_minute_df, merged_events_per_minute_file_path)