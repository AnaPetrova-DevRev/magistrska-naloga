"""Invoke the functions and take care of the state and stats"""

import sys
import os
from datetime import datetime


from dataclasses import dataclass, field, asdict
import logging

import data_preprocessing.calculate_time_delta as calculate_time_delta
import data_preprocessing.merge_columns as merge_columns
import storage_access

from app_actions_constants import CALCULATE_ORIGINAL_TIME_DELTAS, CREATE_MERGED_DATASET, CALCULATE_MERGED_TIME_DELTAS, CALCULATE_MERGED_EVENTS_PER_MINUTE, RESET_STATE
from app_flow_definitions import define_paths_for_input_output_files
import state
import global_vars
import file_stats

from sync_unit import SyncUnit


from orchestrator_message import OrchestratorResponse, OrchestratorMessage

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class DataPreprocessingState:
    """Manages separate states for different data preprocessing functions"""
    calculate_original_time_deltas_state: state.State = field(default_factory=state.State)
    create_merged_dataset_state: state.State = field(default_factory=state.State)
    calculate_merged_time_deltas_state: state.State = field(default_factory=state.State)
    calculate_merged_events_per_minute_state: state.State = field(default_factory=state.State)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "calculate_original_time_deltas_state": asdict(self.calculate_original_time_deltas_state),
            "create_merged_dataset_state": asdict(self.create_merged_dataset_state),
            "calculate_merged_time_deltas_state": asdict(self.calculate_merged_time_deltas_state),
            "calculate_merged_events_per_minute_state": asdict(self.calculate_merged_events_per_minute_state)
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DataPreprocessingState':
        """Create a DataPreprocessingState object from a dictionary"""
        obj = cls()
        if "calculate_original_time_deltas_state" in data and data["calculate_original_time_deltas_state"]:
            obj.calculate_original_time_deltas_state = state.State.from_dict(data["calculate_original_time_deltas_state"])
        if "create_merged_dataset_state" in data and data["create_merged_dataset_state"]:
            obj.create_merged_dataset_state = state.State.from_dict(data["create_merged_dataset_state"])
        if "calculate_merged_time_deltas_state" in data and data["calculate_merged_time_deltas_state"]:
            obj.calculate_merged_time_deltas_state = state.State.from_dict(data["calculate_merged_time_deltas_state"])
        if "calculate_merged_events_per_minute_state" in data and data["calculate_merged_events_per_minute_state"]:
            obj.calculate_merged_events_per_minute_state = state.State.from_dict(data["calculate_merged_events_per_minute_state"])
        return obj

    def get_state_for_function(self, function_name: str) -> state.State:
        """Get the state for a specific function"""
        if function_name == CALCULATE_ORIGINAL_TIME_DELTAS:
            return self.calculate_original_time_deltas_state
        elif function_name == CREATE_MERGED_DATASET:
            return self.create_merged_dataset_state
        elif function_name == CALCULATE_MERGED_TIME_DELTAS:
            return self.calculate_merged_time_deltas_state
        elif function_name == CALCULATE_MERGED_EVENTS_PER_MINUTE:
            return self.calculate_merged_events_per_minute_state
        else:
            raise ValueError(f"Function {function_name} not supported in data preprocessing state")

    def update_state_for_function(self, function_name: str, new_state: state.State) -> None:
        """Update the state for a specific function"""
        if function_name == CALCULATE_ORIGINAL_TIME_DELTAS:
            self.calculate_original_time_deltas_state = new_state
        elif function_name == CREATE_MERGED_DATASET:
            self.create_merged_dataset_state = new_state
        elif function_name == CALCULATE_MERGED_TIME_DELTAS:
            self.calculate_merged_time_deltas_state = new_state
        elif function_name == CALCULATE_MERGED_EVENTS_PER_MINUTE:
            self.calculate_merged_events_per_minute_state = new_state
        else:
            raise ValueError(f"Function {function_name} not supported in data preprocessing state")


def receive_message(message: 'OrchestratorMessage') -> OrchestratorResponse:
    """Process a message from the orchestrator"""
    logger.info("Data preprocessing received message: %s", message.to_string())
    sa = storage_access.StorageAccess()
    sync_unit_obj = message.get_sync_unit()
    app_file_stats, app_state = get_stats_and_state(sa, sync_unit_obj)

    if message.action == CALCULATE_ORIGINAL_TIME_DELTAS:
        try:
            calculate_original_time_deltas_state = app_state.get_state_for_function(CALCULATE_ORIGINAL_TIME_DELTAS)
        except ValueError:
            return OrchestratorResponse(False, "Function calculate_original_time_deltas not supported in data preprocessing state", is_done=True)
        response = invoke_calculate_original_time_deltas(sa, sync_unit_obj, calculate_original_time_deltas_state, app_file_stats)
        app_state.update_state_for_function(CALCULATE_ORIGINAL_TIME_DELTAS, calculate_original_time_deltas_state)
        save_stats_and_state(sa, app_file_stats, app_state, sync_unit_obj)
        return response
    elif message.action == CREATE_MERGED_DATASET:
        try:
            create_merged_dataset_state = app_state.get_state_for_function(CREATE_MERGED_DATASET)
        except ValueError:
            return OrchestratorResponse(False, "Function create_merged_dataset not supported in data preprocessing state", is_done=True)
        response = invoke_create_merged_dataset(sa, sync_unit_obj, create_merged_dataset_state, app_file_stats)
        app_state.update_state_for_function(CREATE_MERGED_DATASET, create_merged_dataset_state)
        save_stats_and_state(sa, app_file_stats, app_state, sync_unit_obj)
        return response
    elif message.action == CALCULATE_MERGED_TIME_DELTAS:
        try:
            calculate_merged_time_deltas_state = app_state.get_state_for_function(CALCULATE_MERGED_TIME_DELTAS)
        except ValueError:
            return OrchestratorResponse(False, "Function calculate_merged_time_deltas not supported in data preprocessing state", is_done=True)
        response = invoke_calculate_merged_time_deltas(sa, sync_unit_obj, calculate_merged_time_deltas_state, app_file_stats)
        app_state.update_state_for_function(CALCULATE_MERGED_TIME_DELTAS, calculate_merged_time_deltas_state)
        save_stats_and_state(sa, app_file_stats, app_state, sync_unit_obj)
        return response
    elif message.action == CALCULATE_MERGED_EVENTS_PER_MINUTE:
        try:
            calculate_merged_events_per_minute_state = app_state.get_state_for_function(CALCULATE_MERGED_EVENTS_PER_MINUTE)
        except ValueError:
            return OrchestratorResponse(False, "Function calculate_merged_events_per_minute not supported in data preprocessing state", is_done=True)
        response = invoke_calculate_merged_events_per_minute(sa, sync_unit_obj, calculate_merged_events_per_minute_state, app_file_stats)
        app_state.update_state_for_function(CALCULATE_MERGED_EVENTS_PER_MINUTE, calculate_merged_events_per_minute_state)
        save_stats_and_state(sa, app_file_stats, app_state, sync_unit_obj)
        return response
    elif message.action == RESET_STATE:
        # Reset state for a specific action
        action_name = message.parameters.get("action_name")
        if not action_name:
            return OrchestratorResponse(
                False,
                "Parameter 'action_name' is required for reset_state action",
                is_done=True
            )

        valid_actions = [CALCULATE_ORIGINAL_TIME_DELTAS, CREATE_MERGED_DATASET, CALCULATE_MERGED_TIME_DELTAS, CALCULATE_MERGED_EVENTS_PER_MINUTE]
        if action_name not in valid_actions:
            return OrchestratorResponse(
                False,
                f"Action '{action_name}' is not a valid action for data_preprocessing app",
                is_done=True
            )

        # Reset the state for the specified action
        try:
            # Ensure the app folder exists before saving state
            sync_unit_path = sync_unit_obj.get_path_for_sync_unit()
            app_folder_path = f"{sync_unit_path}/{global_vars.DATA_PREPROCESSING_APP_FOLDER_NAME}"
            if not sa.check_if_file_exists(app_folder_path):
                sa.create_folder(app_folder_path)

            new_state = state.State()
            app_state.update_state_for_function(action_name, new_state)
            save_stats_and_state(sa, app_file_stats, app_state, sync_unit_obj)
            logger.info("Reset state for action %s in data_preprocessing app", action_name)
            return OrchestratorResponse(
                True,
                f"Successfully reset state for action '{action_name}'",
                is_done=True
            )
        except ValueError as e:
            return OrchestratorResponse(
                False,
                f"Error resetting state for action '{action_name}': {str(e)}",
                is_done=True
            )

    return OrchestratorResponse(False, f"Action '{message.action}' not supported by data preprocessing", is_done=True)

def get_stats_and_state(
    sa: storage_access.StorageAccess,
    sync_unit_obj: 'SyncUnit'
) -> tuple[file_stats.Stats, DataPreprocessingState]:
    """Load file statistics and state for a sync unit"""
    sync_unit_path = sync_unit_obj.get_path_for_sync_unit()
    app_path = f"{sync_unit_path}/{global_vars.DATA_PREPROCESSING_APP_FOLDER_NAME}/"
    state_path = f"{app_path}{global_vars.DATA_PREPROCESSING_STATE_FILE_NAME}"
    file_stats_path = f"{app_path}{global_vars.DATA_PREPROCESSING_FILE_STATS_FILE_NAME}"

    file_stats_exists = sa.check_if_file_exists(file_stats_path)
    if file_stats_exists:
        logger.info("File statistics exists for data preprocessing, loading it")
        app_file_stats = sa.read_json_as_object(file_stats_path, file_stats.Stats.from_dict)
    else:
        logger.info("File statistics does not exist for data preprocessing, creating new one")
        app_file_stats = file_stats.Stats()

    # Read state using the generic storage access method
    app_state = sa.read_json_as_object(state_path, DataPreprocessingState.from_dict)
    if app_state is None:
        logger.info("State does not exist for data preprocessing, creating new one")
        app_state = DataPreprocessingState()

    return app_file_stats, app_state

def save_stats_and_state(
    sa: storage_access.StorageAccess,
    app_file_stats: file_stats.Stats,
    app_state: DataPreprocessingState,
    sync_unit_obj: 'SyncUnit'
) -> None:
    """Save file statistics and state for a sync unit"""
    sync_unit_path = sync_unit_obj.get_path_for_sync_unit()
    app_path = f"{sync_unit_path}/{global_vars.DATA_PREPROCESSING_APP_FOLDER_NAME}/"
    state_path = f"{app_path}{global_vars.DATA_PREPROCESSING_STATE_FILE_NAME}"
    file_stats_path = f"{app_path}{global_vars.DATA_PREPROCESSING_FILE_STATS_FILE_NAME}"
    sa.save_file_stats(app_file_stats, file_stats_path)
    sa.save_state(app_state.to_dict(), state_path)
    logger.info("Saved file statistics and state for data preprocessing")

def invoke_calculate_original_time_deltas(
    sa: storage_access.StorageAccess,
    sync_unit_obj: 'SyncUnit',
    calculate_original_time_deltas_state: state.State,
    app_file_stats: file_stats.Stats
) -> OrchestratorResponse:
    """
    Calculate time deltas for original created and modified dates.
    """

    # Check if the app folder exists, if not, create it
    app_folder_path = f"{sync_unit_obj.get_path_for_sync_unit()}/{global_vars.DATA_PREPROCESSING_APP_FOLDER_NAME}"
    if not sa.check_if_file_exists(app_folder_path):
        sa.create_folder(app_folder_path)

    # While we find a next file to process, the last processed chunk number is incremented and calculate time deltas is called
    while True:
        next_chunk_number = calculate_original_time_deltas_state.last_processed_chunk_number + 1
        #sync_unit_path = sync_unit_obj.get_path_for_sync_unit()
        #file_name = f"{sync_unit_path}/{global_vars.CLEAN_DATA_FOLDER_NAME}/{global_vars.CLEANED_ITEMS_FILE_NAME_FORMAT.format(int=next_chunk_number)}"

        file_paths = define_paths_for_input_output_files(CALCULATE_ORIGINAL_TIME_DELTAS, sync_unit_obj, next_chunk_number)
        file_name = file_paths.input_filepath
        if not sa.check_if_file_exists(file_name):
            logger.info("File %s does not exist - all files processed", file_name)
            calculate_original_time_deltas_state.is_done = True
            return OrchestratorResponse(True, "Processed all existing files", is_done=True)
        else:
            created_time_delta_file_path, modified_time_delta_file_path = calculate_time_delta.calculate_time_delta(file_paths.output_filepaths, file_name)
            if created_time_delta_file_path != "":
            # Add file to stats, update the state with the new last processed chunk number, and save both the stats and the state
                new_file_statistics = file_stats.FileStatistics(
                    file_name=created_time_delta_file_path,
                    output_file=created_time_delta_file_path,
                    input_file=file_name,
                    created_by_function=CALCULATE_ORIGINAL_TIME_DELTAS,
                    size=0,
                    created_date=datetime.now()
                )
                app_file_stats.add_file_statistics(new_file_statistics)

            if modified_time_delta_file_path != "":
                new_file_statistics = file_stats.FileStatistics(
                    file_name=modified_time_delta_file_path,
                    output_file=modified_time_delta_file_path,
                    input_file=file_name,
                    created_by_function=CALCULATE_ORIGINAL_TIME_DELTAS,
                    size=0,
                    created_date=datetime.now()
                )
                app_file_stats.add_file_statistics(new_file_statistics)

            calculate_original_time_deltas_state.last_processed_chunk_number = next_chunk_number

            logger.info("Calculated time deltas for file %s and saved results to %s and %s.", file_name, created_time_delta_file_path, modified_time_delta_file_path)

def invoke_create_merged_dataset(
    sa: storage_access.StorageAccess,
    sync_unit_obj: 'SyncUnit',
    create_merged_dataset_state: state.State,
    app_file_stats: file_stats.Stats
) -> OrchestratorResponse:
    """
    Create a merged dataset with sorted timestamps from the original created and modified time columns.
    """

    # Check if the app folder exists, if not, create it
    sync_unit_path = sync_unit_obj.get_path_for_sync_unit()
    app_folder_path = f"{sync_unit_path}/{global_vars.DATA_PREPROCESSING_APP_FOLDER_NAME}"
    if not sa.check_if_file_exists(app_folder_path):
        sa.create_folder(app_folder_path)

    # While we find a next file to process, the last processed chunk number is incremented and create merged dataset is called
    while True:
        next_chunk_number = create_merged_dataset_state.last_processed_chunk_number + 1
        # file_name = f"{sync_unit_path}/{global_vars.CLEAN_DATA_FOLDER_NAME}/{global_vars.CLEANED_ITEMS_FILE_NAME_FORMAT.format(int=next_chunk_number)}"

        file_paths = define_paths_for_input_output_files(CREATE_MERGED_DATASET, sync_unit_obj, next_chunk_number)
        file_name = file_paths.input_filepath
        if not sa.check_if_file_exists(file_name):
            logger.info("File %s does not exist - all files processed", file_name)
            create_merged_dataset_state.is_done = True
            return OrchestratorResponse(True, "Processed all existing files", is_done=True)
        else:
            merged_file_path = file_paths.output_filepaths[0]
            merge_columns.merge_columns(merged_file_path, file_name)
            # Add file to stats, update the state with the new last processed chunk number, and save both the stats and the state
            new_file_statistics = file_stats.FileStatistics(
                file_name=merged_file_path,
                output_file=merged_file_path,
                input_file=file_name,
                created_by_function=CREATE_MERGED_DATASET,
                size=0,
                created_date=datetime.now()
            )
            app_file_stats.add_file_statistics(new_file_statistics)

            create_merged_dataset_state.last_processed_chunk_number = next_chunk_number
            create_merged_dataset_state.total_processed += 1

            logger.info("Created merged dataset for file %s and saved result to %s.", file_name, merged_file_path)

def invoke_calculate_merged_time_deltas(
    sa: storage_access.StorageAccess,
    sync_unit_obj: 'SyncUnit',
    calculate_merged_time_deltas_state: state.State,
    app_file_stats: file_stats.Stats
) -> OrchestratorResponse:
    """
    Take the merged dataset and create a new column that has the delta between the current row and the previous row.
    """

        # Check if the app folder exists, if not, create it
    sync_unit_path = sync_unit_obj.get_path_for_sync_unit()
    app_folder_path = f"{sync_unit_path}/{global_vars.DATA_PREPROCESSING_APP_FOLDER_NAME}"
    if not sa.check_if_file_exists(app_folder_path):
        sa.create_folder(app_folder_path)

    # While we find a next file to process, the last processed chunk number is incremented and create merged dataset is called
    # The file we are reading here is the merged dataset file.
    while True:
        next_chunk_number = calculate_merged_time_deltas_state.last_processed_chunk_number + 1
        # file_name = f"{sync_unit_path}/{global_vars.DATA_PREPROCESSING_APP_FOLDER_NAME}/{global_vars.MERGED_DATASET_FILE_NAME_FORMAT.format(int=next_chunk_number)}"

        file_paths = define_paths_for_input_output_files(CALCULATE_MERGED_TIME_DELTAS, sync_unit_obj, next_chunk_number)
        file_name = file_paths.input_filepath
        if not sa.check_if_file_exists(file_name):
            logger.info("File %s does not exist - all files processed", file_name)
            calculate_merged_time_deltas_state.is_done = True
            return OrchestratorResponse(True, "Processed all existing files", is_done=True)
        else:
            merged_delta_filepath = file_paths.output_filepaths[0]
            calculate_time_delta.calculate_merged_time_delta(merged_delta_filepath, file_name)
            # Add file to stats, update the state with the new last processed chunk number, and save both the stats and the state
            new_file_statistics = file_stats.FileStatistics(
                file_name=merged_delta_filepath,
                output_file=merged_delta_filepath,
                input_file=file_name,
                created_by_function=CALCULATE_MERGED_TIME_DELTAS,
                size=0,
                created_date=datetime.now()
            )
            app_file_stats.add_file_statistics(new_file_statistics)

            calculate_merged_time_deltas_state.last_processed_chunk_number = next_chunk_number
            calculate_merged_time_deltas_state.total_processed += 1

            logger.info("Calculated time deltas for merged dataset file %s and saved result to %s.", file_name, merged_delta_filepath)


def invoke_calculate_merged_events_per_minute(
    sa: storage_access.StorageAccess,
    sync_unit_obj: 'SyncUnit',
    calculate_merged_events_per_minute_state: state.State,
    app_file_stats: file_stats.Stats
) -> OrchestratorResponse:
    """
    Take the merged dataset and calculate the number of events per minute based on the timestamps.
    """

    # Check if the app folder exists, if not, create it
    sync_unit_path = sync_unit_obj.get_path_for_sync_unit()
    app_folder_path = f"{sync_unit_path}/{global_vars.DATA_PREPROCESSING_APP_FOLDER_NAME}"
    if not sa.check_if_file_exists(app_folder_path):
        sa.create_folder(app_folder_path)

    # While we find a next file to process, the last processed chunk number is incremented and calculate events per minute is called
    # The file we are reading here is the merged dataset file.
    while True:
        next_chunk_number = calculate_merged_events_per_minute_state.last_processed_chunk_number + 1

        file_paths = define_paths_for_input_output_files(CALCULATE_MERGED_EVENTS_PER_MINUTE, sync_unit_obj, next_chunk_number)
        file_name = file_paths.input_filepath
        if not sa.check_if_file_exists(file_name):
            logger.info("File %s does not exist - all files processed", file_name)
            calculate_merged_events_per_minute_state.is_done = True
            return OrchestratorResponse(True, "Processed all existing files", is_done=True)
        else:
            events_per_minute_filepath = file_paths.output_filepaths[0]
            merge_columns.read_and_save_merged_events_per_minute(events_per_minute_filepath, file_name)
            # Add file to stats, update the state with the new last processed chunk number, and save both the stats and the state
            new_file_statistics = file_stats.FileStatistics(
                file_name=events_per_minute_filepath,
                output_file=events_per_minute_filepath,
                input_file=file_name,
                created_by_function=CALCULATE_MERGED_EVENTS_PER_MINUTE,
                size=0,
                created_date=datetime.now()
            )
            app_file_stats.add_file_statistics(new_file_statistics)

            calculate_merged_events_per_minute_state.last_processed_chunk_number = next_chunk_number
            calculate_merged_events_per_minute_state.total_processed += 1

            logger.info("Calculated events per minute for merged dataset file %s and saved result to %s.", file_name, events_per_minute_filepath)