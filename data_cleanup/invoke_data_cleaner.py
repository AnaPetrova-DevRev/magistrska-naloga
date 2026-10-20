# given a sync unit, read the state of the sync unit
# find all the "work_{{int}}.json" files in the data folder for that sync unit
# based on the state's "last processed chunk number", find the next chunk to process
# read the file, parse it, save the parsed items in a new file "parsed_items_{{int}}.pkl" in the same folder
# update the state of the sync unit with the new "last processed chunk number"

# the data cleaner can: parse a json, or it can also do cleanup of the data - for example, deduplication, empty data removal etc.
import sys
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
import logging

from app_actions_constants import PARSE_JSON, CLEANUP_DATA, RESET_STATE
import storage_access
from app_flow_definitions import define_paths_for_input_output_files
import state
import global_vars
import data_cleanup.parse_json
import file_stats
from sync_unit import SyncUnit
from orchestrator_message import OrchestratorResponse, OrchestratorMessage

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class DataCleanerState:
    """Manages separate states for different data cleaner functions"""
    parse_json_state: state.State = field(default_factory=state.State)
    clean_data_state: state.State = field(default_factory=state.State)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "parse_json_state": asdict(self.parse_json_state),
            "clean_data_state": asdict(self.clean_data_state)
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DataCleanerState':
        """Create a DataCleanerState object from a dictionary"""
        obj = cls()
        if "parse_json_state" in data and data["parse_json_state"]:
            obj.parse_json_state = state.State.from_dict(data["parse_json_state"])
        if "clean_data_state" in data and data["clean_data_state"]:
            obj.clean_data_state = state.State.from_dict(data["clean_data_state"])
        return obj

    def get_state_for_function(self, function_name: str) -> state.State:
        """Get the state for a specific function"""
        if function_name == PARSE_JSON:
            return self.parse_json_state
        elif function_name == CLEANUP_DATA:
            return self.clean_data_state
        else:
            raise ValueError(f"Function {function_name} not supported in data cleaner state")

    def update_state_for_function(self, function_name: str, new_state: state.State) -> None:
        """Update the state for a specific function"""
        if function_name == PARSE_JSON:
            self.parse_json_state = new_state
        elif function_name == CLEANUP_DATA:
            self.clean_data_state = new_state
        else:
            raise ValueError(f"Function {function_name} not supported in data cleaner state")



def receive_message(message: 'OrchestratorMessage') -> OrchestratorResponse:
    """Process a message from the orchestrator"""
    logger.info("Data cleaner received message: %s", message.to_string())
    sa = storage_access.StorageAccess()
    sync_unit_obj = message.get_sync_unit()
    app_file_stats, app_state = get_stats_and_state(sa, sync_unit_obj)

    if message.action == PARSE_JSON:
        try:
            parse_json_state = app_state.get_state_for_function(PARSE_JSON)
        except ValueError:
            return OrchestratorResponse(False, "Function parse_json not supported in data cleaner state", is_done=True)
        response = invoke_json_parser(sa, sync_unit_obj, parse_json_state, app_file_stats)
        app_state.update_state_for_function(PARSE_JSON, parse_json_state)
        save_stats_and_state(sa, app_file_stats, app_state, sync_unit_obj)
        return response
    elif message.action == CLEANUP_DATA:
        try:
            clean_data_state = app_state.get_state_for_function(CLEANUP_DATA)
        except ValueError:
            return OrchestratorResponse(False, "Function clean_data not supported in data cleaner state", is_done=True)
        response = invoke_data_cleaner(sa, sync_unit_obj, clean_data_state, app_file_stats)
        app_state.update_state_for_function(CLEANUP_DATA, clean_data_state)
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

        if action_name not in [PARSE_JSON, CLEANUP_DATA, RESET_STATE]:
            return OrchestratorResponse(
                False,
                f"Action '{action_name}' is not a valid action for data_cleanup app",
                is_done=True
            )

        # Reset the state for the specified action
        try:
            # Ensure the app folder exists before saving state
            sync_unit_path = sync_unit_obj.get_path_for_sync_unit()
            app_folder_path = f"{sync_unit_path}/{global_vars.DATA_CLEANUP_APP_FOLDER_NAME}"
            if not sa.check_if_file_exists(app_folder_path):
                sa.create_folder(app_folder_path)

            new_state = state.State()
            app_state.update_state_for_function(action_name, new_state)
            save_stats_and_state(sa, app_file_stats, app_state, sync_unit_obj)
            logger.info(f"Reset state for action '{action_name}' in data_cleanup app")
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
    else:
        return OrchestratorResponse(False, "Action {message.action} not supported by data cleaner", is_done=True)


def get_stats_and_state(
    sa: storage_access.StorageAccess,
    sync_unit_obj: 'SyncUnit'
) -> tuple[file_stats.Stats, DataCleanerState]:
    """Load file statistics and state for a sync unit"""
    sync_unit_path = sync_unit_obj.get_path_for_sync_unit()
    app_path = f"{sync_unit_path}/{global_vars.DATA_CLEANUP_APP_FOLDER_NAME}/"
    state_path = f"{app_path}{global_vars.DATA_CLEANUP_STATE_FILE_NAME}"
    file_stats_path = f"{app_path}{global_vars.DATA_CLEANUP_FILE_STATS_FILE_NAME}"

    file_stats_exists = sa.check_if_file_exists(file_stats_path)
    if file_stats_exists:
        logger.info("File statistics exists for data cleaner, loading it")
        app_file_stats = sa.read_json_as_object(file_stats_path, file_stats.Stats.from_dict)
    else:
        logger.info("File statistics does not exist for data cleaner, creating new one")
        app_file_stats = file_stats.Stats()

    # Read state using the generic storage access method
    app_state = sa.read_json_as_object(state_path, DataCleanerState.from_dict)
    if app_state is None:
        app_state = DataCleanerState()

    return app_file_stats, app_state


def save_stats_and_state(
    sa: storage_access.StorageAccess,
    app_file_stats: file_stats.Stats,
    app_state: DataCleanerState,
    sync_unit_obj: 'SyncUnit'
) -> None:
    """Save file statistics and state for a sync unit"""
    sync_unit_path = sync_unit_obj.get_path_for_sync_unit()
    app_path = f"{sync_unit_path}/{global_vars.DATA_CLEANUP_APP_FOLDER_NAME}/"
    state_path = f"{app_path}{global_vars.DATA_CLEANUP_STATE_FILE_NAME}"
    file_stats_path = f"{app_path}{global_vars.DATA_CLEANUP_FILE_STATS_FILE_NAME}"
    sa.save_file_stats(app_file_stats, file_stats_path)
    sa.save_state(app_state.to_dict(), state_path)
    logger.info("Saved file statistics and state for data cleaner")


def invoke_json_parser(
    sa: storage_access.StorageAccess,
    sync_unit_obj: 'SyncUnit',
    parse_json_state: state.State,
    app_file_stats: file_stats.Stats
) -> OrchestratorResponse:
    """
    Parse JSON files and extract structured data.
    """
    # Check if the app folder exists, if not, create it
    app_folder_path = f"{sync_unit_obj.get_path_for_sync_unit()}/{global_vars.DATA_CLEANUP_APP_FOLDER_NAME}"
    if not sa.check_if_file_exists(app_folder_path):
        sa.create_folder(app_folder_path)

    # While we find a next file to process, the last processed chunk number is incremented and parse json is called
    while True:
        next_chunk_number = parse_json_state.last_processed_chunk_number + 1
        # sync_unit_path = sync_unit_obj.get_path_for_sync_unit()
        # file_name = global_vars.INITIAL_DATA_FILE_NAME_FORMAT.format(int=next_chunk_number)
        # file_path = f"{sync_unit_path}/{global_vars.INITIAL_DATA_FOLDER_NAME}/{file_name}"
        file_paths = define_paths_for_input_output_files(PARSE_JSON, sync_unit_obj, next_chunk_number)
        file_path = file_paths.input_filepath
        if not sa.check_if_file_exists(file_path):
            logger.info("File %s does not exist - all files processed", file_path)
            parse_json_state.is_done = True
            return OrchestratorResponse(True, "Processed all existing files", is_done=True)
        else:
            items, logs = data_cleanup.parse_json.parse_json_file(file_path)
            # Print the logs list of strings
            for log in logs:
                logger.info(log)
            result_file_path = file_paths.output_filepaths[0]
            data_cleanup.parse_json.save_parsed_items(sa, result_file_path, items)
            # Add file to stats, update the state with the new last processed chunk number, and save both the stats and the state
            new_file_statistics = file_stats.FileStatistics(
                file_name=result_file_path,
                output_file=result_file_path,
                input_file=file_path,
                created_by_function=PARSE_JSON,
                size=len(items),
                created_date=datetime.now()
            )
            app_file_stats.add_file_statistics(new_file_statistics)

        # Update the parse_json_state with the new progress
            parse_json_state.last_processed_chunk_number = next_chunk_number
            parse_json_state.total_processed += len(items)

            logger.info("Parsed file %s and saved result to %s.", file_path, result_file_path)


def invoke_data_cleaner(
        sa: storage_access.StorageAccess,
    sync_unit_obj: 'SyncUnit',
    clean_data_state: state.State,
    app_file_stats: file_stats.Stats
) -> OrchestratorResponse:
    """
    The data cleaner makes sure there are no nil values and that there are no duplicates in the data.
    Its input is the parsed items dataframe.
    The output is a cleaned dataframe that is saved as a new file.
    Use global vars for the file name and path.
    """
    sync_unit_path = sync_unit_obj.get_path_for_sync_unit()

    # Check if the app folder exists, if not, create it
    app_folder_path = f"{sync_unit_path}/{global_vars.DATA_CLEANUP_APP_FOLDER_NAME}"
    if not sa.check_if_file_exists(app_folder_path):
        sa.create_folder(app_folder_path)


    # Continue processing until we have processed all files
    while True:
        # check if the next chunk to be processed exists
        next_chunk_number = clean_data_state.last_processed_chunk_number + 1
        # file_name = f"{sync_unit_path}/{global_vars.DATA_CLEANUP_APP_FOLDER_NAME}/{global_vars.PARSED_DATA_FILE_NAME_FORMAT.format(int=next_chunk_number)}"

        file_paths = define_paths_for_input_output_files(CLEANUP_DATA, sync_unit_obj, next_chunk_number)
        file_name = file_paths.input_filepath
        if not sa.check_if_file_exists(file_name):
            logger.info("File %s does not exist - all files processed", file_name)
            return OrchestratorResponse(True, "Processed all existing files", is_done=True)
        else:
            # Load the parsed items dataframe
            df = sa.read_picklefile_as_dataframe(file_name)
            # Clean the data
            df = df.dropna()
            df = df.drop_duplicates()
            # Save the cleaned dataframe
            # result_file_path = f"{sync_unit_path}/{global_vars.CLEAN_DATA_FOLDER_NAME}/{global_vars.CLEANED_ITEMS_FILE_NAME_FORMAT.format(int=next_chunk_number)}"
            result_file_path = file_paths.output_filepaths[0]
            sa.save_dataframe_as_picklefile(df, result_file_path)
            # Add file to stats, update the state with the new last processed chunk number, and save both the stats and the state
            new_file_statistics = file_stats.FileStatistics(
                file_name=result_file_path,
                output_file=result_file_path,
                input_file=file_name,
                created_by_function=CLEANUP_DATA,
                size=len(df),
                created_date=datetime.now()
            )
            app_file_stats.add_file_statistics(new_file_statistics)

            # Update the clean_data_state with the new progress
            clean_data_state.last_processed_chunk_number = next_chunk_number
            clean_data_state.total_processed += len(df)

            logger.info("Cleaned file %s and saved result to %s.", file_name, result_file_path)