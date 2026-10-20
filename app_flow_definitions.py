from os.path import join, basename
import re

import logging
import networkx
import pandas as pd



from app_actions_constants import *
from global_vars import *
from storage_access import StorageAccess
from sync_unit import SyncUnit
from file_stats import Stats
from file_metadata import (
    FileMetadata, DateColumnMetadata, PipelineMetadata, FileFormatColumns,
    FileValidationResult, TrainingReadinessResult, TrainingReadinessStatus
)



logger = logging.getLogger(__name__)

# Goal: for a given action, define the input and output file format.

# Define expected columns for each file format
file_format_to_columns_map: dict[str, FileFormatColumns] = {
    INITIAL_DATA_FILE_NAME_FORMAT: FileFormatColumns(
        id_column="id",
        date_columns=[],  # JSON file, columns not applicable
        other_columns=[]
    ),
    PARSED_DATA_FILE_NAME_FORMAT: FileFormatColumns(
        id_column="id",
        date_columns=["created_date", "modified_date"],
        other_columns=[]
    ),
    CLEANED_ITEMS_FILE_NAME_FORMAT: FileFormatColumns(
        id_column="id",
        date_columns=["created_date", "modified_date"],
        other_columns=[]
    ),
    CREATED_TIME_DELTA_FILE_NAME_FORMAT: FileFormatColumns(
        id_column="id",
        date_columns=["created_date", "modified_date"],
        other_columns=["created_date_delta"]
    ),
    MODIFIED_TIME_DELTA_FILE_NAME_FORMAT: FileFormatColumns(
        id_column="id",
        date_columns=["created_date", "modified_date"],
        other_columns=["modified_date_delta"]
    ),
    MERGED_DATASET_FILE_NAME_FORMAT: FileFormatColumns(
        id_column="id",
        date_columns=["events"],  # events is a timestamp column (one per row)
        other_columns=[]
    ),
    MERGED_TIME_DELTA_FILE_NAME_FORMAT: FileFormatColumns(
        id_column="id",
        date_columns=["events"],  # events is a timestamp column (one per row)
        other_columns=["events_delta"]
    ),
    MERGED_EVENTS_PER_MINUTE_FILE_NAME_FORMAT: FileFormatColumns(
        id_column="",
        date_columns=["minute"],  # JSON file, columns not applicable
        other_columns=["events_per_minute"]
    )
}

# For a given sync unit, action and chunk number, define the input file path and output file path

class InputOutputFiles:
    def __init__(self, input_file: str, input_file_folder, output_files: list[str], output_file_folder: str) -> None:
        self.input_file = input_file
        self.input_file_folder = input_file_folder
        self.output_files = output_files
        self.output_file_folder = output_file_folder

class InputOutputPaths:
    def __init__(self, input_filepath: str, output_filepaths: list[str]) -> None:
        self.input_filepath = input_filepath # input can only be one
        self.output_filepaths = output_filepaths # can have multiple outputs

action_to_input_output_map = {
    PARSE_JSON: InputOutputFiles(
        input_file=INITIAL_DATA_FILE_NAME_FORMAT,
        input_file_folder=INITIAL_DATA_FOLDER_NAME,
        output_files=[PARSED_DATA_FILE_NAME_FORMAT],
        output_file_folder=DATA_CLEANUP_APP_FOLDER_NAME
    ),
    CLEANUP_DATA: InputOutputFiles(
        input_file=PARSED_DATA_FILE_NAME_FORMAT,
        input_file_folder=DATA_CLEANUP_APP_FOLDER_NAME,
        output_files=[CLEANED_ITEMS_FILE_NAME_FORMAT],
        output_file_folder=CLEAN_DATA_FOLDER_NAME
    ),
    CALCULATE_ORIGINAL_TIME_DELTAS: InputOutputFiles(
        input_file=CLEANED_ITEMS_FILE_NAME_FORMAT,
        input_file_folder=CLEAN_DATA_FOLDER_NAME,
        output_files=[CREATED_TIME_DELTA_FILE_NAME_FORMAT, MODIFIED_TIME_DELTA_FILE_NAME_FORMAT],
        output_file_folder=DATA_PREPROCESSING_APP_FOLDER_NAME,
    ),
    CREATE_MERGED_DATASET: InputOutputFiles(
        input_file=CLEANED_ITEMS_FILE_NAME_FORMAT,
        input_file_folder=CLEAN_DATA_FOLDER_NAME,
        output_files=[MERGED_DATASET_FILE_NAME_FORMAT],
        output_file_folder=DATA_PREPROCESSING_APP_FOLDER_NAME
    ),
    CALCULATE_MERGED_TIME_DELTAS: InputOutputFiles(
        input_file=MERGED_DATASET_FILE_NAME_FORMAT,
        input_file_folder=DATA_PREPROCESSING_APP_FOLDER_NAME,
        output_files=[MERGED_TIME_DELTA_FILE_NAME_FORMAT],
        output_file_folder=DATA_PREPROCESSING_APP_FOLDER_NAME
    ),
    CALCULATE_MERGED_EVENTS_PER_MINUTE: InputOutputFiles(
        input_file=MERGED_DATASET_FILE_NAME_FORMAT,
        input_file_folder=DATA_PREPROCESSING_APP_FOLDER_NAME,
        output_files=[MERGED_EVENTS_PER_MINUTE_FILE_NAME_FORMAT],
        output_file_folder=DATA_PREPROCESSING_APP_FOLDER_NAME
    ),
    ANALYSIS_PER_RECORD: InputOutputFiles(
        input_file=CLEANED_ITEMS_FILE_NAME_FORMAT,
        input_file_folder=CLEAN_DATA_FOLDER_NAME,
        output_files=[ANALYSIS_PER_RECORD_FILE_NAME_FORMAT],
        output_file_folder=PIPELINE_EXPLORER_APP_FOLDER_NAME
    )
}

action_to_stats_file_map = {
    PARSE_JSON: join(DATA_CLEANUP_APP_FOLDER_NAME, DATA_CLEANUP_FILE_STATS_FILE_NAME),
    CLEANUP_DATA: join(DATA_CLEANUP_APP_FOLDER_NAME, DATA_CLEANUP_FILE_STATS_FILE_NAME),
    CALCULATE_ORIGINAL_TIME_DELTAS: join(DATA_PREPROCESSING_APP_FOLDER_NAME, DATA_PREPROCESSING_FILE_STATS_FILE_NAME),
    CREATE_MERGED_DATASET: join(DATA_PREPROCESSING_APP_FOLDER_NAME, DATA_PREPROCESSING_FILE_STATS_FILE_NAME),
    CALCULATE_MERGED_TIME_DELTAS: join(DATA_PREPROCESSING_APP_FOLDER_NAME, DATA_PREPROCESSING_FILE_STATS_FILE_NAME),
    ANALYSIS_PER_RECORD: join(PIPELINE_EXPLORER_APP_FOLDER_NAME, PIPELINE_EXPLORER_STATS_FILE_NAME)
}

def define_input_output_files(action: str) -> InputOutputFiles | ValueError:
    """For a given action, return the input and output file formats."""
    if action in action_to_input_output_map:
        return action_to_input_output_map[action]
    else:
        return ValueError(f"{action} is not mapped in the action_to_input_output_map")


def define_paths_for_input_output_files(action: str, sync_unit: SyncUnit, chunk_number: int) -> InputOutputPaths | ValueError:
    """For a given action, sync unit and chunk number, return the full file paths for the input and output files."""
    if action in action_to_input_output_map:
        input_output_files = action_to_input_output_map[action]

        sync_unit_path = sync_unit.get_path_for_sync_unit()

        input_path = join(sync_unit_path, input_output_files.input_file_folder, f"{input_output_files.input_file.format(int=chunk_number)}")
        output_filepaths = [join(sync_unit_path, input_output_files.output_file_folder, f"{output_file.format(int=chunk_number)}") for output_file in input_output_files.output_files]

        return InputOutputPaths(
            input_filepath=input_path,
            output_filepaths=output_filepaths
        )
    else:
        return ValueError(f"{action} is not mapped in the action_to_input_output_map")


def build_full_app_flow_graph():
    """
    Based on the definitions in action to input_output_map, define a
    full graph from beginning to end for the actions. Should be based on the files, input and output.
    """
    start_file = INITIAL_DATA_FILE_NAME_FORMAT # always the top of the hierarchy
    graph = networkx.DiGraph()  # Use DiGraph for directed graph

    # Add the starting node
    graph.add_node(start_file)

    # Iterate through all actions and build edges based on input/output relationships
    for action, input_output_files in action_to_input_output_map.items():
        input_file = input_output_files.input_file
        output_files = input_output_files.output_files

        # Add input node if not already present
        if input_file not in graph:
            graph.add_node(input_file)

        # Add edges from input to each output file
        for output_file in output_files:
            # Add output node if not already present
            if output_file not in graph:
                graph.add_node(output_file)

            # Add edge from input to output with action as edge attribute
            graph.add_edge(input_file, output_file, action=action)

    return graph


def extract_file_name_format(file_path: str) -> str | None:
    """
    Extract the file name format from a full file path.

    Converts a specific file path to its generic format template.
    Example:
        data/ecosphere/zendesk/ecosphere_zendesk/data_cleanup/parsed_items_1.pkl
        -> parsed_items_{int}.pkl

        work_5.json -> work_{int}.json

    Args:
        file_path: Full path or just filename

    Returns:
        The file name format (e.g., "parsed_items_{int}.pkl") or None if not recognized
    """
    # Extract just the filename from the path
    filename = basename(file_path)

    # Define all known file name formats and their patterns
    file_formats = {
        INITIAL_DATA_FILE_NAME_FORMAT: r'^work_\d+\.json$',
        PARSED_DATA_FILE_NAME_FORMAT: r'^parsed_items_\d+\.pkl$',
        CLEANED_ITEMS_FILE_NAME_FORMAT: r'^cleaned_items_\d+\.pkl$',
        CREATED_TIME_DELTA_FILE_NAME_FORMAT: r'^created_time_delta_\d+\.pkl$',
        MODIFIED_TIME_DELTA_FILE_NAME_FORMAT: r'^modified_time_delta_\d+\.pkl$',
        MERGED_DATASET_FILE_NAME_FORMAT: r'^merged_dataset_\d+\.pkl$',
        MERGED_TIME_DELTA_FILE_NAME_FORMAT: r'^merged_time_delta_\d+\.pkl$',
        MERGED_EVENTS_PER_MINUTE_FILE_NAME_FORMAT: r'^merged_events_per_minute_\d+\.pkl$',
    }

    # Try to match the filename against each pattern
    for file_format, pattern in file_formats.items():
        if re.match(pattern, filename):
            return file_format

    return None


def get_file_pipeline(file_path: str) -> networkx.DiGraph | None:
    """
    Get the pipeline for a given file, showing all upstream inputs and downstream outputs.

    Args:
        file_path: Full path or filename (e.g., "data/ecosphere/zendesk/.../parsed_items_1.pkl"
                   or just "parsed_items_1.pkl")
        is_input: If True, treats the file as an input and returns all predecessors and successors.
                  If False, treats the file as an output and returns all predecessors and successors.
                  (Note: The behavior is the same regardless, but the parameter is kept for clarity)

    Returns:
        A directed subgraph containing:
        - All files that came before this file (predecessors/ancestors)
        - The file itself
        - All files that come after this file (successors/descendants)
        Returns None if the file format is not recognized

    Example:
        For "parsed_items_1.pkl":
        - Predecessors: work_{int}.json
        - File itself: parsed_items_{int}.pkl
        - Successors: cleaned_items_{int}.pkl, and all files downstream from it
    """
    # Extract the file name format
    file_format = extract_file_name_format(file_path)

    if file_format is None:
        return None

    # Build the full app flow graph
    full_graph = build_full_app_flow_graph()

    # Check if the file format exists in the graph
    if file_format not in full_graph:
        return None

    # Get all ancestors (files that came before)
    ancestors = networkx.ancestors(full_graph, file_format)

    # Get all descendants (files that come after)
    descendants = networkx.descendants(full_graph, file_format)

    # Combine all nodes: ancestors + the file itself + descendants
    pipeline_nodes = ancestors | descendants | {file_format}

    # Create a subgraph with only these nodes
    pipeline_graph = full_graph.subgraph(pipeline_nodes).copy()

    return pipeline_graph


def read_all_app_stats(sync_unit: SyncUnit) -> list[Stats]:
    """
    Read all app statistics for a given sync unit.
    Returns a list of Stats objects.
    """
    sa = StorageAccess()
    stats = []
    for _, stats_path in action_to_stats_file_map.items():
        full_stats_file_path = join(sync_unit.get_path_for_sync_unit(), stats_path)
        # Use Stats.from_dict which handles both list and dict formats
        stats_object = sa.read_json_as_object(full_stats_file_path, Stats.from_dict)
        if stats_object is not None:
            stats.append(stats_object)
    return stats


def extract_chunk_number(file_path: str) -> int | None:
    """
    Extract the chunk number from a file path.

    Example:
        parsed_items_5.pkl -> 5
        work_10.json -> 10
        data/org/sys/proj/folder/cleaned_items_42.pkl -> 42

    Args:
        file_path: Full path or just filename

    Returns:
        The chunk number as an integer, or None if not found
    """
    # Extract just the filename from the path
    filename = basename(file_path)

    # Pattern to match: <name>_<number>.<extension>
    # This matches files like work_5.json, parsed_items_10.pkl, etc.
    pattern = r'_(\d+)\.'
    match = re.search(pattern, filename)

    if match:
        return int(match.group(1))

    return None


def get_pipeline_file_paths(file_path: str, sync_unit: SyncUnit) -> dict[str, str | None] | None:
    """
    Get the exact file paths for all files in the pipeline for a given file.

    This function:
    1. Extracts the chunk number from the input file path
    2. Gets the pipeline graph for that file format
    3. Reads all app stats for the sync unit
    4. For each file format in the pipeline, tries to find the actual file path

    Args:
        file_path: Full path or filename (e.g., "parsed_items_5.pkl" or full path)
        sync_unit: The sync unit object

    Returns:
        A dictionary mapping file formats to their actual file paths.
        Returns None if the file format is not recognized.
        File paths in the dictionary may be None if the file doesn't exist yet.

    Example:
        For "parsed_items_5.pkl" with sync_unit "org/sys/proj":
        {
            "work_{int}.json": "org/sys/proj/initial_data/work_5.json",
            "parsed_items_{int}.pkl": "org/sys/proj/data_cleanup/parsed_items_5.pkl",
            "cleaned_items_{int}.pkl": "org/sys/proj/clean_data/cleaned_items_5.pkl",
            ...
        }
    """
    # Extract the chunk number from the file path
    chunk_number = extract_chunk_number(file_path)
    if chunk_number is None:
        return None

    # Get the pipeline graph for this file
    pipeline_graph = get_file_pipeline(file_path)
    if pipeline_graph is None:
        return None

    # Read all app stats for the sync unit
    all_stats = read_all_app_stats(sync_unit)

    # Create a mapping of file formats to actual file paths
    result = {}

    # For each file format in the pipeline
    for file_format in pipeline_graph.nodes():
        # Construct the expected file path based on the file format
        actual_file_path = _construct_file_path_for_format(
            file_format,
            sync_unit,
            chunk_number
        )

        # Try to find this file in the stats
        found_in_stats = False
        for stats_obj in all_stats:
            file_stat = stats_obj.find_file_statistics(actual_file_path)
            if file_stat is not None:
                result[file_format] = actual_file_path
                found_in_stats = True
                break

        # If not found in stats, check if the file exists in storage
        if not found_in_stats:
            sa = StorageAccess()
            if sa.check_if_file_exists(actual_file_path):
                logger.warning("File %s exists in storage but is not recorded in stats. This may indicate a discrepancy.", actual_file_path)
                result[file_format] = actual_file_path
            else:
                # File doesn't exist yet
                result[file_format] = None

    return result


def _construct_file_path_for_format(
    file_format: str,
    sync_unit: SyncUnit,
    chunk_number: int
) -> str:
    """
    Construct the full file path for a given file format, sync unit, and chunk number.

    Args:
        file_format: The file format (e.g., "parsed_items_{int}.pkl")
        sync_unit: The sync unit object
        chunk_number: The chunk number

    Returns:
        The full file path (e.g., "org/sys/proj/data_cleanup/parsed_items_5.pkl")
    """
    sync_unit_path = sync_unit.get_path_for_sync_unit()

    # Map file formats to their folders
    format_to_folder = {
        INITIAL_DATA_FILE_NAME_FORMAT: INITIAL_DATA_FOLDER_NAME,
        PARSED_DATA_FILE_NAME_FORMAT: DATA_CLEANUP_APP_FOLDER_NAME,
        CLEANED_ITEMS_FILE_NAME_FORMAT: CLEAN_DATA_FOLDER_NAME,
        CREATED_TIME_DELTA_FILE_NAME_FORMAT: DATA_PREPROCESSING_APP_FOLDER_NAME,
        MODIFIED_TIME_DELTA_FILE_NAME_FORMAT: DATA_PREPROCESSING_APP_FOLDER_NAME,
        MERGED_DATASET_FILE_NAME_FORMAT: DATA_PREPROCESSING_APP_FOLDER_NAME,
        MERGED_TIME_DELTA_FILE_NAME_FORMAT: DATA_PREPROCESSING_APP_FOLDER_NAME,
        MERGED_EVENTS_PER_MINUTE_FILE_NAME_FORMAT: DATA_PREPROCESSING_APP_FOLDER_NAME,
        ANALYSIS_PER_RECORD_FILE_NAME_FORMAT: PIPELINE_EXPLORER_APP_FOLDER_NAME
    }

    folder = format_to_folder.get(file_format)
    if folder is None:
        raise ValueError(f"Unknown file format: {file_format}")

    # Format the filename with the chunk number
    filename = file_format.format(int=chunk_number)

    # Construct the full path
    return join(sync_unit_path, folder, filename)


def validate_pipeline_file_stats(file_path: str, sync_unit: SyncUnit) -> dict[str, FileValidationResult] | None:
    """
    Validate that the input/output files in the stats match the expected pipeline relationships.

    This function:
    1. Gets the pipeline file paths for the given file
    2. Reads all app stats for the sync unit
    3. For each file in the pipeline, checks if the input_file and output_file in stats
       match what we expect from the pipeline graph

    Args:
        file_path: Full path or filename (e.g., "parsed_items_5.pkl")
        sync_unit: The sync unit object

    Returns:
        A dictionary mapping file formats to FileValidationResult objects.
        Returns None if the file format is not recognized.

    Example:
        validation = validate_pipeline_file_stats("parsed_items_5.pkl", sync_unit)
        for file_format, result in validation.items():
            if result.issues:
                print(f"{file_format}: {result.issues}")
            if result.input_matches:
                print(f"{file_format}: Input matches expected")
    """
    # Get the pipeline file paths
    pipeline_paths = get_pipeline_file_paths(file_path, sync_unit)
    if pipeline_paths is None:
        return None

    # Get the pipeline graph to understand relationships
    pipeline_graph = get_file_pipeline(file_path)
    if pipeline_graph is None:
        return None

    # Read all app stats
    all_stats = read_all_app_stats(sync_unit)

    # Create a flat list of all file statistics
    all_file_stats = []
    for stats_obj in all_stats:
        all_file_stats.extend(stats_obj.files)

    # Build a lookup dictionary for quick access
    stats_lookup = {fs.file_name: fs for fs in all_file_stats}

    # Validate each file in the pipeline
    validation_results = {}

    for file_format, actual_path in pipeline_paths.items():
        result = FileValidationResult(
            file_path=actual_path,
            exists=actual_path is not None
        )

        # If file doesn't exist, skip validation
        if actual_path is None:
            result.issues.append("File does not exist")
            validation_results[file_format] = result
            continue

        # Check if we have stats for this file
        file_stat = stats_lookup.get(actual_path)
        if file_stat is None:
            # Initial data files (input JSON files) are not tracked in stats files
            # This is expected behavior, not an issue
            if file_format != INITIAL_DATA_FILE_NAME_FORMAT:
                result.issues.append("No stats found for this file")
            validation_results[file_format] = result
            continue

        result.has_stats = True
        result.actual_input = file_stat.input_file
        # Store actual output as a list for consistency with expected_outputs
        if file_stat.output_file:
            result.actual_outputs = [file_stat.output_file]

        # Find expected input from the pipeline graph
        predecessors = list(pipeline_graph.predecessors(file_format))
        if predecessors:
            # Should have exactly one input
            if len(predecessors) > 1:
                result.issues.append(f"Multiple predecessors found: {predecessors}")

            expected_input_format = predecessors[0]
            expected_input_path = pipeline_paths.get(expected_input_format)
            result.expected_input = expected_input_path

            # Validate input matches
            if expected_input_path:
                result.input_matches = file_stat.input_file == expected_input_path
                if not result.input_matches:
                    result.issues.append(
                        f"Input mismatch: expected '{expected_input_path}', "
                        f"got '{file_stat.input_file}'"
                    )

        # Find expected outputs from the pipeline graph
        successors = list(pipeline_graph.successors(file_format))
        if successors:
            expected_output_paths = []
            for successor_format in successors:
                successor_path = pipeline_paths.get(successor_format)
                if successor_path:
                    expected_output_paths.append(successor_path)

            result.expected_outputs = expected_output_paths

            # Validate output matches (output_file should be the file itself)
            # The output_file field typically contains the file's own path
            result.output_matches = file_stat.output_file == actual_path
            if not result.output_matches:
                result.issues.append(
                    f"Output mismatch: expected '{actual_path}', "
                    f"got '{file_stat.output_file}'"
                )

        validation_results[file_format] = result

    return validation_results


def calculate_pipeline_metadata(sync_unit: SyncUnit) -> dict[str, PipelineMetadata]:
    """
    Calculate metadata for all pipeline files starting from initial data files.

    For each initial data file (work_X.json), this function:
    1. Finds all valid, existing output files in the pipeline
    2. Calculates metadata for each file:
       - Total input rows
       - Total output rows (for the file itself)
       - Unique IDs
       - Date columns with earliest and latest dates
    3. Reports missing files in the pipeline

    Args:
        sync_unit: The sync unit object

    Returns:
        A dictionary mapping initial file paths to PipelineMetadata objects

    Example:
        metadata = calculate_pipeline_metadata(sync_unit)
        for initial_file, pipeline_data in metadata.items():
            print(f"Pipeline for {initial_file}:")
            print(f"  Chunk: {pipeline_data.chunk_number}")
            print(f"  Missing files: {pipeline_data.missing_files}")
            for file_format, file_meta in pipeline_data.files.items():
                if file_meta.exists:
                    print(f"  {file_format}: {file_meta.total_rows} rows")
                    for date_col in file_meta.date_columns:
                        print(f"    {date_col.column_name}: {date_col.earliest_date} to {date_col.latest_date}")
    """
    sa = StorageAccess()

    # Get the initial data folder path
    initial_data_folder = join(sync_unit.get_path_for_sync_unit(), INITIAL_DATA_FOLDER_NAME)

    # List all work_*.json files in the initial data folder
    initial_files = sa.list_files_in_folder(initial_data_folder, "work_*.json")

    if not initial_files:
        return {}

    result = {}

    # Process each initial file
    for initial_file_path in initial_files:
        # Extract chunk number
        chunk_number = extract_chunk_number(initial_file_path)
        if chunk_number is None:
            continue

        # Get pipeline file paths
        pipeline_paths = get_pipeline_file_paths(initial_file_path, sync_unit)
        if pipeline_paths is None:
            continue

        # Create pipeline metadata object
        pipeline_metadata = PipelineMetadata(
            initial_file_path=initial_file_path,
            chunk_number=chunk_number
        )

        # Process each file in the pipeline
        for file_format, file_path in pipeline_paths.items():
            file_metadata = FileMetadata(
                file_path=file_path,
                exists=file_path is not None
            )

            if file_path is None:
                file_metadata.error = "File does not exist"
                pipeline_metadata.missing_files.append(file_format)
            else:
                # Calculate metadata for this file
                try:
                    _calculate_file_metadata(file_path, file_format, file_metadata, sa)
                except Exception as e:
                    file_metadata.error = str(e)

            pipeline_metadata.files[file_format] = file_metadata

        result[initial_file_path] = pipeline_metadata

    return result


def _calculate_file_metadata(
    file_path: str,
    file_format: str,
    file_metadata: FileMetadata,
    sa: StorageAccess
) -> None:
    """
    Calculate metadata for a single file and populate the FileMetadata object.

    Args:
        file_path: Path to the file
        file_format: File format (e.g., "parsed_items_{int}.pkl")
        file_metadata: FileMetadata object to populate
        sa: StorageAccess instance

    This function modifies file_metadata in place.
    """
    # Get expected columns for this file format
    expected_columns = file_format_to_columns_map.get(file_format)

    # Determine how to read the file based on format
    if file_path.endswith('.json'):
        # JSON files are original input files - no detailed metadata needed
        return
    elif file_path.endswith('.pkl'):
        # Read pickle file as dataframe
        df = sa.read_picklefile_as_dataframe(file_path)

        # Calculate total rows
        file_metadata.total_rows = len(df)

        # Calculate unique IDs (if id column exists)
        if expected_columns and expected_columns.id_column in df.columns:
            file_metadata.unique_ids = df[expected_columns.id_column].nunique()
        elif 'id' in df.columns:
            # Fallback to 'id' if no expected columns defined
            file_metadata.unique_ids = df['id'].nunique()

        # Process date columns based on expected columns
        if expected_columns and expected_columns.date_columns:
            for date_col_name in expected_columns.date_columns:
                if date_col_name in df.columns:
                    # Check if it's actually a datetime column
                    if pd.api.types.is_datetime64_any_dtype(df[date_col_name]):
                        # Get earliest and latest dates
                        earliest = df[date_col_name].min()
                        latest = df[date_col_name].max()

                        # Create DateColumnMetadata
                        date_col_metadata = DateColumnMetadata(
                            column_name=date_col_name,
                            earliest_date=earliest if pd.notna(earliest) else None,
                            latest_date=latest if pd.notna(latest) else None
                        )

                        file_metadata.date_columns.append(date_col_metadata)
        else:
            # Fallback: auto-detect date columns if no expected columns defined
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower() or 'events' in col.lower():
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        earliest = df[col].min()
                        latest = df[col].max()

                        date_col_metadata = DateColumnMetadata(
                            column_name=col,
                            earliest_date=earliest if pd.notna(earliest) else None,
                            latest_date=latest if pd.notna(latest) else None
                        )

                        file_metadata.date_columns.append(date_col_metadata)


def validate_training_readiness(sync_unit: SyncUnit) -> TrainingReadinessResult:
    """
    Validate that a sync unit is ready for model training.

    This function checks:
    1. Required files exist: merged_dataset, merged_time_delta, merged_events_per_minute
    2. All files were processed properly (using pipeline explorer validation)
    3. No gaps in processing (if files 1,3 exist but 2 is missing = ERROR)
    4. Trailing unprocessed files (if 1,2 processed but 3 not = WARNING)

    Args:
        sync_unit: The sync unit to validate

    Returns:
        TrainingReadinessResult with status, errors, warnings, and details
    """
    sa = StorageAccess()

    # Required file formats for training
    required_formats = [
        MERGED_DATASET_FILE_NAME_FORMAT,
        MERGED_TIME_DELTA_FILE_NAME_FORMAT,
        MERGED_EVENTS_PER_MINUTE_FILE_NAME_FORMAT,
    ]

    # Get the initial data folder path
    initial_data_folder = join(
        sync_unit.get_path_for_sync_unit(), INITIAL_DATA_FOLDER_NAME
    )

    # List all work_*.json files in the initial data folder
    initial_files = sa.list_files_in_folder(initial_data_folder, "work_*.json")

    if not initial_files:
        return TrainingReadinessResult(
            status=TrainingReadinessStatus.ERROR,
            errors=["No initial data files found in sync unit"],
        )

    # Extract chunk numbers from initial files and sort them
    chunk_numbers = []
    for initial_file in initial_files:
        chunk_num = extract_chunk_number(initial_file)
        if chunk_num is not None:
            chunk_numbers.append(chunk_num)

    chunk_numbers.sort()

    if not chunk_numbers:
        return TrainingReadinessResult(
            status=TrainingReadinessStatus.ERROR,
            errors=["Could not extract chunk numbers from initial files"],
        )

    # Track which chunks have all required formats
    chunk_details: dict[int, dict[str, bool]] = {}
    fully_processed_chunks = []
    partially_processed_chunks = []
    unprocessed_chunks = []

    for chunk_num in chunk_numbers:
        # Get pipeline paths for this chunk
        pipeline_paths = get_pipeline_file_paths(
            f"work_{chunk_num}.json", sync_unit
        )

        if pipeline_paths is None:
            chunk_details[chunk_num] = {fmt: False for fmt in required_formats}
            unprocessed_chunks.append(chunk_num)
            continue

        # Check which required formats exist
        format_exists = {}
        for fmt in required_formats:
            exists = pipeline_paths.get(fmt) is not None
            format_exists[fmt] = exists

        chunk_details[chunk_num] = format_exists

        # Classify this chunk
        all_exist = all(format_exists.values())
        any_exist = any(format_exists.values())

        if all_exist:
            fully_processed_chunks.append(chunk_num)
        elif any_exist:
            partially_processed_chunks.append(chunk_num)
        else:
            unprocessed_chunks.append(chunk_num)

    # Analyze for gaps vs trailing unprocessed
    # Gap: unprocessed chunk between processed chunks
    # Trailing: unprocessed chunks at the end (after all processed)
    errors = []
    warnings = []
    gap_chunks = []
    trailing_chunks = []
    missing_formats = []

    if fully_processed_chunks:
        max_processed = max(fully_processed_chunks)

        for chunk_num in unprocessed_chunks + partially_processed_chunks:
            if chunk_num < max_processed:
                gap_chunks.append(chunk_num)
            else:
                trailing_chunks.append(chunk_num)
    else:
        # No fully processed chunks
        missing_formats = required_formats.copy()
        errors.append("No files have been fully processed for training")

    # Generate error messages for gaps
    if gap_chunks:
        gap_chunks.sort()
        errors.append(
            f"Gap in processing: chunks {gap_chunks} are missing or incomplete "
            f"while later chunks have been processed"
        )

    # Check partially processed for specific missing formats
    for chunk_num in partially_processed_chunks:
        missing_in_chunk = [
            fmt for fmt, exists in chunk_details[chunk_num].items() if not exists
        ]
        if chunk_num in gap_chunks:
            errors.append(
                f"Chunk {chunk_num}: missing {missing_in_chunk}"
            )
        else:
            warnings.append(
                f"Chunk {chunk_num}: partially processed, missing {missing_in_chunk}"
            )

    # Generate warnings for trailing unprocessed
    if trailing_chunks:
        trailing_chunks.sort()
        warnings.append(
            f"New data available: chunks {trailing_chunks} have not been processed"
        )

    # Also validate that processed files have proper stats
    for chunk_num in fully_processed_chunks:
        validation = validate_pipeline_file_stats(f"work_{chunk_num}.json", sync_unit)
        if validation:
            for file_fmt, result in validation.items():
                if result.issues and file_fmt in required_formats:
                    warnings.append(
                        f"Chunk {chunk_num}, {file_fmt}: {result.issues}"
                    )

    # Determine overall status
    if gap_chunks or (not fully_processed_chunks and chunk_numbers):
        status = TrainingReadinessStatus.ERROR
    elif trailing_chunks or partially_processed_chunks:
        status = TrainingReadinessStatus.WARNING
    else:
        status = TrainingReadinessStatus.READY

    return TrainingReadinessResult(
        status=status,
        total_initial_files=len(chunk_numbers),
        processed_files=len(fully_processed_chunks),
        unprocessed_files=len(unprocessed_chunks) + len(partially_processed_chunks),
        errors=errors,
        warnings=warnings,
        missing_required_formats=missing_formats,
        gap_chunks=gap_chunks,
        trailing_chunks=trailing_chunks,
        chunk_details=chunk_details,
    )
