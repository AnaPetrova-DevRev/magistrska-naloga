"""
Pipeline Explorer App - implements the App interface for pipeline exploration operations.

This app provides functionality to calculate and save pipeline metadata and validation
results for a sync unit. While primarily used as a visualization tool through the GUI,
it can also be invoked through the orchestrator for batch processing.
"""
import logging
from dataclasses import dataclass, field, asdict

from app_base import App
from orchestrator_message import OrchestratorResponse
from sync_unit import SyncUnit
from app_actions_constants import (
    CALCULATE_ALL_PIPELINE_METADATA,
    VALIDATE_PIPELINE,
    ANALYSIS_PER_RECORD,
    RESET_STATE
)
from global_vars import (
    PIPELINE_METADATA_FILE_NAME,
    PIPELINE_EXPLORER_APP_FOLDER_NAME,
    PIPELINE_EXPLORER_STATE_FILE_NAME,
    PIPELINE_EXPLORER_STATS_FILE_NAME
)
from app_flow_definitions import validate_training_readiness
from file_metadata import TrainingReadinessStatus
import state
import storage_access
import file_stats

from .pipeline_data import (
    get_combined_pipeline_data,
    save_pipeline_data_to_storage
)
from .analysis_per_record import (
    calculate_analysis_per_record,
    save_analysis_results
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineExplorerState:
    """Manages separate states for different pipeline explorer functions"""
    calculate_all_pipeline_metadata_state: state.State = field(
        default_factory=state.State
    )
    validate_pipeline_state: state.State = field(default_factory=state.State)
    analysis_per_record_state: state.State = field(default_factory=state.State)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "calculate_all_pipeline_metadata_state": asdict(
                self.calculate_all_pipeline_metadata_state
            ),
            "validate_pipeline_state": asdict(self.validate_pipeline_state),
            "analysis_per_record_state": asdict(self.analysis_per_record_state)
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PipelineExplorerState':
        """Create a PipelineExplorerState object from a dictionary"""
        obj = cls()
        key = "calculate_all_pipeline_metadata_state"
        if key in data and data[key]:
            obj.calculate_all_pipeline_metadata_state = state.State.from_dict(
                data[key]
            )
        if "validate_pipeline_state" in data and data["validate_pipeline_state"]:
            obj.validate_pipeline_state = state.State.from_dict(
                data["validate_pipeline_state"]
            )
        if "analysis_per_record_state" in data and data["analysis_per_record_state"]:
            obj.analysis_per_record_state = state.State.from_dict(
                data["analysis_per_record_state"]
            )
        return obj

    def get_state_for_function(self, function_name: str) -> state.State:
        """Get the state for a specific function"""
        if function_name == CALCULATE_ALL_PIPELINE_METADATA:
            return self.calculate_all_pipeline_metadata_state
        elif function_name == VALIDATE_PIPELINE:
            return self.validate_pipeline_state
        elif function_name == ANALYSIS_PER_RECORD:
            return self.analysis_per_record_state
        else:
            raise ValueError(f"Function {function_name} not supported in pipeline explorer state")

    def update_state_for_function(self, function_name: str, new_state: state.State) -> None:
        """Update the state for a specific function"""
        if function_name == CALCULATE_ALL_PIPELINE_METADATA:
            self.calculate_all_pipeline_metadata_state = new_state
        elif function_name == VALIDATE_PIPELINE:
            self.validate_pipeline_state = new_state
        elif function_name == ANALYSIS_PER_RECORD:
            self.analysis_per_record_state = new_state
        else:
            raise ValueError(f"Function {function_name} not supported in pipeline explorer state")


def get_stats_and_state(
    sa: storage_access.StorageAccess,
    sync_unit_obj: SyncUnit
) -> tuple[file_stats.Stats, PipelineExplorerState]:
    """Load file statistics and state for a sync unit"""
    sync_unit_path = sync_unit_obj.get_path_for_sync_unit()
    app_path = f"{sync_unit_path}/{PIPELINE_EXPLORER_APP_FOLDER_NAME}/"
    state_path = f"{app_path}{PIPELINE_EXPLORER_STATE_FILE_NAME}"
    stats_path = f"{app_path}{PIPELINE_EXPLORER_STATS_FILE_NAME}"

    stats_exists = sa.check_if_file_exists(stats_path)
    if stats_exists:
        logger.info("File statistics exists for pipeline explorer, loading it")
        app_file_stats = sa.read_json_as_object(
            stats_path, file_stats.Stats.from_dict
        )
    else:
        logger.info(
            "File statistics does not exist for pipeline explorer, creating new one"
        )
        app_file_stats = file_stats.Stats()

    # Read state using the generic storage access method
    app_state = sa.read_json_as_object(state_path, PipelineExplorerState.from_dict)
    if app_state is None:
        logger.info("State does not exist for pipeline explorer, creating new one")
        app_state = PipelineExplorerState()

    return app_file_stats, app_state


def save_stats_and_state(
    sa: storage_access.StorageAccess,
    app_file_stats: file_stats.Stats,
    app_state: PipelineExplorerState,
    sync_unit_obj: SyncUnit
) -> None:
    """Save file statistics and state for a sync unit"""
    sync_unit_path = sync_unit_obj.get_path_for_sync_unit()
    app_path = f"{sync_unit_path}/{PIPELINE_EXPLORER_APP_FOLDER_NAME}/"
    state_path = f"{app_path}{PIPELINE_EXPLORER_STATE_FILE_NAME}"
    stats_path = f"{app_path}{PIPELINE_EXPLORER_STATS_FILE_NAME}"

    # Ensure the app folder exists
    if not sa.check_if_file_exists(app_path):
        sa.create_folder(app_path)

    sa.save_file_stats(app_file_stats, stats_path)
    sa.save_state(app_state.to_dict(), state_path)
    logger.info("Saved file statistics and state for pipeline explorer")


class PipelineExplorerApp(App):
    """
    Pipeline Explorer app for calculating and visualizing pipeline metadata.

    Available actions:
    - calculate_all_pipeline_metadata: Calculate and save metadata and validation
      results for all pipelines in a sync unit
    - validate_pipeline: Validate that a sync unit is ready for model training
    - reset_state: Reset state for a specific action
    """

    def get_name(self) -> str:
        """Return the app name."""
        return "pipeline_explorer"

    def get_available_actions(self) -> list[str]:
        """Return list of available actions."""
        return [
            CALCULATE_ALL_PIPELINE_METADATA,
            VALIDATE_PIPELINE,
            ANALYSIS_PER_RECORD,
            RESET_STATE
        ]

    def execute_action(
        self,
        sync_unit_obj: SyncUnit,
        action: str,
        parameters: dict = None
    ) -> OrchestratorResponse:
        """
        Execute a pipeline explorer action.

        Args:
            sync_unit_obj: The sync unit to operate on
            action: The action to perform
            parameters: Optional parameters for the action

        Returns:
            OrchestratorResponse: Response with success status and logs

        Raises:
            ValueError: If the action is not supported
        """
        if not self.supports_action(action):
            error_msg = f"Action '{action}' not supported by {self.get_name()}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if parameters is None:
            parameters = {}

        logger.info(
            "Executing action '%s' on sync unit '%s'",
            action,
            sync_unit_obj.get_path_for_sync_unit()
        )

        if action == CALCULATE_ALL_PIPELINE_METADATA:
            return self._calculate_all_metadata(sync_unit_obj, parameters)
        if action == VALIDATE_PIPELINE:
            return self._validate_training_readiness(sync_unit_obj, parameters)
        if action == ANALYSIS_PER_RECORD:
            return self._analysis_per_record(sync_unit_obj, parameters)
        if action == RESET_STATE:
            return self._reset_state(sync_unit_obj, parameters)

        # Should not reach here due to supports_action check
        return OrchestratorResponse(
            success=False,
            message=f"Unknown action: {action}",
            logs=[]
        )

    def _calculate_all_metadata(
        self,
        sync_unit_obj: SyncUnit,
        parameters: dict  # pylint: disable=unused-argument
    ) -> OrchestratorResponse:
        """
        Calculate and save all pipeline metadata and validation for a sync unit.

        This calculates metadata for all pipelines starting from initial data files,
        combines it with validation results, and saves to storage.

        Args:
            sync_unit_obj: The sync unit to process
            parameters: Optional parameters (reserved for future use)

        Returns:
            OrchestratorResponse with success status
        """
        logs = []

        try:
            # Get combined pipeline data
            sync_unit_path = sync_unit_obj.get_path_for_sync_unit()
            logs.append(f"Calculating pipeline metadata for {sync_unit_path}")
            pipelines = get_combined_pipeline_data(sync_unit_obj)

            if not pipelines:
                logs.append("No pipelines found for this sync unit")
                return OrchestratorResponse(
                    success=True,
                    message="No pipelines found",
                    logs=logs
                )

            logs.append(f"Found {len(pipelines)} pipeline(s)")

            # Count errors
            error_count = sum(1 for p in pipelines if p.has_errors())
            if error_count > 0:
                logs.append(f"Found {error_count} pipeline(s) with errors/issues")

            # Save to storage (using orchestrator filename, not cache)
            save_pipeline_data_to_storage(
                sync_unit_obj,
                pipelines,
                use_cache_filename=False
            )

            logs.append(f"Saved pipeline metadata to {PIPELINE_METADATA_FILE_NAME}")

            return OrchestratorResponse(
                success=True,
                message=f"Calculated metadata for {len(pipelines)} pipeline(s)",
                logs=logs
            )

        except Exception as e:
            error_msg = f"Error calculating pipeline metadata: {str(e)}"
            logger.exception(error_msg)
            logs.append(error_msg)
            return OrchestratorResponse(
                success=False,
                message=error_msg,
                logs=logs
            )

    def _validate_training_readiness(
        self,
        sync_unit_obj: SyncUnit,
        parameters: dict  # pylint: disable=unused-argument
    ) -> OrchestratorResponse:
        """
        Validate that a sync unit is ready for model training.

        Checks for:
        - Required files: merged_dataset, merged_time_delta, merged_events_per_minute
        - No gaps in processing (missing chunks between processed ones = ERROR)
        - Trailing unprocessed files (new data available = WARNING)

        Args:
            sync_unit_obj: The sync unit to validate
            parameters: Optional parameters (reserved for future use)

        Returns:
            OrchestratorResponse with:
            - success=True if status is READY or WARNING
            - success=False if status is ERROR
            - data field containing the TrainingReadinessResult as dict
        """
        logs = []

        try:
            sync_unit_path = sync_unit_obj.get_path_for_sync_unit()
            logs.append(f"Validating training readiness for {sync_unit_path}")

            # Run the validation
            result = validate_training_readiness(sync_unit_obj)

            # Log the status
            logs.append(f"Status: {result.status.value.upper()}")
            logs.append(
                f"Files: {result.processed_files} processed, "
                f"{result.unprocessed_files} unprocessed"
            )

            # Log errors
            for error in result.errors:
                logs.append(f"ERROR: {error}")

            # Log warnings
            for warning in result.warnings:
                logs.append(f"WARNING: {warning}")

            # Determine success based on status
            success = result.status != TrainingReadinessStatus.ERROR

            return OrchestratorResponse(
                success=success,
                message=result.get_message(),
                logs=logs,
                data=result.to_dict()
            )

        except Exception as e:
            error_msg = f"Error validating training readiness: {str(e)}"
            logger.exception(error_msg)
            logs.append(error_msg)
            return OrchestratorResponse(
                success=False,
                message=error_msg,
                logs=logs
            )


    def _analysis_per_record(
        self,
        sync_unit_obj: SyncUnit,
        parameters: dict  # pylint: disable=unused-argument
    ) -> OrchestratorResponse:
        """
        Perform analysis per record for a sync unit.

        Reads all clean data pickle files, combines them, and calculates
        per-record statistics including:
        - Date statistics (first created, first modified, last modified)
        - Day of week distribution
        - Hour of day distribution
        - Time delta statistics between modifications
        - Delta bucket categorization

        Args:
            sync_unit_obj: The sync unit to analyze
            parameters: Optional parameters for the analysis

        Returns:
            OrchestratorResponse with success status and logs
        """
        logs = []
        sync_unit_path = sync_unit_obj.get_path_for_sync_unit()
        logs.append(f"Starting analysis per record for {sync_unit_path}")

        try:
            # Calculate analysis per record
            result_df = calculate_analysis_per_record(sync_unit_obj)

            if result_df is None or len(result_df) == 0:
                logs.append("No data available for analysis")
                return OrchestratorResponse(
                    success=False,
                    message="No clean data found for analysis",
                    logs=logs
                )

            logs.append(f"Calculated statistics for {len(result_df)} unique records")
            logs.append(f"Result has {len(result_df.columns)} columns")

            # Save results
            output_path = save_analysis_results(sync_unit_obj, result_df)
            logs.append(f"Saved analysis results to {output_path}")

            return OrchestratorResponse(
                success=True,
                message=f"Analysis complete for {len(result_df)} records",
                logs=logs,
                is_done=True
            )

        except Exception as e:  # pylint: disable=broad-except
            error_msg = f"Error during analysis per record: {str(e)}"
            logs.append(error_msg)
            logger.exception(error_msg)
            return OrchestratorResponse(
                success=False,
                message=error_msg,
                logs=logs
            )

    def _reset_state(
        self,
        sync_unit_obj: SyncUnit,
        parameters: dict
    ) -> OrchestratorResponse:
        """
        Reset state for a specific action in the pipeline explorer.

        Args:
            sync_unit_obj: The sync unit to operate on
            parameters: Must contain 'action_name' key with the action to reset

        Returns:
            OrchestratorResponse with success status
        """
        logs = []
        action_name = parameters.get("action_name")

        if not action_name:
            return OrchestratorResponse(
                success=False,
                message="Parameter 'action_name' is required for reset_state action",
                is_done=True,
                logs=["Missing required parameter 'action_name'"]
            )

        valid_actions = [
            CALCULATE_ALL_PIPELINE_METADATA,
            VALIDATE_PIPELINE,
            ANALYSIS_PER_RECORD
        ]
        if action_name not in valid_actions:
            return OrchestratorResponse(
                success=False,
                message=f"Action '{action_name}' is not a valid action for "
                        f"pipeline_explorer app",
                is_done=True,
                logs=[f"Invalid action_name: {action_name}"]
            )

        try:
            sa = storage_access.StorageAccess()
            app_file_stats, app_state = get_stats_and_state(sa, sync_unit_obj)

            # Reset the state for the specified action
            new_state = state.State()
            app_state.update_state_for_function(action_name, new_state)
            save_stats_and_state(sa, app_file_stats, app_state, sync_unit_obj)

            logger.info(
                "Reset state for action '%s' in pipeline_explorer app",
                action_name
            )
            logs.append(f"Successfully reset state for action '{action_name}'")

            return OrchestratorResponse(
                success=True,
                message=f"Successfully reset state for action '{action_name}'",
                is_done=True,
                logs=logs
            )

        except ValueError as e:
            error_msg = f"Error resetting state: {str(e)}"
            logger.error(error_msg)
            return OrchestratorResponse(
                success=False,
                message=error_msg,
                is_done=True,
                logs=[error_msg]
            )
