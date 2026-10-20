
# Calculate the created date delta and modified date delta columns (delta between current and previous row for both created and modified dates)
# Create a merged dataset where the created and modified dates are sorted in one column
# Take the merged dataset and create a new column that has the delta between the current row and the previous row.

# The input for this app are the cleaned files from the data_cleanup app.

import logging
import data_preprocessing.invoke_data_preprocesser as data_preprocesser
from app_base import App
from orchestrator_message import OrchestratorMessage, OrchestratorResponse
import sync_unit
from app_actions_constants import CALCULATE_ORIGINAL_TIME_DELTAS, CREATE_MERGED_DATASET, CALCULATE_MERGED_TIME_DELTAS, CALCULATE_MERGED_EVENTS_PER_MINUTE, RESET_STATE


logger = logging.getLogger(__name__)

class DataPreprocessingApp(App):
    """
    Data preprocessing app that handles calculating time deltas and creating merged datasets.

    Available actions:
    - calculate_original_time_deltas: Calculate time deltas for original created and modified dates
    - create_merged_dataset: Create a merged dataset with sorted timestamps
    - calculate_merged_time_deltas: Calculate time deltas for the merged dataset
    """

    def get_name(self) -> str:
        """Return the app name."""
        return "data_preprocessing"

    def get_available_actions(self) -> list[str]:
        """Return list of available actions."""
        return [CALCULATE_ORIGINAL_TIME_DELTAS, CREATE_MERGED_DATASET, CALCULATE_MERGED_TIME_DELTAS, CALCULATE_MERGED_EVENTS_PER_MINUTE, RESET_STATE]

    def execute_action(
        self,
        sync_unit_obj: sync_unit.SyncUnit,
        action: str,
        parameters: dict = None
    ) -> OrchestratorResponse:
        """
        Execute a data preprocessing action.

        Args:
            sync_unit_obj: The sync unit to operate on
            action: The action to perform
            parameters: Optional parameters for the action
                       For reset_state: {"action_name": action to reset}

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

        message = OrchestratorMessage(
            sync_unit_obj=sync_unit_obj,
            action=action,
            parameters=parameters
        )

        logger.info(
            "Executing action '%s' on sync unit '%s'",
            action,
            sync_unit_obj.get_name()
        )

        response = data_preprocesser.receive_message(message)

        return response