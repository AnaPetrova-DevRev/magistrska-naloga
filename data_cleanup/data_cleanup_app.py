"""
Data Cleanup App - implements the App interface for data cleaning operations.

This app wraps the existing data_cleanup module and provides a standardized
interface for the orchestrator to invoke data cleaning actions.
"""
import logging
from app_base import App
from orchestrator_message import OrchestratorMessage, OrchestratorResponse
import sync_unit
import data_cleanup.invoke_data_cleaner as data_cleaner
from app_actions_constants import PARSE_JSON, CLEANUP_DATA, RESET_STATE

logger = logging.getLogger(__name__)

class DataCleanupApp(App):
    """
    Data cleanup app that handles parsing and cleaning of data files.

    Available actions:
    - parse_json: Parse JSON files and extract structured data
    - cleanup_data: Clean and deduplicate data (future implementation)
    """

    def get_name(self) -> str:
        """Return the app name."""
        return "data_cleanup"

    def get_available_actions(self) -> list[str]:
        """Return list of available actions."""
        return [PARSE_JSON, CLEANUP_DATA, RESET_STATE]

    def execute_action(
        self,
        sync_unit_obj: sync_unit.SyncUnit,
        action: str,
        parameters: dict = None
    ) -> OrchestratorResponse:
        """
        Execute a data cleanup action.

        Args:
            sync_unit_obj: The sync unit to operate on
            action: The action to perform ("parse_json", "clean_data", or "reset_state")
            parameters: Optional parameters for the action
                       For reset_state: {"action_name": "parse_json" or "clean_data"}

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

        response = data_cleaner.receive_message(message)

        return response
