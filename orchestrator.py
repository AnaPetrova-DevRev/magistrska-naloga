


from datetime import datetime
import logging
import logging_config

from orchestrator_message import OrchestratorMessage, OrchestratorResponse
from app_registry import AppRegistry

import storage_access
import global_vars


logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, app_registry: AppRegistry, enable_sync_unit_logging: bool = True):
        """
        Initialize the Orchestrator.

        Args:
            app_registry: The app registry containing all registered apps
            enable_sync_unit_logging: If True, creates separate log files for each sync unit.
                                     If False, uses the default logging configuration.
        """
        self.app_registry = app_registry
        self.enable_sync_unit_logging = enable_sync_unit_logging
        logger.info("Orchestrator initialized with %d registered apps", len(app_registry.get_app_names()))

    def send_message(self, message: OrchestratorMessage) -> OrchestratorResponse:
        """
        Send a message to the appropriate app based on the action.

        Args:
            message: The orchestrator message containing sync unit, action, and parameters

        Returns:
            OrchestratorResponse: Response from the app execution
        """
        sync_unit_obj = message.get_sync_unit()

        if self.enable_sync_unit_logging:
            log_file = logging_config.setup_sync_unit_logging(
                sync_unit_obj.get_path_for_sync_unit(),
                level="INFO",
                log_to_console=True
            )
            logger.info("=" * 80)
            logger.info("Processing sync unit: %s", sync_unit_obj.get_path_for_sync_unit())
            logger.info("Log file: %s", log_file)
            logger.info("=" * 80)

        logger.info("Received message: %s", message.to_string())

        app = self._find_app_for_message(message)

        if app is None:
            error_msg = f"Action '{message.action}' not supported by any registered app"
            logger.error(error_msg)
            response = OrchestratorResponse(
                success=False,
                message=error_msg,
                is_done=True
            )
        else:
            try:
                response = app.execute_action(
                    sync_unit_obj=sync_unit_obj,
                    action=message.action,
                    parameters=message.parameters
                )
                logger.info("Response: %s", response.to_string())
            except Exception as e:
                error_msg = f"Error executing action '{message.action}': {str(e)}"
                logger.error(error_msg, exc_info=True)
                response = OrchestratorResponse(
                    success=False,
                    message=error_msg,
                    is_done=True
                )

        self.save_operations_logs(sync_unit_obj, message, response)

        return response

    def _find_app_for_message(self, message: OrchestratorMessage):
        """
        Find the appropriate app for a message.

        For most actions, this simply finds the app that supports the action.
        For 'reset_state', it looks at the 'action_name' parameter to find
        which app owns that action, then routes to that app.

        Args:
            message: The orchestrator message

        Returns:
            App instance or None if not found
        """
        from app_actions_constants import RESET_STATE

        if message.action == RESET_STATE:
            # For reset_state, route based on the action_name parameter
            action_name = message.parameters.get("action_name") if message.parameters else None
            if action_name:
                app = self.app_registry.find_app_for_action(action_name)
                if app and app.supports_action(RESET_STATE):
                    return app
                # If app doesn't support reset_state, fall through to normal lookup

        return self.app_registry.find_app_for_action(message.action)

    def get_available_apps(self) -> list[str]:
        """Get list of all available app names."""
        return self.app_registry.get_app_names()

    def get_actions_for_app(self, app_name: str) -> list[str]:
        """Get list of actions for a specific app."""
        return self.app_registry.get_actions_for_app(app_name)

    def save_operations_logs(self, sync_unit_obj, message: OrchestratorMessage, response: OrchestratorResponse) -> None:
        """Save operation logs to storage"""
        sa = storage_access.StorageAccess()
        log_entry = {
            "sync_unit": sync_unit_obj.to_dict(),
            "message": message.to_dict(),
            "response": response.to_dict()
        }
        orchestrator_folder_path = f"{sync_unit_obj.get_path_for_sync_unit()}/{global_vars.ORCHESTRATOR_FOLDER_NAME}"
        if not sa.check_if_file_exists(orchestrator_folder_path):
            sa.create_folder(orchestrator_folder_path)

        file_path = f"{orchestrator_folder_path}/{global_vars.ORCHESTRATOR_LOGS_FORMAT.format(string=datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))}"
        sa.write_json_to_storage(file_path, log_entry)