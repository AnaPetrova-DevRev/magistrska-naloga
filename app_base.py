"""
Base interface for all apps in the orchestrator system.

Each app must implement this interface to be registered and used by the orchestrator.
"""
from abc import ABC, abstractmethod
from orchestrator_message import OrchestratorResponse
import sync_unit


class App(ABC):
    """
    Base class for all apps that can be invoked by the orchestrator.

    Each app must implement:
    - get_name(): Return the unique name of the app
    - get_available_actions(): Return list of actions this app can perform
    - execute_action(): Execute a specific action on a sync unit
    """

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the unique name of this app.

        Returns:
            str: The app name (e.g., "data_cleanup", "data_sync", etc.)
        """
        pass

    @abstractmethod
    def get_available_actions(self) -> list[str]:
        """
        Get the list of actions this app can perform.

        Returns:
            list[str]: List of action names (e.g., ["parse_json", "cleanup_data"])
        """
        pass

    @abstractmethod
    def execute_action(
        self,
        sync_unit_obj: sync_unit.SyncUnit,
        action: str,
        parameters: dict = None
    ) -> OrchestratorResponse:
        """
        Execute a specific action on a sync unit.

        Args:
            sync_unit_obj: The sync unit to operate on
            action: The action to perform (must be in get_available_actions())
            parameters: Optional parameters for the action

        Returns:
            OrchestratorResponse: Response indicating success/failure and any logs

        Raises:
            ValueError: If the action is not supported by this app
        """
        pass

    def supports_action(self, action: str) -> bool:
        """
        Check if this app supports a specific action.

        Args:
            action: The action name to check

        Returns:
            bool: True if the action is supported, False otherwise
        """
        return action in self.get_available_actions()
