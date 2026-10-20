"""
App Registry for managing all registered apps in the orchestrator system.

The registry maintains a collection of all available apps and provides
methods to query apps and their capabilities.
"""
import logging
from typing import Optional
from app_base import App

logger = logging.getLogger(__name__)


class AppRegistry:
    """
    Central registry for all apps that can be invoked by the orchestrator.

    Apps must be registered before they can be used. The registry provides
    methods to:
    - Register new apps
    - Get all registered apps
    - Get a specific app by name
    - Get available actions for an app
    - Find which app supports a specific action
    """

    def __init__(self):
        """Initialize an empty app registry."""
        self._apps: dict[str, App] = {}
        logger.info("App registry initialized")

    def register_app(self, app: App) -> None:
        """
        Register a new app in the registry.

        Args:
            app: The app instance to register

        Raises:
            ValueError: If an app with the same name is already registered
        """
        app_name = app.get_name()

        if app_name in self._apps:
            raise ValueError(f"App '{app_name}' is already registered")

        self._apps[app_name] = app
        logger.info(
            "Registered app '%s' with actions: %s",
            app_name,
            app.get_available_actions()
        )

    def get_all_apps(self) -> dict[str, App]:
        """
        Get all registered apps.

        Returns:
            dict[str, App]: Dictionary mapping app names to app instances
        """
        return self._apps.copy()

    def get_app(self, app_name: str) -> Optional[App]:
        """
        Get a specific app by name.

        Args:
            app_name: The name of the app to retrieve

        Returns:
            App instance if found, None otherwise
        """
        return self._apps.get(app_name)

    def get_app_names(self) -> list[str]:
        """
        Get the names of all registered apps.

        Returns:
            list[str]: List of app names
        """
        return list(self._apps.keys())

    def get_actions_for_app(self, app_name: str) -> list[str]:
        """
        Get the available actions for a specific app.

        Args:
            app_name: The name of the app

        Returns:
            list[str]: List of action names, or empty list if app not found
        """
        app = self.get_app(app_name)
        if app is None:
            logger.warning("App '%s' not found in registry", app_name)
            return []

        return app.get_available_actions()

    def find_app_for_action(self, action: str) -> Optional[App]:
        """
        Find which app supports a specific action.

        Args:
            action: The action name to search for

        Returns:
            App instance that supports the action, or None if not found
        """
        for app in self._apps.values():
            if app.supports_action(action):
                return app

        logger.warning("No app found that supports action '%s'", action)
        return None

    def get_all_actions(self) -> dict[str, list[str]]:
        """
        Get all actions grouped by app.

        Returns:
            dict[str, list[str]]: Dictionary mapping app names to their actions
        """
        return {
            app_name: app.get_available_actions()
            for app_name, app in self._apps.items()
        }

    def is_action_supported(self, action: str) -> bool:
        """
        Check if any registered app supports the given action.

        Args:
            action: The action name to check

        Returns:
            bool: True if any app supports the action, False otherwise
        """
        return self.find_app_for_action(action) is not None
