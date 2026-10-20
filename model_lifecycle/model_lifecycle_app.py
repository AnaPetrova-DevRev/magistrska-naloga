"""Model lifecycle management app for the orchestrator system.

This app handles model versioning, registration, promotion, and rollback
via MLflow Model Registry. It provides actions for managing the full
lifecycle of trained models.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app_base import App
from orchestrator_message import OrchestratorResponse
import sync_unit as sync_unit_module
from app_actions_constants import (
    LIST_CANDIDATE_RUNS,
    REGISTER_RUN_AS_MODEL,
    SET_MODEL_STAGE,
    ROLLBACK_MODEL,
    GET_CURRENT_MODELS,
)
from modeling.core_types import TaskType
from model_lifecycle.mlflow_helpers import (
    list_candidate_runs,
    register_run_as_model,
    set_model_stage,
    rollback_model_to_version,
    get_current_models,
)
from modeling.mlflow_utils import is_mlflow_available, get_registered_model_name

logger = logging.getLogger(__name__)


class ModelLifecycleApp(App):
    """App for managing model lifecycle via MLflow Model Registry.

    Actions:
        list_candidate_runs: List recent training runs for selection.
        register_run_as_model: Register a run as a model version.
        set_model_stage: Promote/demote a model version.
        rollback_model: Rollback to a previous model version.
        get_current_models: Get current Production/Staging models.

    All actions require MLflow to be available and configured.
    """

    def get_name(self) -> str:
        return "model_lifecycle"

    def get_available_actions(self) -> list[str]:
        return [
            LIST_CANDIDATE_RUNS,
            REGISTER_RUN_AS_MODEL,
            SET_MODEL_STAGE,
            ROLLBACK_MODEL,
            GET_CURRENT_MODELS,
        ]

    def execute_action(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        action: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorResponse:
        """Execute a model lifecycle action.

        Args:
            sync_unit_obj: The sync unit.
            action: The action to perform.
            parameters: Action-specific parameters.

        Returns:
            OrchestratorResponse with results.
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
            sync_unit_obj.get_name(),
        )

        if not is_mlflow_available():
            return OrchestratorResponse(
                success=False,
                message="MLflow is not available. Install mlflow to use lifecycle management.",
                is_done=True,
            )

        if action == LIST_CANDIDATE_RUNS:
            return self._handle_list_candidate_runs(sync_unit_obj, parameters)
        elif action == REGISTER_RUN_AS_MODEL:
            return self._handle_register_run(sync_unit_obj, parameters)
        elif action == SET_MODEL_STAGE:
            return self._handle_set_model_stage(sync_unit_obj, parameters)
        elif action == ROLLBACK_MODEL:
            return self._handle_rollback(sync_unit_obj, parameters)
        elif action == GET_CURRENT_MODELS:
            return self._handle_get_current_models(sync_unit_obj, parameters)

        raise ValueError(f"Unknown action: {action}")

    def _handle_list_candidate_runs(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        parameters: Dict[str, Any],
    ) -> OrchestratorResponse:
        """Handle list_candidate_runs action.

        Parameters:
            task_type: "event_rate" or "time_to_event"
            top_k: Maximum number of runs to return (default 10)
            order_by: Metric to sort by (default "val_mae")
            ascending: Sort ascending (default True)
        """
        task_type_str = parameters.get("task_type", "event_rate")
        top_k = int(parameters.get("top_k", 10))
        order_by = parameters.get("order_by", "val_mae")
        ascending = parameters.get("ascending", True)

        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            return OrchestratorResponse(
                success=False,
                message=f"Invalid task_type: {task_type_str}",
                is_done=True,
            )

        try:
            runs = list_candidate_runs(
                sync_unit_name=sync_unit_obj.get_name(),
                task_type=task_type,
                top_k=top_k,
                order_by=order_by,
                ascending=ascending,
            )

            response = OrchestratorResponse(
                success=True,
                message=f"Found {len(runs)} candidate runs",
                is_done=True,
            )
            response.data = {"runs": runs}
            return response

        except Exception as e:
            logger.error("Failed to list candidate runs: %s", e, exc_info=True)
            return OrchestratorResponse(
                success=False,
                message=f"Failed to list candidate runs: {e}",
                is_done=True,
            )

    def _handle_register_run(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        parameters: Dict[str, Any],
    ) -> OrchestratorResponse:
        """Handle register_run_as_model action.

        Parameters:
            task_type: "event_rate" or "time_to_event"
            run_id: MLflow run ID to register
            initial_stage: Initial stage ("None", "Staging", "Production")
        """
        task_type_str = parameters.get("task_type", "event_rate")
        run_id = parameters.get("run_id")
        initial_stage = parameters.get("initial_stage", "None")

        if not run_id:
            return OrchestratorResponse(
                success=False,
                message="run_id is required",
                is_done=True,
            )

        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            return OrchestratorResponse(
                success=False,
                message=f"Invalid task_type: {task_type_str}",
                is_done=True,
            )

        try:
            result = register_run_as_model(
                sync_unit_name=sync_unit_obj.get_name(),
                task_type=task_type,
                run_id=run_id,
                initial_stage=initial_stage,
            )

            if result:
                response = OrchestratorResponse(
                    success=True,
                    message=f"Registered model version {result['version']}",
                    is_done=True,
                )
                response.data = result
                return response
            else:
                return OrchestratorResponse(
                    success=False,
                    message="Failed to register model",
                    is_done=True,
                )

        except Exception as e:
            logger.error("Failed to register run: %s", e, exc_info=True)
            return OrchestratorResponse(
                success=False,
                message=f"Failed to register run: {e}",
                is_done=True,
            )

    def _handle_set_model_stage(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        parameters: Dict[str, Any],
    ) -> OrchestratorResponse:
        """Handle set_model_stage action.

        Parameters:
            task_type: "event_rate" or "time_to_event"
            version: Model version number
            new_stage: Target stage ("None", "Staging", "Production", "Archived")
        """
        task_type_str = parameters.get("task_type", "event_rate")
        version = parameters.get("version")
        new_stage = parameters.get("new_stage")

        if version is None:
            return OrchestratorResponse(
                success=False,
                message="version is required",
                is_done=True,
            )

        if not new_stage:
            return OrchestratorResponse(
                success=False,
                message="new_stage is required",
                is_done=True,
            )

        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            return OrchestratorResponse(
                success=False,
                message=f"Invalid task_type: {task_type_str}",
                is_done=True,
            )

        try:
            success = set_model_stage(
                sync_unit_name=sync_unit_obj.get_name(),
                task_type=task_type,
                version=int(version),
                new_stage=new_stage,
            )

            if success:
                return OrchestratorResponse(
                    success=True,
                    message=f"Set version {version} to stage {new_stage}",
                    is_done=True,
                )
            else:
                return OrchestratorResponse(
                    success=False,
                    message="Failed to set model stage",
                    is_done=True,
                )

        except Exception as e:
            logger.error("Failed to set model stage: %s", e, exc_info=True)
            return OrchestratorResponse(
                success=False,
                message=f"Failed to set model stage: {e}",
                is_done=True,
            )

    def _handle_rollback(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        parameters: Dict[str, Any],
    ) -> OrchestratorResponse:
        """Handle rollback_model action.

        Parameters:
            task_type: "event_rate" or "time_to_event"
            target_version: Version to rollback to
        """
        task_type_str = parameters.get("task_type", "event_rate")
        target_version = parameters.get("target_version")

        if target_version is None:
            return OrchestratorResponse(
                success=False,
                message="target_version is required",
                is_done=True,
            )

        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            return OrchestratorResponse(
                success=False,
                message=f"Invalid task_type: {task_type_str}",
                is_done=True,
            )

        try:
            result = rollback_model_to_version(
                sync_unit_name=sync_unit_obj.get_name(),
                task_type=task_type,
                target_version=int(target_version),
            )

            if result:
                response = OrchestratorResponse(
                    success=True,
                    message=f"Rolled back to version {target_version}",
                    is_done=True,
                )
                response.data = result
                return response
            else:
                return OrchestratorResponse(
                    success=False,
                    message="Failed to rollback model",
                    is_done=True,
                )

        except Exception as e:
            logger.error("Failed to rollback model: %s", e, exc_info=True)
            return OrchestratorResponse(
                success=False,
                message=f"Failed to rollback model: {e}",
                is_done=True,
            )

    def _handle_get_current_models(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        parameters: Dict[str, Any],
    ) -> OrchestratorResponse:
        """Handle get_current_models action.

        Parameters:
            task_type: "event_rate" or "time_to_event"
        """
        task_type_str = parameters.get("task_type", "event_rate")

        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            return OrchestratorResponse(
                success=False,
                message=f"Invalid task_type: {task_type_str}",
                is_done=True,
            )

        try:
            models = get_current_models(
                sync_unit_name=sync_unit_obj.get_name(),
                task_type=task_type,
            )

            response = OrchestratorResponse(
                success=True,
                message="Retrieved current models",
                is_done=True,
            )
            response.data = {
                "models": models,
                "registered_model_name": get_registered_model_name(
                    sync_unit_obj.get_name(), task_type
                ),
            }
            return response

        except Exception as e:
            logger.error("Failed to get current models: %s", e, exc_info=True)
            return OrchestratorResponse(
                success=False,
                message=f"Failed to get current models: {e}",
                is_done=True,
            )
