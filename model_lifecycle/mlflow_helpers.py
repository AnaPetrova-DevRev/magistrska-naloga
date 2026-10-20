"""MLflow helper functions for model lifecycle management.

This module provides low-level functions for interacting with MLflow
tracking server and model registry.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from modeling.core_types import TaskType
from modeling.mlflow_utils import (
    MLFLOW_AVAILABLE,
    MlflowClient,
    get_registered_model_name,
)

logger = logging.getLogger(__name__)


def list_candidate_runs(
    sync_unit_name: str,
    task_type: TaskType,
    top_k: int = 10,
    order_by: str = "val_mae",
    ascending: bool = True,
) -> List[Dict[str, Any]]:
    """List candidate training runs for a sync unit and task type.

    Args:
        sync_unit_name: The sync unit name.
        task_type: The task type.
        top_k: Maximum number of runs to return.
        order_by: Metric to sort by.
        ascending: Sort ascending if True.

    Returns:
        List of run info dicts.
    """
    if not MLFLOW_AVAILABLE:
        return []

    try:
        client = MlflowClient()

        filter_string = (
            f"tags.sync_unit = '{sync_unit_name}' AND "
            f"tags.task_type = '{task_type.value}' AND "
            f"attributes.status = 'FINISHED'"
        )

        order_dir = "ASC" if ascending else "DESC"
        order_clause = f"metrics.{order_by} {order_dir}"

        runs = client.search_runs(
            experiment_ids=[],
            filter_string=filter_string,
            max_results=top_k,
            order_by=[order_clause],
        )

        results = []
        for run in runs:
            results.append({
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "model_type": run.data.tags.get("model_type", "unknown"),
                "template_name": run.data.tags.get("template_name", "unknown"),
                "metrics": dict(run.data.metrics),
            })

        return results

    except Exception as e:
        logger.error("Failed to list candidate runs: %s", e)
        return []


def register_run_as_model(
    sync_unit_name: str,
    task_type: TaskType,
    run_id: str,
    initial_stage: str = "None",
) -> Optional[Dict[str, Any]]:
    """Register a training run as a model version.

    Args:
        sync_unit_name: The sync unit name.
        task_type: The task type.
        run_id: The MLflow run ID to register.
        initial_stage: Initial stage for the version.

    Returns:
        Dict with registration info or None on failure.
    """
    if not MLFLOW_AVAILABLE:
        return None

    try:
        client = MlflowClient()

        model_name = get_registered_model_name(sync_unit_name, task_type)

        try:
            client.get_registered_model(model_name)
        except Exception:
            client.create_registered_model(
                name=model_name,
                description=f"Forecasting model for {sync_unit_name} ({task_type.value})",
            )
            logger.info("Created registered model: %s", model_name)

        run = client.get_run(run_id)
        source = f"{run.info.artifact_uri}/model"

        mv = client.create_model_version(
            name=model_name,
            source=source,
            run_id=run_id,
        )

        if initial_stage and initial_stage != "None":
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage=initial_stage,
            )

        logger.info(
            "Registered model version %s for %s (stage: %s)",
            mv.version, model_name, initial_stage
        )

        return {
            "model_name": model_name,
            "version": mv.version,
            "run_id": run_id,
            "stage": initial_stage,
            "source": source,
        }

    except Exception as e:
        logger.error("Failed to register run as model: %s", e)
        return None


def set_model_stage(
    sync_unit_name: str,
    task_type: TaskType,
    version: int,
    new_stage: str,
) -> bool:
    """Set the stage of a model version.

    Args:
        sync_unit_name: The sync unit name.
        task_type: The task type.
        version: The model version number.
        new_stage: The target stage.

    Returns:
        True if successful.
    """
    if not MLFLOW_AVAILABLE:
        return False

    try:
        client = MlflowClient()

        model_name = get_registered_model_name(sync_unit_name, task_type)

        client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage=new_stage,
        )

        logger.info(
            "Transitioned %s version %s to stage %s",
            model_name, version, new_stage
        )

        return True

    except Exception as e:
        logger.error("Failed to set model stage: %s", e)
        return False


def rollback_model_to_version(
    sync_unit_name: str,
    task_type: TaskType,
    target_version: int,
) -> Optional[Dict[str, Any]]:
    """Rollback Production to a specified version.

    This demotes the current Production version and promotes the target version.

    Args:
        sync_unit_name: The sync unit name.
        task_type: The task type.
        target_version: The version to rollback to.

    Returns:
        Dict with rollback info or None on failure.
    """
    if not MLFLOW_AVAILABLE:
        return None

    try:
        client = MlflowClient()

        model_name = get_registered_model_name(sync_unit_name, task_type)

        current_prod = client.get_latest_versions(model_name, stages=["Production"])

        previous_prod_version = None
        if current_prod:
            previous_prod_version = current_prod[0].version
            client.transition_model_version_stage(
                name=model_name,
                version=str(previous_prod_version),
                stage="Archived",
            )
            logger.info(
                "Archived previous production version %s", previous_prod_version
            )

        client.transition_model_version_stage(
            name=model_name,
            version=str(target_version),
            stage="Production",
        )

        logger.info(
            "Rolled back %s to version %s",
            model_name, target_version
        )

        return {
            "model_name": model_name,
            "previous_production_version": previous_prod_version,
            "new_production_version": target_version,
        }

    except Exception as e:
        logger.error("Failed to rollback model: %s", e)
        return None


def get_current_models(
    sync_unit_name: str,
    task_type: TaskType,
) -> Dict[str, Any]:
    """Get info about current Production and Staging models.

    Args:
        sync_unit_name: The sync unit name.
        task_type: The task type.

    Returns:
        Dict with model info for each stage.
    """
    if not MLFLOW_AVAILABLE:
        return {}

    try:
        client = MlflowClient()

        model_name = get_registered_model_name(sync_unit_name, task_type)

        result = {}

        for stage in ["Production", "Staging", "Archived"]:
            versions = client.get_latest_versions(model_name, stages=[stage])
            if versions:
                mv = versions[0]
                run = client.get_run(mv.run_id)

                result[stage] = {
                    "version": mv.version,
                    "run_id": mv.run_id,
                    "creation_timestamp": mv.creation_timestamp,
                    "model_type": run.data.tags.get("model_type", "unknown"),
                    "template_name": run.data.tags.get("template_name", "unknown"),
                    "metrics": dict(run.data.metrics),
                }

        return result

    except Exception as e:
        logger.error("Failed to get current models: %s", e)
        return {}


def get_all_model_versions(
    sync_unit_name: str,
    task_type: TaskType,
) -> List[Dict[str, Any]]:
    """Get all registered versions for a model.

    Args:
        sync_unit_name: The sync unit name.
        task_type: The task type.

    Returns:
        List of version info dicts.
    """
    if not MLFLOW_AVAILABLE:
        return []

    try:
        client = MlflowClient()

        model_name = get_registered_model_name(sync_unit_name, task_type)

        filter_string = f"name = '{model_name}'"
        versions = client.search_model_versions(filter_string)

        results = []
        for mv in versions:
            results.append({
                "version": mv.version,
                "run_id": mv.run_id,
                "stage": mv.current_stage,
                "status": mv.status,
                "creation_timestamp": mv.creation_timestamp,
            })

        return results

    except Exception as e:
        logger.error("Failed to get model versions: %s", e)
        return []
