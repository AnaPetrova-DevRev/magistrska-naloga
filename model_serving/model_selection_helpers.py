"""MLflow-based model selection helpers for serving.

This module provides utilities for selecting models from the MLflow Model
Registry based on stage (Production/Staging) or explicit run ID.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from modeling.core_types import TaskType
from modeling.mlflow_utils import (
    MLFLOW_AVAILABLE,
    MlflowClient,
    get_registered_model_name,
)

logger = logging.getLogger(__name__)


def select_model_by_stage(
    sync_unit_name: str,
    task_type: TaskType,
    stage: str = "Production",
) -> Optional[Dict[str, Any]]:
    """Select a model from the MLflow registry by stage.

    Args:
        sync_unit_name: The sync unit name.
        task_type: The task type.
        stage: The stage to query ("Production", "Staging", etc.)

    Returns:
        Dict with model info (run_id, source, model_type, model_config)
        or None if not found.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available - cannot select model by stage")
        return None

    try:
        client = MlflowClient()

        model_name = get_registered_model_name(sync_unit_name, task_type)

        versions = client.get_latest_versions(model_name, stages=[stage])

        if not versions:
            logger.info("No model found in stage %s for %s", stage, model_name)
            return None

        mv = versions[0]

        run = client.get_run(mv.run_id)
        tags = run.data.tags
        params = run.data.params

        model_config = _extract_model_config(params)

        return {
            "run_id": mv.run_id,
            "version": mv.version,
            "source": mv.source,
            "model_type": tags.get("model_type", "unknown"),
            "task_type": tags.get("task_type", task_type.value),
            "model_config": model_config,
            "stage": stage,
        }

    except Exception as e:
        logger.error("Failed to select model by stage: %s", e)
        return None


def select_model_by_run_id(run_id: str) -> Optional[Dict[str, Any]]:
    """Select a model by MLflow run ID.

    Args:
        run_id: The MLflow run ID.

    Returns:
        Dict with model info or None if not found.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available - cannot select model by run ID")
        return None

    try:
        client = MlflowClient()

        run = client.get_run(run_id)
        tags = run.data.tags
        params = run.data.params

        source = f"{run.info.artifact_uri}/model"

        model_config = _extract_model_config(params)

        return {
            "run_id": run_id,
            "source": source,
            "model_type": tags.get("model_type", "unknown"),
            "task_type": tags.get("task_type", "unknown"),
            "model_config": model_config,
        }

    except Exception as e:
        logger.error("Failed to select model by run ID: %s", e)
        return None


def get_model_info(
    sync_unit_name: str,
    task_type: TaskType,
) -> Dict[str, Any]:
    """Get information about available models for a sync unit.

    Returns info about Production, Staging, and recent candidate models.

    Args:
        sync_unit_name: The sync unit name.
        task_type: The task type.

    Returns:
        Dict with model info for each stage.
    """
    if not MLFLOW_AVAILABLE:
        return {"mlflow_available": False, "models": {}}

    try:
        client = MlflowClient()
        model_name = get_registered_model_name(sync_unit_name, task_type)

        result = {"mlflow_available": True, "models": {}}

        for stage in ["Production", "Staging", "None"]:
            versions = client.get_latest_versions(model_name, stages=[stage])
            if versions:
                mv = versions[0]
                result["models"][stage] = {
                    "version": mv.version,
                    "run_id": mv.run_id,
                    "creation_timestamp": mv.creation_timestamp,
                }

        return result

    except Exception as e:
        logger.error("Failed to get model info: %s", e)
        return {"mlflow_available": True, "error": str(e), "models": {}}


def _parse_param_value(value: str) -> Any:
    """Parse a string parameter value to its most appropriate Python type.

    Handles ints, floats (including negatives), booleans, and falls back
    to the original string.
    """
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        int_val = int(value)
        # Only return int if round-tripping preserves the string
        # (avoids turning "1.0" into 1 via int("1.0") ValueError)
        return int_val
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _extract_model_config(params: Dict[str, str]) -> Dict[str, Any]:
    """Extract model config from flattened MLflow params.

    Params are stored as "model.hidden_size", "model.num_layers" etc.
    This reconstructs the nested dict.

    Args:
        params: Flattened params from MLflow run.

    Returns:
        Reconstructed model_config dict.
    """
    model_config = {}

    for key, value in params.items():
        if key.startswith("model."):
            config_key = key[6:]
            model_config[config_key] = _parse_param_value(value)

    return model_config


def list_recent_runs(
    sync_unit_name: str,
    task_type: TaskType,
    max_results: int = 10,
    order_by: str = "start_time DESC",
) -> list:
    """List recent training runs for a sync unit.

    Args:
        sync_unit_name: The sync unit name.
        task_type: The task type.
        max_results: Maximum number of runs to return.
        order_by: Ordering for results.

    Returns:
        List of run info dicts.
    """
    if not MLFLOW_AVAILABLE:
        return []

    try:
        client = MlflowClient()

        filter_string = (
            f"tags.sync_unit = '{sync_unit_name}' AND "
            f"tags.task_type = '{task_type.value}'"
        )

        runs = client.search_runs(
            experiment_ids=[],
            filter_string=filter_string,
            max_results=max_results,
            order_by=[order_by],
        )

        results = []
        for run in runs:
            results.append({
                "run_id": run.info.run_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "model_type": run.data.tags.get("model_type", "unknown"),
                "metrics": dict(run.data.metrics),
            })

        return results

    except Exception as e:
        logger.error("Failed to list recent runs: %s", e)
        return []
