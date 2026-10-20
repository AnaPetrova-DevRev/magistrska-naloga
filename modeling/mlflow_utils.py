"""Shared MLflow utilities used by training, serving, and lifecycle modules.

Centralizes the MLflow availability check and the registered model naming
convention so they are defined in exactly one place.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from modeling.core_types import TaskType

logger = logging.getLogger(__name__)

MLFLOW_AVAILABLE = False
mlflow = None
MlflowClient = None
try:
    import mlflow as _mlflow
    from mlflow.tracking import MlflowClient as _MlflowClient
    mlflow = _mlflow
    MlflowClient = _MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    logger.info("MLflow not available")


def is_mlflow_available() -> bool:
    """Check if MLflow is available and configured."""
    return MLFLOW_AVAILABLE


def get_registered_model_name(sync_unit_name: str, task_type: TaskType) -> str:
    """Generate a standardized registered model name.

    Args:
        sync_unit_name: The sync unit identifier.
        task_type: The task type.

    Returns:
        Model name in format: sync_model::{sync_unit}::{task_type}
    """
    task_str = task_type.value if isinstance(task_type, TaskType) else task_type
    return f"sync_model::{sync_unit_name}::{task_str}"


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dict for MLflow param logging.

    Args:
        d: Dict to flatten.
        prefix: Prefix for keys.

    Returns:
        Flattened dict with dot-separated keys.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key))
        elif isinstance(v, (list, tuple)):
            items[new_key] = str(v)
        else:
            items[new_key] = v
    return items
