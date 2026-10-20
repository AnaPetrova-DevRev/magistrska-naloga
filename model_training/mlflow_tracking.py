"""MLflow tracking utilities for model training.

This module provides helpers for logging training runs, metrics, and artifacts
to MLflow. It handles graceful degradation when MLflow is not available.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from modeling.core_types import DataSnapshotMetadata, TaskType, TrainingContext
from modeling.mlflow_utils import (
    MLFLOW_AVAILABLE,
    MlflowClient,
    flatten_dict,
    get_registered_model_name,
    is_mlflow_available,
    mlflow,
)

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Context manager for MLflow run tracking.

    Handles starting runs, logging parameters/metrics/artifacts, and
    graceful degradation when MLflow is unavailable.

    Example:
        with MLflowTracker(training_ctx, snapshot_meta) as tracker:
            model = create_model(training_ctx)
            model.fit(dataset)
            tracker.log_metrics({"val_mae": 0.5})
            tracker.log_model_artifacts(artifact_dir)
    """

    def __init__(
        self,
        training_ctx: TrainingContext,
        snapshot_meta: DataSnapshotMetadata,
        experiment_name: Optional[str] = None,
    ):
        """Initialize the tracker.

        Args:
            training_ctx: Training context with model/task info.
            snapshot_meta: Dataset snapshot metadata.
            experiment_name: MLflow experiment name. Defaults to "model_training".
        """
        self.training_ctx = training_ctx
        self.snapshot_meta = snapshot_meta
        self.experiment_name = (
            experiment_name
            or os.environ.get("MLFLOW_EXPERIMENT_NAME")
            or "model_training"
        )
        self.run = None
        self.run_id = None
        self._active = False

    def __enter__(self) -> "MLflowTracker":
        """Start the MLflow run."""
        if not MLFLOW_AVAILABLE:
            logger.debug("MLflow not available - tracking disabled")
            return self

        try:
            mlflow.set_experiment(self.experiment_name)

            self.run = mlflow.start_run()
            self.run_id = self.run.info.run_id
            self._active = True

            self._log_initial_tags()
            self._log_params()

            logger.info("Started MLflow run: %s", self.run_id)

        except Exception as e:
            logger.warning("Failed to start MLflow run: %s", e)
            self._active = False

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End the MLflow run."""
        if not self._active or not MLFLOW_AVAILABLE:
            return

        try:
            if exc_type is not None:
                mlflow.set_tag("run_status", "failed")
                mlflow.set_tag("error_type", exc_type.__name__)
                mlflow.set_tag("error_message", str(exc_val)[:500])
            else:
                mlflow.set_tag("run_status", "completed")

            mlflow.end_run()
            logger.info("Ended MLflow run: %s", self.run_id)

        except Exception as e:
            logger.warning("Error ending MLflow run: %s", e)

    def _log_initial_tags(self) -> None:
        """Log initial run tags."""
        if not self._active:
            return

        tags = {
            "sync_unit": self.training_ctx.sync_unit_name,
            "task_type": self.training_ctx.task_type.value,
            "model_type": self.training_ctx.model_type,
            "template_name": self.training_ctx.template_name or "none",
            "data_snapshot_metadata_path": self.snapshot_meta.snapshot_metadata_path or "",
            "train_start": self.snapshot_meta.start_time.isoformat(),
            "train_end": self.snapshot_meta.end_time.isoformat(),
        }

        try:
            mlflow.set_tags(tags)
        except Exception as e:
            logger.warning("Failed to log tags: %s", e)

    def _log_params(self) -> None:
        """Log model and training configuration as params."""
        if not self._active:
            return

        try:
            model_params = flatten_dict(self.training_ctx.model_config, "model")
            training_params = flatten_dict(self.training_ctx.training_config, "train")

            all_params = {**model_params, **training_params}

            all_params = {k: str(v)[:500] for k, v in all_params.items()}

            if all_params:
                mlflow.log_params(all_params)

        except Exception as e:
            logger.warning("Failed to log params: %s", e)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow.

        Args:
            metrics: Dict of metric name to value.
            step: Optional step number for time-series metrics.
        """
        if not self._active or not MLFLOW_AVAILABLE:
            return

        try:
            numeric_metrics = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }

            if numeric_metrics:
                mlflow.log_metrics(numeric_metrics, step=step)

        except Exception as e:
            logger.warning("Failed to log metrics: %s", e)

    def log_training_info(self, training_info: Dict[str, Any]) -> None:
        """Log training metadata from model.fit().

        Args:
            training_info: Dict returned by model.fit().
        """
        if not self._active or not MLFLOW_AVAILABLE:
            return

        try:
            metrics = {
                k: v for k, v in training_info.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if metrics:
                prefixed = {f"train_{k}": v for k, v in metrics.items()}
                mlflow.log_metrics(prefixed)

            info_json = json.dumps(training_info, indent=2, default=str)
            mlflow.log_text(info_json, "training_info.json")

        except Exception as e:
            logger.warning("Failed to log training info: %s", e)

    def log_model_artifacts(self, artifact_dir: str) -> None:
        """Log the model artifact directory.

        Args:
            artifact_dir: Path to directory containing model files.
        """
        if not self._active or not MLFLOW_AVAILABLE:
            return

        # Opt-out of duplicating the model directory into the MLflow store. The
        # canonical model already lives under data/{unit}/models/; some models
        # (e.g. SARIMAX on large frames) serialize to multi-GB pickles that,
        # duplicated per run, exhausted the disk. Set MLFLOW_SKIP_ARTIFACTS=1 to
        # keep metric/param tracking but skip the artifact copy.
        if os.environ.get("MLFLOW_SKIP_ARTIFACTS") == "1":
            logger.info("MLFLOW_SKIP_ARTIFACTS=1 set - skipping artifact copy for %s", artifact_dir)
            return

        try:
            if os.path.exists(artifact_dir):
                mlflow.log_artifacts(artifact_dir, artifact_path="model")
                logger.info("Logged model artifacts from %s", artifact_dir)

        except Exception as e:
            logger.warning("Failed to log model artifacts: %s", e)

    def log_snapshot_metadata(self) -> None:
        """Log the dataset snapshot metadata as an artifact."""
        if not self._active or not MLFLOW_AVAILABLE:
            return

        try:
            meta_dict = {
                "snapshot_metadata_path": self.snapshot_meta.snapshot_metadata_path,
                "task_type": self.snapshot_meta.task_type.value,
                "start_time": self.snapshot_meta.start_time.isoformat(),
                "end_time": self.snapshot_meta.end_time.isoformat(),
                "train_indices_count": len(self.snapshot_meta.train_indices),
                "val_indices_count": len(self.snapshot_meta.val_indices),
                "test_indices_count": len(self.snapshot_meta.test_indices),
                "filters": self.snapshot_meta.filters,
                "resample_freq": self.snapshot_meta.resample_freq,
                "created_at": self.snapshot_meta.created_at.isoformat(),
            }

            mlflow.log_text(
                json.dumps(meta_dict, indent=2),
                "snapshot_metadata.json"
            )

        except Exception as e:
            logger.warning("Failed to log snapshot metadata: %s", e)

    def get_run_id(self) -> Optional[str]:
        """Get the MLflow run ID if available."""
        return self.run_id


def register_model_version(
    run_id: str,
    sync_unit_name: str,
    task_type: TaskType,
    artifact_path: str = "model",
) -> Optional[str]:
    """Register a model version in MLflow Model Registry.

    Args:
        run_id: The MLflow run ID containing the model.
        sync_unit_name: Sync unit name for the model name.
        task_type: Task type for the model name.
        artifact_path: Path to model artifacts within the run.

    Returns:
        Model version string if successful, None otherwise.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available - cannot register model")
        return None

    try:
        client = MlflowClient()

        model_name = get_registered_model_name(sync_unit_name, task_type)

        try:
            client.get_registered_model(model_name)
        except Exception:
            client.create_registered_model(
                model_name,
                description=f"Forecasting model for {sync_unit_name} ({task_type.value})"
            )

        run = client.get_run(run_id)
        source = f"{run.info.artifact_uri}/{artifact_path}"

        mv = client.create_model_version(
            name=model_name,
            source=source,
            run_id=run_id,
        )

        logger.info(
            "Registered model version %s for %s",
            mv.version, model_name
        )

        return mv.version

    except Exception as e:
        logger.error("Failed to register model version: %s", e)
        return None
