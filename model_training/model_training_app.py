"""Model training app for the orchestrator system.

This app handles training forecasting models on sync unit data. It supports
training multiple models sequentially, with full MLflow tracking and
error resilience.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Tuple

from app_base import App
from orchestrator_message import OrchestratorResponse
import storage_access
import sync_unit as sync_unit_module
from modeling.core_types import (
    Dataset,
    DatasetConfig,
    DataSnapshotMetadata,
    TaskType,
    TrainingContext,
)
from dataclasses import replace as dc_replace
from modeling.dataset_builder import (
    build_event_rate_dataset,
    build_time_to_event_dataset,
    compute_time_series_cv_splits,
)
from modeling.metrics_registry import build_metric, normalize_metric_defs
from modeling.model_factory import create_model, is_model_supported
from model_training.mlflow_tracking import MLflowTracker, is_mlflow_available
from app_actions_constants import TRAIN_MODELS

logger = logging.getLogger(__name__)

MODEL_TEMPLATES_FOLDER = "model_configs"


class ModelTrainingApp(App):
    """App for training forecasting models on sync unit data.

    Actions:
        train_models: Train one or more models on a sync unit's data.

    The app supports:
    - Multiple models trained sequentially per sync unit
    - Per-model templates with hyperparameters
    - Custom and built-in metrics
    - MLflow tracking (when available)
    - Error resilience (one model failure doesn't stop others)
    """

    def __init__(self, sa: Optional[storage_access.StorageAccess] = None):
        self._sa = sa or storage_access.StorageAccess()

    def get_name(self) -> str:
        return "model_training"

    def get_available_actions(self) -> list[str]:
        return [TRAIN_MODELS]

    def execute_action(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        action: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorResponse:
        """Execute a model training action.

        Args:
            sync_unit_obj: The sync unit to train on.
            action: The action to perform (train_models).
            parameters: Action parameters including task_type, dataset config,
                       models list, and metrics.

        Returns:
            OrchestratorResponse with training results.
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

        if action == TRAIN_MODELS:
            return self._handle_train_models(sync_unit_obj, parameters)

        raise ValueError(f"Unknown action: {action}")

    def _handle_train_models(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        parameters: Dict[str, Any],
    ) -> OrchestratorResponse:
        """Handle the train_models action.

        Expected parameters:
            task_type: "event_rate" or "time_to_event"
            dataset: {
                start_time: ISO datetime string
                end_time: ISO datetime string
                train_ratio: float (default 0.7)
                val_ratio: float (default 0.15)
                filters: dict (optional)
            }
            models: [
                { model_type: "ARIMA", template_name: "arima_small_v1" },
                ...
            ]
            metrics: ["mae", "rmse", ...] or [{"name": "mae"}, ...]
        """
        task_type_str = parameters.get("task_type", "event_rate")
        dataset_params = parameters.get("dataset", {})
        model_specs = parameters.get("models", [])
        raw_metrics = parameters.get("metrics", ["mae"])

        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            return OrchestratorResponse(
                success=False,
                message=f"Invalid task_type: {task_type_str}",
                is_done=True,
            )

        if not model_specs:
            return OrchestratorResponse(
                success=False,
                message="No models specified in 'models' parameter",
                is_done=True,
            )

        metric_defs = normalize_metric_defs(raw_metrics)
        cv_folds = parameters.get("cv_folds", 0)

        try:
            dataset, snapshot_meta = self._build_dataset(
                sync_unit_obj, task_type, dataset_params
            )
        except Exception as e:
            logger.error("Failed to build dataset: %s", e, exc_info=True)
            return OrchestratorResponse(
                success=False,
                message=f"Failed to build dataset: {e}",
                is_done=True,
            )

        results: List[Dict[str, Any]] = []

        for spec in model_specs:
            model_type = spec.get("model_type")
            template_name = spec.get("template_name")

            if not model_type:
                results.append({
                    "model_spec": spec,
                    "success": False,
                    "error": "Missing model_type in spec",
                })
                continue

            try:
                result = self._train_single_model(
                    sync_unit_obj=sync_unit_obj,
                    task_type=task_type,
                    dataset=dataset,
                    snapshot_meta=snapshot_meta,
                    model_type=model_type,
                    template_name=template_name,
                    metric_defs=metric_defs,
                    cv_folds=cv_folds,
                )
                results.append({**result, "success": True})
            except Exception as e:
                logger.error(
                    "Error training model %s: %s", spec, e, exc_info=True
                )
                results.append({
                    "model_type": model_type,
                    "template_name": template_name,
                    "success": False,
                    "error": str(e),
                })

            import gc
            gc.collect()
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass

        overall_success = all(r.get("success", False) for r in results)
        successful_count = sum(1 for r in results if r.get("success", False))
        total_count = len(results)

        message = (
            f"Training complete: {successful_count}/{total_count} models succeeded"
        )

        response = OrchestratorResponse(
            success=overall_success,
            message=message,
            is_done=True,
        )
        response.data = {
            "results": results,
            "sync_unit": sync_unit_obj.get_name(),
            "task_type": task_type.value,
            "snapshot_metadata_path": snapshot_meta.snapshot_metadata_path,
        }

        return response

    def _build_dataset(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        task_type: TaskType,
        dataset_params: Dict[str, Any],
    ) -> Tuple[Dataset, DataSnapshotMetadata]:
        """Build a dataset for training.

        Args:
            sync_unit_obj: The sync unit.
            task_type: EVENT_RATE or TIME_TO_EVENT.
            dataset_params: Dataset configuration from parameters.

        Returns:
            Tuple of (dataset, snapshot_metadata).
        """
        start_time_str = dataset_params.get("start_time")
        end_time_str = dataset_params.get("end_time")

        if start_time_str:
            start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
        else:
            start_time = datetime.min.replace(tzinfo=UTC)

        if end_time_str:
            end_time = datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))
        else:
            end_time = datetime.max.replace(tzinfo=UTC)

        config = DatasetConfig(
            task_type=task_type,
            start_time=start_time,
            end_time=end_time,
            train_ratio=dataset_params.get("train_ratio", 0.7),
            val_ratio=dataset_params.get("val_ratio", 0.15),
            filters=dataset_params.get("filters", {}),
            resample_freq=dataset_params.get("resample_freq"),
            covariates=dataset_params.get("covariates", []),
        )

        if task_type == TaskType.EVENT_RATE:
            return build_event_rate_dataset(sync_unit_obj, config, self._sa)
        elif task_type == TaskType.TIME_TO_EVENT:
            return build_time_to_event_dataset(sync_unit_obj, config, self._sa)
        else:
            raise NotImplementedError(
                f"Dataset building for {task_type.value} not yet implemented"
            )

    def _train_single_model(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        task_type: TaskType,
        dataset: Dataset,
        snapshot_meta: DataSnapshotMetadata,
        model_type: str,
        template_name: Optional[str],
        metric_defs: List[Dict[str, Any]],
        cv_folds: int = 0,
    ) -> Dict[str, Any]:
        """Train a single model and return results.

        Args:
            sync_unit_obj: The sync unit.
            task_type: The task type.
            dataset: The prepared dataset.
            snapshot_meta: Metadata about the data snapshot.
            model_type: The type of model (e.g., "ARIMA").
            template_name: Name of the template file to load.
            metric_defs: List of metric definitions.
            cv_folds: Number of time-series CV folds (0 = disabled).

        Returns:
            Dict with training results including metrics and run info.
        """
        if not is_model_supported(model_type, task_type):
            raise ValueError(
                f"Model type {model_type} not supported for task {task_type.value}"
            )

        template = self._load_template(template_name) if template_name else {}

        if template:
            template_task = template.get("task_type")
            if template_task and template_task != task_type.value:
                raise ValueError(
                    f"Template task_type '{template_task}' doesn't match "
                    f"requested task_type '{task_type.value}'"
                )

        training_ctx = TrainingContext(
            sync_unit_name=sync_unit_obj.get_name(),
            task_type=task_type,
            model_type=model_type,
            model_config=template.get("model_config", {}),
            training_config=template.get("training_config", {}),
            metric_defs=metric_defs,
            template_name=template_name,
        )

        logger.info(
            "Training model %s with template %s on %s",
            model_type, template_name, sync_unit_obj.get_name()
        )

        with MLflowTracker(training_ctx, snapshot_meta) as tracker:
            model = create_model(training_ctx)

            training_info = model.fit(dataset)
            tracker.log_training_info(training_info)

            # --- Evaluation ---
            is_zero_shot = training_info.get("zero_shot", False)
            if is_zero_shot:
                # Zero-shot models: merge val+test into single eval pass
                # (no training occurs, so val/test distinction is meaningless)
                logger.info(
                    "Zero-shot model %s: merging val+test into single eval split",
                    model_type,
                )
                merged_indices = sorted(
                    dataset.val_indices + dataset.test_indices
                )
                eval_dataset = dc_replace(
                    dataset,
                    val_indices=merged_indices,
                    test_indices=[],
                )
                metrics_values = self._evaluate_metrics(
                    training_ctx, model, eval_dataset
                )
            else:
                metrics_values = self._evaluate_metrics(
                    training_ctx, model, dataset
                )
            tracker.log_metrics(metrics_values)

            # --- Time-series cross-validation ---
            cv_results: Optional[Dict[str, Any]] = None
            if is_zero_shot and cv_folds >= 2:
                logger.info(
                    "Skipping CV for zero-shot model %s (no training occurs between folds)",
                    model_type,
                )
            if cv_folds >= 2 and not is_zero_shot:
                cv_results = self._run_cross_validation(
                    training_ctx, dataset, cv_folds,
                )
                tracker.log_metrics(cv_results["combined_metrics"])

            artifact_dir = self._get_artifact_dir(
                sync_unit_obj, model_type, template_name
            )
            model.save(artifact_dir)
            tracker.log_model_artifacts(artifact_dir)
            tracker.log_snapshot_metadata()

            mlflow_run_id = tracker.get_run_id()

            self._save_metadata(
                artifact_dir, training_ctx, snapshot_meta,
                metrics_values, training_info, cv_results,
                mlflow_run_id=mlflow_run_id,
            )

        return {
            "model_type": model_type,
            "template_name": template_name,
            "metrics": metrics_values,
            "training_info": training_info,
            "artifact_path": artifact_dir,
            "mlflow_run_id": mlflow_run_id,
            "cv": cv_results,
        }

    def _load_template(self, template_name: str) -> Dict[str, Any]:
        """Load a model template from the templates folder.

        Args:
            template_name: Name of the template (without .json extension).

        Returns:
            Dict with template configuration.
        """
        if not template_name.endswith(".json"):
            template_name = f"{template_name}.json"

        template_path = os.path.join(MODEL_TEMPLATES_FOLDER, template_name)

        if not os.path.exists(template_path):
            logger.warning("Template not found at %s, using empty config", template_path)
            return {}

        with open(template_path, "r") as f:
            template = json.load(f)

        logger.info("Loaded template from %s", template_path)
        return template

    def _evaluate_metrics(
        self,
        training_ctx: TrainingContext,
        model,
        dataset: Dataset,
    ) -> Dict[str, Any]:
        """Evaluate metrics on the validation and test splits.

        Args:
            training_ctx: The training context.
            model: The trained model.
            dataset: The dataset with val_indices and test_indices.

        Returns:
            Dict mapping metric names to values (or error strings).
        """
        metrics_values: Dict[str, Any] = {}

        splits = {
            "val": dataset.val_indices,
            "test": dataset.test_indices,
        }

        for split, indices in splits.items():
            if not indices:
                continue

            y_true, y_pred = self._get_targets_and_predictions(
                training_ctx, model, dataset, indices
            )

            for m_def in training_ctx.metric_defs:
                try:
                    metric = build_metric(m_def)
                    context = {
                        "task_type": training_ctx.task_type.value,
                        "sync_unit": training_ctx.sync_unit_name,
                        "split": split,
                    }
                    value = metric(y_true, y_pred, context)
                    key = f"{split}_{metric.name}"
                    metrics_values[key] = float(value)
                except Exception as e:
                    logger.error(
                        "Metric %s failed on %s split: %s",
                        m_def.get("name", m_def), split, e, exc_info=True
                    )
                    key = f"{split}_{m_def.get('name', 'unknown')}_error"
                    metrics_values[key] = str(e)

        return metrics_values

    def _run_cross_validation(
        self,
        training_ctx: TrainingContext,
        dataset: Dataset,
        n_folds: int,
    ) -> Dict[str, Any]:
        """Run time-series cross-validation with expanding window.

        Trains a fresh model for each fold and evaluates on that fold's
        validation window.  The held-out test set is not touched.

        Returns:
            Dict with ``fold_metrics`` (list of per-fold dicts),
            ``combined_metrics`` (mean/std across folds), and ``n_folds``.
        """
        n_trainval = len(dataset.train_indices) + len(dataset.val_indices)
        cv_splits = compute_time_series_cv_splits(n_trainval, n_folds)

        logger.info(
            "Running %d-fold time-series CV (%d trainval points)",
            n_folds, n_trainval,
        )

        fold_metrics: List[Dict[str, float]] = []

        for fold_idx, (fold_train, fold_val) in enumerate(cv_splits):
            fold_dataset = dc_replace(
                dataset,
                train_indices=fold_train,
                val_indices=fold_val,
                test_indices=[],
            )

            fold_model = create_model(training_ctx)
            fold_model.fit(fold_dataset)

            fold_result = self._evaluate_metrics(
                training_ctx, fold_model, fold_dataset,
            )
            fold_metrics.append(fold_result)

            logger.info(
                "  Fold %d/%d  train=%d  val=%d  %s",
                fold_idx + 1, n_folds,
                len(fold_train), len(fold_val),
                {k: round(v, 4) for k, v in fold_result.items()
                 if isinstance(v, float)},
            )

            del fold_model
            import gc
            gc.collect()
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass

        # Aggregate: mean and std per metric key
        all_keys: set[str] = set()
        for fm in fold_metrics:
            all_keys.update(
                k for k, v in fm.items() if isinstance(v, (int, float))
            )

        combined: Dict[str, float] = {}
        for key in sorted(all_keys):
            values = [
                fm[key] for fm in fold_metrics
                if key in fm and isinstance(fm[key], (int, float))
            ]
            if not values:
                continue
            mean = sum(values) / len(values)
            combined[f"cv_mean_{key}"] = mean
            if len(values) > 1:
                var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                combined[f"cv_std_{key}"] = var ** 0.5

        return {
            "n_folds": n_folds,
            "fold_metrics": fold_metrics,
            "combined_metrics": combined,
        }

    def _get_targets_and_predictions(
        self,
        training_ctx: TrainingContext,
        model,
        dataset: Dataset,
        indices: List[int],
    ) -> Tuple[List[float], List[float]]:
        """Get true values and predictions for specified indices.

        Args:
            training_ctx: Training context.
            model: Trained model.
            dataset: The dataset.
            indices: Which indices to evaluate.

        Returns:
            Tuple of (y_true, y_pred) as lists.
        """
        import numpy as np

        if training_ctx.task_type == TaskType.EVENT_RATE:
            target_array = np.array(dataset.events_per_minute)
        else:
            target_array = np.array(dataset.minutes_to_next_event)

        y_true = [float(target_array[i]) for i in indices]
        y_pred_array = model.predict_pointwise(dataset, indices)
        y_pred = [float(p) for p in y_pred_array]

        return y_true, y_pred

    def _get_artifact_dir(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        model_type: str,
        template_name: Optional[str] = None,
    ) -> str:
        """Get the directory for storing model artifacts.

        Args:
            sync_unit_obj: The sync unit.
            model_type: Type of model.
            template_name: Template name to include in the directory.

        Returns:
            Path to artifact directory.
        """
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%f")
        suffix = f"{template_name}" if template_name else "no_template"
        base_path = sync_unit_obj.get_path_for_sync_unit()
        artifact_dir = os.path.join(
            "data", base_path, "models", f"{model_type}_{suffix}_{timestamp}"
        )
        os.makedirs(artifact_dir, exist_ok=True)
        return artifact_dir

    def _save_metadata(
        self,
        artifact_dir: str,
        training_ctx: TrainingContext,
        snapshot_meta: DataSnapshotMetadata,
        metrics_values: Dict[str, Any],
        training_info: Dict[str, Any],
        cv_results: Optional[Dict[str, Any]] = None,
        mlflow_run_id: Optional[str] = None,
    ) -> None:
        """Save training metadata to the artifact directory.

        Args:
            artifact_dir: Where to save.
            training_ctx: Training context.
            snapshot_meta: Dataset snapshot metadata.
            metrics_values: Computed metric values.
            training_info: Training statistics from model.fit().
            cv_results: Optional cross-validation results.
            mlflow_run_id: Optional MLflow run ID for this training run.
        """
        metadata = {
            "sync_unit": training_ctx.sync_unit_name,
            "task_type": training_ctx.task_type.value,
            "model_type": training_ctx.model_type,
            "template_name": training_ctx.template_name,
            "model_config": training_ctx.model_config,
            "training_config": training_ctx.training_config,
            "snapshot_metadata_path": snapshot_meta.snapshot_metadata_path,
            "data_start_time": snapshot_meta.start_time.isoformat(),
            "data_end_time": snapshot_meta.end_time.isoformat(),
            "train_indices_count": len(snapshot_meta.train_indices),
            "val_indices_count": len(snapshot_meta.val_indices),
            "test_indices_count": len(snapshot_meta.test_indices),
            "resample_freq": snapshot_meta.resample_freq,
            "metrics": metrics_values,
            "training_info": training_info,
            "created_at": datetime.now(UTC).isoformat(),
        }

        if mlflow_run_id is not None:
            metadata["mlflow_run_id"] = mlflow_run_id

        if cv_results is not None:
            metadata["cv"] = cv_results

        metadata_path = os.path.join(artifact_dir, "training_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info("Saved training metadata to %s", metadata_path)
