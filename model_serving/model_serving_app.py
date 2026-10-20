"""Model serving app for the orchestrator system.

This app handles serving predictions from trained forecasting models.
It supports loading models by stage (Production/Staging) or by explicit
run ID, and can cache predictions for efficient serving.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app_base import App
from orchestrator_message import OrchestratorResponse
import storage_access
import sync_unit as sync_unit_module
from app_actions_constants import GET_PREDICTION, REFRESH_PREDICTIONS
from modeling.core_types import (
    Dataset,
    DatasetConfig,
    EventRateDataset,
    PredictionPoint,
    PredictionResult,
    PredictionTrajectory,
    TaskType,
)
from modeling.dataset_builder import build_event_rate_dataset, build_time_to_event_dataset
from modeling.model_factory import load_model_from_artifacts
from modeling.mlflow_utils import is_mlflow_available, MlflowClient, get_registered_model_name

logger = logging.getLogger(__name__)

PREDICTION_CACHE_FOLDER = "prediction_cache"


class ModelServingApp(App):
    """App for serving predictions from trained models.

    Actions:
        get_prediction: Get a forecast trajectory for a sync unit.
        refresh_predictions: Precompute predictions if cache is stale.

    The app supports:
    - Model selection by stage (Production/Staging) or explicit model ID
    - Caching of predictions for low-latency serving
    - Graceful error handling with structured error responses
    """

    def __init__(self, sa: Optional[storage_access.StorageAccess] = None):
        self._sa = sa or storage_access.StorageAccess()

    def get_name(self) -> str:
        return "model_serving"

    def get_available_actions(self) -> list[str]:
        return [GET_PREDICTION, REFRESH_PREDICTIONS]

    def execute_action(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        action: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorResponse:
        """Execute a model serving action.

        Args:
            sync_unit_obj: The sync unit to serve predictions for.
            action: The action to perform.
            parameters: Action parameters including task_type, horizon,
                       and model_selection.

        Returns:
            OrchestratorResponse with prediction results.
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

        if action == GET_PREDICTION:
            return self._handle_get_prediction(sync_unit_obj, parameters)
        elif action == REFRESH_PREDICTIONS:
            return self._handle_refresh_predictions(sync_unit_obj, parameters)

        raise ValueError(f"Unknown action: {action}")

    def _handle_get_prediction(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        parameters: Dict[str, Any],
    ) -> OrchestratorResponse:
        """Handle the get_prediction action.

        Expected parameters:
            task_type: "event_rate" or "time_to_event"
            horizon_minutes: int (default 60)
            model_selection: {
                mode: "stage", "explicit", or "heuristic"
                stage: "Production" or "Staging" (for mode="stage")
                model_id: str (for mode="explicit")
                granularity: list of grouping keys (for mode="heuristic",
                    default ["hour_of_day", "day_of_week"])
            }
        """
        task_type_str = parameters.get("task_type", "event_rate")
        horizon_minutes = int(parameters.get("horizon_minutes", 60))
        model_selection = parameters.get("model_selection", {"mode": "stage", "stage": "Production"})
        reference_time_raw = parameters.get("reference_time")
        reference_time = (
            datetime.fromisoformat(reference_time_raw)
            if isinstance(reference_time_raw, str)
            else reference_time_raw
        )

        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            result = PredictionResult(
                ok=False,
                error_type="invalid_task_type",
                error_message=f"Invalid task_type: {task_type_str}",
            )
            return self._create_response(False, result)

        mode = model_selection.get("mode", "stage")

        if mode == "heuristic":
            return self._handle_heuristic_prediction(
                sync_unit_obj, task_type, horizon_minutes,
                model_selection,
            )

        if mode == "multi":
            return self._handle_multi_prediction(
                sync_unit_obj, task_type, horizon_minutes,
                model_selection,
            )

        try:
            model, model_info = self._load_model_for_sync_unit(
                sync_unit_obj, task_type, model_selection,
            )

            # Include covariates the model was trained with
            model_covs = getattr(model, "_covariate_names", [])
            history_dataset = self._build_history_dataset(
                sync_unit_obj, task_type,
                covariates=model_covs or None,
                reference_time=reference_time,
                resample_freq=getattr(model, "_freq", None),
            )

            trajectory = model.forecast(
                history_dataset,
                horizon_minutes=horizon_minutes,
                sync_unit_name=sync_unit_obj.get_name(),
                reference_time=reference_time,
            )
            trajectory.model_id = model_info.get(
                "model_id", "unknown",
            )

            result = PredictionResult(ok=True, prediction=trajectory)

            response = self._create_response(True, result)
            response.message = (
                f"Prediction generated for {horizon_minutes} minutes"
            )
            return response

        except Exception as e:
            logger.error(
                "Prediction failed for %s: %s",
                sync_unit_obj.get_name(), e, exc_info=True,
            )
            result = PredictionResult(
                ok=False,
                error_type="prediction_error",
                error_message=str(e),
            )
            return self._create_response(False, result)

    def _handle_refresh_predictions(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        parameters: Dict[str, Any],
    ) -> OrchestratorResponse:
        """Handle the refresh_predictions action.

        Expected parameters:
            task_type: "event_rate" or "time_to_event"
            horizon_minutes: int (default 60)
            min_remaining_minutes: int (default 15)
            model_selection: same as get_prediction
        """
        task_type_str = parameters.get("task_type", "event_rate")
        horizon_minutes = int(parameters.get("horizon_minutes", 60))
        min_remaining = int(parameters.get("min_remaining_minutes", 15))
        model_selection = parameters.get("model_selection", {"mode": "stage", "stage": "Production"})

        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            return OrchestratorResponse(
                success=False,
                message=f"Invalid task_type: {task_type_str}",
                is_done=True,
            )

        try:
            remaining = self._get_remaining_horizon(
                sync_unit_obj, task_type,
            )

            if remaining >= min_remaining:
                return OrchestratorResponse(
                    success=True,
                    message=(
                        f"No refresh needed "
                        f"({remaining} minutes remaining)"
                    ),
                    is_done=True,
                )

            mode = model_selection.get("mode", "stage")

            # Delegate to get_prediction for multi/heuristic so the
            # blending and caching logic stays in one place.
            if mode in ("multi", "heuristic"):
                pred_resp = self._handle_get_prediction(
                    sync_unit_obj,
                    {
                        "task_type": task_type_str,
                        "horizon_minutes": horizon_minutes,
                        "model_selection": model_selection,
                    },
                )
                if pred_resp.success and pred_resp.data:
                    pr = pred_resp.data["prediction_result"]
                    if pr.ok:
                        self._save_prediction_cache(
                            sync_unit_obj, task_type,
                            pr.prediction,
                        )
                return pred_resp

            model, model_info = self._load_model_for_sync_unit(
                sync_unit_obj, task_type, model_selection,
            )

            model_covs = getattr(model, "_covariate_names", [])
            history_dataset = self._build_history_dataset(
                sync_unit_obj, task_type,
                covariates=model_covs or None,
            )

            trajectory = model.forecast(
                history_dataset,
                horizon_minutes=horizon_minutes,
                sync_unit_name=sync_unit_obj.get_name(),
            )
            trajectory.model_id = model_info.get(
                "model_id", "unknown",
            )

            self._save_prediction_cache(
                sync_unit_obj, task_type, trajectory,
            )

            return OrchestratorResponse(
                success=True,
                message=(
                    f"Predictions refreshed for "
                    f"{horizon_minutes} minutes"
                ),
                is_done=True,
            )

        except Exception as e:
            logger.error(
                "Refresh predictions failed for %s: %s",
                sync_unit_obj.get_name(), e, exc_info=True,
            )
            return OrchestratorResponse(
                success=False,
                message=f"Refresh failed: {e}",
                is_done=True,
            )

    def _handle_heuristic_prediction(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        task_type: TaskType,
        horizon_minutes: int,
        model_selection: Dict[str, Any],
    ) -> OrchestratorResponse:
        """Generate predictions from historical averages at similar timestamps.

        Groups historical data by time-of-day and day-of-week (configurable
        via ``model_selection.granularity``), computes the mean value for each
        group, and produces a minute-level prediction trajectory.

        Config example::

            "model_selection": {
                "mode": "heuristic",
                "granularity": ["hour_of_day", "day_of_week"]
            }

        Supported granularity keys: ``hour_of_day``, ``day_of_week``,
        ``minute_of_hour``.  Finer granularity = more specific but noisier.
        """
        try:
            import pandas as pd

            dataset = self._build_history_dataset(sync_unit_obj, task_type)

            timestamps = list(dataset.timestamps)
            if task_type == TaskType.EVENT_RATE:
                values = np.asarray(dataset.events_per_minute, dtype=np.float64)
            else:
                values = np.asarray(dataset.minutes_to_next_event, dtype=np.float64)

            ts_series = pd.Series(timestamps)
            df = pd.DataFrame({
                "value": values,
                "hour_of_day": ts_series.dt.hour,
                "day_of_week": ts_series.dt.dayofweek,
                "minute_of_hour": ts_series.dt.minute,
            })

            granularity = model_selection.get(
                "granularity", ["hour_of_day", "day_of_week"]
            )
            valid_keys = {"hour_of_day", "day_of_week", "minute_of_hour"}
            group_keys = [k for k in granularity if k in valid_keys]
            if not group_keys:
                group_keys = ["hour_of_day", "day_of_week"]

            group_means = df.groupby(group_keys)["value"].mean()

            def _lookup(ts: datetime) -> float:
                key_map = {
                    "hour_of_day": ts.hour,
                    "day_of_week": ts.weekday(),
                    "minute_of_hour": ts.minute,
                }
                key = tuple(key_map[k] for k in group_keys)
                if len(key) == 1:
                    key = key[0]
                if key in group_means.index:
                    return float(group_means.loc[key])
                return float(group_means.mean())

            # Generate minute-level predictions starting from the last timestamp
            last_ts = timestamps[-1]
            points: List[PredictionPoint] = []
            for m in range(1, horizon_minutes + 1):
                future_ts = last_ts + timedelta(minutes=m)
                expected = _lookup(future_ts)
                points.append(PredictionPoint(
                    timestamp=future_ts,
                    expected_events=expected,
                ))

            trajectory = PredictionTrajectory(
                sync_unit_name=sync_unit_obj.get_name(),
                task_type=task_type,
                horizon_minutes=horizon_minutes,
                points=points,
                model_id=f"heuristic_{'_'.join(group_keys)}",
            )

            result = PredictionResult(ok=True, prediction=trajectory)
            response = self._create_response(True, result)
            response.message = (
                f"Heuristic prediction generated for {horizon_minutes} minutes "
                f"(grouped by {', '.join(group_keys)})"
            )
            return response

        except Exception as e:
            logger.error(
                "Heuristic prediction failed for %s: %s",
                sync_unit_obj.get_name(), e, exc_info=True,
            )
            result = PredictionResult(
                ok=False,
                error_type="heuristic_prediction_error",
                error_message=str(e),
            )
            return self._create_response(False, result)

    def _handle_multi_prediction(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        task_type: TaskType,
        horizon_minutes: int,
        model_selection: Dict[str, Any],
    ) -> OrchestratorResponse:
        """Serve predictions from multiple models and blend them.

        Config example::

            "model_selection": {
                "mode": "multi",
                "models": [
                    {"mode": "stage", "stage": "Production"},
                    {"mode": "explicit", "model_id": "path/to/model"}
                ],
                "aggregation": "mean"
            }

        Each entry in ``models`` uses the same single-model selection
        logic (``stage``, ``explicit``, or ``heuristic``).
        ``aggregation`` controls blending: ``"mean"`` (default) averages
        all trajectories point-wise.
        """
        sub_specs = model_selection.get("models", [])
        if not sub_specs:
            result = PredictionResult(
                ok=False,
                error_type="invalid_config",
                error_message=(
                    "mode 'multi' requires a non-empty 'models' list"
                ),
            )
            return self._create_response(False, result)

        aggregation = model_selection.get("aggregation", "mean")

        # Pre-load all non-heuristic models to collect covariate
        # names, then build one dataset with the union of covariates.
        loaded_models: List[tuple] = []  # (spec, model, model_info)
        all_cov_names: set = set()
        load_errors: List[str] = []

        for spec in sub_specs:
            if spec.get("mode", "stage") == "heuristic":
                loaded_models.append((spec, None, None))
                continue
            try:
                model, model_info = self._load_model_for_sync_unit(
                    sync_unit_obj, task_type, spec,
                )
                model_covs = getattr(
                    model, "_covariate_names", [],
                )
                all_cov_names.update(model_covs)
                loaded_models.append((spec, model, model_info))
            except Exception as e:
                load_errors.append(f"{spec}: {e}")
                logger.warning(
                    "Multi-model: failed to load %s: %s",
                    spec, e,
                )

        try:
            history_dataset = self._build_history_dataset(
                sync_unit_obj, task_type,
                covariates=sorted(all_cov_names) or None,
            )
        except Exception as e:
            result = PredictionResult(
                ok=False,
                error_type="dataset_error",
                error_message=str(e),
            )
            return self._create_response(False, result)

        trajectories: List[PredictionTrajectory] = []
        model_ids: List[str] = []
        errors: List[str] = list(load_errors)

        for spec, model, model_info in loaded_models:
            spec_mode = spec.get("mode", "stage")
            try:
                if spec_mode == "heuristic":
                    resp = self._handle_heuristic_prediction(
                        sync_unit_obj, task_type,
                        horizon_minutes, spec,
                    )
                    if resp.success and resp.data:
                        pr = resp.data["prediction_result"]
                        trajectories.append(pr.prediction)
                        model_ids.append(
                            pr.prediction.model_id,
                        )
                    else:
                        errors.append(
                            f"heuristic: {resp.message}"
                        )
                    continue

                if model is None:
                    continue

                traj = model.forecast(
                    history_dataset,
                    horizon_minutes=horizon_minutes,
                    sync_unit_name=sync_unit_obj.get_name(),
                )
                traj.model_id = model_info.get(
                    "model_id", "unknown",
                )
                trajectories.append(traj)
                model_ids.append(traj.model_id)
            except Exception as e:
                errors.append(f"{spec}: {e}")
                logger.warning(
                    "Multi-model: failed to forecast with "
                    "%s: %s", spec, e,
                )

        if not trajectories:
            result = PredictionResult(
                ok=False,
                error_type="multi_prediction_error",
                error_message=(
                    f"All models failed: {errors}"
                ),
            )
            return self._create_response(False, result)

        blended = self._blend_trajectories(
            trajectories, sync_unit_obj.get_name(),
            task_type, horizon_minutes, aggregation,
        )
        blended.model_id = f"multi({', '.join(model_ids)})"

        result = PredictionResult(ok=True, prediction=blended)
        response = self._create_response(True, result)
        n_ok = len(trajectories)
        n_total = len(sub_specs)
        response.message = (
            f"Multi-model prediction: {n_ok}/{n_total} models, "
            f"aggregation={aggregation}, "
            f"horizon={horizon_minutes}min"
        )
        if errors:
            response.message += f" (errors: {errors})"
        return response

    @staticmethod
    def _blend_trajectories(
        trajectories: List[PredictionTrajectory],
        sync_unit_name: str,
        task_type: TaskType,
        horizon_minutes: int,
        aggregation: str = "mean",
    ) -> PredictionTrajectory:
        """Blend multiple trajectories into one.

        Currently supports ``"mean"`` (uniform average).  All
        trajectories must have the same number of points and aligned
        timestamps.
        """
        if len(trajectories) == 1:
            return trajectories[0]

        n_points = len(trajectories[0].points)
        blended_points: List[PredictionPoint] = []

        for i in range(n_points):
            values = [t.points[i].expected_events for t in trajectories]
            avg = sum(values) / len(values)
            blended_points.append(PredictionPoint(
                timestamp=trajectories[0].points[i].timestamp,
                expected_events=avg,
            ))

        return PredictionTrajectory(
            sync_unit_name=sync_unit_name,
            task_type=task_type,
            horizon_minutes=horizon_minutes,
            points=blended_points,
            model_id="blended",
        )

    def _create_response(
        self,
        success: bool,
        result: PredictionResult,
    ) -> OrchestratorResponse:
        """Create an OrchestratorResponse with a PredictionResult."""
        response = OrchestratorResponse(
            success=success,
            message="Prediction completed" if success else "Prediction failed",
            is_done=True,
        )
        response.data = {"prediction_result": result}
        return response

    def _load_model_for_sync_unit(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        task_type: TaskType,
        model_selection: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        """Load a model for the sync unit based on selection criteria.

        Args:
            sync_unit_obj: The sync unit.
            task_type: The task type.
            model_selection: Selection criteria (mode, stage, model_id).

        Returns:
            Tuple of (model instance, model info dict).
        """
        mode = model_selection.get("mode", "stage")

        if mode == "explicit":
            model_id = model_selection.get("model_id")
            if not model_id:
                raise ValueError("model_id required for explicit mode")
            return self._load_model_by_id(model_id, task_type)
        else:
            stage = model_selection.get("stage", "Production")
            return self._load_model_by_stage(sync_unit_obj, task_type, stage)

    def _load_model_by_stage(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        task_type: TaskType,
        stage: str,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Load the model for a sync unit by MLflow stage.

        When MLflow is available, queries the model registry for the
        version at the requested stage, then resolves the local artifact
        directory via the ``mlflow_run_id`` stored in each model's
        ``training_metadata.json``.

        Falls back to the most recent model on disk when MLflow is not
        installed or when no version is registered at the requested stage.
        """
        models_dir = os.path.join(
            "data",
            sync_unit_obj.get_path_for_sync_unit(),
            "models",
        )

        if not os.path.exists(models_dir):
            raise FileNotFoundError(
                f"No models directory found for "
                f"{sync_unit_obj.get_name()}"
            )

        # Scan local model directories once (shared by both paths).
        model_dirs = []
        for name in os.listdir(models_dir):
            path = os.path.join(models_dir, name)
            if not os.path.isdir(path):
                continue
            meta_path = os.path.join(path, "training_metadata.json")
            if os.path.exists(meta_path):
                model_dirs.append(
                    (name, path, os.path.getmtime(meta_path))
                )

        if not model_dirs:
            raise FileNotFoundError(
                f"No trained models found for "
                f"{sync_unit_obj.get_name()}"
            )

        # --- MLflow path: resolve stage → run_id → local dir -----------
        if is_mlflow_available() and MlflowClient is not None:
            artifact_path = self._resolve_stage_via_mlflow(
                sync_unit_obj, task_type, stage, model_dirs,
            )
            if artifact_path is not None:
                return self._load_from_artifact_path(
                    artifact_path, task_type,
                )

            logger.info(
                "No MLflow version at stage '%s' for %s — "
                "falling back to most recent model on disk",
                stage, sync_unit_obj.get_name(),
            )

        # --- Filesystem fallback: newest by mtime ---------------------
        model_dirs.sort(key=lambda x: x[2], reverse=True)
        _, latest_path, _ = model_dirs[0]
        return self._load_from_artifact_path(latest_path, task_type)

    def _resolve_stage_via_mlflow(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        task_type: TaskType,
        stage: str,
        model_dirs: List[Tuple[str, str, float]],
    ) -> Optional[str]:
        """Query MLflow for the version at *stage* and match to a local dir.

        Returns the local artifact path if found, ``None`` otherwise.
        """
        try:
            client = MlflowClient()
            model_name = get_registered_model_name(
                sync_unit_obj.get_name(), task_type,
            )
            versions = client.get_latest_versions(
                model_name, stages=[stage],
            )
            if not versions:
                return None

            target_run_id = versions[0].run_id

            for _, path, _ in model_dirs:
                meta_path = os.path.join(
                    path, "training_metadata.json",
                )
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                if meta.get("mlflow_run_id") == target_run_id:
                    logger.info(
                        "Resolved MLflow %s stage to %s "
                        "(run_id=%s)",
                        stage, path, target_run_id,
                    )
                    return path

            logger.warning(
                "MLflow run_id %s not found in local model dirs",
                target_run_id,
            )
        except Exception as e:
            logger.warning(
                "MLflow stage lookup failed: %s", e,
            )
        return None

    @staticmethod
    def _load_from_artifact_path(
        artifact_path: str,
        task_type: TaskType,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Load a model from a local artifact directory."""
        metadata_path = os.path.join(
            artifact_path, "training_metadata.json",
        )
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        model = load_model_from_artifacts(
            model_type=metadata.get("model_type"),
            task_type=task_type,
            artifact_path=artifact_path,
            model_config=metadata.get("model_config", {}),
        )

        model_info = {
            "model_id": os.path.basename(artifact_path),
            "model_type": metadata.get("model_type"),
            "artifact_path": artifact_path,
        }
        return model, model_info

    def _load_model_by_id(
        self,
        model_id: str,
        task_type: TaskType,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Load a model by explicit ID/path.

        For MLflow integration, model_id would be an MLflow run_id and
        we would use model_selection_helpers.py to resolve it.
        """
        if not os.path.exists(model_id):
            raise FileNotFoundError(f"Model path not found: {model_id}")

        metadata_path = os.path.join(model_id, "training_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"No training_metadata.json in {model_id}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        model_type = metadata.get("model_type")
        model_config = metadata.get("model_config", {})

        model = load_model_from_artifacts(
            model_type=model_type,
            task_type=task_type,
            artifact_path=model_id,
            model_config=model_config,
        )

        model_info = {
            "model_id": os.path.basename(model_id),
            "model_type": model_type,
            "artifact_path": model_id,
        }

        return model, model_info

    def _build_history_dataset(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        task_type: TaskType,
        covariates: Optional[List[str]] = None,
        reference_time: Optional[datetime] = None,
        resample_freq: Optional[str] = None,
    ) -> Dataset:
        """Build a dataset with recent history for forecasting.

        Args:
            sync_unit_obj: The sync unit.
            task_type: The task type.
            covariates: Optional list of covariate names to include
                (e.g. from a model's stored covariate_names).

        Returns:
            Dataset with recent data for model context.
        """
        history_end = (
            reference_time - timedelta(minutes=1)
            if reference_time is not None else datetime.max
        )
        config = DatasetConfig(
            task_type=task_type,
            start_time=datetime.min,
            end_time=history_end,
            train_ratio=1.0,
            val_ratio=0.0,
            covariates=covariates or [],
            resample_freq=resample_freq,
        )

        if task_type == TaskType.EVENT_RATE:
            dataset, _ = build_event_rate_dataset(sync_unit_obj, config, self._sa)
            return dataset
        elif task_type == TaskType.TIME_TO_EVENT:
            dataset, _ = build_time_to_event_dataset(sync_unit_obj, config, self._sa)
            return dataset
        else:
            raise NotImplementedError(
                f"History dataset for {task_type.value} not yet implemented"
            )

    def _get_remaining_horizon(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        task_type: TaskType,
    ) -> int:
        """Get the number of minutes remaining in the cached prediction.

        Returns 0 if no valid cache exists.
        """
        cache_path = self._get_cache_path(sync_unit_obj, task_type)

        if not os.path.exists(cache_path):
            return 0

        try:
            with open(cache_path, "r") as f:
                cache_data = json.load(f)

            created_at = datetime.fromisoformat(cache_data.get("created_at", ""))
            horizon_minutes = cache_data.get("horizon_minutes", 0)

            elapsed = (datetime.now(UTC) - created_at).total_seconds() / 60.0
            remaining = max(0, int(horizon_minutes - elapsed))

            return remaining

        except Exception as e:
            logger.warning("Failed to read prediction cache: %s", e)
            return 0

    def _save_prediction_cache(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        task_type: TaskType,
        trajectory: PredictionTrajectory,
    ) -> None:
        """Save a prediction trajectory to cache."""
        cache_dir = os.path.join(
            "data",
            sync_unit_obj.get_path_for_sync_unit(),
            PREDICTION_CACHE_FOLDER,
        )
        os.makedirs(cache_dir, exist_ok=True)

        cache_path = self._get_cache_path(sync_unit_obj, task_type)

        cache_data = trajectory.to_dict()

        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)

        logger.info("Saved prediction cache to %s", cache_path)

    def _get_cache_path(
        self,
        sync_unit_obj: sync_unit_module.SyncUnit,
        task_type: TaskType,
    ) -> str:
        """Get the path to the prediction cache file."""
        return os.path.join(
            "data",
            sync_unit_obj.get_path_for_sync_unit(),
            PREDICTION_CACHE_FOLDER,
            f"{task_type.value}_predictions.json",
        )
