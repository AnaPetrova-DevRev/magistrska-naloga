"""Ensemble model that combines predictions from multiple base models.

Supports weighted averaging of predictions from any registered BaseModel
subclasses. Weights can be uniform (equal) or optimized on validation MAE.

The ensemble trains each base model from scratch during fit(), then
computes per-model validation predictions to determine optimal weights.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from modeling.base_model import BaseModel
from modeling.core_types import (
    Dataset,
    PredictionTrajectory,
    TaskType,
)

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """Ensemble model that averages predictions from multiple base models.

    Configuration options (model_config):
        base_models: list[dict] - each dict has "model_type" and "template_name".
        aggregation: str - "mean" (uniform average) or "weighted" (MAE-based).
            Default "weighted".

    Training configuration (training_config):
        Passed through to each base model (they use their own template's
        training_config, not this one).
    """

    MODEL_TYPE: str = "Ensemble"
    SUPPORTED_TASK_TYPES: set = {TaskType.EVENT_RATE, TaskType.TIME_TO_EVENT}

    def __init__(
        self,
        task_type: TaskType,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
    ):
        super().__init__(task_type, model_config, training_config)
        self._base_models: List[BaseModel] = []
        self._weights: Optional[np.ndarray] = None
        self._base_model_specs: List[Dict[str, Any]] = model_config.get("base_models", [])
        self._aggregation: str = model_config.get("aggregation", "weighted")

    def _load_template(self, template_name: str) -> Dict[str, Any]:
        """Load a model template JSON file."""
        template_path = os.path.join("model_configs", f"{template_name}.json")
        with open(template_path, "r") as f:
            return json.load(f)

    def _create_base_model(self, spec: Dict[str, Any]) -> BaseModel:
        """Create a base model instance from a spec dict."""
        from modeling.model_factory import create_model_from_config

        template = self._load_template(spec["template_name"])
        model_config = template.get("model_config", {})
        training_config = template.get("training_config", {})

        return create_model_from_config(
            model_type=spec["model_type"],
            task_type=self.task_type,
            model_config=model_config,
            training_config=training_config,
        )

    def fit(self, dataset: Dataset) -> Dict[str, Any]:
        self._validate_dataset_type(dataset)

        if not self._base_model_specs:
            raise ValueError("Ensemble requires at least one base model in model_config.base_models")

        target = self._get_target_array(dataset)
        n_models = len(self._base_model_specs)
        self._base_models = []
        base_infos = []

        # Train each base model
        for i, spec in enumerate(self._base_model_specs):
            logger.info(
                "Ensemble: training base model %d/%d: %s (%s)",
                i + 1, n_models, spec["model_type"], spec["template_name"],
            )
            model = self._create_base_model(spec)
            info = model.fit(dataset)
            self._base_models.append(model)
            base_infos.append({
                "model_type": spec["model_type"],
                "template_name": spec["template_name"],
                "fit_info": {k: v for k, v in info.items() if k != "warnings"},
            })

        # Compute weights from validation predictions
        if dataset.val_indices:
            val_true = target[dataset.val_indices]
            val_preds = []
            val_maes = []

            for model in self._base_models:
                preds = model.predict_pointwise(dataset, dataset.val_indices)
                val_preds.append(preds)
                mae = float(np.mean(np.abs(val_true - preds)))
                val_maes.append(mae)

            if self._aggregation == "weighted" and all(m >= 0 for m in val_maes):
                # Inverse-MAE weighting: better models get higher weight
                # Clamp to avoid infinite weight when a model has perfect MAE
                inv_maes = np.array([1.0 / max(m, 1e-8) for m in val_maes])
                self._weights = inv_maes / inv_maes.sum()
            else:
                self._weights = np.ones(n_models) / n_models

            # Compute ensemble validation MAE for info
            ensemble_preds = np.zeros_like(val_preds[0])
            for w, preds in zip(self._weights, val_preds):
                ensemble_preds += w * preds
            ensemble_val_mae = float(np.mean(np.abs(val_true - ensemble_preds)))
        else:
            self._weights = np.ones(n_models) / n_models
            val_maes = [None] * n_models
            ensemble_val_mae = None

        self._is_fitted = True

        fit_info = {
            "n_base_models": n_models,
            "aggregation": self._aggregation,
            "weights": self._weights.tolist(),
            "base_model_val_maes": val_maes,
            "ensemble_val_mae": ensemble_val_mae,
            "base_models": base_infos,
        }
        return self._run_validation(dataset, fit_info)

    def predict_pointwise(
        self, dataset: Dataset, indices: Sequence[int],
    ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        self._validate_dataset_type(dataset)

        all_preds = []
        for model in self._base_models:
            preds = model.predict_pointwise(dataset, indices)
            all_preds.append(preds)

        ensemble_preds = np.zeros_like(all_preds[0])
        for w, preds in zip(self._weights, all_preds):
            ensemble_preds += w * preds

        return np.maximum(ensemble_preds, 0.0)

    def forecast(
        self,
        dataset: Dataset,
        horizon_minutes: int,
        sync_unit_name: str = "",
        reference_time: Optional[datetime] = None,
    ) -> PredictionTrajectory:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before forecasting")
        self._validate_dataset_type(dataset)

        # Get forecasts from all base models
        trajectories = []
        for model in self._base_models:
            traj = model.forecast(
                dataset, horizon_minutes, sync_unit_name, reference_time,
            )
            trajectories.append(traj)

        # Average the point forecasts
        n_points = horizon_minutes
        point_forecasts = np.zeros(n_points)
        lower_bounds = np.zeros(n_points)
        upper_bounds = np.zeros(n_points)
        uncertainties = np.zeros(n_points)

        for w, traj in zip(self._weights, trajectories):
            for i, pt in enumerate(traj.points[:n_points]):
                point_forecasts[i] += w * pt.expected_events
                if pt.lower_quantile is not None:
                    lower_bounds[i] += w * pt.lower_quantile
                if pt.upper_quantile is not None:
                    upper_bounds[i] += w * pt.upper_quantile
                if pt.uncertainty is not None:
                    uncertainties[i] += w * pt.uncertainty

        point_forecasts = np.maximum(point_forecasts, 0.0)
        lower_bounds = np.maximum(lower_bounds, 0.0)

        # Use timestamps from first trajectory
        timestamps = [pt.timestamp for pt in trajectories[0].points[:n_points]]

        points = self._create_prediction_points(
            timestamps, point_forecasts, lower_bounds, upper_bounds, uncertainties,
        )
        return PredictionTrajectory(
            sync_unit_name=sync_unit_name,
            task_type=self.task_type,
            horizon_minutes=horizon_minutes,
            points=points,
            model_id="ensemble",
        )

    def save(self, path: str) -> None:
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        os.makedirs(path, exist_ok=True)

        # Save each base model
        for i, (model, spec) in enumerate(zip(self._base_models, self._base_model_specs)):
            sub_path = os.path.join(path, f"base_model_{i}")
            model.save(sub_path)

        # Save ensemble config
        config = {
            "model_type": self.MODEL_TYPE,
            "task_type": self.task_type.value,
            "model_config": self.model_config,
            "training_config": self.training_config,
            "weights": self._weights.tolist(),
            "base_model_specs": self._base_model_specs,
            "aggregation": self._aggregation,
        }
        with open(os.path.join(path, "ensemble_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str, task_type: TaskType, model_config: Dict[str, Any]):
        from modeling.model_factory import load_model_from_artifacts

        config_path = os.path.join(path, "ensemble_config.json")
        with open(config_path, "r") as f:
            saved = json.load(f)

        training_config = saved.get("training_config", {})
        ensemble = cls(task_type, model_config, training_config)
        ensemble._weights = np.array(saved["weights"])
        ensemble._aggregation = saved.get("aggregation", "weighted")
        ensemble._base_model_specs = saved["base_model_specs"]

        # Load each base model
        ensemble._base_models = []
        for i, spec in enumerate(ensemble._base_model_specs):
            sub_path = os.path.join(path, f"base_model_{i}")
            template = ensemble._load_template(spec["template_name"])
            sub_model_config = template.get("model_config", {})
            model = load_model_from_artifacts(
                model_type=spec["model_type"],
                task_type=task_type,
                artifact_path=sub_path,
                model_config=sub_model_config,
            )
            ensemble._base_models.append(model)

        ensemble._is_fitted = True
        return ensemble
