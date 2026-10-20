"""Model factory for creating and loading model instances.

This module provides a registry-based factory for instantiating models
based on model type and task type. It supports both creating new models
for training and loading trained models from artifacts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type

from modeling.base_model import BaseModel
from modeling.core_types import TaskType, TrainingContext
from modeling.models.arima_model import ARIMAEventRateModel
from modeling.models.chronos_model import ChronosForecastModel
from modeling.models.cox_ph_model import CoxPHTimeToEventModel
from modeling.models.exponential_model import ExponentialTimeToEventModel
from modeling.models.gamma_model import GammaTimeToEventModel
from modeling.models.kaplan_meier_model import KaplanMeierTimeToEventModel
from modeling.models.loglogistic_model import LogLogisticTimeToEventModel
from modeling.models.lognormal_model import LogNormalTimeToEventModel
from modeling.models.nhits_model import NHiTSForecastModel
from modeling.models.rnn_model import RNNForecastModel
from modeling.models.rnn_gpu_model import RNNGPUForecastModel
from modeling.models.tft_model import TFTForecastModel
from modeling.models.ensemble_model import EnsembleModel
from modeling.models.hurdle_event_rate_model import HurdleEventRateModel
from modeling.models.weibull_model import WeibullTimeToEventModel
from modeling.models.thp_model import TransformerHawkesModel
from modeling.models.zero_inflated_tte_model import ZeroInflatedTTEModel

logger = logging.getLogger(__name__)


_MODEL_REGISTRY: Dict[tuple, Type[BaseModel]] = {
    ("ARIMA", TaskType.EVENT_RATE): ARIMAEventRateModel,
    ("Chronos", TaskType.EVENT_RATE): ChronosForecastModel,
    ("Ensemble", TaskType.EVENT_RATE): EnsembleModel,
    ("Ensemble", TaskType.TIME_TO_EVENT): EnsembleModel,
    ("Chronos", TaskType.TIME_TO_EVENT): ChronosForecastModel,
    ("CoxPH", TaskType.TIME_TO_EVENT): CoxPHTimeToEventModel,
    ("Exponential", TaskType.TIME_TO_EVENT): ExponentialTimeToEventModel,
    ("Gamma", TaskType.TIME_TO_EVENT): GammaTimeToEventModel,
    ("HurdleNB", TaskType.EVENT_RATE): HurdleEventRateModel,
    ("KaplanMeier", TaskType.TIME_TO_EVENT): KaplanMeierTimeToEventModel,
    ("LogLogistic", TaskType.TIME_TO_EVENT): LogLogisticTimeToEventModel,
    ("LogNormal", TaskType.TIME_TO_EVENT): LogNormalTimeToEventModel,
    ("NHiTS", TaskType.EVENT_RATE): NHiTSForecastModel,
    ("NHiTS", TaskType.TIME_TO_EVENT): NHiTSForecastModel,
    ("RNN", TaskType.EVENT_RATE): RNNForecastModel,
    ("RNN", TaskType.TIME_TO_EVENT): RNNForecastModel,
    ("RNN_GPU", TaskType.EVENT_RATE): RNNGPUForecastModel,
    ("RNN_GPU", TaskType.TIME_TO_EVENT): RNNGPUForecastModel,
    ("TFT", TaskType.EVENT_RATE): TFTForecastModel,
    ("TFT", TaskType.TIME_TO_EVENT): TFTForecastModel,
    ("THP", TaskType.EVENT_RATE): TransformerHawkesModel,
    ("THP", TaskType.TIME_TO_EVENT): TransformerHawkesModel,
    ("Weibull", TaskType.TIME_TO_EVENT): WeibullTimeToEventModel,
    ("ZI_Gamma", TaskType.TIME_TO_EVENT): ZeroInflatedTTEModel,
    ("ZI_LogNormal", TaskType.TIME_TO_EVENT): ZeroInflatedTTEModel,
    ("ZI_Weibull", TaskType.TIME_TO_EVENT): ZeroInflatedTTEModel,
}


def _resolve_class(model_type: str, task_type: TaskType) -> Type[BaseModel]:
    """Look up a model class in the registry.

    Coerces string task types to TaskType and raises ValueError with
    a helpful message when the combination is not registered.
    """
    if isinstance(task_type, str):
        task_type = TaskType(task_type)

    key = (model_type, task_type)

    if key not in _MODEL_REGISTRY:
        available = ", ".join(
            f"{mt}/{tt.value}" for (mt, tt) in _MODEL_REGISTRY.keys()
        )
        raise ValueError(
            f"No model registered for {model_type}/{task_type.value}. "
            f"Available: {available}"
        )

    return _MODEL_REGISTRY[key]


def register_model(
    model_type: str,
    task_type: TaskType,
    model_class: Type[BaseModel],
) -> None:
    """Register a new model class in the factory.

    Args:
        model_type: The model type name (e.g., "ARIMA", "NHiTS").
        task_type: The task type this model supports.
        model_class: The BaseModel subclass to register.
    """
    key = (model_type, task_type)
    if key in _MODEL_REGISTRY:
        logger.warning(
            "Overwriting existing model registration for %s/%s",
            model_type, task_type.value
        )
    _MODEL_REGISTRY[key] = model_class
    logger.info("Registered model %s for task %s", model_type, task_type.value)


def get_registered_models() -> Dict[str, list]:
    """Get all registered models grouped by model type.

    Returns:
        Dict mapping model type names to lists of supported TaskTypes.
    """
    result: Dict[str, list] = {}
    for (model_type, task_type), _ in _MODEL_REGISTRY.items():
        if model_type not in result:
            result[model_type] = []
        result[model_type].append(task_type)
    return result


def is_model_supported(model_type: str, task_type: TaskType) -> bool:
    """Check if a model type/task type combination is supported.

    Args:
        model_type: The model type name.
        task_type: The task type.

    Returns:
        True if the combination is registered.
    """
    if isinstance(task_type, str):
        task_type = TaskType(task_type)
    return (model_type, task_type) in _MODEL_REGISTRY


def create_model(training_ctx: TrainingContext) -> BaseModel:
    """Create a new model instance for training.

    Args:
        training_ctx: The training context containing model type, task type,
                     and configuration.

    Returns:
        A BaseModel instance ready for training.

    Raises:
        ValueError: If the model type/task type combination is not registered.
    """
    model_class = _resolve_class(training_ctx.model_type, training_ctx.task_type)

    logger.info(
        "Creating model %s for task %s",
        training_ctx.model_type, training_ctx.task_type.value
    )

    return model_class(
        task_type=training_ctx.task_type,
        model_config=training_ctx.model_config,
        training_config=training_ctx.training_config,
    )


def create_model_from_config(
    model_type: str,
    task_type: TaskType,
    model_config: Dict[str, Any],
    training_config: Optional[Dict[str, Any]] = None,
) -> BaseModel:
    """Create a model instance directly from configuration dicts.

    This is a lower-level alternative to create_model() when you don't
    have a full TrainingContext.

    Args:
        model_type: The model type name.
        task_type: The task type.
        model_config: Model-specific hyperparameters.
        training_config: Training parameters (optional).

    Returns:
        A BaseModel instance ready for training.
    """
    model_class = _resolve_class(model_type, task_type)

    return model_class(
        task_type=task_type,
        model_config=model_config,
        training_config=training_config or {},
    )


def load_model_from_artifacts(
    model_type: str,
    task_type: TaskType,
    artifact_path: str,
    model_config: Dict[str, Any],
) -> BaseModel:
    """Load a trained model from saved artifacts.

    Args:
        model_type: The model type name.
        task_type: The task type.
        artifact_path: Path to the directory containing model artifacts.
        model_config: The model configuration used during training.

    Returns:
        A BaseModel instance ready for prediction.

    Raises:
        ValueError: If the model type/task type combination is not registered.
    """
    model_class = _resolve_class(model_type, task_type)

    logger.info(
        "Loading model %s for task %s from %s",
        model_type, task_type.value, artifact_path
    )

    return model_class.load(
        path=artifact_path,
        task_type=task_type,
        model_config=model_config,
    )


def get_model_class(
    model_type: str,
    task_type: TaskType,
) -> Type[BaseModel]:
    """Get the model class for a given type/task combination.

    Args:
        model_type: The model type name.
        task_type: The task type.

    Returns:
        The registered BaseModel subclass.

    Raises:
        ValueError: If the combination is not registered.
    """
    return _resolve_class(model_type, task_type)
