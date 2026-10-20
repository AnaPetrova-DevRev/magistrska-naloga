"""Policy loader with support for trained RL agents and heuristic baselines.

All policies implement the Policy interface so they can be evaluated
uniformly by the evaluation engine.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Policy(ABC):
    """Abstract interface for sync scheduling policies."""

    @abstractmethod
    def predict(self, observation: np.ndarray) -> int:
        """Return action: 0=skip, 1=sync."""

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable name for this policy."""

    def reset(self) -> None:
        """Reset any internal state. Called at episode start."""
        pass


# Trained RL policy wrapper


class SB3Policy(Policy):
    """Wraps a Stable-Baselines3 model as a Policy."""

    def __init__(self, model, name: str = "sb3_policy"):
        self._model = model
        self._name = name

    def predict(self, observation: np.ndarray) -> int:
        action, _ = self._model.predict(observation, deterministic=True)
        return int(action)

    def get_name(self) -> str:
        return self._name


# Built-in heuristic policies


class PeriodicSyncPolicy(Policy):
    """Sync every N minutes. Classic baseline."""

    def __init__(self, interval_minutes: int = 5):
        self._interval = interval_minutes
        self._steps_since_sync = 0

    def predict(self, observation: np.ndarray) -> int:
        self._steps_since_sync += 1
        if self._steps_since_sync >= self._interval:
            self._steps_since_sync = 0
            return 1
        return 0

    def get_name(self) -> str:
        return f"periodic_{self._interval}min"

    def reset(self) -> None:
        self._steps_since_sync = 0


class ThresholdPolicy(Policy):
    """Sync when a prediction value exceeds a threshold.

    Requires 'predictions' to be in the observation. Uses the first
    prediction value (index specified by prediction_index within the
    observation).
    """

    def __init__(
        self,
        threshold: float = 0.5,
        prediction_start_index: int = 2,
    ):
        self._threshold = threshold
        self._pred_idx = prediction_start_index

    def predict(self, observation: np.ndarray) -> int:
        if self._pred_idx < len(observation):
            return 1 if observation[self._pred_idx] > self._threshold else 0
        return 0

    def get_name(self) -> str:
        return f"threshold_{self._threshold}"


class AlwaysSyncPolicy(Policy):
    """Always sync. Upper bound on sync cost."""

    def predict(self, observation: np.ndarray) -> int:
        return 1

    def get_name(self) -> str:
        return "always_sync"


class NeverSyncPolicy(Policy):
    """Never sync. Upper bound on AoI penalty."""

    def predict(self, observation: np.ndarray) -> int:
        return 0

    def get_name(self) -> str:
        return "never_sync"


class ModelPredictionPolicy(Policy):
    """Policy that uses raw model predictions to decide sync/skip.

    Two strategies:
      - "tte": syncs when predicted minutes-to-next-event < threshold.
      - "er": accumulates predicted events/min since last sync;
        syncs when the running sum exceeds threshold.

    Requires ``set_predictions()`` to be called with the raw
    (unnormalized) prediction array before evaluation begins.
    """

    def __init__(
        self,
        strategy: str = "er",
        threshold: float = 2.0,
        name: Optional[str] = None,
    ):
        if strategy not in ("tte", "er"):
            raise ValueError(
                f"strategy must be 'tte' or 'er', got '{strategy}'"
            )
        self._strategy = strategy
        self._threshold = threshold
        self._name = name
        self._predictions: Optional[np.ndarray] = None
        self._step: int = 0
        self._cumulative_er: float = 0.0

    def set_predictions(self, predictions: np.ndarray) -> None:
        """Inject raw prediction values (called by the evaluation app)."""
        self._predictions = np.asarray(predictions, dtype=np.float32)

    def predict(self, observation: np.ndarray) -> int:
        if self._predictions is None or self._step >= len(self._predictions):
            self._step += 1
            return 0

        value = float(self._predictions[self._step])
        self._step += 1

        if self._strategy == "tte":
            if value < self._threshold:
                return 1
            return 0
        else:  # er
            self._cumulative_er += value
            if self._cumulative_er >= self._threshold:
                self._cumulative_er = 0.0
                return 1
            return 0

    def get_name(self) -> str:
        if self._name:
            return self._name
        return f"model_pred_{self._strategy}_{self._threshold}"

    def reset(self) -> None:
        self._step = 0
        self._cumulative_er = 0.0


# Heuristic registry

_HEURISTIC_REGISTRY: Dict[str, type] = {
    "periodic": PeriodicSyncPolicy,
    "threshold": ThresholdPolicy,
    "always_sync": AlwaysSyncPolicy,
    "never_sync": NeverSyncPolicy,
    "model_prediction": ModelPredictionPolicy,
}


def get_available_heuristics() -> list[str]:
    return sorted(_HEURISTIC_REGISTRY.keys())


# Loader functions


def _get_sb3_class(algorithm: str):
    """Import and return the SB3 algorithm class by name."""
    import stable_baselines3 as sb3
    algo_map = {"PPO": sb3.PPO, "DQN": sb3.DQN, "A2C": sb3.A2C}
    if algorithm not in algo_map:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. "
            f"Supported: {list(algo_map.keys())}"
        )
    return algo_map[algorithm]


def resolve_trained_policy_file(
    policy_path: str, checkpoint: str = "final",
) -> str:
    """Resolve the exact SB3 archive for a requested checkpoint."""
    if checkpoint == "best":
        best_file = os.path.join(policy_path, "best_model", "best_model.zip")
        if not os.path.isfile(best_file):
            raise FileNotFoundError(
                f"Validation-selected checkpoint not found: {best_file}"
            )
        return best_file
    if checkpoint != "final":
        raise ValueError("checkpoint must be 'final' or 'best'")

    agent_file = os.path.join(policy_path, "agent.zip")
    return agent_file if os.path.isfile(agent_file) else os.path.join(
        policy_path, "agent"
    )


def load_trained_policy(
    policy_path: str,
    name: Optional[str] = None,
    checkpoint: str = "final",
) -> Policy:
    """Load a trained SB3 policy from disk.

    Reads training_metadata.json to determine the algorithm (PPO, DQN, A2C).
    Falls back to PPO if metadata is missing.

    Args:
        policy_path: Directory containing agent.zip or agent file.
        name: Optional human-readable name.
        checkpoint: ``final`` or validation-selected ``best``.

    Returns:
        SB3Policy wrapping the loaded model.
    """
    import json as _json

    algorithm = "PPO"
    meta_path = os.path.join(policy_path, "training_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = _json.load(f)
        algorithm = meta.get("agent_config", {}).get("algorithm", "PPO")

    AlgoClass = _get_sb3_class(algorithm)

    agent_file = resolve_trained_policy_file(policy_path, checkpoint)
    model = AlgoClass.load(agent_file)
    policy_name = name or os.path.basename(policy_path.rstrip("/"))
    return SB3Policy(model, name=policy_name)


def load_heuristic_policy(heuristic_config: Dict[str, Any]) -> Policy:
    """Create a heuristic policy from config.

    Args:
        heuristic_config: Dict with 'name' and optional params.
            E.g., {"name": "periodic", "interval_minutes": 5}

    Returns:
        Policy instance.
    """
    name = heuristic_config.get("name", "periodic")
    if name not in _HEURISTIC_REGISTRY:
        raise ValueError(
            f"Unknown heuristic '{name}'. "
            f"Available: {get_available_heuristics()}"
        )

    cls = _HEURISTIC_REGISTRY[name]
    params = {k: v for k, v in heuristic_config.items() if k != "name"}
    return cls(**params)


def load_policy(
    policy_type: str = "trained",
    policy_path: Optional[str] = None,
    heuristic_config: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    checkpoint: str = "final",
) -> Policy:
    """Load a policy by type.

    Args:
        policy_type: "trained" or "heuristic".
        policy_path: Path to trained model (required if trained).
        heuristic_config: Heuristic config dict (required if heuristic).
        name: Optional display name.
        checkpoint: Trained-policy checkpoint selection.

    Returns:
        Policy instance.
    """
    if policy_type == "trained":
        if not policy_path:
            raise ValueError("policy_path required for trained policies")
        return load_trained_policy(policy_path, name, checkpoint)
    elif policy_type == "heuristic":
        if not heuristic_config:
            raise ValueError("heuristic_config required for heuristic policies")
        return load_heuristic_policy(heuristic_config)
    else:
        raise ValueError(f"Unknown policy_type: {policy_type}")
