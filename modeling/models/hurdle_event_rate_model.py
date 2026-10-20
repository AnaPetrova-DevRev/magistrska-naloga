"""Hurdle Negative Binomial model for zero-inflated event rate forecasting.

A two-stage model that explicitly handles excess zeros in event count data:

  Stage 1 (binary): Logistic regression estimating P(events > 0 in bin),
      optionally using temporal covariates (hour_of_day, day_of_week, etc.).

  Stage 2 (count): Negative Binomial fitted on the positive-only subset,
      estimating E[events | events > 0].

Combined prediction: P(Y > 0 | x) * E[Y | Y > 0].

This is preferable to standard zero-inflated models (ZIP/ZINB) when the
zeros represent genuine quiet periods rather than hidden structural zeros.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
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


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    pos = z >= 0
    result = np.empty_like(z, dtype=float)
    result[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    result[~pos] = exp_z / (1.0 + exp_z)
    return result


def _fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    reg: float = 0.01,
    max_iter: int = 200,
) -> np.ndarray:
    """Fit logistic regression via L-BFGS-B with L2 regularisation.

    Args:
        X: Feature matrix (n, d) — should include an intercept column.
        y: Binary labels (n,).
        reg: L2 regularisation strength.
        max_iter: Maximum iterations for L-BFGS-B.

    Returns:
        Coefficient vector of shape (d,).
    """
    from scipy.optimize import minimize

    n, d = X.shape

    # Use balanced class weights to handle extreme imbalance (e.g. 98% zeros)
    n_pos = y.sum()
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        # Degenerate: all same class — return zeros (sigmoid → 0.5)
        return np.zeros(d)
    w_pos = n / (2.0 * n_pos)
    w_neg = n / (2.0 * n_neg)
    sample_weights = np.where(y == 1, w_pos, w_neg)

    def objective(theta):
        logits = X @ theta
        p = _sigmoid(logits)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        nll = -np.sum(sample_weights * (y * np.log(p) + (1 - y) * np.log(1 - p)))
        nll += 0.5 * reg * np.dot(theta, theta)
        return nll

    def gradient(theta):
        logits = X @ theta
        p = _sigmoid(logits)
        residual = sample_weights * (p - y)
        grad = X.T @ residual + reg * theta
        return grad

    theta0 = np.zeros(d)
    result = minimize(objective, theta0, jac=gradient, method="L-BFGS-B",
                      options={"maxiter": max_iter})
    coefs = result.x
    # Guard against NaN/Inf from degenerate optimisation (extreme imbalance)
    if not np.all(np.isfinite(coefs)):
        return np.zeros(d)
    return coefs


def _fit_negbin_mle(data: np.ndarray) -> tuple[float, float]:
    """Fit Negative Binomial parameters (r, p) via MLE on positive counts.

    Uses the method-of-moments as a starting point, then refines with
    scipy's MLE.

    Args:
        data: Positive integer counts (must be > 0).

    Returns:
        (r, p) where r is the number of successes and p is the success
        probability, parameterised so that E[Y] = r*(1-p)/p.
    """
    from scipy.stats import nbinom
    from scipy.optimize import minimize_scalar

    mean = np.mean(data)
    var = np.var(data, ddof=1) if len(data) > 1 else mean + 1.0

    # Method of moments initial estimate
    if var <= mean:
        # Underdispersed or Poisson-like: use large r
        r_init = max(mean ** 2 / max(var - mean, 0.1), 1.0)
    else:
        r_init = mean ** 2 / (var - mean)
    r_init = max(r_init, 0.1)

    # Optimise r via profile likelihood, with p = r / (r + mean) at each r
    int_data = np.round(data).astype(int)

    def neg_loglik(log_r):
        r = np.exp(log_r)
        p = r / (r + mean)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -np.sum(nbinom.logpmf(int_data - 1, r, p))

    result = minimize_scalar(neg_loglik, bounds=(np.log(0.01), np.log(1e6)),
                             method="bounded")
    r = np.exp(result.x)
    p = r / (r + mean)
    return float(r), float(p)


class HurdleEventRateModel(BaseModel):
    """Hurdle Negative Binomial model for event rate forecasting.

    Configuration options (model_config):
        binary_regularization: float — L2 penalty for logistic stage (default 0.01).

    Training configuration (training_config):
        (none required — MLE is used for both stages)
    """

    MODEL_TYPE: str = "HurdleNB"
    SUPPORTED_TASK_TYPES: set = {TaskType.EVENT_RATE}

    def __init__(
        self,
        task_type: TaskType,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
    ):
        super().__init__(task_type, model_config, training_config)
        self._logistic_coefs: Optional[np.ndarray] = None
        self._marginal_p_nonzero: Optional[float] = None
        self._nb_r: Optional[float] = None
        self._nb_p: Optional[float] = None
        self._nb_mean: Optional[float] = None
        self._covariate_names: Optional[List[str]] = None

    def _has_covariates(self, dataset: Dataset) -> bool:
        """Check if dataset has usable covariates."""
        return bool(dataset.covariates)

    def _build_feature_matrix(
        self, dataset: Dataset, indices: Sequence[int],
    ) -> np.ndarray:
        """Build feature matrix with intercept from dataset covariates.

        Returns shape (len(indices), n_features+1) with intercept as first column.
        """
        n = len(indices)
        if not self._has_covariates(dataset):
            return np.ones((n, 1))

        cols = [np.ones(n)]
        for name in sorted(dataset.covariates.keys()):
            values = np.array(dataset.covariates[name])
            cols.append(values[list(indices)])
        return np.column_stack(cols)

    def _build_forecast_features(
        self, timestamps: List[datetime],
    ) -> np.ndarray:
        """Build feature matrix for future timestamps (forecast mode)."""
        if not self._covariate_names:
            return np.ones((len(timestamps), 1))

        from modeling.covariate_builder import build_covariates

        covs = build_covariates(timestamps, self._covariate_names)
        cols = [np.ones(len(timestamps))]
        for name in sorted(covs.keys()):
            cols.append(np.array(covs[name]))
        return np.column_stack(cols)

    def fit(self, dataset: Dataset) -> Dict[str, Any]:
        """Fit the two-stage hurdle model.

        Stage 1: Logistic regression on binary(events > 0).
        Stage 2: Negative Binomial on positive-only counts.
        """
        self._validate_dataset_type(dataset)

        target = self._get_target_array(dataset)
        train_data = target[dataset.train_indices]

        self._covariate_names = sorted(dataset.covariates.keys()) if dataset.covariates else []

        # --- Stage 1: Binary model ---
        binary_labels = (train_data > 0).astype(float)
        n_nonzero = int(binary_labels.sum())
        n_total = len(train_data)
        self._marginal_p_nonzero = n_nonzero / n_total if n_total > 0 else 0.0

        X_train = self._build_feature_matrix(dataset, dataset.train_indices)
        reg = self.model_config.get("binary_regularization", 0.01)
        self._logistic_coefs = _fit_logistic(X_train, binary_labels, reg=reg)

        # Evaluate binary stage on training data
        p_train = _sigmoid(X_train @ self._logistic_coefs)

        # --- Stage 2: Count model on positive subset ---
        positive_mask = train_data > 0
        positive_data = train_data[positive_mask]

        if len(positive_data) == 0:
            # All zeros — degenerate case
            self._nb_r = 1.0
            self._nb_p = 0.5
            self._nb_mean = 0.0
            logger.warning("All training samples are zero — hurdle count stage is degenerate")
        else:
            self._nb_r, self._nb_p = _fit_negbin_mle(positive_data)
            self._nb_mean = float(np.mean(positive_data))

        self._is_fitted = True

        binary_ll = float(np.sum(
            binary_labels * np.log(np.clip(p_train, 1e-12, None))
            + (1 - binary_labels) * np.log(np.clip(1 - p_train, 1e-12, None))
        ))

        count_ll = 0.0
        if len(positive_data) > 0:
            from scipy.stats import nbinom
            int_positive = np.round(positive_data).astype(int)
            count_ll = float(np.sum(nbinom.logpmf(
                np.maximum(int_positive - 1, 0), self._nb_r, self._nb_p,
            )))

        fit_info = {
            "marginal_p_nonzero": self._marginal_p_nonzero,
            "frac_zeros": 1.0 - self._marginal_p_nonzero,
            "n_train": n_total,
            "n_positive": n_nonzero,
            "n_zero": n_total - n_nonzero,
            "nb_r": self._nb_r,
            "nb_p": self._nb_p,
            "nb_mean_positive": self._nb_mean,
            "expected_value": self._marginal_p_nonzero * self._nb_mean,
            "binary_log_likelihood": binary_ll,
            "count_log_likelihood": count_ll,
            "total_log_likelihood": binary_ll + count_ll,
            "n_logistic_features": X_train.shape[1],
            "covariate_names": self._covariate_names,
        }

        logger.info(
            "HurdleNB fitted. P(nonzero)=%.3f, NB(r=%.2f, p=%.4f), "
            "mean_positive=%.2f, n=%d (%d zeros)",
            self._marginal_p_nonzero, self._nb_r, self._nb_p,
            self._nb_mean, n_total, n_total - n_nonzero,
        )

        return self._run_validation(dataset, fit_info)

    def predict_pointwise(
        self,
        dataset: Dataset,
        indices: Sequence[int],
    ) -> np.ndarray:
        """Generate point predictions: P(Y>0 | x) * E[Y | Y>0]."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        self._validate_dataset_type(dataset)

        X = self._build_feature_matrix(dataset, indices)
        p_nonzero = _sigmoid(X @ self._logistic_coefs)
        predictions = p_nonzero * self._nb_mean

        return np.maximum(predictions, 0.0)

    def forecast(
        self,
        dataset: Dataset,
        horizon_minutes: int,
        sync_unit_name: str = "",
        reference_time: Optional[datetime] = None,
    ) -> PredictionTrajectory:
        """Generate a forecast trajectory with covariate-driven predictions."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before forecasting")
        self._validate_dataset_type(dataset)

        if reference_time is None:
            last_ts = dataset.timestamps[-1] if dataset.timestamps else datetime.now(UTC)
            reference_time = last_ts + timedelta(minutes=1)

        # Detect native resolution and compute native-step forecasts
        freq = self._infer_freq(dataset.timestamps)
        step_minutes = self._freq_to_minutes(freq)
        n_native_steps = max(1, int(np.ceil(horizon_minutes / step_minutes)))

        native_timestamps = [
            reference_time + timedelta(minutes=step_minutes * i)
            for i in range(n_native_steps)
        ]

        X = self._build_forecast_features(native_timestamps)
        p_nonzero = _sigmoid(X @ self._logistic_coefs)
        native_forecast = p_nonzero * self._nb_mean

        # Interpolate to minute-level
        is_cumulative = step_minutes > 1.0
        point_forecast = self._interpolate_to_minutes(
            native_forecast, step_minutes, horizon_minutes,
            is_cumulative_rate=is_cumulative,
        )
        point_forecast = np.maximum(point_forecast, 0.0)

        # Uncertainty from the mixture: Var = P*(Var_count + mean^2) - (P*mean)^2
        # Simplified: use std of positive component scaled by P
        from scipy.stats import nbinom
        nb_var = nbinom.var(self._nb_r, self._nb_p) if self._nb_r > 0 else 1.0
        mean_p_nonzero = float(np.mean(p_nonzero))
        mixture_std = float(np.sqrt(
            mean_p_nonzero * (nb_var + self._nb_mean ** 2)
            - (mean_p_nonzero * self._nb_mean) ** 2
        )) if mean_p_nonzero > 0 else 0.0

        uncertainties = np.full(horizon_minutes, mixture_std)
        lower = np.maximum(point_forecast - 1.645 * mixture_std, 0.0)
        upper = point_forecast + 1.645 * mixture_std

        minute_timestamps = [
            reference_time + timedelta(minutes=i)
            for i in range(horizon_minutes)
        ]

        points = self._create_prediction_points(
            minute_timestamps, point_forecast, lower, upper, uncertainties,
        )
        return PredictionTrajectory(
            sync_unit_name=sync_unit_name,
            task_type=self.task_type,
            horizon_minutes=horizon_minutes,
            points=points,
            model_id=f"hurdle_nb_{self.MODEL_TYPE}",
            created_at=datetime.now(UTC),
        )

    def save(self, path: str) -> None:
        """Save the fitted hurdle model to disk."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        os.makedirs(path, exist_ok=True)

        state = {
            "logistic_coefs": self._logistic_coefs,
            "marginal_p_nonzero": self._marginal_p_nonzero,
            "nb_r": self._nb_r,
            "nb_p": self._nb_p,
            "nb_mean": self._nb_mean,
            "covariate_names": self._covariate_names,
        }
        with open(os.path.join(path, "hurdle_nb_model.pkl"), "wb") as f:
            pickle.dump(state, f)

        config = {
            "model_type": self.MODEL_TYPE,
            "task_type": self.task_type.value,
            "model_config": self.model_config,
            "training_config": self.training_config,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Saved HurdleNB model to %s", path)

    @classmethod
    def load(
        cls,
        path: str,
        task_type: TaskType,
        model_config: Dict[str, Any],
    ) -> "HurdleEventRateModel":
        """Load a fitted hurdle model from disk."""
        config_path = os.path.join(path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                saved_config = json.load(f)
            training_config = saved_config.get("training_config", {})
        else:
            training_config = {}

        model = cls(
            task_type=task_type,
            model_config=model_config,
            training_config=training_config,
        )

        with open(os.path.join(path, "hurdle_nb_model.pkl"), "rb") as f:
            state = pickle.load(f)

        model._logistic_coefs = state["logistic_coefs"]
        model._marginal_p_nonzero = state["marginal_p_nonzero"]
        model._nb_r = state["nb_r"]
        model._nb_p = state["nb_p"]
        model._nb_mean = state["nb_mean"]
        model._covariate_names = state.get("covariate_names", [])
        model._is_fitted = True

        logger.info("Loaded HurdleNB model from %s", path)
        return model
