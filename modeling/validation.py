"""Data and fit validation utilities for forecasting models.

Provides reusable check functions that detect data quality issues and
distribution fit problems.  Each check returns ``None`` (pass) or a
``FitWarning`` describing the issue.  Models call these in their
``_pre_fit_validate`` and ``_post_fit_validate`` hooks so that every
``fit()`` result includes a structured ``warnings`` list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Data structures

@dataclass
class FitWarning:
    """A single warning produced during data or fit validation."""
    level: str          # "info", "warning", "error"
    code: str           # machine-readable, e.g. "low_sample_size"
    message: str        # human-readable description

    def to_dict(self) -> Dict[str, str]:
        return {"level": self.level, "code": self.code, "message": self.message}


@dataclass
class ValidationResult:
    """Aggregated result of a validation pass."""
    is_valid: bool = True
    warnings: List[FitWarning] = field(default_factory=list)

    def add(self, warning: Optional[FitWarning]) -> None:
        if warning is None:
            return
        self.warnings.append(warning)
        if warning.level == "error":
            self.is_valid = False

    def merge(self, other: ValidationResult) -> None:
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False

    def to_list(self) -> List[Dict[str, str]]:
        return [w.to_dict() for w in self.warnings]


# Universal pre-fit checks (all models)

def check_nan_values(data: np.ndarray) -> Optional[FitWarning]:
    """Check for NaN values in the training data."""
    n_nan = int(np.isnan(data).sum())
    if n_nan > 0:
        return FitWarning(
            level="error",
            code="nan_in_data",
            message=f"Training data contains {n_nan} NaN values "
                    f"({n_nan / len(data) * 100:.1f}% of data).",
        )
    return None


def check_infinite_values(data: np.ndarray) -> Optional[FitWarning]:
    """Check for infinite values in the training data."""
    n_inf = int(np.isinf(data).sum())
    if n_inf > 0:
        return FitWarning(
            level="error",
            code="inf_in_data",
            message=f"Training data contains {n_inf} infinite values.",
        )
    return None


def check_negative_values(data: np.ndarray) -> Optional[FitWarning]:
    """Check for negative values (invalid for TTE / event rate data)."""
    n_neg = int((data < 0).sum())
    if n_neg > 0:
        return FitWarning(
            level="error",
            code="negative_values",
            message=f"Training data contains {n_neg} negative values. "
                    f"Time-to-event and event-rate data must be non-negative.",
        )
    return None


def check_minimum_samples(data: np.ndarray, min_n: int = 10) -> Optional[FitWarning]:
    """Check that there are enough samples for a reliable fit."""
    n = len(data)
    if n < min_n:
        return FitWarning(
            level="error" if n < 3 else "warning",
            code="low_sample_size",
            message=f"Only {n} training samples (minimum recommended: {min_n}). "
                    f"Model fit may be unreliable.",
        )
    return None


def check_zero_variance(data: np.ndarray) -> Optional[FitWarning]:
    """Check if data has zero or near-zero variance (all-constant)."""
    std = float(np.std(data))
    if std < 1e-10:
        return FitWarning(
            level="error",
            code="zero_variance",
            message="Training data has zero variance (all values identical). "
                    "Cannot fit a distribution.",
        )
    mean = float(np.mean(data))
    if mean != 0 and std / abs(mean) < 1e-6:
        return FitWarning(
            level="warning",
            code="near_zero_variance",
            message=f"Training data has extremely low variance "
                    f"(CV={std / abs(mean):.2e}). Fitted parameters may be degenerate.",
        )
    return None


def check_excessive_zeros(data: np.ndarray, threshold: float = 0.5) -> Optional[FitWarning]:
    """Check if too many values are exactly zero."""
    n_zero = int((data == 0).sum())
    frac = n_zero / len(data) if len(data) > 0 else 0
    if frac > threshold:
        return FitWarning(
            level="warning",
            code="excessive_zeros",
            message=f"{n_zero}/{len(data)} values ({frac:.0%}) are exactly zero. "
                    f"Consider increasing min_delta or using a zero-inflated model "
                    f"(HurdleNB for event_rate, ZI_LogNormal/ZI_Gamma/ZI_Weibull for time_to_event).",
        )
    return None


# Post-fit checks: goodness of fit (parametric TTE models)

def check_ks_test(
    data: np.ndarray,
    dist_name: str,
    dist_params: Tuple,
    significance: float = 0.05,
) -> Optional[FitWarning]:
    """Run a Kolmogorov-Smirnov goodness-of-fit test.

    Args:
        data: The training data.
        dist_name: scipy distribution name (e.g. "expon", "weibull_min").
        dist_params: Parameters to pass to the distribution (shape, loc, scale).
        significance: p-value threshold for a "poor fit" warning.

    Returns:
        FitWarning if p < significance, None otherwise.
    """
    try:
        from scipy.stats import kstest
    except ImportError:
        return None

    stat, p_value = kstest(data, dist_name, args=dist_params)

    if p_value < significance:
        return FitWarning(
            level="warning",
            code="ks_test_poor_fit",
            message=f"Kolmogorov-Smirnov test rejects {dist_name} fit "
                    f"(statistic={stat:.4f}, p={p_value:.4f}). "
                    f"The fitted distribution may not match the data well.",
        )
    return None


def check_fitted_params_finite(params: Dict[str, float]) -> Optional[FitWarning]:
    """Check that all fitted parameters are finite."""
    non_finite = {k: v for k, v in params.items() if not np.isfinite(v)}
    if non_finite:
        names = ", ".join(f"{k}={v}" for k, v in non_finite.items())
        return FitWarning(
            level="warning",
            code="non_finite_params",
            message=f"Fitted parameters contain non-finite values: {names}. "
                    f"Predictions may use fallback values.",
        )
    return None


# Model-specific pre-fit checks

def check_stationarity(data: np.ndarray, significance: float = 0.05) -> Optional[FitWarning]:
    """Run an Augmented Dickey-Fuller test for stationarity (ARIMA)."""
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return None

    if len(data) < 20:
        return None

    result = adfuller(data, autolag="AIC")
    p_value = result[1]

    if p_value > significance:
        return FitWarning(
            level="warning",
            code="non_stationary",
            message=f"Augmented Dickey-Fuller test suggests the series is "
                    f"non-stationary (p={p_value:.4f}). "
                    f"Consider increasing the differencing order (d).",
        )
    return None


def check_data_length_for_windows(
    n_samples: int, window_size: int, min_windows: int = 5,
) -> Optional[FitWarning]:
    """Check that data is long enough for RNN sliding windows."""
    available_windows = n_samples - window_size
    if available_windows < min_windows:
        return FitWarning(
            level="error" if available_windows <= 0 else "warning",
            code="insufficient_data_for_windows",
            message=f"Only {max(0, available_windows)} sliding windows available "
                    f"(need at least {min_windows}). "
                    f"Training data has {n_samples} samples, "
                    f"input_window={window_size}.",
        )
    return None


def check_covariates_for_cox(has_covariates: bool) -> Optional[FitWarning]:
    """Warn if CoxPH is used without covariates."""
    if not has_covariates:
        return FitWarning(
            level="warning",
            code="cox_no_covariates",
            message="CoxPH model has no covariates. It will behave like a "
                    "simple baseline (median survival). Add covariates to the "
                    "dataset config for meaningful hazard ratio estimation.",
        )
    return None


def check_normality_of_log(data: np.ndarray, significance: float = 0.05) -> Optional[FitWarning]:
    """Shapiro-Wilk test on log(data) for LogNormal goodness-of-fit."""
    try:
        from scipy.stats import shapiro
    except ImportError:
        return None

    log_data = np.log(data[data > 0])
    if len(log_data) < 8 or len(log_data) > 5000:
        return None

    stat, p_value = shapiro(log_data)
    if p_value < significance:
        return FitWarning(
            level="warning",
            code="log_not_normal",
            message=f"Shapiro-Wilk test rejects normality of log(data) "
                    f"(p={p_value:.4f}). The LogNormal distribution may not "
                    f"be a good fit.",
        )
    return None
