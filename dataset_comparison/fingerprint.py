"""
Dataset Fingerprinting Module

Computes a fixed-length numerical "fingerprint" vector for each sync unit
by aggregating pre-computed artifacts (analysis_per_record, merged_events_per_minute,
merged_time_delta). No raw data re-processing is needed.

Feature categories (~50 features per sync unit):
- Volume (5): event counts, record counts, density
- Temporal span (4): date range, coverage, active days
- Time-delta distribution (10): statistical moments and percentiles
- Burstiness (4): coefficient of variation, burst ratio
- Periodicity (4): cron pattern counts and frequencies
- Temporal profiles (10): entropy, peak times, business hours fraction
- Record lifecycle (6): lifespan, modification patterns
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from storage_access import StorageAccess
from sync_unit import SyncUnit, get_sync_units
from global_vars import (
    PIPELINE_EXPLORER_APP_FOLDER_NAME,
    ANALYSIS_PER_RECORD_FILE_NAME_FORMAT,
    DATA_PREPROCESSING_APP_FOLDER_NAME,
    MERGED_DATASET_FILE_NAME_FORMAT,
    MERGED_EVENTS_PER_MINUTE_FILE_NAME_FORMAT,
    MERGED_TIME_DELTA_FILE_NAME_FORMAT,
)
from modeling.cron_pattern_detector import detect_cron_patterns, detect_hourly_patterns
from pipeline_explorer.outlier_detection import (
    detect_zscore_outliers,
    detect_modified_zscore_outliers,
    detect_iqr_outliers,
    detect_seasonal_outliers,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetFingerprint:
    """Summary feature vector for a single sync unit dataset."""

    sync_unit_name: str
    org: str
    system: str
    project: str
    features: dict[str, float] = field(default_factory=dict)
    dow_profile: list[float] = field(default_factory=list)  # 7-element normalized
    hod_profile: list[float] = field(default_factory=list)  # 24-element normalized
    computed_at: str = ""
    regime: str = ""  # frozen density-regime class label
    integrity_failures: list[str] = field(default_factory=list)

    def to_feature_vector(self, feature_names: Optional[list[str]] = None) -> np.ndarray:
        """Return the fingerprint as a flat numpy array for clustering.

        Args:
            feature_names: If provided, return only these features in this order.
                           If None, return all features in sorted key order.
        """
        if feature_names is not None:
            return np.array([self.features.get(name, 0.0) for name in feature_names])
        return np.array([self.features[k] for k in sorted(self.features.keys())])

    def get_feature_names(self) -> list[str]:
        """Return sorted list of feature names."""
        return sorted(self.features.keys())

    def to_dict(self) -> dict:
        return {
            "sync_unit_name": self.sync_unit_name,
            "org": self.org,
            "system": self.system,
            "project": self.project,
            "features": self.features,
            "dow_profile": self.dow_profile,
            "hod_profile": self.hod_profile,
            "computed_at": self.computed_at,
            "regime": self.regime,
            "integrity_failures": self.integrity_failures,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetFingerprint":
        return cls(
            sync_unit_name=data["sync_unit_name"],
            org=data["org"],
            system=data["system"],
            project=data["project"],
            features=data.get("features", {}),
            dow_profile=data.get("dow_profile", []),
            hod_profile=data.get("hod_profile", []),
            computed_at=data.get("computed_at", ""),
            regime=data.get("regime", ""),
            integrity_failures=data.get("integrity_failures", []),
        )


def _shannon_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution."""
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def _load_analysis_per_record(sync_unit: SyncUnit) -> Optional[pd.DataFrame]:
    """Load analysis_per_record.pkl for a sync unit."""
    sa = StorageAccess()
    path = (
        f"{sync_unit.get_path_for_sync_unit()}"
        f"/{PIPELINE_EXPLORER_APP_FOLDER_NAME}"
        f"/{ANALYSIS_PER_RECORD_FILE_NAME_FORMAT}"
    )
    if not sa.check_if_file_exists(path):
        logger.warning("analysis_per_record not found for %s", sync_unit.get_name())
        return None
    return sa.read_picklefile_as_dataframe(path)


def _load_events_per_minute(sync_unit: SyncUnit) -> Optional[pd.DataFrame]:
    """Load merged_events_per_minute_1.pkl for a sync unit."""
    sa = StorageAccess()
    file_name = MERGED_EVENTS_PER_MINUTE_FILE_NAME_FORMAT.replace("{int}", "1")
    path = (
        f"{sync_unit.get_path_for_sync_unit()}"
        f"/{DATA_PREPROCESSING_APP_FOLDER_NAME}"
        f"/{file_name}"
    )
    if not sa.check_if_file_exists(path):
        logger.warning("merged_events_per_minute not found for %s", sync_unit.get_name())
        return None
    return sa.read_picklefile_as_dataframe(path)


def _load_time_deltas(sync_unit: SyncUnit) -> Optional[pd.DataFrame]:
    """Load merged_time_delta_1.pkl for a sync unit."""
    sa = StorageAccess()
    file_name = MERGED_TIME_DELTA_FILE_NAME_FORMAT.replace("{int}", "1")
    path = (
        f"{sync_unit.get_path_for_sync_unit()}"
        f"/{DATA_PREPROCESSING_APP_FOLDER_NAME}"
        f"/{file_name}"
    )
    if not sa.check_if_file_exists(path):
        logger.warning("merged_time_delta not found for %s", sync_unit.get_name())
        return None
    return sa.read_picklefile_as_dataframe(path)


def _load_merged_dataset(sync_unit: SyncUnit) -> Optional[pd.DataFrame]:
    """Load merged_dataset_1.pkl (one row per RAW event) for a sync unit."""
    sa = StorageAccess()
    file_name = MERGED_DATASET_FILE_NAME_FORMAT.replace("{int}", "1")
    path = (
        f"{sync_unit.get_path_for_sync_unit()}"
        f"/{DATA_PREPROCESSING_APP_FOLDER_NAME}"
        f"/{file_name}"
    )
    if not sa.check_if_file_exists(path):
        logger.warning("merged_dataset not found for %s", sync_unit.get_name())
        return None
    return sa.read_picklefile_as_dataframe(path)


def check_artifact_integrity(
    epm_df: pd.DataFrame,
    td_df: pd.DataFrame,
    md_df: Optional[pd.DataFrame],
) -> list[str]:
    """Run the integrity gate on a sync unit's artifacts.

    Returns a list of human-readable failure messages (empty = all pass).
    Catches stale/un-floored artifacts like the pre-flooring fingerprint bug.
    """
    failures: list[str] = []

    # 1. events_delta must be integer-minute and >= 1 (floored + deduped)
    if "events_delta" in td_df.columns:
        deltas = td_df["events_delta"].dropna()
        if len(deltas) > 0:
            as_float = deltas.astype(float)
            if not bool((as_float == as_float.round()).all()):
                failures.append(
                    "events_delta has non-integer values (artifact not "
                    "minute-floored — stale?)"
                )
            if float(as_float.min()) < 1:
                failures.append("events_delta has values < 1 minute")
    else:
        failures.append("merged_time_delta missing events_delta column")

    # 2. time_delta and events_per_minute both index active minutes
    if len(td_df) != len(epm_df):
        failures.append(
            f"len(merged_time_delta)={len(td_df)} != "
            f"len(merged_events_per_minute)={len(epm_df)}"
        )

    # 3. sum(events_per_minute) == raw event count (merged_dataset)
    if md_df is not None and "events_per_minute" in epm_df.columns:
        epm_sum = int(epm_df["events_per_minute"].sum())
        raw_count = len(md_df)
        if epm_sum != raw_count:
            failures.append(
                f"sum(events_per_minute)={epm_sum} != "
                f"len(merged_dataset)={raw_count}"
            )

    return failures


def _compute_volume_features(
    apr_df: pd.DataFrame,
    epm_df: pd.DataFrame,
    td_df: pd.DataFrame,
    md_df: Optional[pd.DataFrame] = None,
) -> dict[str, float]:
    """Compute volume metrics on both RAW and MINUTE bases.

    - RAW events = len(merged_dataset) (one row per raw event).
    - MINUTE events = len(merged_time_delta) (one row per active minute).
    Post-flooring, the raw count lives ONLY in merged_dataset, so
    event_density_per_day (RAW) requires md_df. Falls back to the minute
    count when merged_dataset is unavailable.
    """
    minute_events = len(td_df)
    raw_events = len(md_df) if md_df is not None else minute_events
    unique_records = len(apr_df)

    # Compute span from events per minute timestamps
    if "minute" in epm_df.columns:
        ts_col = "minute"
    elif "events" in epm_df.columns:
        ts_col = "events"
    else:
        ts_col = epm_df.columns[0]

    timestamps = pd.to_datetime(epm_df[ts_col])
    span_seconds = (timestamps.max() - timestamps.min()).total_seconds()
    span_days = max(span_seconds / 86400, 1)

    return {
        # RAW basis
        "raw_events": float(raw_events),
        "raw_density_per_day": _safe_div(raw_events, span_days),
        "events_per_record": _safe_div(raw_events, unique_records),
        # MINUTE basis (canonical density axis)
        "minute_events": float(minute_events),
        "minute_density_per_day": _safe_div(minute_events, span_days),
        # dedup / magnitude concentration
        "dedup_ratio": _safe_div(raw_events, minute_events, default=1.0),
        # shared
        "unique_records": float(unique_records),
        "records_per_day": _safe_div(unique_records, span_days),
        # back-compat: total_events kept as an alias of RAW count
        "total_events": float(raw_events),
        "event_density_per_day": _safe_div(raw_events, span_days),
    }


def _compute_temporal_features(epm_df: pd.DataFrame) -> dict[str, float]:
    """Compute temporal span and coverage metrics."""
    if "minute" in epm_df.columns:
        ts_col = "minute"
    elif "events" in epm_df.columns:
        ts_col = "events"
    else:
        ts_col = epm_df.columns[0]

    timestamps = pd.to_datetime(epm_df[ts_col])
    min_ts = timestamps.min()
    max_ts = timestamps.max()
    span_seconds = (max_ts - min_ts).total_seconds()
    span_days = max(span_seconds / 86400, 1)
    total_minutes_in_span = max(span_seconds / 60, 1)

    # Coverage: minutes with events / total minutes
    minutes_with_events = len(epm_df)
    coverage_ratio = _safe_div(minutes_with_events, total_minutes_in_span)

    # Active days
    active_days = timestamps.dt.date.nunique()
    total_days_in_span = max(int(span_days), 1)
    active_days_ratio = _safe_div(active_days, total_days_in_span)

    # Created span (from analysis_per_record if available, else same as span)
    return {
        "span_days": span_days,
        "coverage_ratio": min(coverage_ratio, 1.0),
        "active_days_ratio": min(active_days_ratio, 1.0),
        "total_minutes_with_events": float(minutes_with_events),
    }


def _compute_delta_distribution_features(td_df: pd.DataFrame) -> dict[str, float]:
    """Compute time-delta distribution statistics."""
    if "events_delta" in td_df.columns:
        deltas = td_df["events_delta"].dropna().values.astype(float)
    else:
        logger.warning("No events_delta column found in time delta data")
        return {k: 0.0 for k in [
            "delta_mean", "delta_median", "delta_std", "delta_skew",
            "delta_kurtosis", "delta_q10", "delta_q25", "delta_q75",
            "delta_q90", "delta_iqr",
        ]}

    if len(deltas) == 0:
        return {k: 0.0 for k in [
            "delta_mean", "delta_median", "delta_std", "delta_skew",
            "delta_kurtosis", "delta_q10", "delta_q25", "delta_q75",
            "delta_q90", "delta_iqr",
        ]}

    q10, q25, q75, q90 = np.percentile(deltas, [10, 25, 75, 90])

    return {
        "delta_mean": float(np.mean(deltas)),
        "delta_median": float(np.median(deltas)),
        "delta_std": float(np.std(deltas)),
        "delta_skew": float(scipy_stats.skew(deltas)),
        "delta_kurtosis": float(scipy_stats.kurtosis(deltas)),
        "delta_q10": float(q10),
        "delta_q25": float(q25),
        "delta_q75": float(q75),
        "delta_q90": float(q90),
        "delta_iqr": float(q75 - q25),
    }


def _compute_burstiness_features(epm_df: pd.DataFrame) -> dict[str, float]:
    """Compute burstiness metrics from events per minute data."""
    if "events_per_minute" in epm_df.columns:
        val_col = "events_per_minute"
    else:
        val_col = epm_df.columns[-1]

    values = epm_df[val_col].dropna().values.astype(float)

    if len(values) == 0:
        return {
            "events_per_min_mean": 0.0,
            "events_per_min_max": 0.0,
            "events_per_min_cov": 0.0,
            "burst_ratio": 0.0,
        }

    mean_val = float(np.mean(values))
    max_val = float(np.max(values))
    std_val = float(np.std(values))
    multi_event_minute_fraction = float(np.mean(values > 1))

    return {
        "events_per_min_mean": mean_val,
        "events_per_min_max": max_val,
        "events_per_min_cov": _safe_div(std_val, mean_val),
        "burst_ratio": _safe_div(max_val, mean_val),
        "multi_event_minute_fraction": multi_event_minute_fraction,
    }


# Frozen regime taxonomy on the canonical MINUTE density axis.
# See docs/dataset_characterization_protocol.md §2.
_REGIME_THRESHOLDS = [
    (200.0, "very-dense"),
    (100.0, "dense"),
    (20.0, "moderate"),
    (5.0, "sparse"),
]


def classify_regime(minute_density_per_day: float) -> str:
    """Map minute-floored density/day to a frozen regime class label."""
    for threshold, label in _REGIME_THRESHOLDS:
        if minute_density_per_day >= threshold:
            return label
    return "very-sparse"


# Any fitted shape parameter above this is treated as optimizer divergence, not a
# fit (real IAT shapes are O(0.1-10); divergences reach ~1e40+). See the guard in
# _compute_iat_fit_features.
SHAPE_DIVERGENCE_MAX = 100.0


def _compute_iat_fit_features(td_df: pd.DataFrame) -> dict[str, float]:
    """Fit parametric families to integer-minute inter-arrival deltas.

    Records the best-AIC family and a KS goodness-of-fit p-value, giving a
    defensible distribution-shape label instead of eyeballing skew/kurtosis.
    Fits scipy distributions directly (self-contained; mirrors the families
    in modeling/models/): Exponential, Weibull, Gamma, LogNormal, LogLogistic.
    """
    fam_keys = [
        "iat_best_family", "iat_best_aic", "iat_ks_stat",
        "iat_ks_pvalue", "iat_ks_passes",
    ]
    if "events_delta" not in td_df.columns:
        return {k: (0.0 if k != "iat_best_family" else -1.0) for k in fam_keys}

    deltas = td_df["events_delta"].dropna().values.astype(float)
    # Positive support required for these families
    deltas = deltas[deltas > 0]
    if len(deltas) < 20:
        return {k: (0.0 if k != "iat_best_family" else -1.0) for k in fam_keys}

    # Subsample for speed on huge units (deterministic seed).
    if len(deltas) > 50_000:
        rng = np.random.default_rng(42)
        deltas = rng.choice(deltas, size=50_000, replace=False)

    # (name, scipy dist, code) — code stored numerically for the feature vector.
    families = [
        ("exponential", scipy_stats.expon, 0),
        ("weibull", scipy_stats.weibull_min, 1),
        ("gamma", scipy_stats.gamma, 2),
        ("lognormal", scipy_stats.lognorm, 3),
        ("loglogistic", scipy_stats.fisk, 4),
    ]

    best = None  # (aic, name, code, ks_stat, ks_p)
    for name, dist, code in families:
        try:
            if name == "exponential":
                params = dist.fit(deltas, floc=0)
            else:
                params = dist.fit(deltas, floc=0)
            # Guard against unbounded-optimizer divergence. On the integer-minute
            # deltas of dense units (a large mass of exactly-one-minute gaps),
            # the log-logistic/Weibull shape parameter can run away to ~1e40+,
            # yielding a spuriously huge log-likelihood that wins AIC while the
            # fit is degenerate (kstest returns ks_stat ~= 1.0, a perfect misfit).
            # A shape that large is a divergence, not a fit — reject it so a real
            # (bounded) family is selected instead. Real shapes here are O(0.1-10).
            if name != "exponential" and abs(params[0]) > SHAPE_DIVERGENCE_MAX:
                logger.debug("IAT fit for %s rejected: shape %.3g diverged",
                             name, params[0])
                continue
            ll = float(np.sum(dist.logpdf(deltas, *params)))
            if not np.isfinite(ll):
                continue
            k = len(params)
            aic = 2 * k - 2 * ll
            ks_stat, ks_p = scipy_stats.kstest(deltas, dist.name, args=params)
            if best is None or aic < best[0]:
                best = (aic, name, code, float(ks_stat), float(ks_p))
        except Exception as exc:  # noqa: BLE001 - fit can fail on degenerate data
            logger.debug("IAT fit failed for %s: %s", name, exc)
            continue

    if best is None:
        return {k: (0.0 if k != "iat_best_family" else -1.0) for k in fam_keys}

    _aic, _name, code, ks_stat, ks_p = best
    return {
        "iat_best_family": float(code),
        "iat_best_aic": float(_aic),
        "iat_ks_stat": ks_stat,
        "iat_ks_pvalue": ks_p,
        "iat_ks_passes": float(ks_p >= 0.05),
    }


# Numeric family code -> name, for readable reporting.
IAT_FAMILY_NAMES = {
    -1.0: "none", 0.0: "exponential", 1.0: "weibull",
    2.0: "gamma", 3.0: "lognormal", 4.0: "loglogistic",
}


def _compute_zero_inflation_features(
    epm_df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> dict[str, float]:
    """Zero fraction on the reconstituted regular 1-minute grid.

    The sparse events_per_minute artifact stores only active minutes; the RL
    agent sees a regular grid with zeros filled in. We reconstitute that grid
    over the full span and report the zero fraction overall and per
    chronological split block (matching the train/val/test discipline).
    """
    keys = ["zero_fraction_all", "zero_fraction_train", "zero_fraction_test"]
    if "minute" not in epm_df.columns or "events_per_minute" not in epm_df.columns:
        return {k: 0.0 for k in keys}

    minutes = pd.to_datetime(epm_df["minute"])
    if len(minutes) < 2:
        return {k: 0.0 for k in keys}

    # Total minute-slots in span (denominator of the regular grid).
    span_minutes = int(
        (minutes.max() - minutes.min()).total_seconds() // 60
    ) + 1
    active_minutes = len(epm_df)
    zero_fraction_all = 1.0 - _safe_div(active_minutes, span_minutes)

    # Per-split: chronological blocks over the FULL grid span (not just active).
    # Approximate block zero fractions by allocating active minutes to blocks
    # by their timestamp position within the span.
    start = minutes.min()
    train_end_min = int(span_minutes * train_ratio)
    val_end_min = int(span_minutes * (train_ratio + val_ratio))
    offsets = ((minutes - start).dt.total_seconds() // 60).astype(int)

    active_train = int((offsets < train_end_min).sum())
    active_test = int((offsets >= val_end_min).sum())
    span_train = max(train_end_min, 1)
    span_test = max(span_minutes - val_end_min, 1)

    return {
        "zero_fraction_all": float(zero_fraction_all),
        "zero_fraction_train": float(1.0 - _safe_div(active_train, span_train)),
        "zero_fraction_test": float(1.0 - _safe_div(active_test, span_test)),
    }


def _compute_outlier_features(
    epm_df: pd.DataFrame, td_df: pd.DataFrame
) -> dict[str, float]:
    """Multi-method outlier agreement on both magnitude and IAT series.

    The ground truth is not a single method's flags (method-dependent) but the
    agreement structure across the 4 methods in outlier_detection.py:
    per-method %, a >=3-method consensus %, and the divergence between methods.
    """
    out: dict[str, float] = {}

    series_specs = [
        ("epm", epm_df, "events_per_minute", "minute"),
        ("iat", td_df, "events_delta", "events"),
    ]
    for prefix, df, val_col, date_col in series_specs:
        if val_col not in df.columns:
            for m in ["zscore", "mad", "iqr", "seasonal"]:
                out[f"outlier_pct_{prefix}_{m}"] = 0.0
            out[f"outlier_pct_{prefix}_consensus"] = 0.0
            out[f"outlier_{prefix}_method_divergence"] = 0.0
            continue

        values = df[val_col].dropna().astype(float)
        masks = {
            "zscore": detect_zscore_outliers(values, threshold=3.0),
            "mad": detect_modified_zscore_outliers(values, threshold=3.5),
            "iqr": detect_iqr_outliers(values, multiplier=1.5),
        }
        if date_col in df.columns:
            masks["seasonal"] = detect_seasonal_outliers(
                df.loc[values.index], date_col, val_col, threshold=3.0
            )
        else:
            masks["seasonal"] = pd.Series(False, index=values.index)

        pcts = {}
        for m, mask in masks.items():
            pct = float(mask.mean()) if len(mask) > 0 else 0.0
            out[f"outlier_pct_{prefix}_{m}"] = pct
            pcts[m] = pct

        # Consensus: flagged by >= 3 of the 4 methods.
        aligned = pd.DataFrame({m: mask.astype(int) for m, mask in masks.items()})
        consensus = (aligned.sum(axis=1) >= 3)
        out[f"outlier_pct_{prefix}_consensus"] = (
            float(consensus.mean()) if len(consensus) > 0 else 0.0
        )
        out[f"outlier_{prefix}_method_divergence"] = (
            max(pcts.values()) - min(pcts.values()) if pcts else 0.0
        )

    return out


def _compute_periodicity_features(epm_df: pd.DataFrame) -> dict[str, float]:
    """Compute periodicity metrics using cron pattern detection."""
    if "minute" in epm_df.columns:
        ts_col = "minute"
    elif "events" in epm_df.columns:
        ts_col = "events"
    else:
        ts_col = epm_df.columns[0]

    if "events_per_minute" in epm_df.columns:
        val_col = "events_per_minute"
    else:
        val_col = epm_df.columns[-1]

    timestamps = pd.to_datetime(epm_df[ts_col]).tolist()
    values = epm_df[val_col].values.tolist()

    daily_patterns = detect_cron_patterns(timestamps, values)
    hourly_patterns = detect_hourly_patterns(timestamps, values)

    dominant_freq = 0.0
    if daily_patterns:
        dominant_freq = max(p.frequency for p in daily_patterns)
    elif hourly_patterns:
        dominant_freq = max(p.frequency for p in hourly_patterns)

    # Fraction of events attributable to ANY detected cron slot.
    # Daily patterns match on (hour, minute); hourly patterns match on
    # minute-of-hour (any hour). Counting only daily slots (the previous
    # behaviour) reported 0.0 for units whose only periodicity is hourly.
    cron_event_fraction = 0.0
    total_events = sum(values)
    if total_events > 0 and (daily_patterns or hourly_patterns):
        daily_slots = {(p.hour, p.minute) for p in daily_patterns}
        hourly_minutes = {p.minute for p in hourly_patterns}
        cron_events = sum(
            v for ts, v in zip(timestamps, values)
            if (ts.hour, ts.minute) in daily_slots or ts.minute in hourly_minutes
        )
        cron_event_fraction = cron_events / total_events

    return {
        "num_daily_cron_patterns": float(len(daily_patterns)),
        "num_hourly_cron_patterns": float(len(hourly_patterns)),
        "dominant_cron_frequency": dominant_freq,
        "cron_event_fraction": cron_event_fraction,
    }


def _compute_temporal_profile_features(
    apr_df: pd.DataFrame,
) -> tuple[dict[str, float], list[float], list[float]]:
    """Compute day-of-week and hour-of-day profile features.

    Returns:
        Tuple of (features_dict, dow_profile_7, hod_profile_24)
    """
    # Day-of-week profile from per-record percentages
    dow_cols = [f"pct_changes_{day}" for day in
                ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']]
    existing_dow_cols = [c for c in dow_cols if c in apr_df.columns]

    if existing_dow_cols:
        dow_means = apr_df[existing_dow_cols].mean().values
        dow_total = dow_means.sum()
        dow_profile = (dow_means / dow_total).tolist() if dow_total > 0 else [1/7]*7
    else:
        dow_profile = [1/7] * 7

    dow_arr = np.array(dow_profile)
    dow_entropy = _shannon_entropy(dow_arr)
    peak_dow = int(np.argmax(dow_arr))

    # Weekend fraction
    weekend_fraction = float(dow_arr[5] + dow_arr[6]) if len(dow_arr) == 7 else 0.0

    # Hour-of-day profile from per-record counts
    hod_cols = [f"num_changes_hour_{h:02d}" for h in range(24)]
    existing_hod_cols = [c for c in hod_cols if c in apr_df.columns]

    if existing_hod_cols:
        hod_means = apr_df[existing_hod_cols].mean().values
        hod_total = hod_means.sum()
        hod_profile = (hod_means / hod_total).tolist() if hod_total > 0 else [1/24]*24
    else:
        hod_profile = [1/24] * 24

    hod_arr = np.array(hod_profile)
    hod_entropy = _shannon_entropy(hod_arr)
    peak_hour = int(np.argmax(hod_arr))

    # Night fraction (hours 0-5)
    night_fraction = float(hod_arr[:6].sum())
    # Business hours fraction (hours 9-17)
    business_hours_fraction = float(hod_arr[9:18].sum())

    features = {
        "dow_entropy": dow_entropy,
        "weekend_fraction": weekend_fraction,
        "peak_dow": float(peak_dow),
        "hod_entropy": hod_entropy,
        "peak_hour": float(peak_hour),
        "night_fraction": night_fraction,
        "business_hours_fraction": business_hours_fraction,
    }

    return features, dow_profile, hod_profile


def _compute_lifecycle_features(apr_df: pd.DataFrame) -> dict[str, float]:
    """Compute record lifecycle aggregate features."""
    features = {}

    if "time_first_created_to_last_modified" in apr_df.columns:
        lifespan = apr_df["time_first_created_to_last_modified"].dropna()
        mean_creation_to_last = float(lifespan.mean()) if len(lifespan) > 0 else 0.0
        # Backward-compatible alias; prose and new clustering use the claim-safe key.
        features["avg_record_lifespan"] = mean_creation_to_last
        features["avg_creation_to_last_state"] = mean_creation_to_last
    else:
        features["avg_record_lifespan"] = 0.0
        features["avg_creation_to_last_state"] = 0.0

    if "num_changes_total" in apr_df.columns:
        changes = apr_df["num_changes_total"]
        features["median_changes_per_record"] = float(changes.median())
        features["pct_single_change_records"] = float((changes == 1).mean())
        features["pct_records_gt_10_changes"] = float((changes > 10).mean())
    else:
        features["median_changes_per_record"] = 0.0
        features["pct_single_change_records"] = 0.0
        features["pct_records_gt_10_changes"] = 0.0

    if "was_created_3mo_before_cutoff" in apr_df.columns:
        features["pct_created_before_cutoff"] = float(
            apr_df["was_created_3mo_before_cutoff"].mean()
        )
    else:
        features["pct_created_before_cutoff"] = 0.0

    if "time_to_first_modification" in apr_df.columns:
        ttfm = apr_df["time_to_first_modification"].dropna()
        features["avg_time_to_first_mod"] = float(ttfm.mean()) if len(ttfm) > 0 else 0.0
    else:
        features["avg_time_to_first_mod"] = 0.0

    return features


# pylint: disable=too-many-arguments,too-many-positional-arguments
def compute_fingerprint_from_artifacts(
    sync_unit: SyncUnit,
    apr_df: pd.DataFrame,
    epm_df: pd.DataFrame,
    td_df: pd.DataFrame,
    md_df: Optional[pd.DataFrame] = None,
    profile_apr_df: Optional[pd.DataFrame] = None,
) -> DatasetFingerprint:
    """Compute a fingerprint from supplied artifacts.

    ``profile_apr_df`` allows windowed callers to provide a compact profile frame
    independently from the per-record lifecycle frame. The canonical full-span
    path leaves it unset and preserves the previous behavior.
    """
    # Integrity gate — catches stale/un-floored artifacts. Recorded, not fatal,
    # so the numbers still surface with an explicit warning flag.
    integrity_failures = check_artifact_integrity(epm_df, td_df, md_df)
    if integrity_failures:
        logger.warning(
            "Integrity gate FAILED for %s: %s",
            sync_unit.get_name(),
            "; ".join(integrity_failures),
        )

    features = {}

    # Compute all feature categories
    features.update(_compute_volume_features(apr_df, epm_df, td_df, md_df))
    features.update(_compute_temporal_features(epm_df))
    features.update(_compute_delta_distribution_features(td_df))
    features.update(_compute_burstiness_features(epm_df))
    features.update(_compute_periodicity_features(epm_df))
    features.update(_compute_iat_fit_features(td_df))
    features.update(_compute_zero_inflation_features(epm_df))
    features.update(_compute_outlier_features(epm_df, td_df))

    profile_source = profile_apr_df if profile_apr_df is not None else apr_df
    profile_features, dow_profile, hod_profile = _compute_temporal_profile_features(
        profile_source
    )
    features.update(profile_features)
    features.update(_compute_lifecycle_features(apr_df))

    regime = classify_regime(features.get("minute_density_per_day", 0.0))

    fingerprint = DatasetFingerprint(
        sync_unit_name=sync_unit.get_name(),
        org=sync_unit.organization_id,
        system=sync_unit.external_system_id,
        project=sync_unit.project_id,
        features=features,
        dow_profile=dow_profile,
        hod_profile=hod_profile,
        computed_at=datetime.now().isoformat(),
        regime=regime,
        integrity_failures=integrity_failures,
    )

    logger.info(
        "Fingerprint computed for %s: %d features, regime=%s",
        sync_unit.get_name(),
        len(features),
        regime,
    )
    return fingerprint


def compute_fingerprint(sync_unit: SyncUnit) -> Optional[DatasetFingerprint]:
    """Compute the canonical full-span fingerprint for one sync unit."""
    logger.info("Computing fingerprint for %s", sync_unit.get_name())

    apr_df = _load_analysis_per_record(sync_unit)
    epm_df = _load_events_per_minute(sync_unit)
    td_df = _load_time_deltas(sync_unit)
    md_df = _load_merged_dataset(sync_unit)  # RAW events; None => raw==minute fallback

    if apr_df is None or epm_df is None or td_df is None:
        logger.warning(
            "Missing required artifacts for %s (apr=%s, epm=%s, td=%s)",
            sync_unit.get_name(),
            apr_df is not None,
            epm_df is not None,
            td_df is not None,
        )
        return None

    return compute_fingerprint_from_artifacts(
        sync_unit, apr_df, epm_df, td_df, md_df
    )


def compute_all_fingerprints(
    sync_units: Optional[list[SyncUnit]] = None,
) -> list[DatasetFingerprint]:
    """Compute fingerprints for all (or specified) sync units.

    Args:
        sync_units: List of sync units to process. If None, uses all registered.

    Returns:
        List of successfully computed fingerprints.
    """
    if sync_units is None:
        sync_units = get_sync_units()

    fingerprints = []
    for su in sync_units:
        fp = compute_fingerprint(su)
        if fp is not None:
            fingerprints.append(fp)
        else:
            logger.warning("Skipping %s - missing artifacts", su.get_name())

    logger.info(
        "Computed %d fingerprints out of %d sync units",
        len(fingerprints),
        len(sync_units),
    )
    return fingerprints
