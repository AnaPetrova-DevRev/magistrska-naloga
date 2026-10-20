"""Empirical analysis of prediction volatility for TTE and ER models.

Generates minute-level prediction traces (same way the RL agent gets them)
and analyzes:
1. TTE: minute-to-minute jumps, inconsistency patterns
2. ER: background rate, signal-to-noise ratio
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from pathlib import Path

from sync_unit import SyncUnit
from storage_access import StorageAccess
from modeling.dataset_builder import build_event_rate_dataset, build_time_to_event_dataset
from modeling.core_types import TaskType, DatasetConfig
from model_serving.model_serving_app import ModelServingApp
from app_actions_constants import GET_PREDICTION


DATA_ROOT = Path(__file__).parent.parent / "data"
SYNC_UNIT = SyncUnit("novacore", "jira", "bugs")


def get_predictions_via_serving(
    model_id: str, task_type: str, horizon_minutes: int
) -> np.ndarray:
    """Get predictions the same way the RL agent does."""
    sa = StorageAccess()
    serving = ModelServingApp(sa)
    response = serving.execute_action(
        SYNC_UNIT,
        GET_PREDICTION,
        {
            "task_type": task_type,
            "horizon_minutes": horizon_minutes,
            "model_selection": {"mode": "explicit", "model_id": model_id},
        },
    )
    if not response.success or not response.data:
        raise RuntimeError(f"Failed: {response.message}")
    result = response.data["prediction_result"]
    if not result.ok:
        raise RuntimeError(f"Prediction failed: {result.error}")
    return np.array(
        [p.expected_events for p in result.prediction.points],
        dtype=np.float32,
    )


def get_actual_values(task_type: str, n_points: int = 2000) -> tuple:
    """Load actual values from the test split."""
    if task_type == "event_rate":
        config = DatasetConfig(
            task_type=TaskType.EVENT_RATE, start_time=None, end_time=None,
            train_ratio=0.7, val_ratio=0.15,
        )
        ds, _ = build_event_rate_dataset(SYNC_UNIT, config)
        test_indices = ds.test_indices
        values = np.array([ds.events_per_minute[i] for i in test_indices])
    else:
        config = DatasetConfig(
            task_type=TaskType.TIME_TO_EVENT, start_time=None, end_time=None,
            train_ratio=0.7, val_ratio=0.15,
        )
        ds, _ = build_time_to_event_dataset(SYNC_UNIT, config)
        test_indices = ds.test_indices
        values = np.array([ds.minutes_to_next_event[i] for i in test_indices])

    return values[:n_points]


def analyze_tte_predictions(predictions: np.ndarray, actuals: np.ndarray, model_name: str):
    """Analyze TTE prediction volatility."""
    print("\n" + "="*70)
    print(f"TTE PREDICTION ANALYSIS: {model_name}")
    print("="*70)

    n = min(len(predictions), len(actuals))
    predictions = predictions[:n]
    actuals = actuals[:n]

    print(f"\n  Predictions: n={n}, range=[{predictions.min():.2f}, {predictions.max():.2f}] min")
    print(f"  Mean: {predictions.mean():.2f}, Std: {predictions.std():.2f}, Median: {np.median(predictions):.2f}")
    print(f"\n  Actuals: range=[{actuals.min():.2f}, {actuals.max():.1f}] min")
    print(f"  Mean: {actuals.mean():.2f}, Median: {np.median(actuals):.2f}")

    # Minute-to-minute changes
    diffs = np.diff(predictions)
    abs_diffs = np.abs(diffs)

    print(f"\n  Minute-to-minute prediction changes:")
    print(f"    Mean |Δ|: {abs_diffs.mean():.3f} min")
    print(f"    Median |Δ|: {np.median(abs_diffs):.3f} min")
    print(f"    Max |Δ|: {abs_diffs.max():.2f} min")
    print(f"    Std(Δ): {diffs.std():.3f} min")

    # What fraction of consecutive changes deviate from the expected -1 min/min?
    deviation = diffs - (-1.0)
    print(f"\n  Deviation from ideal countdown (-1 min/min):")
    print(f"    Mean deviation: {deviation.mean():.3f}")
    print(f"    Std: {deviation.std():.3f}")
    print(f"    Within ±0.5: {(np.abs(deviation) < 0.5).mean()*100:.1f}%")
    print(f"    Within ±2:   {(np.abs(deviation) < 2).mean()*100:.1f}%")
    print(f"    Within ±5:   {(np.abs(deviation) < 5).mean()*100:.1f}%")

    # Threshold policy flip analysis
    print(f"\n  Threshold policy flip analysis:")
    for threshold in [5, 10, 15, 30, 60, 120]:
        below = predictions < threshold
        flips = np.sum(np.abs(np.diff(below.astype(int))))
        pct_below = below.mean() * 100
        if pct_below > 0 and pct_below < 100:
            print(f"    T={threshold:4d}min: {pct_below:5.1f}% below, "
                  f"{flips:4d} flips ({flips/n*100:.2f}% rate)")
        else:
            print(f"    T={threshold:4d}min: {pct_below:5.1f}% below (no flips)")

    # The KEY question: when prediction says "5 min to event",
    # does an event actually happen within 5 minutes?
    print(f"\n  Prediction accuracy at different lead times:")
    for lead_time in [5, 10, 15, 30, 60]:
        # Steps where prediction < lead_time (model says "event soon")
        alert_mask = predictions < lead_time
        n_alerts = alert_mask.sum()
        if n_alerts == 0:
            print(f"    Lead={lead_time:3d}min: No alerts")
            continue

        # For each alert, check if actual event happens within lead_time steps
        alert_indices = np.where(alert_mask)[0]
        hits = 0
        for idx in alert_indices:
            # Check if any actual TTE value jumps (event) within lead_time steps
            window_end = min(idx + lead_time, n)
            actual_window = actuals[idx:window_end]
            if len(actual_window) > 1:
                actual_diffs = np.diff(actual_window)
                if np.any(actual_diffs > 1.0):  # event boundary
                    hits += 1
        precision = hits / n_alerts * 100
        print(f"    Lead={lead_time:3d}min: {n_alerts:5d} alerts, "
              f"{hits:4d} true positives ({precision:.1f}% precision)")

    # Large jumps (potential model inconsistency)
    large_jumps = np.where(abs_diffs > 10)[0]
    print(f"\n  Large jumps (>10 min): {len(large_jumps)} "
          f"({len(large_jumps)/max(len(diffs),1)*100:.1f}%)")
    if 0 < len(large_jumps) <= 15:
        print("  Examples:")
        for j in large_jumps[:8]:
            print(f"    Step {j}: {predictions[j]:.1f} -> {predictions[j+1]:.1f} "
                  f"(Δ={diffs[j]:+.1f})")


def analyze_er_predictions(predictions: np.ndarray, actuals: np.ndarray, model_name: str):
    """Analyze ER prediction noise and background rate."""
    print("\n" + "="*70)
    print(f"EVENT RATE PREDICTION ANALYSIS: {model_name}")
    print("="*70)

    n = min(len(predictions), len(actuals))
    predictions = predictions[:n]
    actuals = actuals[:n]

    print(f"\n  Predictions: n={n}, range=[{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  Mean: {predictions.mean():.4f}, Std: {predictions.std():.4f}, "
          f"Median: {np.median(predictions):.4f}")
    print(f"\n  Actuals: range=[{actuals.min():.2f}, {actuals.max():.1f}]")
    print(f"  Mean: {actuals.mean():.3f}, Zero fraction: {(actuals==0).mean()*100:.1f}%")

    # Background vs signal
    no_event_mask = actuals == 0
    event_mask = actuals > 0

    if no_event_mask.any() and no_event_mask.sum() > 10:
        bg_preds = predictions[no_event_mask]
        print(f"\n  Background (actual=0, N={no_event_mask.sum()}):")
        print(f"    Mean: {bg_preds.mean():.4f}, Std: {bg_preds.std():.4f}")
        print(f"    Median: {np.median(bg_preds):.4f}")
        print(f"    90th pct: {np.percentile(bg_preds, 90):.4f}")
        print(f"    95th pct: {np.percentile(bg_preds, 95):.4f}")
        print(f"    99th pct: {np.percentile(bg_preds, 99):.4f}")
    else:
        bg_preds = None
        print(f"\n  No zero-event minutes in this dataset (sparse data = False)")
        # For non-sparse data, background = periods below median
        below_median_mask = actuals <= np.median(actuals)
        bg_preds = predictions[below_median_mask]
        print(f"  Using below-median actuals as 'low activity' (N={below_median_mask.sum()}):")
        print(f"    Mean pred: {bg_preds.mean():.4f}, Std: {bg_preds.std():.4f}")

    if event_mask.any():
        event_preds = predictions[event_mask]
        print(f"\n  Signal (actual>0, N={event_mask.sum()}):")
        print(f"    Mean: {event_preds.mean():.4f}, Std: {event_preds.std():.4f}")
        print(f"    5th pct: {np.percentile(event_preds, 5):.4f}")
        print(f"    10th pct: {np.percentile(event_preds, 10):.4f}")

    # For non-sparse data, look at high vs low activity
    if (actuals == 0).sum() == 0:
        # Correlation between predictions and actuals
        corr = np.corrcoef(predictions, actuals)[0, 1]
        print(f"\n  Prediction-actual correlation: {corr:.4f}")

        # High activity vs low activity discrimination
        high_mask = actuals >= np.percentile(actuals, 75)
        low_mask = actuals <= np.percentile(actuals, 25)
        if high_mask.any() and low_mask.any():
            high_preds = predictions[high_mask]
            low_preds = predictions[low_mask]
            print(f"\n  High activity (top 25%, actual>={np.percentile(actuals, 75):.1f}):")
            print(f"    Mean pred: {high_preds.mean():.4f}")
            print(f"  Low activity (bottom 25%, actual<={np.percentile(actuals, 25):.1f}):")
            print(f"    Mean pred: {low_preds.mean():.4f}")
            separation = (high_preds.mean() - low_preds.mean()) / max(low_preds.std(), 0.001)
            print(f"  Discrimination ratio (high-low)/low_std: {separation:.2f}")

    # Cumulative ER policy simulation
    print(f"\n  Cumulative ER policy simulation:")
    actual_events_total = actuals.sum()
    print(f"    Total actual events in window: {actual_events_total:.0f}")
    print(f"    Actual event minutes: {(actuals > 0).sum()}")

    for threshold in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        cumulative = 0.0
        sync_count = 0
        sync_steps = []
        for i, p in enumerate(predictions):
            cumulative += max(p, 0)
            if cumulative >= threshold:
                cumulative = 0.0
                sync_count += 1
                sync_steps.append(i)

        if sync_count > 0:
            intervals = np.diff(sync_steps) if len(sync_steps) > 1 else [n]
            mean_interval = np.mean(intervals) if len(intervals) > 0 else n
            # What fraction of syncs capture at least 1 event?
            useful_syncs = 0
            prev_sync = 0
            for s in sync_steps:
                if actuals[prev_sync:s+1].sum() > 0:
                    useful_syncs += 1
                prev_sync = s + 1

            print(f"    T={threshold:5.1f}: {sync_count:4d} syncs, "
                  f"avg interval={mean_interval:.0f}min ({mean_interval/60:.1f}h), "
                  f"useful={useful_syncs}/{sync_count} ({useful_syncs/sync_count*100:.0f}%)")
        else:
            print(f"    T={threshold:5.1f}: 0 syncs (threshold never reached)")

    # Volatility
    diffs = np.diff(predictions)
    abs_diffs = np.abs(diffs)
    print(f"\n  Volatility:")
    print(f"    Mean |Δ|: {abs_diffs.mean():.4f}")
    print(f"    Max |Δ|: {abs_diffs.max():.4f}")
    cv = predictions.std() / max(predictions.mean(), 0.001)
    print(f"    Coefficient of variation: {cv:.3f}")

    # Is the model prediction constant/flat?
    unique_vals = len(np.unique(np.round(predictions, 4)))
    print(f"    Unique prediction values (4 decimals): {unique_vals}")
    if unique_vals < 10:
        print(f"    WARNING: Model predictions are nearly constant!")


if __name__ == "__main__":
    # Get actual values for both tasks
    print("Loading actual values...")
    tte_actuals = get_actual_values("time_to_event", 5000)
    er_actuals = get_actual_values("event_rate", 5000)
    print(f"  TTE actuals: {len(tte_actuals)} points")
    print(f"  ER actuals: {len(er_actuals)} points")

    # --- TTE MODELS ---
    # TFT (used by the actual RL agents with predictions)
    tte_model = "data/novacore/jira/bugs/models/TFT_tft_tte_v1_20260328T123754_070168"
    tte_model_path = DATA_ROOT.parent / tte_model
    if tte_model_path.exists():
        print(f"\nGenerating TTE predictions from TFT model...")
        try:
            tte_preds = get_predictions_via_serving(tte_model, "time_to_event", 5000)
            analyze_tte_predictions(tte_preds, tte_actuals, "TFT_tft_tte_v1")
        except Exception as e:
            print(f"  Failed: {e}")

    # NHiTS TTE
    nhits_tte = "data/novacore/jira/bugs/models/NHiTS_nhits_tte_v1_20260328T122651_579485"
    nhits_tte_path = DATA_ROOT.parent / nhits_tte
    if nhits_tte_path.exists():
        print(f"\nGenerating TTE predictions from NHiTS model...")
        try:
            nhits_preds = get_predictions_via_serving(nhits_tte, "time_to_event", 5000)
            analyze_tte_predictions(nhits_preds, tte_actuals, "NHiTS_nhits_tte_v1")
        except Exception as e:
            print(f"  Failed: {e}")

    print("\n\n")

    # --- ER MODELS ---
    # NHiTS ER (used by the actual RL agents with predictions)
    er_model = "data/novacore/jira/bugs/models/NHiTS_nhits_er_cov_v1_20260330T111859_316271"
    er_model_path = DATA_ROOT.parent / er_model
    if er_model_path.exists():
        print(f"\nGenerating ER predictions from NHiTS model...")
        try:
            er_preds = get_predictions_via_serving(er_model, "event_rate", 5000)
            analyze_er_predictions(er_preds, er_actuals, "NHiTS_nhits_er_cov_v1")
        except Exception as e:
            print(f"  Failed: {e}")

    # ARIMA 15min
    arima_model = "data/novacore/jira/bugs/models/ARIMA_sarimax_15min_v1_20260330T111527_295986"
    arima_path = DATA_ROOT.parent / arima_model
    if arima_path.exists():
        print(f"\nGenerating ER predictions from ARIMA 15min model...")
        try:
            arima_preds = get_predictions_via_serving(arima_model, "event_rate", 5000)
            analyze_er_predictions(arima_preds, er_actuals, "ARIMA_sarimax_15min_v1")
        except Exception as e:
            print(f"  Failed: {e}")
