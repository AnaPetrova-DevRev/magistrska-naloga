"""Decision-quality metrics: AoII and SPO Regret.

Computes metrics that evaluate sync policy decisions through the lens
of decision utility rather than raw prediction accuracy.

AoII (Age of Incorrect Information): staleness penalized only when the
local cache is actually stale (events have arrived but not been synced).

SPO Regret: gap between the policy's total cost and the oracle-optimal
schedule's cost, computed with full future knowledge.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


# AoII (Age of Incorrect Information)

def _detect_events_array(
    actual_values: np.ndarray,
    task_type: str,
    tte_min_jump: float = 1.0,
) -> np.ndarray:
    """Return boolean mask where True = event occurred at that step."""
    if task_type == "event_rate":
        return actual_values > 0
    else:
        jumps = np.diff(actual_values, prepend=actual_values[0])
        events = jumps > tte_min_jump
        events[0] = False
        return events


def compute_aoii_series(
    actual_values: np.ndarray,
    actions: np.ndarray,
    task_type: str = "event_rate",
    tte_min_jump: float = 1.0,
) -> np.ndarray:
    """Compute per-step AoII from decisions trace.

    AoII(t) = time_since_correct(t) * pending_items(t)

    where:
    - time_since_correct = minutes since cache was last known correct
      (last sync that captured events, OR last calm minute with no pending)
    - pending_items = cumulative events arrived since last sync

    Returns 1-D float64 array of per-step AoII values.
    """
    n = len(actual_values)
    aoii = np.zeros(n, dtype=np.float64)

    events = _detect_events_array(actual_values, task_type, tte_min_jump)

    pending_items = 0.0
    time_since_correct = 0

    for t in range(n):
        if actions[t] == 1:
            pending_items = 0.0
            time_since_correct = 0
        else:
            if events[t]:
                if task_type == "event_rate":
                    pending_items += actual_values[t]
                else:
                    pending_items += 1.0

            if pending_items > 0:
                time_since_correct += 1
            else:
                time_since_correct = 0

        aoii[t] = time_since_correct * pending_items

    return aoii


def compute_aoii_summary(aoii_series: np.ndarray) -> Dict[str, float]:
    """Aggregate AoII series into summary metrics."""
    if len(aoii_series) == 0:
        return {"avg_aoii": 0.0, "max_aoii": 0.0, "median_aoii": 0.0, "p95_aoii": 0.0}
    return {
        "avg_aoii": float(np.mean(aoii_series)),
        "max_aoii": float(np.max(aoii_series)),
        "median_aoii": float(np.median(aoii_series)),
        "p95_aoii": float(np.percentile(aoii_series, 95)),
    }


# SPO Regret (oracle-relative decision quality)

# Weights for the SPO cost function
_AOI_WEIGHT = 0.02  # per minute of staleness
_EMPTY_SYNC_WEIGHT = 1.0  # per wasted sync


def _compute_true_aoi_series(
    actual_values: np.ndarray,
    actions: np.ndarray,
    task_type: str,
    tte_min_jump: float = 1.0,
) -> np.ndarray:
    """Compute true AoI for a given action sequence.

    A sync at step t captures any event at step t (simultaneous).
    """
    n = len(actual_values)
    true_aoi = np.zeros(n, dtype=np.float64)
    events = _detect_events_array(actual_values, task_type, tte_min_jump)

    first_unsynced_event_step = None

    for t in range(n):
        if events[t] and first_unsynced_event_step is None:
            first_unsynced_event_step = t

        if actions[t] == 1:
            first_unsynced_event_step = None

        if first_unsynced_event_step is not None:
            true_aoi[t] = float(t - first_unsynced_event_step)
        else:
            true_aoi[t] = 0.0

    return true_aoi


def _count_empty_syncs(
    actions: np.ndarray,
    actual_values: np.ndarray,
    task_type: str,
    tte_min_jump: float = 1.0,
) -> int:
    """Count syncs that found no pending items.

    A sync at step t captures any event at step t (simultaneous).
    """
    events = _detect_events_array(actual_values, task_type, tte_min_jump)
    pending = 0.0
    empty = 0

    for t in range(len(actions)):
        if events[t]:
            if task_type == "event_rate":
                pending += actual_values[t]
            else:
                pending += 1.0

        if actions[t] == 1:
            if pending == 0:
                empty += 1
            pending = 0.0

    return empty


def compute_oracle_schedule(
    actual_values: np.ndarray,
    task_type: str = "event_rate",
    tte_min_jump: float = 1.0,
) -> np.ndarray:
    """Compute the oracle sync schedule with perfect knowledge.

    The oracle knows exactly when events occur in the external system.
    It syncs at event minutes and never otherwise — achieving zero
    true_aoi and zero empty syncs.
    """
    n = len(actual_values)
    oracle_actions = np.zeros(n, dtype=np.int64)
    events = _detect_events_array(actual_values, task_type, tte_min_jump)

    pending_items = 0.0

    for t in range(n):
        if events[t]:
            if task_type == "event_rate":
                pending_items += actual_values[t]
            else:
                pending_items += 1.0

        if pending_items > 0:
            oracle_actions[t] = 1
            pending_items = 0.0

    return oracle_actions


def compute_spo_regret(
    decisions_df: pd.DataFrame,
    task_type: str = "event_rate",
    tte_min_jump: float = 1.0,
) -> Dict[str, Any]:
    """Compute SPO Regret: policy cost relative to oracle.

    The oracle achieves zero cost (zero staleness, zero empty syncs).
    Policy cost = sum(true_aoi) * aoi_weight + empty_syncs * empty_weight.
    Regret = policy_cost (since oracle_cost = 0).

    This measures the combined "how much did you miss" (staleness) +
    "how much did you waste" (unnecessary syncs).

    Returns dict with:
        spo_regret_total: total cost (higher = worse)
        spo_regret_per_step: cost normalized by steps
        oracle_sync_count: how many syncs the oracle uses
        oracle_avg_aoi: oracle's avg true AoI (always 0 by construction)
    """
    actual_values = decisions_df["actual_value"].values.astype(np.float64)
    actions = decisions_df["action"].values.astype(np.int64)
    n = len(actual_values)

    if n == 0:
        return {
            "spo_regret_total": 0.0,
            "spo_regret_per_step": 0.0,
            "oracle_sync_count": 0,
            "oracle_avg_aoi": 0.0,
        }

    # Policy cost components
    true_aoi = _compute_true_aoi_series(actual_values, actions, task_type, tte_min_jump)
    empty_syncs = _count_empty_syncs(actions, actual_values, task_type, tte_min_jump)

    policy_cost = float(np.sum(true_aoi)) * _AOI_WEIGHT + empty_syncs * _EMPTY_SYNC_WEIGHT

    # Oracle schedule (for reference stats)
    oracle_actions = compute_oracle_schedule(actual_values, task_type, tte_min_jump)

    return {
        "spo_regret_total": float(policy_cost),
        "spo_regret_per_step": float(policy_cost / n),
        "oracle_sync_count": int(np.sum(oracle_actions)),
        "oracle_avg_aoi": 0.0,
    }
