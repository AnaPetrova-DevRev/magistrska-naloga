"""Core evaluation replay loop for sync policies.

Steps through a Gymnasium environment with a given policy,
recording minute-by-minute decisions and computing summary metrics.

## Age of Information (AoI) — two definitions

This module computes **true AoI**: the time elapsed since the oldest
*unsynced* event appeared in the external system. This measures real
data staleness: if no events have occurred since the last sync, true
AoI is 0 regardless of how long ago the sync was.

This is distinct from the RL training environment's ``current_aoi``,
which is simply "time since last sync" — a monotonically increasing
proxy signal useful for reward shaping during training but NOT an
accurate staleness measure. The env's value is preserved in the
``current_aoi`` column for reward re-scoring; the ``true_aoi`` column
is used for all evaluation metrics (avg_aoi, max_aoi, etc.).

See also: ``rl_training/sync_env.py`` docstrings for the training-side
definition, and ``docs/sync_policy_evaluation.md`` §Age of Information.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from rl_training.reward import RewardConfig
from rl_training.sync_env import SyncSchedulingEnv
from sync_policy_evaluation.policy_loader import Policy

logger = logging.getLogger(__name__)

# Minimum upward TTE jump (in minutes) to count as a real countdown reset.
# At minute resolution a genuine reset always jumps by >= 1 minute; smaller
# increases are interpolation / floating-point noise.
TTE_MIN_JUMP: float = 1.0

DEFAULT_ROLLING_WINDOWS: List[int] = [1440, 10080, 43200]

DEFAULT_PROBLEM_THRESHOLDS: Dict[str, Any] = {
    "aoi_max": 60.0,
    "empty_rate_max": 0.95,
    "reward_percentile": 5,
}


def _detect_event(
    actual_value: float,
    prev_actual_value: Optional[float],
    task_type: str,
    tte_min_jump: float = TTE_MIN_JUMP,
) -> bool:
    """Detect whether an event occurred at the current timestep.

    For event_rate: an event happened if actual_value > 0.
    For time_to_event: an event happened if TTE jumped upward
    by at least *tte_min_jump* minutes (countdown reset after an
    event occurred).  The default filters noise for minute-level
    data — a real reset always jumps by >= 1 minute.
    """
    if task_type == "event_rate":
        return actual_value > 0
    else:
        if prev_actual_value is None:
            return False
        return actual_value > prev_actual_value + tte_min_jump


def run_evaluation(
    policy: Policy,
    env: SyncSchedulingEnv,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Replay a policy on the environment and collect results.

    Args:
        policy: A Policy instance (trained or heuristic).
        env: A SyncSchedulingEnv configured for evaluation
            (episode_length=None, random_start=False).

    Returns:
        Tuple of (decisions_df, summary_metrics).
        decisions_df has columns: timestamp, step, action, actual_value,
            reward, true_aoi, current_aoi, time_since_last_sync,
            cumulative_empty_syncs, last_sync_items.
        summary_metrics is a dict of aggregate stats (AoI metrics
            use true_aoi).
    """
    policy.reset()
    obs, info = env.reset()

    task_type: str = env._task_type
    records: List[Dict[str, Any]] = []
    done = False

    # --- True AoI tracking state ---
    # true_aoi = time since the oldest unsynced event appeared.
    # Stays 0 during calm periods (no events = no staleness).
    first_unsynced_event_step: Optional[int] = None
    prev_actual_value: Optional[float] = None
    step_counter: int = 0

    while not done:
        action = policy.predict(obs)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        actual_value = info["actual_value"]
        event_happened = _detect_event(
            actual_value, prev_actual_value, task_type,
        )

        # Compute true AoI for this step
        if action == 1:  # sync — we captured everything up to now
            first_unsynced_event_step = None
            true_aoi = 0.0
        else:
            if event_happened and first_unsynced_event_step is None:
                first_unsynced_event_step = step_counter
            if first_unsynced_event_step is not None:
                true_aoi = float(step_counter - first_unsynced_event_step)
            else:
                true_aoi = 0.0

        ts = env.current_timestamp
        records.append({
            "timestamp": ts.isoformat() if ts else None,
            "step": info["step"],
            "action": info["action"],
            "actual_value": actual_value,
            "reward": float(reward),
            "true_aoi": true_aoi,
            "current_aoi": info["current_aoi"],
            "time_since_last_sync": info["time_since_last_sync"],
            "cumulative_empty_syncs": info["cumulative_empty_syncs"],
            "last_sync_items": info["last_sync_items"],
        })

        prev_actual_value = actual_value
        step_counter += 1

    decisions_df = pd.DataFrame(records)
    decisions_df = compute_rolling_metrics(decisions_df)

    summary = compute_summary_metrics(records, task_type=task_type)

    problem_periods = detect_problem_periods(decisions_df)
    summary["problem_periods"] = problem_periods
    summary["problem_period_count"] = len(problem_periods)
    summary["problem_period_total_minutes"] = sum(
        p["duration_minutes"] for p in problem_periods
    )

    return decisions_df, summary


def run_evaluation_summary_only(
    policy: Policy,
    env: SyncSchedulingEnv,
) -> Dict[str, Any]:
    """Lightweight evaluation that returns only summary metrics.

    Same replay logic as run_evaluation but skips DataFrame construction,
    rolling metrics, and problem period detection. Used for auxiliary
    train/val splits where only aggregate stats are needed.
    """
    policy.reset()
    obs, info = env.reset()

    task_type: str = env._task_type
    records: List[Dict[str, Any]] = []
    done = False

    first_unsynced_event_step: Optional[int] = None
    prev_actual_value: Optional[float] = None
    step_counter: int = 0

    while not done:
        action = policy.predict(obs)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        actual_value = info["actual_value"]
        event_happened = _detect_event(
            actual_value, prev_actual_value, task_type,
        )

        if action == 1:
            first_unsynced_event_step = None
            true_aoi = 0.0
        else:
            if event_happened and first_unsynced_event_step is None:
                first_unsynced_event_step = step_counter
            if first_unsynced_event_step is not None:
                true_aoi = float(step_counter - first_unsynced_event_step)
            else:
                true_aoi = 0.0

        records.append({
            "action": info["action"],
            "actual_value": actual_value,
            "reward": float(reward),
            "true_aoi": true_aoi,
            "current_aoi": info["current_aoi"],
            "time_since_last_sync": info["time_since_last_sync"],
            "last_sync_items": info["last_sync_items"],
        })

        prev_actual_value = actual_value
        step_counter += 1

    return compute_summary_metrics(records, task_type=task_type)


def compute_summary_metrics(
    records: List[Dict[str, Any]],
    task_type: str = "event_rate",
) -> Dict[str, Any]:
    """Compute aggregate metrics from evaluation records.

    AoI metrics use ``true_aoi`` (real data staleness) when present,
    falling back to ``current_aoi`` (time since last sync) for
    backward compatibility with older artifacts or dashboard-generated
    records that lack the true_aoi field.

    Timing quality metrics (reaction time, event capture rate) require
    ``actual_value`` in the records.  When absent (e.g. re-scored
    records) they default to 0.
    """
    if not records:
        return {}

    total_reward = sum(r["reward"] for r in records)
    total_steps = len(records)
    sync_steps = [r for r in records if r["action"] == 1]
    sync_count = len(sync_steps)

    # Empty syncs: synced but last_sync_items stayed 0
    empty_syncs = sum(1 for r in sync_steps if r["last_sync_items"] == 0.0)

    # Use true_aoi when available, fall back to current_aoi
    aoi_key = "true_aoi" if "true_aoi" in records[0] else "current_aoi"
    aoi_values = [r[aoi_key] for r in records]
    avg_aoi = np.mean(aoi_values)
    max_aoi = np.max(aoi_values)
    median_aoi = np.median(aoi_values)

    # P95 AoI
    p95_aoi = float(np.percentile(aoi_values, 95))

    items_per_sync = [
        r["last_sync_items"] for r in sync_steps
        if r["last_sync_items"] > 0
    ]
    avg_items_per_sync = np.mean(items_per_sync) if items_per_sync else 0

    # --- Timing quality metrics ---
    avg_reaction_time = 0.0
    median_reaction_time = 0.0
    p95_reaction_time = 0.0
    event_capture_rate = 0.0
    total_events_detected = 0

    has_actual = "actual_value" in records[0]
    if has_actual:
        event_steps: List[int] = []
        sync_step_indices = np.array(
            [idx for idx, r in enumerate(records) if r["action"] == 1],
            dtype=np.int64,
        )
        prev_av_m: Optional[float] = None
        for idx, r in enumerate(records):
            av_m = r["actual_value"]
            if _detect_event(av_m, prev_av_m, task_type):
                event_steps.append(idx)
            prev_av_m = av_m

        total_events_detected = len(event_steps)

        if event_steps and len(sync_step_indices) > 0:
            event_arr = np.array(event_steps, dtype=np.int64)
            # For each event, find the first sync at or after it
            insert_pos = np.searchsorted(sync_step_indices, event_arr, side="left")
            reaction_times: List[float] = []
            for j, es in enumerate(event_arr):
                if insert_pos[j] < len(sync_step_indices):
                    reaction_times.append(
                        float(sync_step_indices[insert_pos[j]] - es)
                    )
                else:
                    reaction_times.append(float(total_steps - es))

            avg_reaction_time = float(np.mean(reaction_times))
            median_reaction_time = float(np.median(reaction_times))
            p95_reaction_time = float(np.percentile(reaction_times, 95))

            # Event capture rate: fraction of events synced before next event
            events_captured = 0
            for k in range(len(event_steps)):
                next_event = (
                    event_steps[k + 1] if k + 1 < len(event_steps) else total_steps
                )
                pos = insert_pos[k]
                if (pos < len(sync_step_indices)
                        and sync_step_indices[pos] < next_event):
                    events_captured += 1
            event_capture_rate = events_captured / len(event_steps)

    summary = {
        "total_reward": float(total_reward),
        "mean_reward_per_step": float(total_reward / total_steps),
        "total_steps": total_steps,
        "sync_count": sync_count,
        "sync_frequency": sync_count / total_steps if total_steps else 0,
        "empty_sync_count": empty_syncs,
        "empty_sync_rate": empty_syncs / sync_count if sync_count else 0,
        "avg_aoi": float(avg_aoi),
        "max_aoi": float(max_aoi),
        "median_aoi": float(median_aoi),
        "p95_aoi": p95_aoi,
        "avg_items_per_sync": float(avg_items_per_sync),
        "meaningful_sync_count": len(items_per_sync),
        "avg_reaction_time": avg_reaction_time,
        "median_reaction_time": median_reaction_time,
        "p95_reaction_time": p95_reaction_time,
        "event_capture_rate": event_capture_rate,
        "total_events_detected": total_events_detected,
    }

    # Decision-quality metrics (AoII and SPO Regret)
    if has_actual:
        from sync_policy_evaluation.decision_metrics import (
            compute_aoii_series,
            compute_aoii_summary,
            compute_spo_regret,
        )

        actual_arr = np.array([r["actual_value"] for r in records], dtype=np.float64)
        actions_arr = np.array([r["action"] for r in records], dtype=np.int64)

        aoii = compute_aoii_series(actual_arr, actions_arr, task_type)
        summary.update(compute_aoii_summary(aoii))

        df_for_spo = pd.DataFrame({"actual_value": actual_arr, "action": actions_arr})
        summary.update(compute_spo_regret(df_for_spo, task_type))

    return summary


# Post-hoc reward rescoring


def rescore_decisions(
    decisions_df: pd.DataFrame,
    config: RewardConfig,
) -> np.ndarray:
    """Re-score an evaluation trace with a different reward config.

    The policy's *decisions* are frozen — only the reward assigned to
    each step changes.  Supports all ``aoi_mode`` variants, ``aoi_only``
    reward mode, and ``item_reward_per_unit``.

    Uses ``true_aoi`` (real data staleness) for the AoI penalty when
    the column is present, falling back to ``current_aoi - 1`` (the
    parquet stores the post-increment training proxy) for older
    artifacts that lack it.

    Returns:
        1-D float64 array of rescored per-step rewards.
    """
    actions = decisions_df["action"].values
    actual_values = decisions_df["actual_value"].values
    current_aois = decisions_df["current_aoi"].values
    items = decisions_df["last_sync_items"].values
    true_aois = (
        decisions_df["true_aoi"].values
        if "true_aoi" in decisions_df.columns
        else None
    )

    did_sync = actions == 1
    not_sync = np.logical_not(did_sync)

    # --- Discrete component ---
    if config.reward_mode == "aoi_only":
        discrete = np.where(did_sync, -config.sync_cost, 0.0)
    else:
        if config.mode == "event_rate":
            is_urgent_sync = items > 0
            is_urgent_skip = actual_values > 0
        else:
            is_urgent_sync = actual_values < config.aoi_threshold_minutes
            is_urgent_skip = is_urgent_sync

        not_urgent_sync = np.logical_not(is_urgent_sync)
        discrete = np.select(
            [
                np.logical_and(did_sync, is_urgent_sync),
                np.logical_and(did_sync, not_urgent_sync),
                np.logical_and(not_sync, is_urgent_skip),
            ],
            [
                -config.sync_cost + config.successful_sync_reward,
                -config.empty_sync_penalty,
                -config.late_penalty,
            ],
            default=0.0,
        )

    # --- Continuous AoI penalty ---
    if true_aois is not None:
        reward_aoi = np.where(did_sync, 0.0, true_aois)
    else:
        reward_aoi = np.where(
            did_sync, 0.0, np.maximum(0.0, current_aois - 1.0),
        )

    if config.aoi_mode == "quadratic":
        aoi_penalty = -config.aoi_penalty_per_minute * (reward_aoi ** 2)
    elif config.aoi_mode == "linear_no_threshold":
        aoi_penalty = -config.aoi_penalty_per_minute * reward_aoi
    else:
        aoi_excess = np.maximum(
            0.0, reward_aoi - config.aoi_threshold_minutes,
        )
        aoi_penalty = -config.aoi_penalty_per_minute * aoi_excess

    # --- Item bonus ---
    item_bonus = np.where(
        np.logical_and(did_sync, items > 0),
        config.item_reward_per_unit * items,
        0.0,
    )

    return (discrete + aoi_penalty + item_bonus).astype(np.float64)


# Rolling window metrics


def _numpy_rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    """Rolling mean using numpy cumsum (pandas .rolling() is broken on Python 3.14)."""
    n = len(arr)
    cs = np.empty(n + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(arr, out=cs[1:])
    result = np.empty(n, dtype=np.float64)
    # Partial windows at start
    idx = np.arange(min(w - 1, n))
    result[idx] = cs[idx + 1] / (idx + 1)
    # Full windows
    if n >= w:
        result[w - 1:] = (cs[w:] - cs[:n - w + 1]) / w
    return result


def _numpy_rolling_sum(arr: np.ndarray, w: int) -> np.ndarray:
    """Rolling sum using numpy cumsum."""
    n = len(arr)
    cs = np.empty(n + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(arr, out=cs[1:])
    result = np.empty(n, dtype=np.float64)
    idx = np.arange(min(w - 1, n))
    result[idx] = cs[idx + 1]
    if n >= w:
        result[w - 1:] = cs[w:] - cs[:n - w + 1]
    return result


def compute_rolling_metrics(
    decisions_df: pd.DataFrame,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Add rolling window columns to the decisions DataFrame.

    For each window size *w*, adds:
      - ``rolling_{w}_reward``  — rolling mean reward
      - ``rolling_{w}_aoi``     — rolling mean true AoI
      - ``rolling_{w}_sync_rate`` — rolling fraction of sync actions
      - ``rolling_{w}_empty_rate`` — empty syncs / total syncs in window

    Uses numpy cumsum instead of pandas .rolling() which is broken
    on Python 3.14 / pandas 2.3.3 (returns all NaN on large arrays).
    """
    if windows is None:
        windows = DEFAULT_ROLLING_WINDOWS

    aoi_col = "true_aoi" if "true_aoi" in decisions_df.columns else "current_aoi"
    reward_arr = decisions_df["reward"].values.astype(np.float64)
    aoi_arr = decisions_df[aoi_col].values.astype(np.float64)
    sync_arr = (decisions_df["action"].values == 1).astype(np.float64)
    empty_arr = (
        (decisions_df["action"].values == 1)
        & (decisions_df["last_sync_items"].values == 0.0)
    ).astype(np.float64)

    for w in windows:
        decisions_df[f"rolling_{w}_reward"] = _numpy_rolling_mean(reward_arr, w)
        decisions_df[f"rolling_{w}_aoi"] = _numpy_rolling_mean(aoi_arr, w)
        decisions_df[f"rolling_{w}_sync_rate"] = _numpy_rolling_mean(sync_arr, w)
        rolling_empty = _numpy_rolling_sum(empty_arr, w)
        rolling_sync = _numpy_rolling_sum(sync_arr, w)
        decisions_df[f"rolling_{w}_empty_rate"] = np.where(
            rolling_sync > 0,
            rolling_empty / rolling_sync,
            np.nan,
        )

    return decisions_df


# Problem period detection


def detect_problem_periods(
    decisions_df: pd.DataFrame,
    window: int = 1440,
    thresholds: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Find contiguous stretches where rolling metrics exceed thresholds.

    Returns a list of dicts, each with start/end timestamps, duration,
    which metrics triggered the flag, and severity.
    """
    if thresholds is None:
        thresholds = DEFAULT_PROBLEM_THRESHOLDS

    aoi_col = f"rolling_{window}_aoi"
    empty_col = f"rolling_{window}_empty_rate"
    reward_col = f"rolling_{window}_reward"

    if aoi_col not in decisions_df.columns:
        return []

    n = len(decisions_df)
    aoi_flag = decisions_df[aoi_col] > thresholds["aoi_max"]
    empty_flag = decisions_df[empty_col] > thresholds["empty_rate_max"]

    reward_values = decisions_df[reward_col].dropna()
    if len(reward_values) > 0:
        reward_floor = float(np.nanpercentile(
            reward_values, thresholds["reward_percentile"]
        ))
        reward_flag = decisions_df[reward_col] < reward_floor
    else:
        reward_flag = pd.Series(False, index=decisions_df.index)

    any_flag = (aoi_flag | empty_flag | reward_flag).astype(int)

    # Find contiguous runs of True
    diff = any_flag.diff().fillna(any_flag)
    starts = diff[diff == 1].index.tolist()
    ends = diff[diff == -1].index.tolist()

    # Handle edge: flag active at last row
    if any_flag.iloc[-1] == 1:
        ends.append(decisions_df.index[-1])

    has_ts = "timestamp" in decisions_df.columns
    periods: List[Dict[str, Any]] = []

    for s, e in zip(starts, ends):
        s_iloc = decisions_df.index.get_loc(s)
        e_iloc = decisions_df.index.get_loc(e)
        duration = e_iloc - s_iloc + 1

        triggers = []
        severity: Dict[str, float] = {}
        chunk_aoi = decisions_df.loc[s:e, aoi_col]
        chunk_empty = decisions_df.loc[s:e, empty_col]
        chunk_reward = decisions_df.loc[s:e, reward_col]

        if aoi_flag.loc[s:e].any():
            triggers.append("aoi")
            severity["aoi"] = round(
                float(chunk_aoi.max()) - thresholds["aoi_max"], 2
            )
        if empty_flag.loc[s:e].any():
            triggers.append("empty_rate")
            severity["empty_rate"] = round(
                float(chunk_empty.max()) - thresholds["empty_rate_max"], 2
            )
        if reward_flag.loc[s:e].any():
            triggers.append("reward")
            severity["reward"] = round(
                reward_floor - float(chunk_reward.min()), 4
            )

        period: Dict[str, Any] = {
            "start_idx": int(s_iloc),
            "end_idx": int(e_iloc),
            "duration_minutes": duration,
            "triggers": triggers,
            "severity": severity,
        }
        if has_ts:
            period["start_timestamp"] = str(decisions_df.loc[s, "timestamp"])
            period["end_timestamp"] = str(decisions_df.loc[e, "timestamp"])

        periods.append(period)

    return periods
