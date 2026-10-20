"""Problem period attribution and cross-policy overlap analysis."""

import json
import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


_DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

ATTRIBUTION_CACHE_FILE = "problem_period_attributions.json"


def build_problem_mask(
    decisions_df: pd.DataFrame,
    problem_periods: list[dict[str, Any]],
) -> np.ndarray:
    """Per-minute boolean mask: True inside any problem period."""
    mask = np.zeros(len(decisions_df), dtype=bool)
    for p in problem_periods:
        mask[p["start_idx"]:p["end_idx"] + 1] = True
    return mask


def attribute_problem_period(
    decisions_df: pd.DataFrame,
    period: dict[str, Any],
) -> dict[str, Any]:
    """Compute attribution metrics for a single problem period."""
    s, e = period["start_idx"], period["end_idx"]
    chunk = decisions_df.iloc[s:e + 1]

    actions = chunk["action"].values
    actual = chunk["actual_value"].values
    items = chunk["last_sync_items"].values

    n = len(chunk)
    event_minutes = int((actual > 0).sum())
    event_density = event_minutes / n
    silent_fraction = 1.0 - event_density

    sync_mask = actions == 1
    sync_rate = float(sync_mask.mean())
    total_syncs = int(sync_mask.sum())
    empty_syncs = int(((sync_mask) & (items == 0)).sum())
    empty_sync_rate = empty_syncs / max(total_syncs, 1)

    if silent_fraction > 0.9:
        classification = "data_drought"
    elif event_density > 0.3 and sync_rate < 0.1:
        classification = "missed_burst"
    elif empty_sync_rate > 0.8 and total_syncs > 0:
        classification = "over_syncing"
    else:
        classification = "mixed"

    result: dict[str, Any] = {
        "classification": classification,
        "event_density": round(event_density, 4),
        "sync_rate": round(sync_rate, 4),
        "empty_sync_rate": round(empty_sync_rate, 4),
        "silent_fraction": round(silent_fraction, 4),
        "total_events": event_minutes,
        "total_syncs": total_syncs,
        "empty_syncs": empty_syncs,
        "max_actual_value": round(float(actual.max()), 4),
    }

    if "timestamp" in chunk.columns:
        ts = pd.to_datetime(chunk["timestamp"])
        first_ts = ts.iloc[0]
        if pd.notna(first_ts):
            result["hour_of_day_start"] = int(first_ts.hour)
            result["day_of_week_start"] = int(first_ts.dayofweek)

    return result


def compute_problem_period_attributions(
    decisions_df: pd.DataFrame,
    problem_periods: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attribute all problem periods."""
    return [
        attribute_problem_period(decisions_df, p) for p in problem_periods
    ]


def load_or_compute_attributions(
    eval_dir: str,
    decisions_df: pd.DataFrame,
    problem_periods: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Disk-cached attribution: invalidated when decisions.parquet is newer."""
    cache_path = os.path.join(eval_dir, ATTRIBUTION_CACHE_FILE)
    decisions_path = os.path.join(eval_dir, "decisions.parquet")

    if os.path.exists(cache_path):
        if (not os.path.exists(decisions_path)
                or os.path.getmtime(cache_path) >= os.path.getmtime(decisions_path)):
            try:
                with open(cache_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

    attributions = compute_problem_period_attributions(decisions_df, problem_periods)

    try:
        with open(cache_path, "w") as f:
            json.dump(attributions, f, indent=2)
    except OSError:
        pass

    return attributions


def compute_problem_temporal_concentration(
    decisions_df: pd.DataFrame,
    problem_periods: list[dict[str, Any]],
) -> pd.DataFrame:
    """Hour x day-of-week heatmap: fraction of minutes inside problem periods.

    Returns DataFrame with index=hour (0-23), columns=day names.
    """
    mask = build_problem_mask(decisions_df, problem_periods)
    df = decisions_df[["timestamp"]].copy()
    df["_in_problem"] = mask.astype(np.float64)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["_hour"] = df["timestamp"].dt.hour
    df["_dow"] = df["timestamp"].dt.dayofweek

    pivot = df.groupby(["_dow", "_hour"])["_in_problem"].mean().unstack(level=0)
    pivot.columns = [_DAY_NAMES[int(i)] for i in pivot.columns]

    for h in range(24):
        if h not in pivot.index:
            pivot.loc[h] = np.nan
    pivot = pivot.sort_index()
    for d in _DAY_NAMES:
        if d not in pivot.columns:
            pivot[d] = np.nan
    pivot = pivot[_DAY_NAMES]

    return pivot


# Cross-policy overlap


def compute_problem_overlap(
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Pairwise Jaccard and shared/policy-specific masks."""
    names = list(masks.keys())
    n_policies = len(names)

    stacked = np.stack([masks[n] for n in names])

    shared_mask = stacked.all(axis=0)
    majority_mask = stacked.sum(axis=0) > (n_policies / 2)

    policy_specific = {
        name: masks[name] & ~majority_mask for name in names
    }

    jaccard = np.zeros((n_policies, n_policies))
    for i in range(n_policies):
        for j in range(n_policies):
            intersection = int((stacked[i] & stacked[j]).sum())
            union = int((stacked[i] | stacked[j]).sum())
            jaccard[i, j] = intersection / max(union, 1)

    jaccard_df = pd.DataFrame(jaccard, index=names, columns=names)

    return {
        "jaccard_matrix": jaccard_df,
        "shared_mask": shared_mask,
        "majority_mask": majority_mask,
        "policy_specific": policy_specific,
    }


def create_overlap_timeline_figure(
    timestamps: np.ndarray,
    masks: dict[str, np.ndarray],
    shared_mask: np.ndarray,
) -> go.Figure:
    """Stacked binary timeline: one row per policy, downsampled for rendering."""
    names = list(masks.keys())
    n = len(next(iter(masks.values())))

    bin_size = max(1, n // 2000)
    n_bins = n // bin_size

    ts_binned = timestamps[:n_bins * bin_size:bin_size]
    shared_binned = shared_mask[:n_bins * bin_size].reshape(n_bins, bin_size).any(axis=1)

    rows = []
    for name in names:
        m = masks[name][:n_bins * bin_size].reshape(n_bins, bin_size).any(axis=1)
        # 0=ok, 1=policy-specific, 2=shared
        row = np.where(m & shared_binned, 2, np.where(m, 1, 0))
        rows.append(row)

    matrix = np.stack(rows)

    colorscale = [
        [0.0, "white"],
        [0.33, "white"],
        [0.34, "rgba(239,83,80,0.7)"],
        [0.66, "rgba(239,83,80,0.7)"],
        [0.67, "rgba(126,87,194,0.8)"],
        [1.0, "rgba(126,87,194,0.8)"],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=ts_binned,
        y=names,
        colorscale=colorscale,
        zmin=0,
        zmax=2,
        showscale=False,
    ))

    fig.update_layout(
        height=40 + 50 * len(names),
        margin=dict(l=10, r=10, t=30, b=30),
        title="Problem Period Timeline (red=policy-specific, purple=shared)",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def create_jaccard_heatmap(jaccard_df: pd.DataFrame) -> go.Figure:
    """Annotated heatmap of pairwise Jaccard similarity."""
    fig = px.imshow(
        jaccard_df.values,
        x=jaccard_df.columns.tolist(),
        y=jaccard_df.index.tolist(),
        color_continuous_scale="Blues",
        zmin=0,
        zmax=1,
        text_auto=".2f",
        aspect="auto",
        labels=dict(color="Jaccard"),
    )
    fig.update_layout(
        height=300 + 30 * len(jaccard_df),
        margin=dict(l=10, r=10, t=30, b=30),
    )
    return fig
