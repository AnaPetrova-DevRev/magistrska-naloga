"""Sync Policy Evaluation Dashboard.

Standalone Streamlit app for visualizing and comparing sync scheduling
policies. Reads evaluation artifacts produced by SyncPolicyEvaluationApp.

Launch: streamlit run sync_policy_dashboard.py
"""

import glob
import hashlib
import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rl_training.reward import RewardConfig, RewardFunction
from sync_policy_evaluation.evaluation_engine import (
    _detect_event,
    compute_summary_metrics,
    detect_problem_periods,
)
from sync_policy_evaluation.problem_period_analysis import (
    build_problem_mask,
    compute_problem_temporal_concentration,
    compute_problem_overlap,
    create_jaccard_heatmap,
    create_overlap_timeline_figure,
    load_or_compute_attributions,
)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
RL_POLICIES_FOLDER = "rl_policies"

# LTTB target points for downsampling large datasets
LTTB_TARGET_POINTS = 5000

# Bump when oracle computation logic changes to invalidate cached artifacts.
ORACLE_VERSION = "v3"

# Bump when rescore logic changes to invalidate cached rescored artifacts.
RESCORE_VERSION = "v2"


# Utilities


def _reward_config_hash(rc_dict: dict) -> str:
    """Deterministic hash of a reward config dict for cache keying."""
    canonical = json.dumps(rc_dict, sort_keys=True, default=str)
    return hashlib.md5(canonical.encode()).hexdigest()[:12]


def _get_reward_config(metadata: dict) -> RewardConfig:
    """Extract RewardConfig from evaluation metadata, with defaults."""
    rc = metadata.get("reward_config", {})
    mode = metadata.get("task_type", rc.get("mode", "event_rate"))
    return RewardConfig.from_dict({**rc, "mode": mode})


def _pct_of_optimal(policy_reward: float, oracle_reward: float) -> float:
    """Percentage of optimal that handles negative rewards correctly.

    Returns 100% when policy matches oracle, lower when worse.
    For negative rewards (typical), uses inverted ratio so that
    a more negative (worse) policy yields a lower percentage.
    """
    if oracle_reward == 0:
        return 100.0 if policy_reward >= 0 else 0.0
    if oracle_reward > 0:
        return max(0.0, policy_reward / oracle_reward * 100)
    if policy_reward >= 0:
        return 100.0
    return min(100.0, (oracle_reward / policy_reward) * 100)


# Metrics where lower = better (negative delta is good).
_LOWER_IS_BETTER = {
    "avg_aoi", "max_aoi", "p95_aoi", "median_aoi",
    "empty_sync_rate", "avg_reaction_time", "p95_reaction_time",
}

# Metrics shown in the train/test comparison table.
_SPLIT_COMPARE_METRICS = [
    ("mean_reward_per_step", "Mean Reward/Step", ".4f"),
    ("avg_aoi", "Avg AoI (min)", ",.1f"),
    ("p95_aoi", "P95 AoI (min)", ",.1f"),
    ("max_aoi", "Max AoI (min)", ",.0f"),
    ("avg_aoii", "Avg AoII", ",.1f"),
    ("spo_regret_per_step", "SPO Regret/Step", ".4f"),
    ("sync_frequency", "Sync Frequency", ".2%"),
    ("empty_sync_rate", "Empty Sync Rate", ".2%"),
    ("avg_reaction_time", "Avg Reaction Time", ".1f"),
    ("event_capture_rate", "Event Capture Rate", ".1%"),
]


@st.cache_data
def compute_per_split_metrics(
    eval_dir: str,
    train_end_ts: str | None,
    val_end_ts: str | None,
    task_type: str = "event_rate",
) -> dict[str, dict[str, Any]] | None:
    """Slice decisions by split boundaries and compute metrics per split.

    Persists results to ``split_metrics.json`` inside *eval_dir* so that
    subsequent dashboard loads skip the expensive parquet read.
    """
    if not train_end_ts:
        return None

    cache_path = os.path.join(eval_dir, "split_metrics.json")
    decisions_path = os.path.join(eval_dir, "decisions.parquet")

    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            result = {
                k: v for k, v in cached.items() if not k.startswith("_")
            }
            if len(result) >= 2:
                return result
        except Exception:
            pass

    decisions = load_decisions(eval_dir)
    if decisions is None or decisions.empty:
        return None

    train_end = pd.Timestamp(train_end_ts)
    val_end = pd.Timestamp(val_end_ts) if val_end_ts else None

    train_df = decisions[decisions["timestamp"] <= train_end]
    if val_end is not None:
        val_df = decisions[
            (decisions["timestamp"] > train_end)
            & (decisions["timestamp"] <= val_end)
        ]
        test_df = decisions[decisions["timestamp"] > val_end]
    else:
        val_df = pd.DataFrame()
        test_df = decisions[decisions["timestamp"] > train_end]

    result: dict[str, dict[str, Any]] = {}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if df is not None and not df.empty:
            records = df.to_dict("records")
            result[name] = compute_summary_metrics(records, task_type=task_type)

    try:
        to_save = dict(result) if len(result) >= 2 else {"_empty": True}
        to_save["_train_end_ts"] = train_end_ts
        to_save["_val_end_ts"] = val_end_ts
        with open(cache_path, "w") as f:
            json.dump(to_save, f, default=str)
    except Exception:
        pass

    if len(result) < 2:
        return None
    return result


# LTTB downsampling


def lttb_downsample(
    x: np.ndarray, y: np.ndarray, target: int = LTTB_TARGET_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Largest-Triangle-Three-Buckets downsampling.

    Preserves local peaks and valleys much better than block-mean.
    Returns (x_out, y_out) arrays of length <= target.
    """
    n = len(x)
    if n <= target:
        return x, y

    out_x = np.empty(target)
    out_y = np.empty(target)

    # Always keep first and last point
    out_x[0], out_y[0] = x[0], y[0]
    out_x[-1], out_y[-1] = x[-1], y[-1]

    bucket_size = (n - 2) / (target - 2)

    a_idx = 0  # index of previously selected point
    for i in range(1, target - 1):
        # Bucket boundaries
        b_start = int((i - 1) * bucket_size) + 1
        b_end = int(i * bucket_size) + 1
        b_end = min(b_end, n)

        # Next bucket average (for triangle area)
        c_start = int(i * bucket_size) + 1
        c_end = int((i + 1) * bucket_size) + 1
        c_end = min(c_end, n)
        avg_x = np.mean(x[c_start:c_end].astype(np.float64))
        avg_y = np.mean(y[c_start:c_end])

        # Find point in current bucket with max triangle area
        best_idx = b_start
        max_area = -1.0
        ax, ay = float(x[a_idx]), float(y[a_idx])
        for j in range(b_start, b_end):
            area = abs(
                (float(x[j]) - ax) * (avg_y - ay)
                - (float(y[j]) - ay) * (avg_x - ax)
            )
            if area > max_area:
                max_area = area
                best_idx = j

        out_x[i] = x[best_idx]
        out_y[i] = y[best_idx]
        a_idx = best_idx

    return out_x, out_y


# Cron pattern helpers


def _load_cron_patterns_for_eval(
    metadata: dict,
) -> tuple[list[dict], list[dict]]:
    """Try to load cron patterns from the trained policy's artifact dir.

    Returns:
        (daily_patterns, hourly_patterns) — both may be empty lists.
    """
    policy_path = metadata.get("policy_path")
    if not policy_path:
        return [], []
    cron_file = os.path.join(policy_path, "cron_patterns.json")
    if not os.path.exists(cron_file):
        return [], []
    try:
        with open(cron_file) as f:
            data = json.load(f)
        return data.get("patterns", []), data.get("hourly_patterns", [])
    except Exception:
        return [], []


def _add_cron_vrects(
    fig: go.Figure,
    timestamps: np.ndarray,
    cron_patterns: list[dict],
) -> None:
    """Add semi-transparent vertical bands for cron slots on a timeline chart."""
    if not cron_patterns or len(timestamps) == 0:
        return

    slot_set = {(p["hour"], p["minute"]) for p in cron_patterns}
    ts_series = pd.to_datetime(timestamps)
    start_ts = ts_series.min()
    end_ts = ts_series.max()

    # Generate cron slot timestamps within the visible range
    # To avoid thousands of vrects, cap at 200
    cron_times = []
    current = start_ts.normalize()  # midnight of first day
    while current <= end_ts:
        for h, m in sorted(slot_set):
            slot_ts = current + pd.Timedelta(hours=h, minutes=m)
            if start_ts <= slot_ts <= end_ts:
                cron_times.append(slot_ts)
        current += pd.Timedelta(days=1)

    for ct in cron_times[:200]:
        fig.add_vrect(
            x0=ct, x1=ct + pd.Timedelta(minutes=1),
            fillcolor="rgba(255, 165, 0, 0.15)",
            line_width=0,
            layer="below",
        )

    # Add a single annotation for the legend
    if cron_times:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color="rgba(255, 165, 0, 0.5)", symbol="square"),
            name="Cron window",
            showlegend=True,
        ))


def _add_hourly_cron_vrects(
    fig: go.Figure,
    timestamps: np.ndarray,
    hourly_patterns: list[dict],
) -> None:
    """Add semi-transparent vertical bands for hourly cron slots."""
    if not hourly_patterns or len(timestamps) == 0:
        return

    minute_set = {p["minute"] for p in hourly_patterns}
    ts_series = pd.to_datetime(timestamps)
    start_ts = ts_series.min()
    end_ts = ts_series.max()

    # Generate hourly cron slot timestamps within visible range (cap at 200)
    cron_times = []
    current = start_ts.normalize()
    while current <= end_ts and len(cron_times) < 200:
        for hour in range(24):
            for m in sorted(minute_set):
                slot_ts = current + pd.Timedelta(hours=hour, minutes=m)
                if start_ts <= slot_ts <= end_ts:
                    cron_times.append(slot_ts)
        current += pd.Timedelta(days=1)

    for ct in cron_times[:200]:
        fig.add_vrect(
            x0=ct, x1=ct + pd.Timedelta(minutes=1),
            fillcolor="rgba(100, 149, 237, 0.15)",
            line_width=0,
            layer="below",
        )

    if cron_times:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(
                size=10, color="rgba(100, 149, 237, 0.5)", symbol="square",
            ),
            name="Hourly cron window",
            showlegend=True,
        ))


def _add_problem_period_vrects(
    fig: go.Figure,
    problem_periods: list[dict],
) -> None:
    """Add semi-transparent red vertical bands for problem periods."""
    if not problem_periods:
        return
    for p in problem_periods:
        start = p.get("start_timestamp")
        end = p.get("end_timestamp")
        if not start or not end or str(start) == "None" or str(end) == "None":
            continue
        fig.add_vrect(
            x0=pd.Timestamp(start),
            x1=pd.Timestamp(end),
            fillcolor="rgba(255, 0, 0, 0.10)",
            line_width=0,
            layer="below",
        )
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=10, color="rgba(255, 0, 0, 0.3)", symbol="square"),
        name="Problem period",
        showlegend=True,
    ))


# Plotly chart helpers


def create_timeline_chart(
    timestamps: np.ndarray,
    values: np.ndarray,
    title: str,
    y_label: str = "",
    traces: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    vlines: list[dict] | None = None,
    markers: list[dict] | None = None,
    height: int = 350,
    skip_downsample: bool = False,
) -> go.Figure:
    """Build a Plotly Scattergl figure with rangeslider.

    Args:
        timestamps: x-axis datetime array for the primary trace.
        values: y-axis values for the primary trace.
        title: Chart title.
        y_label: Y-axis label.
        traces: Optional additional named traces {name: (timestamps, values)}.
        vlines: Optional vertical lines [{x: datetime, label: str, color: str}].
        markers: Optional marker overlays. Each dict has keys:
            name (str), timestamps (np.ndarray), values (np.ndarray),
            color (str), symbol (str, default "triangle-up"),
            size (int, default 7).
        height: Figure height in pixels.
        skip_downsample: If True, skip LTTB downsampling (use for log scale).
    """
    fig = go.Figure()

    # LTTB downsample the primary trace (skip for log scale)
    ts_epoch = timestamps.astype(np.int64)
    if skip_downsample:
        ds_epoch, ds_vals = ts_epoch, values
    else:
        ds_epoch, ds_vals = lttb_downsample(ts_epoch, values)
    ds_ts = pd.to_datetime(ds_epoch)

    fig.add_trace(go.Scattergl(
        x=ds_ts, y=ds_vals, mode="lines",
        name=title, line=dict(width=1),
    ))

    # Additional traces
    if traces:
        for name, (t_ts, t_vals) in traces.items():
            t_epoch = t_ts.astype(np.int64)
            if skip_downsample:
                dt_ep, dt_v = t_epoch, t_vals
            else:
                dt_ep, dt_v = lttb_downsample(t_epoch, t_vals)
            fig.add_trace(go.Scattergl(
                x=pd.to_datetime(dt_ep), y=dt_v, mode="lines",
                name=name, line=dict(width=1, dash="dot"),
            ))

    # Marker overlays (e.g. sync points)
    if markers:
        for m in markers:
            m_ts = pd.to_datetime(m["timestamps"])
            fig.add_trace(go.Scattergl(
                x=m_ts,
                y=m["values"],
                mode="markers",
                name=m["name"],
                marker=dict(
                    color=m.get("color", "red"),
                    symbol=m.get("symbol", "triangle-up"),
                    size=m.get("size", 7),
                    opacity=m.get("opacity", 0.7),
                ),
            ))

    # Vertical reference lines — use add_shape + add_annotation instead
    # of add_vline because Plotly's annotation_text handling calls
    # sum() on datetime values, which fails in newer pandas.
    if vlines:
        for vl in vlines:
            vl_x = vl["x"]
            color = vl.get("color", "red")
            fig.add_shape(
                type="line", x0=vl_x, x1=vl_x, y0=0, y1=1,
                yref="paper", line=dict(color=color, dash="dash"),
            )
            label = vl.get("label", "")
            if label:
                fig.add_annotation(
                    x=vl_x, y=1, yref="paper",
                    text=label, showarrow=False,
                    font=dict(color=color, size=10),
                    yanchor="bottom",
                )

    fig.update_layout(
        title=title,
        yaxis_title=y_label,
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
        height=height,
        margin=dict(l=50, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def create_multi_trace_chart(
    series_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str,
    y_label: str = "",
    height: int = 350,
) -> go.Figure:
    """Build a multi-trace Plotly chart for policy comparison."""
    fig = go.Figure()
    for name, (timestamps, values) in series_dict.items():
        ts_epoch = timestamps.astype(np.int64)
        ds_epoch, ds_vals = lttb_downsample(ts_epoch, values)
        fig.add_trace(go.Scattergl(
            x=pd.to_datetime(ds_epoch), y=ds_vals, mode="lines",
            name=name, line=dict(width=1),
        ))

    fig.update_layout(
        title=title,
        yaxis_title=y_label,
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
        height=height,
        margin=dict(l=50, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# Temporal heatmap


_DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def compute_temporal_heatmap_data(
    decisions_df: pd.DataFrame,
    metric: str = "true_aoi",
) -> pd.DataFrame:
    """Compute hour-of-day x day-of-week mean of a metric.

    Returns a DataFrame with index=hour (0-23) and columns=day name.
    """
    df = decisions_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["_hour"] = df["timestamp"].dt.hour
    df["_dow"] = df["timestamp"].dt.dayofweek

    if metric == "sync_rate":
        col = "action"
    elif metric == "empty_rate":
        df["_empty_sync"] = (
            (df["action"] == 1) & (df["last_sync_items"] == 0)
        ).astype(float)
        col = "_empty_sync"
    elif metric == "reward":
        col = "reward"
    else:
        col = "true_aoi" if "true_aoi" in df.columns else "current_aoi"

    pivot = df.groupby(["_dow", "_hour"])[col].mean().unstack(level=0)

    pivot.columns = [_DAY_NAMES[i] for i in pivot.columns]
    for h in range(24):
        if h not in pivot.index:
            pivot.loc[h] = np.nan
    pivot = pivot.sort_index()
    for d in _DAY_NAMES:
        if d not in pivot.columns:
            pivot[d] = np.nan
    pivot = pivot[_DAY_NAMES]

    return pivot


def create_performance_heatmap(
    heatmap_data: pd.DataFrame,
    metric_name: str,
    cron_patterns: list[dict] | None = None,
) -> go.Figure:
    """Create hour x day-of-week heatmap with optional cron overlay."""
    invert = metric_name in ("true_aoi", "aoi", "empty_rate", "problem_rate")
    colorscale = "RdYlGn" if not invert else "RdYlGn_r"

    fig = px.imshow(
        heatmap_data.values,
        x=heatmap_data.columns.tolist(),
        y=[f"{h:02d}:00" for h in heatmap_data.index],
        color_continuous_scale=colorscale,
        aspect="auto",
        labels=dict(x="Day of Week", y="Hour of Day", color=metric_name),
    )

    if cron_patterns:
        for p in cron_patterns:
            hour_label = f"{p['hour']:02d}:00"
            for day in _DAY_NAMES:
                fig.add_trace(go.Scatter(
                    x=[day], y=[hour_label], mode="markers",
                    marker=dict(
                        symbol="square-open", size=14,
                        color="rgba(255, 165, 0, 0.9)",
                        line=dict(width=2),
                    ),
                    showlegend=False,
                ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(
                symbol="square-open", size=12,
                color="rgba(255, 165, 0, 0.9)",
                line=dict(width=2),
            ),
            name="Cron pattern hour",
            showlegend=True,
        ))

    fig.update_layout(
        title=f"Temporal Performance: {metric_name}",
        height=500,
        margin=dict(l=60, r=20, t=40, b=30),
    )
    return fig


# Data loading


@st.cache_data(ttl=60)
def discover_evaluations() -> dict[str, list[dict]]:
    """Scan data/ for evaluation artifacts, grouped by sync unit."""
    evaluations: dict[str, list[dict]] = {}

    pattern = os.path.join(
        DATA_ROOT, "*", "*", "*", RL_POLICIES_FOLDER, "*", "summary_metrics.json",
    )
    for summary_path in sorted(glob.glob(pattern)):
        eval_dir = os.path.dirname(summary_path)
        policy_name = os.path.basename(eval_dir)

        rel = os.path.relpath(eval_dir, DATA_ROOT)
        parts = rel.split(os.sep)
        sync_unit = "/".join(parts[:3]) if len(parts) >= 4 else parts[0]

        with open(summary_path) as f:
            summary = json.load(f)

        meta_path = os.path.join(eval_dir, "policy_metadata.json")
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)

        eval_start, eval_end = _read_eval_time_range(eval_dir, metadata)

        split_meta = metadata.get("data_split", {})
        train_end_ts = split_meta.get("train_end_timestamp")
        val_end_ts = split_meta.get("val_end_timestamp")

        gen_label = "—"
        gen_pct = summary.get("generalization_train_to_test_pct")
        if gen_pct is not None:
            gen_label = f"{gen_pct:+.1f}%"
        elif train_end_ts:
            task_type = metadata.get("task_type", "event_rate")
            sm = compute_per_split_metrics(
                eval_dir, train_end_ts, val_end_ts, task_type=task_type,
            )
            if sm and "train" in sm:
                out_split = "test" if "test" in sm else "val"
                if out_split in sm:
                    tr = sm["train"].get("mean_reward_per_step", 0)
                    outr = sm[out_split].get("mean_reward_per_step", 0)
                    if tr != 0:
                        gen_label = f"{(outr - tr) / abs(tr) * 100:+.1f}%"

        entry = {
            "policy_name": metadata.get("policy_name", policy_name),
            "eval_dir": eval_dir,
            "summary": summary,
            "metadata": metadata,
            "eval_start": eval_start,
            "eval_end": eval_end,
            "train_end_timestamp": train_end_ts,
            "val_end_timestamp": val_end_ts,
            "eval_split": metadata.get("eval_split", "all"),
            "generalization": gen_label,
        }

        evaluations.setdefault(sync_unit, []).append(entry)

    return evaluations


def _read_eval_time_range(
    eval_dir: str, metadata: dict,
) -> tuple[str | None, str | None]:
    """Get eval time range from metadata or parquet.

    Tries ``data_split`` timestamps first (instant), then
    ``eval_start``/``eval_end`` in metadata (also instant).
    Falls back to reading the parquet timestamp column only if
    neither is available, and persists the result into
    ``policy_metadata.json`` so subsequent loads are instant.
    """
    split = metadata.get("data_split", {})
    eval_split = metadata.get("eval_split", "all")
    if split:
        if eval_split == "all" and split.get("data_start_timestamp"):
            return split["data_start_timestamp"], split.get(
                "data_end_timestamp",
                split.get("val_end_timestamp"),
            )
        if eval_split == "val" and split.get("train_end_timestamp"):
            return split["train_end_timestamp"], split.get("val_end_timestamp")
        if eval_split == "test" and split.get("val_end_timestamp"):
            return split["val_end_timestamp"], split.get("data_end_timestamp")

    if metadata.get("eval_start"):
        return metadata["eval_start"], metadata.get("eval_end")

    path = os.path.join(eval_dir, "decisions.parquet")
    if not os.path.exists(path):
        return None, None
    try:
        ts_col = pd.read_parquet(path, columns=["timestamp"])
        if ts_col.empty:
            return None, None
        ts = pd.to_datetime(ts_col["timestamp"]).dropna()
        if ts.empty:
            return None, None
        start, end = str(ts.iloc[0]), str(ts.iloc[-1])
        meta_path = os.path.join(eval_dir, "policy_metadata.json")
        if os.path.exists(meta_path):
            try:
                metadata["eval_start"] = start
                metadata["eval_end"] = end
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)
            except Exception:
                pass
        return start, end
    except Exception:
        return None, None


@st.cache_data
def load_decisions(eval_dir: str) -> pd.DataFrame | None:
    """Load decisions.parquet with parsed timestamps."""
    path = os.path.join(eval_dir, "decisions.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["timestamp"])
    return df


# Oracle ("perfect") policy computation


def _oracle_maintenance_interval(
    rc: RewardConfig,
    maintenance_interval_override: float | None = None,
) -> float | None:
    """Resolve the oracle maintenance sync interval.

    Under true AoI (the default for evaluation), maintenance syncs are
    never beneficial: calm periods have true_aoi=0, so the AoI penalty
    is already zero and a sync only incurs empty_sync_penalty.  Therefore
    the optimal oracle syncs only at events (returns None).

    When ``maintenance_interval_override`` is set, it represents a
    user-imposed constraint ("what if I must sync at least every N
    minutes?") and is returned as-is.
    """
    if maintenance_interval_override is not None:
        return maintenance_interval_override
    return None


def compute_oracle_decisions(
    decisions_df: pd.DataFrame,
    reward_config: RewardConfig,
    maintenance_interval_override: float | None = None,
) -> list[dict[str, Any]]:
    """Compute oracle policy decisions from ground-truth data.

    The oracle has perfect foresight: it syncs at every timestep with
    events, capturing them immediately.  Rewards are scored using true
    AoI (real data staleness), not the training-proxy AoI.

    When ``maintenance_interval_override`` is set, the oracle also does
    periodic maintenance syncs during calm periods (using time-since-
    last-sync for the interval check).  Without it, the oracle never
    does maintenance syncs because true AoI is 0 during calm periods.

    Args:
        maintenance_interval_override: If set, forces maintenance syncs
            every N minutes.  E.g. 60 for "hourly oracle".
    """
    mode = reward_config.mode
    rf = RewardFunction(reward_config)
    maint_interval = _oracle_maintenance_interval(
        reward_config, maintenance_interval_override,
    )

    actual_values = decisions_df["actual_value"].values
    timestamps = decisions_df["timestamp"].values

    records = []
    time_since_sync = 0.0
    cum_empty = 0
    pending_items = 0.0
    first_unsynced_event_step = None
    prev_av = None

    for i in range(len(actual_values)):
        av = float(actual_values[i])

        if mode == "event_rate":
            pending_items += av

        event_happened = _detect_event(av, prev_av, mode)
        has_event = event_happened

        needs_maintenance = (
            maint_interval is not None
            and not has_event
            and time_since_sync >= maint_interval
        )

        should_sync = has_event or needs_maintenance
        action = 1 if should_sync else 0

        # Compute true AoI for this step BEFORE reward (sync resets to 0).
        if action == 1:
            step_true_aoi = 0.0
        else:
            if event_happened and first_unsynced_event_step is None:
                first_unsynced_event_step = i
            step_true_aoi = (
                float(i - first_unsynced_event_step)
                if first_unsynced_event_step is not None
                else 0.0
            )

        # Reward uses true AoI — sync steps see 0 (freshly synced).
        if action == 1 and mode == "event_rate":
            reward_value = pending_items
        else:
            reward_value = av
        reward = rf.compute(
            action=action, actual_value=reward_value,
            current_aoi=step_true_aoi,
        )

        # Update state
        if action == 1:
            if not has_event:
                cum_empty += 1
            time_since_sync = 0.0
            first_unsynced_event_step = None
            true_aoi = 0.0
            if mode == "event_rate":
                last_items = pending_items if has_event else 0.0
            else:
                last_items = 1.0 if has_event else 0.0
            pending_items = 0.0
        else:
            time_since_sync += 1.0
            true_aoi = step_true_aoi
            last_items = 0.0

        records.append({
            "timestamp": timestamps[i],
            "step": i,
            "action": action,
            "actual_value": av,
            "reward": reward,
            "true_aoi": true_aoi,
            "current_aoi": time_since_sync,
            "time_since_last_sync": time_since_sync,
            "cumulative_empty_syncs": cum_empty,
            "last_sync_items": last_items,
        })
        prev_av = av

    return records


@st.cache_data
def load_or_compute_oracle(
    eval_dir: str, rc_dict: dict, mode: str,
    maintenance_interval_override: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Load oracle from disk cache or compute and persist.

    Args:
        maintenance_interval_override: If set, uses this fixed maintenance
            interval instead of the mathematically optimal one.  Causes
            the cache key to include the interval so different variants
            are stored separately.
    """
    # Include override + oracle version in cache key so stale caches
    # are invalidated when oracle logic changes.
    key_extra = f"_maint{maintenance_interval_override}" if maintenance_interval_override is not None else ""
    rc_hash = _reward_config_hash(rc_dict) + key_extra + f"_{ORACLE_VERSION}"
    suffix = f"_{int(maintenance_interval_override)}min" if maintenance_interval_override is not None else ""
    oracle_parquet = os.path.join(eval_dir, f"oracle_decisions{suffix}.parquet")
    oracle_summary_path = os.path.join(eval_dir, f"oracle_summary{suffix}.json")

    # Check disk cache
    if os.path.exists(oracle_parquet) and os.path.exists(oracle_summary_path):
        with open(oracle_summary_path) as f:
            cached = json.load(f)
        if cached.get("_rc_hash") == rc_hash:
            src_parquet = os.path.join(eval_dir, "decisions.parquet")
            if (not os.path.exists(src_parquet)
                    or os.path.getmtime(oracle_parquet) >= os.path.getmtime(src_parquet)):
                oracle_df = pd.read_parquet(oracle_parquet)
                oracle_df["timestamp"] = pd.to_datetime(oracle_df["timestamp"])
                return oracle_df, cached

    # Compute
    decisions = load_decisions(eval_dir)
    if decisions is None:
        return pd.DataFrame(), {}

    rc = RewardConfig.from_dict({**rc_dict, "mode": mode})
    records = compute_oracle_decisions(
        decisions, rc,
        maintenance_interval_override=maintenance_interval_override,
    )
    summary = compute_summary_metrics(records, task_type=mode)
    summary["_rc_hash"] = rc_hash
    if maintenance_interval_override is not None:
        summary["maintenance_interval"] = maintenance_interval_override

    # Persist
    oracle_df = pd.DataFrame(records)
    oracle_df.to_parquet(oracle_parquet, index=False)
    with open(oracle_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    oracle_df["timestamp"] = pd.to_datetime(oracle_df["timestamp"])
    return oracle_df, summary


# Vectorized reward re-scoring


def vectorized_rescore(
    actions: np.ndarray,
    actual_values: np.ndarray,
    current_aois: np.ndarray,
    last_sync_items: np.ndarray,
    config: RewardConfig,
    true_aois: np.ndarray | None = None,
) -> np.ndarray:
    """Re-score decisions with new reward parameters (vectorized).

    Uses true AoI (real data staleness) for the continuous AoI penalty
    when available, falling back to current_aoi (training-proxy) for
    older artifacts.  true_aoi is stored at-step (not post-increment),
    so no -1 adjustment is needed: sync steps are already 0, and skip
    steps reflect actual staleness directly.
    """
    did_sync = actions == 1

    # Effective reward_value: accumulated items for sync (event_rate), else actual
    if config.mode == "event_rate":
        reward_value = np.where(did_sync, last_sync_items, actual_values)
        is_urgent_for_skip = actual_values > 0
        is_urgent_for_sync = reward_value > 0
    else:
        reward_value = actual_values
        is_urgent_for_skip = actual_values < config.aoi_threshold_minutes
        is_urgent_for_sync = is_urgent_for_skip

    # Discrete reward component.
    # Avoid ~ operator on bool arrays — Python 3.14 / numpy 2.1 has
    # inconsistent precedence with & that silently produces wrong results.
    not_urgent_sync = np.logical_not(is_urgent_for_sync)
    not_sync = np.logical_not(did_sync)
    discrete = np.select(
        [
            np.logical_and(did_sync, is_urgent_for_sync),
            np.logical_and(did_sync, not_urgent_sync),
            np.logical_and(not_sync, is_urgent_for_skip),
        ],
        [
            -config.sync_cost + config.successful_sync_reward,
            -config.empty_sync_penalty,
            -config.late_penalty,
        ],
        default=0.0,
    )

    # AoI penalty: use true AoI when available (evaluation semantics),
    # fall back to current_aoi with -1 adjustment (training semantics).
    if true_aois is not None:
        reward_aoi = np.where(did_sync, 0.0, true_aois)
    else:
        reward_aoi = np.where(did_sync, 0.0, np.maximum(0.0, current_aois - 1.0))
    aoi_excess = np.maximum(0.0, reward_aoi - config.aoi_threshold_minutes)
    aoi_penalty = -config.aoi_penalty_per_minute * aoi_excess

    return discrete + aoi_penalty


@st.cache_data
def load_or_compute_rescored(
    eval_dir: str, rc_dict: dict, mode: str,
) -> tuple[np.ndarray, dict]:
    """Load re-scored rewards from disk cache or compute and persist."""
    rc_hash = _reward_config_hash(rc_dict) + f"_{RESCORE_VERSION}"
    cache_dir = os.path.join(eval_dir, "rescored")
    cache_parquet = os.path.join(cache_dir, f"{rc_hash}.parquet")
    cache_summary = os.path.join(cache_dir, f"{rc_hash}_summary.json")

    # Check disk cache
    if os.path.exists(cache_parquet) and os.path.exists(cache_summary):
        src_parquet = os.path.join(eval_dir, "decisions.parquet")
        if (not os.path.exists(src_parquet)
                or os.path.getmtime(cache_parquet) >= os.path.getmtime(src_parquet)):
            cached_df = pd.read_parquet(cache_parquet)
            with open(cache_summary) as f:
                cached_s = json.load(f)
            return cached_df["rescored_reward"].values, cached_s

    # Compute
    decisions = load_decisions(eval_dir)
    if decisions is None:
        return np.array([]), {}

    config = RewardConfig.from_dict({**rc_dict, "mode": mode})
    true_aois = decisions["true_aoi"].values if "true_aoi" in decisions.columns else None
    rescored = vectorized_rescore(
        decisions["action"].values,
        decisions["actual_value"].values,
        decisions["current_aoi"].values,
        decisions["last_sync_items"].values,
        config,
        true_aois=true_aois,
    )

    # Compute summary from re-scored rewards
    # Pass true_aoi through when available (new evaluations have it)
    has_true_aoi = "true_aoi" in decisions.columns
    records = []
    for i in range(len(decisions)):
        row = decisions.iloc[i]
        rec = {
            "action": int(row["action"]),
            "reward": float(rescored[i]),
            "current_aoi": float(row["current_aoi"]),
            "last_sync_items": float(row["last_sync_items"]),
        }
        if has_true_aoi:
            rec["true_aoi"] = float(row["true_aoi"])
        records.append(rec)
    summary = compute_summary_metrics(records)

    # Persist
    os.makedirs(cache_dir, exist_ok=True)
    pd.DataFrame({"rescored_reward": rescored}).to_parquet(cache_parquet, index=False)
    with open(cache_summary, "w") as f:
        json.dump(summary, f, indent=2)

    return rescored, summary


# Sidebar controls


def sidebar_date_filter(evaluations: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Sidebar date range filter — returns filtered evaluations."""
    st.sidebar.header("Filters")

    all_starts, all_ends = [], []
    for evals in evaluations.values():
        for e in evals:
            if e["eval_start"]:
                t = pd.Timestamp(e["eval_start"])
                if not pd.isna(t):
                    all_starts.append(t)
            if e["eval_end"]:
                t = pd.Timestamp(e["eval_end"])
                if not pd.isna(t):
                    all_ends.append(t)

    if not all_starts:
        return evaluations

    global_min = min(all_starts).date()
    global_max = max(all_ends).date() if all_ends else global_min

    date_range = st.sidebar.date_input(
        "Evaluation date range",
        value=(global_min, global_max),
        min_value=global_min,
        max_value=global_max,
    )

    if not isinstance(date_range, tuple) or len(date_range) != 2:
        return evaluations

    filter_start, filter_end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])

    filtered = {}
    for su, evals in evaluations.items():
        kept = []
        for e in evals:
            if not e["eval_start"] or not e["eval_end"]:
                kept.append(e)
                continue
            es = pd.Timestamp(e["eval_start"])
            ee = pd.Timestamp(e["eval_end"])
            if es <= filter_end and ee >= filter_start:
                kept.append(e)
        if kept:
            filtered[su] = kept

    return filtered




# Dashboard pages


def _build_all_evals(evaluations: dict[str, list[dict]]) -> list[tuple[str, dict]]:
    return [
        (su, e)
        for su, evals in evaluations.items()
        for e in evals
    ]


def page_overview(evaluations: dict[str, list[dict]]):
    """Overview: summary table of all evaluations."""
    st.header("Policy Evaluations Overview")

    if not evaluations:
        st.info("No evaluation artifacts found. Run `evaluate_sync_policy` first.")
        return

    all_evals = _build_all_evals(evaluations)

    available_splits = sorted(
        {e.get("eval_split", "all") for _, e in all_evals},
    )
    col_split, col_search = st.columns([1, 2])
    with col_split:
        default_split = next(
            (s for s in ("test", "val", "all") if s in available_splits),
            available_splits[0],
        )
        overview_split = st.radio(
            "Eval split",
            available_splits,
            index=available_splits.index(default_split),
            horizontal=True,
            key="overview_split_filter",
        )
    with col_search:
        search = st.text_input(
            "Filter policies", placeholder="e.g. a2c, pred, periodic",
            key="overview_search",
        ).strip().lower()

    rows = []
    for sync_unit, evals in evaluations.items():
        for e in evals:
            if e.get("eval_split", "all") != overview_split:
                continue
            if search and search not in e["policy_name"].lower():
                continue
            s = e["summary"]
            row = {
                "Sync Unit": sync_unit,
                "Policy": e["policy_name"],
                "Total Reward": f"{s.get('total_reward', 0):,.0f}",
                "Reward/Step": f"{s.get('mean_reward_per_step', 0):.4f}",
                "Generalization": e.get("generalization", "—"),
                "Sync Rate": f"{s.get('sync_frequency', 0):.1%}",
                "Empty Sync Rate": f"{s.get('empty_sync_rate', 0):.1%}",
                "Avg AoI (min)": f"{s.get('avg_aoi', 0):,.0f}",
                "Avg AoII": f"{s.get('avg_aoii', 0):,.1f}" if "avg_aoii" in s else "—",
                "SPO Regret/Step": f"{s.get('spo_regret_per_step', 0):.4f}" if "spo_regret_per_step" in s else "—",
                "Steps": f"{s.get('total_steps', 0):,}",
            }
            rows.append(row)

    if not rows:
        st.info("No policies match the current filters.")
        return
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def page_policy_detail(evaluations: dict[str, list[dict]]):
    """Detailed view for a single policy evaluation."""
    st.header("Policy Detail")

    all_evals = _build_all_evals(evaluations)
    if not all_evals:
        st.info("No evaluations found for this sync unit.")
        return

    search = st.text_input(
        "Filter policies", placeholder="e.g. a2c, pred, periodic",
        key="detail_search",
    ).strip().lower()
    filtered_evals = [
        (su, e) for su, e in all_evals
        if not search or search in e["policy_name"].lower()
    ]
    if not filtered_evals:
        st.info("No policies match the filter.")
        return

    options = [e["policy_name"] for _, e in filtered_evals]
    selected = st.selectbox("Select evaluation", options, key="detail_select")
    idx = options.index(selected)
    sync_unit, eval_data = filtered_evals[idx]

    summary = eval_data["summary"]
    metadata = eval_data["metadata"]
    rc = _get_reward_config(metadata)
    mode = metadata.get("task_type", "event_rate")

    # --- Summary metrics ---
    st.subheader("Summary Metrics")
    cols = st.columns(4)
    cols[0].metric("Total Reward", f"{summary.get('total_reward', 0):,.0f}")
    cols[1].metric("Sync Count", f"{summary.get('sync_count', 0):,}")
    cols[2].metric("Empty Sync Rate", f"{summary.get('empty_sync_rate', 0):.1%}")
    cols[3].metric("Avg AoI", f"{summary.get('avg_aoi', 0):,.0f} min")

    cols2 = st.columns(4)
    cols2[0].metric("Max AoI", f"{summary.get('max_aoi', 0):,.0f} min")
    cols2[1].metric("P95 AoI", f"{summary.get('p95_aoi', 0):,.0f} min")
    cols2[2].metric("Sync Frequency", f"{summary.get('sync_frequency', 0):.1%}")
    cols2[3].metric("Avg Items/Sync", f"{summary.get('avg_items_per_sync', 0):.2f}")

    # Timing quality metrics (present in newer evaluations)
    if summary.get("total_events_detected", 0) > 0:
        cols3 = st.columns(4)
        cols3[0].metric("Avg Reaction Time", f"{summary.get('avg_reaction_time', 0):.1f} min")
        cols3[1].metric("P95 Reaction Time", f"{summary.get('p95_reaction_time', 0):.1f} min")
        cols3[2].metric("Event Capture Rate", f"{summary.get('event_capture_rate', 0):.1%}")
        cols3[3].metric("Events Detected", f"{summary.get('total_events_detected', 0):,}")

    # Decision-quality metrics (AoII + SPO Regret)
    if "avg_aoii" in summary:
        st.subheader("Decision-Quality Metrics")
        cols4 = st.columns(4)
        cols4[0].metric("Avg AoII", f"{summary['avg_aoii']:,.1f}")
        cols4[1].metric("P95 AoII", f"{summary.get('p95_aoii', 0):,.1f}")
        cols4[2].metric("SPO Regret/Step", f"{summary.get('spo_regret_per_step', 0):.4f}")
        cols4[3].metric("Oracle Syncs", f"{summary.get('oracle_sync_count', 0):,}")

        with st.expander("How to interpret these metrics"):
            st.markdown("""
**AoII (Age of Incorrect Information)** — Staleness penalized only when
the cache is actually wrong. Unlike AoI (which grows linearly even
during calm periods), AoII = 0 whenever no events are pending at the
source. When events ARE missed, AoII = time_stale x items_pending,
growing quadratically during bursts.

- `Avg AoII = 0`: perfect — never stale when it matters
- Low AoII + high empty sync rate: over-syncing (waste but no harm)
- High AoII + low empty sync rate: under-syncing (stale data)

**SPO Regret/Step** — How far from optimal. Measures the combined cost
of staleness + wasted syncs relative to an oracle that knows exactly
when events occur (zero AoI, zero empty syncs). Lower = closer to
oracle-optimal behavior.

- `SPO Regret = 0`: policy is the oracle (impossible in practice)
- Regret dominated by AoI cost: policy is too conservative (syncs too rarely)
- Regret dominated by empty syncs: policy is too aggressive (syncs too often)
- Compare across policies: the one with lowest regret makes the best decisions

**Oracle Syncs** — Number of syncs the perfect-knowledge oracle would
use. This equals the number of event-minutes in the evaluation period.
A policy syncing much more than this is wasting resources; much less is
missing events.
""")


    # Date range and split info
    if eval_data.get("eval_start"):
        st.caption(
            f"Evaluation period: {eval_data['eval_start']} to {eval_data['eval_end']}"
        )
    if eval_data.get("eval_split", "all") != "all":
        st.caption(f"Eval split: **{eval_data['eval_split']}**")
    if eval_data.get("train_end_timestamp"):
        st.caption(f"Train/val boundary: {eval_data['train_end_timestamp']}")
    if eval_data.get("val_end_timestamp"):
        st.caption(f"Val/test boundary: {eval_data['val_end_timestamp']}")

    # --- Detected Cron Patterns ---
    cron_patterns, hourly_patterns = _load_cron_patterns_for_eval(metadata)
    if cron_patterns:
        st.subheader("Detected Daily Cron Patterns")
        cron_df = pd.DataFrame([
            {
                "Time Slot": f"{p['hour']:02d}:{p['minute']:02d}",
                "Frequency": f"{p['frequency']:.0%}",
                "Mean Magnitude": f"{p['mean_magnitude']:.2f}",
                "Std Magnitude": f"{p['std_magnitude']:.2f}",
            }
            for p in cron_patterns
        ])
        st.dataframe(cron_df, use_container_width=True, hide_index=True)
    if hourly_patterns:
        st.subheader("Detected Hourly Cron Patterns")
        hourly_df = pd.DataFrame([
            {
                "Minute of Hour": f":{p['minute']:02d}",
                "Frequency": f"{p['frequency']:.0%}",
                "Mean Magnitude": f"{p['mean_magnitude']:.2f}",
                "Std Magnitude": f"{p['std_magnitude']:.2f}",
            }
            for p in hourly_patterns
        ])
        st.dataframe(hourly_df, use_container_width=True, hide_index=True)

    # --- Oracle comparison ---
    st.subheader("vs Oracle (Perfect Foresight)")

    # Custom reward parameters with Recalculate button
    rc_state_key = f"custom_rc_{eval_data['eval_dir']}"
    with st.expander("Custom Reward Parameters"):
        orig_rc_dict = _get_reward_config(metadata).to_dict()
        stored = st.session_state.get(rc_state_key, orig_rc_dict)

        with st.form(key=f"reward_form_{eval_data['eval_dir']}"):
            c1, c2, c3 = st.columns(3)
            cust_sync_cost = c1.number_input(
                "sync_cost", value=float(stored["sync_cost"]),
                step=0.1, format="%.2f",
            )
            cust_late_penalty = c2.number_input(
                "late_penalty", value=float(stored["late_penalty"]),
                step=0.1, format="%.2f",
            )
            cust_success = c3.number_input(
                "successful_sync_reward",
                value=float(stored["successful_sync_reward"]),
                step=0.1, format="%.2f",
            )
            c4, c5, c6 = st.columns(3)
            cust_empty = c4.number_input(
                "empty_sync_penalty",
                value=float(stored["empty_sync_penalty"]),
                step=0.1, format="%.2f",
            )
            cust_aoi_pen = c5.number_input(
                "aoi_penalty_per_minute",
                value=float(stored["aoi_penalty_per_minute"]),
                step=0.001, format="%.4f",
            )
            cust_aoi_thresh = c6.number_input(
                "aoi_threshold_minutes",
                value=float(stored["aoi_threshold_minutes"]),
                step=1.0, format="%.1f",
            )
            btn1, btn2 = st.columns(2)
            recalculate = btn1.form_submit_button("Recalculate")
            reset = btn2.form_submit_button("Reset to original")

        if recalculate:
            st.session_state[rc_state_key] = {
                "sync_cost": cust_sync_cost,
                "late_penalty": cust_late_penalty,
                "successful_sync_reward": cust_success,
                "empty_sync_penalty": cust_empty,
                "aoi_penalty_per_minute": cust_aoi_pen,
                "aoi_threshold_minutes": cust_aoi_thresh,
                "reward_mode": stored.get("reward_mode", "full"),
            }
        elif reset and rc_state_key in st.session_state:
            del st.session_state[rc_state_key]
            st.rerun()

    # Apply custom reward config if set
    if rc_state_key in st.session_state:
        rc = RewardConfig.from_dict({**st.session_state[rc_state_key], "mode": mode})
        st.caption("Using custom reward parameters")

    scoring_rc_dict = rc.to_dict()

    oracle_df, oracle_summary = load_or_compute_oracle(
        eval_data["eval_dir"], scoring_rc_dict, mode,
    )

    # Re-score policy decisions with current scoring params
    rescored_rewards_arr, rescored_summary_detail = load_or_compute_rescored(
        eval_data["eval_dir"], scoring_rc_dict, mode,
    )

    if oracle_summary:
        oracle_reward = oracle_summary.get("total_reward", 0)
        policy_reward = rescored_summary_detail.get("total_reward", 0)
        pct_optimal = _pct_of_optimal(policy_reward, oracle_reward)

        oc = st.columns(4)
        oc[0].metric("Oracle Reward", f"{oracle_reward:,.0f}")
        oc[1].metric("% of Optimal", f"{pct_optimal:.1f}%")
        oc[2].metric(
            "Excess Empty Syncs",
            f"{summary.get('empty_sync_count', 0) - oracle_summary.get('empty_sync_count', 0):,}",
        )
        # True AoI: oracle is always 0 (syncs at every event), so a
        # ratio is meaningless.  Show the policy's absolute avg staleness.
        oc[3].metric("Avg Staleness", f"{summary.get('avg_aoi', 0):.2f} min")

    # --- Load decisions ---
    decisions = load_decisions(eval_data["eval_dir"])
    if decisions is None:
        st.warning("decisions.parquet not found")
        return

    ts = decisions["timestamp"].values
    total_steps = len(decisions)
    st.caption(f"Total steps: {total_steps:,}")

    # Parse split boundaries once — reused for vlines, split metrics, heatmap
    train_end_parsed = None
    val_end_parsed = None
    if eval_data.get("train_end_timestamp"):
        try:
            train_end_parsed = pd.Timestamp(eval_data["train_end_timestamp"])
        except Exception:
            pass
    if eval_data.get("val_end_timestamp"):
        try:
            val_end_parsed = pd.Timestamp(eval_data["val_end_timestamp"])
        except Exception:
            pass

    vlines = []
    if train_end_parsed is not None:
        vlines.append({"x": train_end_parsed, "label": "Train end", "color": "red"})
    if val_end_parsed is not None:
        vlines.append({"x": val_end_parsed, "label": "Val end", "color": "orange"})

    # --- Train vs Test Performance ---
    if train_end_parsed is not None:
        split_metrics = compute_per_split_metrics(
            eval_data["eval_dir"],
            eval_data["train_end_timestamp"],
            eval_data.get("val_end_timestamp"),
            task_type=mode,
        )
        if split_metrics and len(split_metrics) >= 2:
            st.subheader("Train vs Test Performance")

            # Determine which splits to show (test preferred, fall back to val)
            out_split = "test" if "test" in split_metrics else "val"
            out_label = out_split.capitalize()

            # Comparison table
            table_rows = []
            for key, label, fmt in _SPLIT_COMPARE_METRICS:
                train_val = split_metrics["train"].get(key, 0)
                out_val = split_metrics[out_split].get(key, 0)
                val_val = split_metrics.get("val", {}).get(key)

                if isinstance(train_val, (int, float)) and train_val != 0:
                    delta = out_val - train_val
                    pct = delta / abs(train_val) * 100
                    lower_better = key in _LOWER_IS_BETTER
                    improved = (delta < 0) if lower_better else (delta > 0)
                    arrow = "+" if delta > 0 else ""
                    delta_str = f"{arrow}{pct:.1f}%"
                else:
                    delta_str = "—"
                    improved = None

                row = {
                    "Metric": label,
                    "Train": f"{train_val:{fmt}}" if isinstance(train_val, (int, float)) else str(train_val),
                    out_label: f"{out_val:{fmt}}" if isinstance(out_val, (int, float)) else str(out_val),
                }
                if val_val is not None and "val" in split_metrics and out_split != "val":
                    row["Val"] = f"{val_val:{fmt}}" if isinstance(val_val, (int, float)) else str(val_val)
                row["Delta"] = delta_str
                table_rows.append(row)

            st.dataframe(
                pd.DataFrame(table_rows),
                use_container_width=True, hide_index=True,
            )

            # Generalization verdict
            train_r = split_metrics["train"].get("mean_reward_per_step", 0)
            out_r = split_metrics[out_split].get("mean_reward_per_step", 0)
            if train_r != 0:
                reward_pct = (out_r - train_r) / abs(train_r) * 100
                if reward_pct >= 0:
                    st.success(
                        f"Performance improves by {reward_pct:.1f}% "
                        f"on {out_label.lower()} data"
                    )
                elif reward_pct > -10:
                    st.info(
                        f"Performance degrades by {abs(reward_pct):.1f}% "
                        f"on unseen data (within tolerance)"
                    )
                else:
                    st.warning(
                        f"Performance degrades by {abs(reward_pct):.1f}% "
                        f"on unseen data"
                    )

    # Actual events
    st.subheader("Actual Events Over Time")
    ev_c1, ev_c2, ev_c3 = st.columns([1, 2, 2])
    log_scale_events = ev_c1.checkbox("Log scale", value=False, key="log_scale_events")
    show_syncs = ev_c2.checkbox("Show sync decisions", value=False, key="show_sync_markers")
    show_problems_on_events = ev_c3.checkbox("Show problem periods", value=False, key="show_problems_events")

    sync_markers = []
    if show_syncs:
        sync_mask = decisions["action"] == 1
        sync_rows = decisions[sync_mask]
        if len(sync_rows) > 0:
            # A sync is meaningful if it captured accumulated events since
            # the previous sync (last_sync_items > 0), not just whether
            # an event happened at that exact minute.
            meaningful_mask = sync_rows["last_sync_items"] > 0
            empty_mask = ~meaningful_mask

            if meaningful_mask.any():
                sync_markers.append({
                    "name": "Sync (found events)",
                    "timestamps": sync_rows.loc[meaningful_mask, "timestamp"].values,
                    "values": sync_rows.loc[meaningful_mask, "actual_value"].values,
                    "color": "#2ca02c",
                    "symbol": "triangle-up",
                    "size": 7,
                    "opacity": 0.8,
                })
            if empty_mask.any():
                empty_vals = sync_rows.loc[empty_mask, "actual_value"].values
                if log_scale_events:
                    empty_vals = np.full_like(empty_vals, 0.1)
                sync_markers.append({
                    "name": "Sync (empty)",
                    "timestamps": sync_rows.loc[empty_mask, "timestamp"].values,
                    "values": empty_vals,
                    "color": "#d62728",
                    "symbol": "x",
                    "size": 6,
                    "opacity": 0.6,
                })

    fig_events = create_timeline_chart(
        ts, decisions["actual_value"].values,
        "Actual Events", y_label="Events/min", vlines=vlines,
        markers=sync_markers if sync_markers else None,
        skip_downsample=log_scale_events,
    )
    if log_scale_events:
        fig_events.update_yaxes(type="log")
    if cron_patterns:
        _add_cron_vrects(fig_events, ts, cron_patterns)
    if hourly_patterns:
        _add_hourly_cron_vrects(fig_events, ts, hourly_patterns)
    problem_periods = summary.get("problem_periods", [])
    if show_problems_on_events:
        _add_problem_period_vrects(fig_events, problem_periods)
    st.plotly_chart(fig_events, use_container_width=True)

    sync_count = (decisions["action"] == 1).sum()
    st.caption(f"Sync points: {sync_count:,} / {total_steps:,} ({sync_count/total_steps:.1%})")

    # AoI with oracle overlay — use true_aoi (real staleness) when
    # available, fall back to current_aoi for older artifacts.
    aoi_col = "true_aoi" if "true_aoi" in decisions.columns else "current_aoi"
    oracle_aoi_col = "true_aoi" if (not oracle_df.empty and "true_aoi" in oracle_df.columns) else "current_aoi"
    st.subheader("Age of Information Over Time")
    oracle_traces = {}
    if not oracle_df.empty:
        oracle_traces["Oracle AoI"] = (
            oracle_df["timestamp"].values,
            oracle_df[oracle_aoi_col].values,
        )

    fig_aoi = create_timeline_chart(
        ts, decisions[aoi_col].values,
        "Policy AoI", y_label="AoI (min)",
        traces=oracle_traces, vlines=vlines,
    )
    _add_problem_period_vrects(fig_aoi, problem_periods)
    st.plotly_chart(fig_aoi, use_container_width=True)

    # Cumulative reward
    st.subheader("Cumulative Reward")
    if len(rescored_rewards_arr) == len(ts):
        cum_reward = np.cumsum(rescored_rewards_arr)
    else:
        cum_reward = np.cumsum(decisions["reward"].values)
    oracle_reward_traces = {}
    if not oracle_df.empty:
        oracle_reward_traces["Oracle Reward"] = (
            oracle_df["timestamp"].values,
            np.cumsum(oracle_df["reward"].values),
        )
    fig_reward = create_timeline_chart(
        ts, cum_reward,
        "Cumulative Reward", y_label="Reward",
        traces=oracle_reward_traces, vlines=vlines,
    )
    _add_problem_period_vrects(fig_reward, problem_periods)
    st.plotly_chart(fig_reward, use_container_width=True)

    # Cumulative empty syncs
    st.subheader("Cumulative Empty Syncs")
    fig_empty = create_timeline_chart(
        ts, decisions["cumulative_empty_syncs"].values,
        "Cumulative Empty Syncs", y_label="Count", vlines=vlines,
    )
    st.plotly_chart(fig_empty, use_container_width=True)

    # Problem periods summary
    if problem_periods:
        st.subheader("Problem Periods")
        total_problem_min = sum(p["duration_minutes"] for p in problem_periods)
        pp_cols = st.columns(3)
        pp_cols[0].metric("Problem Periods", len(problem_periods))
        pp_cols[1].metric("Total Problem Duration", f"{total_problem_min:,} min")
        pp_cols[2].metric(
            "% of Eval", f"{total_problem_min / total_steps:.1%}"
        )
        pp_rows = []
        for p in problem_periods:
            pp_rows.append({
                "Start": p.get("start_timestamp", p["start_idx"]),
                "End": p.get("end_timestamp", p["end_idx"]),
                "Duration (min)": p["duration_minutes"],
                "Triggers": ", ".join(p["triggers"]),
                "Severity": ", ".join(
                    f"{k}: {v:.2f}" for k, v in p["severity"].items()
                ),
            })
        st.dataframe(
            pd.DataFrame(pp_rows), use_container_width=True, hide_index=True,
        )

        with st.expander("Problem Period Deep Analysis", expanded=False):
            attributions = load_or_compute_attributions(
                eval_data["eval_dir"], decisions, problem_periods,
            )

            from collections import Counter
            class_counts = Counter(a["classification"] for a in attributions)
            cc = st.columns(4)
            cc[0].metric("Data Droughts", class_counts.get("data_drought", 0))
            cc[1].metric("Missed Bursts", class_counts.get("missed_burst", 0))
            cc[2].metric("Over-syncing", class_counts.get("over_syncing", 0))
            cc[3].metric("Mixed", class_counts.get("mixed", 0))

            attr_rows = []
            for p, attr in zip(problem_periods, attributions):
                attr_rows.append({
                    "Start": p.get("start_timestamp", p["start_idx"]),
                    "Duration (min)": p["duration_minutes"],
                    "Classification": attr["classification"],
                    "Event Density": f"{attr['event_density']:.2%}",
                    "Sync Rate": f"{attr['sync_rate']:.2%}",
                    "Empty Sync %": f"{attr['empty_sync_rate']:.0%}",
                    "Silent %": f"{attr['silent_fraction']:.0%}",
                })
            sorted_rows = sorted(attr_rows, key=lambda r: r["Duration (min)"], reverse=True)
            st.dataframe(
                pd.DataFrame(sorted_rows[:20]),
                use_container_width=True, hide_index=True,
            )
            if len(sorted_rows) > 20:
                with st.expander(f"Show all {len(sorted_rows)} periods"):
                    st.dataframe(
                        pd.DataFrame(sorted_rows),
                        use_container_width=True, hide_index=True,
                    )

            st.markdown("**Problem Period Concentration by Time**")
            concentration = compute_problem_temporal_concentration(
                decisions, problem_periods,
            )
            fig_conc = create_performance_heatmap(concentration, "problem_rate")
            st.plotly_chart(fig_conc, use_container_width=True)

    # Temporal performance heatmap
    st.subheader("Temporal Performance Patterns")
    hm_c1, hm_c2 = st.columns([2, 3])
    heatmap_metric = hm_c1.selectbox(
        "Heatmap metric",
        ["true_aoi", "reward", "sync_rate", "empty_rate"],
        index=0,
        key=f"heatmap_metric_{eval_data['eval_dir']}",
    )
    heatmap_split = "All"
    if train_end_parsed is not None:
        split_opts = ["All", "Train"]
        if val_end_parsed is not None:
            split_opts.append("Val")
        split_opts.append("Test")
        heatmap_split = hm_c2.radio(
            "Data split", split_opts, horizontal=True,
            key=f"heatmap_split_{eval_data['eval_dir']}",
        )

    if heatmap_split == "Train" and train_end_parsed is not None:
        heatmap_df = decisions[decisions["timestamp"] <= train_end_parsed]
    elif heatmap_split == "Val" and train_end_parsed is not None and val_end_parsed is not None:
        heatmap_df = decisions[
            (decisions["timestamp"] > train_end_parsed)
            & (decisions["timestamp"] <= val_end_parsed)
        ]
    elif heatmap_split == "Test" and train_end_parsed is not None:
        boundary = val_end_parsed if val_end_parsed is not None else train_end_parsed
        heatmap_df = decisions[decisions["timestamp"] > boundary]
    else:
        heatmap_df = decisions

    if not heatmap_df.empty:
        heatmap_data = compute_temporal_heatmap_data(heatmap_df, heatmap_metric)
        fig_heatmap = create_performance_heatmap(
            heatmap_data, heatmap_metric, cron_patterns=cron_patterns,
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("No data for the selected split.")

    # Items per sync histogram
    st.subheader("Items per Sync Distribution")
    sync_mask = decisions["action"] == 1
    if sync_mask.any():
        items = decisions.loc[sync_mask, "last_sync_items"].values
        fig_hist = go.Figure(data=[go.Histogram(
            x=items, nbinsx=50, name="Items per Sync",
        )])
        fig_hist.update_layout(
            xaxis_title="Items", yaxis_title="Count",
            height=300, margin=dict(l=50, r=20, t=30, b=30),
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        zero_pct = (items == 0).sum() / len(items) * 100
        st.caption(f"Empty syncs (0 items): {zero_pct:.1f}% of all syncs")

    # Action distribution
    st.subheader("Action Distribution")
    action_counts = decisions["action"].value_counts()
    action_labels = {0: "Skip", 1: "Sync"}
    action_df = pd.DataFrame({
        "Action": [action_labels.get(k, str(k)) for k in action_counts.index],
        "Count": action_counts.values,
    })
    st.bar_chart(action_df.set_index("Action"), use_container_width=True)

    # Metadata
    with st.expander("Policy Metadata"):
        st.json(eval_data["metadata"])


def page_compare(evaluations: dict[str, list[dict]]):
    """Compare multiple policies side-by-side."""
    st.header("Policy Comparison")

    all_evals = _build_all_evals(evaluations)
    if len(all_evals) < 2:
        st.info("Need at least 2 evaluations for this sync unit to compare.")
        return

    available_splits = sorted(
        {e.get("eval_split", "all") for _, e in all_evals},
    )
    default_split = next(
        (s for s in ("test", "val", "all") if s in available_splits),
        available_splits[0],
    )
    split_filter = st.radio(
        "Eval split",
        available_splits,
        index=available_splits.index(default_split),
        horizontal=True,
        key="compare_split_filter",
    )
    compare_search = st.text_input(
        "Filter policies", placeholder="e.g. a2c, pred, periodic",
        key="compare_search",
    ).strip().lower()
    filtered_evals = [
        (su, e) for su, e in all_evals
        if e.get("eval_split", "all") == split_filter
        and (not compare_search or compare_search in e["policy_name"].lower())
    ]
    if len(filtered_evals) < 2:
        st.info(f"Fewer than 2 evaluations matching filters.")
        return

    options = [e["policy_name"] for _, e in filtered_evals]
    selected = st.multiselect(
        "Select policies to compare",
        options,
        default=options[:min(3, len(options))],
    )

    if len(selected) < 2:
        st.warning("Select at least 2 policies.")
        return

    selected_evals = [filtered_evals[options.index(s)] for s in selected]

    oracle_is_valid = True

    first_su, first_e = selected_evals[0]
    first_meta = first_e["metadata"]
    first_rc = _get_reward_config(first_meta)
    first_mode = first_meta.get("task_type", "event_rate")

    # Custom reward parameters with Recalculate button
    compare_rc_key = "compare_custom_rc"
    with st.expander("Custom Reward Parameters"):
        orig_compare_rc = first_rc.to_dict()
        stored_cmp = st.session_state.get(compare_rc_key, orig_compare_rc)

        with st.form(key="compare_reward_form"):
            c1, c2, c3 = st.columns(3)
            cmp_sync_cost = c1.number_input(
                "sync_cost", value=float(stored_cmp["sync_cost"]),
                step=0.1, format="%.2f", key="cmp_sync_cost",
            )
            cmp_late_penalty = c2.number_input(
                "late_penalty", value=float(stored_cmp["late_penalty"]),
                step=0.1, format="%.2f", key="cmp_late_penalty",
            )
            cmp_success = c3.number_input(
                "successful_sync_reward",
                value=float(stored_cmp["successful_sync_reward"]),
                step=0.1, format="%.2f", key="cmp_success",
            )
            c4, c5, c6 = st.columns(3)
            cmp_empty = c4.number_input(
                "empty_sync_penalty",
                value=float(stored_cmp["empty_sync_penalty"]),
                step=0.1, format="%.2f", key="cmp_empty",
            )
            cmp_aoi_pen = c5.number_input(
                "aoi_penalty_per_minute",
                value=float(stored_cmp["aoi_penalty_per_minute"]),
                step=0.001, format="%.4f", key="cmp_aoi_pen",
            )
            cmp_aoi_thresh = c6.number_input(
                "aoi_threshold_minutes",
                value=float(stored_cmp["aoi_threshold_minutes"]),
                step=1.0, format="%.1f", key="cmp_aoi_thresh",
            )
            btn1, btn2 = st.columns(2)
            cmp_recalc = btn1.form_submit_button("Recalculate")
            cmp_reset = btn2.form_submit_button("Reset to original")

        if cmp_recalc:
            st.session_state[compare_rc_key] = {
                "sync_cost": cmp_sync_cost,
                "late_penalty": cmp_late_penalty,
                "successful_sync_reward": cmp_success,
                "empty_sync_penalty": cmp_empty,
                "aoi_penalty_per_minute": cmp_aoi_pen,
                "aoi_threshold_minutes": cmp_aoi_thresh,
                "reward_mode": stored_cmp.get("reward_mode", "full"),
            }
        elif cmp_reset and compare_rc_key in st.session_state:
            del st.session_state[compare_rc_key]
            st.rerun()

    # Apply custom reward config if set
    if compare_rc_key in st.session_state:
        first_rc = RewardConfig.from_dict(
            {**st.session_state[compare_rc_key], "mode": first_mode},
        )
        st.caption("Using custom reward parameters")

    compare_rc_dict = first_rc.to_dict()

    oracle_df_compare = pd.DataFrame()
    oracle_summary = None
    if oracle_is_valid:
        oracle_df_compare, oracle_summary = load_or_compute_oracle(
            first_e["eval_dir"], compare_rc_dict, first_mode,
        )

    # Comparison table
    metrics_keys = [
        "total_reward", "mean_reward_per_step", "sync_count",
        "sync_frequency", "empty_sync_count", "empty_sync_rate",
        "avg_aoi", "max_aoi", "p95_aoi", "median_aoi",
        "avg_aoii", "p95_aoii", "spo_regret_per_step", "oracle_sync_count",
        "avg_items_per_sync",
        "avg_reaction_time", "event_capture_rate",
    ]

    oracle_label = "Oracle"
    rows = {}
    if oracle_summary:
        rows[oracle_label] = {
            k: oracle_summary.get(k, 0) for k in metrics_keys
        }
    use_custom_scoring = compare_rc_key in st.session_state
    gen_values = {}
    for su, e in selected_evals:
        name = e["policy_name"]
        if use_custom_scoring:
            _, rescored_s = load_or_compute_rescored(
                e["eval_dir"], compare_rc_dict, first_mode,
            )
            rows[name] = {k: rescored_s.get(k, 0) for k in metrics_keys}
        else:
            s = e["summary"]
            rows[name] = {k: s.get(k, 0) for k in metrics_keys}

        gen_values[name] = e.get("generalization", "—")

    comparison_df = pd.DataFrame(rows).T
    comparison_df.index.name = "Policy"

    if oracle_summary and "total_reward" in comparison_df.columns:
        oracle_reward = oracle_summary.get("total_reward", 0)
        comparison_df.insert(
            1, "% of Optimal",
            comparison_df["total_reward"].apply(
                lambda r: f"{_pct_of_optimal(r, oracle_reward):.1f}%"
                if isinstance(r, (int, float)) else "—"
            ),
        )

    gen_col_pos = 3 if "% of Optimal" in comparison_df.columns else 2
    comparison_df.insert(
        gen_col_pos, "Generalization",
        [gen_values.get(idx, "—") for idx in comparison_df.index],
    )

    for col in ["sync_frequency", "empty_sync_rate", "event_capture_rate"]:
        if col in comparison_df.columns:
            comparison_df[col] = comparison_df[col].apply(
                lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else x
            )

    st.dataframe(comparison_df, use_container_width=True)

    # Chart sections — guarded behind checkbox to avoid loading
    # multi-million-row parquets until explicitly requested.
    show_charts = st.checkbox("Show comparison charts", value=False, key="compare_show_charts")

    if show_charts:
        with st.expander("AoI Comparison Charts", expanded=True):
            aoi_series = {}
            if oracle_is_valid and not oracle_df_compare.empty:
                o_col = "true_aoi" if "true_aoi" in oracle_df_compare.columns else "current_aoi"
                aoi_series[oracle_label] = (
                    oracle_df_compare["timestamp"].values,
                    oracle_df_compare[o_col].values,
                )
            for su, e in selected_evals:
                decisions = load_decisions(e["eval_dir"])
                if decisions is not None:
                    col = "true_aoi" if "true_aoi" in decisions.columns else "current_aoi"
                    aoi_series[e["policy_name"]] = (
                        decisions["timestamp"].values,
                        decisions[col].values,
                    )
            if aoi_series:
                fig = create_multi_trace_chart(aoi_series, "AoI Comparison", "AoI (min)")
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("Cumulative Reward Comparison Charts", expanded=True):
            reward_series = {}
            if oracle_is_valid and not oracle_df_compare.empty:
                reward_series[oracle_label] = (
                    oracle_df_compare["timestamp"].values,
                    np.cumsum(oracle_df_compare["reward"].values),
                )
            for su, e in selected_evals:
                decisions = load_decisions(e["eval_dir"])
                if decisions is not None:
                    if use_custom_scoring:
                        rescored_r, _ = load_or_compute_rescored(
                            e["eval_dir"], compare_rc_dict, first_mode,
                        )
                        rewards = rescored_r if len(rescored_r) == len(decisions) else decisions["reward"].values
                    else:
                        rewards = decisions["reward"].values
                    reward_series[e["policy_name"]] = (
                        decisions["timestamp"].values,
                        np.cumsum(rewards),
                    )
            if reward_series:
                fig = create_multi_trace_chart(
                    reward_series, "Cumulative Reward Comparison", "Reward",
                )
                st.plotly_chart(fig, use_container_width=True)

    # Per-split comparison (only for policies with split boundaries)
    evals_with_splits = [
        (su, e) for su, e in selected_evals
        if e.get("train_end_timestamp")
    ]
    if evals_with_splits:
        with st.expander("Per-Split Performance Breakdown", expanded=False):
            all_splits: dict[str, dict[str, dict]] = {}
            for su, e in evals_with_splits:
                sm = compute_per_split_metrics(
                    e["eval_dir"],
                    e["train_end_timestamp"],
                    e.get("val_end_timestamp"),
                    task_type=first_mode,
                )
                if sm:
                    all_splits[e["policy_name"]] = sm

            if not all_splits:
                st.info("No split data available for selected policies.")
            else:
                common_splits = set.intersection(*(
                    set(s.keys()) for s in all_splits.values()
                ))
                compare_keys = [
                    ("mean_reward_per_step", "Mean Reward/Step", ".4f"),
                    ("avg_aoi", "Avg AoI", ",.1f"),
                    ("p95_aoi", "P95 AoI", ",.1f"),
                    ("sync_frequency", "Sync Freq", ".2%"),
                    ("empty_sync_rate", "Empty Rate", ".2%"),
                    ("event_capture_rate", "Capture Rate", ".1%"),
                ]

                split_tabs = st.tabs([s.capitalize() for s in sorted(common_splits)])
                for tab, split_name in zip(split_tabs, sorted(common_splits)):
                    with tab:
                        rows = []
                        for key, label, fmt in compare_keys:
                            row: dict[str, Any] = {"Metric": label}
                            for pol_name, pol_splits in all_splits.items():
                                val = pol_splits.get(split_name, {}).get(key, 0)
                                row[pol_name] = (
                                    f"{val:{fmt}}"
                                    if isinstance(val, (int, float))
                                    else str(val)
                                )
                            rows.append(row)
                        st.dataframe(
                            pd.DataFrame(rows),
                            use_container_width=True, hide_index=True,
                        )

    # Problem period overlap analysis
    with st.expander("Problem Period Overlap Analysis", expanded=False):
        pp_masks: dict[str, np.ndarray] = {}
        pp_timestamps = None
        for su, e in selected_evals:
            dec = load_decisions(e["eval_dir"])
            if dec is None:
                continue
            pp_list = e["summary"].get("problem_periods", [])
            if not pp_list:
                pp_list = detect_problem_periods(dec)
            mask = build_problem_mask(dec, pp_list)
            pp_masks[e["policy_name"]] = mask
            if pp_timestamps is None:
                pp_timestamps = dec["timestamp"].values

        if len(pp_masks) < 2:
            st.info("Need at least 2 policies with decision data for overlap analysis.")
        else:
            min_len = min(len(m) for m in pp_masks.values())
            pp_masks = {k: v[:min_len] for k, v in pp_masks.items()}
            if pp_timestamps is not None:
                pp_timestamps = pp_timestamps[:min_len]

            overlap = compute_problem_overlap(pp_masks)

            any_problem = np.any(
                np.stack(list(pp_masks.values())), axis=0,
            )
            oc = st.columns(3)
            oc[0].metric(
                "Shared Problem Minutes",
                f"{int(overlap['shared_mask'].sum()):,}",
            )
            oc[1].metric(
                "% Shared (of any problem)",
                f"{overlap['shared_mask'].sum() / max(int(any_problem.sum()), 1):.1%}",
            )
            upper_tri = overlap["jaccard_matrix"].values[
                np.triu_indices(len(pp_masks), k=1)
            ]
            oc[2].metric("Avg Jaccard", f"{upper_tri.mean():.3f}")

            st.markdown("**Pairwise Problem Period Overlap (Jaccard)**")
            fig_jaccard = create_jaccard_heatmap(overlap["jaccard_matrix"])
            st.plotly_chart(fig_jaccard, use_container_width=True)

            if pp_timestamps is not None:
                st.markdown("**Problem Period Timeline**")
                fig_timeline = create_overlap_timeline_figure(
                    pp_timestamps, pp_masks, overlap["shared_mask"],
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

            st.markdown("**Policy-Specific Problems** (not shared with majority)")
            spec_rows = []
            for name, spec_mask in overlap["policy_specific"].items():
                total_problems = int(pp_masks[name].sum())
                specific = int(spec_mask.sum())
                spec_rows.append({
                    "Policy": name,
                    "Total Problem Min": f"{total_problems:,}",
                    "Specific Problem Min": f"{specific:,}",
                    "% Unique": f"{specific / max(total_problems, 1):.1%}",
                })
            st.dataframe(
                pd.DataFrame(spec_rows),
                use_container_width=True, hide_index=True,
            )


def page_reward_experiment(evaluations: dict[str, list[dict]]):
    """Reward parameter experimentation — re-score existing decisions."""
    st.header("Reward Experimentation")
    st.caption(
        "Re-score existing policy decisions with different reward weights. "
        "Actions and AoI are fixed — only reward scoring changes."
    )

    all_evals = _build_all_evals(evaluations)
    if not all_evals:
        st.info("No evaluations found for this sync unit.")
        return

    rescore_search = st.text_input(
        "Filter policies", placeholder="e.g. a2c, pred, periodic",
        key="rescore_search",
    ).strip().lower()
    rescore_filtered = [
        (su, e) for su, e in all_evals
        if not rescore_search or rescore_search in e["policy_name"].lower()
    ]
    if not rescore_filtered:
        st.info("No policies match the filter.")
        return

    options = [e["policy_name"] for _, e in rescore_filtered]
    selected = st.selectbox("Select evaluation", options, key="rescore_select")
    idx = options.index(selected)
    sync_unit, eval_data = rescore_filtered[idx]

    metadata = eval_data["metadata"]
    mode = metadata.get("task_type", "event_rate")

    # Custom reward parameters with Recalculate button
    original_rc = metadata.get("reward_config", {})
    rescore_rc_key = f"rescore_rc_{eval_data['eval_dir']}"
    with st.expander("Reward Parameters"):
        stored_rescore = st.session_state.get(rescore_rc_key, original_rc)

        with st.form(key=f"rescore_form_{eval_data['eval_dir']}"):
            st.caption("Original values shown as defaults. Change and click Recalculate.")
            c1, c2, c3 = st.columns(3)
            rs_sync_cost = c1.number_input(
                "sync_cost",
                value=float(stored_rescore.get("sync_cost", 1.0)),
                step=0.1, format="%.2f", key="rs_sync_cost",
            )
            rs_late_penalty = c2.number_input(
                "late_penalty",
                value=float(stored_rescore.get("late_penalty", 2.0)),
                step=0.1, format="%.2f", key="rs_late_penalty",
            )
            rs_success = c3.number_input(
                "successful_sync_reward",
                value=float(stored_rescore.get("successful_sync_reward", 0.5)),
                step=0.1, format="%.2f", key="rs_success",
            )
            c4, c5, c6 = st.columns(3)
            rs_empty = c4.number_input(
                "empty_sync_penalty",
                value=float(stored_rescore.get("empty_sync_penalty", 1.0)),
                step=0.1, format="%.2f", key="rs_empty",
            )
            rs_aoi_pen = c5.number_input(
                "aoi_penalty_per_minute",
                value=float(stored_rescore.get("aoi_penalty_per_minute", 0.01)),
                step=0.001, format="%.4f", key="rs_aoi_pen",
            )
            rs_aoi_thresh = c6.number_input(
                "aoi_threshold_minutes",
                value=float(stored_rescore.get("aoi_threshold_minutes", 5.0)),
                step=1.0, format="%.1f", key="rs_aoi_thresh",
            )
            btn1, btn2 = st.columns(2)
            rs_recalc = btn1.form_submit_button("Recalculate")
            rs_reset = btn2.form_submit_button("Reset to original")

        if rs_recalc:
            st.session_state[rescore_rc_key] = {
                "sync_cost": rs_sync_cost,
                "late_penalty": rs_late_penalty,
                "successful_sync_reward": rs_success,
                "empty_sync_penalty": rs_empty,
                "aoi_penalty_per_minute": rs_aoi_pen,
                "aoi_threshold_minutes": rs_aoi_thresh,
                "reward_mode": stored_rescore.get("reward_mode", "full"),
            }
        elif rs_reset and rescore_rc_key in st.session_state:
            del st.session_state[rescore_rc_key]
            st.rerun()

    custom_rc = st.session_state.get(rescore_rc_key, original_rc).copy()
    custom_rc["mode"] = mode
    if rescore_rc_key in st.session_state:
        st.caption("Using custom reward parameters")

    # Load decisions
    decisions = load_decisions(eval_data["eval_dir"])
    if decisions is None:
        st.warning("decisions.parquet not found")
        return

    ts = decisions["timestamp"].values
    original_rewards = decisions["reward"].values

    # Re-score with custom params
    rescored_rewards, rescored_summary = load_or_compute_rescored(
        eval_data["eval_dir"], custom_rc, mode,
    )

    # Oracle with custom params
    _, oracle_custom = load_or_compute_oracle(
        eval_data["eval_dir"], custom_rc, mode,
    )

    # --- Summary comparison ---
    st.subheader("Metrics Comparison")
    original_summary = eval_data["summary"]

    metric_names = [
        "total_reward", "mean_reward_per_step", "sync_count",
        "sync_frequency", "empty_sync_count", "empty_sync_rate",
        "avg_aoi", "max_aoi", "p95_aoi",
        "avg_reaction_time", "event_capture_rate",
    ]

    comp_rows = {"Original": {}, "Re-scored": {}, "Oracle (custom)": {}}
    for k in metric_names:
        comp_rows["Original"][k] = original_summary.get(k, 0)
        comp_rows["Re-scored"][k] = rescored_summary.get(k, 0)
        if oracle_custom:
            comp_rows["Oracle (custom)"][k] = oracle_custom.get(k, 0)

    if not oracle_custom:
        del comp_rows["Oracle (custom)"]

    comp_df = pd.DataFrame(comp_rows).T
    comp_df.index.name = "Variant"

    for col in ["sync_frequency", "empty_sync_rate"]:
        if col in comp_df.columns:
            comp_df[col] = comp_df[col].apply(
                lambda x: f"{x:.2%}" if isinstance(x, float) else x
            )

    st.dataframe(comp_df, use_container_width=True)

    # % of optimal with new params
    if oracle_custom:
        oracle_r = oracle_custom.get("total_reward", 0)
        rescored_r = rescored_summary.get("total_reward", 0)
        pct = _pct_of_optimal(rescored_r, oracle_r)
        st.metric("% of Optimal (custom params)", f"{pct:.1f}%")

    # --- Cumulative reward comparison ---
    st.subheader("Cumulative Reward: Original vs Re-scored")
    if len(rescored_rewards) == len(ts):
        fig = go.Figure()

        orig_cum = np.cumsum(original_rewards)
        rescore_cum = np.cumsum(rescored_rewards)

        ep_orig, v_orig = lttb_downsample(ts.astype(np.int64), orig_cum)
        ep_rescore, v_rescore = lttb_downsample(ts.astype(np.int64), rescore_cum)

        fig.add_trace(go.Scattergl(
            x=pd.to_datetime(ep_orig), y=v_orig,
            mode="lines", name="Original",
        ))
        fig.add_trace(go.Scattergl(
            x=pd.to_datetime(ep_rescore), y=v_rescore,
            mode="lines", name="Re-scored",
        ))

        fig.update_layout(
            xaxis=dict(rangeslider=dict(visible=True), type="date"),
            height=400,
            margin=dict(l=50, r=20, t=30, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Per-step reward delta ---
    st.subheader("Per-Step Reward Delta (Re-scored - Original)")
    if len(rescored_rewards) == len(ts):
        delta = rescored_rewards - original_rewards
        fig_delta = create_timeline_chart(
            ts, delta, "Reward Delta", y_label="Delta",
        )
        st.plotly_chart(fig_delta, use_container_width=True)


# Main


def main():
    st.set_page_config(
        page_title="Sync Policy Dashboard",
        page_icon="",
        layout="wide",
    )
    st.title("Sync Policy Evaluation Dashboard")

    all_evaluations = discover_evaluations()
    all_evaluations = sidebar_date_filter(all_evaluations)

    if not all_evaluations:
        st.info("No evaluation artifacts found. Run `evaluate_sync_policy` first.")
        return

    # Sync unit selector in sidebar — scopes all tabs
    sync_units = sorted(all_evaluations.keys())
    st.sidebar.header("Sync Unit")
    selected_su = st.sidebar.selectbox(
        "Select sync unit", sync_units, key="sync_unit_select",
    )
    evaluations = {selected_su: all_evaluations[selected_su]}

    tab_overview, tab_detail, tab_compare, tab_rescore = st.tabs([
        "Overview", "Policy Detail", "Compare Policies", "Reward Experimentation",
    ])

    with tab_overview:
        page_overview(evaluations)

    with tab_detail:
        page_policy_detail(evaluations)

    with tab_compare:
        page_compare(evaluations)

    with tab_rescore:
        page_reward_experiment(evaluations)


if __name__ == "__main__":
    main()
