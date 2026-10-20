#!/usr/bin/env python3
"""Compare observable-stream metrics under alternative timestamp cutoffs."""

# pylint: disable=too-many-locals

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.cron_pattern_detector import (  # pylint: disable=import-error
    detect_cron_patterns,
    detect_hourly_patterns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTER_PATH = PROJECT_ROOT / "data" / "sync_units_register.json"
ARTIFACT_NAME = "merged_events_per_minute_1.pkl"
CLEAN_ARTIFACT_NAME = "cleaned_items_1.pkl"
REGIME_THRESHOLDS = (
    (200.0, "very-dense"),
    (100.0, "dense"),
    (20.0, "moderate"),
    (5.0, "sparse"),
)
STANDARD_WINDOWS = (
    ("full", None),
    ("post_2021", "2021-01-01T00:00:00Z"),
    ("post_2022", "2022-01-01T00:00:00Z"),
    ("post_2023", "2023-01-01T00:00:00Z"),
)
MODELED_STARTS = {
    "novacore_zendesk_novacore_zendesk": "2022-01-01T00:00:00Z",
}


def classify_regime(minute_density_per_day: float) -> str:
    """Apply the canonical frozen minute-density taxonomy."""
    for threshold, label in REGIME_THRESHOLDS:
        if minute_density_per_day >= threshold:
            return label
    return "very-sparse"


def _utc_timestamp(value: str) -> pd.Timestamp:
    """Normalize a naive or aware timestamp to UTC."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _filter_event_minutes(
    events_per_minute: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Validate, normalize, and filter an active-minute artifact."""
    required = {"minute", "events_per_minute"}
    missing = required.difference(events_per_minute.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = events_per_minute.loc[:, ["minute", "events_per_minute"]].copy()
    frame["minute"] = pd.to_datetime(frame["minute"], utc=True)
    frame = frame.dropna(subset=["minute", "events_per_minute"])
    if start is not None:
        frame = frame[frame["minute"] >= _utc_timestamp(start)]
    if end is not None:
        frame = frame[frame["minute"] <= _utc_timestamp(end)]
    if frame.empty:
        raise ValueError("No event-minutes remain in the requested window")
    if not frame["minute"].is_monotonic_increasing:
        order = np.argsort(frame["minute"].astype("int64").to_numpy(), kind="stable")
        frame = frame.iloc[order]
    return frame

def compute_window_metrics(
    events_per_minute: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Compute canonical minute-basis metrics after filtering one window."""
    frame = _filter_event_minutes(events_per_minute, start, end)

    timestamps = frame["minute"]
    span_minutes = max(
        (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 60.0,
        1.0,
    )
    span_days = max(span_minutes / 1440.0, 1.0)
    event_minutes = len(frame)
    observed_events = float(frame["events_per_minute"].sum())
    coverage = min(event_minutes / span_minutes, 1.0)

    deltas = timestamps.diff().dropna().dt.total_seconds() / 60.0
    by_minute = frame.groupby(timestamps.dt.minute)["events_per_minute"].sum()
    top_minutes = by_minute.sort_values(ascending=False).head(3)

    return {
        "effective_start": timestamps.iloc[0].isoformat(),
        "effective_end": timestamps.iloc[-1].isoformat(),
        "span_days": span_days,
        "observed_events": int(observed_events),
        "event_minutes": event_minutes,
        "minute_density_per_day": event_minutes / span_days,
        "coverage_ratio": coverage,
        "zero_fraction": 1.0 - coverage,
        "mean_iat_minutes": float(deltas.mean()) if not deltas.empty else 0.0,
        "median_iat_minutes": float(deltas.median()) if not deltas.empty else 0.0,
        "events_per_active_minute": observed_events / event_minutes,
        "multi_event_minute_fraction": float(
            (frame["events_per_minute"] > 1).mean()
        ),
        "peak_minute": int(top_minutes.index[0]),
        "peak_minute_share": float(top_minutes.iloc[0] / observed_events),
        "top3_minutes": ",".join(f"{int(value):02d}" for value in top_minutes.index),
        "top3_minute_share": float(top_minutes.sum() / observed_events),
        "regime": classify_regime(event_minutes / span_days),
    }


def compute_split_coverage(split: dict, cutoff: str) -> dict:
    """Measure minute-grid overlap at or after a connection cutoff by split."""
    minute = pd.Timedelta(minutes=1)
    cutoff_minute = _utc_timestamp(cutoff).ceil("min")
    boundaries = {
        "train": (
            _utc_timestamp(split["data_start"]),
            _utc_timestamp(split["train_end"]),
        ),
        "validation": (
            _utc_timestamp(split["train_end"]) + minute,
            _utc_timestamp(split["val_end"]),
        ),
        "test": (
            _utc_timestamp(split["val_end"]) + minute,
            _utc_timestamp(split["data_end"]),
        ),
    }
    result = {}
    for split_name, (start, end) in boundaries.items():
        total = max(int((end - start) / minute) + 1, 0)
        overlap_start = max(start, cutoff_minute)
        post_cutoff = max(int((end - overlap_start) / minute) + 1, 0)
        ratio = post_cutoff / total if total else 0.0
        result[split_name] = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "total_minutes": total,
            "post_cutoff_minutes": post_cutoff,
            "post_cutoff_ratio": ratio,
            "classification": (
                "none" if post_cutoff == 0 else "full" if post_cutoff == total else "partial"
            ),
        }
    return result


def compute_pattern_metrics(
    events_per_minute: pd.DataFrame,
    start: str,
) -> dict:
    """Run production-default cron detectors on a regular post-cutoff grid."""
    frame = _filter_event_minutes(events_per_minute, start)
    values = frame.groupby("minute")["events_per_minute"].sum()
    grid = pd.date_range(values.index.min(), values.index.max(), freq="1min")
    regular = values.reindex(grid, fill_value=0.0)
    timestamps = [timestamp.to_pydatetime() for timestamp in grid]
    magnitudes = regular.astype(float).tolist()
    daily = detect_cron_patterns(timestamps, magnitudes)
    hourly = detect_hourly_patterns(timestamps, magnitudes)
    return {
        "regular_grid_minutes": len(grid),
        "presence_ratio": float((regular > 0).mean()),
        "daily_pattern_count": len(daily),
        "daily_pattern_slot_fraction": len(daily) / 1440.0,
        "hourly_pattern_count": len(hourly),
        "daily_patterns": [
            {"hour": item.hour, "minute": item.minute, "frequency": item.frequency}
            for item in daily
        ],
        "hourly_patterns": [
            {"minute": item.minute, "frequency": item.frequency}
            for item in hourly
        ],
    }


def compute_lifecycle_metrics(clean_data: pd.DataFrame, start: str) -> dict:
    """Summarize distinct exported record states at or after a cutoff."""
    required = {"id", "created_date", "modified_date"}
    missing = required.difference(clean_data.columns)
    if missing:
        raise ValueError(f"Missing clean-data columns: {sorted(missing)}")
    states = clean_data.loc[:, sorted(required)].copy()
    states["created_date"] = pd.to_datetime(states["created_date"], utc=True)
    states["modified_date"] = pd.to_datetime(states["modified_date"], utc=True)
    states = states[states["modified_date"] >= _utc_timestamp(start)]
    states = states.dropna(subset=["id", "created_date", "modified_date"])
    states = states.drop_duplicates(subset=["id", "modified_date"])
    if states.empty:
        raise ValueError("No exported record states remain after the cutoff")

    grouped = states.groupby("id", sort=False)["modified_date"]
    counts = grouped.size().astype(int)
    spans = (grouped.max() - grouped.min()).dt.total_seconds().astype(float)
    descending = np.sort(counts.to_numpy())[::-1]
    top_n = max(int(np.ceil(len(descending) * 0.10)), 1)
    total = int(counts.sum())
    return {
        "records_with_state": int(len(counts)),
        "observed_states": total,
        "median_states_per_record": float(counts.median()),
        "pct_one_state_records": float((counts == 1).mean()),
        "pct_records_gt_10_states": float((counts > 10).mean()),
        "max_states_single_record": int(counts.max()),
        "top_10pct_state_share": float(descending[:top_n].sum() / total),
        "median_observed_activity_span_seconds": float(spans.median()),
    }


def windows_for_unit(sync_unit_name: str) -> list[tuple[str, str | None]]:
    """Return generic sensitivity windows plus the unit's modeled window."""
    return [*STANDARD_WINDOWS, ("modeled", MODELED_STARTS.get(sync_unit_name))]


def load_registered_units(
    register_path: Path = REGISTER_PATH,
    data_root: Path = PROJECT_ROOT / "data",
) -> list[dict]:
    """Read sync-unit identities and per-minute artifact paths."""
    with register_path.open() as file:
        units = json.load(file)
    for unit in units:
        unit["name"] = "_".join(
            [unit["organization"], unit["external_system"], unit["project"]]
        )
        unit["path"] = (
            data_root
            / unit["organization"]
            / unit["external_system"]
            / unit["project"]
            / "data_preprocessing"
            / ARTIFACT_NAME
        )
        unit["clean_path"] = (
            data_root
            / unit["organization"]
            / unit["external_system"]
            / unit["project"]
            / "clean_data"
            / CLEAN_ARTIFACT_NAME
        )
    return units


def load_connection_config(path: Path) -> dict:
    """Load and minimally validate connection-informed window metadata."""
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    if not config.get("units"):
        raise ValueError("Connection cutoff metadata has no units")
    for name, unit in config["units"].items():
        if not unit.get("scenarios") or not unit.get("split"):
            raise ValueError(f"Incomplete connection metadata for {name}")
    return config


def analyze_all_units(
    register_path: Path = REGISTER_PATH,
    data_root: Path = PROJECT_ROOT / "data",
) -> pd.DataFrame:
    """Compute all five sensitivity scenarios for every registered unit."""
    rows = []
    for unit in load_registered_units(register_path, data_root):
        events_per_minute = pd.read_pickle(unit["path"])
        for window_id, start in windows_for_unit(unit["name"]):
            metrics = compute_window_metrics(
                events_per_minute, start=start
            )
            rows.append(
                {
                    "sync_unit": unit["name"],
                    "window": window_id,
                    "requested_start": start or "full",
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def analyze_connection_windows(
    config_path: Path,
    register_path: Path = REGISTER_PATH,
    data_root: Path = PROJECT_ROOT / "data",
) -> dict:
    """Compute stream, lifecycle, pattern, and split evidence per cutoff scenario."""
    config = load_connection_config(config_path)
    registered = {
        unit["name"]: unit for unit in load_registered_units(register_path, data_root)
    }
    output = {
        "semantics_version": config["semantics_version"],
        "units": {},
    }
    missing = set(config["units"]).difference(registered)
    if missing:
        raise ValueError(f"Cutoff units missing from registry: {sorted(missing)}")

    for name, metadata in config["units"].items():
        unit = registered[name]
        events = pd.read_pickle(unit["path"])
        clean = pd.read_pickle(unit["clean_path"])
        scenarios = []
        for scenario in metadata["scenarios"]:
            start = scenario["start"]
            scenarios.append(
                {
                    **scenario,
                    "stream": compute_window_metrics(events, start=start),
                    "lifecycle": compute_lifecycle_metrics(clean, start),
                    "patterns": compute_pattern_metrics(events, start),
                    "split_coverage": compute_split_coverage(metadata["split"], start),
                }
            )
        output["units"][name] = {
            "display_name": metadata["display_name"],
            "cutoff_status": metadata["cutoff_status"],
            "sync_mode_before_oct_2024": metadata["sync_mode_before_oct_2024"],
            "interval_minutes_before_oct_2024": metadata[
                "interval_minutes_before_oct_2024"
            ],
            "full_stream": compute_window_metrics(events),
            "scenarios": scenarios,
        }
    return output


def render_connection_markdown(result: dict) -> str:
    """Render compact thesis-facing evidence tables from the JSON result."""
    lines = [
        "# Connection-informed observation-window sensitivity",
        "",
        "## Post-cutoff stream characterization",
        "",
        "| Unit | Scenario | Span (d) | Event-min/day | Regime | Records | Daily/hourly patterns |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for unit in result["units"].values():
        for scenario in unit["scenarios"]:
            stream = scenario["stream"]
            lifecycle = scenario["lifecycle"]
            patterns = scenario["patterns"]
            lines.append(
                f"| {unit['display_name']} | {scenario['id']} | {stream['span_days']:.1f} "
                f"| {stream['minute_density_per_day']:.1f} | {stream['regime']} "
                f"| {lifecycle['records_with_state']:,} "
                f"| {patterns['daily_pattern_count']}/{patterns['hourly_pattern_count']} |"
            )
    lines.extend(
        [
            "",
            "## As-trained split coverage after cutoff",
            "",
            "| Unit | Scenario | Train | Validation | Test |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for unit in result["units"].values():
        for scenario in unit["scenarios"]:
            coverage = scenario["split_coverage"]
            cells = [
                f"{100 * coverage[name]['post_cutoff_ratio']:.1f}% "
                f"({coverage[name]['classification']})"
                for name in ("train", "validation", "test")
            ]
            lines.append(
                f"| {unit['display_name']} | {scenario['id']} | "
                + " | ".join(cells)
                + " |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the sensitivity analysis and print its compact comparison view."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTER_PATH)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--cutoffs", type=Path,
                        help="Connection cutoff configuration for connection analysis")
    parser.add_argument("--csv", type=Path, help="Also write the full result as CSV")
    parser.add_argument(
        "--connection-json", type=Path, help="Write connection-informed evidence as JSON"
    )
    parser.add_argument(
        "--connection-markdown", type=Path, help="Write a compact connection audit"
    )
    args = parser.parse_args()

    if args.connection_json or args.connection_markdown:
        if args.cutoffs is None:
            parser.error("--cutoffs is required for connection analysis")
        connection_result = analyze_connection_windows(
            args.cutoffs, args.registry, args.data_root,
        )
        if args.connection_json:
            args.connection_json.write_text(
                json.dumps(connection_result, indent=2) + "\n", encoding="utf-8"
            )
        if args.connection_markdown:
            args.connection_markdown.write_text(
                render_connection_markdown(connection_result), encoding="utf-8"
            )
        return

    result = analyze_all_units(args.registry, args.data_root)
    if args.csv:
        result.to_csv(args.csv, index=False)
    columns = [
        "sync_unit",
        "window",
        "minute_density_per_day",
        "coverage_ratio",
        "median_iat_minutes",
        "top3_minutes",
        "top3_minute_share",
        "regime",
    ]
    print(result[columns].to_string(index=False))


if __name__ == "__main__":
    main()
