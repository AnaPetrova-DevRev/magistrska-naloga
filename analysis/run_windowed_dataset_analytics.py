#!/usr/bin/env python3
"""Generate isolated per-window lifecycle, fingerprint, and clustering artifacts."""

import argparse
import gc
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_comparison.weighted_comparison import (  # pylint: disable=import-error
    compute as compute_comparison,
)
from dataset_comparison.windowed import (  # pylint: disable=import-error
    AnalysisWindow,
    build_windowed_time_deltas,
    compute_windowed_fingerprint,
    filter_timestamp_window,
    windows_for_unit,
)
from sync_unit import SyncUnit, get_sync_units  # pylint: disable=import-error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEMANTICS_VERSION = "observable-stream-window-v1"


def _json_safe(value):
    """Replace non-finite scalars recursively for standards-compliant JSON."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _write_json(path: Path, value) -> None:
    """Write one derived artifact without touching canonical pipeline files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(_json_safe(value), file, indent=2)


def _unit_root(sync_unit: SyncUnit) -> Path:
    return PROJECT_ROOT / "data" / sync_unit.get_path_for_sync_unit()


def _load_unit_artifacts(sync_unit: SyncUnit) -> tuple[pd.DataFrame, ...]:
    """Load cleaned states and stream artifacts once for one sync unit."""
    root = _unit_root(sync_unit)
    clean_paths = sorted((root / "clean_data").glob("cleaned_items_*.pkl"))
    if not clean_paths:
        raise FileNotFoundError(f"No cleaned data found for {sync_unit.get_name()}")
    clean_parts = [pd.read_pickle(path) for path in clean_paths]
    clean = clean_parts[0] if len(clean_parts) == 1 else pd.concat(clean_parts)
    epm_path = root / "data_preprocessing" / "merged_events_per_minute_1.pkl"
    merged_path = root / "data_preprocessing" / "merged_dataset_1.pkl"
    return clean, pd.read_pickle(epm_path), pd.read_pickle(merged_path)


def _temporal_summary(
    epm_df: pd.DataFrame,
    window: AnalysisWindow,
    fingerprint: dict,
) -> tuple[dict, np.ndarray]:
    """Extract compact temporal output and deterministic IAT samples."""
    window_epm = filter_timestamp_window(epm_df, "minute", window)
    timestamps = pd.to_datetime(window_epm["minute"], utc=True)
    deltas = build_windowed_time_deltas(window_epm)["events_delta"].dropna()
    values = deltas.astype(float).to_numpy()
    if len(values) > 10_000:
        values = np.random.default_rng(42).choice(values, 10_000, replace=False)
    features = fingerprint["features"]
    keys = [
        "raw_events", "minute_events", "minute_density_per_day",
        "coverage_ratio", "dedup_ratio", "delta_mean", "delta_median",
        "multi_event_minute_fraction", "zero_fraction_all",
        "business_hours_fraction", "weekend_fraction",
        "num_hourly_cron_patterns", "num_daily_cron_patterns",
        "cron_event_fraction", "iat_ks_pvalue",
    ]
    summary = {key: features.get(key) for key in keys}
    summary.update(
        {
            "effective_start": timestamps.min().isoformat(),
            "effective_end": timestamps.max().isoformat(),
            "regime": fingerprint["regime"],
            "dow_profile": fingerprint["dow_profile"],
            "hod_profile": fingerprint["hod_profile"],
        }
    )
    return summary, values


def _write_unit_window(
    sync_unit: SyncUnit,
    window: AnalysisWindow,
    fingerprint: dict,
    lifecycle: dict,
    temporal: dict,
) -> None:
    """Write one window under the unit's isolated derived-artifact directory."""
    output = _unit_root(sync_unit) / "windowed_analysis" / window.window_id
    manifest = {
        "semantics_version": SEMANTICS_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sync_unit": sync_unit.get_name(),
        "requested_window": window.to_dict(),
        "effective_start": temporal["effective_start"],
        "effective_end": temporal["effective_end"],
        "sources": [
            "clean_data/cleaned_items_*.pkl",
            "data_preprocessing/merged_events_per_minute_1.pkl",
            "data_preprocessing/merged_dataset_1.pkl",
        ],
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "lifecycle_summary.json", lifecycle)
    _write_json(output / "temporal_summary.json", temporal)
    _write_json(output / "fingerprint.json", fingerprint)


def analyze_unit(sync_unit: SyncUnit) -> dict[str, dict]:
    """Analyze all unique bounds for one unit and write five named windows."""
    print(f"Loading {sync_unit.get_name()}...", flush=True)
    clean, epm, merged = _load_unit_artifacts(sync_unit)
    cache = {}
    result = {}
    for window in windows_for_unit(sync_unit.get_name()):
        key = (window.start, window.end)
        if key not in cache:
            fingerprint, lifecycle, records = compute_windowed_fingerprint(
                sync_unit, clean, epm, merged, window
            )
            fingerprint_dict = fingerprint.to_dict()
            temporal, deltas = _temporal_summary(epm, window, fingerprint_dict)
            cache[key] = (fingerprint_dict, lifecycle, temporal, deltas)
            del records
            gc.collect()
        fingerprint_dict, lifecycle, temporal, deltas = cache[key]
        _write_unit_window(
            sync_unit, window, fingerprint_dict, lifecycle, temporal
        )
        result[window.window_id] = {
            "fingerprint": fingerprint_dict,
            "lifecycle": lifecycle,
            "temporal": temporal,
            "deltas": deltas,
            "requested_window": window.to_dict(),
        }
        print(f"  wrote {window.window_id}", flush=True)
    del clean, epm, merged, cache
    gc.collect()
    return result


def run_all_units() -> None:
    """Generate per-unit outputs followed by suite-level comparisons."""
    suite = {window.window_id: [] for window in windows_for_unit("placeholder")}
    samples = {window_id: {} for window_id in suite}
    requested = {window_id: {} for window_id in suite}
    for sync_unit in get_sync_units():
        unit_result = analyze_unit(sync_unit)
        for window_id, result in unit_result.items():
            suite[window_id].append(result["fingerprint"])
            samples[window_id][sync_unit.get_name()] = result["deltas"]
            requested[window_id][sync_unit.get_name()] = result["requested_window"]

    base = PROJECT_ROOT / "data" / "dataset_comparison" / "windowed"
    for window_id, fingerprints in suite.items():
        output = base / window_id
        fingerprint_path = output / "fingerprints.json"
        _write_json(fingerprint_path, fingerprints)
        _write_json(
            output / "manifest.json",
            {
                "semantics_version": SEMANTICS_VERSION,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "requested_windows": requested[window_id],
                "sync_unit_count": len(fingerprints),
            },
        )
        compute_comparison(
            out_dir=str(output),
            fingerprint_path=str(fingerprint_path),
            delta_samples=samples[window_id],
            runs=("lifecycle_excluded", "behavioral", "full"),
        )


def main() -> None:
    """Parse CLI arguments and execute the requested analysis scope."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", help="Run one registered sync-unit name only")
    args = parser.parse_args()
    if args.unit:
        units = {unit.get_name(): unit for unit in get_sync_units()}
        if args.unit not in units:
            parser.error(f"Unknown sync unit: {args.unit}")
        analyze_unit(units[args.unit])
        return
    run_all_units()


if __name__ == "__main__":
    main()
