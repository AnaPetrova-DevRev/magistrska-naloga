#!/usr/bin/env python3
"""Audit observation clipping across the frozen eligible RL candidates."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# pylint: disable=too-many-locals


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_SPECS = {
    "time_since_last_sync": ("time_since_last_sync", 1440.0),
    "current_aoi": ("current_aoi", 1440.0),
    "items_synced_last": ("last_sync_items", 1000.0),
    "cumulative_empty_syncs": ("cumulative_empty_syncs", 1440.0),
}


def _resolve(path: str, code_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else code_root / candidate


def audit_observation_clipping(
    inventory_path: Path,
    code_root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Summarize clipping only when a candidate actually used the feature."""
    inventory = pd.read_csv(inventory_path)
    candidates = inventory[
        (inventory["family"] == "RL") & inventory["family_eligible"]
    ].drop_duplicates("decisions_path")
    totals = {}

    for row in candidates.itertuples(index=False):
        metadata_path = (
            _resolve(row.source_policy_path, code_root) / "training_metadata.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        enabled = set(
            metadata.get("env_config", {}).get("observation_features", [])
        )
        active = {
            feature: spec
            for feature, spec in FEATURE_SPECS.items()
            if feature in enabled
        }
        if not active:
            continue

        trace = pq.read_table(
            _resolve(row.decisions_path, code_root),
            columns=[column for column, _ in active.values()],
        )
        for feature, (column, limit) in active.items():
            values = trace[column].to_numpy(zero_copy_only=False)
            values = values[np.isfinite(values)]
            for unit in (row.unit, "ALL"):
                key = (unit, feature)
                summary = totals.setdefault(
                    key,
                    {
                        "candidate_count": 0,
                        "minute_count": 0,
                        "values_above_limit": 0,
                        "maximum": 0.0,
                        "limit": limit,
                    },
                )
                summary["candidate_count"] += 1
                summary["minute_count"] += int(values.size)
                summary["values_above_limit"] += int(
                    np.count_nonzero(values > limit)
                )
                if values.size:
                    summary["maximum"] = max(
                        summary["maximum"], float(values.max())
                    )

    records = []
    for (unit, feature), summary in totals.items():
        count = summary["minute_count"]
        above = summary["values_above_limit"]
        records.append(
            {
                "unit": unit,
                "feature": feature,
                **summary,
                "share_above_limit": above / count if count else 0.0,
            }
        )
    return pd.DataFrame(records).sort_values(["unit", "feature"])


def main() -> None:
    """Run the audit and optionally persist its tabular result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=Path.cwd(),
                        help="Root for relative paths stored in the inventory")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    result = audit_observation_clipping(args.inventory, args.code_root)
    if args.csv:
        result.to_csv(args.csv, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
