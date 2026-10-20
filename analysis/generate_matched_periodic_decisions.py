"""Generate deterministic periodic decisions on each matched thesis test window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Support both ``python -m analysis...`` and direct script execution.
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

# pylint: disable=import-error,wrong-import-position
from sync_policy_evaluation.decision_metrics import _compute_true_aoi_series


OUTPUT_ROOT = Path(__file__).resolve().parent / "generated_periodic_decisions"
INTERVALS = (1, 2, 5, 10, 15, 20, 30, 45, 60)
UNITS = (
    ("ecosphere_zendesk", "ecosphere/zendesk/ecosphere_zendesk"),
    ("brightpath_salesforce", "brightpath/salesforce/brightpath_salesforce"),
    ("novacore_servicenow", "novacore/servicenow/novacore_servicenow"),
    ("novacore_jira_bugs", "novacore/jira/bugs"),
    ("brightpath_servicenow", "brightpath/servicenow/brightpath_servicenow"),
    ("novacore_jira_support", "novacore/jira/support"),
    ("novacore_zendesk", "novacore/zendesk/novacore_zendesk"),
    ("novacore_jira_incidents", "novacore/jira/incidents"),
)


def periodic_actions(step_count: int, interval: int) -> np.ndarray:
    """Match PeriodicSyncPolicy: first sync occurs at step interval - 1."""
    if interval < 1:
        raise ValueError("interval must be at least one minute")
    return (((np.arange(step_count) + 1) % interval) == 0).astype(np.int8)


def build_periodic_decisions(template: pd.DataFrame, interval: int) -> pd.DataFrame:
    """Apply one periodic action sequence to a matched event-rate test trace."""
    actions = periodic_actions(len(template), interval)
    actual_values = template["actual_value"].to_numpy(dtype=float)
    true_aoi = _compute_true_aoi_series(actual_values, actions, "event_rate")
    return pd.DataFrame({
        "timestamp": template["timestamp"].to_numpy(),
        "step": template["step"].to_numpy(),
        "action": actions,
        "actual_value": actual_values,
        "true_aoi": true_aoi,
    })


def matched_group_traces(
    unit_path: str,
    data_root: Path = CODE_ROOT / "data",
) -> tuple[dict[str, Path], int]:
    """Select the largest standard-reward window containing RL and periodic traces."""
    policy_dir = data_root / unit_path / "rl_policies"
    groups: dict[int, dict[str, Path]] = {}
    for path in sorted(policy_dir.glob("*_test_*/decisions.parquet")):
        summary_path = path.with_name("summary_metrics.json")
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary.get("is_standard_reward"):
            continue
        steps = int(summary["total_steps"])
        name = path.parent.name.rsplit("_test_", 1)[0]
        groups.setdefault(steps, {})[name] = path

    eligible = [
        (steps, traces)
        for steps, traces in groups.items()
        if any(name.startswith(("a2c_", "ppo_", "dqn_")) for name in traces)
        and any(name.startswith("periodic_") for name in traces)
    ]
    if not eligible:
        raise RuntimeError(f"No matched comparison group for {unit_path}")
    steps, traces = max(eligible, key=lambda item: (item[0], len(item[1])))
    return traces, steps


def validate_against_existing(generated: pd.DataFrame, existing_path: Path) -> None:
    """Require exact agreement with an existing evaluation decision trace."""
    existing = pd.read_parquet(
        existing_path, columns=["step", "action", "actual_value", "true_aoi"],
    )
    for column in ("step", "action", "actual_value", "true_aoi"):
        if not np.array_equal(generated[column].to_numpy(), existing[column].to_numpy()):
            raise RuntimeError(f"Generated {column} differs from {existing_path}")


def archived_trace(
    unit_path: str,
    interval: int,
    actual_values: np.ndarray,
    data_root: Path = CODE_ROOT / "data",
) -> Path | None:
    """Find an archived periodic trace only when its full event sequence matches."""
    archive = data_root / unit_path / "rl_policies_prefix_archive"
    for path in sorted(archive.glob(f"periodic_{interval}min_test_*/decisions.parquet")):
        archived_values = pd.read_parquet(path, columns=["actual_value"])[
            "actual_value"
        ].to_numpy()
        if np.array_equal(archived_values, actual_values):
            return path
    return None


def _record_path(path: Path, reference_root: Path) -> str:
    """Preserve relative references when possible, including external outputs."""
    try:
        return str(path.resolve().relative_to(reference_root.resolve()))
    except ValueError:
        return str(path.resolve())


def generate_unit(
    unit_key: str,
    unit_path: str,
    *,
    data_root: Path = CODE_ROOT / "data",
    output_root: Path = OUTPUT_ROOT,
) -> list[dict]:
    """Generate and validate all requested intervals for one unit."""
    traces, matched_steps = matched_group_traces(unit_path, data_root)
    source_name = "periodic_15min"
    if source_name not in traces:
        raise RuntimeError(f"No {source_name} source trace for {unit_path}")
    source_path = traces[source_name]
    template = pd.read_parquet(source_path, columns=["timestamp", "step", "actual_value"])
    if len(template) != matched_steps:
        raise RuntimeError(f"Source length mismatch for {unit_path}")

    records = []
    for interval in INTERVALS:
        generated = build_periodic_decisions(template, interval)
        existing = traces.get(f"periodic_{interval}min")
        if existing is None:
            existing = archived_trace(
                unit_path, interval, template["actual_value"].to_numpy(), data_root,
            )
        if existing is not None:
            validate_against_existing(generated, existing)

        destination = output_root / unit_key / f"periodic_{interval}min"
        destination.mkdir(parents=True, exist_ok=True)
        decisions_path = destination / "decisions.parquet"
        generated.to_parquet(decisions_path, index=False)
        metadata = {
            "unit_path": unit_path,
            "interval_minutes": interval,
            "matched_steps": matched_steps,
            "task_type": "event_rate",
            "source_decisions": _record_path(source_path, data_root.parent),
            "validated_against": (
                _record_path(existing, data_root.parent) if existing is not None else None
            ),
            "columns": list(generated.columns),
            "sync_count": int(generated["action"].sum()),
            "mean_true_aoi": float(generated["true_aoi"].mean()),
        }
        (destination / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
        )
        records.append(metadata | {"decisions": _record_path(decisions_path, data_root.parent)})
    return records


def main() -> None:
    """Generate matched periodic traces from a supplied evaluation archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Input data root containing unit rl_policies directories")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--unit", action="append", metavar="ORG/SYSTEM/PROJECT",
                        help="Generate one supplied unit; repeat for multiple units")
    args = parser.parse_args()
    units = UNITS
    if args.unit:
        units = []
        for unit_path in args.unit:
            parts = unit_path.split("/")
            if len(parts) != 3 or any(part in ("", ".", "..") for part in parts):
                parser.error("--unit must have the form ORG/SYSTEM/PROJECT")
            units.append(("_".join(parts), unit_path))
    records = []
    for unit_key, unit_path in units:
        records.extend(generate_unit(
            unit_key, unit_path, data_root=args.data_root, output_root=args.output_dir,
        ))
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(records)} traces under {args.output_dir}")
    for record in records:
        status = record["validated_against"] or "new deterministic baseline"
        print(record["unit_path"], record["interval_minutes"], record["matched_steps"], status)


if __name__ == "__main__":
    main()
