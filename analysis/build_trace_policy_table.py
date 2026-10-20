#!/usr/bin/env python3
"""Build the eight-unit trace-property and canonical-policy comparison table."""

# pylint: disable=too-many-locals

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

if __package__:
    from analysis.rl_periodic_reward_analysis import load_manifest, run_analysis
else:
    from rl_periodic_reward_analysis import load_manifest, run_analysis


ANALYSIS_DIR = Path(__file__).resolve().parent
CODE_ROOT = ANALYSIS_DIR.parent
SYSTEM_LABELS = {
    "salesforce": "Salesforce",
    "servicenow": "ServiceNow",
    "zendesk": "Zendesk",
}


def _unit_name(path: Path, data_root: Path = CODE_ROOT / "data") -> str:
    """Return the display name used by the canonical policy manifest."""
    relative = path.relative_to(data_root)
    org, system, project = relative.parts[:3]
    if system == "jira":
        return f"{org.title()} Jira {project.title()}"
    return f"{org.title()} {SYSTEM_LABELS[system]}"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_window(
    unit_name: str,
    window_manifest: dict,
    expected_steps: int,
    decisions_path: Path,
) -> tuple[str, str]:
    """Require the canonical test trace to be the modeled window's 15% suffix."""
    if window_manifest.get("semantics_version") != "observable-stream-window-v1":
        raise ValueError(f"{unit_name}: unsupported window semantics")
    start = pd.Timestamp(window_manifest["effective_start"])
    end = pd.Timestamp(window_manifest["effective_end"])
    total_steps = int((end - start).total_seconds() // 60) + 1
    test_steps = total_steps - int(total_steps * 0.70) - int(total_steps * 0.15)
    expected_start = start + pd.Timedelta(minutes=total_steps - test_steps)
    if test_steps != expected_steps:
        raise ValueError(
            f"{unit_name}: modeled test has {test_steps} steps, expected {expected_steps}"
        )
    timestamps = pd.read_parquet(decisions_path, columns=["timestamp"])["timestamp"]
    parsed = pd.to_datetime(timestamps, utc=True, errors="coerce")
    dated = parsed.dropna()
    if len(parsed) != expected_steps or len(dated) != expected_steps - 1:
        raise ValueError(f"{unit_name}: unexpected dated/terminal decision count")
    if not pd.isna(parsed.iloc[-1]) or parsed.iloc[:-1].isna().any():
        raise ValueError(f"{unit_name}: terminal timestamp convention changed")
    if dated.iloc[0] != expected_start + pd.Timedelta(minutes=1):
        raise ValueError(f"{unit_name}: policy test start does not match modeled window")
    if dated.iloc[-1] != end:
        raise ValueError(f"{unit_name}: policy test end does not match modeled window")
    return dated.iloc[0].isoformat(), dated.iloc[-1].isoformat()


def build_rows(policy_manifest: Path, data_root: Path = CODE_ROOT / "data") -> list[dict]:
    """Load, validate, and density-order the eight comparison rows."""
    _, units = load_manifest(policy_manifest)
    policy = run_analysis(units, thresholds=(0.0,), costs=(1.0,), include_family=False)
    policy_rows = policy["canonical_pairs"].set_index("unit")
    unit_specs = {unit.name: unit for unit in units}
    rows = []
    fingerprints = sorted(
        data_root.glob("*/*/*/windowed_analysis/modeled/fingerprint.json")
    )
    for fingerprint_path in fingerprints:
        unit_name = _unit_name(fingerprint_path, data_root)
        if unit_name not in unit_specs:
            continue
        directory = fingerprint_path.parent
        fingerprint = _read_json(fingerprint_path)
        features = fingerprint["features"]
        window_manifest = _read_json(directory / "manifest.json")
        policy_row = policy_rows.loc[unit_name]
        spec = unit_specs[unit_name]
        policy_start, policy_end = _validate_window(
            unit_name, window_manifest, spec.expected_steps, spec.fixed_rl.path
        )
        median_iat = float(features["delta_median"])
        if median_iat <= 0:
            raise ValueError(f"{unit_name}: non-positive median IAT")
        rows.append({
            "unit": unit_name,
            "coverage_overall": float(features["coverage_ratio"]),
            "coverage_train": 1.0 - float(features["zero_fraction_train"]),
            "coverage_test": 1.0 - float(features["zero_fraction_test"]),
            "coverage_test_minus_train": (
                float(features["zero_fraction_train"])
                - float(features["zero_fraction_test"])
            ),
            "median_iat_minutes": median_iat,
            "mean_median_iat_ratio": float(features["delta_mean"]) / median_iat,
            "multi_event_minute_fraction": float(
                features["multi_event_minute_fraction"]
            ),
            "detected_slot_count": int(
                features["num_daily_cron_patterns"]
                + features["num_hourly_cron_patterns"]
            ),
            "cron_event_fraction": float(features["cron_event_fraction"]),
            "rl_periodic_reward_difference_pct": float(
                policy_row["reward_difference_pct"]
            ),
            "policy_first_decision": policy_start,
            "policy_last_dated_decision": policy_end,
        })
    if len(rows) != len(units):
        raise ValueError(f"Expected {len(units)} units, built {len(rows)}")
    return sorted(rows, key=lambda row: row["coverage_overall"], reverse=True)


def _num(value: float, decimals: int = 1, signed: bool = False) -> str:
    text = f"{value:{'+' if signed else ''}.{decimals}f}"
    return text.replace(".", "{,}")


def _latex(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{l r r r r r r}",
        r"\toprule",
        r"Enota & Aktivne minute & Mediana IAT & Razmerje & Več dogodkov & "
        r"Zaznane reže & $\Delta R$ \\",
        r"\midrule",
    ]
    for row in rows:
        coverage = (
            f"{_num(100 * row['coverage_overall'])}\\% "
            f"({_num(100 * row['coverage_train'])}$\\to$"
            f"{_num(100 * row['coverage_test'])}\\%)"
        )
        slots = (
            f"{_num(100 * row['cron_event_fraction'])}\\% "
            f"({row['detected_slot_count']})"
        )
        lines.append(
            f"{row['unit']} & {coverage} & {_num(row['median_iat_minutes'])} & "
            f"{_num(row['mean_median_iat_ratio'])} & "
            f"{_num(100 * row['multi_event_minute_fraction'])}\\% & {slots} & "
            f"{_num(row['rl_periodic_reward_difference_pct'], signed=True)}\\% \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write machine-readable and LaTeX outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Input data root containing modeled-window fingerprints")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = build_rows(args.manifest, args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "trace_policy_comparison.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(args.output_dir / "trace_policy_comparison.csv", index=False)
    (args.output_dir / "trace_policy_comparison.tex").write_text(
        _latex(rows), encoding="utf-8"
    )
    print(f"Wrote {len(rows)} validated rows to {args.output_dir}")


if __name__ == "__main__":
    main()
