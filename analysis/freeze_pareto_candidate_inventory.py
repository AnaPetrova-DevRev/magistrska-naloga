#!/usr/bin/env python3
"""Freeze a metadata-verified candidate inventory for Pareto analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    "unit", "matched_steps", "family", "policy", "decisions_path",
}
IDENTITY_FIELDS = (
    "task_type", "total_timesteps", "agent_config", "env_config",
    "reward_config", "prediction_config", "data_split",
)
VOLATILE_KEYS = {"agent_name", "seed", "trained_at", "mlflow_run_id"}
SEED_PATTERN = re.compile(r"_seed(?:[0-9]+)(?:_|$)")
EXCLUDE_PATTERN = re.compile(r"(?:^|_)(?:pilot|smoke|gate)[a-z0-9]*(?:_|$)")


def _clean_identity(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _clean_identity(item)
            for key, item in sorted(value.items())
            if key not in VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [_clean_identity(item) for item in value]
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _training_metadata(decisions_path: Path, code_root: Path) -> tuple[dict, Path]:
    policy_metadata_path = decisions_path.parent / "policy_metadata.json"
    policy_metadata = json.loads(policy_metadata_path.read_text(encoding="utf-8"))
    policy_path = _resolve(code_root, policy_metadata["policy_path"])
    training_path = policy_path / "training_metadata.json"
    return json.loads(training_path.read_text(encoding="utf-8")), policy_path


def _portable_path(path: Path, code_root: Path) -> str:
    """Prefer a repository-relative provenance path when possible."""
    try:
        return str(path.relative_to(code_root))
    except ValueError:
        return str(path)


def _configuration_digest(metadata: dict) -> str:
    identity = _clean_identity({key: metadata.get(key) for key in IDENTITY_FIELDS})
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _inventory_record(row: dict, code_root: Path) -> dict:
    """Add reproducibility fields to one source-inventory row."""
    record = dict(row)
    record.update({
        "configuration_digest": "", "seed": "", "family_eligible": False,
        "selection_reason": "generated_periodic_set_replaces_legacy_grid",
        "representative_policy": "", "repetition_count": 1,
        "source_policy_path": "", "source_trained_at": "",
    })
    if row["family"] != "RL":
        return record
    decisions = _resolve(code_root, row["decisions_path"])
    metadata, policy_path = _training_metadata(decisions, code_root)
    record.update({
        "configuration_digest": _configuration_digest(metadata),
        "seed": metadata.get("agent_config", {}).get("seed", ""),
        "source_policy_path": _portable_path(policy_path, code_root),
        "source_trained_at": metadata.get("trained_at", ""),
        "selection_reason": (
            "protocol_excluded" if EXCLUDE_PATTERN.search(row["policy"])
            else "pending_representative_selection"
        ),
    })
    return record


def _mark_representatives(result: pd.DataFrame) -> None:
    """Select one deterministic representative per substantive RL configuration."""
    rl = result[result["family"] == "RL"]
    for _, group in rl.groupby(["unit", "configuration_digest"], sort=True):
        eligible = group[~group["policy"].str.contains(EXCLUDE_PATTERN)]
        if eligible.empty:
            continue
        canonical = eligible[~eligible["policy"].str.contains(SEED_PATTERN)]
        reason = "canonical_unsuffixed"
        if canonical.empty:
            eligible = eligible.assign(
                _seed_order=eligible["seed"].replace("", float("inf")).astype(float),
            ).sort_values(["_seed_order", "policy"], kind="stable")
            representative_index = eligible.index[0]
            reason = "lowest_seed_fallback"
        else:
            canonical = canonical.sort_values(
                ["source_trained_at", "policy"], kind="stable",
            )
            representative_index = canonical.index[0]
            if len(canonical) > 1:
                reason = "earliest_unsuffixed"
        representative = str(result.loc[representative_index, "policy"])
        result.loc[group.index, "representative_policy"] = representative
        result.loc[group.index, "repetition_count"] = len(eligible)
        seed_repetitions = eligible[eligible["policy"].str.contains(SEED_PATTERN)]
        unseeded_repetitions = eligible.drop(seed_repetitions.index)
        result.loc[seed_repetitions.index, "selection_reason"] = "seed_robustness_only"
        result.loc[
            unseeded_repetitions.index, "selection_reason"
        ] = "unseeded_robustness_only"
        result.loc[representative_index, "family_eligible"] = True
        result.loc[representative_index, "selection_reason"] = reason


def freeze_inventory(source: Path, output: Path, code_root: Path) -> pd.DataFrame:
    """Build and persist a deterministic, metadata-audited inventory."""
    frame = pd.read_csv(source)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Candidate inventory lacks columns: {missing}")

    result = pd.DataFrame([
        _inventory_record(row, code_root) for row in frame.to_dict("records")
    ])
    _mark_representatives(result)

    result = result.sort_values(["unit", "family", "policy"], kind="stable")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result.reset_index(drop=True)


def main() -> None:
    """Parse command-line arguments and write the frozen inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = freeze_inventory(args.source, args.output, args.code_root.resolve())
    print(f"Wrote {len(result)} candidates to {args.output}")


if __name__ == "__main__":
    main()
