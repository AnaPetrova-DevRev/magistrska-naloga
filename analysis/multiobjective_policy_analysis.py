#!/usr/bin/env python3
"""Reusable Pareto and objective-weight analysis for sync policies.

The input is the ``candidate_metrics.csv`` table emitted by
``rl_periodic_reward_analysis.py``. It supports both the original three-objective
analysis and separate two-objective variants that avoid request-cost overlap.
"""

# pylint: disable=too-many-locals

from __future__ import annotations

import argparse
import math
import re
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FRESHNESS_VARIANTS = {
    "aoii": "cvar95_aoii",
    "aoi": "cvar95_true_aoi",
}
DEFAULT_NORMALIZATIONS = ("raw", "log1p")
PRIORITY_SCENARIOS = {
    "request_focused": (0.70, 0.15, 0.15),
    "empty_sync_focused": (0.15, 0.70, 0.15),
    "freshness_focused": (0.15, 0.15, 0.70),
    "balanced": (1 / 3, 1 / 3, 1 / 3),
}
REQUIRED_COLUMNS = {
    "unit", "family", "policy", "decisions_path", "sync_rate",
    "empty_sync_step_rate", *FRESHNESS_VARIANTS.values(),
}
GROUP_COLUMNS = ["unit", "freshness_variant", "normalization"]
SCORE_COLUMNS = ["normalized_sync", "normalized_empty", "normalized_freshness"]
WEIGHT_COLUMNS = ["weight_sync", "weight_empty", "weight_freshness"]
TWO_OBJECTIVE_PAIRS = {
    "sync_vs_aoii": ("sync_rate", "cvar95_aoii", "aoii"),
    "empty_vs_aoii": ("empty_sync_step_rate", "cvar95_aoii", "aoii"),
    "sync_vs_aoi": ("sync_rate", "cvar95_true_aoi", "aoi"),
    "empty_vs_aoi": ("empty_sync_step_rate", "cvar95_true_aoi", "aoi"),
}
TWO_OBJECTIVE_SCENARIOS = {
    "freshness_focused": 0.20,
    "moderately_freshness_focused": 0.35,
    "balanced": 0.50,
    "operational_focused": 0.80,
}
TWO_GROUP_COLUMNS = ["unit", "objective_pair", "normalization"]
TWO_SCORE_COLUMNS = ["normalized_operational", "normalized_freshness"]
TWO_WEIGHT_COLUMNS = ["weight_operational", "weight_freshness"]


def _validate_candidates(frame: pd.DataFrame) -> None:
    """Reject malformed candidate tables before optimization."""
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Candidate metrics lack required columns: {missing}")
    if frame.empty:
        raise ValueError("Candidate metrics are empty")
    if frame.duplicated(["unit", "family", "policy"]).any():
        raise ValueError("Candidate unit/family/policy identifiers must be unique")
    families = set(frame["family"].astype(str))
    if not families <= {"RL", "periodic"}:
        raise ValueError(f"Unsupported policy families: {sorted(families)}")
    identifiers = frame["family"].astype(str) + frame["policy"].astype(str)
    if identifiers.str.contains(r"[:;]", regex=True).any():
        raise ValueError("Family and policy identifiers cannot contain ':' or ';'")
    values = frame[["sync_rate", "empty_sync_step_rate", *FRESHNESS_VARIANTS.values()]]
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Objective values must be finite")
    if (values < 0).any().any():
        raise ValueError("Objective values must be non-negative")


def _family_eligibility(values: pd.Series) -> pd.Series:
    """Parse boolean eligibility from in-memory or CSV-backed tables."""
    if pd.api.types.is_bool_dtype(values):
        return values
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "1": True, "false": False, "0": False}
    if not normalized.isin(mapping).all():
        raise ValueError("family_eligible must contain only boolean values")
    return normalized.map(mapping).astype(bool)


def pareto_mask(values: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    """Return a mask selecting rows not dominated under minimization."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or len(array) == 0:
        raise ValueError("Pareto values must be a non-empty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("Pareto values must be finite")
    result = np.ones(len(array), dtype=bool)
    for index, value in enumerate(array):
        no_worse = np.all(array <= value + tolerance, axis=1)
        strictly_better = np.any(array < value - tolerance, axis=1)
        result[index] = not np.any(no_worse & strictly_better)
    return result


def _dominator_indices(
    values: np.ndarray, index: int, tolerance: float,
) -> np.ndarray:
    value = values[index]
    no_worse = np.all(values <= value + tolerance, axis=1)
    strictly_better = np.any(values < value - tolerance, axis=1)
    return np.flatnonzero(no_worse & strictly_better)


def minmax_normalize(values: Iterable[float]) -> np.ndarray:
    """Normalize a vector to [0, 1], mapping a constant vector to zero."""
    array = np.asarray(tuple(values), dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Normalization requires a finite non-empty vector")
    value_range = float(array.max() - array.min())
    if value_range <= 1e-15:
        return np.zeros(len(array), dtype=float)
    return (array - array.min()) / value_range


def build_candidate_objectives(
    candidate_metrics: pd.DataFrame,
    normalizations: Iterable[str] = DEFAULT_NORMALIZATIONS,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Add objective, Pareto, and normalized fields to candidate metrics."""
    _validate_candidates(candidate_metrics)
    if "family_eligible" in candidate_metrics:
        candidate_metrics = candidate_metrics[
            _family_eligibility(candidate_metrics["family_eligible"])
        ]
        if candidate_metrics.empty:
            raise ValueError("No family-eligible candidates are available")
    modes = tuple(dict.fromkeys(normalizations))
    unknown = sorted(set(modes) - {"raw", "log1p"})
    if not modes or unknown:
        raise ValueError(f"Unsupported freshness normalizations: {unknown}")
    rows: list[dict] = []
    for _, source in candidate_metrics.groupby("unit", sort=True):
        source = source.sort_values(["family", "policy"]).reset_index(drop=True)
        for variant, freshness_column in FRESHNESS_VARIANTS.items():
            raw_values = source[["sync_rate", "empty_sync_step_rate", freshness_column]].to_numpy()
            frontier = pareto_mask(raw_values, tolerance)
            dominators = [_dominator_indices(raw_values, i, tolerance) for i in range(len(source))]
            for mode in modes:
                transformed = (
                    np.log1p(source[freshness_column].to_numpy(dtype=float))
                    if mode == "log1p" else source[freshness_column].to_numpy(dtype=float)
                )
                normalized = np.column_stack([
                    minmax_normalize(source["sync_rate"]),
                    minmax_normalize(source["empty_sync_step_rate"]),
                    minmax_normalize(transformed),
                ])
                for index, (_, candidate) in enumerate(source.iterrows()):
                    ids = sorted(
                        f"{source.iloc[item]['family']}:{source.iloc[item]['policy']}"
                        for item in dominators[index]
                    )
                    rows.append({
                        **candidate.to_dict(),
                        "freshness_variant": variant,
                        "freshness_metric": freshness_column,
                        "normalization": mode,
                        "objective_sync_rate": float(raw_values[index, 0]),
                        "objective_empty_sync_step_rate": float(raw_values[index, 1]),
                        "objective_freshness": float(raw_values[index, 2]),
                        "transformed_freshness": float(transformed[index]),
                        "normalized_sync": float(normalized[index, 0]),
                        "normalized_empty": float(normalized[index, 1]),
                        "normalized_freshness": float(normalized[index, 2]),
                        "is_pareto": bool(frontier[index]),
                        "dominator_count": len(ids),
                        "dominators": ";".join(ids),
                    })
    return pd.DataFrame(rows).sort_values(
        [*GROUP_COLUMNS, "family", "policy"], kind="stable",
    ).reset_index(drop=True)


def generate_weight_grid(step: float = 0.05) -> pd.DataFrame:
    """Generate every non-negative three-weight composition at a fixed step."""
    fraction = Fraction(str(step))
    if fraction <= 0 or fraction > 1:
        raise ValueError("Weight step must be in (0, 1]")
    levels = Fraction(1, 1) / fraction
    if levels.denominator != 1:
        raise ValueError("Weight step must divide one exactly")
    count = levels.numerator
    rows = []
    for sync in range(count + 1):
        for empty in range(count - sync + 1):
            freshness = count - sync - empty
            rows.append({
                "weight_sync": sync / count,
                "weight_empty": empty / count,
                "weight_freshness": freshness / count,
            })
    result = pd.DataFrame(rows)
    result.insert(0, "weight_id", np.arange(len(result), dtype=int))
    result["is_interior"] = result[WEIGHT_COLUMNS].min(axis=1) >= 0.10 - 1e-12
    return result


def _score_groups(
    candidates: pd.DataFrame,
    weights: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    frontier = candidates[candidates["is_pareto"]]
    for keys, group in frontier.groupby(GROUP_COLUMNS, sort=True):
        group = group.sort_values(["family", "policy"]).reset_index(drop=True)
        objective_values = group[SCORE_COLUMNS].to_numpy(dtype=float)
        for _, weight in weights.iterrows():
            weight_values = weight[WEIGHT_COLUMNS].to_numpy(dtype=float)
            scores = objective_values @ weight_values
            best_score = float(scores.min())
            winner_indices = np.flatnonzero(
                np.isclose(scores, best_score, rtol=0.0, atol=tolerance)
            )
            winner_ids = [
                f"{group.iloc[i]['family']}:{group.iloc[i]['policy']}"
                for i in winner_indices
            ]
            winner_families = sorted({str(group.iloc[i]["family"]) for i in winner_indices})
            larger = scores[scores > best_score + tolerance]
            rows.append({
                **dict(zip(GROUP_COLUMNS, keys)),
                **weight.to_dict(),
                "winner_family": (
                    winner_families[0] if len(winner_families) == 1 else "mixed_tie"
                ),
                "winner_count": len(winner_indices),
                "winner_families": ";".join(winner_families),
                "winner_policies": ";".join(winner_ids),
                "winning_score": best_score,
                "runner_up_gap": float(larger.min() - best_score) if len(larger) else math.nan,
            })
    return pd.DataFrame(rows).sort_values(
        [*GROUP_COLUMNS, *[column for column in weights.columns if column not in WEIGHT_COLUMNS]],
        kind="stable",
    ).reset_index(drop=True)


def score_weight_sweep(
    candidates: pd.DataFrame,
    step: float = 0.05,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Score every Pareto candidate across the simplex weight grid."""
    return _score_groups(candidates, generate_weight_grid(step), tolerance)


def score_priority_scenarios(
    candidates: pd.DataFrame,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Score the four human-readable operational priority scenarios."""
    weights = pd.DataFrame([
        {
            "scenario": name,
            "weight_sync": values[0],
            "weight_empty": values[1],
            "weight_freshness": values[2],
        }
        for name, values in PRIORITY_SCENARIOS.items()
    ])
    return _score_groups(candidates, weights, tolerance)


def build_two_objective_candidates(
    candidate_metrics: pd.DataFrame,
    normalizations: Iterable[str] = DEFAULT_NORMALIZATIONS,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Prepare independent request/freshness and waste/freshness frontiers."""
    _validate_candidates(candidate_metrics)
    if "family_eligible" in candidate_metrics:
        candidate_metrics = candidate_metrics[
            _family_eligibility(candidate_metrics["family_eligible"])
        ]
        if candidate_metrics.empty:
            raise ValueError("No family-eligible candidates are available")
    modes = tuple(dict.fromkeys(normalizations))
    unknown = sorted(set(modes) - {"raw", "log1p"})
    if not modes or unknown:
        raise ValueError(f"Unsupported freshness normalizations: {unknown}")

    rows: list[dict] = []
    for _, source in candidate_metrics.groupby("unit", sort=True):
        source = source.sort_values(["family", "policy"]).reset_index(drop=True)
        for pair, pair_columns in TWO_OBJECTIVE_PAIRS.items():
            operational_column, freshness_column, freshness_variant = pair_columns
            raw_values = source[[operational_column, freshness_column]].to_numpy()
            frontier = pareto_mask(raw_values, tolerance)
            dominators = [_dominator_indices(raw_values, i, tolerance) for i in range(len(source))]
            for mode in modes:
                freshness = source[freshness_column].to_numpy(dtype=float)
                transformed = np.log1p(freshness) if mode == "log1p" else freshness
                normalized = np.column_stack([
                    minmax_normalize(source[operational_column]),
                    minmax_normalize(transformed),
                ])
                for index, (_, candidate) in enumerate(source.iterrows()):
                    ids = sorted(
                        f"{source.iloc[item]['family']}:{source.iloc[item]['policy']}"
                        for item in dominators[index]
                    )
                    rows.append({
                        **candidate.to_dict(),
                        "objective_pair": pair,
                        "operational_metric": operational_column,
                        "freshness_variant": freshness_variant,
                        "freshness_metric": freshness_column,
                        "normalization": mode,
                        "objective_operational": float(raw_values[index, 0]),
                        "objective_freshness": float(raw_values[index, 1]),
                        "transformed_freshness": float(transformed[index]),
                        "normalized_operational": float(normalized[index, 0]),
                        "normalized_freshness": float(normalized[index, 1]),
                        "is_pareto": bool(frontier[index]),
                        "dominator_count": len(ids),
                        "dominators": ";".join(ids),
                    })
    return pd.DataFrame(rows).sort_values(
        [*TWO_GROUP_COLUMNS, "family", "policy"], kind="stable",
    ).reset_index(drop=True)


def generate_two_objective_weight_grid(step: float = 0.01) -> pd.DataFrame:
    """Generate complementary operational and freshness weights."""
    fraction = Fraction(str(step))
    if fraction <= 0 or fraction > 1:
        raise ValueError("Weight step must be in (0, 1]")
    levels = Fraction(1, 1) / fraction
    if levels.denominator != 1:
        raise ValueError("Weight step must divide one exactly")
    count = levels.numerator
    result = pd.DataFrame({
        "weight_id": np.arange(count + 1, dtype=int),
        "weight_operational": np.arange(count + 1, dtype=float) / count,
    })
    result["weight_freshness"] = 1.0 - result["weight_operational"]
    result["is_interior"] = result["weight_operational"].between(
        0.10 - 1e-12, 0.90 + 1e-12,
    )
    return result


def generate_two_objective_weight_plane(step: float = 0.05) -> pd.DataFrame:
    """Generate a two-axis priority plane and its normalized weight ratios."""
    fraction = Fraction(str(step))
    if fraction <= 0 or fraction > 1:
        raise ValueError("Weight-plane step must be in (0, 1]")
    levels = Fraction(1, 1) / fraction
    if levels.denominator != 1:
        raise ValueError("Weight-plane step must divide one exactly")
    count = levels.numerator
    rows = []
    for operational in range(count + 1):
        for freshness in range(count + 1):
            if operational == 0 and freshness == 0:
                continue
            total = operational + freshness
            rows.append({
                "plane_id": len(rows),
                "operational_priority": operational / count,
                "freshness_priority": freshness / count,
                "weight_operational": operational / total,
                "weight_freshness": freshness / total,
            })
    return pd.DataFrame(rows)


def _score_two_objective_groups(
    candidates: pd.DataFrame,
    weights: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    frontier = candidates[candidates["is_pareto"]]
    for keys, group in frontier.groupby(TWO_GROUP_COLUMNS, sort=True):
        group = group.sort_values(["family", "policy"]).reset_index(drop=True)
        objective_values = group[TWO_SCORE_COLUMNS].to_numpy(dtype=float)
        for _, weight in weights.iterrows():
            weight_values = weight[TWO_WEIGHT_COLUMNS].to_numpy(dtype=float)
            scores = objective_values @ weight_values
            best_score = float(scores.min())
            winner_indices = np.flatnonzero(
                np.isclose(scores, best_score, rtol=0.0, atol=tolerance)
            )
            winner_ids = [
                f"{group.iloc[i]['family']}:{group.iloc[i]['policy']}"
                for i in winner_indices
            ]
            winner_families = sorted({str(group.iloc[i]["family"]) for i in winner_indices})
            larger = scores[scores > best_score + tolerance]
            rows.append({
                **dict(zip(TWO_GROUP_COLUMNS, keys)),
                **weight.to_dict(),
                "winner_family": (
                    winner_families[0] if len(winner_families) == 1 else "mixed_tie"
                ),
                "winner_count": len(winner_indices),
                "winner_families": ";".join(winner_families),
                "winner_policies": ";".join(winner_ids),
                "winning_score": best_score,
                "runner_up_gap": float(larger.min() - best_score) if len(larger) else math.nan,
            })
    sort_extras = [column for column in weights.columns if column not in TWO_WEIGHT_COLUMNS]
    return pd.DataFrame(rows).sort_values(
        [*TWO_GROUP_COLUMNS, *sort_extras], kind="stable",
    ).reset_index(drop=True)


def score_two_objective_weight_sweep(
    candidates: pd.DataFrame,
    step: float = 0.01,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Score both independent objective pairs over complementary weights."""
    return _score_two_objective_groups(
        candidates, generate_two_objective_weight_grid(step), tolerance,
    )


def score_two_objective_weight_plane(
    candidates: pd.DataFrame,
    step: float = 0.05,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Score the redundant but intuitive independent-axis priority plane."""
    return _score_two_objective_groups(
        candidates, generate_two_objective_weight_plane(step), tolerance,
    )


def score_two_objective_scenarios(
    candidates: pd.DataFrame,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Score named freshness, balanced, and operational priorities."""
    weights = pd.DataFrame([
        {
            "scenario": name,
            "weight_operational": operational,
            "weight_freshness": 1.0 - operational,
        }
        for name, operational in TWO_OBJECTIVE_SCENARIOS.items()
    ])
    return _score_two_objective_groups(candidates, weights, tolerance)


def _winner_ids(row: pd.Series) -> list[str]:
    return [value for value in str(row["winner_policies"]).split(";") if value]


def _family_credit(row: pd.Series) -> tuple[float, float]:
    """Split mixed-family ties equally, independently of policy multiplicity."""
    families = {value for value in str(row["winner_families"]).split(";") if value}
    if families == {"RL", "periodic"}:
        return 0.5, 0.5
    return (1.0, 0.0) if families == {"RL"} else (0.0, 1.0)


def summarize_family_shares(sweep: pd.DataFrame) -> pd.DataFrame:
    """Summarize exclusive outcomes and fractional family credit."""
    rows = []
    for scope, scoped in (
        ("full", sweep),
        ("interior", sweep[sweep["is_interior"]]),
    ):
        for keys, group in scoped.groupby(GROUP_COLUMNS, sort=True):
            count = len(group)
            rl_credit = periodic_credit = 0.0
            for _, winner in group.iterrows():
                rl_share, periodic_share = _family_credit(winner)
                rl_credit += rl_share
                periodic_credit += periodic_share
            outcomes = group["winner_family"].value_counts()
            rows.append({
                **dict(zip(GROUP_COLUMNS, keys)),
                "grid_scope": scope,
                "combination_count": count,
                "rl_fractional_share": rl_credit / count,
                "periodic_fractional_share": periodic_credit / count,
                "rl_only_share": outcomes.get("RL", 0) / count,
                "periodic_only_share": outcomes.get("periodic", 0) / count,
                "mixed_tie_share": outcomes.get("mixed_tie", 0) / count,
            })
    return pd.DataFrame(rows).sort_values(
        [*GROUP_COLUMNS, "grid_scope"], kind="stable",
    ).reset_index(drop=True)


def summarize_policy_shares(sweep: pd.DataFrame) -> pd.DataFrame:
    """Give tied policies equal credit within full and interior grids."""
    rows = []
    for scope, scoped in (
        ("full", sweep),
        ("interior", sweep[sweep["is_interior"]]),
    ):
        for keys, group in scoped.groupby(GROUP_COLUMNS, sort=True):
            policy_credits: dict[str, float] = {}
            for _, winner in group.iterrows():
                ids = _winner_ids(winner)
                for candidate_id in ids:
                    policy_credits[candidate_id] = (
                        policy_credits.get(candidate_id, 0.0) + 1 / len(ids)
                    )
            for candidate_id, credit in sorted(policy_credits.items()):
                family, policy = candidate_id.split(":", 1)
                rows.append({
                    **dict(zip(GROUP_COLUMNS, keys)),
                    "grid_scope": scope,
                    "family": family,
                    "policy": policy,
                    "fractional_wins": credit,
                    "weight_space_share": credit / len(group),
                })
    return pd.DataFrame(rows).sort_values(
        [*GROUP_COLUMNS, "grid_scope", "family", "policy"], kind="stable",
    ).reset_index(drop=True)


def summarize_two_objective_family_shares(sweep: pd.DataFrame) -> pd.DataFrame:
    """Summarize family outcomes over complete and non-extreme line sweeps."""
    rows = []
    for scope, scoped in (
        ("full", sweep),
        ("interior", sweep[sweep["is_interior"]]),
    ):
        for keys, group in scoped.groupby(TWO_GROUP_COLUMNS, sort=True):
            count = len(group)
            rl_credit = periodic_credit = 0.0
            for _, winner in group.iterrows():
                rl_share, periodic_share = _family_credit(winner)
                rl_credit += rl_share
                periodic_credit += periodic_share
            outcomes = group["winner_family"].value_counts()
            rows.append({
                **dict(zip(TWO_GROUP_COLUMNS, keys)),
                "grid_scope": scope,
                "combination_count": count,
                "rl_fractional_share": rl_credit / count,
                "periodic_fractional_share": periodic_credit / count,
                "rl_only_share": outcomes.get("RL", 0) / count,
                "periodic_only_share": outcomes.get("periodic", 0) / count,
                "mixed_tie_share": outcomes.get("mixed_tie", 0) / count,
            })
    return pd.DataFrame(rows).sort_values(
        [*TWO_GROUP_COLUMNS, "grid_scope"], kind="stable",
    ).reset_index(drop=True)


def summarize_two_objective_policy_shares(sweep: pd.DataFrame) -> pd.DataFrame:
    """Give tied policies equal credit over both two-objective sweep scopes."""
    rows = []
    for scope, scoped in (
        ("full", sweep),
        ("interior", sweep[sweep["is_interior"]]),
    ):
        for keys, group in scoped.groupby(TWO_GROUP_COLUMNS, sort=True):
            policy_credits: dict[str, float] = {}
            for _, winner in group.iterrows():
                ids = _winner_ids(winner)
                for candidate_id in ids:
                    policy_credits[candidate_id] = (
                        policy_credits.get(candidate_id, 0.0) + 1 / len(ids)
                    )
            for candidate_id, credit in sorted(policy_credits.items()):
                family, policy = candidate_id.split(":", 1)
                rows.append({
                    **dict(zip(TWO_GROUP_COLUMNS, keys)),
                    "grid_scope": scope,
                    "family": family,
                    "policy": policy,
                    "fractional_wins": credit,
                    "weight_space_share": credit / len(group),
                })
    return pd.DataFrame(rows).sort_values(
        [*TWO_GROUP_COLUMNS, "grid_scope", "family", "policy"], kind="stable",
    ).reset_index(drop=True)


def summarize_two_objective_winner_ranges(sweep: pd.DataFrame) -> pd.DataFrame:
    """Collapse consecutive sampled weights with identical policy winners."""
    rows = []
    for keys, group in sweep.groupby(TWO_GROUP_COLUMNS, sort=True):
        group = group.sort_values("weight_operational").reset_index(drop=True)
        signature = group["winner_family"].astype(str) + "|" + group["winner_policies"].astype(str)
        run_ids = signature.ne(signature.shift()).cumsum()
        for range_index, (_, run) in enumerate(group.groupby(run_ids, sort=False)):
            first = run.iloc[0]
            rows.append({
                **dict(zip(TWO_GROUP_COLUMNS, keys)),
                "range_index": range_index,
                "operational_weight_min": float(run["weight_operational"].min()),
                "operational_weight_max": float(run["weight_operational"].max()),
                "freshness_weight_min": float(run["weight_freshness"].min()),
                "freshness_weight_max": float(run["weight_freshness"].max()),
                "sampled_weight_count": len(run),
                "winner_family": first["winner_family"],
                "winner_count": int(first["winner_count"]),
                "winner_families": first["winner_families"],
                "winner_policies": first["winner_policies"],
            })
    return pd.DataFrame(rows).sort_values(
        [*TWO_GROUP_COLUMNS, "range_index"], kind="stable",
    ).reset_index(drop=True)


def summarize_two_objective_normalization_bounds(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Export candidate-pool bounds and the policies that define each bound."""
    rows = []
    for keys, group in candidates.groupby(TWO_GROUP_COLUMNS, sort=True):
        row = dict(zip(TWO_GROUP_COLUMNS, keys))
        for label, column in (
            ("operational", "objective_operational"),
            ("freshness", "objective_freshness"),
            ("transformed_freshness", "transformed_freshness"),
        ):
            values = group[column]
            minimum, maximum = float(values.min()), float(values.max())
            row[f"{label}_min"] = minimum
            row[f"{label}_max"] = maximum
            for suffix, target in (("min", minimum), ("max", maximum)):
                selected = group[np.isclose(values, target, rtol=0.0, atol=1e-12)]
                row[f"{label}_{suffix}_candidates"] = ";".join(sorted(
                    f"{item.family}:{item.policy}" for item in selected.itertuples()
                ))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(TWO_GROUP_COLUMNS, kind="stable").reset_index(
        drop=True,
    )


def build_multiobjective_results(
    candidate_metrics: pd.DataFrame,
    weight_step: float = 0.05,
    normalizations: Iterable[str] = DEFAULT_NORMALIZATIONS,
    tolerance: float = 1e-12,
) -> dict[str, pd.DataFrame]:
    """Build all reusable multi-objective evidence tables."""
    candidates = build_candidate_objectives(candidate_metrics, normalizations, tolerance)
    frontier = candidates[candidates["is_pareto"]].reset_index(drop=True)
    sweep = score_weight_sweep(candidates, weight_step, tolerance)
    scenarios = score_priority_scenarios(candidates, tolerance)
    return {
        "multiobjective_candidates": candidates,
        "pareto_frontiers": frontier,
        "weight_sweep_winners": sweep,
        "priority_scenario_winners": scenarios,
        "weight_space_family_share": summarize_family_shares(sweep),
        "weight_space_policy_share": summarize_policy_shares(sweep),
    }


def build_two_objective_results(
    candidate_metrics: pd.DataFrame,
    weight_step: float = 0.01,
    normalizations: Iterable[str] = DEFAULT_NORMALIZATIONS,
    tolerance: float = 1e-12,
    plane_step: float = 0.05,
) -> dict[str, pd.DataFrame]:
    """Build all evidence tables for the two independent objective pairs."""
    candidates = build_two_objective_candidates(
        candidate_metrics, normalizations, tolerance,
    )
    frontier = candidates[candidates["is_pareto"]].reset_index(drop=True)
    sweep = score_two_objective_weight_sweep(candidates, weight_step, tolerance)
    return {
        "two_objective_candidates": candidates,
        "two_objective_pareto_frontiers": frontier,
        "two_objective_weight_sweep_winners": sweep,
        "two_objective_weight_plane_winners": score_two_objective_weight_plane(
            candidates, plane_step, tolerance,
        ),
        "two_objective_priority_scenario_winners": score_two_objective_scenarios(
            candidates, tolerance,
        ),
        "two_objective_weight_space_family_share": (
            summarize_two_objective_family_shares(sweep)
        ),
        "two_objective_weight_space_policy_share": (
            summarize_two_objective_policy_shares(sweep)
        ),
        "two_objective_winner_ranges": summarize_two_objective_winner_ranges(sweep),
        "two_objective_normalization_bounds": (
            summarize_two_objective_normalization_bounds(candidates)
        ),
    }


def write_multiobjective_outputs(
    results: dict[str, pd.DataFrame], output_dir: Path,
) -> None:
    """Write every evidence table using stable CSV names."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)


def write_weight_triangle_html(sweep: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write one interactive family-winner weight triangle per normalization."""
    import plotly.express as px  # pylint: disable=import-error,import-outside-toplevel

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for normalization, frame in sweep.groupby("normalization", sort=True):
        plot = frame.copy()
        plot["triangle_x"] = plot["weight_empty"] + 0.5 * plot["weight_freshness"]
        plot["triangle_y"] = (math.sqrt(3) / 2) * plot["weight_freshness"]
        figure = px.scatter(
            plot, x="triangle_x", y="triangle_y", color="winner_family",
            facet_row="unit", facet_col="freshness_variant",
            hover_data=[*WEIGHT_COLUMNS, "winner_policies", "winning_score"],
            category_orders={"winner_family": ["RL", "periodic", "mixed_tie"]},
            title=(
                f"Policy-family winner across objective weights ({normalization})"
                "<br><sup>Corners: synchronization (left), empty-sync (right), "
                "freshness (top)</sup>"
            ),
        )
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
        figure.update_layout(height=300 * plot["unit"].nunique())
        safe_name = re.sub(r"[^a-z0-9]+", "_", str(normalization).lower()).strip("_")
        path = output_dir / f"weight_triangle_{safe_name}.html"
        figure.write_html(path, include_plotlyjs="cdn")
        paths.append(path)
    return paths


def write_two_objective_html(
    candidates: pd.DataFrame,
    sweep: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Write readable Pareto scatters and family-winner strips."""
    import plotly.express as px  # pylint: disable=import-error,import-outside-toplevel

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for keys, candidate_group in candidates.groupby(
        ["objective_pair", "normalization"], sort=True,
    ):
        pair, normalization = keys
        pair_label = (
            "Synchronization rate vs CVaR95 AoII"
            if pair == "sync_vs_aoii"
            else "Empty synchronizations per step vs CVaR95 AoII"
        )
        figure = px.scatter(
            candidate_group,
            x="normalized_operational",
            y="normalized_freshness",
            color="family",
            symbol="is_pareto",
            facet_col="unit",
            facet_col_wrap=2,
            hover_data=[
                "policy", "objective_operational", "objective_freshness",
                "dominator_count",
            ],
            category_orders={"family": ["RL", "periodic"]},
            title=f"{pair_label} Pareto candidates ({normalization})",
        )
        figure.update_xaxes(title="Normalized operational cost")
        figure.update_yaxes(title="Normalized freshness risk")
        figure.update_layout(height=320 * math.ceil(candidate_group["unit"].nunique() / 2))
        stem = f"two_objective_pareto_{pair}_{normalization}.html"
        path = output_dir / stem
        figure.write_html(path, include_plotlyjs="cdn")
        paths.append(path)

        sweep_group = sweep[
            (sweep["objective_pair"] == pair)
            & (sweep["normalization"] == normalization)
        ]
        strip = px.scatter(
            sweep_group,
            x="weight_operational",
            y="unit",
            color="winner_family",
            hover_data=["weight_freshness", "winner_policies", "winning_score"],
            category_orders={"winner_family": ["RL", "periodic", "mixed_tie"]},
            title=f"{pair_label}: winner by operational weight ({normalization})",
        )
        strip.update_traces(marker={"size": 9, "symbol": "square"})
        strip.update_xaxes(title="Operational-cost weight", range=[-0.01, 1.01])
        strip.update_yaxes(title=None)
        strip.update_layout(height=100 + 65 * sweep_group["unit"].nunique())
        strip_path = output_dir / f"two_objective_winner_strip_{pair}_{normalization}.html"
        strip.write_html(strip_path, include_plotlyjs="cdn")
        paths.append(strip_path)
    return paths


def write_two_objective_weight_plane_html(
    plane: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Write one eight-panel relative-priority winner map per analysis variant."""
    import plotly.express as px  # pylint: disable=import-error,import-outside-toplevel

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for keys, group in plane.groupby(["objective_pair", "normalization"], sort=True):
        pair, normalization = keys
        operational_label = (
            "Total-request priority"
            if pair == "sync_vs_aoii"
            else "Empty-request priority"
        )
        figure = px.scatter(
            group,
            x="operational_priority",
            y="freshness_priority",
            color="winner_family",
            facet_col="unit",
            facet_col_wrap=2,
            hover_data={
                "winner_policies": True,
                "weight_operational": ":.3f",
                "weight_freshness": ":.3f",
                "winning_score": ":.5f",
                "operational_priority": ":.2f",
                "freshness_priority": ":.2f",
            },
            category_orders={"winner_family": ["RL", "periodic", "mixed_tie"]},
            color_discrete_map={
                "RL": "#2a9d8f",
                "periodic": "#e76f51",
                "mixed_tie": "#8d99ae",
            },
            title=(
                f"Policy winner by relative priority ({pair}, {normalization})"
                "<br><sup>Hover for policy; axes are normalized by their sum, "
                "so points on the same ray are equivalent.</sup>"
            ),
        )
        figure.update_traces(marker={"size": 12, "symbol": "square"})
        figure.update_xaxes(title=operational_label, range=[-0.03, 1.03])
        figure.update_yaxes(title="Freshness priority", range=[-0.03, 1.03])
        figure.update_layout(
            width=1100,
            height=340 * math.ceil(group["unit"].nunique() / 2),
        )
        path = output_dir / f"two_objective_weight_plane_{pair}_{normalization}.html"
        figure.write_html(path, include_plotlyjs="cdn")
        paths.append(path)
    return paths


def parse_normalizations(raw: str) -> tuple[str, ...]:
    """Parse a comma-separated normalization list for argparse."""
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one normalization is required")
    return values


def main() -> None:
    """Run multi-objective analysis from an existing candidate-metrics CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("multiobjective_results"))
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--two-objective-weight-step", type=float, default=0.01)
    parser.add_argument("--two-objective-plane-step", type=float, default=0.05)
    parser.add_argument(
        "--normalizations", type=parse_normalizations,
        default=DEFAULT_NORMALIZATIONS,
    )
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    candidate_metrics = pd.read_csv(
        args.candidate_metrics, float_precision="round_trip",
    )
    results = build_multiobjective_results(
        candidate_metrics, args.weight_step, args.normalizations,
    )
    results.update(build_two_objective_results(
        candidate_metrics,
        args.two_objective_weight_step,
        args.normalizations,
        plane_step=args.two_objective_plane_step,
    ))
    write_multiobjective_outputs(results, args.output_dir)
    if args.plots:
        write_weight_triangle_html(results["weight_sweep_winners"], args.output_dir)
        write_two_objective_html(
            results["two_objective_candidates"],
            results["two_objective_weight_sweep_winners"],
            args.output_dir,
        )
        write_two_objective_weight_plane_html(
            results["two_objective_weight_plane_winners"], args.output_dir,
        )
    print("Pareto candidates:", len(results["pareto_frontiers"]))
    print("Weight-sweep rows:", len(results["weight_sweep_winners"]))
    print("Two-objective Pareto candidates:", len(results["two_objective_pareto_frontiers"]))
    print("Two-objective weight-sweep rows:", len(results["two_objective_weight_sweep_winners"]))


if __name__ == "__main__":
    main()
