#!/usr/bin/env python3
"""Authoritative trace-based comparison of RL and periodic sync policies.

The script evaluates frozen, matched-window ``decisions.parquet`` traces. It
keeps the canonical standardized reward separate from the operational objective

    J(T, C) = mean(max(0, true_aoi - T)) + C * sync_rate.

Post-hoc scoring never changes a policy's recorded actions.
"""

# pylint: disable=import-error,too-many-instance-attributes,too-many-locals
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-branches,too-many-statements
# pylint: disable=wrong-import-position

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from rl_training.reward import RewardConfig
from analysis.multiobjective_policy_analysis import (
    DEFAULT_NORMALIZATIONS,
    build_multiobjective_results,
    build_two_objective_results,
    parse_normalizations,
    write_two_objective_html,
    write_two_objective_weight_plane_html,
    write_weight_triangle_html,
)
from sync_policy_evaluation.decision_metrics import (
    _detect_events_array,
    compute_aoii_series,
)


DEFAULT_THRESHOLDS = (0.0, 1.0, 5.0, 10.0, 30.0, 60.0)
DEFAULT_COSTS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
DEFAULT_PERIODIC_INTERVALS = (1, 2, 5, 10, 15, 20, 30, 45, 60)
REQUIRED_COLUMNS = ("timestamp", "action", "actual_value", "true_aoi")


@dataclass(frozen=True)
class CandidateRef:
    """One explicitly inventoried policy trace."""

    policy: str
    family: str
    path: Path


@dataclass(frozen=True)
class UnitSpec:
    """Manifest configuration for one synchronization unit."""

    name: str
    expected_steps: int
    expected_margin_pct: float
    fixed_rl: CandidateRef
    fixed_periodic: CandidateRef
    family_rl: tuple[CandidateRef, ...]
    family_periodic: tuple[CandidateRef, ...]


@dataclass(frozen=True)
class TraceMetrics:
    """Policy behavior and sufficient statistics from one frozen trace."""

    steps: int
    timestamp_digest: str
    event_digest: str
    start_timestamp: str
    end_timestamp: str
    initial_true_aoi: float
    sync_count: int
    successful_sync_count: int
    empty_sync_count: int
    late_skip_count: int
    true_aoi_sum: float
    avg_true_aoi: float
    max_true_aoi: float
    p95_true_aoi: float
    p95_true_aoi_when_stale: float
    cvar95_true_aoi: float
    avg_aoii: float
    max_aoii: float
    p95_aoii: float
    cvar95_aoii: float
    avg_reaction_time: float
    p95_reaction_time: float
    avg_items_per_meaningful_sync: float
    mean_excess_aoi: dict[float, float] = field(repr=False)
    aoi_exceedance_rate: dict[float, float] = field(repr=False)

    @property
    def sync_rate(self) -> float:
        """Fraction of evaluation minutes containing a synchronization."""
        return self.sync_count / self.steps

    @property
    def empty_sync_rate(self) -> float:
        """Fraction of synchronizations that returned no pending items."""
        return self.empty_sync_count / self.sync_count if self.sync_count else 0.0

    @property
    def empty_sync_step_rate(self) -> float:
        """Fraction of evaluation minutes containing an empty synchronization."""
        return self.empty_sync_count / self.steps

    def standard_reward_mean(self) -> float:
        """Reconstruct the canonical event-rate evaluation reward."""
        config = RewardConfig.standard_eval(mode="event_rate")
        total = (
            (-config.sync_cost + config.successful_sync_reward)
            * self.successful_sync_count
            - config.empty_sync_penalty * self.empty_sync_count
            - config.late_penalty * self.late_skip_count
            - config.aoi_penalty_per_minute * self.true_aoi_sum
        )
        return float(total / self.steps)

    def operational_score(self, threshold: float, sync_cost: float) -> float:
        """Return J(T,C), where every sync incurs the same cost."""
        return self.mean_excess_aoi[float(threshold)] + sync_cost * self.sync_rate

    def flat_dict(self) -> dict[str, Any]:
        """Return scalar fields suitable for one CSV row."""
        result = asdict(self)
        result.pop("mean_excess_aoi")
        result.pop("aoi_exceedance_rate")
        result["sync_rate"] = self.sync_rate
        result["empty_sync_rate"] = self.empty_sync_rate
        result["empty_sync_step_rate"] = self.empty_sync_step_rate
        result["standard_reward_mean"] = self.standard_reward_mean()
        return result


@dataclass(frozen=True)
class BreakEvenResult:
    """Fixed-pair operational switching point for one AoI threshold."""

    threshold: float
    break_even_cost: float | None
    winner_below: str
    winner_above: str
    mean_excess_aoi_difference: float
    sync_rate_difference: float


def _require_columns(frame: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Decision trace lacks required columns: {missing}")


def _series_digest(values: pd.Series | np.ndarray) -> str:
    """Return a stable digest for one ordered trace input series."""
    hashes = pd.util.hash_pandas_object(values, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def empirical_cvar(values: np.ndarray, alpha: float = 0.95) -> float:
    """Return the mean of the fixed-size empirical upper tail."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("CVaR requires a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("CVaR values must be finite")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("CVaR alpha must be in [0, 1)")
    tail_count = max(1, math.ceil((1.0 - alpha) * len(array) - 1e-12))
    tail = np.partition(array, len(array) - tail_count)[-tail_count:]
    return float(tail.mean())


def derive_sync_items(
    actual_values: np.ndarray,
    actions: np.ndarray,
    task_type: str = "event_rate",
) -> np.ndarray:
    """Derive items returned by each sync from events and frozen actions."""
    events = _detect_events_array(actual_values, task_type)
    sync_items = np.zeros(len(actions), dtype=np.float64)
    pending = 0.0
    for index, did_event in enumerate(events):
        if did_event:
            pending += actual_values[index] if task_type == "event_rate" else 1.0
        if actions[index] == 1:
            sync_items[index] = pending
            pending = 0.0
    return sync_items


def _reaction_times(
    actual_values: np.ndarray,
    actions: np.ndarray,
    task_type: str,
) -> np.ndarray:
    event_steps = np.flatnonzero(_detect_events_array(actual_values, task_type))
    sync_steps = np.flatnonzero(actions == 1)
    if len(event_steps) == 0:
        return np.array([], dtype=np.float64)
    positions = np.searchsorted(sync_steps, event_steps, side="left")
    result = np.empty(len(event_steps), dtype=np.float64)
    for index, event_step in enumerate(event_steps):
        result[index] = (
            sync_steps[positions[index]] - event_step
            if positions[index] < len(sync_steps)
            else len(actions) - event_step
        )
    return result


def summarize_decisions(
    frame: pd.DataFrame,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    task_type: str = "event_rate",
) -> TraceMetrics:
    """Compute reward, freshness, tail, efficiency, and AoII statistics."""
    _require_columns(frame)
    if frame.empty:
        raise ValueError("Decision trace is empty")
    actions = frame["action"].to_numpy(dtype=np.int8)
    if not np.isin(actions, [0, 1]).all():
        raise ValueError("Decision actions must be binary")
    actual = frame["actual_value"].to_numpy(dtype=np.float64)
    true_aoi = frame["true_aoi"].to_numpy(dtype=np.float64)
    if not np.isfinite(actual).all() or not np.isfinite(true_aoi).all():
        raise ValueError("Decision values and true AoI must be finite")
    if (true_aoi < 0).any():
        raise ValueError("True AoI cannot be negative")

    true_aoi = np.where(actions == 1, 0.0, true_aoi)
    sync_items = derive_sync_items(actual, actions, task_type)
    sync_mask = actions == 1
    successful = np.logical_and(sync_mask, sync_items > 0)
    empty = np.logical_and(sync_mask, sync_items == 0)
    events = _detect_events_array(actual, task_type)
    late = np.logical_and(actions == 0, events)
    aoii = compute_aoii_series(actual, actions, task_type)
    reactions = _reaction_times(actual, actions, task_type)
    stale_aoi = true_aoi[true_aoi > 0]
    meaningful_items = sync_items[successful]

    normalized_thresholds = tuple(dict.fromkeys(float(value) for value in thresholds))
    mean_excess = {}
    for value in normalized_thresholds:
        stale_above_threshold = true_aoi[true_aoi > value]
        mean_excess[value] = float(
            (stale_above_threshold - value).sum() / len(true_aoi)
        )
    exceedance = {
        value: float((true_aoi > value).mean()) for value in normalized_thresholds
    }
    return TraceMetrics(
        steps=len(frame),
        timestamp_digest=_series_digest(frame["timestamp"]),
        event_digest=_series_digest(frame["actual_value"]),
        start_timestamp=str(frame["timestamp"].iloc[0]),
        end_timestamp=str(frame["timestamp"].iloc[-1]),
        initial_true_aoi=float(true_aoi[0]),
        sync_count=int(sync_mask.sum()),
        successful_sync_count=int(successful.sum()),
        empty_sync_count=int(empty.sum()),
        late_skip_count=int(late.sum()),
        true_aoi_sum=float(true_aoi.sum()),
        avg_true_aoi=float(true_aoi.mean()),
        max_true_aoi=float(true_aoi.max()),
        p95_true_aoi=float(np.percentile(true_aoi, 95)),
        p95_true_aoi_when_stale=(
            float(np.percentile(stale_aoi, 95)) if len(stale_aoi) else float("nan")
        ),
        cvar95_true_aoi=empirical_cvar(true_aoi),
        avg_aoii=float(aoii.mean()),
        max_aoii=float(aoii.max()),
        p95_aoii=float(np.percentile(aoii, 95)),
        cvar95_aoii=empirical_cvar(aoii),
        avg_reaction_time=float(reactions.mean()) if len(reactions) else 0.0,
        p95_reaction_time=float(np.percentile(reactions, 95)) if len(reactions) else 0.0,
        avg_items_per_meaningful_sync=(
            float(meaningful_items.mean()) if len(meaningful_items) else 0.0
        ),
        mean_excess_aoi=mean_excess,
        aoi_exceedance_rate=exceedance,
    )


def load_trace_metrics(
    path: Path,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> TraceMetrics:
    """Read only required parquet columns and summarize the trace."""
    frame = pd.read_parquet(path, columns=list(REQUIRED_COLUMNS))
    return summarize_decisions(frame, thresholds=thresholds)


def validate_same_window(metrics: Iterable[TraceMetrics], expected_steps: int) -> None:
    """Require identical windows, event inputs, and initial freshness state."""
    items = tuple(metrics)
    if not items:
        raise ValueError("No traces supplied for window validation")
    lengths = {item.steps for item in items}
    digests = {item.timestamp_digest for item in items}
    if lengths != {expected_steps}:
        raise ValueError(f"Expected {expected_steps} steps, found {sorted(lengths)}")
    if len(digests) != 1:
        raise ValueError("Compared traces do not have identical timestamps")
    event_digests = {item.event_digest for item in items}
    if len(event_digests) != 1:
        raise ValueError("Compared traces do not have identical event sequences")
    initial_states = {item.initial_true_aoi for item in items}
    if len(initial_states) != 1:
        raise ValueError("Compared traces do not have identical initial true AoI")


def _winner(difference: float, tolerance: float = 1e-12) -> str:
    if difference < -tolerance:
        return "RL"
    if difference > tolerance:
        return "periodic"
    return "tie"


def fixed_pair_break_even(
    rl: TraceMetrics,
    periodic: TraceMetrics,
    threshold: float,
) -> BreakEvenResult:
    """Find C where fixed RL and periodic traces have equal J(T,C)."""
    threshold = float(threshold)
    excess_difference = rl.mean_excess_aoi[threshold] - periodic.mean_excess_aoi[threshold]
    rate_difference = rl.sync_rate - periodic.sync_rate
    if abs(rate_difference) <= 1e-12:
        winner = _winner(excess_difference)
        return BreakEvenResult(
            threshold, None, winner, winner, excess_difference, rate_difference,
        )

    crossing = -excess_difference / rate_difference
    if crossing > 1e-12:
        below = _winner(excess_difference + 0.5 * crossing * rate_difference)
        above_cost = crossing + max(1.0, crossing)
        above = _winner(excess_difference + above_cost * rate_difference)
        return BreakEvenResult(
            threshold, float(crossing), below, above,
            excess_difference, rate_difference,
        )

    positive_winner = _winner(excess_difference + rate_difference)
    return BreakEvenResult(
        threshold, None, positive_winner, positive_winner,
        excess_difference, rate_difference,
    )


def _candidate_from_row(row: pd.Series, code_root: Path) -> CandidateRef:
    return CandidateRef(
        policy=str(row["policy"]), family=str(row["family"]),
        path=(code_root / str(row["decisions_path"])).resolve(),
    )


def _find_candidate(
    rows: pd.DataFrame,
    unit_name: str,
    policy: str,
    family: str,
    code_root: Path,
) -> CandidateRef:
    """Resolve one unique inventory row to an explicit candidate reference."""
    matches = rows[(rows["policy"] == policy) & (rows["family"] == family)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {unit_name} {family}/{policy}, found {len(matches)}"
        )
    return _candidate_from_row(matches.iloc[0], code_root)


def _eligible_rl_rows(rows: pd.DataFrame, exclusions: list[re.Pattern]) -> pd.DataFrame:
    """Prefer frozen eligibility decisions, with legacy name filtering as fallback."""
    rl_rows = rows[rows["family"] == "RL"]
    if "family_eligible" in rl_rows.columns:
        eligibility = rl_rows["family_eligible"].map(
            lambda value: str(value).strip().lower() == "true",
        )
        return rl_rows[eligibility]
    included = ~rl_rows["policy"].map(
        lambda value: any(pattern.search(str(value)) for pattern in exclusions),
    )
    return rl_rows[included]


def _validate_file_digest(path: Path, expected: str | None, label: str) -> None:
    """Reject a frozen input whose bytes no longer match its manifest digest."""
    if not expected:
        return
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def load_manifest(path: Path) -> tuple[dict[str, Any], tuple[UnitSpec, ...]]:
    """Load explicit inventory paths and fixed-pair selections."""
    path = path.resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    code_root = (path.parent / raw["code_root"]).resolve()
    inventory_path = (path.parent / raw["candidate_inventory_csv"]).resolve()
    _validate_file_digest(
        inventory_path, raw.get("candidate_inventory_sha256"), "Candidate inventory",
    )
    inventory = pd.read_csv(inventory_path)
    periodic_records = None
    if "periodic_decisions_manifest" in raw:
        periodic_path = (path.parent / raw["periodic_decisions_manifest"]).resolve()
        periodic_records = pd.DataFrame(
            json.loads(periodic_path.read_text(encoding="utf-8"))
        )
    exclusions = [re.compile(value) for value in raw.get("family_exclude_patterns", [])]
    units = []
    for configured in raw["units"]:
        rows = inventory[inventory["unit"] == configured["name"]]
        if rows.empty:
            raise ValueError(f"No inventory rows for {configured['name']}")

        include_rl = [
            _candidate_from_row(row, code_root)
            for _, row in _eligible_rl_rows(rows, exclusions).iterrows()
        ]
        intervals = configured.get(
            "periodic_intervals",
            raw.get("periodic_intervals", DEFAULT_PERIODIC_INTERVALS),
        )
        periodic_names = [f"periodic_{int(value)}min" for value in intervals]
        periodic_rows = rows[rows["family"] == "periodic"]
        if periodic_records is not None:
            selected = periodic_records[
                (periodic_records["unit_path"] == configured["unit_path"])
                & (periodic_records["matched_steps"] == configured["expected_steps"])
                & (periodic_records["interval_minutes"].isin(intervals))
            ].copy()
            selected["unit"] = configured["name"]
            selected["family"] = "periodic"
            selected["policy"] = selected["interval_minutes"].map(
                lambda value: f"periodic_{int(value)}min"
            )
            selected["decisions_path"] = selected["decisions"]
            periodic_rows = selected
        units.append(UnitSpec(
            name=configured["name"],
            expected_steps=int(configured["expected_steps"]),
            expected_margin_pct=float(configured["expected_margin_pct"]),
            fixed_rl=_find_candidate(
                rows, configured["name"], configured["fixed_pair"]["rl"],
                "RL", code_root,
            ),
            fixed_periodic=_find_candidate(
                periodic_rows, configured["name"],
                configured["fixed_pair"]["periodic"],
                "periodic", code_root,
            ),
            family_rl=tuple(include_rl),
            family_periodic=tuple(
                _find_candidate(
                    periodic_rows, configured["name"], name, "periodic", code_root,
                )
                for name in periodic_names
            ),
        ))
    return raw, tuple(units)


def _comparison_category(
    winner: str,
    rl: TraceMetrics,
    periodic: TraceMetrics,
    threshold: float,
) -> str:
    if winner != "RL":
        return "tie" if winner == "tie" else "periodic_wins"
    rl_excess = rl.mean_excess_aoi[threshold]
    periodic_excess = periodic.mean_excess_aoi[threshold]
    if rl.sync_rate <= periodic.sync_rate and rl_excess <= periodic_excess:
        return "rl_dominates"
    if rl.sync_rate > periodic.sync_rate:
        return "rl_fresher_more_syncs"
    return "rl_fewer_syncs_more_excess_aoi"


def _best_candidate(
    candidates: Iterable[CandidateRef],
    metric_loader,
    threshold: float,
    sync_cost: float,
) -> CandidateRef:
    """Select the lowest operational-cost candidate with a stable tie-break."""
    scored = (
        (
            metric_loader(candidate).operational_score(threshold, sync_cost),
            candidate.policy,
            candidate,
        )
        for candidate in candidates
    )
    return min(scored, key=lambda item: (item[0], item[1]))[2]


def _least_sync_candidate(
    candidates: Iterable[CandidateRef],
    metric_loader,
    p95_boundary: float | None,
) -> CandidateRef | None:
    """Select the least-request candidate satisfying an optional P95 AoI boundary."""
    eligible = [
        candidate for candidate in candidates
        if p95_boundary is None
        or metric_loader(candidate).p95_true_aoi <= p95_boundary
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda candidate: (
        metric_loader(candidate).sync_count,
        metric_loader(candidate).empty_sync_count,
        metric_loader(candidate).p95_true_aoi,
        candidate.policy,
    ))


def run_analysis(
    units: Iterable[UnitSpec],
    thresholds: Iterable[float],
    costs: Iterable[float],
    include_family: bool = True,
    weight_step: float = 0.05,
    two_objective_weight_step: float = 0.01,
    freshness_normalizations: Iterable[str] = DEFAULT_NORMALIZATIONS,
    two_objective_plane_step: float = 0.05,
) -> dict[str, pd.DataFrame]:
    """Load traces, validate windows, and construct auditable result tables."""
    thresholds = tuple(float(value) for value in thresholds)
    costs = tuple(float(value) for value in costs)
    metric_cache: dict[Path, TraceMetrics] = {}

    def metrics(candidate: CandidateRef) -> TraceMetrics:
        if candidate.path not in metric_cache:
            if not candidate.path.exists():
                raise FileNotFoundError(candidate.path)
            metric_cache[candidate.path] = load_trace_metrics(candidate.path, thresholds)
        return metric_cache[candidate.path]

    fixed_rows: list[dict[str, Any]] = []
    break_even_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    constrained_rows: list[dict[str, Any]] = []

    for unit in units:
        candidates = [unit.fixed_rl, unit.fixed_periodic]
        if include_family:
            candidates.extend(unit.family_rl)
            candidates.extend(unit.family_periodic)
        unique = {candidate.path: candidate for candidate in candidates}
        family_paths = {
            candidate.path for candidate in (*unit.family_rl, *unit.family_periodic)
        }
        loaded = {path: metrics(candidate) for path, candidate in unique.items()}
        validate_same_window(loaded.values(), unit.expected_steps)

        rl = metrics(unit.fixed_rl)
        periodic = metrics(unit.fixed_periodic)
        rl_reward = rl.standard_reward_mean()
        periodic_reward = periodic.standard_reward_mean()
        margin = 100.0 * (rl_reward - periodic_reward) / abs(periodic_reward)
        if abs(margin - unit.expected_margin_pct) > 0.15:
            raise ValueError(
                f"{unit.name} canonical margin {margin:.3f}% does not reproduce "
                f"expected {unit.expected_margin_pct:.3f}%"
            )
        row = {
            "unit": unit.name,
            "rl_policy": unit.fixed_rl.policy,
            "periodic_policy": unit.fixed_periodic.policy,
            "rl_standard_reward_per_step": rl_reward,
            "periodic_standard_reward_per_step": periodic_reward,
            "absolute_reward_difference": rl_reward - periodic_reward,
            "reward_difference_pct": margin,
        }
        for prefix, item in (("rl", rl), ("periodic", periodic)):
            for key, value in item.flat_dict().items():
                row[f"{prefix}_{key}"] = value
        fixed_rows.append(row)
        for threshold in thresholds:
            result = fixed_pair_break_even(rl, periodic, threshold)
            break_even_rows.append({"unit": unit.name, **asdict(result)})

        for candidate in unique.values():
            item = metrics(candidate)
            candidate_rows.append({
                "unit": unit.name, "family": candidate.family,
                "policy": candidate.policy, "decisions_path": str(candidate.path),
                "family_eligible": candidate.path in family_paths,
                **item.flat_dict(),
            })
            for threshold in thresholds:
                threshold_rows.append({
                    "unit": unit.name, "family": candidate.family,
                    "policy": candidate.policy, "threshold": threshold,
                    "mean_excess_aoi": item.mean_excess_aoi[threshold],
                    "aoi_exceedance_rate": item.aoi_exceedance_rate[threshold],
                })

        if include_family:
            for boundary in (*thresholds, None):
                for family, family_candidates in (
                    ("RL", unit.family_rl),
                    ("periodic", unit.family_periodic),
                ):
                    selected = _least_sync_candidate(
                        family_candidates, metrics, boundary,
                    )
                    if selected is None:
                        constrained_rows.append({
                            "unit": unit.name,
                            "family": family,
                            "p95_boundary_minutes": boundary,
                            "is_unbounded": boundary is None,
                            "eligible": False,
                        })
                        continue
                    item = metrics(selected)
                    constrained_rows.append({
                        "unit": unit.name,
                        "family": family,
                        "p95_boundary_minutes": boundary,
                        "is_unbounded": boundary is None,
                        "eligible": True,
                        "policy": selected.policy,
                        "mean_excess_aoi": (
                            item.mean_excess_aoi[boundary]
                            if boundary is not None else float("nan")
                        ),
                        "aoi_exceedance_rate": (
                            item.aoi_exceedance_rate[boundary]
                            if boundary is not None else float("nan")
                        ),
                        **item.flat_dict(),
                    })
            for threshold in thresholds:
                for sync_cost in costs:
                    best_rl = _best_candidate(
                        unit.family_rl, metrics, threshold, sync_cost,
                    )
                    best_periodic = _best_candidate(
                        unit.family_periodic, metrics, threshold, sync_cost,
                    )
                    rl_item = metrics(best_rl)
                    periodic_item = metrics(best_periodic)
                    rl_score = rl_item.operational_score(threshold, sync_cost)
                    periodic_score = periodic_item.operational_score(threshold, sync_cost)
                    winner = _winner(rl_score - periodic_score)
                    family_rows.append({
                        "unit": unit.name, "threshold": threshold,
                        "sync_cost": sync_cost, "winner": winner,
                        "comparison_type": _comparison_category(
                            winner, rl_item, periodic_item, threshold,
                        ),
                        "rl_policy": best_rl.policy, "rl_cost_per_step": rl_score,
                        "rl_sync_rate": rl_item.sync_rate,
                        "rl_mean_excess_aoi": rl_item.mean_excess_aoi[threshold],
                        "periodic_policy": best_periodic.policy,
                        "periodic_cost_per_step": periodic_score,
                        "periodic_sync_rate": periodic_item.sync_rate,
                        "periodic_mean_excess_aoi": periodic_item.mean_excess_aoi[threshold],
                    })

    results = {
        "canonical_pairs": pd.DataFrame(fixed_rows),
        "break_even": pd.DataFrame(break_even_rows),
        "candidate_metrics": pd.DataFrame(candidate_rows),
        "candidate_threshold_metrics": pd.DataFrame(threshold_rows),
        "p95_constrained_best": pd.DataFrame(constrained_rows),
        "family_envelope": pd.DataFrame(family_rows),
    }
    if include_family:
        results.update(build_multiobjective_results(
            results["candidate_metrics"],
            weight_step=weight_step,
            normalizations=freshness_normalizations,
        ))
        results.update(build_two_objective_results(
            results["candidate_metrics"],
            weight_step=two_objective_weight_step,
            normalizations=freshness_normalizations,
            plane_step=two_objective_plane_step,
        ))
    return results


def write_outputs(results: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Write each evidence table to a stable CSV name."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)


def _file_digest(path: Path) -> str:
    """Return the SHA-256 digest of one artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_csv(table_name: str) -> str | None:
    """Identify the immediate machine-readable source for a result table."""
    source = "candidate_metrics.csv"
    if table_name == "candidate_metrics":
        source = None
    elif table_name == "multiobjective_candidates":
        source = "candidate_metrics.csv"
    elif table_name.startswith("two_objective_"):
        if table_name == "two_objective_candidates":
            source = "candidate_metrics.csv"
        elif "weight_space" in table_name or table_name == "two_objective_winner_ranges":
            source = "two_objective_weight_sweep_winners.csv"
        else:
            source = "two_objective_candidates.csv"
    elif table_name in {
        "pareto_frontiers", "weight_sweep_winners", "priority_scenario_winners",
    }:
        source = "multiobjective_candidates.csv"
    elif table_name.startswith("weight_space_"):
        source = "weight_sweep_winners.csv"
    return source


def _figure_source(path: Path) -> str:
    """Identify the CSV rendered by one generated interactive figure."""
    if "weight_plane" in path.name:
        return "two_objective_weight_plane_winners.csv"
    if "winner_strip" in path.name:
        return "two_objective_weight_sweep_winners.csv"
    if "two_objective_pareto" in path.name:
        return "two_objective_candidates.csv"
    return "weight_sweep_winners.csv"


def write_artifact_manifest(
    results: dict[str, pd.DataFrame],
    output_dir: Path,
    protocol: dict[str, Any],
    protocol_path: Path,
    command: list[str],
    figure_paths: Iterable[Path] = (),
) -> Path:
    """Link regenerated artifacts to frozen inputs, code, and source tables."""
    code_paths = (
        Path(__file__),
        Path(__file__).with_name("multiobjective_policy_analysis.py"),
        Path(__file__).with_name("freeze_pareto_candidate_inventory.py"),
    )
    inputs = []
    for key in (
        "candidate_inventory_csv", "candidate_inventory_source_csv",
        "periodic_decisions_manifest",
    ):
        if key in protocol:
            input_path = (protocol_path.parent / protocol[key]).resolve()
            inputs.append({
                "role": key, "path": str(input_path),
                "sha256": _file_digest(input_path),
            })
    tables = []
    for name in sorted(results):
        artifact = output_dir / f"{name}.csv"
        tables.append({
            "path": artifact.name,
            "sha256": _file_digest(artifact),
            "source_csv": _source_csv(name),
        })
    figures = [{
        "path": path.name,
        "sha256": _file_digest(path),
        "source_csv": _figure_source(path),
    } for path in sorted(figure_paths)]
    payload = {
        "protocol_version": protocol.get("version"),
        "evidence_protocol": protocol.get("evidence_protocol", {}),
        "regeneration_command": shlex.join(command),
        "output_directory": str(output_dir.resolve()),
        "input_fingerprint_table": "candidate_metrics.csv",
        "input_fingerprint_fields": [
            "decisions_path", "timestamp_digest", "event_digest", "initial_true_aoi",
        ],
        "inputs": inputs,
        "analysis_code_sha256": {
            path.name: _file_digest(path) for path in code_paths
        },
        "tables": tables,
        "figures": figures,
    }
    destination = output_dir / "artifact_manifest.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def _parse_values(raw: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one numeric value is required")
    return values


def main() -> None:
    """Run manifest validation and write the requested evidence tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, required=True,
        help="Input manifest with candidate inventory and matched trace paths",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).with_name("rl_periodic_reward_results"),
    )
    parser.add_argument("--thresholds", type=_parse_values, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--costs", type=_parse_values, default=DEFAULT_COSTS)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--two-objective-weight-step", type=float, default=0.01)
    parser.add_argument("--two-objective-plane-step", type=float, default=0.05)
    parser.add_argument(
        "--freshness-normalizations", type=parse_normalizations,
        default=DEFAULT_NORMALIZATIONS,
    )
    parser.add_argument("--multiobjective-plots", action="store_true")
    parser.add_argument("--fixed-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    raw_manifest, units = load_manifest(args.manifest)
    results = run_analysis(
        units, args.thresholds, args.costs,
        include_family=not args.fixed_only,
        weight_step=args.weight_step,
        two_objective_weight_step=args.two_objective_weight_step,
        freshness_normalizations=args.freshness_normalizations,
        two_objective_plane_step=args.two_objective_plane_step,
    )
    if not args.validate_only:
        write_outputs(results, args.output_dir)
        figure_paths = []
        if args.multiobjective_plots and not args.fixed_only:
            figure_paths.extend(write_weight_triangle_html(
                results["weight_sweep_winners"], args.output_dir,
            ))
            figure_paths.extend(write_two_objective_html(
                results["two_objective_candidates"],
                results["two_objective_weight_sweep_winners"],
                args.output_dir,
            ))
            figure_paths.extend(write_two_objective_weight_plane_html(
                results["two_objective_weight_plane_winners"], args.output_dir,
            ))
        write_artifact_manifest(
            results, args.output_dir, raw_manifest, args.manifest.resolve(),
            [sys.executable, *sys.argv], figure_paths,
        )
    print(f"Validated {len(units)} units and {len(results['canonical_pairs'])} fixed pairs")
    canonical_wins = int(
        (results["canonical_pairs"]["reward_difference_pct"] > 0).sum()
    )
    print("Canonical RL wins:", canonical_wins)
    if not args.fixed_only:
        print("Family-envelope cells:", len(results["family_envelope"]))


if __name__ == "__main__":
    main()
