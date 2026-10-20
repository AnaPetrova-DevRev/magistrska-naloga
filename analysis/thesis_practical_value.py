"""Reproduce thesis practical differences from frozen CSVs, never decision traces.

Run with: python -m analysis.thesis_practical_value --input-dir PATH
Add --output-dir PATH to export new CSVs and a provenance manifest. Existing
output directories are refused. No training, parquet access or new dependencies.
"""

# Explicit audit loops keep every validation and intermediate quantity visible.
# pylint: disable=too-many-locals,too-many-branches,too-many-nested-blocks

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import quantiles


INPUT_NAMES = ("candidate_metrics.csv", "two_objective_candidates.csv",
               "two_objective_normalization_bounds.csv",
               "two_objective_weight_sweep_winners.csv")
GROUP_KEYS = ("unit", "objective_pair", "normalization")
TRACE_KEYS = ("steps", "timestamp_digest", "event_digest", "initial_true_aoi")
PAIRS = {
    "sync_vs_aoii": ("sync_rate", "cvar95_aoii"),
    "empty_vs_aoii": ("empty_sync_step_rate", "cvar95_aoii"),
    "sync_vs_aoi": ("sync_rate", "cvar95_true_aoi"),
    "empty_vs_aoi": ("empty_sync_step_rate", "cvar95_true_aoi"),
}
DELTAS = {
    "delta_syncs_per_1440min": ("sync_rate", 1440),
    "delta_empty_per_1440min": ("empty_sync_step_rate", 1440),
    "delta_cvar95_aoii_RL_minus_P": ("cvar95_aoii", 1),
    "delta_cvar95_true_aoi_RL_minus_P": ("cvar95_true_aoi", 1),
}
METRICS = tuple(metric for metric, _ in DELTAS.values())
TOLERANCE = 1e-12


def read_csv(path):
    """Load one small named CSV, not any paths referenced by its records."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    """Fingerprint a named small artifact."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(directory):
    """Verify only the named small tables; never follow stored trace paths."""
    directory = Path(directory)
    manifest_path = directory / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {r["path"]: r["sha256"] for r in manifest["tables"]}
    tables, hashes = {}, {manifest_path.name: sha256(manifest_path)}
    for name in INPUT_NAMES:
        digest = sha256(directory / name)
        if digest != records.get(name):
            raise ValueError(f"Frozen input SHA-256 mismatch: {name}")
        hashes[name] = digest
        tables[name] = read_csv(directory / name)
    protocol = manifest["evidence_protocol"]
    if protocol["endpoint_rule"] != "restrict_winners_to_raw_pareto_frontier":
        raise ValueError("Unsupported endpoint rule")
    return tables, hashes


def eligible_candidates(rows):
    """Keep frozen eligibility, rejecting mismatched inputs and invalid metrics."""
    selected, identities = [], set()
    for original in rows:
        if str(original["family_eligible"]).lower() != "true":
            continue
        row = dict(original)
        if row["family"] not in ("RL", "periodic"):
            raise ValueError("Unsupported candidate family")
        identity = tuple(row[k] for k in ("unit", "family", "policy"))
        if identity in identities:
            raise ValueError("Duplicate candidate identity")
        identities.add(identity)
        row["steps"] = int(row["steps"])
        if row["steps"] <= 0:
            raise ValueError("Trace length must be positive")
        for key in (*METRICS, "initial_true_aoi"):
            row[key] = float(row[key])
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError(f"Invalid nonnegative finite metric: {key}")
        for metric, count in (("sync_rate", "sync_count"),
                              ("empty_sync_step_rate", "empty_sync_count")):
            if not 0 <= row[metric] <= 1:
                raise ValueError(f"Rate outside [0, 1]: {metric}")
            if not math.isclose(row[metric], int(row[count]) / row["steps"],
                                abs_tol=TOLERANCE, rel_tol=0):
                raise ValueError(f"Rate/count mismatch: {metric}")
        if row["empty_sync_step_rate"] > row["sync_rate"] + TOLERANCE:
            raise ValueError("Empty synchronizations exceed all synchronizations")
        if any(not row[k] for k in ("timestamp_digest", "event_digest", "decisions_path")):
            raise ValueError("Missing trace identity")
        selected.append(row)
    if not selected:
        raise ValueError("No eligible candidates")
    for unit in sorted({r["unit"] for r in selected}):
        group = [r for r in selected if r["unit"] == unit]
        if {r["family"] for r in group} != {"RL", "periodic"}:
            raise ValueError(f"Both families required: {unit}")
        for key in TRACE_KEYS:
            if len({r[key] for r in group}) != 1:
                raise ValueError(f"Mismatched trace {key}: {unit}")
    return sorted(selected, key=lambda r: (r["unit"], r["family"], r["policy"]))


def normalize(values):
    """Use common min/max bounds and map a constant objective to zero."""
    low, high = min(values), max(values)
    return [0.0 if high - low <= 1e-15 else (v - low) / (high - low)
            for v in values]


def prepare_group(rows, pair, mode):
    """Normalize the complete shared pool, independently of Pareto membership."""
    if mode not in ("raw", "log1p"):
        raise ValueError("Unknown freshness transformation")
    operational, freshness = PAIRS[pair]
    raw = [(r[operational], r[freshness]) for r in rows]
    transformed = [math.log1p(f) if mode == "log1p" else f for _, f in raw]
    bounds = {}
    for name, values in (("operational", [o for o, _ in raw]),
                         ("freshness", [f for _, f in raw]),
                         ("transformed_freshness", transformed)):
        bounds.update({f"{name}_min": min(values), f"{name}_max": max(values)})
    prepared = []
    for row, value, op, fresh in zip(rows, raw, normalize([o for o, _ in raw]),
                                    normalize(transformed)):
        dominated = any(all(a <= b + TOLERANCE for a, b in zip(other, value))
                        and any(a < b - TOLERANCE for a, b in zip(other, value))
                        for other in raw)
        prepared.append({**row, "normalized_operational": op,
                         "normalized_freshness": fresh, "is_pareto": not dominated})
    return prepared, bounds


def compare_at_weight(rows, bounds, weight_id):
    """Retain tied alternatives and raw ranges; never synthesize a mean policy."""
    if not isinstance(weight_id, int) or not 0 <= weight_id <= 100:
        raise ValueError("Weight ID must be an integer in [0, 100]")
    weight = weight_id / 100
    scored = [{**r, "score": weight * r["normalized_operational"]
               + (1 - weight) * r["normalized_freshness"]} for r in rows]
    result = {"weight_id": weight_id, "weight_operational": weight,
              "is_interior": 10 <= weight_id <= 90, **bounds}
    chosen = {}
    for family, prefix in (("RL", "rl"), ("periodic", "periodic")):
        pool = [r for r in scored if r["family"] == family]
        best = min(r["score"] for r in pool)
        chosen[family] = [r for r in pool if abs(r["score"] - best) <= TOLERANCE]
        result[f"{prefix}_score"] = best
        result[f"{prefix}_candidates"] = [
            {k: r[k] for k in ("policy", "decisions_path", "score", "is_pareto",
                               "normalized_operational", "normalized_freshness", *METRICS)}
            for r in chosen[family]]
    gap = result["periodic_score"] - result["rl_score"]
    result["score_gap_P_minus_RL"] = gap
    result["family_score_winner"] = ("mixed_tie" if abs(gap) <= TOLERANCE
                                     else "RL" if gap > 0 else "periodic")
    # Published endpoint winners exclude dominated ties; family comparators do not.
    front = [r for r in scored if r["is_pareto"]]
    result["winning_score"] = min(r["score"] for r in front)
    winners = [r for r in front if abs(r["score"] - result["winning_score"]) <= TOLERANCE]
    families = {r["family"] for r in winners}
    result["winner_family"] = next(iter(families)) if len(families) == 1 else "mixed_tie"
    result["winner_policies"] = ";".join(sorted(f'{r["family"]}:{r["policy"]}' for r in winners))
    for name, (metric, factor) in DELTAS.items():
        differences = [(rl[metric] - p[metric]) * factor
                       for rl in chosen["RL"] for p in chosen["periodic"]]
        low, high = min(differences), max(differences)
        result[f"{name}_min"], result[f"{name}_max"] = low, high
        result[name] = low if low == high else None
    return result


def summarize(pairs):
    """Inclusive quartiles of all 81 weights, including periodic wins; not CIs."""
    groups = defaultdict(list)
    for row in pairs:
        if row["is_interior"]:
            groups[tuple(row[k] for k in GROUP_KEYS)].append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        if sorted(r["weight_id"] for r in group) != list(range(10, 91)):
            raise ValueError("Summary requires every interior weight exactly once")
        row = dict(zip(GROUP_KEYS, key))
        row["weight_count"] = len(group)
        for metric in ("score_gap_P_minus_RL", *DELTAS):
            values = [r[metric] for r in group]
            if any(v is None for v in values):
                raise ValueError("Ambiguous interior raw differences: retain tie ranges")
            for suffix, value in zip(("q25", "median", "q75"),
                                     quantiles(values, n=4, method="inclusive")):
                row[f"{metric}_{suffix}"] = value
        counts = Counter(r["family_score_winner"] for r in group)
        row.update({f"{family}_weights": counts[family]
                    for family in ("RL", "periodic", "mixed_tie")})
        row["rl_fractional_share"] = (counts["RL"] + 0.5 * counts["mixed_tie"]) / len(group)
        summaries.append(row)
    return summaries


def indexed(rows, keys):
    """Index a cross-check table without silently accepting duplicate keys."""
    result = {tuple(str(r[k]) for k in keys): r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate cross-check key")
    return result


def check_number(actual, expected, label):
    """Compare independent floating-point calculations at numerical precision."""
    if not math.isclose(actual, float(expected), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"Frozen cross-check failed: {label}")


def analyze(directory):
    """Cross-check bounds, candidate coordinates and all published winners."""
    tables, hashes = load_inputs(directory)
    rows = eligible_candidates(tables["candidate_metrics.csv"])
    saved_candidates = indexed(tables["two_objective_candidates.csv"],
                               (*GROUP_KEYS, "family", "policy"))
    saved_bounds = indexed(tables["two_objective_normalization_bounds.csv"], GROUP_KEYS)
    saved_winners = indexed(tables["two_objective_weight_sweep_winners.csv"],
                            (*GROUP_KEYS, "weight_id"))
    pairs, trace_metadata = [], {}
    for unit in sorted({r["unit"] for r in rows}):
        group = [r for r in rows if r["unit"] == unit]
        intervals = {r["policy"] for r in group if r["family"] == "periodic"}
        if intervals != {f"periodic_{n}min" for n in (1, 2, 5, 10, 15, 20, 30, 45, 60)}:
            raise ValueError(f"Incomplete declared periodic set: {unit}")
        trace_metadata[unit] = {k: group[0][k] for k in TRACE_KEYS}
        for pair in sorted(PAIRS):
            for mode in ("log1p", "raw"):
                key = (unit, pair, mode)
                prepared, bounds = prepare_group(group, pair, mode)
                expected_bounds = saved_bounds.pop(key)
                for field, value in bounds.items():
                    check_number(value, expected_bounds[field], (*key, field))
                for candidate in prepared:
                    identity = (*key, candidate["family"], candidate["policy"])
                    expected = saved_candidates.pop(identity)
                    for field in (*METRICS, "normalized_operational", "normalized_freshness"):
                        check_number(candidate[field], expected[field], (*key, field))
                    if (expected["is_pareto"] == "True") != candidate["is_pareto"]:
                        raise ValueError("Pareto membership mismatch")
                    for field in ("decisions_path", "timestamp_digest", "event_digest"):
                        if candidate[field] != expected[field]:
                            raise ValueError(f"Candidate identity mismatch: {field}")
                for weight_id in range(101):
                    row = {**dict(zip(GROUP_KEYS, key)),
                           **compare_at_weight(prepared, bounds, weight_id)}
                    expected = saved_winners.pop((*key, str(weight_id)))
                    for field in ("winner_family", "winner_policies"):
                        if row[field] != expected[field]:
                            raise ValueError(f"Frozen winner mismatch: {key}, {weight_id}, {field}")
                    check_number(row["winning_score"], expected["winning_score"], key)
                    check_number(row["weight_operational"], expected["weight_operational"], key)
                    if row["is_interior"] and row["family_score_winner"] != row["winner_family"]:
                        raise ValueError("Interior family/frontier winner mismatch")
                    pairs.append(row)
    if saved_candidates or saved_bounds or saved_winners:
        raise ValueError("Unmatched frozen cross-check rows")
    return {"pairs": pairs, "summaries": summarize(pairs), "input_sha256": hashes,
            "trace_metadata": trace_metadata,
            "candidate_counts": dict(Counter(r["family"] for r in rows))}


def write_report(report, directory):
    """Write only to a new directory, keeping existing evidence untouched."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for key in ("pairs", "summaries"):
        with (directory / f"{key}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report[key][0]))
            writer.writeheader()
            for row in report[key]:
                writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, list) else v
                                 for k, v in row.items()})
    manifest = {k: v for k, v in report.items() if k not in ("pairs", "summaries")}
    manifest.update({"analysis_sha256": sha256(__file__), "full_pair_count": len(report["pairs"]),
                     "interior_pair_count": sum(r["is_interior"] for r in report["pairs"]),
                     "summary_count": len(report["summaries"]),
                     "quartiles": "linear interpolation, inclusive; all 81 interior weights",
                     "score_gap": "periodic minus RL; dimensionless, not percent savings",
                     "raw_differences": "RL minus periodic; 1440-minute replay-rate equivalents",
                     "trace_check": "saved metadata only; no trace files reloaded",
                     "output_sha256": {f"{k}.csv": sha256(directory / f"{k}.csv")
                                       for k in ("pairs", "summaries")}})
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    """Validate in memory by default; export only when explicitly requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing input tables and artifact_manifest.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = analyze(args.input_dir)
    if args.output_dir:
        write_report(report, args.output_dir)
    print(f'{len(report["pairs"])} weight pairs; {len(report["summaries"])} interior summaries')
    print("Eligible candidates:", report["candidate_counts"])
    for pair in ("sync_vs_aoii", "empty_vs_aoii"):
        rows = [r for r in report["summaries"] if r["objective_pair"] == pair
                and r["normalization"] == "log1p"]
        count = sum(r["RL_weights"] for r in rows)
        total = sum(r["weight_count"] for r in rows)
        print(f"{pair}, log1p: {count}/{total} strict RL selections")


if __name__ == "__main__":
    main()
