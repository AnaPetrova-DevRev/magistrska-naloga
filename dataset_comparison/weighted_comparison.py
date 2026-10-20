"""Block-weighted pairwise comparison + dual clustering.

Comparison procedure:
- curated feature set (excludes categorical codes, zero-variance, aliases)
- 10 semantic blocks; each feature z-scored then scaled by 1/sqrt(block_size)
  so each BLOCK contributes equally; volume down-weighted to 0.5 in full run
- three independent signals: block-weighted Euclidean/cosine, KS on events_delta,
  JSD on dow/hod profiles
- two clustering runs (behavioral-only / full), Ward + k-sweep 2..5 with silhouette
- documented transfer-candidate rule

Usage:
    python -m dataset_comparison.weighted_comparison --fingerprints PATH [--output-dir PATH]
Default out_dir: data/dataset_comparison
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from itertools import combinations

import numpy as np
from scipy import stats as scipy_stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Protocol constants

KS_SAMPLE_SIZE = 10_000
SEED = 42
JSD_EPS = 1e-10

# Excluded from the Euclidean/cosine vector, with reason (§2.1).
EXCLUDED_FEATURES = {
    "iat_best_family",       # categorical code (1/3/4)
    "peak_dow", "peak_hour", # cyclic categoricals -> compared via JSD
    "iat_ks_passes",         # zero variance (all 0)
    "num_daily_cron_patterns",  # zero variance (all 0)
    "total_events",          # exact alias of raw_events
    "event_density_per_day", # exact alias of raw_density_per_day
    "iat_best_aic",          # scale-dependent fit artifact
    "avg_record_lifespan",   # backward-compatible alias; use claim-safe key
}

# Semantic blocks (§3). Order is fixed for determinism.
BLOCKS: dict[str, list[str]] = {
    "volume": ["raw_events", "minute_events", "unique_records", "events_per_record",
               "records_per_day", "span_days", "total_minutes_with_events"],
    "density": ["minute_density_per_day", "raw_density_per_day", "coverage_ratio",
                "active_days_ratio"],
    "dedup": ["dedup_ratio", "multi_event_minute_fraction", "events_per_min_mean",
              "events_per_min_max"],
    "burstiness": ["events_per_min_cov", "burst_ratio"],
    "iat_shape": ["delta_mean", "delta_median", "delta_std", "delta_skew",
                  "delta_kurtosis", "delta_iqr", "delta_q10", "delta_q25",
                  "delta_q75", "delta_q90", "iat_ks_stat", "iat_ks_pvalue"],
    "zero_inflation": ["zero_fraction_all", "zero_fraction_train", "zero_fraction_test"],
    "periodicity": ["num_hourly_cron_patterns", "dominant_cron_frequency",
                    "cron_event_fraction"],
    "rhythm": ["dow_entropy", "hod_entropy", "weekend_fraction", "night_fraction",
               "business_hours_fraction"],
    "lifecycle": ["pct_single_change_records", "pct_records_gt_10_changes",
                  "median_changes_per_record", "avg_creation_to_last_state",
                  "avg_time_to_first_mod", "pct_created_before_cutoff"],
    "outliers": ["outlier_pct_epm_zscore", "outlier_pct_epm_mad", "outlier_pct_epm_iqr",
                 "outlier_pct_epm_seasonal", "outlier_pct_epm_consensus",
                 "outlier_epm_method_divergence", "outlier_pct_iat_zscore",
                 "outlier_pct_iat_mad", "outlier_pct_iat_iqr", "outlier_pct_iat_seasonal",
                 "outlier_pct_iat_consensus", "outlier_iat_method_divergence"],
}

# Per-run block weights (§3 / §5). None => block excluded from that run.
RUN_WEIGHTS = {
    "lifecycle_excluded": {
        b: (None if b in {"volume", "lifecycle"} else 1.0) for b in BLOCKS
    },
    "behavioral": {b: (None if b == "volume" else 1.0) for b in BLOCKS},
    "full": {b: (0.5 if b == "volume" else 1.0) for b in BLOCKS},
}


def _selected_features(run: str) -> list[tuple[str, str]]:
    """Return ordered `(feature, block)` pairs enabled for one run."""
    weights = RUN_WEIGHTS[run]
    return [
        (feature, block)
        for block, features in BLOCKS.items()
        if weights[block] is not None
        for feature in features
        if feature not in EXCLUDED_FEATURES
    ]


def _build_weighted_matrix(fps: list[dict], run: str) -> tuple[np.ndarray, list[str]]:
    """Build the z-scored, block-weighted feature matrix for a run.

    Each feature is StandardScaler-z-scored across units, then multiplied by
    sqrt(block_weight / block_size / total_weight). Each block's squared
    contribution is therefore its normalized block weight. Returns
    (matrix, unit_names).
    """
    names = [fp["sync_unit_name"] for fp in fps]
    weights = RUN_WEIGHTS[run]
    columns = _selected_features(run)

    def feature_value(fp: dict, feature: str) -> float:
        features = fp["features"]
        if feature == "avg_creation_to_last_state":
            return features.get(feature, features.get("avg_record_lifespan", 0.0))
        return features.get(feature, 0.0)

    raw = np.array(
        [[feature_value(fp, feature) for feature, _ in columns] for fp in fps],
        dtype=float,
    )
    scaled = StandardScaler().fit_transform(raw)  # zero-variance cols -> 0

    # Per-feature multiplier: sqrt(block_weight / block_size_in_run)
    block_sizes = {}
    for _, b in columns:
        block_sizes[b] = block_sizes.get(b, 0) + 1
    mult = np.array([
        np.sqrt(weights[b] / block_sizes[b]) for _, b in columns
    ])
    total_weight = sum(weight for weight in weights.values() if weight is not None)
    return scaled * mult / np.sqrt(total_weight), names


def _euclidean_cosine(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(mat)
    euc = np.zeros((n, n))
    cos = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(mat[i] - mat[j]))
            ni, nj = np.linalg.norm(mat[i]), np.linalg.norm(mat[j])
            if ni == 0 or nj == 0:
                c = 1.0 if (ni == 0 and nj == 0) else 0.0
            else:
                c = float(np.dot(mat[i], mat[j]) / (ni * nj))
            euc[i, j] = euc[j, i] = d
            cos[i, j] = cos[j, i] = c
    return euc, cos


def _load_delta_samples(names: list[str]) -> dict[str, np.ndarray]:
    """Load events_delta samples per unit (subsampled) for KS tests."""
    import pandas as pd
    from sync_unit import get_sync_units
    su_by_name = {su.get_name(): su for su in get_sync_units()}
    rng = np.random.default_rng(SEED)
    out = {}
    for name in names:
        su = su_by_name.get(name)
        if su is None:
            continue
        path = f"data/{su.organization_id}/{su.external_system_id}/{su.project_id}/data_preprocessing/merged_time_delta_1.pkl"
        if not os.path.exists(path):
            continue
        d = pd.read_pickle(path)["events_delta"].dropna().values.astype(float)
        if len(d) > KS_SAMPLE_SIZE:
            d = rng.choice(d, size=KS_SAMPLE_SIZE, replace=False)
        out[name] = d
    return out


def _cluster(mat: np.ndarray, names: list[str]) -> dict:
    """Ward hierarchical clustering with k=2..5 silhouette sweep + kmeans cross-check."""
    n = len(mat)
    Z = linkage(mat, method="ward")
    sweep = []
    best = None
    for k in range(2, min(6, n)):
        labels = fcluster(Z, t=k, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        sil = float(silhouette_score(mat, labels))
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(mat)
        km_sil = float(silhouette_score(mat, km.labels_))
        entry = {"k": k, "ward_silhouette": round(sil, 4),
                 "kmeans_silhouette": round(km_sil, 4),
                 "ward_labels": {names[i]: int(labels[i]) for i in range(n)}}
        sweep.append(entry)
        if best is None or sil > best["ward_silhouette"]:
            best = entry
    continuum = best is None or best["ward_silhouette"] < 0.25
    return {
        "silhouette_sweep": sweep,
        "chosen_k": best["k"] if best else None,
        "best_silhouette": best["ward_silhouette"] if best else None,
        "labels": best["ward_labels"] if best else None,
        "verdict": "continuum" if continuum else "clustered",
        "linkage": Z.tolist(),
    }


def compute(
    out_dir: str = "data/dataset_comparison",
    *,
    fingerprint_path: str,
    delta_samples: dict[str, np.ndarray] | None = None,
    runs: tuple[str, ...] = ("behavioral", "full"),
) -> dict:
    """Compute pairwise and clustering outputs from an explicit fingerprint set."""
    with open(fingerprint_path, encoding="utf-8") as file:
        fps = json.load(file)
    names = [fp["sync_unit_name"] for fp in fps]
    n = len(fps)

    # --- Signal 1: block-weighted Euclidean/cosine, both runs ---
    matrices = {}
    clusters = {}
    for run in runs:
        mat, _ = _build_weighted_matrix(fps, run)
        matrices[run] = mat
        clusters[run] = _cluster(mat, names)
    distance_matrices = {
        run: _euclidean_cosine(matrix) for run, matrix in matrices.items()
    }

    # --- Signal 2: KS on events_delta ---
    samples = delta_samples if delta_samples is not None else _load_delta_samples(names)

    # --- Signal 3: JSD on dow/hod ---
    profiles = {fp["sync_unit_name"]: fp for fp in fps}

    # Assemble pairwise records
    pairs = []
    idx = {name: i for i, name in enumerate(names)}
    for a, b in combinations(names, 2):
        i, j = idx[a], idx[b]
        # KS
        sa, sb = samples.get(a), samples.get(b)
        if sa is not None and sb is not None and len(sa) and len(sb):
            ks_stat, ks_p = scipy_stats.ks_2samp(sa, sb)
        else:
            ks_stat, ks_p = 0.0, 1.0
        # JSD
        da = np.maximum(np.array(profiles[a]["dow_profile"]), JSD_EPS)
        db = np.maximum(np.array(profiles[b]["dow_profile"]), JSD_EPS)
        ha = np.maximum(np.array(profiles[a]["hod_profile"]), JSD_EPS)
        hb = np.maximum(np.array(profiles[b]["hod_profile"]), JSD_EPS)
        same_fam = (profiles[a]["features"].get("iat_best_family")
                    == profiles[b]["features"].get("iat_best_family"))
        pair = {
            "su_a": a, "su_b": b,
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_p), 6),
            "dow_jsd": round(float(jensenshannon(da, db)), 4),
            "hod_jsd": round(float(jensenshannon(ha, hb)), 4),
            "same_iat_family": bool(same_fam),
        }
        for run, (euclidean, cosine) in distance_matrices.items():
            pair[f"euclidean_{run}"] = round(float(euclidean[i, j]), 4)
            pair[f"cosine_{run}"] = round(float(cosine[i, j]), 4)
        pairs.append(pair)

    # --- Transfer-candidate rule (§6, relative-tercile) ---
    # All three gates are relative to the 28-pair distribution, because absolute
    # thresholds are miscalibrated: block-weighting caps cosine well below 0.8, and
    # 10k-sample KS makes every p-value ~0. We rank by KS *statistic* (effect size).
    euc_arr = np.array([p["euclidean_behavioral"] for p in pairs])
    cos_arr = np.array([p["cosine_behavioral"] for p in pairs])
    ks_arr = np.array([p["ks_statistic"] for p in pairs])
    euc_cut = float(np.percentile(euc_arr, 100 / 3))   # bottom tercile (<=)
    cos_cut = float(np.percentile(cos_arr, 200 / 3))   # top tercile (>=)
    ks_cut = float(np.percentile(ks_arr, 100 / 3))     # bottom tercile (<=)
    for p in pairs:
        p["transfer_candidate"] = bool(
            p["euclidean_behavioral"] <= euc_cut
            and p["cosine_behavioral"] >= cos_cut
            and p["ks_statistic"] <= ks_cut
        )
    tercile = {"euclidean_bottom_tercile": round(euc_cut, 4),
               "cosine_top_tercile": round(cos_cut, 4),
               "ks_stat_bottom_tercile": round(ks_cut, 4)}

    os.makedirs(out_dir, exist_ok=True)
    pairwise_payload = {"pairs": pairs, "tercile_cutoff": tercile, "n_units": n}
    for file_name in ("pairwise_distances.json", "pairwise_distances_v2.json"):
        with open(
            os.path.join(out_dir, file_name), "w", encoding="utf-8"
        ) as file:
            json.dump(pairwise_payload, file, indent=2)
    for run in runs:
        with open(
            os.path.join(out_dir, f"clusters_{run}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(clusters[run], f, indent=2)

    return {"pairs": pairs, "clusters": clusters, "tercile_cutoff": tercile,
            "names": names}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fingerprints", required=True,
                        help="Input fingerprint JSON produced by dataset analysis")
    parser.add_argument("--output-dir", default="data/dataset_comparison")
    args = parser.parse_args()
    res = compute(args.output_dir, fingerprint_path=args.fingerprints)
    print(f"Computed {len(res['pairs'])} pairs over {len(res['names'])} units.")
    for run in res["clusters"]:
        c = res["clusters"][run]
        print(f"[{run}] verdict={c['verdict']} chosen_k={c['chosen_k']} "
              f"best_sil={c['best_silhouette']}")
    tc = [p for p in res["pairs"] if p["transfer_candidate"]]
    print(f"transfer candidates ({len(tc)}):")
    for p in tc:
        print(f"  {p['su_a']} <-> {p['su_b']}  euc={p['euclidean_behavioral']} "
              f"cos={p['cosine_behavioral']} ks_p={p['ks_pvalue']}")


if __name__ == "__main__":
    main()
