"""
Pairwise Dataset Comparison Module

Computes similarity and distance metrics between dataset fingerprints:
- Euclidean and cosine distance on normalized fingerprint vectors
- Kolmogorov-Smirnov test on time-delta distributions
- Jensen-Shannon divergence on temporal profiles (hour-of-day, day-of-week)
"""
import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np
from scipy import stats as scipy_stats
from scipy.spatial.distance import cosine, jensenshannon
from sklearn.preprocessing import StandardScaler

from .fingerprint import DatasetFingerprint

logger = logging.getLogger(__name__)

# Maximum events to sample for KS-test (for efficiency with large datasets)
KS_SAMPLE_SIZE = 10_000


@dataclass
class PairwiseComparison:
    """Comparison metrics between two datasets."""

    su_a: str
    su_b: str
    euclidean_distance: float = 0.0
    cosine_similarity: float = 0.0
    ks_statistic: float = 0.0
    ks_pvalue: float = 1.0
    dow_jsd: float = 0.0
    hod_jsd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "su_a": self.su_a,
            "su_b": self.su_b,
            "euclidean_distance": self.euclidean_distance,
            "cosine_similarity": self.cosine_similarity,
            "ks_statistic": self.ks_statistic,
            "ks_pvalue": self.ks_pvalue,
            "dow_jsd": self.dow_jsd,
            "hod_jsd": self.hod_jsd,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PairwiseComparison":
        return cls(**data)


def _compute_euclidean_cosine(
    fingerprints: list[DatasetFingerprint],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pairwise Euclidean distance and cosine similarity matrices.

    Returns:
        (euclidean_matrix, cosine_similarity_matrix) each of shape (n, n)
    """
    if not fingerprints:
        return np.array([[]]), np.array([[]])

    feature_names = fingerprints[0].get_feature_names()
    vectors = np.array([fp.to_feature_vector(feature_names) for fp in fingerprints])

    # Standardize
    scaler = StandardScaler()
    scaled = scaler.fit_transform(vectors)

    n = len(fingerprints)
    euclidean_matrix = np.zeros((n, n))
    cosine_matrix = np.ones((n, n))  # diagonal = 1.0 (self-similarity)

    for i in range(n):
        for j in range(i + 1, n):
            euc = float(np.linalg.norm(scaled[i] - scaled[j]))

            # Cosine distance returns NaN for zero-norm vectors (identical
            # inputs after StandardScaler). Treat as identical (similarity=1).
            norm_i = np.linalg.norm(scaled[i])
            norm_j = np.linalg.norm(scaled[j])
            if norm_i == 0 or norm_j == 0:
                cos_sim = 1.0 if (norm_i == 0 and norm_j == 0) else 0.0
            else:
                cos_sim = 1.0 - cosine(scaled[i], scaled[j])

            euclidean_matrix[i, j] = euc
            euclidean_matrix[j, i] = euc
            cosine_matrix[i, j] = cos_sim
            cosine_matrix[j, i] = cos_sim

    return euclidean_matrix, cosine_matrix


def _compute_ks_tests(
    fingerprints: list[DatasetFingerprint],
    delta_samples: Optional[dict[str, np.ndarray]] = None,
) -> dict[tuple[str, str], tuple[float, float]]:
    """Compute pairwise KS-tests on time-delta distributions.

    Args:
        fingerprints: List of fingerprints (used for sync unit names).
        delta_samples: Pre-loaded time-delta samples per sync unit name.
            If None, KS-test is skipped and defaults are returned.

    Returns:
        Dict mapping (su_a, su_b) to (ks_statistic, ks_pvalue).
    """
    results = {}

    if delta_samples is None:
        for i, fp_a in enumerate(fingerprints):
            for fp_b in fingerprints[i + 1:]:
                results[(fp_a.sync_unit_name, fp_b.sync_unit_name)] = (0.0, 1.0)
        return results

    for i, fp_a in enumerate(fingerprints):
        for fp_b in fingerprints[i + 1:]:
            sample_a = delta_samples.get(fp_a.sync_unit_name)
            sample_b = delta_samples.get(fp_b.sync_unit_name)

            if sample_a is None or sample_b is None or len(sample_a) == 0 or len(sample_b) == 0:
                results[(fp_a.sync_unit_name, fp_b.sync_unit_name)] = (0.0, 1.0)
                continue

            stat, pvalue = scipy_stats.ks_2samp(sample_a, sample_b)
            results[(fp_a.sync_unit_name, fp_b.sync_unit_name)] = (
                float(stat),
                float(pvalue),
            )

    return results


def _compute_profile_jsd(
    fingerprints: list[DatasetFingerprint],
) -> dict[tuple[str, str], tuple[float, float]]:
    """Compute pairwise Jensen-Shannon divergence on temporal profiles.

    Returns:
        Dict mapping (su_a, su_b) to (dow_jsd, hod_jsd).
    """
    results = {}

    for i, fp_a in enumerate(fingerprints):
        for fp_b in fingerprints[i + 1:]:
            # Day-of-week JSD
            dow_a = np.array(fp_a.dow_profile) if fp_a.dow_profile else np.ones(7) / 7
            dow_b = np.array(fp_b.dow_profile) if fp_b.dow_profile else np.ones(7) / 7
            # Ensure valid distributions (non-negative, sum > 0)
            dow_a = np.maximum(dow_a, 1e-10)
            dow_b = np.maximum(dow_b, 1e-10)
            dow_jsd = float(jensenshannon(dow_a, dow_b))

            # Hour-of-day JSD
            hod_a = np.array(fp_a.hod_profile) if fp_a.hod_profile else np.ones(24) / 24
            hod_b = np.array(fp_b.hod_profile) if fp_b.hod_profile else np.ones(24) / 24
            hod_a = np.maximum(hod_a, 1e-10)
            hod_b = np.maximum(hod_b, 1e-10)
            hod_jsd = float(jensenshannon(hod_a, hod_b))

            results[(fp_a.sync_unit_name, fp_b.sync_unit_name)] = (dow_jsd, hod_jsd)

    return results


def compute_all_pairwise(
    fingerprints: list[DatasetFingerprint],
    delta_samples: Optional[dict[str, np.ndarray]] = None,
) -> list[PairwiseComparison]:
    """Compute all pairwise comparison metrics.

    Args:
        fingerprints: List of dataset fingerprints.
        delta_samples: Optional pre-loaded time-delta samples per sync unit.
            Keys are sync_unit_name, values are 1D numpy arrays of delta values.
            If provided, KS-tests are computed. If None, KS-tests are skipped.

    Returns:
        List of PairwiseComparison objects for all unique pairs.
    """
    if len(fingerprints) < 2:
        return []

    logger.info("Computing pairwise comparisons for %d datasets", len(fingerprints))

    euclidean_matrix, cosine_matrix = _compute_euclidean_cosine(fingerprints)
    ks_results = _compute_ks_tests(fingerprints, delta_samples)
    jsd_results = _compute_profile_jsd(fingerprints)

    comparisons = []
    for i, fp_a in enumerate(fingerprints):
        for j, fp_b in enumerate(fingerprints[i + 1:], start=i + 1):
            key = (fp_a.sync_unit_name, fp_b.sync_unit_name)
            ks_stat, ks_pval = ks_results.get(key, (0.0, 1.0))
            dow_jsd, hod_jsd = jsd_results.get(key, (0.0, 0.0))

            comp = PairwiseComparison(
                su_a=fp_a.sync_unit_name,
                su_b=fp_b.sync_unit_name,
                euclidean_distance=float(euclidean_matrix[i, j]),
                cosine_similarity=float(cosine_matrix[i, j]),
                ks_statistic=ks_stat,
                ks_pvalue=ks_pval,
                dow_jsd=dow_jsd,
                hod_jsd=hod_jsd,
            )
            comparisons.append(comp)

    logger.info("Computed %d pairwise comparisons", len(comparisons))
    return comparisons


def compute_distance_matrix(
    fingerprints: list[DatasetFingerprint],
    metric: str = "euclidean",
) -> tuple[np.ndarray, list[str]]:
    """Compute a distance matrix for the given fingerprints.

    Args:
        fingerprints: List of fingerprints.
        metric: "euclidean" or "cosine" (as distance, i.e. 1 - similarity).

    Returns:
        (distance_matrix, labels) where labels are sync unit names.
    """
    euclidean_matrix, cosine_matrix = _compute_euclidean_cosine(fingerprints)
    labels = [fp.sync_unit_name for fp in fingerprints]

    if metric == "cosine":
        # Convert similarity to distance
        return 1.0 - cosine_matrix, labels

    return euclidean_matrix, labels


def load_delta_samples(
    sync_units: list,
    max_samples: int = KS_SAMPLE_SIZE,
) -> dict[str, np.ndarray]:
    """Load time-delta samples for KS-test computation.

    Loads merged_time_delta for each sync unit and subsamples if needed.

    Args:
        sync_units: List of SyncUnit objects.
        max_samples: Maximum number of delta values to keep per dataset.

    Returns:
        Dict mapping sync_unit_name to 1D array of delta values.
    """
    from .fingerprint import _load_time_deltas

    samples = {}
    for su in sync_units:
        td_df = _load_time_deltas(su)
        if td_df is None:
            continue

        if "events_delta" in td_df.columns:
            deltas = td_df["events_delta"].dropna().values.astype(float)
        else:
            continue

        if len(deltas) > max_samples:
            rng = np.random.default_rng(42)
            deltas = rng.choice(deltas, size=max_samples, replace=False)

        samples[su.get_name()] = deltas

    return samples
