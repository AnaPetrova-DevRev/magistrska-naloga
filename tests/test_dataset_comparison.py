"""
Tests for dataset comparison module.

Tests cover:
- DatasetFingerprint dataclass and serialization
- Feature computation helpers (entropy, volume, temporal)
- PairwiseComparison computation
- Distance matrix computation
- Dataset-level clustering
- Record-level clustering
- Feature importance computation
"""
import unittest

import numpy as np
import pandas as pd

from dataset_comparison.fingerprint import (
    DatasetFingerprint,
    _shannon_entropy,
    _safe_div,
    _compute_volume_features,
    _compute_delta_distribution_features,
    _compute_burstiness_features,
    _compute_temporal_profile_features,
    _compute_lifecycle_features,
    _compute_iat_fit_features,
    SHAPE_DIVERGENCE_MAX,
    IAT_FAMILY_NAMES,
)
from dataset_comparison.comparison import (
    PairwiseComparison,
    compute_all_pairwise,
    compute_distance_matrix,
    _compute_euclidean_cosine,
    _compute_profile_jsd,
)
from dataset_comparison.clustering import (
    DatasetClusterResult,
    cluster_datasets,
    _compute_feature_importance,
)


def _make_fingerprint(name, org="test_org", system="test_sys", project="test_proj",
                      features=None, dow_profile=None, hod_profile=None):
    """Helper to create a fingerprint for testing."""
    if features is None:
        features = {"f1": 1.0, "f2": 2.0, "f3": 3.0}
    if dow_profile is None:
        dow_profile = [1/7] * 7
    if hod_profile is None:
        hod_profile = [1/24] * 24
    return DatasetFingerprint(
        sync_unit_name=name,
        org=org,
        system=system,
        project=project,
        features=features,
        dow_profile=dow_profile,
        hod_profile=hod_profile,
        computed_at="2026-04-12T10:00:00",
    )


class TestDatasetFingerprint(unittest.TestCase):
    """Tests for DatasetFingerprint dataclass."""

    def test_to_feature_vector(self):
        fp = _make_fingerprint("test", features={"a": 1.0, "b": 2.0, "c": 3.0})
        vec = fp.to_feature_vector()
        np.testing.assert_array_equal(vec, [1.0, 2.0, 3.0])

    def test_to_feature_vector_with_names(self):
        fp = _make_fingerprint("test", features={"a": 1.0, "b": 2.0, "c": 3.0})
        vec = fp.to_feature_vector(feature_names=["c", "a"])
        np.testing.assert_array_equal(vec, [3.0, 1.0])

    def test_get_feature_names(self):
        fp = _make_fingerprint("test", features={"z": 1.0, "a": 2.0, "m": 3.0})
        self.assertEqual(fp.get_feature_names(), ["a", "m", "z"])

    def test_to_dict_and_from_dict(self):
        fp = _make_fingerprint("su1", org="brightpath", system="salesforce")
        d = fp.to_dict()
        fp2 = DatasetFingerprint.from_dict(d)
        self.assertEqual(fp2.sync_unit_name, "su1")
        self.assertEqual(fp2.org, "brightpath")
        self.assertEqual(fp2.features, fp.features)
        self.assertEqual(fp2.dow_profile, fp.dow_profile)

    def test_missing_feature_defaults_to_zero(self):
        fp = _make_fingerprint("test", features={"a": 1.0})
        vec = fp.to_feature_vector(feature_names=["a", "missing"])
        np.testing.assert_array_equal(vec, [1.0, 0.0])


class TestHelperFunctions(unittest.TestCase):
    """Tests for helper computation functions."""

    def test_shannon_entropy_uniform(self):
        """Uniform distribution has max entropy."""
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        self.assertAlmostEqual(_shannon_entropy(probs), 2.0)

    def test_shannon_entropy_concentrated(self):
        """Fully concentrated distribution has 0 entropy."""
        probs = np.array([1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(_shannon_entropy(probs), 0.0)

    def test_shannon_entropy_empty(self):
        probs = np.array([0.0, 0.0])
        self.assertEqual(_shannon_entropy(probs), 0.0)

    def test_safe_div_normal(self):
        self.assertEqual(_safe_div(10, 5), 2.0)

    def test_safe_div_zero_denominator(self):
        self.assertEqual(_safe_div(10, 0), 0.0)

    def test_safe_div_custom_default(self):
        self.assertEqual(_safe_div(10, 0, default=-1.0), -1.0)


class TestVolumeFeatures(unittest.TestCase):
    """Tests for volume feature computation."""

    def test_basic_volume(self):
        apr_df = pd.DataFrame({
            "id": ["a", "b", "c"],
            "num_changes_total": [10, 20, 30],
        })
        epm_df = pd.DataFrame({
            "minute": pd.date_range("2024-01-01", periods=1440, freq="min"),
            "events_per_minute": np.random.randint(0, 5, 1440),
        })
        td_df = pd.DataFrame({
            "events": pd.date_range("2024-01-01", periods=100, freq="h"),
            "events_delta": np.random.rand(100) * 3600,
        })

        features = _compute_volume_features(apr_df, epm_df, td_df)

        self.assertEqual(features["total_events"], 100.0)
        self.assertEqual(features["unique_records"], 3.0)
        self.assertAlmostEqual(features["events_per_record"], 100.0 / 3.0, places=1)
        self.assertGreater(features["event_density_per_day"], 0)


class TestDeltaDistributionFeatures(unittest.TestCase):
    """Tests for time-delta distribution statistics."""

    def test_normal_deltas(self):
        td_df = pd.DataFrame({
            "events_delta": [10, 20, 30, 40, 50],
        })
        features = _compute_delta_distribution_features(td_df)

        self.assertAlmostEqual(features["delta_mean"], 30.0)
        self.assertAlmostEqual(features["delta_median"], 30.0)
        self.assertGreater(features["delta_std"], 0)
        self.assertAlmostEqual(features["delta_q25"], 20.0)
        self.assertAlmostEqual(features["delta_q75"], 40.0)
        self.assertAlmostEqual(features["delta_iqr"], 20.0)

    def test_empty_deltas(self):
        td_df = pd.DataFrame({"events_delta": pd.Series([], dtype=float)})
        features = _compute_delta_distribution_features(td_df)
        self.assertEqual(features["delta_mean"], 0.0)

    def test_missing_column(self):
        td_df = pd.DataFrame({"other_col": [1, 2, 3]})
        features = _compute_delta_distribution_features(td_df)
        self.assertEqual(features["delta_mean"], 0.0)


class TestIatFitFeatures(unittest.TestCase):
    """Tests for parametric IAT-fit selection, incl. the shape-divergence guard.

    Regression guard for the fingerprint defect where, on integer-minute deltas
    with a large mass of exactly-one-minute gaps (dense units), the unbounded
    log-logistic/Weibull optimizer diverges (shape ~1e40+) and returns a
    degenerate "best fit" with iat_ks_stat ~= 1.0 that wins AIC. The guard must
    reject such fits so a real (bounded) family is selected.
    """

    def _dense_one_minute_deltas(self, seed=0):
        # ~55% exactly-one-minute gaps + a heavy tail: reproduces the divergence
        # regime seen in Zendesk A / Salesforce.
        rng = np.random.default_rng(seed)
        ones = np.ones(1100)
        tail = np.ceil(rng.lognormal(mean=1.5, sigma=1.2, size=900)).astype(float)
        deltas = np.concatenate([ones, tail])
        return pd.DataFrame({"events_delta": deltas})

    def test_no_divergent_shape_is_selected(self):
        td_df = self._dense_one_minute_deltas()
        feats = _compute_iat_fit_features(td_df)
        # The degenerate fit reported ks_stat ~= 1.0; a real fit is well below it.
        self.assertLess(feats["iat_ks_stat"], 0.99,
                        "divergent (ks_stat~=1.0) fit was not rejected")
        # And a real family must have been chosen (not the sentinel -1 = none).
        self.assertGreaterEqual(feats["iat_best_family"], 0.0)

    def test_clean_deltas_still_fit(self):
        # A well-behaved (non-divergent) sample: a real family fits with a
        # finite, non-degenerate ks_stat.
        rng = np.random.default_rng(1)
        deltas = np.ceil(rng.weibull(1.5, size=2000) * 10).astype(float)
        feats = _compute_iat_fit_features(pd.DataFrame({"events_delta": deltas}))
        self.assertGreaterEqual(feats["iat_best_family"], 0.0)
        self.assertLessEqual(feats["iat_ks_stat"], 1.0)
        self.assertTrue(np.isfinite(feats["iat_best_aic"]))

    def test_insufficient_data_returns_sentinel(self):
        feats = _compute_iat_fit_features(pd.DataFrame({"events_delta": [1.0, 2.0]}))
        self.assertEqual(feats["iat_best_family"], -1.0)


class TestBurstinessFeatures(unittest.TestCase):
    """Tests for burstiness metrics."""

    def test_uniform_events(self):
        epm_df = pd.DataFrame({
            "events_per_minute": [5.0] * 100,
        })
        features = _compute_burstiness_features(epm_df)
        self.assertAlmostEqual(features["events_per_min_mean"], 5.0)
        self.assertAlmostEqual(features["events_per_min_max"], 5.0)
        self.assertAlmostEqual(features["events_per_min_cov"], 0.0)
        self.assertAlmostEqual(features["burst_ratio"], 1.0)

    def test_bursty_events(self):
        values = [1.0] * 99 + [100.0]
        epm_df = pd.DataFrame({"events_per_minute": values})
        features = _compute_burstiness_features(epm_df)
        self.assertAlmostEqual(features["events_per_min_max"], 100.0)
        self.assertGreater(features["burst_ratio"], 10.0)
        self.assertGreater(features["events_per_min_cov"], 1.0)


class TestTemporalProfileFeatures(unittest.TestCase):
    """Tests for temporal profile features."""

    def test_uniform_profile(self):
        # All days/hours equal
        data = {}
        for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
            data[f'pct_changes_{day}'] = [14.28]
        for h in range(24):
            data[f'num_changes_hour_{h:02d}'] = [1.0]

        apr_df = pd.DataFrame(data)
        features, dow, hod = _compute_temporal_profile_features(apr_df)

        # High entropy for uniform distribution
        self.assertGreater(features["dow_entropy"], 2.5)
        self.assertGreater(features["hod_entropy"], 4.0)
        self.assertAlmostEqual(len(dow), 7)
        self.assertAlmostEqual(len(hod), 24)

    def test_concentrated_profile(self):
        # All activity on Monday, hour 9
        data = {"pct_changes_monday": [100.0]}
        for day in ['tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
            data[f'pct_changes_{day}'] = [0.0]
        for h in range(24):
            data[f'num_changes_hour_{h:02d}'] = [100.0 if h == 9 else 0.0]

        apr_df = pd.DataFrame(data)
        features, _, _ = _compute_temporal_profile_features(apr_df)

        self.assertAlmostEqual(features["dow_entropy"], 0.0)
        self.assertEqual(features["peak_dow"], 0.0)  # Monday
        self.assertEqual(features["peak_hour"], 9.0)
        self.assertAlmostEqual(features["weekend_fraction"], 0.0)


class TestLifecycleFeatures(unittest.TestCase):
    """Tests for record lifecycle features."""

    def test_basic_lifecycle(self):
        apr_df = pd.DataFrame({
            "time_first_created_to_last_modified": [100.0, 200.0, 300.0],
            "num_changes_total": [1, 5, 15],
            "was_created_3mo_before_cutoff": [True, False, True],
            "time_to_first_modification": [10.0, 20.0, 30.0],
        })

        features = _compute_lifecycle_features(apr_df)

        self.assertAlmostEqual(features["avg_record_lifespan"], 200.0)
        self.assertAlmostEqual(features["avg_creation_to_last_state"], 200.0)
        self.assertAlmostEqual(features["median_changes_per_record"], 5.0)
        self.assertAlmostEqual(features["pct_single_change_records"], 1/3)
        self.assertAlmostEqual(features["pct_records_gt_10_changes"], 1/3)
        self.assertAlmostEqual(features["pct_created_before_cutoff"], 2/3)
        self.assertAlmostEqual(features["avg_time_to_first_mod"], 20.0)


class TestPairwiseComparison(unittest.TestCase):
    """Tests for pairwise comparison computation."""

    def test_identical_fingerprints(self):
        features = {"f1": 1.0, "f2": 2.0, "f3": 3.0}
        fp1 = _make_fingerprint("a", features=features)
        fp2 = _make_fingerprint("b", features=features.copy())

        comparisons = compute_all_pairwise([fp1, fp2])

        self.assertEqual(len(comparisons), 1)
        self.assertAlmostEqual(comparisons[0].euclidean_distance, 0.0)
        self.assertAlmostEqual(comparisons[0].cosine_similarity, 1.0, places=5)

    def test_different_fingerprints(self):
        fp1 = _make_fingerprint("a", features={"f1": 0.0, "f2": 0.0, "f3": 0.0})
        fp2 = _make_fingerprint("b", features={"f1": 10.0, "f2": 10.0, "f3": 10.0})

        comparisons = compute_all_pairwise([fp1, fp2])

        self.assertEqual(len(comparisons), 1)
        self.assertGreater(comparisons[0].euclidean_distance, 0)

    def test_three_datasets(self):
        fp1 = _make_fingerprint("a", features={"f1": 1.0, "f2": 2.0})
        fp2 = _make_fingerprint("b", features={"f1": 3.0, "f2": 4.0})
        fp3 = _make_fingerprint("c", features={"f1": 5.0, "f2": 6.0})

        comparisons = compute_all_pairwise([fp1, fp2, fp3])
        self.assertEqual(len(comparisons), 3)  # 3 choose 2

    def test_single_fingerprint_no_comparisons(self):
        fp1 = _make_fingerprint("a")
        comparisons = compute_all_pairwise([fp1])
        self.assertEqual(len(comparisons), 0)

    def test_serialization(self):
        comp = PairwiseComparison(
            su_a="a", su_b="b",
            euclidean_distance=1.5,
            cosine_similarity=0.9,
        )
        d = comp.to_dict()
        comp2 = PairwiseComparison.from_dict(d)
        self.assertEqual(comp2.su_a, "a")
        self.assertAlmostEqual(comp2.euclidean_distance, 1.5)


class TestDistanceMatrix(unittest.TestCase):
    """Tests for distance matrix computation."""

    def test_symmetric_matrix(self):
        fp1 = _make_fingerprint("a", features={"f1": 1.0, "f2": 0.0})
        fp2 = _make_fingerprint("b", features={"f1": 0.0, "f2": 1.0})

        matrix, labels = compute_distance_matrix([fp1, fp2], metric="euclidean")

        self.assertEqual(labels, ["a", "b"])
        self.assertEqual(matrix.shape, (2, 2))
        self.assertAlmostEqual(matrix[0, 0], 0.0)
        self.assertAlmostEqual(matrix[0, 1], matrix[1, 0])  # Symmetric

    def test_cosine_distance(self):
        fp1 = _make_fingerprint("a", features={"f1": 1.0, "f2": 0.0})
        fp2 = _make_fingerprint("b", features={"f1": 1.0, "f2": 0.0})

        matrix, _ = compute_distance_matrix([fp1, fp2], metric="cosine")
        self.assertAlmostEqual(matrix[0, 1], 0.0, places=5)  # Identical vectors


class TestProfileJSD(unittest.TestCase):
    """Tests for Jensen-Shannon divergence on profiles."""

    def test_identical_profiles(self):
        fp1 = _make_fingerprint("a", dow_profile=[1/7]*7, hod_profile=[1/24]*24)
        fp2 = _make_fingerprint("b", dow_profile=[1/7]*7, hod_profile=[1/24]*24)

        results = _compute_profile_jsd([fp1, fp2])
        key = ("a", "b")
        dow_jsd, hod_jsd = results[key]

        self.assertAlmostEqual(dow_jsd, 0.0, places=3)
        self.assertAlmostEqual(hod_jsd, 0.0, places=3)

    def test_different_profiles(self):
        # Monday-only vs Friday-only
        fp1 = _make_fingerprint("a", dow_profile=[1, 0, 0, 0, 0, 0, 0])
        fp2 = _make_fingerprint("b", dow_profile=[0, 0, 0, 0, 1, 0, 0])

        results = _compute_profile_jsd([fp1, fp2])
        dow_jsd, _ = results[("a", "b")]

        self.assertGreater(dow_jsd, 0.5)


class TestDatasetClustering(unittest.TestCase):
    """Tests for dataset-level clustering."""

    def test_hierarchical_clustering(self):
        fps = [
            _make_fingerprint("a", features={"f1": 0.0, "f2": 0.0}),
            _make_fingerprint("b", features={"f1": 0.1, "f2": 0.1}),
            _make_fingerprint("c", features={"f1": 10.0, "f2": 10.0}),
            _make_fingerprint("d", features={"f1": 10.1, "f2": 10.1}),
        ]

        result = cluster_datasets(fps, method="hierarchical", n_clusters=2)

        self.assertEqual(result.method, "hierarchical")
        self.assertEqual(result.n_clusters, 2)
        self.assertEqual(len(result.labels), 4)
        # a and b should be in same cluster, c and d in another
        self.assertEqual(result.labels["a"], result.labels["b"])
        self.assertEqual(result.labels["c"], result.labels["d"])
        self.assertNotEqual(result.labels["a"], result.labels["c"])
        self.assertIsNotNone(result.linkage_matrix)
        self.assertIsNotNone(result.pca_2d)

    def test_kmeans_clustering(self):
        fps = [
            _make_fingerprint("a", features={"f1": 0.0, "f2": 0.0}),
            _make_fingerprint("b", features={"f1": 0.1, "f2": 0.1}),
            _make_fingerprint("c", features={"f1": 10.0, "f2": 10.0}),
        ]

        result = cluster_datasets(fps, method="kmeans", n_clusters=2)

        self.assertEqual(result.method, "kmeans")
        self.assertEqual(result.n_clusters, 2)
        self.assertIsNone(result.linkage_matrix)  # No dendrogram for kmeans

    def test_too_few_fingerprints(self):
        fp = _make_fingerprint("a")
        result = cluster_datasets([fp], n_clusters=2)
        self.assertEqual(result.n_clusters, 0)
        self.assertEqual(result.labels["a"], 0)

    def test_silhouette_positive(self):
        fps = [
            _make_fingerprint("a", features={"f1": 0.0, "f2": 0.0}),
            _make_fingerprint("b", features={"f1": 0.0, "f2": 0.0}),
            _make_fingerprint("c", features={"f1": 100.0, "f2": 100.0}),
            _make_fingerprint("d", features={"f1": 100.0, "f2": 100.0}),
        ]
        result = cluster_datasets(fps, n_clusters=2)
        self.assertGreater(result.silhouette, 0)


class TestFeatureImportance(unittest.TestCase):
    """Tests for feature importance computation."""

    def test_clear_separation(self):
        features = np.array([
            [0, 0],
            [0, 0],
            [10, 0],
            [10, 0],
        ], dtype=float)
        labels = np.array([0, 0, 1, 1])
        names = ["f1", "f2"]

        importances = _compute_feature_importance(features, labels, names)

        self.assertIn("f1", importances)
        self.assertGreater(importances["f1"], 0.5)

    def test_single_cluster_no_importance(self):
        features = np.array([[1, 2], [3, 4]])
        labels = np.array([0, 0])

        importances = _compute_feature_importance(features, labels, ["f1", "f2"])
        self.assertEqual(importances, {})


class TestDatasetClusterResultSerialization(unittest.TestCase):
    """Tests for DatasetClusterResult serialization."""

    def test_to_dict(self):
        result = DatasetClusterResult(
            method="hierarchical",
            n_clusters=2,
            labels={"a": 0, "b": 1},
            silhouette=0.75,
            feature_importances={"f1": 0.8},
        )
        d = result.to_dict()
        self.assertEqual(d["method"], "hierarchical")
        self.assertEqual(d["labels"]["a"], 0)
        self.assertAlmostEqual(d["silhouette"], 0.75)

    def test_roundtrip(self):
        result = DatasetClusterResult(
            method="hierarchical",
            n_clusters=2,
            labels={"a": 0, "b": 1},
            silhouette=0.75,
            feature_importances={"f1": 0.8},
            linkage_matrix=[[0, 1, 1.0, 2]],
            pca_2d={"a": [0.1, 0.2], "b": [0.3, 0.4]},
            tsne_2d={"a": [0.5, 0.6], "b": [0.7, 0.8]},
        )
        d = result.to_dict()
        restored = DatasetClusterResult.from_dict(d)
        self.assertEqual(restored.method, "hierarchical")
        self.assertEqual(restored.n_clusters, 2)
        self.assertEqual(restored.labels, {"a": 0, "b": 1})
        self.assertAlmostEqual(restored.silhouette, 0.75)
        self.assertEqual(restored.feature_importances, {"f1": 0.8})
        self.assertEqual(restored.linkage_matrix, [[0, 1, 1.0, 2]])
        self.assertEqual(restored.pca_2d["a"], [0.1, 0.2])
        self.assertEqual(restored.tsne_2d["b"], [0.7, 0.8])


if __name__ == "__main__":
    unittest.main()
