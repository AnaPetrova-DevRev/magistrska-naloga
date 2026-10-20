"""Tests for parameterized block-weighted dataset comparison."""

import unittest

import numpy as np

from dataset_comparison.weighted_comparison import _cluster, _selected_features


class TestWeightedComparisonRuns(unittest.TestCase):
    """Validate feature inclusion for the three sensitivity variants."""

    def test_lifecycle_excluded_run_omits_both_sensitive_blocks(self):
        """Primary robustness run excludes volume and lifecycle features."""
        names = [name for name, _ in _selected_features("lifecycle_excluded")]

        self.assertNotIn("raw_events", names)
        self.assertNotIn("avg_record_lifespan", names)
        self.assertIn("delta_mean", names)

    def test_behavioral_run_keeps_lifecycle_but_not_volume(self):
        """Protocol-preserving behavioral run still includes lifecycle."""
        names = [name for name, _ in _selected_features("behavioral")]

        self.assertNotIn("raw_events", names)
        self.assertIn("avg_creation_to_last_state", names)
        self.assertNotIn("avg_record_lifespan", names)

    def test_full_run_keeps_all_available_blocks(self):
        """Full run includes the volume, behavioral, and lifecycle features."""
        names = [name for name, _ in _selected_features("full")]

        self.assertIn("raw_events", names)
        self.assertIn("avg_creation_to_last_state", names)
        self.assertIn("delta_mean", names)

    def test_clustering_is_deterministic(self):
        """Ward and seeded K-means return identical outputs on repeated calls."""
        matrix = np.array(
            [[0.0, 0.0], [0.1, 0.2], [3.0, 3.0], [3.2, 3.1], [6.0, 0.0]]
        )
        names = ["a", "b", "c", "d", "e"]

        first = _cluster(matrix, names)
        second = _cluster(matrix, names)

        self.assertEqual(first["silhouette_sweep"], second["silhouette_sweep"])
        self.assertEqual(first["labels"], second["labels"])


if __name__ == "__main__":
    unittest.main()
