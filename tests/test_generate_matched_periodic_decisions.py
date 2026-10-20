"""Tests for deterministic matched-window periodic decision generation."""

import unittest

import numpy as np
import pandas as pd

from analysis.generate_matched_periodic_decisions import (
    build_periodic_decisions,
    periodic_actions,
)


class TestMatchedPeriodicDecisionGeneration(unittest.TestCase):
    """Verify periodic alignment, AoI reconstruction, and invalid intervals."""

    def test_periodic_actions_match_policy_alignment(self):
        """The first sync occurs at interval minus one, as in PeriodicSyncPolicy."""
        np.testing.assert_array_equal(periodic_actions(5, 1), [1, 1, 1, 1, 1])
        np.testing.assert_array_equal(periodic_actions(7, 5), [0, 0, 0, 0, 1, 0, 0])

    def test_build_periodic_decisions_recomputes_true_aoi(self):
        """Generated actions reset true AoI on simultaneous syncs."""
        template = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="min", tz="UTC"),
            "step": np.arange(5),
            "actual_value": [1.0, 0.0, 0.0, 1.0, 0.0],
        })

        decisions = build_periodic_decisions(template, interval=3)

        np.testing.assert_array_equal(decisions["action"], [0, 0, 1, 0, 0])
        np.testing.assert_array_equal(decisions["true_aoi"], [0.0, 1.0, 0.0, 0.0, 1.0])
        self.assertEqual(
            list(decisions.columns),
            ["timestamp", "step", "action", "actual_value", "true_aoi"],
        )

    def test_interval_must_be_positive(self):
        """Reject zero-length and negative periodic intervals."""
        with self.assertRaises(ValueError):
            periodic_actions(5, 0)


if __name__ == "__main__":
    unittest.main()
