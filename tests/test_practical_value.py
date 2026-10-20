"""Synthetic contracts for practical policy comparisons."""

import unittest

from analysis import thesis_practical_value as practical


def candidate(family, policy, sync, freshness):
    """A thousand-minute synthetic trace with consistent counts and rates."""
    return {"unit": "fixture", "family": family, "policy": policy,
            "family_eligible": True, "steps": 1000, "timestamp_digest": "timestamps",
            "event_digest": "events", "initial_true_aoi": 0,
            "decisions_path": f"fixture/{policy}/decisions.parquet",
            "sync_rate": sync, "sync_count": round(sync * 1000),
            "empty_sync_step_rate": sync, "empty_sync_count": round(sync * 1000),
            "cvar95_aoii": freshness, "cvar95_true_aoi": freshness}


def comparisons(rows, mode="raw"):
    """Build the full grid for one synthetic unit/objective pair."""
    prepared, bounds = practical.prepare_group(
        practical.eligible_candidates(rows), "sync_vs_aoii", mode)
    return [{"unit": "fixture", "objective_pair": "sync_vs_aoii", "normalization": mode,
             **practical.compare_at_weight(prepared, bounds, weight)} for weight in range(101)]


class TestPracticalValueContracts(unittest.TestCase):
    """Guard signs, population, ties, normalization and metadata validation."""

    def test_dominated_family_comparator_is_retained(self):
        """Do not lose the family comparator just because it is dominated."""
        rows = [candidate("RL", "rl", .5, 5), candidate("periodic", "p", .2, 2)]
        result = comparisons(rows)[50]
        self.assertEqual(result["rl_candidates"][0]["policy"], "rl")
        self.assertFalse(result["rl_candidates"][0]["is_pareto"])
        self.assertEqual(result["score_gap_P_minus_RL"], -1)
        self.assertEqual(result["winner_family"], "periodic")
        self.assertAlmostEqual(result["delta_syncs_per_1440min"], 432)

    def test_shared_bounds_and_log_transform(self):
        """Both families use one pool; log1p precedes freshness normalization."""
        rows = practical.eligible_candidates([
            candidate("RL", "rl", .25, 3), candidate("periodic", "p", .75, 15),
            candidate("periodic", "middle", .5, 7)])
        prepared, bounds = practical.prepare_group(rows, "sync_vs_aoii", "log1p")
        middle = next(r for r in prepared if r["policy"] == "middle")
        self.assertAlmostEqual(middle["normalized_operational"], .5)
        self.assertAlmostEqual(middle["normalized_freshness"], .5)
        self.assertEqual((bounds["operational_min"], bounds["operational_max"]), (.25, .75))
        raw, _ = practical.prepare_group(rows, "sync_vs_aoii", "raw")
        self.assertAlmostEqual(next(r for r in raw if r["policy"] == "middle")
                               ["normalized_freshness"], 1 / 3)
        with self.assertRaises(ValueError):
            practical.prepare_group(rows, "sync_vs_aoii", "unknown")

    def test_tied_raw_alternatives_are_not_averaged(self):
        """Equal scores can correspond to different raw operating points."""
        rows = [candidate("RL", "r1", .2, 8), candidate("RL", "r2", .8, 2),
                candidate("periodic", "p", .5, 5)]
        pairs = comparisons(rows)
        result = pairs[50]
        self.assertEqual(len(result["rl_candidates"]), 2)
        self.assertEqual(result["family_score_winner"], "mixed_tie")
        self.assertIsNone(result["delta_syncs_per_1440min"])
        self.assertAlmostEqual(result["delta_syncs_per_1440min_min"], -432)
        self.assertAlmostEqual(result["delta_syncs_per_1440min_max"], 432)
        with self.assertRaisesRegex(ValueError, "Ambiguous interior"):
            practical.summarize(pairs)

    def test_endpoint_frontier_rule_is_separate_from_family_ties(self):
        """At zero weight a dominated comparator may tie without being a winner."""
        rows = [candidate("RL", "rl", .5, 0), candidate("periodic", "p", .2, 0)]
        result = comparisons(rows)[0]
        self.assertEqual(result["family_score_winner"], "mixed_tie")
        self.assertEqual(result["winner_family"], "periodic")
        self.assertEqual(result["winner_policies"], "periodic:p")

    def test_constant_zero_axes_and_family_tie_credit(self):
        """Zero denominators are safe and policy multiplicity does not bias credit."""
        rows = [candidate("RL", "r1", 0, 0), candidate("RL", "r2", 0, 0),
                candidate("periodic", "p", 0, 0)]
        pairs = comparisons(rows, "log1p")
        summary = practical.summarize(pairs)[0]
        self.assertEqual(summary["mixed_tie_weights"], 81)
        self.assertEqual(summary["rl_fractional_share"], .5)
        self.assertEqual(summary["delta_syncs_per_1440min_median"], 0)

    def test_all_interior_weights_include_periodic_wins(self):
        """Use 81 weights, not just positive gaps, with inclusive quartiles."""
        pairs = comparisons([candidate("RL", "rl", .2, 8),
                             candidate("periodic", "p", .8, 2)])
        self.assertEqual(len(pairs), 101)
        self.assertEqual(sum(r["is_interior"] for r in pairs), 81)
        self.assertLess(pairs[10]["score_gap_P_minus_RL"], 0)
        self.assertGreater(pairs[90]["score_gap_P_minus_RL"], 0)
        summary = practical.summarize(pairs)[0]
        self.assertEqual((summary["RL_weights"], summary["periodic_weights"]), (40, 40))
        self.assertEqual(summary["mixed_tie_weights"], 1)
        for suffix, expected in (("q25", -.4), ("median", 0), ("q75", .4)):
            self.assertAlmostEqual(summary[f"score_gap_P_minus_RL_{suffix}"], expected)
        for malformed in (pairs[11:], pairs + [pairs[50]]):
            with self.assertRaisesRegex(ValueError, "every interior weight"):
                practical.summarize(malformed)

    def test_trace_identity_mismatches_are_rejected(self):
        """Every required trace-identity field must agree within a unit."""
        for key, value in (("steps", 2000), ("timestamp_digest", "different"),
                           ("event_digest", "different"), ("initial_true_aoi", 1)):
            with self.subTest(field=key):
                rl = candidate("RL", "rl", 0, 0)
                periodic = candidate("periodic", "p", 0, 0)
                periodic[key] = value
                with self.assertRaisesRegex(ValueError, f"Mismatched trace {key}"):
                    practical.eligible_candidates([rl, periodic])

    def test_invalid_metrics_duplicates_and_eligibility(self):
        """Reject malformed included records without promoting excluded runs."""
        rl, periodic = candidate("RL", "rl", .2, 3), candidate("periodic", "p", .8, 5)
        for key, value in (("cvar95_aoii", float("nan")), ("cvar95_true_aoi", -1),
                           ("steps", 0), ("sync_rate", 1.1), ("sync_count", 999)):
            with self.subTest(field=key, value=value), self.assertRaises(ValueError):
                practical.eligible_candidates([{**rl, key: value}, periodic])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            practical.eligible_candidates([rl, rl, periodic])
        excluded = {**rl, "policy": "excluded", "family_eligible": False, "sync_rate": 0}
        self.assertEqual(len(practical.eligible_candidates([rl, periodic, excluded])), 2)


if __name__ == "__main__":
    unittest.main()
