"""Tests for reusable Pareto and objective-weight policy analysis."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.multiobjective_policy_analysis import (
    build_candidate_objectives,
    build_multiobjective_results,
    build_two_objective_candidates,
    build_two_objective_results,
    generate_weight_grid,
    generate_two_objective_weight_grid,
    generate_two_objective_weight_plane,
    minmax_normalize,
    pareto_mask,
    score_two_objective_scenarios,
    score_two_objective_weight_plane,
    score_priority_scenarios,
    summarize_two_objective_winner_ranges,
    write_multiobjective_outputs,
)


def candidate_frame(rows) -> pd.DataFrame:
    """Build the minimal reusable candidate-metrics input table."""
    return pd.DataFrame(rows, columns=[
        "unit", "family", "policy", "decisions_path", "sync_rate",
        "empty_sync_step_rate", "cvar95_aoii", "cvar95_true_aoi",
        "family_eligible",
    ])


class TestParetoAnalysis(unittest.TestCase):
    """Verify dominance, normalization, and candidate preparation."""

    def test_dominated_and_tradeoff_candidates(self):
        """A worse row is removed while a request-freshness trade-off remains."""
        values = np.array([
            [0.10, 0.05, 1.0],
            [0.20, 0.10, 2.0],
            [0.05, 0.02, 5.0],
        ])

        np.testing.assert_array_equal(
            pareto_mask(values), np.array([True, False, True]),
        )

    def test_identical_candidates_both_remain(self):
        """Equal objective vectors should not be resolved by policy name."""
        values = np.array([[0.1, 0.0, 1.0], [0.1, 0.0, 1.0]])

        np.testing.assert_array_equal(pareto_mask(values), [True, True])

    def test_constant_normalization_maps_to_zero(self):
        """An objective with no candidate variation should not affect scores."""
        np.testing.assert_array_equal(
            minmax_normalize([3.0, 3.0, 3.0]), np.zeros(3),
        )

    def test_family_ineligible_candidate_is_excluded(self):
        """Canonical-only candidates must not enter family optimization."""
        frame = candidate_frame([
            ["U", "RL", "eligible", "a", 0.2, 0.1, 2.0, 2.0, True],
            ["U", "RL", "excluded", "b", 0.1, 0.0, 1.0, 1.0, False],
            ["U", "periodic", "periodic", "c", 0.3, 0.2, 3.0, 3.0, True],
        ])

        result = build_candidate_objectives(frame, normalizations=["raw"])

        self.assertEqual(set(result["policy"]), {"eligible", "periodic"})


class TestWeightAnalysis(unittest.TestCase):
    """Verify simplex generation, priorities, ties, and stable output."""

    def setUp(self):
        self.frame = candidate_frame([
            ["U", "RL", "request", "a", 0.1, 0.1, 1.0, 1.0, True],
            ["U", "periodic", "empty", "b", 0.2, 0.0, 1.0, 1.0, True],
            ["U", "RL", "fresh", "c", 0.3, 0.1, 0.0, 0.0, True],
        ])

    def test_weight_grid_sizes_and_sums(self):
        """The 0.05 simplex grid should contain complete and interior sweeps."""
        grid = generate_weight_grid(0.05)

        self.assertEqual(len(grid), 231)
        self.assertEqual(int(grid["is_interior"].sum()), 120)
        np.testing.assert_allclose(grid[[
            "weight_sync", "weight_empty", "weight_freshness",
        ]].sum(axis=1), 1.0)

    def test_named_scenarios_change_winner(self):
        """Focused weights should select candidates strong on that objective."""
        candidates = build_candidate_objectives(
            self.frame, normalizations=["raw"],
        )
        scenarios = score_priority_scenarios(candidates)
        aoii = scenarios.query("freshness_variant == 'aoii'").set_index("scenario")

        self.assertEqual(aoii.loc["request_focused", "winner_policies"], "RL:request")
        self.assertEqual(
            aoii.loc["empty_sync_focused", "winner_policies"], "periodic:empty",
        )
        self.assertEqual(aoii.loc["freshness_focused", "winner_policies"], "RL:fresh")

    def test_mixed_ties_receive_equal_family_credit(self):
        """An exact cross-family tie should not be broken alphabetically."""
        tied = candidate_frame([
            ["U", "RL", "rl", "a", 0.1, 0.0, 1.0, 1.0, True],
            ["U", "periodic", "periodic", "b", 0.1, 0.0, 1.0, 1.0, True],
        ])

        results = build_multiobjective_results(tied, normalizations=["raw"])
        sweep = results["weight_sweep_winners"]
        shares = results["weight_space_family_share"]

        self.assertTrue((sweep["winner_family"] == "mixed_tie").all())
        self.assertTrue(np.allclose(shares["rl_fractional_share"], 0.5))
        self.assertTrue(np.allclose(shares["periodic_fractional_share"], 0.5))
        self.assertTrue(np.allclose(shares["mixed_tie_share"], 1.0))

    def test_results_are_stable_under_input_reordering(self):
        """Candidate input order must not change machine-readable evidence."""
        first = build_multiobjective_results(self.frame, normalizations=["raw"])
        second = build_multiobjective_results(
            self.frame.iloc[::-1].reset_index(drop=True), normalizations=["raw"],
        )

        pd.testing.assert_frame_equal(
            first["weight_sweep_winners"], second["weight_sweep_winners"],
        )

    def test_dominated_candidate_never_wins(self):
        """Weight scoring must only consider objective-efficient candidates."""
        dominated = candidate_frame([
            *self.frame.to_numpy().tolist(),
            ["U", "periodic", "dominated", "d", 0.4, 0.2, 2.0, 2.0, True],
        ])

        results = build_multiobjective_results(dominated, normalizations=["raw"])

        self.assertFalse(
            results["weight_sweep_winners"]["winner_policies"]
            .str.contains("periodic:dominated")
            .any()
        )

    def test_writes_every_result_table(self):
        """The reusable writer should persist every returned evidence table."""
        results = build_multiobjective_results(self.frame, normalizations=["raw"])
        with tempfile.TemporaryDirectory() as directory:
            write_multiobjective_outputs(results, Path(directory))

            self.assertEqual(
                {path.stem for path in Path(directory).glob("*.csv")},
                set(results),
            )


class TestTwoObjectiveAnalysis(unittest.TestCase):
    """Verify independent cost/freshness variants and one-dimensional weights."""

    def setUp(self):
        self.frame = candidate_frame([
            ["U", "RL", "low_sync", "a", 0.1, 0.09, 10.0, 10.0, True],
            ["U", "periodic", "fresh", "b", 0.2, 0.01, 5.0, 5.0, True],
            ["U", "RL", "low_empty", "c", 0.3, 0.00, 8.0, 8.0, True],
        ])

    def test_pair_specific_frontiers_do_not_double_count_requests(self):
        """A policy may be dominated for total requests but efficient for waste."""
        candidates = build_two_objective_candidates(
            self.frame, normalizations=["raw"],
        )
        frontiers = candidates[candidates["is_pareto"]].groupby(
            "objective_pair",
        )["policy"].apply(set)

        self.assertEqual(frontiers["sync_vs_aoii"], {"low_sync", "fresh"})
        self.assertEqual(frontiers["empty_vs_aoii"], {"fresh", "low_empty"})
        self.assertEqual(frontiers["sync_vs_aoi"], {"low_sync", "fresh"})
        self.assertEqual(frontiers["empty_vs_aoi"], {"fresh", "low_empty"})

    def test_weight_grid_is_complete_and_has_non_extreme_scope(self):
        """A 0.01 line sweep has 101 complete and 81 non-extreme points."""
        grid = generate_two_objective_weight_grid(0.01)

        self.assertEqual(len(grid), 101)
        self.assertEqual(int(grid["is_interior"].sum()), 81)
        np.testing.assert_allclose(
            grid[["weight_operational", "weight_freshness"]].sum(axis=1), 1.0,
        )

    def test_weight_plane_normalizes_independent_axis_priorities(self):
        """The plane omits its undefined origin and preserves priority ratios."""
        plane = generate_two_objective_weight_plane(0.05)

        self.assertEqual(len(plane), 440)
        self.assertFalse((
            (plane["operational_priority"] == 0)
            & (plane["freshness_priority"] == 0)
        ).any())
        np.testing.assert_allclose(
            plane[["weight_operational", "weight_freshness"]].sum(axis=1), 1.0,
        )
        equal = plane[
            (plane["operational_priority"] == 1)
            & (plane["freshness_priority"] == 1)
        ].iloc[0]
        self.assertEqual(equal["weight_operational"], 0.5)
        self.assertEqual(equal["weight_freshness"], 0.5)

    def test_weight_plane_agrees_with_equivalent_named_scenario(self):
        """Independent priorities must produce the same normalized-ratio winner."""
        candidates = build_two_objective_candidates(
            self.frame, normalizations=["raw"],
        )
        plane = score_two_objective_weight_plane(candidates, step=0.2)
        scenarios = score_two_objective_scenarios(candidates)
        plane_row = plane.query(
            "objective_pair == 'sync_vs_aoii' "
            "and operational_priority == 0.2 and freshness_priority == 0.8"
        ).iloc[0]
        scenario_row = scenarios.query(
            "objective_pair == 'sync_vs_aoii' and scenario == 'freshness_focused'"
        ).iloc[0]

        self.assertEqual(plane_row["winner_policies"], scenario_row["winner_policies"])

    def test_named_scenarios_select_opposite_tradeoff_ends(self):
        """Focused scenarios should select the corresponding frontier end."""
        simple = candidate_frame([
            ["U", "RL", "cheap", "a", 0.0, 0.0, 10.0, 10.0, True],
            ["U", "periodic", "fresh", "b", 1.0, 1.0, 0.0, 0.0, True],
        ])
        candidates = build_two_objective_candidates(simple, normalizations=["raw"])
        scenarios = score_two_objective_scenarios(candidates)
        sync_pair = scenarios.query("objective_pair == 'sync_vs_aoii'").set_index(
            "scenario",
        )

        self.assertEqual(sync_pair.loc["freshness_focused", "winner_policies"], "periodic:fresh")
        self.assertEqual(sync_pair.loc["operational_focused", "winner_policies"], "RL:cheap")
        self.assertEqual(sync_pair.loc["balanced", "winner_family"], "mixed_tie")

        results = build_two_objective_results(
            simple, weight_step=0.5, normalizations=["raw"],
        )
        share = results["two_objective_weight_space_family_share"].query(
            "objective_pair == 'sync_vs_aoii' and grid_scope == 'full'",
        ).iloc[0]
        self.assertAlmostEqual(share["rl_fractional_share"], 0.5)
        self.assertAlmostEqual(share["periodic_fractional_share"], 0.5)

    def test_family_tie_credit_ignores_number_of_tied_policies(self):
        """Mixed-family credit should be 50/50 even with more tied RL policies."""
        tied = candidate_frame([
            ["U", "RL", "rl_a", "a", 0.1, 0.1, 1.0, 1.0, True],
            ["U", "RL", "rl_b", "b", 0.1, 0.1, 1.0, 1.0, True],
            ["U", "periodic", "periodic", "c", 0.1, 0.1, 1.0, 1.0, True],
        ])

        results = build_two_objective_results(
            tied, weight_step=0.5, normalizations=["raw"],
        )
        shares = results["two_objective_weight_space_family_share"]

        self.assertTrue(np.allclose(shares["rl_fractional_share"], 0.5))
        self.assertTrue(np.allclose(shares["periodic_fractional_share"], 0.5))

    def test_endpoint_winners_are_restricted_to_raw_frontier(self):
        """An ignored objective must not reintroduce a dominated endpoint tie."""
        frame = candidate_frame([
            ["U", "RL", "dominant", "a", 0.1, 0.1, 0.0, 0.0, True],
            ["U", "periodic", "dominated", "b", 0.1, 0.1, 5.0, 5.0, True],
        ])
        results = build_two_objective_results(
            frame, weight_step=1.0, normalizations=["raw"],
        )
        operational_endpoint = results["two_objective_weight_sweep_winners"].query(
            "objective_pair == 'sync_vs_aoii' and weight_operational == 1.0"
        ).iloc[0]

        self.assertEqual(operational_endpoint["winner_policies"], "RL:dominant")

    def test_normalization_bounds_record_extreme_candidates(self):
        """Preference specifications should export their pool-defining bounds."""
        results = build_two_objective_results(
            self.frame, weight_step=0.5, normalizations=["raw"],
        )
        bounds = results["two_objective_normalization_bounds"]
        row = bounds.query("objective_pair == 'sync_vs_aoii'").iloc[0]

        self.assertEqual(row["operational_min_candidates"], "RL:low_sync")
        self.assertEqual(row["freshness_min_candidates"], "periodic:fresh")

    def test_two_objective_results_are_stable_under_input_reordering(self):
        """Input ordering must not alter the two-objective evidence tables."""
        first = build_two_objective_results(self.frame, normalizations=["raw"])
        second = build_two_objective_results(
            self.frame.iloc[::-1].reset_index(drop=True), normalizations=["raw"],
        )

        for name, frame in first.items():
            pd.testing.assert_frame_equal(frame, second[name])

    def test_winner_ranges_collapse_only_consecutive_identical_winners(self):
        """Sampled winner intervals preserve family and policy transitions."""
        results = build_two_objective_results(
            self.frame, weight_step=0.5, normalizations=["raw"],
        )
        sweep = results["two_objective_weight_sweep_winners"].query(
            "objective_pair == 'sync_vs_aoii'",
        )
        ranges = summarize_two_objective_winner_ranges(sweep)

        self.assertEqual(int(ranges["sampled_weight_count"].sum()), len(sweep))
        self.assertTrue((
            ranges["operational_weight_min"] <= ranges["operational_weight_max"]
        ).all())

    def test_two_objective_results_are_written_with_distinct_names(self):
        """The variation should preserve all eight separately named artifacts."""
        results = build_two_objective_results(
            self.frame, weight_step=0.5, normalizations=["raw"],
        )
        with tempfile.TemporaryDirectory() as directory:
            write_multiobjective_outputs(results, Path(directory))

            self.assertEqual(
                {path.stem for path in Path(directory).glob("*.csv")}, set(results),
            )
            self.assertEqual(len(results), 9)


if __name__ == "__main__":
    unittest.main()
