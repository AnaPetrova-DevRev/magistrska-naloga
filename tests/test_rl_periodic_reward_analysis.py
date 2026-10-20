"""Tests for the authoritative RL-versus-periodic trace analysis."""

# pylint: disable=too-many-locals

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.generate_matched_periodic_decisions import (
    INTERVALS,
    periodic_actions,
)
from analysis.rl_periodic_reward_analysis import (
    CandidateRef,
    UnitSpec,
    derive_sync_items,
    empirical_cvar,
    fixed_pair_break_even,
    load_manifest,
    run_analysis,
    summarize_decisions,
    validate_same_window,
    write_artifact_manifest,
    write_outputs,
)


def decision_frame(actual, actions, true_aoi, start="2025-01-01"):
    """Build a small event-rate decision trace."""
    return pd.DataFrame({
        "timestamp": pd.date_range(start, periods=len(actual), freq="min"),
        "action": actions,
        "actual_value": actual,
        "true_aoi": true_aoi,
    })


class TestTraceMetrics(unittest.TestCase):
    """Verify reward decomposition and operational diagnostics."""

    def test_derive_sync_items_accumulates_until_sync(self):
        """Pending event counts should be returned and reset at sync steps."""
        actual = np.array([0.0, 2.0, 0.0, 1.0, 0.0, 0.0])
        actions = np.array([0, 0, 1, 0, 0, 1])
        np.testing.assert_array_equal(
            derive_sync_items(actual, actions),
            np.array([0.0, 0.0, 2.0, 0.0, 0.0, 1.0]),
        )

    def test_requested_periodic_intervals_and_two_minute_cadence(self):
        """The periodic family should contain every requested interval."""
        self.assertEqual(INTERVALS, (1, 2, 5, 10, 15, 20, 30, 45, 60))
        np.testing.assert_array_equal(
            periodic_actions(6, 2),
            np.array([0, 1, 0, 1, 0, 1]),
        )

    def test_summary_separates_true_aoi_and_aoii(self):
        """AoII should weight pending volume rather than duplicate true AoI."""
        frame = decision_frame(
            [0, 2, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
        )
        result = summarize_decisions(frame, thresholds=[0, 1])

        self.assertEqual(result.successful_sync_count, 2)
        self.assertEqual(result.empty_sync_count, 0)
        self.assertEqual(result.late_skip_count, 2)
        self.assertAlmostEqual(result.avg_true_aoi, 1 / 6)
        self.assertAlmostEqual(result.avg_aoii, 5 / 6)
        self.assertAlmostEqual(result.max_aoii, 2.0)
        self.assertAlmostEqual(result.cvar95_true_aoi, 1.0)
        self.assertAlmostEqual(result.cvar95_aoii, 2.0)
        self.assertAlmostEqual(result.avg_reaction_time, 1.5)
        self.assertAlmostEqual(result.p95_reaction_time, 1.95)
        self.assertAlmostEqual(result.p95_true_aoi_when_stale, 1.0)
        self.assertAlmostEqual(result.standard_reward_mean(), -5.02 / 6)

    def test_all_minute_p95_can_hide_sparse_staleness(self):
        """Conditional P95 should expose a sparse nonzero staleness tail."""
        true_aoi = np.zeros(100)
        true_aoi[-4:] = [1, 2, 3, 4]
        frame = decision_frame(np.zeros(100), np.zeros(100), true_aoi)
        result = summarize_decisions(frame, thresholds=[0])

        self.assertEqual(result.p95_true_aoi, 0.0)
        self.assertGreater(result.p95_true_aoi_when_stale, 3.0)

    def test_empirical_cvar_uses_exact_worst_five_percent(self):
        """Sparse percentile ties must not include more than the fixed tail."""
        values = np.zeros(100)
        values[-5:] = [6, 7, 8, 9, 10]

        self.assertAlmostEqual(empirical_cvar(values), 8.0)
        self.assertAlmostEqual(empirical_cvar(np.array([1.0, 3.0])), 3.0)

    def test_thresholds_do_not_mutate_large_true_aoi_array(self):
        """Operational thresholds must not alter canonical reward statistics."""
        steps = 260_000
        true_aoi = np.zeros(steps)
        true_aoi[::10] = 2.0
        frame = decision_frame(np.zeros(steps), np.zeros(steps), true_aoi)

        baseline = summarize_decisions(frame, thresholds=[0])
        result = summarize_decisions(frame, thresholds=[0, 1, 5])

        self.assertEqual(result.true_aoi_sum, baseline.true_aoi_sum)
        self.assertEqual(result.standard_reward_mean(), baseline.standard_reward_mean())
        self.assertEqual(result.mean_excess_aoi, {0.0: 0.2, 1.0: 0.1, 5.0: 0.0})

    def test_empty_sync_is_counted_without_last_sync_items_column(self):
        """Minimal generated periodic traces should support empty-sync metrics."""
        frame = decision_frame([0, 0, 1], [1, 0, 1], [0, 0, 0])
        result = summarize_decisions(frame, thresholds=[0])

        self.assertEqual(result.sync_count, 2)
        self.assertEqual(result.empty_sync_count, 1)
        self.assertEqual(result.successful_sync_count, 1)
        self.assertAlmostEqual(result.empty_sync_rate, 0.5)
        self.assertAlmostEqual(result.empty_sync_step_rate, 1 / 3)


class TestBreakEvenAnalysis(unittest.TestCase):
    """Verify fixed-pair switching costs and window checks."""

    def test_fresher_more_frequent_rl_has_positive_break_even(self):
        """A fresher but more frequent RL trace should switch at finite cost."""
        rl = summarize_decisions(
            decision_frame([0] * 4, [1, 0, 1, 0], [0, 0, 0, 0]),
            thresholds=[0],
        )
        periodic = summarize_decisions(
            decision_frame([0] * 4, [0, 0, 0, 1], [0, 1, 2, 0]),
            thresholds=[0],
        )
        result = fixed_pair_break_even(rl, periodic, 0)

        self.assertAlmostEqual(result.break_even_cost, 3.0)
        self.assertEqual(result.winner_below, "RL")
        self.assertEqual(result.winner_above, "periodic")

    def test_equal_rate_lower_aoi_has_no_break_even(self):
        """Lower AoI at equal request rate should favor RL at every cost."""
        rl = summarize_decisions(
            decision_frame([0] * 4, [0, 0, 0, 1], [0, 0, 1, 0]),
            thresholds=[0],
        )
        periodic = summarize_decisions(
            decision_frame([0] * 4, [0, 0, 0, 1], [0, 1, 2, 0]),
            thresholds=[0],
        )
        result = fixed_pair_break_even(rl, periodic, 0)

        self.assertIsNone(result.break_even_cost)
        self.assertEqual(result.winner_below, "RL")
        self.assertEqual(result.winner_above, "RL")

    def test_window_validation_checks_exact_timestamps(self):
        """Equal trace lengths must not mask different chronological windows."""
        first = summarize_decisions(
            decision_frame([0, 0], [0, 1], [0, 0]), thresholds=[0],
        )
        second = summarize_decisions(
            decision_frame([0, 0], [0, 1], [0, 0], start="2025-02-01"),
            thresholds=[0],
        )
        with self.assertRaisesRegex(ValueError, "identical timestamps"):
            validate_same_window([first, second], expected_steps=2)

    def test_window_validation_checks_exact_event_sequence(self):
        """Equal timestamps must not mask different underlying event inputs."""
        first = summarize_decisions(
            decision_frame([0, 1], [0, 1], [0, 0]), thresholds=[0],
        )
        second = summarize_decisions(
            decision_frame([0, 2], [0, 1], [0, 0]), thresholds=[0],
        )

        with self.assertRaisesRegex(ValueError, "identical event sequences"):
            validate_same_window([first, second], expected_steps=2)

    def test_window_validation_checks_initial_true_aoi(self):
        """Equal event inputs must not mask a different initial freshness state."""
        first = summarize_decisions(
            decision_frame([0, 0], [0, 0], [0, 0]), thresholds=[0],
        )
        second = summarize_decisions(
            decision_frame([0, 0], [0, 0], [1, 0]), thresholds=[0],
        )

        with self.assertRaisesRegex(ValueError, "identical initial true AoI"):
            validate_same_window([first, second], expected_steps=2)


class TestManifestAndPipeline(unittest.TestCase):
    """Verify manifest exclusions and end-to-end synthetic table generation."""

    def test_manifest_keeps_fixed_run_but_excludes_it_from_family(self):
        """Generated periodic traces should replace the legacy periodic inventory."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                ["Example", 2, "RL", "ppo_seed123", "fixed.parquet"],
                ["Example", 2, "RL", "ppo_base", "base.parquet"],
                ["Example", 2, "RL", "ppo_protocol_pilot10k", "pilot.parquet"],
                ["Example", 2, "RL", "ppo_protocol_smoke1k", "smoke.parquet"],
                ["Example", 2, "RL", "ppo_protocol_gate100k", "gate.parquet"],
                ["Example", 2, "periodic", "periodic_15min", "legacy.parquet"],
            ]
            pd.DataFrame(rows, columns=[
                "unit", "matched_steps", "family", "policy", "decisions_path",
            ]).to_csv(root / "inventory.csv", index=False)
            periodic_records = [
                {
                    "unit_path": "example/unit/path",
                    "matched_steps": 2,
                    "interval_minutes": value,
                    "decisions": f"generated/p{value}.parquet",
                }
                for value in INTERVALS
            ]
            (root / "periodic.json").write_text(
                json.dumps(periodic_records), encoding="utf-8",
            )
            manifest = {
                "code_root": ".",
                "candidate_inventory_csv": "inventory.csv",
                "periodic_decisions_manifest": "periodic.json",
                "family_exclude_patterns": [
                    "_seed[0-9]+$",
                    "(?:^|_)(?:pilot|smoke|gate)[a-z0-9]*(?:_|$)",
                ],
                "units": [{
                    "name": "Example", "unit_path": "example/unit/path",
                    "expected_steps": 2,
                    "expected_margin_pct": 0,
                    "fixed_pair": {"rl": "ppo_seed123", "periodic": "periodic_15min"},
                }],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            _, units = load_manifest(root / "manifest.json")

            self.assertEqual(units[0].fixed_rl.policy, "ppo_seed123")
            self.assertEqual([item.policy for item in units[0].family_rl], ["ppo_base"])
            self.assertEqual(len(units[0].family_periodic), 9)
            self.assertEqual(
                units[0].fixed_periodic.path,
                (root / "generated/p15.parquet").resolve(),
            )

    def test_manifest_prefers_frozen_family_eligibility(self):
        """An inventory eligibility column overrides legacy policy-name patterns."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                ["Example", 2, "RL", "canonical", "canonical.parquet", True],
                ["Example", 2, "RL", "other", "other.parquet", False],
                ["Example", 2, "periodic", "periodic_15min", "periodic.parquet", False],
            ]
            pd.DataFrame(rows, columns=[
                "unit", "matched_steps", "family", "policy", "decisions_path",
                "family_eligible",
            ]).to_csv(root / "inventory.csv", index=False)
            manifest = {
                "code_root": ".", "candidate_inventory_csv": "inventory.csv",
                "periodic_intervals": [15],
                "family_exclude_patterns": ["canonical"],
                "units": [{
                    "name": "Example", "unit_path": "example/unit/path",
                    "expected_steps": 2, "expected_margin_pct": 0,
                    "fixed_pair": {"rl": "other", "periodic": "periodic_15min"},
                }],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            _, units = load_manifest(root / "manifest.json")

            self.assertEqual([item.policy for item in units[0].family_rl], ["canonical"])

    def test_manifest_rejects_changed_frozen_inventory(self):
        """A declared digest turns unnoticed inventory drift into a hard failure."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inventory.csv").write_text("unit\nExample\n", encoding="utf-8")
            manifest = {
                "code_root": ".", "candidate_inventory_csv": "inventory.csv",
                "candidate_inventory_sha256": "0" * 64, "units": [],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_manifest(root / "manifest.json")

    def test_artifact_manifest_links_tables_and_figures(self):
        """Regeneration records code hashes, commands, and immediate CSV sources."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = root / "protocol.json"
            protocol_path.write_text("{}", encoding="utf-8")
            results = {
                "candidate_metrics": pd.DataFrame({"value": [1]}),
                "two_objective_candidates": pd.DataFrame({"value": [2]}),
            }
            write_outputs(results, root)

            path = write_artifact_manifest(
                results, root, {"version": 2}, protocol_path,
                ["python", "analysis.py"],
            )
            manifest = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["protocol_version"], 2)
            self.assertEqual(manifest["regeneration_command"], "python analysis.py")
            tables = {item["path"]: item for item in manifest["tables"]}
            self.assertIsNone(tables["candidate_metrics.csv"]["source_csv"])
            self.assertEqual(
                tables["two_objective_candidates.csv"]["source_csv"],
                "candidate_metrics.csv",
            )
            self.assertIn("rl_periodic_reward_analysis.py", manifest["analysis_code_sha256"])

    def test_run_analysis_emits_all_evidence_tables(self):
        """The pipeline should produce canonical, break-even, and family rows."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rl_path = root / "rl.parquet"
            periodic_path = root / "periodic.parquet"
            rl_frame = decision_frame([0, 1, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0])
            periodic_frame = decision_frame([0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0])
            rl_frame.to_parquet(rl_path, index=False)
            periodic_frame.to_parquet(periodic_path, index=False)
            rl_ref = CandidateRef("rl", "RL", rl_path)
            periodic_ref = CandidateRef("periodic", "periodic", periodic_path)
            rl_metrics = summarize_decisions(rl_frame, thresholds=[0])
            periodic_metrics = summarize_decisions(periodic_frame, thresholds=[0])
            expected_margin = 100 * (
                rl_metrics.standard_reward_mean() - periodic_metrics.standard_reward_mean()
            ) / abs(periodic_metrics.standard_reward_mean())
            unit = UnitSpec(
                "Example", 4, expected_margin, rl_ref, periodic_ref,
                (rl_ref,), (periodic_ref,),
            )

            results = run_analysis([unit], thresholds=[2], costs=[0.5], include_family=True)

            self.assertEqual(len(results["canonical_pairs"]), 1)
            self.assertEqual(len(results["break_even"]), 1)
            self.assertEqual(len(results["family_envelope"]), 1)
            self.assertEqual(len(results["p95_constrained_best"]), 4)
            self.assertEqual(len(results["pareto_frontiers"]), 8)
            self.assertEqual(len(results["weight_sweep_winners"]), 924)
            self.assertEqual(len(results["two_objective_weight_sweep_winners"]), 808)
            self.assertEqual(len(results["two_objective_weight_plane_winners"]), 3520)
            self.assertIn("two_objective_winner_ranges", results)
            self.assertIn("two_objective_normalization_bounds", results)
            bounded = results["p95_constrained_best"].query("not is_unbounded")
            self.assertEqual(set(bounded["policy"]), {"rl", "periodic"})
            self.assertEqual(results["family_envelope"].iloc[0]["winner"], "periodic")

            fixed_only = run_analysis(
                [unit], thresholds=[2], costs=[0.5], include_family=False,
            )
            self.assertNotIn("pareto_frontiers", fixed_only)
            self.assertNotIn("two_objective_pareto_frontiers", fixed_only)


if __name__ == "__main__":
    unittest.main()
