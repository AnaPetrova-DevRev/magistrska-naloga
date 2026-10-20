"""Tests for deterministic Pareto candidate-inventory freezing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.freeze_pareto_candidate_inventory import freeze_inventory


class TestFreezeCandidateInventory(unittest.TestCase):
    """Verify metadata grouping and representative selection."""

    @staticmethod
    def _write_candidate(
        root: Path, policy: str, seed, trained_at: str = "2025-01-01T00:00:00Z",
    ) -> str:
        evaluation = root / "evaluations" / policy
        agent = root / "agents" / policy
        evaluation.mkdir(parents=True)
        agent.mkdir(parents=True)
        (evaluation / "policy_metadata.json").write_text(json.dumps({
            "policy_path": str(agent.relative_to(root)),
        }), encoding="utf-8")
        (agent / "training_metadata.json").write_text(json.dumps({
            "task_type": "event_rate",
            "total_timesteps": 100,
            "agent_name": policy,
            "agent_config": {"algorithm": "ppo", "seed": seed},
            "env_config": {"features": ["current_aoi"]},
            "reward_config": {"aoi_mode": "linear"},
            "prediction_config": {"mode": "none"},
            "data_split": {"test_start": "2025-01-01"},
            "trained_at": trained_at,
            "mlflow_run_id": "ignored",
        }), encoding="utf-8")
        return str((evaluation / "decisions.parquet").relative_to(root))

    def test_unsuffixed_run_represents_seed_repetitions(self):
        """The canonical run is eligible and matching seeds are robustness-only."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for policy, seed in (
                ("ppo_base", None),
                ("ppo_base_seed7", 7),
                ("ppo_base_pilot10k", 42),
            ):
                rows.append([
                    "Unit", 10, "RL", policy,
                    self._write_candidate(root, policy, seed),
                ])
            rows.append(["Unit", 10, "periodic", "periodic_15min", "periodic"])
            source = root / "source.csv"
            output = root / "frozen.csv"
            pd.DataFrame(rows, columns=[
                "unit", "matched_steps", "family", "policy", "decisions_path",
            ]).to_csv(source, index=False)

            result = freeze_inventory(source, output, root).set_index("policy")

            self.assertTrue(result.loc["ppo_base", "family_eligible"])
            self.assertFalse(result.loc["ppo_base_seed7", "family_eligible"])
            self.assertEqual(
                result.loc["ppo_base", "configuration_digest"],
                result.loc["ppo_base_seed7", "configuration_digest"],
            )
            self.assertEqual(
                result.loc["ppo_base_seed7", "selection_reason"],
                "seed_robustness_only",
            )
            self.assertEqual(
                result.loc["ppo_base_pilot10k", "selection_reason"],
                "protocol_excluded",
            )
            self.assertTrue(output.exists())

    def test_earliest_unsuffixed_run_breaks_repetition_tie(self):
        """Repeated unseeded training uses provenance time, never performance."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for policy, trained_at in (
                ("later", "2025-02-01T00:00:00Z"),
                ("earlier", "2025-01-01T00:00:00Z"),
            ):
                rows.append([
                    "Unit", 10, "RL", policy,
                    self._write_candidate(root, policy, None, trained_at),
                ])
            source = root / "source.csv"
            pd.DataFrame(rows, columns=[
                "unit", "matched_steps", "family", "policy", "decisions_path",
            ]).to_csv(source, index=False)

            result = freeze_inventory(source, root / "frozen.csv", root).set_index(
                "policy",
            )

            self.assertTrue(result.loc["earlier", "family_eligible"])
            self.assertEqual(result.loc["earlier", "selection_reason"], "earliest_unsuffixed")
            self.assertEqual(
                result.loc["later", "selection_reason"],
                "unseeded_robustness_only",
            )


if __name__ == "__main__":
    unittest.main()
