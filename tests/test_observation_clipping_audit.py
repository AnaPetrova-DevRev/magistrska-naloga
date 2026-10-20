"""Tests for the frozen-candidate observation clipping audit."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.audit_observation_clipping import audit_observation_clipping


class TestObservationClippingAudit(unittest.TestCase):
    """Validate feature-aware clipping counts on tiny synthetic traces."""

    def test_only_enabled_features_and_eligible_candidates_are_counted(self):
        """Ignore excluded candidates and trace fields absent from observations."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            records = []
            for name, eligible, features in (
                ("used", True, ["current_aoi"]),
                ("excluded", False, ["time_since_last_sync"]),
            ):
                agent = root / name / "agent"
                policy = root / name / "policy"
                agent.mkdir(parents=True)
                policy.mkdir(parents=True)
                (agent / "training_metadata.json").write_text(
                    json.dumps(
                        {"env_config": {"observation_features": features}}
                    ),
                    encoding="utf-8",
                )
                pd.DataFrame(
                    {
                        "current_aoi": [0.0, 1440.0, 1500.0],
                        "time_since_last_sync": [2000.0, 2000.0, 2000.0],
                    }
                ).to_parquet(policy / "decisions.parquet", index=False)
                records.append(
                    {
                        "unit": "Test Unit",
                        "family": "RL",
                        "family_eligible": eligible,
                        "decisions_path": str(policy / "decisions.parquet"),
                        "source_policy_path": str(agent),
                    }
                )

            inventory = root / "inventory.csv"
            pd.DataFrame(records).to_csv(inventory, index=False)
            result = audit_observation_clipping(inventory, root)
            global_rows = result[result["unit"] == "ALL"]

            self.assertEqual(global_rows["feature"].tolist(), ["current_aoi"])
            row = global_rows.iloc[0]
            self.assertEqual(row["candidate_count"], 1)
            self.assertEqual(row["minute_count"], 3)
            self.assertEqual(row["values_above_limit"], 1)
            self.assertAlmostEqual(row["share_above_limit"], 1 / 3)
            self.assertEqual(row["maximum"], 1500.0)


if __name__ == "__main__":
    unittest.main()
