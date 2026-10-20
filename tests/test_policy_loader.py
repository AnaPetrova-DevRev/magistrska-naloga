"""Tests for explicit final/best trained-policy checkpoint selection."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from sync_policy_evaluation.policy_loader import (
    load_trained_policy,
    resolve_trained_policy_file,
)


class TestPolicyCheckpointSelection(unittest.TestCase):
    """Verify checkpoint resolution and SB3 loader routing."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        with open(
            os.path.join(self.root, "training_metadata.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump({"agent_config": {"algorithm": "PPO"}}, f)
        Path(self.root, "agent.zip").touch()
        os.makedirs(os.path.join(self.root, "best_model"))
        Path(self.root, "best_model", "best_model.zip").touch()

    def tearDown(self):
        self.tmp.cleanup()

    def test_final_checkpoint_is_default(self):
        """Legacy calls continue to select the final agent archive."""
        resolved = resolve_trained_policy_file(self.root)
        self.assertEqual(resolved, os.path.join(self.root, "agent.zip"))

    def test_best_checkpoint_is_selected_explicitly(self):
        """The best selector resolves EvalCallback's saved archive."""
        resolved = resolve_trained_policy_file(self.root, "best")
        self.assertEqual(
            resolved,
            os.path.join(self.root, "best_model", "best_model.zip"),
        )

    def test_loader_passes_selected_archive_to_sb3(self):
        """SB3 receives the resolved best archive path."""
        algo = MagicMock()
        algo.load.return_value = MagicMock()
        with patch(
            "sync_policy_evaluation.policy_loader._get_sb3_class",
            return_value=algo,
        ):
            load_trained_policy(self.root, checkpoint="best")
        algo.load.assert_called_once_with(
            os.path.join(self.root, "best_model", "best_model.zip")
        )

    def test_missing_best_checkpoint_fails_closed(self):
        """A missing requested best checkpoint cannot fall back silently."""
        os.remove(os.path.join(self.root, "best_model", "best_model.zip"))
        with self.assertRaisesRegex(FileNotFoundError, "checkpoint not found"):
            resolve_trained_policy_file(self.root, "best")

    def test_invalid_checkpoint_is_rejected(self):
        """Only the two documented checkpoint modes are accepted."""
        with self.assertRaisesRegex(ValueError, "final.*best"):
            resolve_trained_policy_file(self.root, "latest")


if __name__ == "__main__":
    unittest.main()
