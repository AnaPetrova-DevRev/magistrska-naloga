"""Tests for isolated windowed-analysis artifact writing."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.run_windowed_dataset_analytics import _write_unit_window
from dataset_comparison.windowed import AnalysisWindow
from sync_unit import SyncUnit


class TestWindowedArtifactWriter(unittest.TestCase):
    """Ensure derived files cannot replace canonical pipeline artifacts."""

    def test_outputs_are_nested_under_windowed_analysis(self):
        """All unit artifacts use a separate window-specific directory."""
        unit = SyncUnit("org", "system", "project")
        temporal = {
            "effective_start": "2022-01-01T00:00:00+00:00",
            "effective_end": "2022-01-02T00:00:00+00:00",
        }
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "analysis.run_windowed_dataset_analytics.PROJECT_ROOT",
                Path(temporary),
            ):
                _write_unit_window(
                    unit,
                    AnalysisWindow("post_2022", "2022-01-01T00:00:00Z"),
                    {"features": {}},
                    {"records_with_state": 1},
                    temporal,
                )

            output = (
                Path(temporary)
                / "data/org/system/project/windowed_analysis/post_2022"
            )
            self.assertEqual(len(list(output.glob("*.json"))), 4)
            canonical = (
                Path(temporary)
                / "data/org/system/project/pipeline_explorer/analysis_per_record.pkl"
            )
            self.assertFalse(canonical.exists())


if __name__ == "__main__":
    unittest.main()
