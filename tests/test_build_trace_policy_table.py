"""Tests for the matched-window trace/property table generator."""

# pylint: disable=missing-function-docstring,protected-access

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis import build_trace_policy_table as table


class TestBuildTracePolicyTable(unittest.TestCase):
    """Verify naming, formatting, and policy-window alignment."""

    def test_unit_name_handles_standard_and_jira_units(self):
        standard = table.CODE_ROOT / "data/brightpath/salesforce/project/file.json"
        jira = table.CODE_ROOT / "data/novacore/jira/incidents/file.json"

        self.assertEqual(table._unit_name(standard), "Brightpath Salesforce")
        self.assertEqual(table._unit_name(jira), "Novacore Jira Incidents")

    def test_num_uses_protected_decimal_comma(self):
        self.assertEqual(table._num(1.32), "1{,}3")
        self.assertEqual(table._num(5.9, signed=True), "+5{,}9")

    def test_validate_window_accepts_terminal_null_timestamp(self):
        manifest = {
            "semantics_version": "observable-stream-window-v1",
            "effective_start": "2025-01-01T00:00:00Z",
            "effective_end": "2025-01-01T00:19:00Z",
        }
        timestamps = pd.Series([
            pd.Timestamp("2025-01-01T00:18:00Z"),
            pd.Timestamp("2025-01-01T00:19:00Z"),
            pd.NaT,
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.parquet"
            pd.DataFrame({"timestamp": timestamps}).to_parquet(path)
            start, end = table._validate_window("Unit", manifest, 3, path)

        self.assertEqual(start, "2025-01-01T00:18:00+00:00")
        self.assertEqual(end, "2025-01-01T00:19:00+00:00")

    def test_validate_window_rejects_missing_terminal_null(self):
        manifest = {
            "semantics_version": "observable-stream-window-v1",
            "effective_start": "2025-01-01T00:00:00Z",
            "effective_end": "2025-01-01T00:19:00Z",
        }
        timestamps = pd.date_range("2025-01-01T00:17:00Z", periods=3, freq="min")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.parquet"
            pd.DataFrame({"timestamp": timestamps}).to_parquet(path)
            with self.assertRaisesRegex(ValueError, "dated/terminal decision count"):
                table._validate_window("Unit", manifest, 3, path)


if __name__ == "__main__":
    unittest.main()
