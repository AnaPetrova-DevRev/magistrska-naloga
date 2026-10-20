"""Analysis commands require explicit external inputs before reading artifacts."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def dependencies_available(*names):
    return all(importlib.util.find_spec(name) is not None for name in names)


class TestAnalysisInputArguments(unittest.TestCase):
    def assert_required_input(self, module, argument, *arguments):
        result = subprocess.run(
            [sys.executable, "-m", module, *arguments],
            cwd=ROOT, capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn(argument, result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_practical_value_requires_input_directory(self):
        self.assert_required_input("analysis.thesis_practical_value", "--input-dir")

    @unittest.skipUnless(dependencies_available("numpy", "pandas", "scipy"),
                         "Requires NumPy, pandas and SciPy")
    def test_connection_analysis_requires_cutoff_configuration(self):
        self.assert_required_input(
            "analysis.observation_cutoff_sensitivity", "--cutoffs",
            "--connection-json", "unused-output.json",
        )

    @unittest.skipUnless(dependencies_available("numpy", "pandas", "scipy"),
                         "Requires NumPy, pandas and SciPy")
    def test_connection_analysis_reads_supplied_external_inputs(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            data_root = directory / "external-data"
            unit_root = data_root / "synthetic" / "system" / "project"
            (unit_root / "data_preprocessing").mkdir(parents=True)
            (unit_root / "clean_data").mkdir()
            times = pd.date_range("2025-01-01", periods=20, freq="min", tz="UTC")
            pd.DataFrame({
                "minute": times[[0, 5, 19]], "events_per_minute": [1, 2, 1],
            }).to_pickle(unit_root / "data_preprocessing/merged_events_per_minute_1.pkl")
            pd.DataFrame({
                "id": ["synthetic-record", "synthetic-record"],
                "created_date": [times[0], times[0]],
                "modified_date": times[[5, 19]],
            }).to_pickle(unit_root / "clean_data/cleaned_items_1.pkl")
            registry = directory / "registry.json"
            registry.write_text(json.dumps([{
                "organization": "synthetic", "external_system": "system", "project": "project",
            }]))
            cutoffs = directory / "cutoffs.json"
            cutoffs.write_text(json.dumps({
                "semantics_version": "synthetic-fixture-v1",
                "units": {"synthetic_system_project": {
                    "display_name": "Synthetic unit", "cutoff_status": "synthetic",
                    "sync_mode_before_oct_2024": "synthetic",
                    "interval_minutes_before_oct_2024": 15,
                    "split": {
                        "data_start": times[0].isoformat(), "train_end": times[9].isoformat(),
                        "val_end": times[14].isoformat(), "data_end": times[19].isoformat(),
                    },
                    "scenarios": [{"id": "fixture", "start": times[5].isoformat()}],
                }},
            }))
            output = directory / "result.json"
            result = subprocess.run([
                sys.executable, "-m", "analysis.observation_cutoff_sensitivity",
                "--registry", str(registry), "--data-root", str(data_root),
                "--cutoffs", str(cutoffs), "--connection-json", str(output),
            ], cwd=ROOT, capture_output=True, text=True, timeout=30, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            unit = json.loads(output.read_text())["units"]["synthetic_system_project"]
            self.assertEqual(unit["full_stream"]["observed_events"], 4)
            self.assertEqual(unit["scenarios"][0]["stream"]["observed_events"], 3)
            self.assertEqual(unit["scenarios"][0]["lifecycle"]["observed_states"], 2)

    @unittest.skipUnless(dependencies_available("numpy", "pandas", "pyarrow"),
                         "Requires NumPy, pandas and PyArrow")
    def test_clipping_audit_requires_inventory(self):
        self.assert_required_input("analysis.audit_observation_clipping", "--inventory")

    @unittest.skipUnless(dependencies_available("numpy", "pandas", "scipy", "sklearn", "gymnasium"),
                         "Requires the forecasting and RL runtime dependencies")
    def test_reward_analysis_requires_manifest(self):
        self.assert_required_input("analysis.rl_periodic_reward_analysis", "--manifest")

    @unittest.skipUnless(dependencies_available("numpy", "pandas", "scipy", "sklearn", "gymnasium"),
                         "Requires the forecasting and RL runtime dependencies")
    def test_periodic_generation_requires_input_and_output_roots(self):
        self.assert_required_input(
            "analysis.generate_matched_periodic_decisions", "--data-root",
        )
        self.assert_required_input(
            "analysis.generate_matched_periodic_decisions", "--output-dir",
            "--data-root", "unused-data",
        )

    @unittest.skipUnless(dependencies_available("numpy", "pandas", "scipy", "sklearn", "gymnasium"),
                         "Requires the forecasting and RL runtime dependencies")
    def test_trace_table_requires_manifest(self):
        self.assert_required_input("analysis.build_trace_policy_table", "--manifest")

    @unittest.skipUnless(dependencies_available("numpy", "pandas", "scipy", "sklearn"),
                         "Requires the dataset comparison runtime dependencies")
    def test_weighted_comparison_requires_fingerprints(self):
        self.assert_required_input("dataset_comparison.weighted_comparison", "--fingerprints")


if __name__ == "__main__":
    unittest.main()
