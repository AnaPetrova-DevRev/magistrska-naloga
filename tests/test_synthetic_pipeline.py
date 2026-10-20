"""Exercise the public fixture workflow in isolated temporary workspaces."""

from contextlib import chdir
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from modeling.core_types import DatasetConfig, TaskType
from modeling.dataset_builder import build_event_rate_dataset, build_time_to_event_dataset
from scripts.create_synthetic_fixture import create_fixture, run_pipeline
from sync_unit import SyncUnit


UNIT = SyncUnit("test_org", "test_system", "test_project")


class TestSyntheticFixture(unittest.TestCase):
    def test_fixture_creates_two_parseable_chunks_with_only_invented_ids(self):
        from data_cleanup.parse_json import parse_json_file

        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            create_fixture(Path(directory))
            registry = json.loads(Path("data/sync_units_register.json").read_text())
            self.assertEqual(registry, [UNIT.to_dict()])
            records = []
            for chunk in (1, 2):
                path = f"{UNIT.get_path_for_sync_unit()}/initial_data/work_{chunk}.json"
                parsed, logs = parse_json_file(path)
                self.assertEqual(logs, [])
                self.assertEqual(len(parsed), 15)
                records.extend(parsed)
            self.assertEqual(len({record.id for record in records}), 24)
            self.assertTrue(all(record.id.startswith("synthetic-") for record in records))

    def test_existing_data_is_rejected_without_changing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            sentinel = data / "sync_units_register.json"
            sentinel.write_text('[{"organization": "existing"}]')
            before = sentinel.read_bytes()
            with self.assertRaises(FileExistsError):
                create_fixture(root)
            self.assertEqual(sentinel.read_bytes(), before)
            self.assertEqual(list(data.iterdir()), [sentinel])

    def test_invalid_fixture_is_rejected_before_creating_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root / "invalid.json"
            invalid.write_text('{"schema_version": 1, "chunks": []}')
            with self.assertRaises(ValueError):
                create_fixture(root, fixture_path=invalid)
            self.assertFalse((root / "data").exists())


class TestSyntheticPipeline(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.cwd = chdir(self.workspace)
        self.cwd.__enter__()
        self.addCleanup(self.cwd.__exit__, None, None, None)
        create_fixture(self.workspace)
        self.responses = run_pipeline(self.workspace)
        self.start = datetime(2024, 1, 1)
        self.end = datetime(2024, 1, 1, 1, 56)

    def event_rate_dataset(self):
        return build_event_rate_dataset(UNIT, DatasetConfig(
            task_type=TaskType.EVENT_RATE,
            start_time=self.start,
            end_time=self.end,
            resample_freq="1min",
        ))[0]

    def test_real_pipeline_deduplicates_snapshots_and_builds_minute_counts(self):
        self.assertEqual(len(self.responses), 6)
        self.assertTrue(all(response.success for response in self.responses))
        root = Path("data") / UNIT.get_path_for_sync_unit()
        for chunk in (1, 2):
            cleaned = pd.read_pickle(root / "clean_data" / f"cleaned_items_{chunk}.pkl")
            self.assertEqual(len(cleaned), 14)
            self.assertEqual(cleaned["id"].nunique(), 12)
        dataset = self.event_rate_dataset()
        self.assertEqual(len(dataset.timestamps), 117)
        self.assertEqual(sum(dataset.events_per_minute), 52)
        self.assertEqual(list(dataset.events_per_minute).count(0), 65)
        self.assertEqual(dataset.events_per_minute[60], 1)
        self.assertEqual(len(dataset.train_indices), 81)
        self.assertEqual(len(dataset.val_indices), 17)
        self.assertEqual(len(dataset.test_indices), 19)

    def test_tte_keeps_the_interval_across_input_chunks(self):
        dataset, _ = build_time_to_event_dataset(UNIT, DatasetConfig(
            task_type=TaskType.TIME_TO_EVENT,
            start_time=self.start,
            end_time=self.end,
        ))
        self.assertEqual(len(dataset.timestamps), 51)
        self.assertTrue(all(value > 0 for value in dataset.minutes_to_next_event))
        boundary = list(dataset.timestamps).index(datetime(2024, 1, 1, 1))
        self.assertEqual(dataset.minutes_to_next_event[boundary], 4)
        self.assertEqual(sum(dataset.minutes_to_next_event), 116)

    @unittest.skipUnless(importlib.util.find_spec("statsmodels"), "statsmodels is optional")
    def test_small_arima_fit_predict_and_save_load(self):
        from modeling.models.arima_model import ARIMAEventRateModel

        dataset = self.event_rate_dataset()
        model = ARIMAEventRateModel(
            TaskType.EVENT_RATE, {"order": [1, 0, 0]}, {"max_iter": 30},
        )
        result = model.fit(dataset)
        self.assertEqual(result["n_train"], 81)
        predictions = model.predict_pointwise(dataset, dataset.test_indices)
        self.assertEqual(predictions.shape, (19,))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue((predictions >= 0).all())
        model.save("tiny_model")
        restored = ARIMAEventRateModel.load(
            "tiny_model", TaskType.EVENT_RATE, {"order": [1, 0, 0]},
        )
        np.testing.assert_allclose(
            restored.predict_pointwise(dataset, dataset.test_indices), predictions,
        )


if __name__ == "__main__":
    unittest.main()
