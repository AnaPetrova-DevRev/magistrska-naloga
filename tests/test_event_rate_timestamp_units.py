"""Event aggregation must preserve timestamps at each pandas time resolution."""

from datetime import datetime
import unittest
from unittest.mock import MagicMock

import pandas as pd

from modeling.dataset_builder import _load_events_per_minute_dataframe
from sync_unit import SyncUnit


class TestEventRateTimestampUnits(unittest.TestCase):
    def test_aggregation_preserves_dates_and_merges_duplicate_minutes(self):
        timestamps = pd.to_datetime([
            datetime(2025, 1, 1, 12, 1),
            datetime(2025, 1, 1, 12, 0),
            datetime(2025, 1, 1, 12, 1),
        ])
        for unit in ("ms", "us", "ns"):
            with self.subTest(unit=unit):
                storage = MagicMock()
                storage.list_files_in_folder.return_value = ["synthetic.pkl"]
                storage.read_picklefile_as_dataframe.return_value = pd.DataFrame({
                    "minute": timestamps.as_unit(unit),
                    "events_per_minute": [2.0, 1.0, 3.0],
                })
                result = _load_events_per_minute_dataframe(
                    storage, SyncUnit("test_org", "test_system", "test_project"),
                )
                self.assertEqual(result["minute"].tolist(), sorted(set(timestamps)))
                self.assertEqual(result["events_per_minute"].tolist(), [1.0, 5.0])
