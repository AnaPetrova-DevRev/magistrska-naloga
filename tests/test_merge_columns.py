"""
Tests for merge_columns functionality
"""
import unittest
from datetime import datetime

import pandas as pd

from data_preprocessing.merge_columns import calculate_merge, calculate_events_per_minute


class TestMergeColumns(unittest.TestCase):
    """Test cases for the merge_columns module"""

    def test_calculate_merge_different_dates(self):
        """Test merging when created_date and modified_date are different"""
        # Create test dataframe - each row has different created and modified dates
        df = pd.DataFrame({
            'id': ['1', '2', '3'],
            'created_date': [
                datetime(2024, 1, 1, 10, 0, 0),
                datetime(2024, 1, 2, 10, 0, 0),
                datetime(2024, 1, 3, 10, 0, 0)
            ],
            'modified_date': [
                datetime(2024, 1, 1, 15, 0, 0),
                datetime(2024, 1, 2, 16, 0, 0),
                datetime(2024, 1, 3, 17, 0, 0)
            ]
        })

        # Apply merge
        result = calculate_merge(df)

        # Check that created_date and modified_date columns are dropped
        self.assertNotIn('created_date', result.columns)
        self.assertNotIn('modified_date', result.columns)

        # Check that events column exists
        self.assertIn('events', result.columns)

        # Each row with different dates becomes 2 rows (one per event)
        # Total: 3 input rows with different dates = 6 output rows
        self.assertEqual(len(result), 6)

        # Results are sorted by events datetime
        # Check that all events are present and sorted
        events = result['events'].tolist()
        self.assertEqual(events[0], datetime(2024, 1, 1, 10, 0, 0))  # id=1 created
        self.assertEqual(events[1], datetime(2024, 1, 1, 15, 0, 0))  # id=1 modified
        self.assertEqual(events[2], datetime(2024, 1, 2, 10, 0, 0))  # id=2 created
        self.assertEqual(events[3], datetime(2024, 1, 2, 16, 0, 0))  # id=2 modified
        self.assertEqual(events[4], datetime(2024, 1, 3, 10, 0, 0))  # id=3 created
        self.assertEqual(events[5], datetime(2024, 1, 3, 17, 0, 0))  # id=3 modified

    def test_calculate_merge_same_dates(self):
        """Test merging when created_date and modified_date are the same"""
        # Create test dataframe
        df = pd.DataFrame({
            'id': ['1', '2'],
            'created_date': [
                datetime(2024, 1, 1, 10, 0, 0),
                datetime(2024, 1, 2, 10, 0, 0)
            ],
            'modified_date': [
                datetime(2024, 1, 1, 10, 0, 0),  # Same as created_date
                datetime(2024, 1, 2, 10, 0, 0)   # Same as created_date
            ]
        })

        # Apply merge
        result = calculate_merge(df)

        # When dates are the same, only one row per input row
        self.assertEqual(len(result), 2)

        # Check events are correct and sorted
        events = result['events'].tolist()
        self.assertEqual(events[0], datetime(2024, 1, 1, 10, 0, 0))
        self.assertEqual(events[1], datetime(2024, 1, 2, 10, 0, 0))

        # Check IDs
        ids = result['id'].tolist()
        self.assertEqual(ids[0], '1')
        self.assertEqual(ids[1], '2')

    def test_calculate_merge_mixed_dates(self):
        """Test merging with a mix of same and different dates"""
        # Create test dataframe
        df = pd.DataFrame({
            'id': ['1', '2', '3'],
            'created_date': [
                datetime(2024, 1, 1, 10, 0, 0),
                datetime(2024, 1, 2, 10, 0, 0),
                datetime(2024, 1, 3, 10, 0, 0)
            ],
            'modified_date': [
                datetime(2024, 1, 1, 10, 0, 0),  # Same
                datetime(2024, 1, 2, 15, 0, 0),  # Different
                datetime(2024, 1, 3, 10, 0, 0)   # Same
            ]
        })

        # Apply merge
        result = calculate_merge(df)

        # id=1: same dates = 1 row
        # id=2: different dates = 2 rows
        # id=3: same dates = 1 row
        # Total: 4 rows
        self.assertEqual(len(result), 4)

        # Results sorted by events datetime
        events = result['events'].tolist()
        ids = result['id'].tolist()

        self.assertEqual(events[0], datetime(2024, 1, 1, 10, 0, 0))
        self.assertEqual(ids[0], '1')

        self.assertEqual(events[1], datetime(2024, 1, 2, 10, 0, 0))
        self.assertEqual(ids[1], '2')

        self.assertEqual(events[2], datetime(2024, 1, 2, 15, 0, 0))
        self.assertEqual(ids[2], '2')

        self.assertEqual(events[3], datetime(2024, 1, 3, 10, 0, 0))
        self.assertEqual(ids[3], '3')

    def test_calculate_merge_only_id_and_events_columns(self):
        """Test that result only has id and events columns"""
        # Create test dataframe with additional columns
        df = pd.DataFrame({
            'id': ['1', '2'],
            'name': ['Alice', 'Bob'],
            'created_date': [
                datetime(2024, 1, 1, 10, 0, 0),
                datetime(2024, 1, 2, 10, 0, 0)
            ],
            'modified_date': [
                datetime(2024, 1, 1, 15, 0, 0),
                datetime(2024, 1, 2, 16, 0, 0)
            ]
        })

        # Apply merge
        result = calculate_merge(df)

        # Result should only have id and events columns
        self.assertEqual(list(result.columns), ['id', 'events'])

        # Each row with different dates = 2 output rows
        self.assertEqual(len(result), 4)

    def test_calculate_merge_sorting(self):
        """Test that events are sorted in ascending order by datetime"""
        # Create test dataframe with dates in non-chronological order
        df = pd.DataFrame({
            'id': ['a', 'b'],
            'created_date': [
                datetime(2024, 1, 5, 10, 0, 0),  # Later date
                datetime(2024, 1, 1, 10, 0, 0)   # Earlier date
            ],
            'modified_date': [
                datetime(2024, 1, 5, 15, 0, 0),
                datetime(2024, 1, 1, 15, 0, 0)
            ]
        })

        # Apply merge
        result = calculate_merge(df)

        # Should be sorted by events datetime (ascending)
        events = result['events'].tolist()
        self.assertEqual(events[0], datetime(2024, 1, 1, 10, 0, 0))  # b created
        self.assertEqual(events[1], datetime(2024, 1, 1, 15, 0, 0))  # b modified
        self.assertEqual(events[2], datetime(2024, 1, 5, 10, 0, 0))  # a created
        self.assertEqual(events[3], datetime(2024, 1, 5, 15, 0, 0))  # a modified

    def test_calculate_merge_repeated_id_emits_creation_once(self):
        """Test that a record spanning several modification snapshots yields one creation event

        Cleaned rows are modification snapshots, not records: the source API returns one
        {id, created_date, modified_date} object per change, so created_date is a repeated
        attribute of the record rather than a per-row occurrence.
        """
        # One record, three modification snapshots - the shape of the real data
        df = pd.DataFrame({
            'id': ['1', '1', '1'],
            'created_date': [
                datetime(2024, 1, 1, 10, 0, 0),
                datetime(2024, 1, 1, 10, 0, 0),
                datetime(2024, 1, 1, 10, 0, 0)
            ],
            'modified_date': [
                datetime(2024, 1, 1, 11, 0, 0),
                datetime(2024, 1, 1, 12, 0, 0),
                datetime(2024, 1, 1, 13, 0, 0)
            ]
        })

        # Apply merge
        result = calculate_merge(df)

        # 1 creation + 3 modifications = 4 events, not 2 per input row
        self.assertEqual(len(result), 4)

        events = result['events'].tolist()
        self.assertEqual(events, [
            datetime(2024, 1, 1, 10, 0, 0),
            datetime(2024, 1, 1, 11, 0, 0),
            datetime(2024, 1, 1, 12, 0, 0),
            datetime(2024, 1, 1, 13, 0, 0)
        ])

    def test_calculate_merge_no_duplicate_id_timestamp_pairs(self):
        """Test that no (id, events) pair is emitted more than once"""
        # Two records, each with several snapshots, plus a redundant source object
        # for id=2 (pagination overlap repeats the same (id, modified_date))
        df = pd.DataFrame({
            'id': ['1', '1', '2', '2', '2'],
            'created_date': [
                datetime(2024, 1, 1, 10, 0, 0),
                datetime(2024, 1, 1, 10, 0, 0),
                datetime(2024, 1, 2, 10, 0, 0),
                datetime(2024, 1, 2, 10, 0, 0),
                datetime(2024, 1, 2, 10, 0, 0)
            ],
            'modified_date': [
                datetime(2024, 1, 1, 11, 0, 0),
                datetime(2024, 1, 1, 12, 0, 0),
                datetime(2024, 1, 2, 11, 0, 0),
                datetime(2024, 1, 2, 11, 0, 0),  # Repeat of the previous snapshot
                datetime(2024, 1, 2, 12, 0, 0)
            ]
        })

        # Apply merge
        result = calculate_merge(df)

        # No (id, events) pair may appear twice
        self.assertEqual(len(result), len(result.drop_duplicates(subset=['id', 'events'])))

        # id=1: 1 creation + 2 modifications = 3
        # id=2: 1 creation + 2 distinct modifications = 3
        self.assertEqual(len(result), 6)

    def test_calculate_merge_repeated_id_created_equals_modified(self):
        """Test that a snapshot taken at the creation instant is not double-counted"""
        # id=1 was modified in the same instant it was created, then modified again
        df = pd.DataFrame({
            'id': ['1', '1'],
            'created_date': [
                datetime(2024, 1, 1, 10, 0, 0),
                datetime(2024, 1, 1, 10, 0, 0)
            ],
            'modified_date': [
                datetime(2024, 1, 1, 10, 0, 0),  # Same as created_date
                datetime(2024, 1, 1, 11, 0, 0)   # Different
            ]
        })

        # Apply merge
        result = calculate_merge(df)

        # The creation instant counts once, plus the later modification
        self.assertEqual(len(result), 2)

        events = result['events'].tolist()
        self.assertEqual(events[0], datetime(2024, 1, 1, 10, 0, 0))
        self.assertEqual(events[1], datetime(2024, 1, 1, 11, 0, 0))

    def test_calculate_events_per_minute_repeated_id(self):
        """Test that per-minute counts are not inflated by a record's creation instant"""
        # The merged stream for one record with three snapshots, all within the same hour
        df = pd.DataFrame({
            'id': ['1', '1', '1'],
            'created_date': [
                datetime(2024, 1, 1, 10, 0, 0),
                datetime(2024, 1, 1, 10, 0, 0),
                datetime(2024, 1, 1, 10, 0, 0)
            ],
            'modified_date': [
                datetime(2024, 1, 1, 10, 5, 0),
                datetime(2024, 1, 1, 10, 6, 0),
                datetime(2024, 1, 1, 10, 7, 0)
            ]
        })

        # Apply merge, then calculate events per minute
        result = calculate_events_per_minute(calculate_merge(df))

        # The creation minute holds one event, not one per snapshot
        creation_minute = result[result['minute'] == datetime(2024, 1, 1, 10, 0, 0)]
        self.assertEqual(creation_minute.iloc[0]['events_per_minute'], 1)

        # Four active minutes, one event each
        self.assertEqual(len(result), 4)
        self.assertEqual(result['events_per_minute'].sum(), 4)

    def test_calculate_events_per_minute(self):
        """Test that events per minute are calculated correctly"""
        # Create test dataframe with events at different minutes
        df = pd.DataFrame({
            'events': [
                datetime(2024, 1, 1, 10, 0, 0),
                datetime(2024, 1, 1, 10, 1, 1),
                datetime(2024, 1, 1, 10, 1, 2),
                datetime(2024, 1, 1, 10, 2, 0),
                datetime(2024, 1, 1, 11, 0, 0),
                datetime(2024, 1, 1, 11, 1, 1)
            ]
        })

        # Apply calculation        result = calculate_events_per_minute(df)
        result = calculate_events_per_minute(df)

        # Check that result has correct columns
        self.assertEqual(list(result.columns), ['minute', 'events_per_minute'])

        # Check that result has correct rows and values
        self.assertEqual(len(result), 5)
        self.assertEqual(result.iloc[0]['minute'], datetime(2024, 1, 1, 10, 0, 0))
        self.assertEqual(result.iloc[0]['events_per_minute'], 1)
        self.assertEqual(result.iloc[1]['minute'], datetime(2024, 1, 1, 10, 1, 0))
        self.assertEqual(result.iloc[1]['events_per_minute'], 2)
        self.assertEqual(result.iloc[2]['minute'], datetime(2024, 1, 1, 10, 2, 0))
        self.assertEqual(result.iloc[2]['events_per_minute'], 1)


if __name__ == '__main__':
    unittest.main()
