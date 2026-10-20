"""
Tests for calculate_time_delta functionality
"""
import unittest
from datetime import datetime

import pandas as pd

from data_preprocessing.calculate_time_delta import (
    calculate_time_deltas_for_column,
    save_time_deltas
)
import storage_access


class TestCalculateTimeDeltasForColumn(unittest.TestCase):
    """Test cases for calculate_time_deltas_for_column function"""

    def test_calculate_time_deltas_basic(self):
        """Test basic time delta calculation without unique_id"""
        # Create test dataframe with sequential timestamps
        df = pd.DataFrame({
            'id': ['1', '2', '3'],
            'created_date': [
                datetime(2024, 1, 1, 10, 0, 0),
                datetime(2024, 1, 1, 10, 5, 0),  # 5 minutes later
                datetime(2024, 1, 1, 10, 15, 0)  # 10 minutes later
            ]
        })

        result = calculate_time_deltas_for_column(
            df,
            unique_id=False,
            time_column='created_date'
        )

        # Check that delta column was created
        self.assertIn('created_date_delta', result.columns)

        # First row should have NaN delta
        self.assertTrue(pd.isna(result.iloc[0]['created_date_delta']))

        # Second row: 5 minutes
        self.assertEqual(result.iloc[1]['created_date_delta'], 5)

        # Third row: 10 minutes
        self.assertEqual(result.iloc[2]['created_date_delta'], 10)

    def test_calculate_time_deltas_with_unique_id(self):
        """Test time delta calculation with unique_id=True"""
        # Create test dataframe with duplicate IDs
        df = pd.DataFrame({
            'id': ['1', '1', '2', '2', '3'],
            'created_date': [
                datetime(2024, 1, 1, 10, 0, 0),
                datetime(2024, 1, 1, 10, 5, 0),  # Duplicate ID 1
                datetime(2024, 1, 1, 10, 10, 0),
                datetime(2024, 1, 1, 10, 15, 0),  # Duplicate ID 2
                datetime(2024, 1, 1, 10, 20, 0)
            ]
        })

        result = calculate_time_deltas_for_column(
            df,
            unique_id=True,
            time_column='created_date',
            id_column='id'
        )

        # Should only have 3 rows (one per unique ID)
        self.assertEqual(len(result), 3)

        # Check that we kept the first occurrence of each ID
        self.assertEqual(result.iloc[0]['id'], '1')
        self.assertEqual(result.iloc[1]['id'], '2')
        self.assertEqual(result.iloc[2]['id'], '3')

        # Check deltas
        self.assertTrue(pd.isna(result.iloc[0]['created_date_delta']))
        self.assertEqual(result.iloc[1]['created_date_delta'], 10)  # 10 min
        self.assertEqual(result.iloc[2]['created_date_delta'], 10)  # 10 min

    def test_calculate_time_deltas_sorting(self):
        """Test that dataframe is sorted by time column"""
        # Create unsorted dataframe
        df = pd.DataFrame({
            'id': ['1', '2', '3'],
            'created_date': [
                datetime(2024, 1, 1, 10, 15, 0),  # Out of order
                datetime(2024, 1, 1, 10, 0, 0),   # Earliest
                datetime(2024, 1, 1, 10, 5, 0)    # Middle
            ]
        })

        result = calculate_time_deltas_for_column(
            df,
            unique_id=False,
            time_column='created_date'
        )

        # Check that result is sorted
        self.assertEqual(
            result.iloc[0]['created_date'],
            datetime(2024, 1, 1, 10, 0, 0)
        )
        self.assertEqual(
            result.iloc[1]['created_date'],
            datetime(2024, 1, 1, 10, 5, 0)
        )
        self.assertEqual(
            result.iloc[2]['created_date'],
            datetime(2024, 1, 1, 10, 15, 0)
        )

    def test_calculate_time_deltas_modified_date(self):
        """Test time delta calculation for modified_date column"""
        df = pd.DataFrame({
            'id': ['1', '1', '1'],
            'modified_date': [
                datetime(2024, 1, 1, 10, 0, 0),
                datetime(2024, 1, 1, 11, 0, 0),  # 1 hour later
                datetime(2024, 1, 1, 11, 30, 0)  # 30 minutes later
            ]
        })

        result = calculate_time_deltas_for_column(
            df,
            unique_id=False,
            time_column='modified_date'
        )

        # Check delta column name
        self.assertIn('modified_date_delta', result.columns)

        # Check deltas
        self.assertTrue(pd.isna(result.iloc[0]['modified_date_delta']))
        self.assertEqual(result.iloc[1]['modified_date_delta'], 60)  # 1 hour
        self.assertEqual(result.iloc[2]['modified_date_delta'], 30)  # 30 min


class TestSaveTimeDeltasAndIntegration(unittest.TestCase):
    """Test cases for save_time_deltas and integration functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.sa = storage_access.StorageAccess()
        self.test_files = []

    def tearDown(self):
        """Clean up test files"""
        for file_path in self.test_files:
            try:
                if self.sa.check_if_file_exists(file_path):
                    self.sa.remove_from_storage(file_path)
            except (FileNotFoundError, PermissionError, IOError):
                pass

    def test_save_time_deltas(self):
        """Test saving time deltas to pickle file"""
        df = pd.DataFrame({
            'id': ['1', '2'],
            'created_date_delta': [None, 300.0]
        })

        file_path = "test_time_delta.pkl"
        self.test_files.append(file_path)

        result_path = save_time_deltas(self.sa, file_path, df)

        # Check that file was created
        self.assertTrue(self.sa.check_if_file_exists(file_path))
        self.assertEqual(result_path, file_path)

        # Read back and verify
        loaded_df = self.sa.read_picklefile_as_dataframe(file_path)
        self.assertEqual(len(loaded_df), 2)
        self.assertIn('created_date_delta', loaded_df.columns)


if __name__ == '__main__':
    unittest.main()
