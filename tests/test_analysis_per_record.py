"""
Tests for analysis per record functionality.

Tests cover:
- Day of week statistics calculation
- Hour of day statistics calculation
- Time delta statistics calculation
- Delta bucket categorization
- Single record analysis
- Full analysis pipeline
"""
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from pipeline_explorer.analysis_per_record import (
    calculate_day_of_week_stats,
    calculate_hour_of_day_stats,
    calculate_time_delta_stats,
    calculate_delta_bucket_stats,
    analyze_single_record,
    load_all_clean_data,
    calculate_analysis_per_record,
    ONE_SECOND,
    TEN_SECONDS,
    ONE_MINUTE,
    ONE_HOUR,
    FOUR_HOURS,
    ONE_DAY,
    DAY_NAMES
)


class TestDayOfWeekStats(unittest.TestCase):
    """Test calculate_day_of_week_stats function."""

    def test_all_monday_changes(self):
        """Test when all changes occur on Monday."""
        # Create data with all Mondays (2024-01-01 was a Monday)
        df = pd.DataFrame({
            'id': ['1', '1', '1'],
            'modified_date': pd.to_datetime([
                '2024-01-01 10:00:00',
                '2024-01-08 11:00:00',
                '2024-01-15 12:00:00'
            ])
        })

        stats = calculate_day_of_week_stats(df)

        self.assertEqual(stats['num_changes_monday'], 3)
        self.assertEqual(stats['pct_changes_monday'], 100.0)
        self.assertEqual(stats['num_changes_tuesday'], 0)
        self.assertEqual(stats['num_changes_weekday'], 3)
        self.assertEqual(stats['num_changes_weekend'], 0)

    def test_weekend_changes(self):
        """Test when changes occur on weekends."""
        # 2024-01-06 was Saturday, 2024-01-07 was Sunday
        df = pd.DataFrame({
            'id': ['1', '1'],
            'modified_date': pd.to_datetime([
                '2024-01-06 10:00:00',  # Saturday
                '2024-01-07 10:00:00'   # Sunday
            ])
        })

        stats = calculate_day_of_week_stats(df)

        self.assertEqual(stats['num_changes_saturday'], 1)
        self.assertEqual(stats['num_changes_sunday'], 1)
        self.assertEqual(stats['num_changes_weekend'], 2)
        self.assertEqual(stats['num_changes_weekday'], 0)

    def test_mixed_days(self):
        """Test with changes across multiple days."""
        df = pd.DataFrame({
            'id': ['1'] * 5,
            'modified_date': pd.to_datetime([
                '2024-01-01 10:00:00',  # Monday
                '2024-01-02 10:00:00',  # Tuesday
                '2024-01-03 10:00:00',  # Wednesday
                '2024-01-06 10:00:00',  # Saturday
                '2024-01-07 10:00:00'   # Sunday
            ])
        })

        stats = calculate_day_of_week_stats(df)

        self.assertEqual(stats['num_changes_monday'], 1)
        self.assertEqual(stats['num_changes_tuesday'], 1)
        self.assertEqual(stats['num_changes_wednesday'], 1)
        self.assertEqual(stats['num_changes_weekday'], 3)
        self.assertEqual(stats['num_changes_weekend'], 2)
        self.assertAlmostEqual(stats['pct_changes_monday'], 20.0, places=1)


class TestHourOfDayStats(unittest.TestCase):
    """Test calculate_hour_of_day_stats function."""

    def test_single_hour(self):
        """Test when all changes occur in one hour."""
        df = pd.DataFrame({
            'id': ['1', '1', '1'],
            'modified_date': pd.to_datetime([
                '2024-01-01 10:00:00',
                '2024-01-02 10:30:00',
                '2024-01-03 10:45:00'
            ])
        })

        stats = calculate_hour_of_day_stats(df)

        self.assertEqual(stats['num_changes_hour_10'], 3)
        self.assertEqual(stats['num_changes_hour_00'], 0)
        self.assertEqual(stats['num_changes_hour_23'], 0)

    def test_multiple_hours(self):
        """Test with changes across multiple hours."""
        df = pd.DataFrame({
            'id': ['1'] * 4,
            'modified_date': pd.to_datetime([
                '2024-01-01 00:30:00',  # Hour 0
                '2024-01-01 12:30:00',  # Hour 12
                '2024-01-01 12:45:00',  # Hour 12
                '2024-01-01 23:30:00'   # Hour 23
            ])
        })

        stats = calculate_hour_of_day_stats(df)

        self.assertEqual(stats['num_changes_hour_00'], 1)
        self.assertEqual(stats['num_changes_hour_12'], 2)
        self.assertEqual(stats['num_changes_hour_23'], 1)

    def test_all_24_columns_exist(self):
        """Test that all 24 hour columns are created."""
        df = pd.DataFrame({
            'id': ['1'],
            'modified_date': pd.to_datetime(['2024-01-01 10:00:00'])
        })

        stats = calculate_hour_of_day_stats(df)

        for hour in range(24):
            self.assertIn(f'num_changes_hour_{hour:02d}', stats)


class TestTimeDeltaStats(unittest.TestCase):
    """Test calculate_time_delta_stats function."""

    def test_empty_deltas(self):
        """Test with no deltas (single modification)."""
        deltas = np.array([])

        stats = calculate_time_delta_stats(deltas)

        self.assertIsNone(stats['avg_time_between_modifications'])
        self.assertIsNone(stats['min_time_between_modifications'])
        self.assertIsNone(stats['max_time_between_modifications'])
        self.assertIsNone(stats['std_time_between_modifications'])
        self.assertIsNone(stats['q1_time_between_modifications'])
        self.assertIsNone(stats['q3_time_between_modifications'])

    def test_single_delta(self):
        """Test with a single delta."""
        deltas = np.array([3600.0])  # 1 hour in seconds

        stats = calculate_time_delta_stats(deltas)

        self.assertEqual(stats['avg_time_between_modifications'], 3600.0)
        self.assertEqual(stats['min_time_between_modifications'], 3600.0)
        self.assertEqual(stats['max_time_between_modifications'], 3600.0)
        self.assertEqual(stats['std_time_between_modifications'], 0.0)

    def test_multiple_deltas(self):
        """Test with multiple deltas."""
        deltas = np.array([60.0, 120.0, 180.0, 240.0])  # 1, 2, 3, 4 minutes

        stats = calculate_time_delta_stats(deltas)

        self.assertEqual(stats['avg_time_between_modifications'], 150.0)
        self.assertEqual(stats['min_time_between_modifications'], 60.0)
        self.assertEqual(stats['max_time_between_modifications'], 240.0)
        # Q1 and Q3 use numpy's default linear interpolation
        self.assertAlmostEqual(
            stats['q1_time_between_modifications'], 105.0, places=1
        )
        self.assertAlmostEqual(
            stats['q3_time_between_modifications'], 195.0, places=1
        )


class TestDeltaBucketStats(unittest.TestCase):
    """Test calculate_delta_bucket_stats function."""

    def test_empty_deltas(self):
        """Test with no deltas."""
        deltas = np.array([])

        stats = calculate_delta_bucket_stats(deltas)

        self.assertEqual(stats['num_deltas_lt_1s'], 0)
        self.assertEqual(stats['num_deltas_1s_to_10s'], 0)
        self.assertEqual(stats['num_deltas_gt_1d'], 0)

    def test_sub_second_deltas(self):
        """Test deltas less than 1 second."""
        deltas = np.array([0.1, 0.5, 0.9])

        stats = calculate_delta_bucket_stats(deltas)

        self.assertEqual(stats['num_deltas_lt_1s'], 3)
        self.assertEqual(stats['num_deltas_1s_to_10s'], 0)

    def test_various_buckets(self):
        """Test deltas falling into various buckets."""
        deltas = np.array([
            0.5,           # < 1s
            5.0,           # 1s-10s
            30.0,          # 10s-1min
            1800.0,        # 1min-1h (30 min)
            7200.0,        # 1h-4h (2 hours)
            43200.0,       # 4h-1d (12 hours)
            172800.0,      # > 1d (2 days)
            864000.0,      # > 1w (10 days)
            3456000.0      # > 1mo (40 days)
        ])

        stats = calculate_delta_bucket_stats(deltas)

        self.assertEqual(stats['num_deltas_lt_1s'], 1)
        self.assertEqual(stats['num_deltas_1s_to_10s'], 1)
        self.assertEqual(stats['num_deltas_10s_to_1min'], 1)
        self.assertEqual(stats['num_deltas_1min_to_1h'], 1)
        self.assertEqual(stats['num_deltas_1h_to_4h'], 1)
        self.assertEqual(stats['num_deltas_4h_to_1d'], 1)
        self.assertEqual(stats['num_deltas_gt_1d'], 3)  # 2d, 10d, 40d
        self.assertEqual(stats['num_deltas_gt_1w'], 2)  # 10d, 40d
        self.assertEqual(stats['num_deltas_gt_1mo'], 1)  # 40d

    def test_boundary_values(self):
        """Test boundary values between buckets."""
        deltas = np.array([
            ONE_SECOND - 0.1,      # Should be < 1s
            ONE_SECOND,            # Should be 1s-10s
            TEN_SECONDS,           # Should be 10s-1min
            ONE_MINUTE,            # Should be 1min-1h
            ONE_HOUR,              # Should be 1h-4h
            FOUR_HOURS,            # Should be 4h-1d
            ONE_DAY,               # Should be > 1d
        ])

        stats = calculate_delta_bucket_stats(deltas)

        self.assertEqual(stats['num_deltas_lt_1s'], 1)
        self.assertEqual(stats['num_deltas_1s_to_10s'], 1)
        self.assertEqual(stats['num_deltas_10s_to_1min'], 1)
        self.assertEqual(stats['num_deltas_1min_to_1h'], 1)
        self.assertEqual(stats['num_deltas_1h_to_4h'], 1)
        self.assertEqual(stats['num_deltas_4h_to_1d'], 1)
        self.assertEqual(stats['num_deltas_gt_1d'], 1)


class TestAnalyzeSingleRecord(unittest.TestCase):
    """Test analyze_single_record function."""

    def test_basic_analysis(self):
        """Test basic single record analysis."""
        df = pd.DataFrame({
            'id': ['record1', 'record1', 'record1'],
            'created_date': pd.to_datetime([
                '2024-01-01 10:00:00',
                '2024-01-01 10:00:00',
                '2024-01-01 10:00:00'
            ]),
            'modified_date': pd.to_datetime([
                '2024-01-01 10:00:00',  # Same as created (creation event)
                '2024-01-01 11:00:00',  # First actual modification
                '2024-01-01 12:00:00'   # Second modification
            ])
        })
        cutoff_date = pd.Timestamp('2024-07-01')

        stats = analyze_single_record(df, cutoff_date)

        self.assertEqual(stats['id'], 'record1')
        self.assertEqual(
            stats['first_created_date'],
            pd.Timestamp('2024-01-01 10:00:00')
        )
        self.assertEqual(
            stats['first_modified_date'],
            pd.Timestamp('2024-01-01 11:00:00')
        )
        self.assertEqual(
            stats['last_modified_date'],
            pd.Timestamp('2024-01-01 12:00:00')
        )
        self.assertEqual(stats['num_changes_total'], 3)

    def test_time_spans(self):
        """Test time span calculations."""
        df = pd.DataFrame({
            'id': ['record1', 'record1'],
            'created_date': pd.to_datetime([
                '2024-01-01 10:00:00',
                '2024-01-01 10:00:00'
            ]),
            'modified_date': pd.to_datetime([
                '2024-01-01 10:00:00',  # Creation
                '2024-01-01 12:00:00'   # 2 hours later
            ])
        })
        cutoff_date = pd.Timestamp('2024-07-01')

        stats = analyze_single_record(df, cutoff_date)

        # Time from creation to last modified: 2 hours = 7200 seconds
        self.assertEqual(stats['time_first_created_to_last_modified'], 7200.0)
        # Time to first modification: 2 hours = 7200 seconds
        self.assertEqual(stats['time_to_first_modification'], 7200.0)

    def test_no_modifications_after_creation(self):
        """Test when modified_date equals created_date (no actual modifications)."""
        df = pd.DataFrame({
            'id': ['record1'],
            'created_date': pd.to_datetime(['2024-01-01 10:00:00']),
            'modified_date': pd.to_datetime(['2024-01-01 10:00:00'])
        })
        cutoff_date = pd.Timestamp('2024-07-01')

        stats = analyze_single_record(df, cutoff_date)

        self.assertIsNone(stats['first_modified_date'])
        self.assertIsNone(stats['time_to_first_modification'])

    def test_was_created_3mo_before_cutoff(self):
        """Test 3-month cutoff check."""
        # Created 4 months before cutoff
        df_old = pd.DataFrame({
            'id': ['record1'],
            'created_date': pd.to_datetime(['2024-01-01 10:00:00']),
            'modified_date': pd.to_datetime(['2024-01-01 10:00:00'])
        })
        cutoff_date = pd.Timestamp('2024-05-01')  # 4 months later

        stats_old = analyze_single_record(df_old, cutoff_date)
        self.assertTrue(stats_old['was_created_3mo_before_cutoff'])

        # Created 2 months before cutoff
        df_new = pd.DataFrame({
            'id': ['record2'],
            'created_date': pd.to_datetime(['2024-03-01 10:00:00']),
            'modified_date': pd.to_datetime(['2024-03-01 10:00:00'])
        })

        stats_new = analyze_single_record(df_new, cutoff_date)
        self.assertFalse(stats_new['was_created_3mo_before_cutoff'])

    def test_all_columns_present(self):
        """Test that all expected columns are present in output."""
        df = pd.DataFrame({
            'id': ['record1', 'record1'],
            'created_date': pd.to_datetime([
                '2024-01-01 10:00:00',
                '2024-01-01 10:00:00'
            ]),
            'modified_date': pd.to_datetime([
                '2024-01-01 10:00:00',
                '2024-01-02 10:00:00'
            ])
        })
        cutoff_date = pd.Timestamp('2024-07-01')

        stats = analyze_single_record(df, cutoff_date)

        # Check key columns exist
        expected_cols = [
            'id', 'first_created_date', 'first_modified_date', 'last_modified_date',
            'time_first_created_to_last_modified', 'time_to_first_modification',
            'num_changes_weekday', 'num_changes_weekend',
            'avg_time_between_modifications', 'min_time_between_modifications',
            'max_time_between_modifications', 'std_time_between_modifications',
            'q1_time_between_modifications', 'q3_time_between_modifications',
            'num_deltas_lt_1s', 'num_deltas_gt_1d',
            'was_created_3mo_before_cutoff', 'num_changes_total'
        ]

        for col in expected_cols:
            self.assertIn(col, stats, f"Missing column: {col}")

        # Check all day columns
        for day in DAY_NAMES:
            self.assertIn(f'num_changes_{day}', stats)
            self.assertIn(f'pct_changes_{day}', stats)

        # Check all hour columns
        for hour in range(24):
            self.assertIn(f'num_changes_hour_{hour:02d}', stats)


class TestLoadAllCleanData(unittest.TestCase):
    """Test load_all_clean_data function with mocking."""

    @patch('pipeline_explorer.analysis_per_record.StorageAccess')
    def test_no_clean_data_files(self, mock_sa_class):
        """Test when no clean data files exist (folder missing or empty)."""
        mock_sa = MagicMock()
        mock_sa_class.return_value = mock_sa
        mock_sa.check_if_file_exists.return_value = False

        mock_sync_unit = MagicMock()
        mock_sync_unit.get_path_for_sync_unit.return_value = 'org/sys/proj'

        result = load_all_clean_data(mock_sync_unit)

        self.assertIsNone(result)

    @patch('pipeline_explorer.analysis_per_record.StorageAccess')
    def test_load_single_file(self, mock_sa_class):
        """Test loading a single clean data file."""
        mock_sa = MagicMock()
        mock_sa_class.return_value = mock_sa

        test_df = pd.DataFrame({
            'id': ['1', '2'],
            'created_date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'modified_date': pd.to_datetime(['2024-01-01', '2024-01-02'])
        })

        def check_exists(path):
            if 'cleaned_items_1.pkl' in path:
                return True
            if 'cleaned_items_2.pkl' in path:
                return False
            return 'clean_data' in path

        mock_sa.check_if_file_exists.side_effect = check_exists
        mock_sa.read_picklefile_as_dataframe.return_value = test_df

        mock_sync_unit = MagicMock()
        mock_sync_unit.get_path_for_sync_unit.return_value = 'org/sys/proj'

        result = load_all_clean_data(mock_sync_unit)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    @patch('pipeline_explorer.analysis_per_record.StorageAccess')
    def test_load_multiple_files(self, mock_sa_class):
        """Test loading and combining multiple clean data files."""
        mock_sa = MagicMock()
        mock_sa_class.return_value = mock_sa

        df1 = pd.DataFrame({
            'id': ['1', '2'],
            'created_date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'modified_date': pd.to_datetime(['2024-01-01', '2024-01-02'])
        })
        df2 = pd.DataFrame({
            'id': ['3', '4'],
            'created_date': pd.to_datetime(['2024-01-03', '2024-01-04']),
            'modified_date': pd.to_datetime(['2024-01-03', '2024-01-04'])
        })

        def check_exists(path):
            if 'cleaned_items_1.pkl' in path:
                return True
            if 'cleaned_items_2.pkl' in path:
                return True
            if 'cleaned_items_3.pkl' in path:
                return False
            return 'clean_data' in path

        def read_pickle(path):
            if 'cleaned_items_1.pkl' in path:
                return df1
            if 'cleaned_items_2.pkl' in path:
                return df2
            return None

        mock_sa.check_if_file_exists.side_effect = check_exists
        mock_sa.read_picklefile_as_dataframe.side_effect = read_pickle

        mock_sync_unit = MagicMock()
        mock_sync_unit.get_path_for_sync_unit.return_value = 'org/sys/proj'

        result = load_all_clean_data(mock_sync_unit)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)  # Combined 2 + 2


class TestCalculateAnalysisPerRecord(unittest.TestCase):
    """Test calculate_analysis_per_record function."""

    @patch('pipeline_explorer.analysis_per_record.load_all_clean_data')
    def test_no_data_returns_none(self, mock_load):
        """Test when no data is available."""
        mock_load.return_value = None

        mock_sync_unit = MagicMock()
        result = calculate_analysis_per_record(mock_sync_unit)

        self.assertIsNone(result)

    @patch('pipeline_explorer.analysis_per_record.load_all_clean_data')
    def test_empty_dataframe_returns_none(self, mock_load):
        """Test when empty DataFrame is returned."""
        mock_load.return_value = pd.DataFrame()

        mock_sync_unit = MagicMock()
        result = calculate_analysis_per_record(mock_sync_unit)

        self.assertIsNone(result)

    @patch('pipeline_explorer.analysis_per_record.load_all_clean_data')
    def test_single_record_analysis(self, mock_load):
        """Test analysis with a single record."""
        test_df = pd.DataFrame({
            'id': ['record1', 'record1', 'record1'],
            'created_date': pd.to_datetime([
                '2024-01-01 10:00:00',
                '2024-01-01 10:00:00',
                '2024-01-01 10:00:00'
            ]),
            'modified_date': pd.to_datetime([
                '2024-01-01 10:00:00',
                '2024-01-01 11:00:00',
                '2024-01-01 12:00:00'
            ])
        })
        mock_load.return_value = test_df

        mock_sync_unit = MagicMock()
        result = calculate_analysis_per_record(mock_sync_unit)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['id'], 'record1')
        self.assertEqual(result.iloc[0]['num_changes_total'], 3)

    @patch('pipeline_explorer.analysis_per_record.load_all_clean_data')
    def test_multiple_records_analysis(self, mock_load):
        """Test analysis with multiple records."""
        test_df = pd.DataFrame({
            'id': ['record1', 'record1', 'record2', 'record2', 'record3'],
            'created_date': pd.to_datetime([
                '2024-01-01 10:00:00',
                '2024-01-01 10:00:00',
                '2024-01-02 10:00:00',
                '2024-01-02 10:00:00',
                '2024-01-03 10:00:00'
            ]),
            'modified_date': pd.to_datetime([
                '2024-01-01 10:00:00',
                '2024-01-01 11:00:00',
                '2024-01-02 10:00:00',
                '2024-01-02 12:00:00',
                '2024-01-03 10:00:00'
            ])
        })
        mock_load.return_value = test_df

        mock_sync_unit = MagicMock()
        result = calculate_analysis_per_record(mock_sync_unit)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)

        # Check each record
        record_ids = result['id'].tolist()
        self.assertIn('record1', record_ids)
        self.assertIn('record2', record_ids)
        self.assertIn('record3', record_ids)


if __name__ == '__main__':
    unittest.main()
