"""Synthetic correctness tests for outlier detection."""
import unittest
import numpy as np
import pandas as pd
from pipeline_explorer.outlier_detection import OutlierMethod, detect_zscore_outliers, detect_modified_zscore_outliers, detect_iqr_outliers, detect_seasonal_outliers, detect_outliers, can_detect_outliers

class TestDetectZscoreOutliers(unittest.TestCase):
    """Tests for Z-score outlier detection."""

    def test_basic_outlier_detection(self):
        """Should detect obvious outliers."""
        values = pd.Series([10, 10, 10, 10, 10, 100])
        result = detect_zscore_outliers(values, threshold=2.0)
        self.assertTrue(result.iloc[-1] is True or result.iloc[-1] == True)
        self.assertTrue(result.iloc[0] is False or result.iloc[0] == False)

    def test_empty_series(self):
        """Should return empty series for empty input."""
        values = pd.Series([], dtype=float)
        result = detect_zscore_outliers(values)
        self.assertTrue(len(result) == 0)

    def test_single_value(self):
        """Should return False for single value (std is NaN or 0)."""
        values = pd.Series([42.0])
        result = detect_zscore_outliers(values)
        self.assertTrue(len(result) == 1)
        self.assertTrue(result.iloc[0] == False)

    def test_all_same_values(self):
        """Should return all False when all values are identical."""
        values = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])
        result = detect_zscore_outliers(values)
        self.assertTrue(all(result == False))

    def test_preserves_index(self):
        """Should preserve the original index."""
        values = pd.Series([1, 2, 100], index=['a', 'b', 'c'])
        result = detect_zscore_outliers(values, threshold=2.0)
        self.assertTrue(list(result.index) == ['a', 'b', 'c'])

    def test_nan_values(self):
        """Should handle NaN values gracefully."""
        values = pd.Series([10, 10, np.nan, 10, 100])
        result = detect_zscore_outliers(values, threshold=2.0)
        self.assertTrue(len(result) == 5)

    def test_negative_values(self):
        """Should work with negative values."""
        values = pd.Series([-10] * 20 + [-100])
        result = detect_zscore_outliers(values, threshold=2.0)
        self.assertTrue(result.iloc[-1] == True)

class TestDetectModifiedZscoreOutliers(unittest.TestCase):
    """Tests for Modified Z-score outlier detection."""

    def test_basic_outlier_detection(self):
        """Should detect obvious outliers."""
        values = pd.Series([10, 10, 10, 10, 10, 100])
        result = detect_modified_zscore_outliers(values, threshold=2.0)
        self.assertTrue(result.iloc[-1] == True)

    def test_empty_series(self):
        """Should return empty series for empty input."""
        values = pd.Series([], dtype=float)
        result = detect_modified_zscore_outliers(values)
        self.assertTrue(len(result) == 0)

    def test_all_same_values_zero_mad(self):
        """Should handle zero MAD (all values identical)."""
        values = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])
        result = detect_modified_zscore_outliers(values)
        self.assertTrue(all(result == False))

    def test_single_value(self):
        """Should return False for single value."""
        values = pd.Series([42.0])
        result = detect_modified_zscore_outliers(values)
        self.assertTrue(len(result) == 1)
        self.assertTrue(result.iloc[0] == False)

    def test_two_identical_values(self):
        """Should handle two identical values (MAD = 0)."""
        values = pd.Series([5.0, 5.0])
        result = detect_modified_zscore_outliers(values)
        self.assertTrue(all(result == False))

class TestDetectIqrOutliers(unittest.TestCase):
    """Tests for IQR outlier detection."""

    def test_basic_outlier_detection(self):
        """Should detect outliers outside IQR bounds."""
        values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        result = detect_iqr_outliers(values, multiplier=1.5)
        self.assertTrue(result.iloc[-1] == True)

    def test_empty_series(self):
        """Should return empty series for empty input."""
        values = pd.Series([], dtype=float)
        result = detect_iqr_outliers(values)
        self.assertTrue(len(result) == 0)

    def test_all_same_values_zero_iqr(self):
        """Should handle zero IQR (all values identical)."""
        values = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])
        result = detect_iqr_outliers(values)
        self.assertTrue(all(result == False))

    def test_single_value(self):
        """Should return False for single value."""
        values = pd.Series([42.0])
        result = detect_iqr_outliers(values)
        self.assertTrue(len(result) == 1)
        self.assertTrue(result.iloc[0] == False)

class TestDetectSeasonalOutliers(unittest.TestCase):
    """Tests for seasonal outlier detection."""

    def test_basic_detection(self):
        """Should detect outliers after removing hourly pattern."""
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        values = [10] * 99 + [1000]
        df = pd.DataFrame({'date': dates, 'value': values})
        result = detect_seasonal_outliers(df, 'date', 'value', threshold=2.0)
        self.assertTrue(result.iloc[-1] == True)

    def test_empty_dataframe(self):
        """Should handle empty DataFrame."""
        df = pd.DataFrame({'date': [], 'value': []})
        result = detect_seasonal_outliers(df, 'date', 'value')
        self.assertTrue(len(result) == 0)

    def test_single_hour_data(self):
        """Should handle data from single hour."""
        dates = pd.to_datetime(['2024-01-01 10:00'] * 5)
        values = [10, 10, 10, 10, 100]
        df = pd.DataFrame({'date': dates, 'value': values})
        result = detect_seasonal_outliers(df, 'date', 'value', threshold=2.0)
        self.assertTrue(len(result) == 5)

class TestDetectOutliers(unittest.TestCase):
    """Tests for the main detect_outliers dispatcher function."""

    def test_zscore_method(self):
        """Should route to Z-score detection."""
        df = pd.DataFrame({'value': [10] * 20 + [100]})
        result = detect_outliers(df, 'value', OutlierMethod.ZSCORE, threshold=2.0)
        self.assertTrue(result.iloc[-1] == True)

    def test_modified_zscore_method(self):
        """Should route to Modified Z-score detection."""
        df = pd.DataFrame({'value': [10, 10, 10, 100]})
        result = detect_outliers(df, 'value', OutlierMethod.MODIFIED_ZSCORE, threshold=2.0)
        self.assertTrue(result.iloc[-1] == True)

    def test_iqr_method(self):
        """Should route to IQR detection with threshold conversion."""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 100]})
        result = detect_outliers(df, 'value', OutlierMethod.IQR, threshold=3.0)
        self.assertTrue(result.iloc[-1] == True)

    def test_seasonal_method_with_date(self):
        """Should use seasonal detection when date column provided."""
        dates = pd.date_range('2024-01-01', periods=50, freq='h')
        values = [10] * 49 + [1000]
        df = pd.DataFrame({'date': dates, 'value': values})
        result = detect_outliers(df, 'value', OutlierMethod.SEASONAL, date_column='date', threshold=2.0)
        self.assertTrue(result.iloc[-1] == True)

    def test_seasonal_method_without_date_falls_back(self):
        """Should fall back to Z-score when no date column."""
        df = pd.DataFrame({'value': [10, 10, 10, 100]})
        result = detect_outliers(df, 'value', OutlierMethod.SEASONAL, threshold=2.0)
        self.assertTrue(len(result) == 4)

class TestCanDetectOutliers(unittest.TestCase):
    """Tests for can_detect_outliers function."""

    def test_supported_file_types(self):
        """Should return True for supported file types."""
        self.assertTrue(can_detect_outliers('path/to/cleaned_items_1.pkl') is True)
        self.assertTrue(can_detect_outliers('path/to/created_time_delta_1.pkl') is True)
        self.assertTrue(can_detect_outliers('path/to/modified_time_delta_1.pkl') is True)
        self.assertTrue(can_detect_outliers('path/to/merged_time_delta_1.pkl') is True)

    def test_unsupported_file_types(self):
        """Should return False for unsupported file types."""
        self.assertTrue(can_detect_outliers('path/to/work_1.json') is False)
        self.assertTrue(can_detect_outliers('path/to/parsed_items_1.pkl') is False)
        self.assertTrue(can_detect_outliers('path/to/random_file.csv') is False)

class TestEdgeCases(unittest.TestCase):
    """Tests for various edge cases and potential bugs."""

    def test_very_small_std(self):
        """Should handle very small standard deviation."""
        values = pd.Series([1.0] * 50 + [100.0])
        result = detect_zscore_outliers(values, threshold=3.0)
        self.assertTrue(result.iloc[-1] == True)

    def test_inf_values(self):
        """Should handle infinity values."""
        values = pd.Series([10, 10, 10, np.inf])
        result = detect_zscore_outliers(values, threshold=3.0)
        self.assertTrue(len(result) == 4)

    def test_mixed_nan_and_outliers(self):
        """Should handle mix of NaN and real outliers."""
        values = pd.Series([10, np.nan, 10, 10, 100])
        result = detect_iqr_outliers(values, multiplier=1.5)
        self.assertTrue(len(result) == 5)
        self.assertTrue(result.iloc[-1] == True)

    def test_timedelta_values(self):
        """Should handle timedelta values (common in our use case)."""
        values = pd.Series([pd.Timedelta(hours=1)] * 20 + [pd.Timedelta(hours=100)])
        numeric_values = values.dt.total_seconds()
        result = detect_zscore_outliers(numeric_values, threshold=2.0)
        self.assertTrue(result.iloc[-1] == True)


if __name__ == "__main__":
    unittest.main()
