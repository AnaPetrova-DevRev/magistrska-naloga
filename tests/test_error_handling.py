"""
Tests for error handling in storage operations and parsing.
"""
import unittest
import tempfile
import shutil
from pathlib import Path
import os

from storage_access import StorageAccess
from data_cleanup.parse_json import parse_json_file, ParsedItem


class TestStorageErrorHandling(unittest.TestCase):
    """Test error handling in StorageAccess"""

    def setUp(self):
        """Create a temporary directory for testing"""
        self.test_dir = tempfile.mkdtemp()
        self.sa = StorageAccess()
        self.sa.bucket_path = Path(self.test_dir)

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.test_dir)

    def test_read_nonexistent_file_raises_error(self):
        """Test reading nonexistent file raises FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            self.sa.read_from_storage("nonexistent.txt")

    def test_read_lines_nonexistent_file_raises_error(self):
        """Test reading lines from nonexistent file raises FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            self.sa.read_lines_from_storage("nonexistent.txt")

    def test_remove_nonexistent_file_raises_error(self):
        """Test removing nonexistent file raises FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            self.sa.remove_from_storage("nonexistent.txt")

    def test_read_pickle_nonexistent_file_raises_error(self):
        """Test reading nonexistent pickle raises FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            self.sa.read_picklefile_as_dataframe("nonexistent.pkl")

    def test_transform_nonexistent_pickle_raises_error(self):
        """Test transforming nonexistent pickle raises FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            self.sa.transform_pkl_dataframe_to_csv("nonexistent.pkl", "output.csv")


class TestParseJsonErrorHandling(unittest.TestCase):
    """Test error handling in JSON parsing"""

    def setUp(self):
        """Create a temporary directory for testing"""
        self.test_dir = tempfile.mkdtemp()
        self.sa = StorageAccess()
        self.sa.bucket_path = Path(self.test_dir)

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.test_dir)

    def test_parse_empty_file(self):
        """Test parsing empty file returns empty list"""
        # Create empty file
        empty_file = self.sa.bucket_path / "empty.json"
        empty_file.write_text("")

        items, logs = parse_json_file(str(empty_file))

        self.assertEqual(len(items), 0)
        self.assertIsInstance(logs, list)

    def test_parse_file_with_empty_lines(self):
        """Test parsing file with empty lines (should skip them)"""
        content = """
{
  "id": "123"
\x20\x20
  "created_date": "2026-02-07 12:00:00"
  "modified_date": "2026-02-07 13:00:00"
}
"""
        test_file = self.sa.bucket_path / "test.json"
        test_file.write_text(content)

        items, logs = parse_json_file(str(test_file))

        # Should handle empty lines gracefully
        self.assertIsInstance(items, list)
        self.assertIsInstance(logs, list)

    def test_parse_file_with_missing_fields(self):
        """Test parsing file with missing required fields"""
        content = """{
  "id": "123"
  "created_date": "2026-02-07 12:00:00"
}"""
        test_file = self.sa.bucket_path / "test.json"
        test_file.write_text(content)

        items, logs = parse_json_file(str(test_file))

        # Should log the error
        self.assertGreater(len(logs), 0)
        # Should not add incomplete item
        self.assertEqual(len(items), 0)

    def test_parse_file_with_invalid_date_format(self):
        """Test parsing file with invalid date format"""
        content = """{
  "id": "123",
  "created_date": "invalid-date",
  "modified_date": "2026-02-07 13:00:00"
}"""
        test_file = self.sa.bucket_path / "test.json"
        test_file.write_text(content)

        items, logs = parse_json_file(str(test_file))

        # Should log the parsing error
        self.assertGreater(len(logs), 0)

    def test_parse_valid_file(self):
        """Test parsing a valid file"""
        content = """{
  "id": "123",
  "created_date": "2026-02-07 12:00:00",
  "modified_date": "2026-02-07 13:00:00"
}"""
        test_file = self.sa.bucket_path / "test.json"
        test_file.write_text(content)

        items, logs = parse_json_file(str(test_file))

        # Should successfully parse
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].id, "123")
        self.assertIsNotNone(items[0].created_date)
        self.assertIsNotNone(items[0].modified_date)


class TestParsedItemValidation(unittest.TestCase):
    """Test ParsedItem validation"""

    def test_validate_complete_item(self):
        """Test validating a complete item"""
        from datetime import datetime
        item = ParsedItem(
            id="123",
            created_date=datetime.now(),
            modified_date=datetime.now()
        )

        self.assertTrue(item.validate())

    def test_validate_incomplete_item_no_id(self):
        """Test validating item without ID"""
        from datetime import datetime
        item = ParsedItem(
            id="",
            created_date=datetime.now(),
            modified_date=datetime.now()
        )

        self.assertFalse(item.validate())

    def test_validate_incomplete_item_no_dates(self):
        """Test validating item without dates"""
        item = ParsedItem(id="123")

        self.assertFalse(item.validate())

    def test_reset_items(self):
        """Test resetting item fields"""
        from datetime import datetime
        item = ParsedItem(
            id="123",
            created_date=datetime.now(),
            modified_date=datetime.now()
        )

        item.reset_items()

        self.assertEqual(item.id, "")
        self.assertIsNone(item.created_date)
        self.assertIsNone(item.modified_date)


if __name__ == '__main__':
    unittest.main()
