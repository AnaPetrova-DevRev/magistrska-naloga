"""
Tests for StorageAccess class with pathlib implementation.
"""
import unittest
import json
import tempfile
import shutil
from pathlib import Path
import pandas as pd

from storage_access import StorageAccess
from file_stats import FileStatistics, Stats
from state import State


class TestStorageAccessPathlib(unittest.TestCase):
    """Test StorageAccess with pathlib implementation"""

    def setUp(self):
        """Create a temporary directory for testing"""
        self.test_dir = tempfile.mkdtemp()
        self.sa = StorageAccess()
        # Override bucket_path to use temp directory
        self.sa.bucket_path = Path(self.test_dir)

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.test_dir)

    def test_bucket_path_is_path_object(self):
        """Test that bucket_path is a Path object"""
        self.assertIsInstance(self.sa.bucket_path, Path)

    def test_check_if_file_exists(self):
        """Test checking if file exists"""
        # Create a test file
        test_file = self.sa.bucket_path / "test.txt"
        test_file.write_text("test content")

        # Check it exists
        self.assertTrue(self.sa.check_if_file_exists("test.txt"))
        self.assertFalse(self.sa.check_if_file_exists("nonexistent.txt"))

    def test_write_and_read_from_storage(self):
        """Test writing and reading text files"""
        content = "Hello, World!"
        self.sa.write_to_storage("test.txt", content)

        read_content = self.sa.read_from_storage("test.txt")
        self.assertEqual(read_content, content)

    def test_write_and_read_json(self):
        """Test writing and reading JSON files"""
        data = {"key": "value", "number": 42}
        self.sa.write_json_to_storage("test.json", data)

        read_data = self.sa.read_json_from_storage("test.json")
        self.assertEqual(read_data["key"], "value")
        self.assertEqual(read_data["number"], 42)

    def test_read_nonexistent_json_returns_none(self):
        """Test reading nonexistent JSON returns None"""
        result = self.sa.read_json_from_storage("nonexistent.json")
        self.assertIsNone(result)

    def test_read_lines_from_storage(self):
        """Test reading file as lines"""
        content = "Line 1\nLine 2\nLine 3"
        self.sa.write_to_storage("test.txt", content)

        lines = self.sa.read_lines_from_storage("test.txt")
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0].strip(), "Line 1")

    def test_create_folder(self):
        """Test creating folders"""
        self.sa.create_folder("test_folder/subfolder")

        folder_path = self.sa.bucket_path / "test_folder" / "subfolder"
        self.assertTrue(folder_path.exists())
        self.assertTrue(folder_path.is_dir())

    def test_remove_from_storage(self):
        """Test removing files"""
        self.sa.write_to_storage("test.txt", "content")
        self.assertTrue(self.sa.check_if_file_exists("test.txt"))

        self.sa.remove_from_storage("test.txt")
        self.assertFalse(self.sa.check_if_file_exists("test.txt"))

    def test_create_full_path_string(self):
        """Test creating full path string"""
        full_path = self.sa.create_full_path_string("subfolder/file.txt")

        # Should be a string
        self.assertIsInstance(full_path, str)
        # Should contain the bucket path
        self.assertIn(str(self.sa.bucket_path), full_path)
        # Should contain the relative path
        self.assertIn("subfolder/file.txt", full_path)

    def test_save_and_read_dataframe_pickle(self):
        """Test saving and reading DataFrame as pickle"""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })

        self.sa.save_dataframe_as_picklefile(df, "test.pkl")

        read_df = self.sa.read_picklefile_as_dataframe("test.pkl")
        self.assertEqual(len(read_df), 3)
        self.assertEqual(list(read_df['name']), ['Alice', 'Bob', 'Charlie'])

    def test_transform_pkl_to_csv(self):
        """Test converting pickle to CSV"""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })

        self.sa.save_dataframe_as_picklefile(df, "test.pkl")
        self.sa.transform_pkl_dataframe_to_csv("test.pkl", "test.csv")

        # Check CSV was created
        csv_path = self.sa.bucket_path / "test.csv"
        self.assertTrue(csv_path.exists())

        # Read and verify
        read_df = pd.read_csv(csv_path)
        self.assertEqual(len(read_df), 3)

    def test_save_and_read_file_stats(self):
        """Test saving and reading file statistics"""
        stats = Stats()
        stats.add_file_statistics(FileStatistics(file_name="test1.pkl", size=100))
        stats.add_file_statistics(FileStatistics(file_name="test2.pkl", size=200))

        self.sa.save_file_stats(stats, "stats.json")

        # Read back as JSON
        data = self.sa.read_json_from_storage("stats.json")
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["file_name"], "test1.pkl")

    def test_save_and_read_state(self):
        """Test saving and reading state"""
        state_data = {
            "last_processed_chunk_number": 5,
            "is_done": False,
            "total_processed": 100
        }

        self.sa.save_state(state_data, "state.json")

        read_data = self.sa.read_json_from_storage("state.json")
        self.assertEqual(read_data["last_processed_chunk_number"], 5)
        self.assertFalse(read_data["is_done"])

    def test_read_json_as_object(self):
        """Test reading JSON and converting to object"""
        state_data = {
            "last_processed_chunk_number": 5,
            "is_done": False,
            "total_processed": 100,
            "function_name": "parse_json"
        }

        self.sa.write_json_to_storage("state.json", state_data)

        # Read as State object
        state = self.sa.read_json_as_object("state.json", State.from_dict)

        self.assertIsInstance(state, State)
        self.assertEqual(state.last_processed_chunk_number, 5)
        self.assertEqual(state.function_name, "parse_json")

    def test_list_files_in_folder(self):
        """Test listing files in a folder"""
        # Create a test folder with some files
        self.sa.create_folder("test_folder")
        self.sa.write_to_storage("test_folder/file1.txt", "content1")
        self.sa.write_to_storage("test_folder/file2.txt", "content2")
        self.sa.write_to_storage("test_folder/data.json", "content3")

        # List all files
        all_files = self.sa.list_files_in_folder("test_folder")
        self.assertEqual(len(all_files), 3)
        self.assertIn("test_folder/file1.txt", all_files)
        self.assertIn("test_folder/file2.txt", all_files)
        self.assertIn("test_folder/data.json", all_files)

        # List only .txt files
        txt_files = self.sa.list_files_in_folder("test_folder", "*.txt")
        self.assertEqual(len(txt_files), 2)
        self.assertIn("test_folder/file1.txt", txt_files)
        self.assertIn("test_folder/file2.txt", txt_files)

        # List only .json files
        json_files = self.sa.list_files_in_folder("test_folder", "*.json")
        self.assertEqual(len(json_files), 1)
        self.assertIn("test_folder/data.json", json_files)

    def test_list_files_in_nonexistent_folder(self):
        """Test listing files in a folder that doesn't exist"""
        files = self.sa.list_files_in_folder("nonexistent_folder")
        self.assertEqual(files, [])

    def test_list_files_with_pattern(self):
        """Test listing files with specific pattern"""
        # Create test files with numbers
        self.sa.create_folder("numbered_folder")
        self.sa.write_to_storage("numbered_folder/work_1.json", "content")
        self.sa.write_to_storage("numbered_folder/work_2.json", "content")
        self.sa.write_to_storage("numbered_folder/work_10.json", "content")
        self.sa.write_to_storage("numbered_folder/other.txt", "content")

        # List work_*.json files
        work_files = self.sa.list_files_in_folder("numbered_folder", "work_*.json")
        self.assertEqual(len(work_files), 3)
        self.assertIn("numbered_folder/work_1.json", work_files)
        self.assertIn("numbered_folder/work_2.json", work_files)
        self.assertIn("numbered_folder/work_10.json", work_files)


if __name__ == '__main__':
    unittest.main()
