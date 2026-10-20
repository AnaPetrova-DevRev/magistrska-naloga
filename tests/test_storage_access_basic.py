import unittest
import storage_access

class TestStorageAccess(unittest.TestCase):
    def test_check_if_file_exists(self):
        sa = storage_access.StorageAccess()
        self.assertFalse(sa.check_if_file_exists("test.txt"))
    def test_write_and_read_from_storage(self):
        sa = storage_access.StorageAccess()
        sa.write_to_storage("test.txt", "Hello, World!")
        self.assertTrue(sa.check_if_file_exists("test.txt"))
        content = sa.read_from_storage("test.txt")
        self.assertEqual(content, "Hello, World!")
        sa.remove_from_storage("test.txt")

if __name__ == '__main__':
    unittest.main()