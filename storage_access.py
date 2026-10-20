
# in our case the s3 bucket is the local "data" folder.
from pathlib import Path
from typing import Optional, Callable, TypeVar, Any
import json
import pandas as pd

T = TypeVar('T')


class StorageAccess:
    """Local storage access layer that simulates S3-like storage"""

    def __init__(self) -> None:
        self.bucket_path: Path = Path("./data")

    def check_if_file_exists(self, path: str) -> bool:
        """Check if a file exists in storage"""
        return (self.bucket_path / path).is_file()

    def read_json_from_storage(self, file_name: str) -> Optional[dict]:
        """Read JSON file from storage, returns None if file doesn't exist"""
        try:
            file_path = self.bucket_path / file_name
            with open(file_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def read_json_as_object(self, file_name: str, object_constructor: Callable[[dict], T]) -> Optional[T]:
        """
        Generic function to read a JSON file and convert it to an object.

        Args:
            file_name: The name of the JSON file to read
            object_constructor: A callable that takes the JSON dict and returns an object
                               This can be a class constructor, a factory function, or a lambda

        Returns:
            The object created by object_constructor, or None if file doesn't exist

        Example:
            # Simple case - class with __init__ that accepts **kwargs
            state = sa.read_json_as_object("state.json", lambda d: State(**d))

            # Complex case - custom conversion logic
            def create_data_cleaner_state(data):
                obj = DataCleanerState()
                obj.parse_json_state = State(**data["parse_json_state"])
                obj.clean_data_state = State(**data["clean_data_state"])
                return obj
            app_state = sa.read_json_as_object("state.json", create_data_cleaner_state)
        """
        json_data = self.read_json_from_storage(file_name)
        if json_data is None:
            return None
        return object_constructor(json_data)

    def read_from_storage(self, file_name: str) -> str:
        """Read file content as string"""
        try:
            file_path = self.bucket_path / file_name
            with open(file_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_name}")
        except PermissionError:
            raise PermissionError(f"Permission denied reading file: {file_name}")
        except Exception as e:
            raise IOError(f"Error reading file {file_name}: {str(e)}")

    def read_lines_from_storage(self, file_name: str) -> list[str]:
        """Read file content as list of lines"""
        try:
            file_path = self.bucket_path / file_name
            with open(file_path, "r") as f:
                return f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_name}")
        except PermissionError:
            raise PermissionError(f"Permission denied reading file: {file_name}")
        except Exception as e:
            raise IOError(f"Error reading file {file_name}: {str(e)}")

    def write_to_storage(self, file_name: str, content: str) -> None:
        """Write string content to file"""
        try:
            file_path = self.bucket_path / file_name
            with open(file_path, "w") as f:
                f.write(content)
        except PermissionError:
            raise PermissionError(f"Permission denied writing to file: {file_name}")
        except Exception as e:
            raise IOError(f"Error writing to file {file_name}: {str(e)}")

    def write_json_to_storage(self, file_name: str, content: Any) -> None:
        """Write content as JSON to file"""
        try:
            file_path = self.bucket_path / file_name
            with open(file_path, "w") as f:
                json.dump(content, f, indent=2)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Error serializing content to JSON for {file_name}: {str(e)}")
        except PermissionError:
            raise PermissionError(f"Permission denied writing to file: {file_name}")
        except Exception as e:
            raise IOError(f"Error writing JSON to file {file_name}: {str(e)}")

    def write_to_path(self, path: str, content: str) -> None:
        """Write content to a specific path"""
        try:
            with open(path, "w") as f:
                f.write(content)
        except PermissionError:
            raise PermissionError(f"Permission denied writing to path: {path}")
        except Exception as e:
            raise IOError(f"Error writing to path {path}: {str(e)}")

    def remove_from_storage(self, file_name: str) -> None:
        """Remove a file from storage"""
        try:
            file_path = self.bucket_path / file_name
            file_path.unlink()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found for removal: {file_name}")
        except PermissionError:
            raise PermissionError(f"Permission denied removing file: {file_name}")
        except Exception as e:
            raise IOError(f"Error removing file {file_name}: {str(e)}")

    def create_full_path_string(self, path: str) -> str:
        """Create full path string from relative path"""
        return str(self.bucket_path / path)

    def create_folder(self, folder_name: str) -> None:
        """Create a folder in storage"""
        folder_path = self.bucket_path / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)

    def save_dataframe_as_picklefile(self, df: pd.DataFrame, file_name: str) -> None:
        """Save a pandas DataFrame as a pickle file"""
        try:
            file_path = self.bucket_path / file_name
            df.to_pickle(str(file_path))
        except PermissionError:
            raise PermissionError(f"Permission denied saving pickle file: {file_name}")
        except Exception as e:
            raise IOError(f"Error saving dataframe as pickle {file_name}: {str(e)}")

    def transform_pkl_dataframe_to_csv(self, pkl_file_name: str, csv_file_name: str) -> None:
        """Convert a pickle DataFrame to CSV format"""
        try:
            pkl_path = self.bucket_path / pkl_file_name
            csv_path = self.bucket_path / csv_file_name
            df = pd.read_pickle(str(pkl_path))
            df.to_csv(str(csv_path), index=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"Pickle file not found: {pkl_file_name}")
        except PermissionError:
            raise PermissionError(f"Permission denied during pickle to CSV conversion")
        except Exception as e:
            raise IOError(f"Error transforming pickle {pkl_file_name} to CSV {csv_file_name}: {str(e)}")

    def read_picklefile_as_dataframe(self, file_name: str) -> pd.DataFrame:
        """Read a pickle file as a pandas DataFrame"""
        try:
            file_path = self.bucket_path / file_name
            return pd.read_pickle(str(file_path))
        except FileNotFoundError:
            raise FileNotFoundError(f"Pickle file not found: {file_name}")
        except PermissionError:
            raise PermissionError(f"Permission denied reading pickle file: {file_name}")
        except Exception as e:
            raise IOError(f"Error reading pickle file {file_name}: {str(e)}")

    def save_file_stats(self, file_stats: Any, file_name: str) -> None:
        """Save file statistics as JSON"""
        try:
            file_path = self.bucket_path / file_name
            with open(file_path, "w") as f:
                json.dump([fs.__dict__ for fs in file_stats.files], f, default=str, indent=2)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Error serializing file stats to JSON for {file_name}: {str(e)}")
        except PermissionError:
            raise PermissionError(f"Permission denied saving file stats: {file_name}")
        except Exception as e:
            raise IOError(f"Error saving file stats to {file_name}: {str(e)}")

    def save_state(self, state: dict, file_name: str) -> None:
        """Save state as JSON"""
        try:
            file_path = self.bucket_path / file_name
            with open(file_path, "w") as f:
                json.dump(state, f, default=str, indent=2)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Error serializing state to JSON for {file_name}: {str(e)}")
        except PermissionError:
            raise PermissionError(f"Permission denied saving state: {file_name}")
        except Exception as e:
            raise IOError(f"Error saving state to {file_name}: {str(e)}")

    def list_files_in_folder(self, folder_path: str, pattern: str = "*") -> list[str]:
        """
        List all files in a folder that match the given pattern.

        Args:
            folder_path: Relative path to the folder (e.g., "org/sys/proj/initial_data")
            pattern: Glob pattern to match files (default: "*" for all files)

        Returns:
            List of file paths relative to bucket_path

        Example:
            files = sa.list_files_in_folder("org/sys/proj/initial_data", "work_*.json")
            # Returns: ["org/sys/proj/initial_data/work_1.json", "org/sys/proj/initial_data/work_2.json", ...]
        """
        try:
            full_folder_path = self.bucket_path / folder_path
            if not full_folder_path.exists():
                return []

            if not full_folder_path.is_dir():
                return []

            matching_files = []
            for file_path in full_folder_path.glob(pattern):
                if file_path.is_file():
                    relative_path = file_path.relative_to(self.bucket_path)
                    matching_files.append(str(relative_path))

            return sorted(matching_files)
        except Exception as e:
            raise IOError(f"Error listing files in folder {folder_path}: {str(e)}")
