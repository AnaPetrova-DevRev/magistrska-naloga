from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class FileStatistics:
    """Statistics for a file processed by the system"""
    file_name: str = ""
    size: int = 0
    last_modified: Optional[datetime] = None
    created_date: Optional[datetime] = None
    input_file: str = ""
    output_file: str = ""
    created_by_function: str = ""

    def to_string(self) -> str:
        return f"File name: {self.file_name}, Size: {self.size}, Last modified: {self.last_modified}, Created date: {self.created_date}, Input file: {self.input_file}, Output file: {self.output_file}, Created by function: {self.created_by_function}"

    @staticmethod
    def from_dict(data: dict) -> 'FileStatistics':
        """Create a FileStatistics object from a dictionary"""
        return FileStatistics(
            file_name=data.get("file_name", ""),
            size=data.get("size", 0),
            last_modified=data.get("last_modified", None),
            created_date=data.get("created_date", None),
            input_file=data.get("input_file", ""),
            output_file=data.get("output_file", ""),
            created_by_function=data.get("created_by_function", "")
        )


@dataclass
class Stats:
    """Collection of file statistics"""
    files: list[FileStatistics] = field(default_factory=list)

    def add_file_statistics(self, file_statistics: FileStatistics) -> None:
        """Add or update file statistics. If file already exists, replace it."""
        existing_file = self.find_file_statistics(file_statistics.file_name)
        if existing_file is not None:
            self.files.remove(existing_file)
        self.files.append(file_statistics)

    def find_file_statistics(self, file_name: str) -> Optional[FileStatistics]:
        """Find file statistics by file name"""
        for file_statistics in self.files:
            if file_statistics.file_name == file_name:
                return file_statistics
        return None

    @staticmethod
    def from_dict(data: dict | list) -> 'Stats':
        """
        Create a Stats object from JSON data.
        The data can be either:
        - A list of file statistics dictionaries (as saved by save_file_stats)
        - A dictionary with a "files" key containing the list
        """
        obj = Stats()
        if isinstance(data, list):
            # Data is directly an array of file statistics
            obj.files = [FileStatistics.from_dict(fs) for fs in data]
        elif isinstance(data, dict) and "files" in data:
            # Data is a dictionary with a "files" key
            obj.files = [FileStatistics.from_dict(fs) for fs in data["files"]]
        return obj