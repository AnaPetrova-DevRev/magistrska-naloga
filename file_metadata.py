"""
Data classes for file metadata.

This module defines classes for storing metadata about files in the pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime


class TrainingReadinessStatus(Enum):
    """Status levels for training readiness validation."""
    READY = "ready"  # All good - ready to train
    WARNING = "warning"  # New unprocessed files exist, but can train
    ERROR = "error"  # Missing files or gaps - cannot train properly


@dataclass
class DateColumnMetadata:
    """Metadata for a date column in a file"""
    column_name: str
    earliest_date: Optional[datetime] = None
    latest_date: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "column_name": self.column_name,
            "earliest_date": str(self.earliest_date) if self.earliest_date else None,
            "latest_date": str(self.latest_date) if self.latest_date else None
        }


@dataclass
class FileMetadata:
    """Metadata for a single file in the pipeline"""
    file_path: Optional[str] = None
    exists: bool = False
    total_rows: Optional[int] = None
    unique_ids: Optional[int] = None
    date_columns: list[DateColumnMetadata] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "file_path": self.file_path,
            "exists": self.exists,
            "total_rows": self.total_rows,
            "unique_ids": self.unique_ids,
            "date_columns": [dc.to_dict() for dc in self.date_columns],
            "error": self.error
        }


@dataclass
class PipelineMetadata:
    """Metadata for an entire pipeline starting from an initial file"""
    initial_file_path: str
    chunk_number: int
    files: dict[str, FileMetadata] = field(default_factory=dict)
    missing_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "initial_file_path": self.initial_file_path,
            "chunk_number": self.chunk_number,
            "files": {fmt: meta.to_dict() for fmt, meta in self.files.items()},
            "missing_files": self.missing_files
        }


@dataclass
class FileFormatColumns:
    """Expected columns for a file format"""
    id_column: str = "id"
    date_columns: list[str] = field(default_factory=list)
    other_columns: list[str] = field(default_factory=list)

    def get_all_columns(self) -> list[str]:
        """Get all expected columns"""
        return [self.id_column] + self.date_columns + self.other_columns


@dataclass
class FileValidationResult:
    """Validation result for a single file in the pipeline"""
    file_path: Optional[str] = None
    exists: bool = False
    has_stats: bool = False
    expected_input: Optional[str] = None
    actual_input: Optional[str] = None
    input_matches: Optional[bool] = None
    expected_outputs: list[str] = field(default_factory=list)
    actual_outputs: list[str] = field(default_factory=list)
    output_matches: Optional[bool] = None
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "file_path": self.file_path,
            "exists": self.exists,
            "has_stats": self.has_stats,
            "expected_input": self.expected_input,
            "actual_input": self.actual_input,
            "input_matches": self.input_matches,
            "expected_outputs": self.expected_outputs,
            "actual_outputs": self.actual_outputs,
            "output_matches": self.output_matches,
            "issues": self.issues
        }


@dataclass
class PipelineValidationResult:
    """Validation results for an entire pipeline"""
    initial_file_path: str
    chunk_number: int
    files: dict[str, FileValidationResult] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "initial_file_path": self.initial_file_path,
            "chunk_number": self.chunk_number,
            "files": {fmt: result.to_dict() for fmt, result in self.files.items()}
        }


@dataclass
class TrainingReadinessResult:
    """Result of training readiness validation for a sync unit.

    Attributes:
        status: Overall status (READY, WARNING, ERROR)
        total_initial_files: Number of initial data files (work_X.json)
        processed_files: Number of files fully processed through pipeline
        unprocessed_files: Number of new files that haven't been processed yet
        errors: List of critical issues that prevent training
        warnings: List of non-critical issues
        missing_required_formats: List of required file formats that are missing
        gap_chunks: List of chunk numbers with gaps (e.g., 2 missing between 1 and 3)
        trailing_chunks: List of chunk numbers at the end that aren't processed
        chunk_details: Details for each chunk about what files exist
    """
    status: TrainingReadinessStatus
    total_initial_files: int = 0
    processed_files: int = 0
    unprocessed_files: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_required_formats: list[str] = field(default_factory=list)
    gap_chunks: list[int] = field(default_factory=list)
    trailing_chunks: list[int] = field(default_factory=list)
    chunk_details: dict[int, dict[str, bool]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "status": self.status.value,
            "total_initial_files": self.total_initial_files,
            "processed_files": self.processed_files,
            "unprocessed_files": self.unprocessed_files,
            "errors": self.errors,
            "warnings": self.warnings,
            "missing_required_formats": self.missing_required_formats,
            "gap_chunks": self.gap_chunks,
            "trailing_chunks": self.trailing_chunks,
            "chunk_details": self.chunk_details
        }

    def get_message(self) -> str:
        """Get a human-readable message summarizing the validation result."""
        if self.status == TrainingReadinessStatus.READY:
            return (
                f"Ready for training. All {self.processed_files} files "
                "processed successfully."
            )
        if self.status == TrainingReadinessStatus.WARNING:
            return (
                f"Can train with existing data. {self.processed_files} files "
                f"processed, {self.unprocessed_files} new files available."
            )
        # ERROR status
        return (
            f"Cannot train. {len(self.errors)} critical issue(s) found. "
            f"Gap in chunks: {self.gap_chunks}" if self.gap_chunks else
            f"Cannot train. {len(self.errors)} critical issue(s) found."
        )
