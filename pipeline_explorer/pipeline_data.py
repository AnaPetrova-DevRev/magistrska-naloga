"""
Pipeline Data Module

Provides functionality to combine pipeline metadata and validation results,
and save/load them for exploration in the Pipeline Explorer app.
"""
import logging
from dataclasses import dataclass, field
from os.path import join
from typing import Optional

from storage_access import StorageAccess
from sync_unit import SyncUnit
from file_metadata import FileMetadata, FileValidationResult
from app_flow_definitions import calculate_pipeline_metadata, validate_pipeline_file_stats
from global_vars import (
    INITIAL_DATA_FILE_NAME_FORMAT,
    PIPELINE_EXPLORER_APP_FOLDER_NAME,
    PIPELINE_METADATA_FILE_NAME,
    PIPELINE_METADATA_CACHE_FILE_NAME,
)

logger = logging.getLogger(__name__)


@dataclass
class FileDisplayData:
    """Combined metadata and validation data for a single file"""
    file_format: str
    file_path: Optional[str]
    exists: bool
    is_json: bool  # True for JSON files (non-clickable)

    # Metadata
    total_rows: Optional[int] = None
    unique_ids: Optional[int] = None
    date_columns: list = field(default_factory=list)  # List of DateColumnMetadata dicts
    metadata_error: Optional[str] = None

    # Validation
    has_stats: bool = False
    input_matches: Optional[bool] = None
    output_matches: Optional[bool] = None
    validation_issues: list[str] = field(default_factory=list)

    def has_errors(self) -> bool:
        """Check if this file has any errors or issues"""
        return bool(self.metadata_error or self.validation_issues)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "file_format": self.file_format,
            "file_path": self.file_path,
            "exists": self.exists,
            "is_json": self.is_json,
            "total_rows": self.total_rows,
            "unique_ids": self.unique_ids,
            "date_columns": self.date_columns,
            "metadata_error": self.metadata_error,
            "has_stats": self.has_stats,
            "input_matches": self.input_matches,
            "output_matches": self.output_matches,
            "validation_issues": self.validation_issues
        }


@dataclass
class PipelineDisplayData:
    """Combined metadata and validation data for an entire pipeline"""
    initial_file_path: str
    chunk_number: int
    files: dict[str, FileDisplayData] = field(default_factory=dict)
    missing_files: list[str] = field(default_factory=list)

    def has_errors(self) -> bool:
        """Check if any file in this pipeline has errors"""
        return any(f.has_errors() for f in self.files.values())

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "initial_file_path": self.initial_file_path,
            "chunk_number": self.chunk_number,
            "files": {fmt: f.to_dict() for fmt, f in self.files.items()},
            "missing_files": self.missing_files
        }


def get_combined_pipeline_data(sync_unit: SyncUnit) -> list[PipelineDisplayData]:
    """
    Get combined metadata and validation data for all pipelines in a sync unit.

    Args:
        sync_unit: The sync unit to get pipeline data for

    Returns:
        List of PipelineDisplayData objects, one per initial file,
        sorted by chunk number
    """
    # Get metadata for all pipelines
    metadata_dict = calculate_pipeline_metadata(sync_unit)

    result = []

    for initial_file_path, pipeline_metadata in metadata_dict.items():
        # Get validation results for this pipeline
        validation_results = validate_pipeline_file_stats(initial_file_path, sync_unit)

        # Create combined pipeline data
        pipeline_data = PipelineDisplayData(
            initial_file_path=initial_file_path,
            chunk_number=pipeline_metadata.chunk_number,
            missing_files=pipeline_metadata.missing_files.copy()
        )

        # Combine metadata and validation for each file
        for file_format, file_metadata in pipeline_metadata.files.items():
            file_display = _create_file_display_data(
                file_format,
                file_metadata,
                validation_results.get(file_format) if validation_results else None
            )
            pipeline_data.files[file_format] = file_display

        result.append(pipeline_data)

    # Sort by chunk number
    result.sort(key=lambda p: p.chunk_number)

    return result


def _create_file_display_data(
    file_format: str,
    metadata: FileMetadata,
    validation: Optional[FileValidationResult]
) -> FileDisplayData:
    """
    Create FileDisplayData by combining metadata and validation.

    Args:
        file_format: The file format string (e.g., "parsed_items_{int}.pkl")
        metadata: FileMetadata object
        validation: FileValidationResult object (may be None)

    Returns:
        Combined FileDisplayData object
    """
    is_json = file_format == INITIAL_DATA_FILE_NAME_FORMAT

    display = FileDisplayData(
        file_format=file_format,
        file_path=metadata.file_path,
        exists=metadata.exists,
        is_json=is_json,
        total_rows=metadata.total_rows,
        unique_ids=metadata.unique_ids,
        date_columns=[dc.to_dict() for dc in metadata.date_columns],
        metadata_error=metadata.error
    )

    # Add validation data if available
    if validation:
        display.has_stats = validation.has_stats
        display.input_matches = validation.input_matches
        display.output_matches = validation.output_matches
        display.validation_issues = validation.issues.copy()

    return display


def save_pipeline_data_to_storage(
    sync_unit: SyncUnit,
    pipelines: list[PipelineDisplayData],
    use_cache_filename: bool = False
) -> None:
    """
    Save pipeline data (metadata + validation) to storage.

    Args:
        sync_unit: The sync unit
        pipelines: List of PipelineDisplayData to save
        use_cache_filename: If True, use cache filename; if False, use orchestrator filename
    """
    sa = StorageAccess()

    # Ensure the pipeline explorer folder exists
    folder_path = join(sync_unit.get_path_for_sync_unit(), PIPELINE_EXPLORER_APP_FOLDER_NAME)
    sa.create_folder(folder_path)

    # Convert to dict for JSON serialization
    data = {
        "sync_unit": sync_unit.get_path_for_sync_unit(),
        "pipelines": [p.to_dict() for p in pipelines]
    }

    # Determine filename
    if use_cache_filename:
        metadata_filename = PIPELINE_METADATA_CACHE_FILE_NAME
    else:
        metadata_filename = PIPELINE_METADATA_FILE_NAME

    file_path = join(folder_path, metadata_filename)
    sa.write_json_to_storage(file_path, data)

    logger.info("Saved pipeline data to %s", file_path)


def load_cached_pipeline_data(sync_unit: SyncUnit) -> Optional[list[dict]]:
    """
    Load cached pipeline data from storage.

    Args:
        sync_unit: The sync unit

    Returns:
        List of pipeline data dicts, or None if not found
    """
    sa = StorageAccess()

    file_path = join(
        sync_unit.get_path_for_sync_unit(),
        PIPELINE_EXPLORER_APP_FOLDER_NAME,
        PIPELINE_METADATA_CACHE_FILE_NAME
    )

    data = sa.read_json_from_storage(file_path)
    if data and "pipelines" in data:
        return data["pipelines"]

    return None


def get_pipeline_page(
    pipelines: list[PipelineDisplayData],
    page_num: int,
    page_size: int = 10
) -> tuple[list[PipelineDisplayData], int, int]:
    """
    Get a page of pipelines for display.

    Args:
        pipelines: Full list of pipelines
        page_num: 1-based page number
        page_size: Number of pipelines per page (default: 10)

    Returns:
        Tuple of (page_pipelines, total_pages, current_page)
    """
    total_pipelines = len(pipelines)
    total_pages = max(1, (total_pipelines + page_size - 1) // page_size)

    # Clamp page number
    page_num = max(1, min(page_num, total_pages))

    start_idx = (page_num - 1) * page_size
    end_idx = start_idx + page_size

    return pipelines[start_idx:end_idx], total_pages, page_num
