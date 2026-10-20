""" Should parse the original json that exists in the data folder. """

import sys
import os
import json

from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import pandas as pd
import storage_access

from sync_unit import SyncUnit
from file_stats import Stats

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

@dataclass
class ParsedItem:
    """Represents a parsed item from JSON data"""
    id: str = ""
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None

    def validate(self) -> bool:
        """Check if all required fields are filled"""
        return self.id != "" and self.created_date is not None and self.modified_date is not None

    def reset_items(self) -> None:
        """Reset all fields to default values"""
        self.id = ""
        self.created_date = None
        self.modified_date = None

class JsonDataParser:
    """Parser for JSON data files"""
    def __init__(self, sync_unit_obj: SyncUnit, file_stats: Stats) -> None:
        self.sync_unit_obj: SyncUnit = sync_unit_obj
        self.chunk_number: int = 0
        self.items: list[ParsedItem] = []
        self.file_stats: Stats = file_stats


def parse_json_file(file_path: str) -> tuple[list[ParsedItem], list[str]]:
    """
    Parse JSON file containing one JSON object per line.

    This function now uses proper JSON parsing instead of string splitting,
    which handles null values and various JSON formats correctly.
    """
    items = []
    logs = []

    sa = storage_access.StorageAccess()
    file_content = sa.read_lines_from_storage(file_path)

    # Accumulate lines for each JSON object
    current_json_lines = []

    for line_num, line in enumerate(file_content, start=1):
        try:
            # Strip whitespace
            line = line.strip()
            if not line:
                continue

            # Accumulate lines until we have a complete JSON object
            current_json_lines.append(line)

            # Check if we have a complete JSON object (ends with '}')
            if line == '}':
                # Join all lines and parse as JSON
                json_str = '\n'.join(current_json_lines)
                line_start = line_num - len(current_json_lines) + 1

                try:
                    # Parse the JSON object
                    json_obj = json.loads(json_str)

                    # Extract fields
                    item = ParsedItem()
                    item.id = json_obj.get('id', '')

                    # Handle created_date (might be null)
                    created_date_str = json_obj.get('created_date')
                    if created_date_str is not None:
                        try:
                            item.created_date = pd.to_datetime(created_date_str, format='ISO8601')
                        except (ValueError, TypeError):
                            try:
                                item.created_date = pd.to_datetime(created_date_str, format='%Y-%m-%d %H:%M:%S')
                            except (ValueError, TypeError):
                                logs.append(f"Line {line_start}: Could not parse created_date: {created_date_str}")
                    else:
                        logs.append(f"Line {line_start}: created_date is null for id={item.id}")

                    # Handle modified_date (might be null)
                    modified_date_str = json_obj.get('modified_date')
                    if modified_date_str is not None:
                        try:
                            item.modified_date = pd.to_datetime(modified_date_str, format='ISO8601')
                        except (ValueError, TypeError):
                            try:
                                item.modified_date = pd.to_datetime(modified_date_str, format='%Y-%m-%d %H:%M:%S')
                            except (ValueError, TypeError):
                                logs.append(f"Line {line_start}: Could not parse modified_date: {modified_date_str}")
                    else:
                        logs.append(f"Line {line_start}: modified_date is null for id={item.id}")

                    # Validate and add item
                    if item.validate():
                        items.append(item)
                    else:
                        logs.append(f"Line {line_start}: Skipping item with missing required fields")
                        logs.append(f"Item data: id={item.id}, created_date={item.created_date}, modified_date={item.modified_date}")

                except json.JSONDecodeError as e:
                    logs.append(f"Line {line_start}: JSON parsing error - {str(e)}")
                except Exception as e:
                    logs.append(f"Line {line_start}: Unexpected error - {str(e)}")

                # Reset for next object
                current_json_lines = []

        except Exception as e:
            logs.append(f"Line {line_num}: Unexpected error - {str(e)}")
            continue

    return items, logs

def save_parsed_items(
    sa: storage_access.StorageAccess,
    result_file_path: str,
    items: list[ParsedItem],
) -> None:
    """Save parsed items as a pickle file and return the file path"""
    df = pd.DataFrame(items)
    sa.save_dataframe_as_picklefile(df, result_file_path)