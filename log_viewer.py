"""
Log Viewer Utility - Read and filter log files for sync units and actions.

This module provides functionality to:
1. List available log files for a sync unit
2. Read and parse log files
3. Filter log lines based on search criteria
4. Group logs by action type
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import re
import sync_unit

logger = logging.getLogger(__name__)


class LogEntry:
    """Represents a single log entry"""
    def __init__(self, timestamp: str, logger_name: str, level: str, message: str, raw_line: str):
        self.timestamp = timestamp
        self.logger_name = logger_name
        self.level = level
        self.message = message
        self.raw_line = raw_line

    def matches_filter(self, search_text: str = "", level_filter: Optional[str] = None) -> bool:
        """Check if this log entry matches the given filters"""
        if level_filter and self.level != level_filter:
            return False

        if search_text:
            search_lower = search_text.lower()
            return (search_lower in self.message.lower() or
                   search_lower in self.logger_name.lower() or
                   search_lower in self.raw_line.lower())

        return True


class LogFile:
    """Represents a log file with metadata"""
    def __init__(self, file_path: Path, log_type: str, action: Optional[str] = None):
        self.file_path = file_path
        self.log_type = log_type  # "processing" or "orchestrator"
        self.file_name = file_path.name
        self.timestamp = self._extract_timestamp()
        self.action = action  # Action name extracted from the log file

    def _extract_timestamp(self) -> Optional[datetime]:
        """Extract timestamp from filename"""
        try:
            # Format: processing_2026-02-14_13-08-57.log or orchestrator_logs_14-02-2026_13-08-07.txt
            if "processing_" in self.file_name:
                # Extract: 2026-02-14_13-08-57
                match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', self.file_name)
                if match:
                    return datetime.strptime(match.group(1), '%Y-%m-%d_%H-%M-%S')
            elif "orchestrator_logs_" in self.file_name:
                # Extract: 14-02-2026_13-08-07
                match = re.search(r'(\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2})', self.file_name)
                if match:
                    return datetime.strptime(match.group(1), '%d-%m-%Y_%H-%M-%S')
        except Exception as e:
            logger.warning(f"Could not parse timestamp from {self.file_name}: {e}")
        return None


def get_log_files_for_sync_unit(sync_unit_obj: sync_unit.SyncUnit) -> Dict[str, List[LogFile]]:
    """
    Get all log files for a sync unit, grouped by type.

    Args:
        sync_unit_obj: The sync unit to get logs for

    Returns:
        Dictionary with keys "processing" and "orchestrator", each containing a list of LogFile objects
    """
    base_path = Path("./data") / sync_unit_obj.get_path_for_sync_unit()

    result = {
        "processing": [],
        "orchestrator": []
    }

    # Get processing logs
    processing_logs_path = base_path / "logs"
    if processing_logs_path.exists():
        for log_file_path in processing_logs_path.glob("processing_*.log"):
            # Extract action name from the log file
            action = get_action_from_log_file_path(log_file_path)
            log_file = LogFile(log_file_path, "processing", action)
            result["processing"].append(log_file)

    # Get orchestrator logs
    orchestrator_logs_path = base_path / "orchestrator"
    if orchestrator_logs_path.exists():
        for log_file_path in orchestrator_logs_path.glob("orchestrator_logs_*.txt"):
            # Extract action name from the log file
            action = get_action_from_log_file_path(log_file_path)
            log_file = LogFile(log_file_path, "orchestrator", action)
            result["orchestrator"].append(log_file)

    # Sort by timestamp (newest first)
    for log_type in result:
        result[log_type].sort(key=lambda x: x.timestamp if x.timestamp else datetime.min, reverse=True)

    return result


def parse_log_line(line: str) -> Optional[LogEntry]:
    """
    Parse a single log line into a LogEntry object.

    Expected format: 2026-02-14 13:08:57 - logger_name - LEVEL - message
    """
    # Pattern: timestamp - logger_name - level - message
    pattern = r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - ([^ ]+) - ([A-Z]+) - (.+)$'
    match = re.match(pattern, line)

    if match:
        timestamp, logger_name, level, message = match.groups()
        return LogEntry(timestamp, logger_name, level, message, line)

    return None


def read_log_file(log_file: LogFile,
                  search_text: str = "",
                  level_filter: Optional[str] = None,
                  max_lines: int = 1000) -> Tuple[List[LogEntry], int]:
    """
    Read and parse a log file with optional filtering.

    Args:
        log_file: The LogFile object to read
        search_text: Optional text to search for in log lines
        level_filter: Optional log level to filter by (INFO, WARNING, ERROR, etc.)
        max_lines: Maximum number of lines to return

    Returns:
        Tuple of (list of LogEntry objects, total number of matching lines)
    """
    entries = []
    total_matches = 0
    current_entry = None

    try:
        with open(log_file.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line:
                    continue

                entry = parse_log_line(line)

                # If we can parse it as a structured log, it's a new entry
                if entry is not None:
                    # Save the previous entry if it exists and matches filters
                    if current_entry and current_entry.matches_filter(search_text, level_filter):
                        total_matches += 1
                        if len(entries) < max_lines:
                            entries.append(current_entry)

                    current_entry = entry
                else:
                    # This is a continuation line (e.g., traceback)
                    # Append it to the current entry's message and raw_line
                    if current_entry:
                        current_entry.message += "\n" + line
                        current_entry.raw_line += "\n" + line
                    else:
                        # No current entry, treat as standalone
                        current_entry = LogEntry("", "", "", line, line)

            # Don't forget the last entry
            if current_entry and current_entry.matches_filter(search_text, level_filter):
                total_matches += 1
                if len(entries) < max_lines:
                    entries.append(current_entry)

    except Exception as e:
        logger.error(f"Error reading log file {log_file.file_path}: {e}")

    return entries, total_matches


def get_action_from_log_file_path(file_path: Path) -> Optional[str]:
    """
    Extract the action name from a log file by reading its content.

    Args:
        file_path: Path to the log file

    Returns:
        The action name if found, None otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Look for "Action: <action_name>" pattern
                match = re.search(r'Action:\s+(\w+)', line)
                if match:
                    return match.group(1)
    except Exception as e:
        logger.error(f"Error reading log file {file_path}: {e}")

    return None


def get_action_from_log_file(log_file: LogFile) -> Optional[str]:
    """
    Get the action name from a LogFile object.

    Args:
        log_file: The LogFile object

    Returns:
        The action name if available, None otherwise
    """
    return log_file.action


def get_unique_actions_from_logs(log_files: List[LogFile]) -> List[str]:
    """
    Get a sorted list of unique action names from a list of log files.

    Args:
        log_files: List of LogFile objects

    Returns:
        Sorted list of unique action names (excluding None)
    """
    actions = set()
    for log_file in log_files:
        if log_file.action:
            actions.add(log_file.action)

    return sorted(list(actions))
