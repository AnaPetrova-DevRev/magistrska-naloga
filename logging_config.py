"""
Logging configuration for the application.
Import this module at the start of your application to configure logging.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(level: str = "INFO", log_file: str = None, log_to_console: bool = True) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, logs only to console.
        log_to_console: Whether to also log to console (default: True)

    Example:
        # Console only (default)
        setup_logging(level="INFO")

        # Console + file
        setup_logging(level="DEBUG", log_file="./data/logs/app.log")

        # File only (no console output)
        setup_logging(level="INFO", log_file="./data/logs/app.log", log_to_console=False)
    """
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler (optional)
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    if log_to_console:
        logging.info("Logging configured with level: %s", level)


def get_log_file_path(base_path: str = "./data/logs") -> str:
    """
    Generate a timestamped log file path.

    Args:
        base_path: Base directory for logs

    Returns:
        Full path to log file with timestamp

    Example:
        log_path = get_log_file_path()
        # Returns: "./data/logs/app_2026-02-07_14-30-45.log"
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return f"{base_path}/app_{timestamp}.log"


def setup_sync_unit_logging(sync_unit_path: str, level: str = "INFO", log_to_console: bool = True) -> str:
    """
    Configure logging for a specific sync unit.
    Creates a timestamped log file in the sync unit's logs folder.

    Args:
        sync_unit_path: Path to the sync unit (e.g., "org1/sys1/proj1")
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_console: Whether to also log to console (default: True)

    Returns:
        Path to the created log file

    Example:
        log_file = setup_sync_unit_logging("org1/sys1/proj1")
        # Logs will be saved to: ./data/org1/sys1/proj1/logs/processing_2026-02-07_14-30-45.log
    """
    # Create log directory path
    log_dir = f"./data/{sync_unit_path}/logs"

    # Generate timestamped log file name
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = f"{log_dir}/processing_{timestamp}.log"

    # Setup logging with the generated path
    setup_logging(level=level, log_file=log_file, log_to_console=log_to_console)

    return log_file


# Default configuration - can be overridden by calling setup_logging()
setup_logging(level="INFO")
