import logging
import sys
import os
from typing import Optional

# Default log format
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def get_logger(
    name: str = "jaxflow",
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    fmt: str = DEFAULT_FORMAT
) -> logging.Logger:
    """
    Get or configure a logger with specific settings.

    Args:
        name: The name of the logger (default: "jaxflow").
        level: The logging level (e.g., "INFO", "DEBUG"). Defaults to "INFO" or JAXFLOW_LOG_LEVEL env var.
        log_file: Path to a file to log to. If None, logs only to console.
        fmt: The logging format string.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # If logger already has handlers, assume it's configured unless we force it.
    # But here we just return it to avoid duplicate logs.
    if logger.handlers:
        return logger

    # Determine log level
    if level is None:
        level = os.environ.get("JAXFLOW_LOG_LEVEL", "INFO").upper()
    
    try:
        log_level = getattr(logging, level)
    except AttributeError:
        log_level = logging.INFO

    logger.setLevel(log_level)
    logger.propagate = False  # Prevent propagation to root logger to avoid double logging

    formatter = logging.Formatter(fmt, datefmt=DATE_FORMAT)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # Fallback to console warning if file logging fails
            print(f"Warning: Could not set up file logging to {log_file}: {e}", file=sys.stderr)

    return logger
