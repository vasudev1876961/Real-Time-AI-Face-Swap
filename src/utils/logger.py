"""
Structured Logging Module for Real-Time AI Face Swap Application.
"""

import os
import sys
import logging
from typing import Optional

_LOGGERS = {}
DEFAULT_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = "logs/app.log",
    enable_console: bool = True,
) -> None:
    """Configures the root logging handlers and formatting."""
    handlers = []
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DATE_FORMAT)

    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        handlers.append(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(level=log_level, handlers=handlers, force=True)


def get_logger(name: str = "FaceSwapApp") -> logging.Logger:
    """Returns a named logger instance with standardized formatting."""
    if name not in _LOGGERS:
        logger = logging.getLogger(name)
        if not logger.handlers and not logging.getLogger().handlers:
            # Fallback initialization if setup_logging hasn't been called
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DATE_FORMAT))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        _LOGGERS[name] = logger
    return _LOGGERS[name]
