"""
Minimal stdlib-only logging setup with console and JSON file output.
"""

import json
import logging
import logging.handlers
import os
import sys
import traceback
from datetime import datetime, timezone


class ConsoleFormatter(logging.Formatter):
    """Single-line console formatter: YYYY-MM-DD HH:MM:SS | LEVEL | logger.name | message"""

    def format(self, record):
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        base = (
            f"{timestamp} | {record.levelname} | {record.name} | {record.getMessage()}"
        )

        # If there is exception info, append formatted traceback for console readability
        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            formatted = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            return f"{base}\n{formatted}"

        return base


class JSONFormatter(logging.Formatter):
    """JSON formatter with fields: ts, level, logger, message, pathname, lineno, funcName, process, threadName"""

    def __init__(self, ecs_compatible=False):
        super().__init__()
        self.ecs_compatible = ecs_compatible

    def format(self, record):
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()

        if self.ecs_compatible:
            log_obj = {
                "@timestamp": timestamp,
                "log.level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "pathname": record.pathname,
                "lineno": record.lineno,
                "funcName": record.funcName,
                "process": record.process,
                "threadName": record.threadName,
            }
        else:
            log_obj = {
                "ts": timestamp,
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "pathname": record.pathname,
                "lineno": record.lineno,
                "funcName": record.funcName,
                "process": record.process,
                "threadName": record.threadName,
            }

        # Attach structured exception information when available
        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            # Full formatted exception as list of strings (keeps newlines intact when joined)
            formatted = traceback.format_exception(exc_type, exc_value, exc_tb)
            # Extract frame-level info for better programmatic analysis
            tb_frames = traceback.extract_tb(exc_tb)
            frames = [
                {
                    "filename": f.filename,
                    "lineno": f.lineno,
                    "name": f.name,
                    "line": f.line,
                }
                for f in tb_frames
            ]

            log_obj["exc"] = {
                "type": exc_type.__name__ if exc_type is not None else None,
                "message": str(exc_value),
                "traceback": formatted,
                "frames": frames,
            }

        # Include any stack_info (from logging.stack_info=True calls)
        if record.stack_info:
            log_obj["stack_info"] = record.stack_info

        return json.dumps(log_obj, ensure_ascii=False)


def _uncaught_exception_handler(exc_type, exc_value, exc_traceback):
    """Log uncaught exceptions"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger = logging.getLogger("uncaught")
    logger.error(
        f"Uncaught exception: {exc_type.__name__}: {exc_value}",
        exc_info=(exc_type, exc_value, exc_traceback),
    )


_logging_initialized = False


def init_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    log_file: str = "app.log",
    rotate_max_bytes: int = 10 * 1024 * 1024,
    rotate_backups: int = 5,
    ecs_compatible: bool = False,
) -> None:
    """
    Initialize logging with console and rotating file handlers (idempotent).

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        log_file: Log file name
        rotate_max_bytes: Max bytes per log file before rotation
        rotate_backups: Number of backup files to keep
        ecs_compatible: If True, use ECS-compatible JSON field names
    """
    global _logging_initialized
    if _logging_initialized:
        return

    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ConsoleFormatter())
    root_logger.addHandler(console_handler)

    # Rotating file handler
    log_path = os.path.join(log_dir, log_file)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=rotate_max_bytes,
        backupCount=rotate_backups,
        encoding="utf-8",
    )
    file_handler.setFormatter(JSONFormatter(ecs_compatible=ecs_compatible))
    root_logger.addHandler(file_handler)

    # Install exception handler
    sys.excepthook = _uncaught_exception_handler

    _logging_initialized = True


def get_logger(name: str = __name__) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)
