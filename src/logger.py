"""
Centralized logging setup.

Logs to both console (concise) and file (verbose with full prompts/responses).
Log files are stored in logs/ with timestamped filenames.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path


LOG_DIR = Path("logs")

# Custom level for per-example verbose traces (prompts, raw outputs)
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


class TraceLogger(logging.Logger):
    def trace(self, msg, *args, **kwargs):
        if self.isEnabledFor(TRACE):
            self._log(TRACE, msg, args, **kwargs)


logging.setLoggerClass(TraceLogger)


class ConsoleFormatter(logging.Formatter):
    """Compact colored console output."""

    COLORS = {
        "DEBUG": "\033[90m",     # grey
        "TRACE": "\033[90m",     # grey
        "INFO": "\033[0m",      # default
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",    # red
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        return f"{color}[{timestamp}] {record.getMessage()}{self.RESET}"


class FileFormatter(logging.Formatter):
    """Verbose file output with full details."""

    def format(self, record):
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return f"[{timestamp}] [{record.levelname:<7}] {record.getMessage()}"


def setup_logger(
    name: str = "vlm-eval",
    log_file: str = None,
    console_level: int = logging.INFO,
    file_level: int = TRACE,
) -> logging.Logger:
    """Set up logger with console + file handlers.

    Args:
        name: logger name
        log_file: explicit log file path. If None, auto-generates in logs/
        console_level: minimum level for console output
        file_level: minimum level for file output (TRACE=5 logs everything)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(min(console_level, file_level))
    logger.handlers.clear()

    # Console handler -- concise
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(ConsoleFormatter())
    logger.addHandler(console)

    # File handler -- verbose
    if log_file is None:
        LOG_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"{name}_{ts}.log"

    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(FileFormatter())
    logger.addHandler(file_handler)

    logger.info(f"Logging to file: {log_file}")
    return logger


def format_time(seconds: float) -> str:
    """Human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m {int(s)}s"


def format_progress_bar(current: int, total: int, width: int = 30) -> str:
    """Simple text progress bar."""
    pct = current / total if total else 0
    filled = int(width * pct)
    bar = "=" * filled + ">" + "." * (width - filled - 1)
    return f"[{bar}] {current}/{total} ({pct*100:.1f}%)"
