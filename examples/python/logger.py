"""Logger module for logging data in the backend to the console/files or any other location."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)  # Console logger for printing module logs


class Logger(ABC):
    """Abstract class for the logging interface."""

    def __init__(self):
        """Initialize the logger."""
        ...

    @abstractmethod
    def log(self, data: Dict[str, Any], flush: bool = True):
        """Log the data.

        Args:
            data: The data to log.
            flush: Whether to flush the log immediately after logging the data.
        """
        ...

    def flush(self):
        """Flush the log."""
        ...


class FileLogger(Logger):
    """File logger to save logs to a file."""

    def __init__(self, path: Path):
        """Initialize the file logger.

        Args:
            path: The path to the log file.
        """
        super().__init__()
        self.path = path
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        if self.path.is_file():
            logger.warning(f"Log file {path} already exists. Overwriting...")
        self._log = []

    def log(self, data: Dict[str, Any], flush: bool = True):
        """Log the data to the file.

        Note:
            The log is not written to the file unless the flush parameter is set to True.

        Args:
            data: The data to log.
            flush: Whether to flush the log immediately after logging the data.
        """
        self._log.append(data)
        flush and self.flush()

    def flush(self):
        """Flush the log to the file."""
        log = self.jsonify(self._log)
        with open(self.path, "w") as f:
            json.dump(log, f)
        # We don't clear the log here because we do not append to the file, we overwrite it

    def jsonify(self, log: Any) -> Any:
        """Convert the log to a JSON serializable format.

        Some data types like numpy arrays are not JSON serializable, so we recursively convert the
        log to a JSON compatible format.

        Args:
            log: The log data to convert to a JSON serializable format.
        """
        if isinstance(log, np.ndarray):
            return log.tolist()
        if isinstance(log, Mapping):
            return {key: self.jsonify(value) for key, value in log.items()}
        if isinstance(log, Iterable):
            return [self.jsonify(value) for value in log]
        return log