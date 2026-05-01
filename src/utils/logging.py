"""Logging configuration shared across scripts."""
from __future__ import annotations

import logging
import sys


def setup_logger(name: str = "pce", level: int = logging.INFO) -> logging.Logger:
    """Module-level logger with console handler. Idempotent."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
