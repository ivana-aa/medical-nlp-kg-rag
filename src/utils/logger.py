"""Simple logging helpers used by training and RAG scripts."""

from __future__ import annotations

import logging
import sys


def setup_logger(name: str = "medical_nlp", level: int = logging.INFO) -> logging.Logger:
    """Create a console logger with a consistent, beginner-friendly format."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
