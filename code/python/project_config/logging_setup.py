"""Centralized logging setup for the repository.

Usage:
    from project_config.logging_setup import get_logger
    logger = get_logger(__name__)

This configures a consistent formatter and logs to both stderr and a file under logs/.
Call ensure_dirs() before heavy jobs to guarantee the logs directory exists.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .paths import LOGS_DIR, ensure_dirs


_FORMAT = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _setup_handlers() -> list[logging.Handler]:
    ensure_dirs()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = []

    # Stream handler (stderr)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    handlers.append(sh)

    # Rotating file handler
    log_file = Path(LOGS_DIR) / "qspectro2d.log"
    fh = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3)
    fh.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    handlers.append(fh)

    return handlers


def get_logger(name: str = "qspectro2d", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        for h in _setup_handlers():
            logger.addHandler(h)
        logger.propagate = False
    return logger
