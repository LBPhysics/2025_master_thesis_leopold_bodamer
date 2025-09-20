"""No-op logger shim.

This module intentionally disables logging across the project while keeping the
same import surface. All calls like::

    from project_config.logging_setup import get_logger
    logger = get_logger(__name__)
    logger.info("...")

silently do nothing and will not write to stderr or files.
"""

from __future__ import annotations

import logging


def get_logger(name: str = "qspectro2d", level: int = logging.INFO) -> logging.Logger:
    """Return a disabled logger that emits nothing.

    - Attaches a NullHandler
    - Sets a very high level
    - Disables propagation
    """
    logger = logging.getLogger(name)
    # Remove any pre-existing handlers and ensure a NullHandler is present
    try:
        logger.handlers.clear()
    except Exception:
        # Older Python versions: fall back to reassign
        logger.handlers = []  # type: ignore[attr-defined]
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL + 10)
    logger.propagate = False
    return logger
