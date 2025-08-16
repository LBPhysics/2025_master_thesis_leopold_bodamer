"""Compatibility shim for legacy import path.

Moved to ``project_config.paths`` to reflect repository-wide scope (incl. LaTeX).
This module re-exports the new locations and emits a one-time warning when imported
directly.
"""

from __future__ import annotations

from project_config.paths import (
    PROJECT_ROOT,
    DATA_DIR,
    FIGURES_DIR,
    FIGURES_PYTHON_DIR,
    FIGURES_TESTS_DIR,
    PYTHON_CODE_DIR,
    SCRIPTS_DIR,
    NOTEBOOKS_DIR,
    LATEX_DIR,
    ensure_dirs,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "FIGURES_DIR",
    "FIGURES_PYTHON_DIR",
    "FIGURES_TESTS_DIR",
    "PYTHON_CODE_DIR",
    "SCRIPTS_DIR",
    "NOTEBOOKS_DIR",
    "LATEX_DIR",
    "ensure_dirs",
]
