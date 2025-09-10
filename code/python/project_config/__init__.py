"""Project-level configuration helpers.

This package contains repository-wide configuration that is not specific to the
``qspectro2d`` runtime library API, such as resolved filesystem paths for data,
figures, LaTeX sources, and related utilities.

Public surface:
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
"""

from .paths import (
    PROJECT_ROOT,
    DATA_DIR,
    FIGURES_DIR,
    FIGURES_PYTHON_DIR,
    FIGURES_TESTS_DIR,
    LOGS_DIR,
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
    "LOGS_DIR",
    "PYTHON_CODE_DIR",
    "SCRIPTS_DIR",
    "NOTEBOOKS_DIR",
    "LATEX_DIR",
    "ensure_dirs",
]
