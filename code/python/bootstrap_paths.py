"""Unified project path bootstrap (central location).

Provides path constants and ensure_dirs() without relying on a separate paths.py file.
Safe to import from scripts, notebooks, or external tools once the parent directory
(code/python) is on sys.path.

Usage (inside scripts directory):
    import sys, pathlib
    p = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(p))  # if needed
    from bootstrap_paths import DATA_DIR, SCRIPTS_DIR, ensure_dirs

Behavior:
    - Determines repository root by ascending from this file until a .git directory is found.
    - Exposes canonical directories and a helper to create them.
"""

from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# Root discovery
# ---------------------------------------------------------------------------
_DEF_FILE = Path(__file__).resolve()
for _parent in _DEF_FILE.parents:
    if (_parent / ".git").is_dir():
        PROJECT_ROOT = _parent
        break
else:  # fallback to two levels up
    PROJECT_ROOT = _DEF_FILE.parents[2]

# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------
DATA_DIR = (PROJECT_ROOT / "data").resolve()
FIGURES_DIR = (PROJECT_ROOT / "figures").resolve()
PYTHON_CODE_DIR = PROJECT_ROOT / "code" / "python"
SCRIPTS_DIR = PYTHON_CODE_DIR / "scripts"
NOTEBOOKS_DIR = PYTHON_CODE_DIR / "notebooks"
LATEX_DIR = PROJECT_ROOT / "latex"
FIGURES_PYTHON_DIR = FIGURES_DIR / "figures_from_python"
FIGURES_TESTS_DIR = FIGURES_PYTHON_DIR / "tests"

# ---------------------------------------------------------------------------
# Directory creation helper
# ---------------------------------------------------------------------------

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
]
