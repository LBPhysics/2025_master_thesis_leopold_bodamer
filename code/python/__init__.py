"""my_project package root.

Central bootstrap for project-wide canonical paths. Eliminates the need for a
separate `project_paths.py` module; import directly:

    from my_project import DATA_DIR, SCRIPTS_DIR

Root discovery strategy:
1. Ascend from this file until a directory containing a `.git` folder is found.
2. Fallback: two levels up from this file (repo layout assumption).
"""

from __future__ import annotations
from pathlib import Path

_DEF_FILE = Path(__file__).resolve()
for _parent in _DEF_FILE.parents:
    if (_parent / ".git").is_dir():
        PROJECT_ROOT = _parent
        break
else:  # fallback if .git not found (e.g., packaged source distribution)
    PROJECT_ROOT = _DEF_FILE.parents[2]

# Canonical paths
DATA_DIR = (PROJECT_ROOT / "data").resolve()
FIGURES_DIR = (PROJECT_ROOT / "figures").resolve()
PYTHON_CODE_DIR = PROJECT_ROOT / "code" / "python"
SCRIPTS_DIR = PYTHON_CODE_DIR / "scripts"
NOTEBOOKS_DIR = PYTHON_CODE_DIR / "notebooks"
LATEX_DIR = PROJECT_ROOT / "latex"
FIGURES_PYTHON_DIR = FIGURES_DIR / "figures_from_python"
FIGURES_TESTS_DIR = FIGURES_PYTHON_DIR / "tests"

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
