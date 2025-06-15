"""
Path configurations for the qspectro2d project.

This module provides cross-platform path definitions for data and figure directories.
Paths are defined relative to the project root to ensure compatibility across different
operating systems and user environments.
"""

from pathlib import Path
import os

# Get the project root directory (master_thesis folder)
# Start from current file location and go up to find master_thesis
current_file = Path(__file__).resolve()
project_root = None

# Walk up the directory tree to find master_thesis folder
for parent in current_file.parents:
    if parent.name == 'master_thesis':
        project_root = parent
        break

if project_root is None:
    raise RuntimeError("Could not find 'master_thesis' directory in parent directories")

# Define directory paths
DATA_DIR = project_root / 'code' / 'python' / 'data'
FIGURES_DIR = project_root / 'figures'

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Convert to absolute paths for consistency
DATA_DIR = DATA_DIR.resolve()
FIGURES_DIR = FIGURES_DIR.resolve()

# Additional useful paths
PYTHON_CODE_DIR = project_root / 'code' / 'python'
SCRIPTS_DIR = PYTHON_CODE_DIR / 'scripts'
NOTEBOOKS_DIR = PYTHON_CODE_DIR / 'notebooks'
LATEX_DIR = project_root / 'latex'

# Figures subdirectories (commonly used)
FIGURES_PYTHON_DIR = FIGURES_DIR / 'figures_from_python'
FIGURES_1D_DIR = FIGURES_PYTHON_DIR / '1d_spectroscopy'
FIGURES_2D_DIR = FIGURES_PYTHON_DIR / '2d_spectroscopy'
FIGURES_BATH_DIR = FIGURES_PYTHON_DIR / 'bath_correlator'
FIGURES_PULSES_DIR = FIGURES_PYTHON_DIR / 'pulses'

# Create figure subdirectories if they don't exist
for fig_dir in [FIGURES_PYTHON_DIR, FIGURES_1D_DIR, FIGURES_2D_DIR, 
                FIGURES_BATH_DIR, FIGURES_PULSES_DIR]:
    fig_dir.mkdir(parents=True, exist_ok=True)

# Print paths for debugging (only when directly executed)
if __name__ == "__main__":
    print("Project paths configuration:")
    print(f"Project root: {project_root}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Figures directory: {FIGURES_DIR}")
    print(f"Python code directory: {PYTHON_CODE_DIR}")
    print(f"Scripts directory: {SCRIPTS_DIR}")
    print(f"Notebooks directory: {NOTEBOOKS_DIR}")
    print(f"LaTeX directory: {LATEX_DIR}")
