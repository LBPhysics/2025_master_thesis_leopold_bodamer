"""
Path configurations for the qspectro2d project.

This module provides cross-platform path definitions for data and figure directories.
Paths are defined relative to the project root to ensure compatibility across different
operating systems and user environments.
"""

import os
from pathlib import Path


def find_project_root():
    """Find the project root directory by looking for 'Master_thesis' directory"""
    current_path = Path(__file__).resolve()

    # Look for Master_thesis directory (case-insensitive search)
    for parent in current_path.parents:
        if parent.name.lower() == "master_thesis":
            return parent

    # Alternative: look for specific marker files/directories
    for parent in current_path.parents:
        if (parent / "Master_thesis").exists() or parent.name == "Master_thesis":
            return (
                parent if parent.name == "Master_thesis" else parent / "Master_thesis"
            )

    raise RuntimeError(
        f"Could not find 'Master_thesis' directory. Current path: {current_path}"
    )


PROJECT_ROOT = find_project_root()

# Define directory paths
DATA_DIR = PROJECT_ROOT / "code" / "python" / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Convert to absolute paths for consistency
DATA_DIR = DATA_DIR.resolve()
FIGURES_DIR = FIGURES_DIR.resolve()

# Additional useful paths
PYTHON_CODE_DIR = PROJECT_ROOT / "code" / "python"
SCRIPTS_DIR = PYTHON_CODE_DIR / "scripts"
NOTEBOOKS_DIR = PYTHON_CODE_DIR / "notebooks"
LATEX_DIR = PROJECT_ROOT / "latex"

# Figures subdirectories (commonly used)
FIGURES_PYTHON_DIR = FIGURES_DIR / "figures_from_python"
FIGURES_1D_DIR = FIGURES_PYTHON_DIR / "1d_spectroscopy"
FIGURES_2D_DIR = FIGURES_PYTHON_DIR / "2d_spectroscopy"
FIGURES_BATH_DIR = FIGURES_PYTHON_DIR / "bath_correlator"
FIGURES_PULSES_DIR = FIGURES_PYTHON_DIR / "pulses"

# Create figure subdirectories if they don't exist
for fig_dir in [
    FIGURES_PYTHON_DIR,
    FIGURES_1D_DIR,
    FIGURES_2D_DIR,
    FIGURES_BATH_DIR,
    FIGURES_PULSES_DIR,
]:
    fig_dir.mkdir(parents=True, exist_ok=True)

# Print paths for debugging (only when directly executed)
if __name__ == "__main__":
    print("Project paths configuration:")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Figures directory: {FIGURES_DIR}")
    print(f"Python code directory: {PYTHON_CODE_DIR}")
    print(f"Scripts directory: {SCRIPTS_DIR}")
    print(f"Notebooks directory: {NOTEBOOKS_DIR}")
    print(f"LaTeX directory: {LATEX_DIR}")
