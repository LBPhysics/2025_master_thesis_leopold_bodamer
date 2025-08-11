"""Project path definitions (no side effects).

This module exposes lazily-resolved paths relative to the detected project root.
It intentionally performs **no filesystem mutations at import time** (no directory
creation) to keep imports pure and to avoid surprising side effects in contexts like
unit tests, packaging, or documentation builds.

Call :func:`ensure_dirs` explicitly in entry‑point scripts (e.g. simulation runners,
CLI tools, notebooks) before writing data / figures.

Rationale for removing side effects:
    * Deterministic imports (import does not touch disk)
    * Easier mocking of filesystem in tests
    * Clearer responsibility boundary (callers decide when to create output dirs)
    * Avoids accidental creation of large directory trees during static analysis

Example::

    from qspectro2d.config.paths import DATA_DIR, FIGURES_DIR, ensure_dirs
    ensure_dirs()  # creates the base output directory structure if missing
    (DATA_DIR / "1d_spectroscopy").mkdir(exist_ok=True)

Future extension: accept a ``base=Path`` override or environment variable to allow
redirecting outputs (e.g. for scratch / temporary runs) without modifying code.
"""

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

#########################
# Base directories (pure)
#########################
DATA_DIR = (PROJECT_ROOT / "data").resolve()
FIGURES_DIR = (PROJECT_ROOT / "figures").resolve()

# Additional useful paths (NOT used)
PYTHON_CODE_DIR = PROJECT_ROOT / "code" / "python"
SCRIPTS_DIR = PYTHON_CODE_DIR / "scripts"
NOTEBOOKS_DIR = PYTHON_CODE_DIR / "notebooks"
LATEX_DIR = PROJECT_ROOT / "latex"

##############################
# Figure subdirectories (lazy)
##############################
FIGURES_PYTHON_DIR = FIGURES_DIR / "figures_from_python"
FIGURES_BATH_DIR = FIGURES_PYTHON_DIR / "bath_correlator"
FIGURES_PULSES_DIR = FIGURES_PYTHON_DIR / "pulses"
FIGURES_TESTS_DIR = FIGURES_PYTHON_DIR / "tests"  # For test figures


def ensure_dirs() -> None:
    """Create the canonical output directory hierarchy if missing.

    Safe to call multiple times. Kept separate from import to avoid side effects.
    """

    for path in [
        DATA_DIR,
        FIGURES_DIR,
        FIGURES_PYTHON_DIR,
        FIGURES_BATH_DIR,
        FIGURES_PULSES_DIR,
        FIGURES_TESTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


# Print paths for debugging (only when directly executed)
if __name__ == "__main__":  # manual debug helper
    print("Project root:", PROJECT_ROOT)
    print("Data directory:", DATA_DIR)
    print("Figures directory:", FIGURES_DIR)
    print("(Call ensure_dirs() to create directories)")
